"""RIFE frame interpolation, the kernel.

This is everything vs-rife's ``rife()`` does per frame, minus what Nelux does
for every interpolator:

========================================  =====================================
vs-rife                                   here
========================================  =====================================
frame_cache / encode_cache dicts, the     ``prepare``: Nelux runs it once per
10-frame eviction, ``encoding()`` run     source frame and shares the result
through FrameEval + ModifyFrame           between both pairs, on any thread
5 locks, 5 CUDA streams, synchronize()    none: values are immutable (design §6)
Interleave / DuplicateFrames / [::den],   the timeline: ``factor=``, ``fps=``
``real_n``, ``t``                         (design §6.1)
``t == 0`` returns f[0]                   ``passthrough`` (Interpolator default)
``_SceneChangeNext`` returns f[0]         ``scene="duplicate"``
pad to ``tmp``, crop ``[:h, :w]``         ``size_multiple`` + ``pad_mode``
frame_to_tensor / tensor_to_frame         none: frames are already tensors
RGBH runs FP16, RGBS runs FP32            ``precision`` sets the frame format
TensorRT engine build at first run        AOTInductor at ``nelux-plugin build``,
                                          prebuilt; local compile is opt-in
``auto_download`` / ``python -m vsrife``  ``weights.fetch`` at build time
========================================  =====================================
"""

import torch
import torch.nn as nn

from nelux_plugin_sdk import Interpolator, weights

from .versions import PRECISIONS, SCALES, VERSIONS, WEIGHTS_URL


class Rife(Interpolator):
    namespace = "rife"
    name = "interpolate"

    timesteps = "any"  # every IFNet takes an arbitrary timestep
    pad_mode = "constant"  # vs-rife: F.pad(img, padding), zeros
    min_size = 2  # vs-rife accepts any size; padding covers the rest
    max_batch = 1  # vs-rife runs batch 1; see models/IFNet_HDv3_v4_0.py before raising it

    # One artifact per combination; ensemble only where the model supports it.
    # Proposed SDK extension: an explicit list of combinations (see PORTING.md).
    variants = [
        {"model": model, "scale": scale, "ensemble": ensemble, "precision": precision}
        for model, version in VERSIONS.items()
        for scale in SCALES
        for ensemble in ((False, True) if version.ensemble else (False,))
        for precision in PRECISIONS
    ]

    def __init__(self, model: str = "4.25", scale: float = 1.0, ensemble: bool = False, precision: str = "fp32"):
        super().__init__()
        version = VERSIONS[model]
        self.dtype = torch.half if precision == "fp16" else torch.float

        # Attributes set in __init__ override the class for this variant (design §7).
        #
        # vs-rife: RGBH clips run in FP16 and RGBS clips in FP32, frames
        # included. FP16 variants take and return FP16 frames, so Nelux converts
        # decoded frames straight to half precision and nothing is cast here.
        # The output format follows the input.
        self.input = dict(dtype="float16" if precision == "fp16" else "float32")
        if precision == "fp16":
            # The build compares compiled with eager FP16, and FP16 rounding differs
            # between them. Measured on 4.25: on the SDK's noisy validation frames,
            # compiled differs from eager by up to 0.073, while eager FP16 is itself
            # 0.086 from FP32 and compiled 0.073, so this is not a compile error. On
            # real motion all three agree within 0.003.
            self.tolerance = 0.1
        # vs-rife: tmp = max(modulo, int(modulo / scale)), then pad up to it.
        self.size_multiple = max(version.modulo, int(version.modulo / scale))

        # vs-rife's init_module().
        state_dict = torch.load(
            weights.fetch(WEIGHTS_URL.format(model), version.sha256), map_location="cpu", mmap=True
        )
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k}

        self.flownet = version.ifnet()(scale, ensemble)
        self.flownet.load_state_dict(state_dict, strict=False)
        self.flownet.to(self.dtype)

        self.encode: nn.Module | None = version.make_head()
        if self.encode is not None:
            encode_state_dict = {k.replace("encode.", ""): v for k, v in state_dict.items() if "encode." in k}
            self.encode.load_state_dict(encode_state_dict)
            self.encode.to(self.dtype)

    def prepare(self, frame):
        # vs-rife's encoding(): the padded frame and its features, once per source frame.
        if self.encode is None:
            return frame
        return frame, self.encode(frame)

    def forward(self, a, b, t):
        if self.encode is None:
            img0, img1 = a, b
            features = ()
        else:
            (img0, f0), (img1, f1) = a, b
            features = (f0, f1)

        n, _, ph, pw = img0.shape

        # vs-rife precomputes these for one padded size. Here they follow the
        # frame's shape so one artifact serves every resolution. The divisor is a
        # pair of numbers rather than a tensor; warplayer.py only indexes it.
        tenFlow_div = ((pw - 1.0) / 2.0, (ph - 1.0) / 2.0)
        tenHorizontal = torch.linspace(-1.0, 1.0, pw, dtype=torch.float, device=img0.device)
        tenHorizontal = tenHorizontal.view(1, 1, 1, pw).expand(-1, -1, ph, -1)
        tenVertical = torch.linspace(-1.0, 1.0, ph, dtype=torch.float, device=img0.device)
        tenVertical = tenVertical.view(1, 1, ph, 1).expand(-1, -1, -1, pw)
        backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

        # vs-rife: torch.full([1, 1, ph, pw], t, dtype=dtype), one t per frame here.
        # t always arrives as float32, so this is the one cast left.
        timestep = t.to(self.dtype).view(n, 1, 1, 1).expand(n, 1, ph, pw)

        return self.flownet(img0, img1, timestep, tenFlow_div, backwarp_tenGrid, *features)
