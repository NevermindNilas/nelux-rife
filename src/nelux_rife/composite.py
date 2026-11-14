"""``clip.rife.rife(...)``: vs-rife's signature, wiring the two kernels.

A ``composite`` runs once when the graph is built (design §3.1) and never on the
frame path. It only checks arguments and connects ``sc_detect`` and
``interpolate``; the frames themselves never touch Python.

Differences from vs-rife's ``rife()``:

- ``device_index`` becomes the reserved ``device`` argument.
- ``fp16`` replaces vs-rife's choice by clip format (RGBH or RGBS). Nelux
  converts formats between nodes, so precision is a choice, not a property of
  the clip.
- ``auto_download`` is gone: prebuilt artifacts have the weights compiled in. A user who
  opts in to compiling a configuration locally fetches them then, as vs-rife does.
- The ``trt_*`` arguments are gone: the kernels were compiled with AOTInductor
  when the plugin was built.
- ``model``, ``scale`` and ``ensemble`` pick a precompiled variant. A
  combination that is neither in the installed wheel nor compiled locally (opt-in)
  fails at graph build.
"""

from fractions import Fraction

SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


def rife(
    clip,
    device_index: int = 0,
    model: str = "4.25",
    factor_num: int = 2,
    factor_den: int = 1,
    fps_num: int | None = None,
    fps_den: int | None = None,
    scale: float = 1.0,
    ensemble: bool = False,
    sc: bool = False,
    sc_threshold: float | None = 0.15,
    fp16: bool = False,
):
    """Real-Time Intermediate Flow Estimation for Video Frame Interpolation

    :param clip:            Clip to process.
    :param device_index:    Device ordinal of the GPU.
    :param model:           Model to use.
    :param factor_num:      Numerator of factor for target frame rate.
    :param factor_den:      Denominator of factor for target frame rate.
                            For example `factor_num=5, factor_den=2` will multiply the frame rate by 2.5.
    :param fps_num:         Numerator of target frame rate.
    :param fps_den:         Denominator of target frame rate.
                            Override `factor_num` and `factor_den` if specified.
    :param scale:           Control the process resolution for optical flow model. Try scale=0.5 for 4K video.
                            Must be 0.25, 0.5, 1.0, 2.0, or 4.0.
    :param ensemble:        Smooth predictions in areas where the estimation is uncertain.
    :param sc:              Avoid interpolating frames over scene changes.
    :param sc_threshold:    Threshold for scene change detection. Must be between 0.0 and 1.0.
                            Leave the argument as None if the frames already have _SceneChangeNext property set.
    :param fp16:            Run inference in FP16 instead of FP32.
    """
    if clip.num_frames < 2:
        raise ValueError("rife: clip's number of frames must be at least 2")

    if factor_num < 1:
        raise ValueError("rife: factor_num must be at least 1")

    if factor_den < 1:
        raise ValueError("rife: factor_den must be at least 1")

    if fps_num is not None and fps_num < 1:
        raise ValueError("rife: fps_num must be at least 1")

    if fps_den is not None and fps_den < 1:
        raise ValueError("rife: fps_den must be at least 1")

    if fps_num is not None and fps_den is not None and not clip.fps:
        raise ValueError("rife: clip does not have a valid frame rate and hence fps_num and fps_den cannot be used")

    if scale not in SCALES:
        raise ValueError("rife: scale must be 0.25, 0.5, 1.0, 2.0, or 4.0")

    factor = Fraction(factor_num, factor_den)
    if fps_num is not None and fps_den is not None:
        factor = Fraction(fps_num, fps_den) / clip.fps

    if factor <= 1:
        raise ValueError(
            "rife: target frame rate must be higher than source frame rate. consider using change_fps from "
            "https://github.com/Jaded-Encoding-Thaumaturgy/vs-tools if you want to reduce the frame rate"
        )

    if sc and sc_threshold is not None:
        clip = clip.rife.sc_detect(threshold=sc_threshold)

    return clip.rife.interpolate(
        model=model,
        scale=scale,
        ensemble=ensemble,
        precision="fp16" if fp16 else "fp32",
        factor=factor,
        scene="duplicate" if sc else "ignore",
        device=f"cuda:{device_index}",
    )
