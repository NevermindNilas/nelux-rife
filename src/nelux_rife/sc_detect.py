"""Scene-change detection, as vs-rife's ``sc_detect``.

vs-rife builds it from four VapourSynth calls::

    sc_clip = clip.resize.Bicubic(format=vs.GRAY8, matrix_s="709")
    sc_next = sc_clip[1:] + sc_clip[-1]
    sc_next = sc_next.std.PlaneStats(sc_clip)
    clip.std.ModifyFrame([clip, sc_next], set_sc_props)   # _SceneChangeNext = threshold < PlaneStatsDiff

Here it is one filter that looks one frame ahead and returns only a property,
so the pixels pass through untouched (design §6.1, "Properties out"). The last
frame is compared with itself, as ``sc_clip[-1]`` does.
"""

import torch

from nelux_plugin_sdk import Filter


def _gray8(rgb: torch.Tensor) -> torch.Tensor:
    # RGB -> GRAY8 with the BT.709 matrix: limited range (16-235), rounded to
    # nearest, no dithering, which are VapourSynth's resize defaults for a
    # YUV-family output.
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return torch.round(16.0 + 219.0 * y).clamp(0.0, 255.0)


class SceneChange(Filter):
    namespace = "rife"
    name = "sc_detect"

    after = 1  # forward(current, next)
    max_batch = 8

    def forward(self, current, following, threshold: float = 0.15):
        # PlaneStatsDiff: mean absolute difference, normalized to [0, 1].
        diff = (_gray8(current) - _gray8(following)).abs().mean(dim=(1, 2, 3)) / 255.0
        return {"_SceneChangeNext": (diff > threshold).to(torch.int64)}
