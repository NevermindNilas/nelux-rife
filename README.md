# nelux-rife

RIFE frame interpolation for [Nelux](https://github.com/NevermindNilas/Nelux), a 1:1 port of
[vs-rife](https://github.com/HolyWu/vs-rife) 5.7.0 (upstream commit `3488617`).

**Status: design test, not runnable.** This is written against the Nelux plugin SDK as *designed*
(`docs/plugin-system/design.md` on branch `concept/plugin-system`). Several features it uses aren't
built yet. [PORTING.md](PORTING.md) lists them and compares the port with vs-rife piece by piece.

## Usage

vs-rife:

```python
from vsrife import rife

clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")
clip = rife(clip, model="4.25", factor_num=5, factor_den=2, sc=True)
```

nelux-rife, with the same arguments:

```python
import nelux

clip = nelux.VideoReader("in.mp4", decode_accelerator="nvdec").clip()
clip = clip.rife.rife(model="4.25", factor_num=5, factor_den=2, sc=True)
clip.encode("out.mp4", codec="hevc_nvenc")
```

Or the two kernels directly, which is what `rife()` wires up:

```python
from fractions import Fraction

clip = clip.rife.sc_detect(threshold=0.15)
clip = clip.rife.interpolate(model="4.25", factor=Fraction(5, 2), scene="duplicate")
# or fps=Fraction(60000, 1001) to target a frame rate directly
```

## Layout

| Path | What it is |
|---|---|
| `src/nelux_rife/interpolate.py` | The `Rife` interpolator: vs-rife's `init_module`, `encoding` and `inference` |
| `src/nelux_rife/sc_detect.py` | The `SceneChange` filter: vs-rife's `sc_detect` |
| `src/nelux_rife/composite.py` | `rife()` with vs-rife's signature |
| `src/nelux_rife/versions.py` | All 36 models: module, head, padding modulo, ensemble support, weights SHA-256 |
| `src/nelux_rife/models/` | vs-rife's `IFNet_HDv3_*.py` and `warplayer.py`, byte for byte, except v4.0 and v4.1 (see PORTING.md) |

## Licence

All rights reserved; see [LICENSE](LICENSE). The code taken from and ported from vs-rife,
Copyright (c) 2021 HolyWu, remains under the MIT licence; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
