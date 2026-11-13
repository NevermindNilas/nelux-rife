# Porting vs-rife to Nelux

This compares the port with vs-rife 5.7.0 piece by piece. It also lists the SDK features the port
assumes and the problems it exposed.

## Size

| | vs-rife | nelux-rife |
|---|---|---|
| Glue code (non-blank, non-comment lines, docstrings included) | 708 (`__init__.py`, `__main__.py`) | 279 (`interpolate.py`, `sc_detect.py`, `composite.py`, `versions.py`) |
| Model code (`IFNet_*.py`, `warplayer.py`) | 37 files | The same 37 files: 35 unchanged, 2 patched |
| Locks, CUDA streams, `synchronize()` calls | 5, 5, 6 | 0, 0, 0 |
| Hand-written frame caches | 2 dicts, evicted 10 frames back | None |

## Piece by piece

| vs-rife | nelux-rife |
|---|---|
| `frame_cache` and `encode_cache` dicts, filled by `encoding()` through `FrameEval` + `ModifyFrame`, evicted at `real_n - 10` | `Rife.prepare` returns `(img, encode(img))`. Nelux runs it once per source frame, shares the result between both pairs on any thread, and evicts through its LRU cache |
| 5 locks and 5 CUDA streams, each with its own `synchronize()` | None. Values are immutable and dependencies are declared (design §6) |
| `Interleave([clip] * factor_num)`, `DuplicateFrames`, `[::factor_den]`, then `real_n = n * den // num` and `t = n * den % num / num` | The timeline: `factor=Fraction(num, den)` gives the same positions and the same frame count, with exact rationals (design §6.1) |
| `fps_num` / `fps_den` converted to a factor | The same conversion in `composite.py`. The kernel also accepts `fps=` directly |
| `t == 0` returns `f[0]` | `passthrough`, which `Interpolator` sets by default |
| `sc and f[0].props["_SceneChangeNext"]` returns `f[0]` | `scene="duplicate"`. With `sc=False` the composite passes `scene="ignore"`, so existing properties are ignored, as in vs-rife |
| `sc_detect`: GRAY8 resize, `PlaneStats` against the next frame, `ModifyFrame` | `SceneChange`: one filter with `after = 1` that returns `{"_SceneChangeNext": ...}` and passes pixels through |
| Padding to `max(modulo, int(modulo / scale))` with zeros, then cropping `[:h, :w]` | `size_multiple` with `pad_mode = "constant"`. Nelux pads and crops |
| `tenFlow_div` and `backwarp_tenGrid` precomputed for one padded size | Computed from the frame's shape inside `forward`, so one artifact serves every resolution. `tenFlow_div` is a pair of numbers, not a tensor; `warplayer.py` only indexes it |
| `frame_to_tensor` / `tensor_to_frame` through NumPy, with a host copy each way | None. Frames stay tensors on the GPU |
| `torch.export` + Torch-TensorRT on the user's machine at first use, cached per GPU, shape and TensorRT version | AOTInductor at `nelux-plugin build`, on the author's machine, once |
| `auto_download`, `python -m vsrife` | `weights.fetch(url, sha256)` at build time. Users never download weights |
| RGBH means FP16, RGBS means FP32, frames included | `precision` build-time argument. FP16 variants declare FP16 frames, so Nelux converts decoded frames straight to half precision and nothing is cast inside the plugin. `composite.py` exposes it as `fp16=` |
| `device_index` | The reserved `device` argument |
| `torch.set_float32_matmul_precision("high")`, so TF32 | The host's TF32 setting (design §7) |
| The 200-line `match model:` block | `versions.py`, one line of data per model |

## Changes to the model code

Only `IFNet_HDv3_v4_0.py` and `IFNet_HDv3_v4_1.py` change; `diff` them against upstream to see the
change. Both contain this:

```python
if i == 1 and f0[:, :2].abs().max() > 32 and f0[:, 2:4].abs().max() > 32:
    # restart the coarse levels at twice the scale
```

That is a Python `if` on tensor data, which `torch.export` can't capture. Its `.max()` also spans
the whole batch, so frames in a batch would affect each other. The patch computes both paths and
picks per frame with `torch.where`. At batch 1 that equals vs-rife's output exactly, but v4.0 and
v4.1 cost about twice as much. `torch.cond` could avoid the doubled cost, if AOTInductor runs it
without the proxy executor; that hasn't been tested.

## SDK features this port assumes

| Feature | Used for | Status |
|---|---|---|
| Fractional `factor=` and `fps=` timelines | `factor_num` / `factor_den`, `fps_num` / `fps_den` | Designed (§6.1), not built |
| Property outputs from `forward` | `SceneChange` | Designed (§6.1), not built |
| The `scene` policy on retiming kernels | `sc` | Designed (§6.1), not built |
| Windows as separate `forward` arguments | `SceneChange.forward(current, following)` | Proposed (§7, open decision 4) |
| `timesteps = "any"` | Rejecting midpoint-only models | Designed (§6.1), not built |
| `variants` as an explicit list of combinations | Excluding `ensemble=True` for the 9 models that raise on it | **New, not in the design** |
| Attributes set in `__init__` override the class per variant | `size_multiple` depends on the model and `scale`; the frame format `input` depends on `precision` | Built in the SDK |
| `[tool.nelux-plugin.variants]` in `pyproject.toml` to pick which variants ship | Keeping the wheel buildable | **New, not in the design** |
| `composites` in `pyproject.toml`, shipped as source in the wheel | `clip.rife.rife(...)` | Runtime named in §3.1; registration and packaging not designed |
| `min_size = 2` | vs-rife accepts any size | Supported today |

## Problems the port exposed

1. **Variants explode.** vs-rife's model, scale, ensemble and precision options give 630 combinations
   for the CUDA target. That's (36 + 27 models with ensemble) × 5 scales × 2 precisions. At 10–90 MB
   each, shipping them all is impossible. vs-rife avoids this because it compiles on the user's
   machine, eagerly in PyTorch or with TensorRT at first use. `scale` is a build-time argument because
   it sets the network's internal resolution, so it can't become a per-call scalar. Options:
   - ship a curated subset, as `pyproject.toml` does here;
   - compile one artifact per architecture with weights loaded at run time, since many versions
     share code (4.2–4.4, 4.7–4.9, 4.10–4.12);
   - reconsider compiling on the user's machine as an opt-in.
2. **Per-variant attributes.** `size_multiple` depends on the model and `scale`, and the frame
   format depends on `precision`, but the SDK only reads class attributes. Without per-variant
   formats, FP16 variants would receive FP32 frames and cast them inside the plugin: twice the
   frame memory and bandwidth at both ends, unlike vs-rife's RGBH path. Solved: the SDK now lets
   attributes set in `__init__` override the class for that variant (design §7), and this port uses
   that for both.
3. **Composite packaging.** A composite is plugin Python run at graph build. The design keeps plugin
   code out of `import nelux`, so this is allowed, but the wheel layout, the import rules and how a
   composite is exposed as `clip.ns.name` aren't designed yet.
4. **Model choice fails late.** vs-rife checks `model in models` up front. Here an unknown model,
   or one the wheel doesn't ship, fails when the graph is built. That's the same point in the
   script, but the error has to come from the variant lookup, and it should list the combinations
   that are installed.
