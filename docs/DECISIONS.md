# Decisions

Record **architectural and behavioral decisions** here so future work does not re-litigate settled choices.

**Newest first.** Related choices that landed the same day **may** be merged into one dated section (subsections + a short consequences line) instead of many micro-entries.

## 2026-05-03 — `crop` / `anon` output ``--format`` and ``--quality``

**Context:** Callers wanted alternative encodings (WebP, PNG) and a quality knob without changing the existing JPEG behavior. WebP in particular is preferred lossy at the slowest/best-compression setting; PNG is lossless and has no comparable knob.

**Decision:** Add ``--format / -f`` (``jpg`` default, ``webp``, ``png``) and ``--quality / -q`` (positive int in ``1..100``) to ``crop`` and ``anon`` (full-frame and ``--crop``). The chosen format drives **both** the encoder **and** the file extension for sidecar primary crops (``{stem}-cropped.{ext}``, ``{stem}-cropped-NN.{ext}``), the ``anon`` sidecar (``{stem}-anon.{ext}``), and the ``--debug`` overlay (``{stem}-debug.{ext}``). When ``-o`` / ``--output`` is given, its suffix is **rewritten** to match ``--format`` (e.g. ``-o foo.jpg --format png`` writes ``foo.png``); ``.jpg`` and ``.jpeg`` are both treated as already-jpg. WebP is encoded lossy with Pillow ``method=6`` (slowest, best compression) and ``lossless=False``; the user-facing knob is just ``--quality``. PNG is lossless: ``--quality`` is **accepted but silently ignored** (no separate ``compress_level`` knob to keep the surface small). Default ``--quality`` is **per-format**: ``95`` for jpg (matches the old hard-coded default), ``90`` for webp, N/A for png. Out-of-range ``--quality`` is a Click usage error.

**Consequences:** **CLI-005**, **CLI-006**, **CLI-010**, **CLI-011**, **DATA-001**, **DATA-006** plus new **DATA-008** in ``docs/SPEC.md``; ``crop_one`` / ``anon_one`` / ``write_crop_debug_jpeg`` accept ``output_format`` and ``quality`` kwargs; README options tables and the "Outputs:" blurb gain the new flags.

## 2026-05-03 — `anon` subcommand (full-frame vs `--crop`)

**Context:** Callers wanted face anonymization without changing framing or resolution; they also wanted a single obvious way to get the existing crop+anon pipeline.

**Decision:** Add CLI ``anon``: default mode loads the full image, runs the same SSD + **CLI-008** inpaint stack on the native-sized raster (highest-confidence face only, unchanged from ``crop --anon``), writes ``{stem}-anon.jpg``, and **never** runs GFPGAN. Flag ``--crop`` reuses ``crop``’s width/height/strategy/enhance/debug options and calls ``crop_one(..., anon=True)``, so ``anon --crop …`` matches ``crop --anon …``. Supplying crop-only flags without ``--crop`` is a usage error.

**Consequences:** **CLI-010**–**CLI-012**, **DATA-007**, **ERR-002**; new ``ImageCropper.anon_one``.

## 2026-05-03 — `crop --output-dir` vs basename collisions

**Context:** Users asked to gather crops from many inputs into one directory; default naming uses ``{stem}-cropped.jpg`` only, so two inputs with the same basename (e.g. ``dir1/a.jpg`` and ``dir2/a.jpg``) would target the same path under a shared ``--output-dir``.

**Decision:** Add optional ``--output-dir`` (mutually exclusive with ``-o``). Primary crops use the same sidecar basenames under that directory; debug JPEGs stay beside each input (**DATA-006**). **No** CLI preflight for duplicate targets: the **existing** phase-2 overwrite refusal applies to the second conflicting write unless ``--force`` is set.

**Consequences:** **CLI-005**, **CLI-006** in ``docs/SPEC.md``; ``crop_one`` accepts optional ``output_dir`` (explicit ``output_path`` wins when both are passed at the API).

## 2026-04-21 — Verbose stderr: per-crop progress callback

**Context:** Long-running ``crop_one`` (YOLO, GFPGAN, multiple subjects) produced no stderr until the full per-input summary line, which felt unresponsive under captured stderr.

**Decision:** ``crop_one`` accepts an optional ``progress: Callable[[str], None]``; the CLI passes a styled writer when not ``--quiet``, emitting lines before region selection and before each ``crop i/n`` save. **CLI-007** distinguishes interim lines from the final per-input summary line.

**Consequences:** Tests may assert callback strings; quiet mode passes ``progress=None``.

## 2026-04-21 — Multi-subject crops (automatic numbered sidecars)

**Context:** Group photos often contain several people or faces; the tool previously kept only the highest-confidence detection.

**Decision:** When YOLO returns multiple COCO person boxes or the SSD returns multiple faces above the existing confidence threshold, ``crop`` writes **one** resized output per detection (same aspect expansion and post-resize enhance/anon as today), ordered by **descending confidence**. A single output uses ``{stem}-cropped.jpg``; multiple outputs use ``{stem}-cropped-01.jpg`` … with pad width ``max(2, len(str(N)))``. If ``--output`` names a single file but ``N > 1``, fail that input with **no** crop writes. **No** IoU deduplication or NMS between overlapping boxes in this version. Progress line strategy text uses the **first** crop’s suffix chain plus `` ×N``; later crops may differ (e.g. enhance skipped) without being listed individually. Overwrite checks for implicit sidecars run **after** detection so ``N`` is known; only the debug JPEG is pre-checked before load when ``--output`` is omitted.

**Consequences:** **DATA-003**, **CLI-006**, **CLI-007**, **DATA-006** updates in ``docs/SPEC.md``; ``detect_human_bboxes`` / ``detect_face_padded_bbox_list`` and ``CropResult.output_paths``.

Suggested shape for a standalone decision:

```text
## YYYY-MM-DD — Short title

**Context:** …
**Decision:** …
**Consequences:** …
```

## 2026-04-21 — DATA-004: left/right primary and empty vertical interval use `y_lo` / bbox-top

**Context:** For a wide person box, **left** or **right** often wins the nearest-edge tie; vertical **centering** shifted **`y0`** down and clipped the head on the final crop. When **`min(H - h, y1)`** was below **`y2 - h`** (empty legal **`y`** interval, e.g. box flush to the top), the fallback still used a **centered** **`y0`** clamped to **`[0, H - h]`**, which could start **below** the detection top and clip the head.

**Decision:** For **left** and **right** primary edges, set **`y0 = y_lo`**. In the **`else`** branch, when **`y_lo > y_hi`**, set **`y0 = max(0, min(y1c, h_img - h_i))`**.

**Consequences:** **`docs/SPEC.md`** **DATA-004**; new **`tests/test_crop.py`** case for left-primary portrait aspect.

## 2026-04-21 — `crop --debug` overlay JPEG

**Context:** Diagnosing crop geometry (detector boxes vs. final crop) required ad-hoc scripts.

**Decision:** Add optional ``--debug`` on ``imagecropper crop``. When set, write ``{stem}-debug.jpg`` beside each input after load: full source resolution with rectangles for the same windows used for strategy labeling (person, SSD + padded face, or center window) and corner ``(x,y)`` text. Respect ``--force`` for the debug path alongside the main output.

**Consequences:** **DATA-006**, **CLI-005** / **CLI-007** updates in **`docs/SPEC.md`**; **`README.md`** options table; per-file progress may show ``dbg=`` basename.

## 2026-04-21 — DATA-004: when bottom edge is primary, use minimum `y0`

**Context:** With nearest-edge placement, choosing the **bottom** edge as primary (smallest slack under the detection box) set **`y0 = y_hi`**, shifting the aspect crop **down** within the legal vertical band. That often clipped heads on portrait outputs when feet sat near the frame bottom, even though the person box was fully contained.

**Decision:** Keep **which** edge is primary (minimum slack among top, bottom, left, right; same tie order). When the primary edge is **bottom**, set **`y0 = y_lo`** (minimum legal vertical origin), matching the **top**-primary case and **favoring headroom** above the detection box top. **Human** and **face** paths both use **`_expand_bbox_to_aspect_crop`**, so they inherit this rule.

**Consequences:** **`docs/SPEC.md`** **DATA-004** documents the exception; **`tests/test_crop.py`** bottom-edge expectation aligns with **`y_lo`**.

## 2026-04-21 — MIT License for the repository

**Context:** The project is distributed as software and needs a simple, permissive license.

**Decision:** Ship **`LICENSE`** as the **MIT** grant at the repository root; set **`license = "MIT"`** and **`license-files = ["LICENSE"]`** in **`pyproject.toml`** (SPDX; setuptools **≥77**).

**Consequences:** Downstream use, modification, and redistribution follow **MIT**; **`README.md`** states the license and points to **`LICENSE`**.

## 2026-04-21 — GFPGAN as a core runtime dependency

**Context:** Face restoration should work after a normal install without users opting into a separate **`[enhance]`** extra.

**Decision:** Declare **gfpgan** in **`[project.dependencies]`** and remove the **`enhance`** optional dependency group from **`pyproject.toml`**.

**Consequences:** Every standard install pulls GFPGAN and its transitive requirements; spec items **NFR-BLD-003**, **NFR-BLD-006**, and **CLI-005** / **CLI-009** are updated so **`(enhance failed)`** applies to runtime/model failures, not to a deliberately omitted optional extra.

## 2026-04-21 — Smart `crop` CLI: stack, detection, geometry, status, enhance, anonymize

**Context:** First shippable `crop` behavior and supporting toolchain.

**Decision:**

### Tooling and layout

- **CLI:** Click; package under setuptools **`src/`** layout; **uv**; **pytest-cov** with fail-under **80**; **Black**, **Ruff**, **Mypy**, **pre-commit** as enforced in `pyproject.toml`.
- **Imaging:** **Pillow** for file read/write; **OpenCV** + **NumPy** in memory for arrays, detection, and crop math.
- **Heavy imports:** **Ultralytics** is a direct runtime dependency; load YOLO inside human detection so **`--help`** stays cheap.

### `crop` contract

- **Subcommand** `crop` with **`--strategy` / `-s`**: `auto` (default), `human`, `face`, `center`.
- **Size:** **`--width` / `-W`** and **`--height` / `-H`** default to **1024** when omitted (square **1024×1024** when both omitted).
- **Paths:** **`--output` / `-o`** when exactly one input; default output **`{stem}-cropped.jpg`** beside each input; **`--force`** overwrites; **`--model-dir`** defaults to **`~/.imagecropper`**.
- **Flags:** **`--quiet`** suppresses banner, per-file lines, and summary (errors still emitted where required).

### Detection

- **Human:** **Ultralytics `YOLO("yolov8m.pt")`**; **COCO class 0 (person)**; when several person boxes exist, take **highest confidence**.
- **Face:** **OpenCV DNN Caffe** SSD; confidence threshold **0.5** (shared semantics for crop, anonymize eligibility, and enhance gating where the spec ties them).
- **Margins before aspect math:** expand each **`xyxy`** by **fixed fractions of box width/height**, clipped to the image; **face** boxes get **extra top** padding for headroom (**person** padding is symmetric).

### Crop geometry

- **Aspect-safe window:** smallest axis-aligned rectangle with the **same aspect ratio as target W:H** that contains the padded box, centered on the box; **uniform** downscale if it exceeds the image; shift to stay in bounds; **integer** crop width/height preserve that aspect before **`cv2.resize`** to the target.
- **Containment:** after integer **`w_i` / `h_i`**, **widen** along the output aspect (when the image allows) until the window is at least the padded bbox size; place the window by **clamping** the centered origin so the padded box stays inside the crop, then clip to image bounds.

### Status on stderr

- **Per input:** one stderr line with **status** (`ok` / `fail`), **elapsed ms**, **input basename**, **output basename** (em dash if no output path), **strategy label**, **target WxH**, truncated **error** on failure.
- **Banner + summary** when not **`--quiet`**; **Click** `style` for ANSI color when the environment supports it; footer is a compact summary (dot-separated fields, path arrow).

### Post-resize GFPGAN (`--enhance` / `--no-enhance`)

- **Default:** enhancement **on**. Run **GFPGAN** on the **resized** output when enhancement is on, **`--anon`** is off, and SSD finds a face on that resized frame; **`--anon`** forces **no** GFPGAN (strategy label notes enhancement skipped for anon).
- **Packaging:** **gfpgan** is a **core** dependency; cache **GFPGANv1.4.pth** under **`--model-dir`**.
- **Resilience:** on enhancement failure (load, inference), **save the unenhanced** resized image and record **`(enhance failed)`**; **`--no-enhance`** skips the GAN path entirely.
- **Compatibility:** small **torchvision** import shim for **basicsr** when **`torchvision.transforms.functional_tensor`** is absent.

### `crop --anon` (post-resize)

- **Dependency boundary:** **OpenCV SSD** box → expanded head rectangle → ellipse mask and **`cv2.inpaint`**; **no** portrait-segmentation stack (e.g. rembg) in the anonymize path.
- **Pipeline order:** region selection and **uniform resize** on an **unaltered** copy of the loaded image; **anonymize** runs on the **output-sized** BGR frame **after** resize (GFPGAN does not run when **`--anon`** is set).
- **Eligibility:** SSD on that **output-sized** frame; if **no** face → **skip** anonymization, **still save**, strategy label **`(anon skipped)`**.
- **Mask:** same style of **expanded axis-aligned head bbox** as in `anon.expand_face_bbox_for_head`, then a **filled ellipse** (**`LINE_AA`**) inscribed in that rectangle with each **semi-axis lengthened by 10 px** (clipped to raster bounds); **`cv2.inpaint`** (**TELEA**, fixed radius) with that binary mask.
- **Obscuring:** **GaussianBlur** on the **full** inpainted frame with **21×21** kernel (**10 px** half-size).
- **Edge composite:** **premultiplied alpha** on the **full** raster — soften the binary mask with **`(2n+1)×(2n+1)`** Gaussian where **`n = max(1, round(0.10 · d_min))`** and **`d_min`** is the ellipse **minor diameter** after the **+10 px** semi-axis grow; **renormalize** peak α to **1**; **`premult = blurred_BGR × α`**; stack **α** as a fourth channel; **`(2n+1)`** Gaussian on all four channels; **`out = src × (1 − α′) + premult′`** with **`src`** the **pre-inpaint** output-sized frame and **`α′`** clipped to **[0, 1]**. No tight ROI for this layer pass (Gaussian tails must not clip to a subrectangle).

**Consequences:** Normative detail lives in **`docs/SPEC.md`** (**CLI-005**–**CLI-009**, **DATA-002**, **DATA-004**–**DATA-005**, **NFR-BLD-006**); README points here for rationale. Runtime footprint includes **torch** and **OpenCV**; CI tests mock network and heavy model paths.
