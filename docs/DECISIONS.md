# Decisions

Record **architectural and behavioral decisions** here so future work does not re-litigate settled choices.

**Newest first.** Related choices that landed the same day **may** be merged into one dated section (subsections + a short consequences line) instead of many micro-entries.

Suggested shape for a standalone decision:

```text
## YYYY-MM-DD — Short title

**Context:** …
**Decision:** …
**Consequences:** …
```


## 2026-04-21 — Creative Commons BY-SA 4.0 for the repository

**Context:** The project needed an explicit public license aligned with **share-alike** redistribution of adaptations.

**Decision:** Apply **Creative Commons Attribution-ShareAlike 4.0 International** (**CC BY-SA 4.0**): ship the full legal text as **`LICENSE`** at the repository root; record **`license = "CC-BY-SA-4.0"`** and **`license-files = ["LICENSE"]`** in **`pyproject.toml`** (setuptools **≥77**). Remove the PyPI trove classifier that claimed **MIT**.

**Consequences:** Downstream users must follow **BY** (attribution) and **SA** (compatible license on adapted material) as defined in the legal text. Creative Commons positions these licenses mainly for creative works rather than software; if that mismatch matters for a consumer, they should treat this as a deliberate project choice or negotiate a separate grant.

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
- **Packaging:** **gfpgan** under optional **`[enhance]`**; cache **GFPGANv1.4.pth** under **`--model-dir`**.
- **Resilience:** on enhancement failure (missing extra, load, inference), **save the unenhanced** resized image and record **`(enhance failed)`**; **`--no-enhance`** skips the GAN path entirely.
- **Compatibility:** small **torchvision** import shim for **basicsr** when **`torchvision.transforms.functional_tensor`** is absent.

### `crop --anon` (post-resize)

- **Dependency boundary:** **OpenCV SSD** box → expanded head rectangle → ellipse mask and **`cv2.inpaint`**; **no** portrait-segmentation stack (e.g. rembg) in the anonymize path.
- **Pipeline order:** region selection and **uniform resize** on an **unaltered** copy of the loaded image; **anonymize** runs on the **output-sized** BGR frame **after** resize (GFPGAN does not run when **`--anon`** is set).
- **Eligibility:** SSD on that **output-sized** frame; if **no** face → **skip** anonymization, **still save**, strategy label **`(anon skipped)`**.
- **Mask:** same style of **expanded axis-aligned head bbox** as in `anon.expand_face_bbox_for_head`, then a **filled ellipse** (**`LINE_AA`**) inscribed in that rectangle with each **semi-axis lengthened by 10 px** (clipped to raster bounds); **`cv2.inpaint`** (**TELEA**, fixed radius) with that binary mask.
- **Obscuring:** **GaussianBlur** on the **full** inpainted frame with **21×21** kernel (**10 px** half-size).
- **Edge composite:** **premultiplied alpha** on the **full** raster — soften the binary mask with **`(2n+1)×(2n+1)`** Gaussian where **`n = max(1, round(0.10 · d_min))`** and **`d_min`** is the ellipse **minor diameter** after the **+10 px** semi-axis grow; **renormalize** peak α to **1**; **`premult = blurred_BGR × α`**; stack **α** as a fourth channel; **`(2n+1)`** Gaussian on all four channels; **`out = src × (1 − α′) + premult′`** with **`src`** the **pre-inpaint** output-sized frame and **`α′`** clipped to **[0, 1]**. No tight ROI for this layer pass (Gaussian tails must not clip to a subrectangle).

**Consequences:** Normative detail lives in **`docs/SPEC.md`** (**CLI-005**–**CLI-009**, **DATA-002**, **DATA-004**–**DATA-005**, **NFR-BLD-006**); README points here for rationale. Runtime footprint includes **torch** and **OpenCV**; CI tests mock network and heavy model paths.
