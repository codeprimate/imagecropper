# imagecropper — specification

| Field | Value |
|--------|--------|
| **SPEC-ID** | `IC-SPEC-001` |
| **Title** | imagecropper — software requirements |
| **Status** | Draft |
| **Normative sources** | `pyproject.toml`, `.pre-commit-config.yaml`, `AGENTS.md` (process only) |

Changes to externally visible behavior, CLI contracts, or normative tooling rules **SHALL** be reflected here in the same change batch as the implementation, unless recorded only as rationale in [`DECISIONS.md`](DECISIONS.md) where that file explicitly defers normative text to this spec.

---

## 1. Identifier convention

Requirements use immutable codes. Once assigned, a code **SHALL NOT** be reused for a different meaning; superseded statements **SHALL** be struck through or replaced in place with a dated note.

| Prefix | Domain |
|--------|--------|
| `DOC-*` | Document meta and maintenance rules |
| `PROD-*` | Product intent and scope |
| `CLI-*` | Command-line interface |
| `DATA-*` | Inputs, outputs, and on-disk formats |
| `ERR-*` | Exit codes and error reporting |
| `NFR-BLD-*` | Build, packaging, and runtime platform |
| `NFR-QA-*` | Verification, coverage, style, and static analysis |
| `NFR-HOST-*` | Repository-hosted automation and editor defaults |
| `OOS-*` | Explicitly out of scope (non-requirements) |

---

## 2. Document control

**DOC-001 — Normative precedence**  
For conflicts among project artifacts, precedence for implementable behavior **SHALL** be: this specification, then `pyproject.toml` tool configuration where it encodes enforced checks, then the working tree source code. Rationale not duplicated here **MAY** appear in `DECISIONS.md`.

**DOC-002 — Traceability**  
Each `CLI-*`, `DATA-*`, and `ERR-*` item **SHALL** be satisfiable by a test or observable CLI behavior unless marked *Deferred*.

---

## 3. Product

**PROD-001 — Purpose**  
The product **SHALL** be a distributable Python package providing a command-line tool for image cropping and related image utilities. *Deferred:* detailed user stories and acceptance criteria until commands beyond bootstrap exist.

**PROD-002 — Distribution form**  
The product **SHALL** be installable as a setuptools-based package with a console script entry point as specified under `CLI-*`.

---

## 4. Command-line interface

**CLI-001 — Console entry name**  
The installed console script **SHALL** be named `imagecropper` and **SHALL** invoke `imagecropper.cli:main`.

**CLI-002 — Module execution**  
The package **SHALL** support invocation as `python -m imagecropper` with the same functional entry as `CLI-001`.

**CLI-003 — Version and help**  
The CLI **SHALL** expose `--version` reporting the package version and **SHALL** expose `--help` on the top-level group and on each subcommand describing available options for that command.

**CLI-004 — Default invocation**  
Invoking `imagecropper` with no subcommand **SHALL** emit top-level usage text and exit successfully (exit code `0`).

**CLI-005 — `crop` subcommand**  
The CLI **SHALL** expose a `crop` subcommand that accepts one or more existing regular file paths as `INPUT` arguments, accepts optional `--width` / `-W` and `--height` / `-H` positive integers (each defaulting to **1024** when omitted), accepts `--strategy` / `-s` with values `auto` (default), `human`, `face`, or `center`, accepts optional `--model-dir` (default `~/.imagecropper`), optional `--output` / `-o` when exactly one `INPUT` is given (**SHALL NOT** be combined with `--output-dir`), optional `--output-dir` naming a directory for primary crop JPEGs when `--output` is omitted (see **CLI-006**), optional `--force` to overwrite an existing output file, optional `--quiet` to suppress the banner, per-file progress lines, and summary while still emitting per-file error lines to standard error, optional `--anon` to run **post-resize** face anonymization on the output-sized raster after region selection and uniform resize (see **DATA-002**), ``--enhance`` / ``--no-enhance`` to enable or disable **post-resize** GFPGAN face enhancement when the conditions in **CLI-009** hold (**default:** enhancement **on**; **GFPGAN** is a core runtime dependency per **NFR-BLD-003** and **NFR-BLD-006**), and optional ``--debug`` to write a per-input debug JPEG as specified in **DATA-006**.

**CLI-008 — `crop --anon` (post-resize inpaint)**  
When `--anon` is set, the implementation **SHALL** complete the normal ``--strategy`` region selection and **uniform resize** to ``--width``×``--height`` on an **unaltered** in-memory copy of the source (no anonymization before region selection). It **SHALL** then run OpenCV SSD face detection on that **output-sized** BGR raster (same confidence rule as face cropping). If a face is detected, it **SHALL** compute the same **axis-aligned expanded** rectangle around that box as for the anonymize mask today, then build a **filled ellipse** inscribed in that rectangle whose **semi-axes are each larger by 10 pixels** than the strict inscribed semi-axes (subject to raster bounds), **SHALL** ``cv2.inpaint`` using that mask (OpenCV ``inpaint`` only; no extra segmentation dependency), **SHALL** then apply **Gaussian blur** with a **21×21** kernel (10-pixel blur scale) to the inpainted image for stronger obscuring, **SHALL** derive the ellipse **semi-axes** from the same expanded head bbox and **+10 px** semi-axis grow used for the mask, **SHALL** set the edge-feather Gaussian **kernel half-size** ``n`` to **``max(1, round(0.10 · d_min))``** where ``d_min`` is the oval’s **minor diameter** ``2 · min(semi-axis width, semi-axis height)``, **SHALL** build **soft α** by Gaussian-blurring the binary mask with kernel **``2n+1``** and **renormalizing** peak **α** to **1**, **SHALL** multiply that blurred BGR by **soft** α to form premultiplied ``(B·α, G·α, R·α)``, **SHALL** stack **α** as a fourth float channel, **SHALL** apply a second Gaussian with the **same ``2n+1``** kernel to all four channels **on the full output-sized raster** (cropping **SHALL NOT** clip Gaussian tails), and **SHALL** composite over the **pre-inpaint source** with **``out = src·(1 − α′) + premult′``** (clipping ``α′`` to ``[0, 1]``), then write that result. This **SHALL NOT** rely on linear interpolation between ``src`` and blurred BGR using a blurred mask alone (which leaves a visible inpaint discontinuity); the premultiplied blur is the normative edge treatment. If no face is detected on the output-sized raster, it **SHALL** skip anonymization, **SHALL** still save the current in-memory output-sized raster (subject to **CLI-009** for GFPGAN when ``--anon`` is not set), and **SHALL** record that the anonymization step was skipped in the per-file strategy label (e.g. suffix ``(anon skipped)``). When anonymization runs, the label **SHALL** indicate that (e.g. suffix ``(anon)``).

**CLI-006 — `crop` default output path**  
When `--output` is omitted and `--output-dir` is omitted, each input **SHALL** be written beside that input. When `--output-dir` is set and `--output` is omitted, each primary crop **SHALL** use the **same basenames** as in the beside-input case but rooted under ``output_dir`` (the implementation **MAY** create ``output_dir`` and parents on write). When exactly **one** subject-driven crop is produced for that input (``human``, ``face``, or ``auto`` with a single person or single face, or ``center``), that file **SHALL** be named ``{stem}-cropped.jpg`` (JPEG), matching the legacy script naming. When **two or more** subject-driven crops are produced from the same input, the implementation **SHALL** write ``{stem}-cropped-{i}.jpg`` for ``i = 1 .. N`` with a **uniform** zero-pad width ``W = max(2, len(str(N)))`` applied to every index in that run (so lexicographic order matches confidence order).

When ``--output`` / ``-o`` is given (exactly one ``INPUT``), it **SHALL** name the output file **only** when that run produces **exactly one** crop for that input; ``--output-dir`` **SHALL NOT** appear in the same invocation. If region selection yields **more than one** crop, the implementation **SHALL NOT** write any crop file for that input and **SHALL** report a per-input error explaining that a single ``--output`` path cannot name multiple subjects and that ``--output`` should be omitted to use sidecar numbered filenames.

**CLI-007 — `crop` status output channel**  
Progress and summary lines for `crop` (banner, separator, colored per-file lines, footer) **SHALL** be written to standard error so standard output remains available for piping. After each such write, the implementation **SHALL** flush standard error (and **SHALL** enable line-buffered standard error at startup when the platform supports it) so lines become visible promptly when standard error is captured or piped. When not ``--quiet``, the implementation **SHALL** emit interim status lines to standard error while each input is processed (for example before region detection runs and before each per-subject resize, enhancement, and save), and **SHALL** flush standard error immediately after each interim line the same as for other progress writes. When not ``--quiet``, each successful or failed input **SHALL** produce one **summary** line containing status, elapsed milliseconds, input basename, output basename (or an em dash when there is no output path), strategy label, and target ``WIDTHxHEIGHT``. When more than one crop file is written for that input, the output basename cell **SHALL** show the first written file’s basename followed by `` (+M more)`` where ``M = N - 1``. When ``--debug`` is set and a debug JPEG path applies to that input, the successful per-file line **SHALL** also indicate the debug file basename (for example ``dbg=`` followed by the filename). ANSI coloring **SHALL** follow terminal capability and common no-color conventions (for example ``NO_COLOR``). The first banner line **SHALL** include the effective output raster dimensions as ``out=WIDTHxHEIGHT``, matching the ``--width`` and ``--height`` values used for that invocation (after defaults are applied). When ``--no-enhance`` is in effect, that banner line **SHALL** also include ``enhance=off``.

**CLI-009 — `crop --enhance` (post-resize GFPGAN)**  
By default, the effective **enhance** flag **SHALL** be **on** (GFPGAN may run per the rules below). When ``--no-enhance`` is given, the implementation **SHALL NOT** run GFPGAN and **SHALL NOT** add enhancement-related suffixes to the strategy label. When enhancement is **on** and ``--anon`` **is** also set, the implementation **SHALL NOT** run GFPGAN; the per-file strategy label **SHALL** include a suffix such as ``(enhance skipped: anon)``. When enhancement is **on**, ``--anon`` is not set, and OpenCV SSD face detection on the **resized** output (same confidence rule as face cropping) finds **no** face, the implementation **SHALL NOT** run GFPGAN and **SHALL** record a suffix such as ``(enhance skipped)``. When enhancement is **on**, ``--anon`` is not set, and a face **is** detected on the resized raster, the implementation **SHALL** run GFPGAN on that resized BGR image, then save the result; the strategy label **SHALL** include a suffix such as ``(enhance)``. If enhancement is attempted but fails (for example model load error or inference error), the implementation **SHALL** save the **unenhanced** resized image unchanged and **SHALL** record a suffix such as ``(enhance failed)``.

---

## 5. Data formats

**DATA-001 — Raster images**  
The implementation **SHALL** use Pillow for reading and writing raster files on disk. Arrays passed to OpenCV or PyTorch for inference **MAY** be derived from Pillow-loaded pixels. *Deferred:* supported formats, color spaces, and metadata handling beyond common RGB/RGBA inputs and JPEG/PNG outputs.

**DATA-002 — ``--anon`` processing order**  
With ``--anon``, region selection and uniform resize **SHALL** use the loaded pixels **without** anonymization. Anonymization **SHALL** mutate only the in-memory **output-sized** BGR raster **after** resize to ``--width``×``--height`` and **after** any GFPGAN step permitted by **CLI-009** (when ``--anon`` is set, GFPGAN does not run, so anonymization applies directly to the resize). On-disk inputs **SHALL NOT** be modified.

**DATA-003 — Detector box (person or face)**  
For ``human`` (and for ``auto`` when the human branch runs), person detection **SHALL** use YOLO COCO class 0 only. Each person ``xyxy`` **SHALL** be clipped to the source image (**no** relative-margin expansion around that box). When **several** person boxes appear, the implementation **SHALL** emit **one** crop per box, ordered by **descending detector confidence** (highest first). APIs that return a **single** person box (for example the highest-confidence crop helper) **SHALL** continue to use the **highest-confidence** clipped box. Each clipped rectangle is a **detection box** fed into **DATA-004** for its respective output file.

For ``face`` (and for ``auto`` when only the face branch runs), the implementation **SHALL** consider every SSD detection whose confidence is **strictly greater** than the same threshold used for single-face selection. Detections **SHALL** be ordered by **descending confidence**. For each detection, the implementation **SHALL** expand the SSD face ``xyxy`` by **fixed relative margins** (fractions of box width and height), clipping to the source image, **before** the aspect step. Face cropping **SHALL** use **stronger upward** than downward margin so headroom is preserved relative to a tight face rectangle. The resulting padded rectangle for each face is the **detection box** fed into **DATA-004** for that face’s output. Single-face helpers **SHALL** use the **highest-confidence** face above the threshold.

When multiple subject crops are written, the per-file **strategy** label on stderr **SHALL** append `` ×N`` (space, Unicode multiplication sign, count) after the first crop’s full suffix chain (including anonymization and enhancement suffixes per **CLI-008** / **CLI-009**); per-output suffix differences on later crops **MAY** differ from that string without being enumerated on the progress line.

**DATA-004 — Aspect expansion from the detection box**  
Given the **detection box** from **DATA-003** and the requested output ratio ``--width`` : ``--height``, the implementation **SHALL** choose a source **crop rectangle** that:

1. **Contains** the entire detection box (no part of that box lies outside the crop).
2. Has aspect ratio **matching** ``--width`` : ``--height`` (up to integer rounding of pixel width and height).
3. Lies **entirely inside** the source image (clip to image extents; no crop pixel outside the raster).

**Expansion order (normative intent):** grow from the detection box toward the **target aspect** while respecting (3). **SHALL** extend **toward the nearest image boundary first** (i.e. prefer growth along the direction(s) that reach the closest source-image edge—top, bottom, left, or right—from the current crop or box, using the shortest slack to an edge as the primary expansion axis or axes), **then** extend the **other** axis or axes as needed so the crop matches the output aspect and still obeys (1)–(3). When several edges tie or geometry is ambiguous, the implementation **MAY** choose among valid crops arbitrarily provided (1)–(3) hold.

When the primary edge is the **bottom** (smallest slack to the image bottom), vertical placement **SHALL** use the **minimum** legal crop origin **`y0`** within the valid vertical interval (equivalently: the same **`y0`** rule as when the primary edge is the **top**—favor **headroom** above the detection box top rather than anchoring to the bottom of the valid interval). When the primary edge is the **left** or **right**, vertical placement **SHALL** use that same **minimum** legal **`y0`** (not vertical centering on the detection box), so wide person boxes do not shift the crop downward and clip the head. When no legal **`y0`** simultaneously contains the full detection box and fits the image at the chosen aspect window (empty vertical interval), the implementation **SHALL** set **`y0 = max(0, min(y1, H - h))`** where **`y1`** is the clipped detection top, **`H`** the source height, and **`h`** the crop height—i.e. align the crop top with the detection top when possible—then continue to satisfy horizontal rules as in the existing degenerate branch.

When no rectangle satisfies (1)–(3), the implementation **SHALL** use a **best-effort** in-bounds crop at the output aspect that minimizes violation of (1) (largest possible crop inside the image at that aspect); *Deferred:* exact tie-breaking for impossible cases.

``crop_one`` **SHALL** ``resize`` the extracted patch to ``--width``×``--height`` with **uniform** scaling (no anamorphic squish).

**DATA-005 — ``--enhance`` processing order**  
GFPGAN enhancement **SHALL** run only on the in-memory **output-sized** BGR raster **after** uniform resize to ``--width``×``--height`` and **before** writing the output file, and only under the rules in **CLI-009**. The default ``GFPGANv1.4`` weights file **SHALL** be stored under ``--model-dir`` (same cache root as other models) when first needed. On-disk inputs **SHALL NOT** be modified by this step.

**DATA-006 — ``crop --debug`` overlay image**  
When ``--debug`` is set, after the source raster is loaded and **before** region selection, the implementation **SHALL** write ``{input-stem}-debug.jpg`` beside that input (same directory rule as **CLI-006** for the stem), as a JPEG. The image **SHALL** be the **full-size** loaded source (not the cropped patch) with axis-aligned rectangles and text overlays for the same detector/crop windows used for labeling in that run: for ``center``, the ``center`` crop window; for ``human``, **every** clipped person ``xyxy`` used for cropping, with role labels ``person_01``, ``person_02``, … in the same order as confidence-descending enumeration; for ``face``, for each face above threshold, both the raw SSD box and the margin-padded box, with labels ``face_ssd_01`` / ``face_padded_01``, ``face_ssd_02`` / ``face_padded_02``, …; for ``auto``, **only** person boxes when at least one person exists (same indexing as ``human``); otherwise, when no person exists, all face pairs as for ``face``; otherwise the center window. Each drawn rectangle **SHALL** include a role label and **SHALL** label each corner with its ``(x, y)`` pixel coordinates in source-image space (integer coordinates after the same clipping applied for drawing). Role and corner text **SHALL** be laid out **inside** that rectangle (with padding from its edges) using OpenCV’s bottom-left text origin so strings are not clipped when the rectangle touches the image border; when a string does not fit inside the rectangle, it **MAY** be omitted for that box. If ``--force`` is not set and ``{stem}-debug.jpg`` already exists, the command **SHALL** refuse that input with the same overwrite policy as the primary crop output. The debug image **SHALL** be written even when the subsequent crop step fails (for example ``human`` with no person), so long as the source was loaded and overwrite checks passed.

Overwrite refusal for crop outputs **SHALL** apply to **every** output path that would be written for that input (including all numbered sidecars when ``N > 1``) before any crop file is written; when ``--output`` is omitted, the pre-load overwrite check **SHALL** apply only to the debug JPEG when ``--debug`` is set (because the count ``N`` of sidecar names is not known until after detection).

---

## 6. Errors and exit codes

**ERR-001 — Success**  
Successful completion of a command **SHALL** use process exit code `0`.

**ERR-002 — `crop` partial or total failure**  
If any input file fails during `imagecropper crop`, the process **SHALL** exit with code `1` after processing all inputs. Usage or parameter errors **SHALL** use Click’s normal non-zero exit conventions.

---

## 7. Non-functional — build and packaging

**NFR-BLD-001 — Python language version**  
The package **SHALL** declare and support `requires-python >= 3.10` as specified in `pyproject.toml`.

**NFR-BLD-002 — Source layout**  
Application source **SHALL** reside under `src/imagecropper/` and **SHALL** be discovered via setuptools `where = ["src"]`.

**NFR-BLD-003 — Runtime dependencies**  
The runtime **SHALL** depend on **Click**, **Pillow**, **NumPy**, **opencv-python-headless**, **requests**, **PyTorch**, **ultralytics** (default human detection: **YOLOv8m** COCO weights via `YOLO("yolov8m.pt")`), and **gfpgan** (post-resize face enhancement per **CLI-009**) with lower bounds as declared in `[project.dependencies]` in `pyproject.toml`.

**NFR-BLD-004 — Reproducible resolution**  
The repository **SHALL** contain a `uv.lock` file produced and updated with **uv** for locked dependency resolution.

**NFR-BLD-005 — Development environment**  
Development workflows **SHALL** assume **uv**. Installing development tools **SHALL** be achievable with `uv sync --extra dev`, which **SHALL** include at minimum: pytest, pytest-cov, pytest-mock, Black, Ruff, Mypy, types-Pillow, types-requests, pre-commit, build, twine (as listed under `[project.optional-dependencies].dev`).

**NFR-BLD-006 — GFPGAN packaging**  
The package **SHALL** list **gfpgan** (and its declared lower bound) in `[project.dependencies]` so a normal install includes post-resize GFPGAN support without enabling an optional extra. A standard ``uv sync`` or ``pip install`` of the package **SHALL** therefore satisfy **CLI-009**’s dependency on **GFPGAN** for the happy path; **CLI-009** ``(enhance failed)`` **SHALL** still apply when enhancement is attempted but fails at runtime (for example weights or load/inference errors), not because **gfpgan** was omitted from install-time resolution.

**NFR-BLD-007 — Documented first-run downloads**  
The repository **SHALL** ship user-facing documentation (``README.md``) that enumerates each weight or definition file the implementation may download at runtime, with approximate on-disk size and the feature path that triggers the fetch. That documentation **SHALL** distinguish artifacts stored under ``--model-dir`` (OpenCV SSD files and ``GFPGANv1.4.pth``) from **YOLOv8m** checkpoint resolution performed by **Ultralytics** (``yolov8m.pt``), which **SHALL NOT** be implied to live under ``--model-dir`` by default.

---

## 8. Non-functional — quality and verification

**NFR-QA-001 — Test layout**  
Automated tests **SHALL** live under `tests/` and **SHALL** be collected by pytest per `[tool.pytest.ini_options]` in `pyproject.toml`.

**NFR-QA-002 — Markers**  
Pytest **SHALL** run with strict marker registration (`--strict-markers`).

**NFR-QA-003 — Line coverage threshold**  
The pytest run configured for this repository **SHALL** enforce a minimum line coverage of **80%** on the package under measurement (`--cov-fail-under=80`).

**NFR-QA-004 — Coverage scope**  
Coverage measurement **SHALL** include `src/imagecropper` and **SHALL** omit `src/imagecropper/__main__.py` from coverage accounting.

**NFR-QA-005 — Formatter**  
Python sources **SHALL** conform to **Black** with line length **100** and Black `target-version` through **py312**, as configured under `[tool.black]`.

**NFR-QA-006 — Linter**  
Python sources **SHALL** pass **Ruff** check with line length **100**, `target-version` **py310**, and the rule set under `[tool.ruff.lint]` in `pyproject.toml`.

**NFR-QA-007 — Static typing**  
The `src/imagecropper` tree **SHALL** type-check under **Mypy** with the strictness options under `[tool.mypy]` in `pyproject.toml`, including `ignore_missing_imports` for `click.*` as configured; additional overrides **MAY** be added when new third-party modules are introduced.

**NFR-QA-008 — Pre-commit hooks**  
The repository **SHALL** ship `.pre-commit-config.yaml` invoking, at minimum: Black; Ruff with fix and non-zero exit on fix; Mypy with `types-Pillow` and `types-requests` as additional dependencies for the hook environment.

**NFR-QA-009 — Pre-commit Mypy scope**  
The Mypy pre-commit hook **SHALL** be restricted to paths matching `^src/` so the hook environment does not require pytest or test-only typing stubs.

---

## 9. Non-functional — hosting and editor

**NFR-HOST-001 — Hosted continuous integration**  
*Deferred:* There **SHALL** be no requirement for GitHub Actions or other hosted CI until an `OOS-*` or `NFR-HOST-*` item is superseded by a new requirement.

**NFR-HOST-002 — Editor defaults**  
The repository **MAY** include committed `.vscode/settings.json` for Black, Ruff, and pytest; `.vscode/` **SHALL NOT** be listed in `.gitignore` so those defaults **MAY** remain versioned.

---

## 10. Out of scope

**OOS-001 — Hosted CI**  
Hosted continuous integration (e.g. GitHub Actions) is **not** required by this specification at `IC-SPEC-001` status Draft.

**OOS-002 — Non-Python runtimes**  
Native rewrites or alternate language implementations are out of scope unless introduced under a new `PROD-*` or `NFR-BLD-*` requirement.
