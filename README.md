# imagecropper

**imagecropper** is a small command-line tool that crops images to a **fixed width and height** (defaults: **1024×1024**). It tries to keep **people and faces** in frame using detectors, can **blur faces** on the result (`--anon`), and by default **sharpens faces** on the resized output with **GFPGAN** when a face is detected (turn off with `--no-enhance`).

- **Python 3.10+**
- **Entry point:** `imagecropper` (same behavior as `python -m imagecropper`)
- **Outputs:** JPEG by default (`{input-stem}-cropped.jpg` beside each input); see [Quick start](#quick-start)

**On this page:** [Requirements](#requirements) · [Install](#install) · [First-run downloads](#first-run-downloads-approximate-sizes) · [Quick start](#quick-start) · [Commands](#commands) · [Enhancement](#enhancement) · [Development](#development) · [Documentation](#documentation) · [License](#license) · [Status](#status)

---

## Requirements

- **Python** 3.10 or newer.
- **Disk and time:** the runtime stack includes **PyTorch**, **OpenCV**, **Ultralytics (YOLO)**, and **GFPGAN** (`gfpgan`). First runs download model weights (see [First-run downloads](#first-run-downloads-approximate-sizes)). A **virtual environment** is strongly recommended so this does not mix with other projects.

Inputs are read with **Pillow** (common raster formats). Details are in [docs/SPEC.md](docs/SPEC.md).

---

## Install

### Install from GitHub with pip (`git+`)

You do **not** need to clone the repo. Use a [PEP 508](https://peps.python.org/pep-0508/) direct URL requirement:

**Latest `main` branch:**

```bash
pip install "imagecropper @ git+https://github.com/codeprimate/imagecropper.git"
```

**Pinned release (reproducible installs):**

```bash
pip install "imagecropper @ git+https://github.com/codeprimate/imagecropper.git@v0.1.1"
```

**With development tools** (pytest, ruff, black, mypy, pre-commit):

```bash
pip install "imagecropper[dev] @ git+https://github.com/codeprimate/imagecropper.git"
```

Replace the URL’s branch or tag (`@main`, `@v0.1.1`, etc.) to match the revision you want. The package uses a `src/` layout; `pip` builds from the repository root automatically. **GFPGAN** is included in the default dependency set (no separate extra).

---

### Clone the repo, then sync with uv (recommended for contributors)

From the repository root:

```bash
uv sync
```

---

### Editable install from a clone (pip)

After cloning:

```bash
pip install -e ".[dev]"
```

---

### Check that it worked

```bash
imagecropper --help
# or
python -m imagecropper --help
```

---

## First-run downloads (approximate sizes)

Sizes are **HTTP `Content-Length`** values checked in April 2026; release assets can change slightly over time.

| File | When it is needed | Where it is stored (defaults) | Approx. size |
|------|-------------------|-------------------------------|----------------|
| `deploy.prototxt` | Face-related paths (SSD): `face` / `auto` face branch, `--anon`, default **enhance** face gate | `--model-dir` (default `~/.imagecropper`) | ~28 KB |
| `res10_300x300_ssd_iter_140000.caffemodel` | Same as above | `--model-dir` | ~10 MB |
| `yolov8m.pt` | `human` or `auto` when the human branch runs | **Not** under `--model-dir`. Ultralytics resolves `yolov8m.pt` in the **current working directory** first, then under its global **`weights_dir`**, then downloads into the working directory as `./yolov8m.pt`. Inspect or change that directory with `yolo settings` (see [Ultralytics docs](https://docs.ultralytics.com/quickstart/#ultralytics-settings)). | ~50 MB |
| `GFPGANv1.4.pth` | Default **enhance** path when a face is seen on the resized output and enhancement is on | `--model-dir` | ~330 MB |

**Rough totals:** face / auto / anon / enhance checks that only need SSD files: **~10 MB** under `--model-dir`. A run that also needs person detection adds **~50 MB** (YOLO, location as above). A successful GFPGAN run adds **~330 MB** under `--model-dir`.

---

## Quick start

```bash
# Default: 1024×1024, strategy auto, writes beside inputs as <stem>-cropped.jpg
imagecropper crop photo.jpg

# Explicit size and strategy
imagecropper crop -W 800 -H 800 -s face *.jpg

# Single input, explicit output
imagecropper crop -o out.jpg input.png

# Same as the console script
python -m imagecropper crop --help
```

Progress and summaries go to **stderr** so stdout stays pipe-friendly. Use `--quiet` to suppress banner and per-file progress (errors still print).

---

## Commands

| Command | Purpose |
|--------|---------|
| `imagecropper` | Print top-level help and exit `0`. |
| `imagecropper crop …` | Crop each input to `--width` × `--height` (defaults **1024**). |

### `crop` options (summary)

| Option | Description |
|--------|-------------|
| `-W`, `--width` / `-H`, `--height` | Output dimensions (positive integers). |
| `-s`, `--strategy` | `auto` (default), `human`, `face`, or `center`. |
| `--model-dir` | Cache for OpenCV SSD + GFPGAN weights (default: `~/.imagecropper`); does not relocate Ultralytics `yolov8m.pt` (see [First-run downloads](#first-run-downloads-approximate-sizes)). |
| `-o`, `--output` | Output path; only when exactly one input is given. |
| `--force` | Overwrite an existing output file. |
| `--quiet` | Less chatter on stderr. |
| `--anon` | After crop and resize: inpaint + blur inside an expanded SSD face oval on the output-sized frame. |
| `--enhance` / `--no-enhance` | Post-resize GFPGAN when a face is seen on the resized image (default: **on**; skipped with `--anon` or if deps/models fail). |

Default output when `-o` is omitted: `{input-stem}-cropped.jpg` next to each input.

### Exit codes

- **0** — all inputs processed successfully for that command.
- **1** — `crop`: one or more inputs failed (all inputs are still attempted).
- Non-zero — Click usage / bad parameters.

---

## Enhancement

**GFPGAN** (`gfpgan`) is a **normal runtime dependency**: a standard install can run the default post-resize enhancement path when a face is detected. Use **`--no-enhance`** to skip it entirely. If enhancement fails at runtime (for example model load or inference), the tool still saves the unenhanced crop and records **`(enhance failed)`** per [docs/SPEC.md](docs/SPEC.md).

---

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run black .
uv run mypy src
uv run pre-commit run --all-files
```

Line coverage is enforced at **≥80%** (see `pyproject.toml`).

---

## Documentation

- [github.com/codeprimate/imagecropper](https://github.com/codeprimate/imagecropper) — source and issue tracker.
- [docs/SPEC.md](docs/SPEC.md) — CLI behavior, data formats, and requirements.
- [docs/DECISIONS.md](docs/DECISIONS.md) — design notes and trade-offs.
- [AGENTS.md](AGENTS.md) — contributor and agent workflow for this repo.

---

## License

This project is licensed under [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) (**CC BY-SA 4.0**). See the [`LICENSE`](LICENSE) file for the full legal text.

---

## Status

Alpha (**0.1.x**). Behavior is versioned with the package; treat CLI flags and defaults as the contract, with detail in the spec.

When reporting issues, include your **Python version**, **install method** (e.g. `git+` vs editable), and the **exact command** you ran.
