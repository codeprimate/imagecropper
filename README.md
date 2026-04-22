# imagecropper

Small command-line tool for **square (or fixed-size) image crops** with person- and face-aware region selection, optional post-resize face anonymization, and optional GFPGAN enhancement after resize.

**Python 3.10+** · installs as the `imagecropper` console script.

## Install

From the repository root (recommended: [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
```

With optional GFPGAN enhancement (see [Enhancement](#enhancement)):

```bash
uv sync --extra enhance
```

Using pip (editable dev install):

```bash
pip install -e ".[dev]"
```

Runtime pulls in **PyTorch**, **OpenCV**, **Ultralytics (YOLO)**, and related stack. Some weights are fetched on first use (see below).

### First-run downloads (approximate sizes)

Sizes are **HTTP `Content-Length`** values checked in April 2026; release assets can change slightly over time.

| File | When it is needed | Where it is stored (defaults) | Approx. size |
|------|-------------------|-------------------------------|----------------|
| `deploy.prototxt` | Face-related paths (SSD): `face` / `auto` face branch, `--anon`, default **enhance** face gate | `--model-dir` (default `~/.imagecropper`) | ~28 KB |
| `res10_300x300_ssd_iter_140000.caffemodel` | Same as above | `--model-dir` | ~10 MB |
| `yolov8m.pt` | `human` or `auto` when the human branch runs | **Not** under `--model-dir`. Ultralytics resolves `yolov8m.pt` in the **current working directory** first, then under its global **`weights_dir`**, then downloads into the working directory as `./yolov8m.pt`. Inspect or change that directory with `yolo settings` (see [Ultralytics docs](https://docs.ultralytics.com/quickstart/#ultralytics-settings)). | ~50 MB |
| `GFPGANv1.4.pth` | Default **enhance** path when a face is seen on the resized output and the **`enhance`** extra is installed | `--model-dir` | ~330 MB |

**Rough totals:** face / auto / anon / enhance checks that only need SSD files: **~10 MB** under `--model-dir`. A run that also needs person detection adds **~50 MB** (YOLO, location as above). A successful GFPGAN run adds **~330 MB** under `--model-dir`.

## Quick start

```bash
# Default: 1024×1024, strategy auto, writes beside inputs as <stem>-cropped.jpg
imagecropper crop photo.jpg

# Explicit size and strategy
imagecropper crop -W 800 -H 800 -s face *.jpg

# Single input, explicit output
imagecropper crop -o out.jpg input.png

# Module invocation (same as the console script)
python -m imagecropper crop --help
```

Progress and summaries go to **stderr** so stdout stays pipe-friendly. Use `--quiet` to suppress banner and per-file progress (errors still print).

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

## Enhancement

Default behavior tries **GFPGAN** on the resized result when a face is detected. That path needs the optional **`enhance`** extra (`gfpgan`). Without it, crops still succeed; the run is labeled as enhancement failed or skipped per [docs/SPEC.md](docs/SPEC.md).

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

## Documentation

- [github.com/codeprimate/imagecropper](https://github.com/codeprimate/imagecropper) — source and issue tracker.
- [docs/SPEC.md](docs/SPEC.md) — CLI behavior, data formats, and requirements.
- [docs/DECISIONS.md](docs/DECISIONS.md) — design notes and trade-offs.
- [AGENTS.md](AGENTS.md) — contributor/agent workflow for this repo.

## License

This project is licensed under [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) (**CC BY-SA 4.0**). See the [`LICENSE`](LICENSE) file for the full legal text.

## Status

Alpha (**0.1.x**). Behavior is versioned with the package; treat CLI flags and defaults as the contract, with detail in the spec.
