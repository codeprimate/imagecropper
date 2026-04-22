# Agent notes — imagecropper

## What this is

Small **Python CLI** package (**Python 3.10+**, setuptools, `src/` layout). Primary command: **`imagecropper crop`** — fixed-size crops with person/face-aware selection, optional **face anonymization** (`--anon`), and optional **GFPGAN** post-resize enhancement (see `docs/SPEC.md`).

Runtime stack includes **Click**, **Pillow**, **NumPy**, **OpenCV (headless)**, **PyTorch**, **Ultralytics (YOLO)** for detection, plus **requests** for model downloads. OpenCV SSD assets and **GFPGAN** weights cache under **`--model-dir`** (default **`~/.imagecropper`**); **YOLOv8m** (`yolov8m.pt`) is handled by **Ultralytics** (typically **cwd** or its **`weights_dir`**, not `imagecropper`’s model dir—see README). Optional **`enhance`** extra installs **GFPGAN** (`uv sync --extra enhance`); without it, crops still work and enhancement is skipped per the spec.

There is **no GitHub Actions CI** in this repository yet; rely on local checks and pre-commit.

## Layout

- `src/imagecropper/` — application code; console entry `imagecropper.cli:main`
- `tests/` — pytest + coverage (`--cov-fail-under=80`)
- `doc/` — optional scratch notes (see `doc/README.md`); normative docs live under `docs/`
- `docs/DECISIONS.md` — decision log (why something was chosen or changed)
- `docs/SPEC.md` — product/CLI specification (what the tool does and promises)

## Development workflow (required)

You **MUST** follow this end-to-end for any change beyond trivial fixes (typos, comments, formatting only). The goal is **mindful, minimal work**: the smallest correct change, with **no drive-by refactors**, no speculative features, and no scope creep beyond what was asked and what the spec allows.

**1 — Scope and intent**  
Name the smallest slice that satisfies the request. If requirements are ambiguous, resolve that **before** writing code (questions, spec/decision updates, or an explicit note in `docs/DECISIONS.md`). Do not “improve” unrelated modules in the same batch.

**2 — Read the existing contract**  
Skim `docs/SPEC.md` and `docs/DECISIONS.md`. Open the tests and modules that already cover the behavior path (`tests/` names usually mirror `src/imagecropper/`). Treat those tests as the living definition of current behavior.

**3 — Align docs before coding when needed**  
If the planned work **contradicts** the spec or recorded decisions, update `docs/DECISIONS.md` (dated entry, or adjust spec if that is the agreed direction) **before** substantive implementation so intent and artifacts stay consistent.

**4 — Implement minimally**  
Touch only what the scoped change requires. Match existing patterns, types, and structure; every line in the diff should serve the request.

**5 — Test comprehensively (same batch)**  
Treat **`tests/`** as part of the shipped product. Add or extend tests **in the same PR / change batch** as the code: cover the **happy path**, **relevant error and edge paths** (invalid inputs, missing resources, strategy boundaries, CLI exits), and regressions for anything you fixed. Prefer extending an existing test file unless the surface is genuinely new.

Run `uv run pytest` and fix failures before the work is done. New or changed behavior **SHALL** be exercised by tests unless truly impractical—in that case, record why in `docs/DECISIONS.md` and still keep line coverage **≥80%**. Pure refactors keep tests green; change tests only when they encoded wrong expectations.

**6 — Document comprehensively (same batch)**  
For **any** user-visible change (CLI flags, defaults, exit codes, outputs, formats, guarantees): update `docs/SPEC.md` so it matches the implementation. For **non-obvious** engineering choices (trade-offs, rejected alternatives, coupling, compatibility): append a dated entry to `docs/DECISIONS.md`.

**7 — Verify**  
Run the **Tooling** commands below (at least `pytest`, `ruff`, `black`, `mypy`, and `pre-commit` when you touched hooks or wide formatting). After CLI changes, confirm `uv run imagecropper --help` matches the spec.

**Trivial changes** — Documentation and new tests are not required for changes that alter **no** behavior and **no** public contract (e.g. typo-only comments). Still keep the tree green if you run checks.

## Tooling

Use **uv** for environments and commands.

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run black .
uv run mypy src
uv run pre-commit run --all-files
uv run imagecropper --help
```

For work that exercises **GFPGAN** paths, use `uv sync --extra dev --extra enhance` so the optional dependency is installed.

The mypy pre-commit hook is limited to `src/` so it does not need pytest stubs in the hook environment.

## Quality bar

- Follow **Development workflow** above for every substantive change; **`uv run pytest`** is part of step 7, not optional.
- Coverage must stay **≥80%** (pytest-cov).
- Format: **Black**; lint: **Ruff**; types: **Mypy** (strict settings in `pyproject.toml`).
