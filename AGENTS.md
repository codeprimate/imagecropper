# Agent notes — imagecropper

## What this is

Small **Python CLI** package (installable with setuptools, `src/` layout). Runtime stack: **Click** + **Pillow**. There is **no GitHub Actions CI** in this repository yet; rely on local checks and pre-commit.

## Layout

- `src/imagecropper/` — application code; console entry `imagecropper.cli:main`
- `tests/` — pytest + coverage (`--cov-fail-under=80`)
- `doc/` — optional scratch or auxiliary notes
- `docs/DECISIONS.md` — decision log (why something was chosen or changed)
- `docs/SPEC.md` — product/CLI specification (what the tool does and promises)

## Documentation workflow (required)

When you **add, change, or remove** user-visible behavior, CLI flags, defaults, file formats, or architectural boundaries:

1. **Before** substantive implementation: skim `docs/DECISIONS.md` and `docs/SPEC.md`; if the work contradicts them, update those docs first (or add a dated decision) so the plan matches agreed intent.
2. **After** the feature is implemented (same PR / same change batch): update `docs/SPEC.md` so it reflects current behavior (commands, options, exit codes, examples as needed).
3. Whenever you make a **non-obvious choice** (trade-offs, rejected alternatives, coupling, compatibility): append a dated entry to `docs/DECISIONS.md`.

If a change is trivial (typos, internal refactor with no behavior change), doc updates are not required.

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

The mypy pre-commit hook is limited to `src/` so it does not need pytest stubs in the hook environment.

## Quality bar

- Coverage must stay **≥80%** (pytest-cov).
- Format: **Black**; lint: **Ruff**; types: **Mypy** (strict settings in `pyproject.toml`).
