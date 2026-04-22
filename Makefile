.PHONY: help install install-dev build check publish uninstall clean format lint typecheck test test-unit coverage check-all

# ==============================================================================
# Build and Installing (inspired by todo_agent)
# ==============================================================================
# Use project venv: .venv/bin/pip, .venv/bin/python

help:
	@echo "Available commands:"
	@echo "  make install       - Build wheel and install package locally (from dist/*.whl)"
	@echo "  make install-dev   - Install package in development mode with [dev] deps"
	@echo "  make build         - Clean and build package (sdist + wheel)"
	@echo "  make check         - Validate distribution with twine"
	@echo "  make publish       - Clean, build, check, then upload to PyPI"
	@echo "  make uninstall     - Uninstall imagecropper from current environment"
	@echo ""
	@echo "  make format        - Format code with ruff"
	@echo "  make lint          - Lint with ruff"
	@echo "  make typecheck     - Type check with mypy"
	@echo "  make test          - Run tests"
	@echo "  make test-unit     - Run tests marked @pytest.mark.unit"
	@echo "  make coverage      - Run tests with coverage report"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make check-all     - format, lint, typecheck, test"

install: build uninstall
	@echo "📦 Installing built package locally..."
	pip install dist/*.whl

install-dev: uninstall
	@echo "🔧 Installing package in development mode with dev dependencies..."
	pip install -e ".[dev]"

build: clean
	@echo "🔨 Building package..."
	.venv/bin/python -m build

check:
	@echo "✅ Checking distribution files..."
	.venv/bin/twine check dist/*

publish: clean build check
	@echo "🚀 Publishing to PyPI..."
	.venv/bin/twine upload dist/*

uninstall:
	@echo "🧹 Uninstalling package..."
	.venv/bin/pip uninstall -y imagecropper 2>/dev/null || true

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/ dist/ .eggs/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ==============================================================================
# Linting and Code Quality
# ==============================================================================

format:
	.venv/bin/ruff format src/ tests/
	.venv/bin/ruff check --fix src/ tests/

lint:
	.venv/bin/ruff check src/ tests/

typecheck:
	.venv/bin/mypy src/

# ==============================================================================
# Testing
# ==============================================================================

test:
	.venv/bin/pytest

test-unit:
	.venv/bin/pytest -m unit

coverage:
	.venv/bin/pytest --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

check-all: format lint typecheck test
	@echo "All quality checks passed!"
