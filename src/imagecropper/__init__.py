"""Image cropping CLI package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("imagecropper")
except PackageNotFoundError:  # e.g. bare ``PYTHONPATH=src`` without an install
    __version__ = "0.0.0+unknown"
