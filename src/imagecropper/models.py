"""Face detector model files under the imagecropper cache directory."""

from __future__ import annotations

from pathlib import Path

import requests

FACE_PROTOTXT_NAME = "deploy.prototxt"
FACE_CAFFEMODEL_NAME = "res10_300x300_ssd_iter_140000.caffemodel"

FACE_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/"
    + FACE_PROTOTXT_NAME
)
FACE_CAFFEMODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
    + FACE_CAFFEMODEL_NAME
)


def default_model_dir() -> Path:
    """Return the default directory for cached models (``~/.imagecropper``)."""
    return Path("~/.imagecropper").expanduser()


def ensure_model_directory(model_dir: Path) -> None:
    """Create the model directory if it does not exist."""
    model_dir.mkdir(parents=True, exist_ok=True)


def face_prototxt_path(model_dir: Path) -> Path:
    return model_dir / FACE_PROTOTXT_NAME


def face_caffemodel_path(model_dir: Path) -> Path:
    return model_dir / FACE_CAFFEMODEL_NAME


def download_face_models(model_dir: Path) -> None:
    """Download OpenCV DNN face detector Caffe files if missing."""
    ensure_model_directory(model_dir)
    targets: dict[Path, str] = {
        face_prototxt_path(model_dir): FACE_PROTOTXT_URL,
        face_caffemodel_path(model_dir): FACE_CAFFEMODEL_URL,
    }
    for path, url in targets.items():
        if path.exists():
            continue
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
