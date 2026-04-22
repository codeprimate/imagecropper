"""Tests for face model download helpers."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from imagecropper.models import (
    default_model_dir,
    download_face_models,
    ensure_model_directory,
    face_caffemodel_path,
    face_prototxt_path,
)


def test_default_model_dir_is_under_home() -> None:
    d = default_model_dir()
    assert d.name == ".imagecropper"


def test_ensure_model_directory_creates(tmp_path: Path) -> None:
    d = tmp_path / "nested" / "models"
    ensure_model_directory(d)
    assert d.is_dir()


def test_download_face_models_writes_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_get = MagicMock()
    mock_get.return_value.content = b"fake-bytes"
    mock_get.return_value.raise_for_status = MagicMock()
    monkeypatch.setattr("imagecropper.models.requests.get", mock_get)

    download_face_models(tmp_path)

    assert face_prototxt_path(tmp_path).read_bytes() == b"fake-bytes"
    assert face_caffemodel_path(tmp_path).read_bytes() == b"fake-bytes"
    assert mock_get.call_count == 2


def test_download_face_models_skips_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    face_prototxt_path(tmp_path).write_text("x", encoding="utf-8")
    face_caffemodel_path(tmp_path).write_text("y", encoding="utf-8")
    mock_get = MagicMock()
    monkeypatch.setattr("imagecropper.models.requests.get", mock_get)
    download_face_models(tmp_path)
    mock_get.assert_not_called()
