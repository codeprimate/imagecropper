"""Tests for crop logic."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

from imagecropper.crop import (
    _HUMAN_YOLO_WEIGHTS,
    CropResult,
    ImageCropper,
    _expand_bbox_to_aspect_crop,
    _pad_detection_bbox,
)


def _rgb_png(path: Path, w: int, h: int) -> None:
    Image.new("RGB", (w, h), color=(10, 20, 30)).save(path)


def test_center_crop_geometry() -> None:
    cropper = ImageCropper(model_dir=Path("/tmp/unused"))
    img = np.zeros((40, 20, 3), dtype=np.uint8)
    out = cropper.center_crop(img, 10, 10)
    assert out.shape[0] > 0 and out.shape[1] > 0


def test_crop_one_center_writes_jpg(tmp_path: Path) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 30, 40)
    cropper = ImageCropper(model_dir=tmp_path)
    r = cropper.crop_one(inp, 5, 5, "center", None, True, enhance=False)
    assert r.error is None
    assert r.output_path == tmp_path / "x-cropped.jpg"
    assert r.strategy_used == "center"
    with Image.open(r.output_path) as out:
        assert out.size == (5, 5)


def test_crop_one_human_strategy_no_detection(tmp_path: Path, mocker: Any) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 8, 8)
    cropper = ImageCropper(model_dir=tmp_path)
    mocker.patch.object(ImageCropper, "detect_human", return_value=None)
    r = cropper.crop_one(inp, 4, 4, "human", None, True, enhance=False)
    assert r.error is not None
    assert "human" in r.error.lower()


def test_crop_one_human_strategy_with_detection(tmp_path: Path, mocker: Any) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 20, 20)
    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    cropper = ImageCropper(model_dir=tmp_path)
    mocker.patch.object(ImageCropper, "detect_human", return_value=patch)
    r = cropper.crop_one(inp, 4, 4, "human", None, True, enhance=False)
    assert r.error is None
    assert r.strategy_used == "human"


def test_crop_one_face_strategy_no_detection(tmp_path: Path, mocker: Any) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 8, 8)
    cropper = ImageCropper(model_dir=tmp_path)
    mocker.patch.object(ImageCropper, "detect_face", return_value=None)
    r = cropper.crop_one(inp, 4, 4, "face", None, True, enhance=False)
    assert r.error is not None


def test_crop_one_face_strategy_with_detection(tmp_path: Path, mocker: Any) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 20, 20)
    patch = np.zeros((8, 8, 3), dtype=np.uint8)
    cropper = ImageCropper(model_dir=tmp_path)
    mocker.patch.object(ImageCropper, "detect_face", return_value=patch)
    r = cropper.crop_one(inp, 3, 3, "face", None, True, enhance=False)
    assert r.error is None
    assert r.strategy_used == "face"


def test_crop_auto_prefers_human(tmp_path: Path, mocker: Any) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 20, 20)
    patch = np.ones((6, 6, 3), dtype=np.uint8) * 200
    cropper = ImageCropper(model_dir=tmp_path)
    mocker.patch.object(ImageCropper, "detect_human", return_value=patch)
    mocker.patch.object(ImageCropper, "detect_face", return_value=None)
    r = cropper.crop_one(inp, 4, 4, "auto", None, True, enhance=False)
    assert r.error is None
    assert r.strategy_used == "human"


def test_crop_auto_face_fallback(tmp_path: Path, mocker: Any) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 20, 20)
    face_patch = np.ones((5, 5, 3), dtype=np.uint8) * 100
    cropper = ImageCropper(model_dir=tmp_path)
    mocker.patch.object(ImageCropper, "detect_human", return_value=None)
    mocker.patch.object(ImageCropper, "detect_face", return_value=face_patch)
    r = cropper.crop_one(inp, 4, 4, "auto", None, True, enhance=False)
    assert r.error is None
    assert r.strategy_used == "face"


def test_crop_auto_center_fallback(tmp_path: Path, mocker: Any) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 12, 16)
    cropper = ImageCropper(model_dir=tmp_path)
    mocker.patch.object(ImageCropper, "detect_human", return_value=None)
    mocker.patch.object(ImageCropper, "detect_face", return_value=None)
    r = cropper.crop_one(inp, 4, 4, "auto", None, True, enhance=False)
    assert r.error is None
    assert r.strategy_used == "center"


def test_crop_refuse_overwrite(tmp_path: Path) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 10, 10)
    out = tmp_path / "x-cropped.jpg"
    out.write_bytes(b"existing")
    cropper = ImageCropper(model_dir=tmp_path)
    r = cropper.crop_one(inp, 4, 4, "center", None, False, enhance=False)
    assert r.error is not None
    assert "overwrite" in r.error.lower()


def test_crop_explicit_output_path(tmp_path: Path) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 10, 10)
    custom = tmp_path / "out" / "custom.png"
    cropper = ImageCropper(model_dir=tmp_path)
    r = cropper.crop_one(inp, 3, 3, "center", custom, True, enhance=False)
    assert r.error is None
    assert r.output_path == custom
    assert custom.exists()


def test_select_region_unknown_strategy() -> None:
    cropper = ImageCropper(model_dir=Path("/tmp"))
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="unknown strategy"):
        cropper._select_region(img, 4, 4, cast(Any, "invalid"))


def test_detect_human_empty_boxes_no_boxes_object(tmp_path: Path) -> None:
    cropper = ImageCropper(model_dir=tmp_path)

    class _R0:
        boxes = None

    class _Model:
        def predict(self, _img: Any, verbose: bool = False) -> list[Any]:
            return [_R0()]

    cropper._human_net = _Model()
    assert cropper.detect_human(np.zeros((20, 20, 3), dtype=np.uint8), 16, 16) is None


def test_detect_human_empty_boxes_len_zero(tmp_path: Path) -> None:
    cropper = ImageCropper(model_dir=tmp_path)

    class _Boxes:
        xyxy = np.zeros((0, 4), dtype=np.float32)

        def __len__(self) -> int:
            return 0

    class _R0:
        boxes = _Boxes()

    class _Model:
        def predict(self, _img: Any, verbose: bool = False) -> list[Any]:
            return [_R0()]

    cropper._human_net = _Model()
    assert cropper.detect_human(np.zeros((20, 20, 3), dtype=np.uint8), 16, 16) is None


def test_detect_human_valid_crop(tmp_path: Path) -> None:
    cropper = ImageCropper(model_dir=tmp_path)
    img = np.zeros((30, 30, 3), dtype=np.uint8)

    class _Boxes:
        xyxy = np.array([[5.0, 5.0, 15.0, 20.0]], dtype=np.float32)
        cls = np.array([0.0], dtype=np.float32)
        conf = np.array([0.95], dtype=np.float32)

        def __len__(self) -> int:
            return 1

    class _R0:
        boxes = _Boxes()

    class _Model:
        def predict(self, _x: Any, verbose: bool = False) -> list[Any]:
            return [_R0()]

    cropper._human_net = _Model()
    out = cropper.detect_human(img, 30, 30)
    assert out is not None
    # Square output aspect → minimal window height max(15,10)=15, width 15.
    assert out.shape[0] == out.shape[1] == 15


def test_detect_human_invalid_box_dims(tmp_path: Path) -> None:
    cropper = ImageCropper(model_dir=tmp_path)

    class _Boxes:
        xyxy = np.array([[5.0, 5.0, 4.0, 6.0]], dtype=np.float32)
        cls = np.array([0.0], dtype=np.float32)
        conf = np.array([0.9], dtype=np.float32)

        def __len__(self) -> int:
            return 1

    class _R0:
        boxes = _Boxes()

    class _Model:
        def predict(self, _img: Any, verbose: bool = False) -> list[Any]:
            return [_R0()]

    cropper._human_net = _Model()
    assert cropper.detect_human(np.zeros((20, 20, 3), dtype=np.uint8), 8, 8) is None


def test_detect_human_only_non_person_returns_none(tmp_path: Path) -> None:
    """COCO class 27 = tie; must not crop to tie when no person box exists."""
    cropper = ImageCropper(model_dir=tmp_path)

    class _Boxes:
        xyxy = np.array([[60.0, 70.0, 65.0, 75.0]], dtype=np.float32)
        cls = np.array([27.0], dtype=np.float32)
        conf = np.array([0.99], dtype=np.float32)

        def __len__(self) -> int:
            return 1

    class _R0:
        boxes = _Boxes()

    class _Model:
        def predict(self, _img: Any, verbose: bool = False) -> list[Any]:
            return [_R0()]

    cropper._human_net = _Model()
    assert cropper.detect_human(np.zeros((100, 100, 3), dtype=np.uint8), 64, 64) is None


def test_detect_human_prefers_highest_confidence_person(tmp_path: Path) -> None:
    """Among COCO persons, pick max confidence (not largest area)."""
    cropper = ImageCropper(model_dir=tmp_path)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    class _Boxes:
        xyxy = np.array(
            [
                [10.0, 10.0, 30.0, 35.0],
                [5.0, 5.0, 95.0, 95.0],
            ],
            dtype=np.float32,
        )
        cls = np.array([0.0, 0.0], dtype=np.float32)
        conf = np.array([0.99, 0.2], dtype=np.float32)

        def __len__(self) -> int:
            return 2

    class _R0:
        boxes = _Boxes()

    class _Model:
        def predict(self, _img: Any, verbose: bool = False) -> list[Any]:
            return [_R0()]

    cropper._human_net = _Model()
    out = cropper.detect_human(img, 100, 100)
    assert out is not None
    # High-conf box 20×25; square aspect crop → 25×25.
    assert out.shape == (25, 25, 3)


def test_expand_bbox_detector_region_fully_inside_crop_when_geometry_allows() -> None:
    """DATA-004 (containment): detection-marked region is fully present in the aspect crop."""
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[10:30, 10:40, 0] = 255
    out = _expand_bbox_to_aspect_crop(img, 10, 10, 40, 30, 16, 16)
    assert np.count_nonzero(out[:, :, 0] == 255) == 20 * 30


def test_expand_bbox_edge_first_top_anchor() -> None:
    """DATA-004: smallest slack to top → ``y0 = y_lo`` (orthogonal axis centered)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[2:22, 40:60, 0] = 17
    out = _expand_bbox_to_aspect_crop(img, 40, 2, 60, 22, 32, 32)
    assert np.array_equal(out, img[2:22, 40:60, :])


def test_expand_bbox_edge_first_bottom_anchor() -> None:
    """DATA-004: smallest slack to bottom → ``y0 = y_hi``."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[85:99, 40:60, 0] = 9
    out = _expand_bbox_to_aspect_crop(img, 40, 85, 60, 99, 32, 32)
    assert np.array_equal(out, img[80:100, 40:60, :])


def test_expand_bbox_edge_first_left_anchor() -> None:
    """DATA-004: smallest slack to left → ``x0 = x_lo``."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 2:22, 0] = 11
    out = _expand_bbox_to_aspect_crop(img, 2, 40, 22, 60, 32, 32)
    assert np.array_equal(out, img[40:60, 2:22, :])


def test_expand_bbox_edge_first_right_anchor() -> None:
    """DATA-004: smallest slack to right → ``x0 = x_hi``."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 78:98, 0] = 13
    out = _expand_bbox_to_aspect_crop(img, 78, 40, 98, 60, 32, 32)
    assert np.array_equal(out, img[40:60, 78:98, :])


def test_pad_detection_bbox_symmetric_clips() -> None:
    x1, y1, x2, y2 = _pad_detection_bbox(
        10, 10, 20, 20, 64, 64, pad_x=0.2, pad_y_top=0.2, pad_y_bottom=0.2
    )
    assert x1 == 8 and y1 == 8 and x2 == 22 and y2 == 22


def test_pad_detection_bbox_asymmetric_top() -> None:
    x1, y1, x2, y2 = _pad_detection_bbox(
        40, 40, 50, 50, 100, 100, pad_x=0.1, pad_y_top=0.5, pad_y_bottom=0.1
    )
    top_out = 40 - y1
    bottom_out = y2 - 50
    assert top_out > bottom_out


def test_expand_bbox_aspect_matches_target_ratio() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = _expand_bbox_to_aspect_crop(img, 40, 45, 60, 55, 200, 50)
    ar = out.shape[1] / out.shape[0]
    assert abs(ar - 4.0) < 0.06


def test_expand_bbox_contains_bbox_pixels() -> None:
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    out = _expand_bbox_to_aspect_crop(img, 10, 10, 20, 30, 10, 10)
    assert out.shape[0] >= 20
    assert out.shape[1] >= 20


def test_expand_bbox_nudges_when_centering_misses_bbox() -> None:
    """Wide crop + bbox on the right: crop must still include the bbox (DATA-004 containment)."""
    img = np.zeros((40, 200, 3), dtype=np.uint8)
    img[:, :, 0] = np.arange(200, dtype=np.uint8)
    # Target aspect 5:1 matches image 200×40 so scale=1; wide crop + bbox on the right edge.
    out = _expand_bbox_to_aspect_crop(img, 150, 5, 199, 35, 400, 80)
    assert out.shape[1] >= 199 - 150
    assert out.shape[0] >= 35 - 5
    # Value 198 only appears at source column 198; crop must still include that column.
    assert (out[:, :, 0] == 198).any()


def test_expand_bbox_rebalances_when_rounded_height_exceeds_image() -> None:
    """Extreme target aspect can make ``h_i`` from width exceed ``h_img``; code caps and recomputes."""
    img = np.zeros((4, 10, 3), dtype=np.uint8)
    out = _expand_bbox_to_aspect_crop(img, 0, 0, 10, 4, 1, 100)
    assert out.shape[0] == 4
    assert out.shape[1] <= 10


def test_expand_bbox_rebalances_when_rounded_width_exceeds_image() -> None:
    img = np.zeros((10, 4, 3), dtype=np.uint8)
    out = _expand_bbox_to_aspect_crop(img, 0, 0, 4, 10, 100, 1)
    assert out.shape[1] == 4
    assert out.shape[0] <= 10


def test_detect_face_low_confidence(tmp_path: Path, mocker: Any) -> None:
    cropper = ImageCropper(model_dir=tmp_path)
    net = mocker.MagicMock()
    forward = np.zeros((1, 1, 1, 7), dtype=np.float32)
    forward[0, 0, 0, 2] = 0.1
    net.forward.return_value = forward
    cropper._face_net = net
    assert cropper.detect_face(np.zeros((30, 30, 3), dtype=np.uint8), 16, 16) is None


def test_detect_face_high_confidence(tmp_path: Path, mocker: Any) -> None:
    cropper = ImageCropper(model_dir=tmp_path)
    net = mocker.MagicMock()
    d = np.zeros((1, 1, 2, 7), dtype=np.float32)
    d[0, 0, 0, 2] = 0.9
    d[0, 0, 0, 3:7] = [0.25, 0.25, 0.75, 0.75]
    d[0, 0, 1, 2] = 0.2
    net.forward.return_value = d
    cropper._face_net = net
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = cropper.detect_face(img, 64, 64)
    assert out is not None
    assert out.shape[0] == out.shape[1]
    assert out.size > 0


def test_detect_face_invalid_dims_after_scale(tmp_path: Path, mocker: Any) -> None:
    cropper = ImageCropper(model_dir=tmp_path)
    net = mocker.MagicMock()
    d = np.zeros((1, 1, 1, 7), dtype=np.float32)
    d[0, 0, 0, 2] = 0.99
    d[0, 0, 0, 3:7] = [0.5, 0.5, 0.5, 0.5]
    net.forward.return_value = d
    cropper._face_net = net
    assert cropper.detect_face(np.zeros((10, 10, 3), dtype=np.uint8), 4, 4) is None


def test_ensure_face_net_loads(mocker: Any, tmp_path: Path) -> None:
    cropper = ImageCropper(model_dir=tmp_path)
    mocker.patch("imagecropper.crop.download_face_models")
    fake_net = object()
    mocker.patch("imagecropper.crop.cv2.dnn.readNetFromCaffe", return_value=fake_net)
    n1 = cropper._ensure_face_net()
    n2 = cropper._ensure_face_net()
    assert n1 is fake_net is n2


def test_crop_result_dataclass() -> None:
    r = CropResult(
        input_path=Path("a"),
        output_path=Path("b"),
        strategy_used="center",
        elapsed_ms=1,
        target_width=2,
        target_height=3,
        error=None,
    )
    assert r.strategy_used == "center"


def test_ensure_human_net_lazy_load(mocker: Any, tmp_path: Path) -> None:
    fake = mocker.MagicMock()
    mock_yolo = mocker.patch("ultralytics.YOLO", return_value=fake)
    cropper = ImageCropper(model_dir=tmp_path)
    assert cropper._ensure_human_net() is fake
    assert cropper._ensure_human_net() is fake
    mock_yolo.assert_called_once_with(_HUMAN_YOLO_WEIGHTS)


def test_crop_one_anon_applied_calls_inpaint_helper(tmp_path: Path, mocker: Any) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 40, 40)
    cropper = ImageCropper(model_dir=tmp_path)
    mocker.patch.object(ImageCropper, "detect_face_bbox", return_value=(10, 10, 20, 20))
    anon_mock = mocker.patch(
        "imagecropper.crop.anonymize_face_inpaint",
        side_effect=lambda im, bb: im,
    )
    r = cropper.crop_one(inp, 5, 5, "center", None, True, anon=True)
    assert r.error is None
    anon_mock.assert_called_once()
    assert r.strategy_used == "center (anon) (enhance skipped: anon)"


def test_crop_one_enhance_skipped_anon(tmp_path: Path, mocker: Any) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 20, 20)
    cropper = ImageCropper(model_dir=tmp_path)
    mocker.patch.object(ImageCropper, "detect_face_bbox", return_value=(2, 2, 10, 10))
    enh_mock = mocker.patch("imagecropper.crop.enhance_bgr_gfpgan")
    r = cropper.crop_one(inp, 4, 4, "center", None, True, anon=True)
    assert r.error is None
    enh_mock.assert_not_called()
    assert "(enhance skipped: anon)" in r.strategy_used


def test_crop_one_enhance_skipped_no_face_on_resized(tmp_path: Path, mocker: Any) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 20, 20)
    cropper = ImageCropper(model_dir=tmp_path)

    def _bbox_side_effect(im: Any) -> Any:
        h, w = im.shape[:2]
        if w >= 20:
            return (5, 5, 15, 15)
        return None

    mocker.patch.object(ImageCropper, "detect_face_bbox", side_effect=_bbox_side_effect)
    enh_mock = mocker.patch("imagecropper.crop.enhance_bgr_gfpgan")
    r = cropper.crop_one(inp, 4, 4, "center", None, True)
    assert r.error is None
    enh_mock.assert_not_called()
    assert "(enhance skipped)" in r.strategy_used


def test_crop_one_enhance_applies_when_face_on_resized(tmp_path: Path, mocker: Any) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 20, 20)
    cropper = ImageCropper(model_dir=tmp_path)
    mocker.patch.object(ImageCropper, "detect_face_bbox", return_value=(1, 1, 3, 3))

    def _fake_enh(im: Any, _md: Any) -> Any:
        return im + 1

    mocker.patch("imagecropper.crop.enhance_bgr_gfpgan", side_effect=_fake_enh)
    r = cropper.crop_one(inp, 4, 4, "center", None, True)
    assert r.error is None
    assert "(enhance)" in r.strategy_used


def test_crop_one_enhance_failed_falls_back(tmp_path: Path, mocker: Any) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 20, 20)
    cropper = ImageCropper(model_dir=tmp_path)
    mocker.patch.object(ImageCropper, "detect_face_bbox", return_value=(1, 1, 3, 3))
    mocker.patch("imagecropper.crop.enhance_bgr_gfpgan", side_effect=RuntimeError("boom"))
    r = cropper.crop_one(inp, 4, 4, "center", None, True)
    assert r.error is None
    assert "(enhance failed)" in r.strategy_used


def test_crop_one_anon_skipped_no_face(tmp_path: Path, mocker: Any) -> None:
    inp = tmp_path / "x.png"
    _rgb_png(inp, 20, 20)
    cropper = ImageCropper(model_dir=tmp_path)
    mocker.patch.object(ImageCropper, "detect_face_bbox", return_value=None)
    anon_mock = mocker.patch("imagecropper.crop.anonymize_face_inpaint")
    r = cropper.crop_one(inp, 4, 4, "center", None, True, anon=True)
    assert r.error is None
    anon_mock.assert_not_called()
    assert r.strategy_used == "center (anon skipped) (enhance skipped: anon)"


def test_detect_face_bbox_low_confidence(tmp_path: Path, mocker: Any) -> None:
    cropper = ImageCropper(model_dir=tmp_path)
    net = mocker.MagicMock()
    forward = np.zeros((1, 1, 1, 7), dtype=np.float32)
    forward[0, 0, 0, 2] = 0.1
    net.forward.return_value = forward
    cropper._face_net = net
    assert cropper.detect_face_bbox(np.zeros((30, 30, 3), dtype=np.uint8)) is None


def test_detect_face_bbox_high_confidence(tmp_path: Path, mocker: Any) -> None:
    cropper = ImageCropper(model_dir=tmp_path)
    net = mocker.MagicMock()
    d = np.zeros((1, 1, 2, 7), dtype=np.float32)
    d[0, 0, 0, 2] = 0.9
    d[0, 0, 0, 3:7] = [0.25, 0.25, 0.75, 0.75]
    d[0, 0, 1, 2] = 0.2
    net.forward.return_value = d
    cropper._face_net = net
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = cropper.detect_face_bbox(img)
    assert bbox is not None
    sx, sy, ex, ey = bbox
    assert ex > sx and ey > sy


def test_crop_one_image_open_oserror(tmp_path: Path, mocker: Any) -> None:
    inp = tmp_path / "x.png"
    inp.write_bytes(b"x")
    cropper = ImageCropper(model_dir=tmp_path)
    mocker.patch("imagecropper.crop.Image.open", side_effect=OSError("read fail"))
    r = cropper.crop_one(inp, 2, 2, "center", None, True, enhance=False)
    assert r.error is not None
    assert "read fail" in r.error
