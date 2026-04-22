"""Tests for anonymization helpers."""

import numpy as np

from imagecropper.anon import (
    _feather_kernel_half_px,
    _mask_bbox_roi,
    _oval_semi_axes,
    anonymize_face_inpaint,
    expand_face_bbox_for_head,
)


def test_expand_face_bbox_clipped_to_image() -> None:
    x1, y1, x2, y2 = expand_face_bbox_for_head(50, 50, 60, 60, 64, 64)
    assert 0 <= x1 < x2 <= 64
    assert 0 <= y1 < y2 <= 64


def test_expand_face_bbox_non_empty() -> None:
    x1, y1, x2, y2 = expand_face_bbox_for_head(10, 10, 20, 20, 100, 100)
    assert x2 > x1 and y2 > y1
    assert x1 < 10 or y1 < 10 or x2 > 20 or y2 > 20


def test_feather_half_is_ten_percent_of_minor_diameter() -> None:
    # Bbox 0,0,100,100 → semi-axes 50+10=60 each → minor diameter 120 → 10% = 12
    assert _feather_kernel_half_px(0, 0, 100, 100, semi_axis_expand_px=10) == 12
    # Narrow tall oval: bw=40,bh=100 → axis_w=20+0=20, axis_h=50 → minor diameter 40 → 10% = 4
    assert _feather_kernel_half_px(0, 0, 40, 100, semi_axis_expand_px=0) == 4


def test_oval_semi_axes_match_expand() -> None:
    aw, ah = _oval_semi_axes(0, 0, 100, 80, semi_axis_expand_px=5)
    assert aw == 50 + 5 and ah == 40 + 5


def test_mask_bbox_roi_empty_returns_none() -> None:
    empty = np.zeros((8, 8), dtype=np.uint8)
    assert _mask_bbox_roi(empty, 8, 8, 3) is None


def test_anonymize_face_inpaint_changes_masked_region() -> None:
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:, :] = (80, 40, 20)
    img[15:25, 15:25] = (220, 220, 220)
    out = anonymize_face_inpaint(img, (15, 15, 25, 25))
    assert out.shape == img.shape
    assert not np.array_equal(out[18:22, 18:22], img[18:22, 18:22])
