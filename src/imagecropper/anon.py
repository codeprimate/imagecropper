"""Face anonymization on a BGR raster using OpenCV inpaint (geometry mask from face box)."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

# Fractions of SSD face box width/height applied outward from each edge.
_ANON_PAD_X = 0.22
_ANON_PAD_TOP = 0.55
_ANON_PAD_BOTTOM = 0.12

_INPAINT_RADIUS = 5
# Grow inscribed semi-axes before inpaint (pixels outward along each axis).
_ANON_OVAL_EXPAND_PX = 10
# Gaussian blur kernel half-size (kernel = 2 * n + 1); full-frame blur after inpaint.
_ANON_OVAL_BLUR_PX = 10
# Edge feather: Gaussian kernel half-size = this fraction × oval **minor diameter** (2×min semi-axis).
_ANON_FEATHER_FRACTION_OF_MINOR_DIAMETER = 0.10


def _oval_semi_axes(
    ex1: int,
    ey1: int,
    ex2: int,
    ey2: int,
    *,
    semi_axis_expand_px: int,
) -> tuple[int, int]:
    """Semi-axes (px) of the anonymize ellipse inscribed in the expanded head bbox."""
    bw = ex2 - ex1
    bh = ey2 - ey1
    axis_w = max(1, bw // 2) + max(0, semi_axis_expand_px)
    axis_h = max(1, bh // 2) + max(0, semi_axis_expand_px)
    return axis_w, axis_h


def _feather_kernel_half_px(
    ex1: int,
    ey1: int,
    ex2: int,
    ey2: int,
    *,
    semi_axis_expand_px: int,
) -> int:
    """Odd Gaussian uses ``2 * n + 1``; ``n`` is 10% of minor diameter, at least 1."""
    aw, ah = _oval_semi_axes(ex1, ey1, ex2, ey2, semi_axis_expand_px=semi_axis_expand_px)
    minor_diameter = 2 * min(aw, ah)
    return max(1, int(round(_ANON_FEATHER_FRACTION_OF_MINOR_DIAMETER * minor_diameter)))


def _fill_expanded_bbox_oval(
    mask: npt.NDArray[np.uint8],
    ex1: int,
    ey1: int,
    ex2: int,
    ey2: int,
    *,
    semi_axis_expand_px: int = 0,
) -> None:
    """Paint an axis-aligned ellipse on ``mask`` (inscribed in bbox, semi-axes + expand)."""
    axis_w, axis_h = _oval_semi_axes(
        ex1,
        ey1,
        ex2,
        ey2,
        semi_axis_expand_px=semi_axis_expand_px,
    )
    cx = (ex1 + ex2) // 2
    cy = (ey1 + ey2) // 2
    cv2.ellipse(
        mask,
        (cx, cy),
        (axis_w, axis_h),
        0.0,
        0.0,
        360.0,
        255,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


def expand_face_bbox_for_head(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Expand a face rectangle upward (hair) and slightly sideways/down; clip to image."""
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    nx1 = x1 - int(round(bw * _ANON_PAD_X))
    nx2 = x2 + int(round(bw * _ANON_PAD_X))
    ny1 = y1 - int(round(bh * _ANON_PAD_TOP))
    ny2 = y2 + int(round(bh * _ANON_PAD_BOTTOM))
    nx1 = int(np.clip(nx1, 0, max(0, image_width - 1)))
    ny1 = int(np.clip(ny1, 0, max(0, image_height - 1)))
    nx2 = int(np.clip(nx2, nx1 + 1, image_width))
    ny2 = int(np.clip(ny2, ny1 + 1, image_height))
    return nx1, ny1, nx2, ny2


def _mask_bbox_roi(
    mask: npt.NDArray[np.uint8],
    image_height: int,
    image_width: int,
    pad_px: int,
) -> tuple[int, int, int, int] | None:
    """Return ``y0, y1, x0, x1`` covering mask nonzero pixels plus ``pad_px``, or ``None``."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0 = max(0, int(ys.min()) - pad_px)
    y1 = min(image_height, int(ys.max()) + pad_px + 1)
    x0 = max(0, int(xs.min()) - pad_px)
    x1 = min(image_width, int(xs.max()) + pad_px + 1)
    return y0, y1, x0, x1


def anonymize_face_inpaint(
    image_bgr: npt.NDArray[np.uint8],
    face_bbox: tuple[int, int, int, int],
) -> npt.NDArray[np.uint8]:
    """Anonymize the face oval via inpaint, content blur, then premultiplied-alpha layer composite."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = face_bbox
    ex1, ey1, ex2, ey2 = expand_face_bbox_for_head(x1, y1, x2, y2, w, h)
    mask_filled = np.zeros((h, w), dtype=np.uint8)
    _fill_expanded_bbox_oval(
        mask_filled,
        ex1,
        ey1,
        ex2,
        ey2,
        semi_axis_expand_px=_ANON_OVAL_EXPAND_PX,
    )
    src = image_bgr.copy()
    inpainted = cast(
        npt.NDArray[np.uint8],
        cv2.inpaint(src, mask_filled, _INPAINT_RADIUS, cv2.INPAINT_TELEA),
    )
    bk = 2 * _ANON_OVAL_BLUR_PX + 1
    blurred = cast(
        npt.NDArray[np.uint8],
        cv2.GaussianBlur(inpainted, (bk, bk), 0),
    )

    # Full-raster layer: a tight ROI clips Gaussian tails and leaves a hard *rectangle*
    # around the head; premult must be blurred and composited on the full frame.
    feather_half = _feather_kernel_half_px(
        ex1,
        ey1,
        ex2,
        ey2,
        semi_axis_expand_px=_ANON_OVAL_EXPAND_PX,
    )
    k_feather = 2 * feather_half + 1

    mask_f = mask_filled.astype(np.float32) / 255.0
    alpha_soft = cast(
        npt.NDArray[np.float32],
        cv2.GaussianBlur(mask_f, (k_feather, k_feather), 0),
    )
    amax = float(alpha_soft.max())
    if amax <= 1e-6:
        return cast(npt.NDArray[np.uint8], image_bgr.copy())
    alpha_soft /= amax
    alpha_h = alpha_soft[..., np.newaxis]

    blurred_f = blurred.astype(np.float32)
    premult = blurred_f * alpha_h
    layer = np.concatenate([premult, alpha_h], axis=2)
    layer_b = cast(
        npt.NDArray[np.float32],
        cv2.GaussianBlur(layer, (k_feather, k_feather), 0),
    )
    prem_b = layer_b[..., :3]
    alpha_b = np.clip(layer_b[..., 3:4], 0.0, 1.0)

    out = src.astype(np.float32) * (1.0 - alpha_b) + prem_b
    return cast(npt.NDArray[np.uint8], np.clip(np.round(out), 0, 255).astype(np.uint8))
