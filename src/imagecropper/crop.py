"""Smart crop: human (YOLOv8 via Ultralytics), face (OpenCV DNN), or center crop."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from imagecropper.anon import anonymize_face_inpaint
from imagecropper.enhance import enhance_bgr_gfpgan
from imagecropper.models import (
    default_model_dir,
    download_face_models,
    face_caffemodel_path,
    face_prototxt_path,
)

StrategyName = Literal["auto", "human", "face", "center"]

# COCO 80-class id for Ultralytics COCO pretrained detectors
_COCO_PERSON_CLASS = 0
# YOLOv8m: better person localization (incl. head) than yolov5su at higher compute/weight cost.
_HUMAN_YOLO_WEIGHTS = "yolov8m.pt"

# OpenCV SSD face detector confidence threshold (matches prior ``detect_face`` behavior)
_FACE_CONFIDENCE_THRESHOLD = 0.5

# Extra margin around detector boxes before aspect expansion (fractions of box width/height).
_FACE_BBOX_PAD_X = 0.15
_FACE_BBOX_PAD_TOP = 0.2
_FACE_BBOX_PAD_BOTTOM = 0.2


@dataclass(frozen=True)
class CropResult:
    """Outcome of cropping one input file."""

    input_path: Path
    output_path: Path | None
    strategy_used: str
    elapsed_ms: int
    target_width: int
    target_height: int
    error: str | None = None
    debug_output_path: Path | None = None


def _pad_detection_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
    *,
    pad_x: float,
    pad_y_top: float,
    pad_y_bottom: float,
) -> tuple[int, int, int, int]:
    """Grow ``xyxy`` outward by per-edge fractions of box size; clip to image bounds."""
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    nx1 = x1 - int(round(bw * pad_x))
    nx2 = x2 + int(round(bw * pad_x))
    ny1 = y1 - int(round(bh * pad_y_top))
    ny2 = y2 + int(round(bh * pad_y_bottom))
    nx1 = int(np.clip(nx1, 0, max(0, image_width - 1)))
    ny1 = int(np.clip(ny1, 0, max(0, image_height - 1)))
    nx2 = int(np.clip(nx2, nx1 + 1, image_width))
    ny2 = int(np.clip(ny2, ny1 + 1, image_height))
    return nx1, ny1, nx2, ny2


def _pil_to_bgr(pil: Image.Image) -> npt.NDArray[np.uint8]:
    rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    return cast(npt.NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _bgr_to_pil(bgr: npt.NDArray[np.uint8]) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _center_crop_window_xyxy(
    image_bgr: npt.NDArray[np.uint8],
    target_width: int,
    target_height: int,
) -> tuple[int, int, int, int]:
    """Axis-aligned window ``(x1, y1, x2, y2)`` with ``x2``, ``y2`` exclusive, same as ``center_crop``."""
    h, w = image_bgr.shape[:2]
    new_width = min(w, int(target_height * w / h))
    new_height = min(h, int(target_width * h / w))
    start_x = w // 2 - new_width // 2
    start_y = h // 2 - new_height // 2
    return start_x, start_y, start_x + new_width, start_y + new_height


def _put_text_stroked_baseline(
    vis: npt.NDArray[np.uint8],
    text: str,
    org_x: int,
    org_y_baseline: int,
    font: int,
    font_scale: float,
    fg_bgr: tuple[int, int, int],
    stroke_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    """``putText`` with ``org`` = bottom-left; ``org_y`` = baseline (OpenCV convention)."""
    cv2.putText(
        vis,
        text,
        (org_x, org_y_baseline),
        font,
        font_scale,
        stroke_bgr,
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        text,
        (org_x, org_y_baseline),
        font,
        font_scale,
        fg_bgr,
        thickness,
        cv2.LINE_AA,
    )


def write_crop_debug_jpeg(
    image_bgr: npt.NDArray[np.uint8],
    labeled_boxes: list[tuple[str, tuple[int, int, int, int]]],
    out_path: Path,
) -> None:
    """Draw ``labeled_boxes`` on a copy of ``image_bgr`` and save as JPEG (BGR in, RGB file)."""
    vis = image_bgr.copy()
    h_img, w_img = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(max(0.35, min(h_img, w_img) / 900.0))
    thick = max(1, int(round(scale * 2)))
    pad = max(2, int(round(4 * scale)))
    colors: dict[str, tuple[int, int, int]] = {
        "person": (0, 220, 0),
        "face_ssd": (0, 128, 255),
        "face_padded": (255, 96, 0),
        "center_crop": (0, 220, 220),
    }
    for label, (x1, y1, x2, y2) in labeled_boxes:
        color = colors.get(label, (200, 200, 255))
        x1c = int(np.clip(x1, 0, max(0, w_img - 1)))
        y1c = int(np.clip(y1, 0, max(0, h_img - 1)))
        x2c = int(np.clip(x2, x1c + 1, w_img))
        y2c = int(np.clip(y2, y1c + 1, h_img))
        br_x = x2c - 1
        br_y = y2c - 1
        cv2.rectangle(vis, (x1c, y1c), (br_x, br_y), color, thickness=max(1, thick))

        tx_scale = max(0.28, scale * 0.72)
        tthick = max(1, thick - 1)
        corner_txts = (
            ("tl", f"({x1c},{y1c})"),
            ("tr", f"({br_x},{y1c})"),
            ("bl", f"({x1c},{br_y})"),
            ("br", f"({br_x},{br_y})"),
        )
        max_top_to_bl = 1
        for _cid, t in corner_txts:
            (_tw, th), bl = cv2.getTextSize(t, font, tx_scale, tthick)
            max_top_to_bl = max(max_top_to_bl, th - bl)

        # First horizontal band inside the box: optional role label, then TL/TR coords.
        inner_top = y1c + pad
        (lw, lh), lbl = cv2.getTextSize(label, font, scale, thick)
        by_role = inner_top + (lh - lbl)
        if by_role + lbl <= y2c - pad and x1c + pad + lw <= x2c - pad:
            _put_text_stroked_baseline(
                vis,
                label,
                x1c + pad,
                by_role,
                font,
                scale,
                color,
                (0, 0, 0),
                thick,
            )
            inner_top += lh + max(1, pad // 2)

        by_top_row = inner_top + max_top_to_bl
        for corner_id, txt in corner_txts:
            (tw, th), bl = cv2.getTextSize(txt, font, tx_scale, tthick)
            top_to_baseline_t = th - bl
            org_x: int
            by: int
            if corner_id == "tl":
                org_x = x1c + pad
                by = by_top_row
            elif corner_id == "tr":
                org_x = x2c - pad - tw
                by = by_top_row
            elif corner_id == "bl":
                org_x = x1c + pad
                by = y2c - pad - bl
            else:
                org_x = x2c - pad - tw
                by = y2c - pad - bl
            top_y = by - top_to_baseline_t
            if (
                org_x >= x1c + pad
                and org_x + tw <= x2c - pad
                and top_y >= y1c + pad
                and by + bl <= y2c - pad
            ):
                _put_text_stroked_baseline(
                    vis,
                    txt,
                    org_x,
                    by,
                    font,
                    tx_scale,
                    (255, 255, 255),
                    (0, 0, 0),
                    tthick,
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_pil = _bgr_to_pil(vis)
    out_pil.save(out_path, format="JPEG", quality=92)


def _expand_bbox_to_aspect_crop(
    image_bgr: npt.NDArray[np.uint8],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    target_width: int,
    target_height: int,
) -> npt.NDArray[np.uint8]:
    """Crop a region whose aspect matches ``target_width:target_height`` (**DATA-004**)."""
    h_img, w_img = image_bgr.shape[:2]
    tw, th = int(target_width), int(target_height)
    x1c = int(np.clip(x1, 0, max(0, w_img - 1)))
    y1c = int(np.clip(y1, 0, max(0, h_img - 1)))
    x2c = int(np.clip(x2, x1c + 1, w_img))
    y2c = int(np.clip(y2, y1c + 1, h_img))

    bw = float(x2c - x1c)
    bh = float(y2c - y1c)
    ar = tw / th

    ch0 = max(bh, bw / ar)
    cw0 = ar * ch0
    cx = (x1c + x2c) / 2.0
    cy = (y1c + y2c) / 2.0

    scale = min(1.0, float(w_img) / cw0, float(h_img) / ch0)
    cw = cw0 * scale

    def _dims_from_width(w: int) -> tuple[int, int]:
        w = max(1, min(int(w), w_img))
        h = max(1, int(round(w * th / tw)))
        if h > h_img:
            h = h_img
            w = max(1, min(int(round(h * tw / th)), w_img))
        else:
            w = min(w, w_img)
            h = max(1, min(int(round(w * th / tw)), h_img))
        return w, h

    w_i, h_i = _dims_from_width(max(1, int(round(cw))))

    need_w = max(1, ceil(bw - 1e-9))
    need_h = max(1, ceil(bh - 1e-9))
    guard = 0
    while (w_i < need_w or h_i < need_h) and guard < w_img + h_img + 5:
        guard += 1
        if w_i < w_img:
            w_i, h_i = _dims_from_width(w_i + 1)
        else:
            break

    x_lo = max(0, x2c - w_i)
    x_hi = min(w_img - w_i, x1c)
    y_lo = max(0, y2c - h_i)
    y_hi = min(h_img - h_i, y1c)

    def _center_x0() -> int:
        x0_t = int(np.floor(cx - w_i / 2.0 + 1e-9))
        return int(np.clip(x0_t, x_lo, x_hi))

    if x_lo <= x_hi and y_lo <= y_hi:
        # DATA-004: slack to each image edge from the detection box; nearest edge first (tie: T,B,L,R).
        slacks = (
            ("top", y1c),
            ("bottom", h_img - y2c),
            ("left", x1c),
            ("right", w_img - x2c),
        )
        edge_i = min(range(4), key=lambda i: (slacks[i][1], i))
        edge = slacks[edge_i][0]
        if edge == "top":
            y0 = y_lo
            x0 = _center_x0()
        elif edge == "bottom":
            # Prefer minimum y0 (same as top-primary): maximize headroom above bbox top; avoids
            # shifting the window down when feet sit near the bottom edge (DATA-004).
            y0 = y_lo
            x0 = _center_x0()
        elif edge == "left":
            x0 = x_lo
            # Same vertical preference as top/bottom primary: avoid centering downward into y_hi.
            y0 = y_lo
        else:
            x0 = x_hi
            y0 = y_lo
    else:
        x0_t = int(np.floor(cx - w_i / 2.0 + 1e-9))
        y0_t = int(np.floor(cy - h_i / 2.0 + 1e-9))
        x0 = int(np.clip(x0_t, x_lo, x_hi)) if x_lo <= x_hi else max(0, min(x0_t, w_img - w_i))
        # If no y0 contains the full bbox at this window size, anchor crop top to detection top.
        y0 = (
            int(np.clip(y0_t, y_lo, y_hi))
            if y_lo <= y_hi
            else max(0, min(y1c, h_img - h_i))
        )

    return image_bgr[y0 : y0 + h_i, x0 : x0 + w_i]


class ImageCropper:
    """YOLOv8m person + aspect crop, SSD face + aspect crop (**DATA-004**), or ``center`` crop."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir if model_dir is not None else default_model_dir()
        self._face_net: Any = None
        self._human_net: Any = None

    def _ensure_face_net(self) -> Any:
        if self._face_net is None:
            download_face_models(self.model_dir)
            self._face_net = cv2.dnn.readNetFromCaffe(
                str(face_prototxt_path(self.model_dir)),
                str(face_caffemodel_path(self.model_dir)),
            )
        return self._face_net

    def _ensure_human_net(self) -> Any:
        if self._human_net is None:
            from ultralytics import YOLO  # defer heavy import until human detection runs

            self._human_net = YOLO(_HUMAN_YOLO_WEIGHTS)
        return self._human_net

    def detect_human_bbox(self, image_bgr: npt.NDArray[np.uint8]) -> tuple[int, int, int, int] | None:
        """Highest-confidence COCO person ``xyxy``, clipped to the image (**DATA-003**)."""
        model = self._ensure_human_net()
        results = model.predict(image_bgr, verbose=False)
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return None
        xyxy_t = r0.boxes.xyxy
        cls_t = r0.boxes.cls
        xyxy = xyxy_t.cpu().numpy() if hasattr(xyxy_t, "cpu") else np.asarray(xyxy_t)
        cls_vals = cls_t.cpu().numpy() if hasattr(cls_t, "cpu") else np.asarray(cls_t)
        conf_t = r0.boxes.conf
        conf = conf_t.cpu().numpy() if hasattr(conf_t, "cpu") else np.asarray(conf_t)
        person_mask = np.round(cls_vals).astype(np.int64) == _COCO_PERSON_CLASS
        idxs = np.flatnonzero(person_mask)
        if idxs.size == 0:
            return None
        j = int(idxs[int(np.argmax(conf[idxs]))])
        x1, y1, x2, y2 = (int(round(float(c))) for c in xyxy[j])
        if x2 <= x1 or y2 <= y1:
            return None
        h_img, w_img = image_bgr.shape[:2]
        x1 = int(np.clip(x1, 0, max(0, w_img - 1)))
        y1 = int(np.clip(y1, 0, max(0, h_img - 1)))
        x2 = int(np.clip(x2, x1 + 1, w_img))
        y2 = int(np.clip(y2, y1 + 1, h_img))
        return x1, y1, x2, y2

    def detect_human(
        self,
        image_bgr: npt.NDArray[np.uint8],
        target_width: int,
        target_height: int,
    ) -> npt.NDArray[np.uint8] | None:
        """Highest-confidence COCO person ``xyxy`` (clipped); aspect crop then ``crop_one`` resizes."""
        bbox = self.detect_human_bbox(image_bgr)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        return _expand_bbox_to_aspect_crop(image_bgr, x1, y1, x2, y2, target_width, target_height)

    def detect_face_bbox(
        self, image_bgr: npt.NDArray[np.uint8]
    ) -> tuple[int, int, int, int] | None:
        """Highest-confidence SSD face box in pixel coordinates, or ``None`` if below threshold."""
        h, w = image_bgr.shape[:2]
        blob_size = (300, 300)
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image_bgr, blob_size),
            1.0,
            blob_size,
            (104.0, 177.0, 123.0),
        )
        face_net = self._ensure_face_net()
        face_net.setInput(blob)
        detections = face_net.forward()
        if detections[0, 0, :, 2].max() <= _FACE_CONFIDENCE_THRESHOLD:
            return None
        i = int(np.argmax(detections[0, 0, :, 2]))
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        start_x, start_y, end_x, end_y = box.astype("int")
        if end_x <= start_x or end_y <= start_y:
            return None
        return start_x, start_y, end_x, end_y

    def detect_face_padded_bbox(
        self, image_bgr: npt.NDArray[np.uint8]
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
        """SSD ``xyxy`` and padded ``xyxy`` (both half-open), or ``None`` if no face above threshold."""
        bbox = self.detect_face_bbox(image_bgr)
        if bbox is None:
            return None
        start_x, start_y, end_x, end_y = bbox
        h_img, w_img = image_bgr.shape[:2]
        px1, py1, px2, py2 = _pad_detection_bbox(
            start_x,
            start_y,
            end_x,
            end_y,
            w_img,
            h_img,
            pad_x=_FACE_BBOX_PAD_X,
            pad_y_top=_FACE_BBOX_PAD_TOP,
            pad_y_bottom=_FACE_BBOX_PAD_BOTTOM,
        )
        return (start_x, start_y, end_x, end_y), (px1, py1, px2, py2)

    def detect_face(
        self,
        image_bgr: npt.NDArray[np.uint8],
        target_width: int,
        target_height: int,
    ) -> npt.NDArray[np.uint8] | None:
        pair = self.detect_face_padded_bbox(image_bgr)
        if pair is None:
            return None
        _raw, (start_x, start_y, end_x, end_y) = pair
        return _expand_bbox_to_aspect_crop(
            image_bgr,
            start_x,
            start_y,
            end_x,
            end_y,
            target_width,
            target_height,
        )

    def center_crop(
        self,
        image_bgr: npt.NDArray[np.uint8],
        target_width: int,
        target_height: int,
    ) -> npt.NDArray[np.uint8]:
        x1, y1, x2, y2 = _center_crop_window_xyxy(image_bgr, target_width, target_height)
        return image_bgr[y1:y2, x1:x2]

    def debug_annotation_boxes(
        self,
        image_bgr: npt.NDArray[np.uint8],
        strategy: StrategyName,
        target_width: int,
        target_height: int,
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        """Boxes drawn on ``--debug`` JPEG (same semantics as region selection)."""
        if strategy == "center":
            return [("center_crop", _center_crop_window_xyxy(image_bgr, target_width, target_height))]
        if strategy == "human":
            bbox = self.detect_human_bbox(image_bgr)
            return [("person", bbox)] if bbox is not None else []
        if strategy == "face":
            pair = self.detect_face_padded_bbox(image_bgr)
            if pair is None:
                return []
            raw, padded = pair
            return [("face_ssd", raw), ("face_padded", padded)]
        if strategy == "auto":
            hb = self.detect_human_bbox(image_bgr)
            if hb is not None:
                return [("person", hb)]
            fp = self.detect_face_padded_bbox(image_bgr)
            if fp is not None:
                raw, padded = fp
                return [("face_ssd", raw), ("face_padded", padded)]
            return [("center_crop", _center_crop_window_xyxy(image_bgr, target_width, target_height))]
        msg = f"unknown strategy: {strategy}"
        raise ValueError(msg)

    def _select_region(
        self,
        image_bgr: npt.NDArray[np.uint8],
        target_width: int,
        target_height: int,
        strategy: StrategyName,
    ) -> tuple[npt.NDArray[np.uint8], str]:
        if strategy == "center":
            return self.center_crop(image_bgr, target_width, target_height), "center"

        if strategy == "auto":
            cropped = self.detect_human(image_bgr, target_width, target_height)
            if cropped is not None:
                return cropped, "human"
            cropped = self.detect_face(image_bgr, target_width, target_height)
            if cropped is not None:
                return cropped, "face"
            return self.center_crop(image_bgr, target_width, target_height), "center"

        if strategy == "human":
            cropped = self.detect_human(image_bgr, target_width, target_height)
            if cropped is None:
                msg = "no human detection"
                raise ValueError(msg)
            return cropped, "human"

        if strategy == "face":
            cropped = self.detect_face(image_bgr, target_width, target_height)
            if cropped is None:
                msg = "no face detection"
                raise ValueError(msg)
            return cropped, "face"

        msg = f"unknown strategy: {strategy}"
        raise ValueError(msg)

    def crop_one(
        self,
        input_path: Path,
        target_width: int,
        target_height: int,
        strategy: StrategyName,
        output_path: Path | None,
        overwrite: bool,
        anon: bool = False,
        enhance: bool = True,
        *,
        debug: bool = False,
    ) -> CropResult:
        """Load image, select region, resize, save. Returns timing and strategy used."""
        t0 = perf_counter()
        if output_path is None:
            out = input_path.parent / f"{input_path.stem}-cropped.jpg"
        else:
            out = output_path

        debug_out: Path | None = (
            (input_path.parent / f"{input_path.stem}-debug.jpg") if debug else None
        )
        overwrite_blocks: list[str] = []
        if out.exists() and not overwrite:
            overwrite_blocks.append(str(out))
        if debug_out is not None and debug_out.exists() and not overwrite:
            overwrite_blocks.append(str(debug_out))
        if overwrite_blocks:
            elapsed_ms = int((perf_counter() - t0) * 1000)
            return CropResult(
                input_path=input_path,
                output_path=None,
                strategy_used="",
                elapsed_ms=elapsed_ms,
                target_width=target_width,
                target_height=target_height,
                error="refusing to overwrite existing file(s): " + "; ".join(overwrite_blocks),
            )

        try:
            with Image.open(input_path) as pil:
                pil.load()
                image_bgr = _pil_to_bgr(pil)
            if debug_out is not None:
                boxes = self.debug_annotation_boxes(
                    image_bgr, strategy, target_width, target_height
                )
                write_crop_debug_jpeg(image_bgr, boxes, debug_out)
            region, strategy_used = self._select_region(
                image_bgr, target_width, target_height, strategy
            )
            resized_bgr = cast(
                npt.NDArray[np.uint8],
                cv2.resize(region, (target_width, target_height)),
            )
            enhance_suffix = ""
            if enhance:
                if anon:
                    enhance_suffix = " (enhance skipped: anon)"
                elif self.detect_face_bbox(resized_bgr) is None:
                    enhance_suffix = " (enhance skipped)"
                else:
                    try:
                        resized_bgr = enhance_bgr_gfpgan(resized_bgr, self.model_dir)
                        enhance_suffix = " (enhance)"
                    except Exception:
                        enhance_suffix = " (enhance failed)"
            anon_suffix = ""
            if anon:
                face_bbox = self.detect_face_bbox(resized_bgr)
                if face_bbox is None:
                    anon_suffix = " (anon skipped)"
                else:
                    resized_bgr = anonymize_face_inpaint(resized_bgr, face_bbox)
                    anon_suffix = " (anon)"
            strategy_used = f"{strategy_used}{anon_suffix}{enhance_suffix}"
            out_pil = _bgr_to_pil(resized_bgr)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.suffix.lower() in {".jpg", ".jpeg"}:
                out_pil.save(out, format="JPEG", quality=95)
            else:
                out_pil.save(out)
        except (OSError, ValueError) as exc:
            elapsed_ms = int((perf_counter() - t0) * 1000)
            dbg_written: Path | None = (
                debug_out if (debug_out is not None and debug_out.exists()) else None
            )
            return CropResult(
                input_path=input_path,
                output_path=None,
                strategy_used="",
                elapsed_ms=elapsed_ms,
                target_width=target_width,
                target_height=target_height,
                error=str(exc),
                debug_output_path=dbg_written,
            )

        elapsed_ms = int((perf_counter() - t0) * 1000)
        return CropResult(
            input_path=input_path,
            output_path=out,
            strategy_used=strategy_used,
            elapsed_ms=elapsed_ms,
            target_width=target_width,
            target_height=target_height,
            error=None,
            debug_output_path=debug_out,
        )
