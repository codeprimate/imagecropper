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

    def _center_y0() -> int:
        y0_t = int(np.floor(cy - h_i / 2.0 + 1e-9))
        return int(np.clip(y0_t, y_lo, y_hi))

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
            y0 = y_hi
            x0 = _center_x0()
        elif edge == "left":
            x0 = x_lo
            y0 = _center_y0()
        else:
            x0 = x_hi
            y0 = _center_y0()
    else:
        x0_t = int(np.floor(cx - w_i / 2.0 + 1e-9))
        y0_t = int(np.floor(cy - h_i / 2.0 + 1e-9))
        x0 = int(np.clip(x0_t, x_lo, x_hi)) if x_lo <= x_hi else max(0, min(x0_t, w_img - w_i))
        y0 = int(np.clip(y0_t, y_lo, y_hi)) if y_lo <= y_hi else max(0, min(y0_t, h_img - h_i))

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

    def detect_human(
        self,
        image_bgr: npt.NDArray[np.uint8],
        target_width: int,
        target_height: int,
    ) -> npt.NDArray[np.uint8] | None:
        """Highest-confidence COCO person ``xyxy`` (clipped); aspect crop then ``crop_one`` resizes."""
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

    def detect_face(
        self,
        image_bgr: npt.NDArray[np.uint8],
        target_width: int,
        target_height: int,
    ) -> npt.NDArray[np.uint8] | None:
        bbox = self.detect_face_bbox(image_bgr)
        if bbox is None:
            return None
        start_x, start_y, end_x, end_y = bbox
        h_img, w_img = image_bgr.shape[:2]
        start_x, start_y, end_x, end_y = _pad_detection_bbox(
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
        h, w = image_bgr.shape[:2]
        new_width = min(w, int(target_height * w / h))
        new_height = min(h, int(target_width * h / w))
        start_x = w // 2 - new_width // 2
        start_y = h // 2 - new_height // 2
        return image_bgr[start_y : start_y + new_height, start_x : start_x + new_width]

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
    ) -> CropResult:
        """Load image, select region, resize, save. Returns timing and strategy used."""
        t0 = perf_counter()
        if output_path is None:
            out = input_path.parent / f"{input_path.stem}-cropped.jpg"
        else:
            out = output_path

        if out.exists() and not overwrite:
            elapsed_ms = int((perf_counter() - t0) * 1000)
            return CropResult(
                input_path=input_path,
                output_path=None,
                strategy_used="",
                elapsed_ms=elapsed_ms,
                target_width=target_width,
                target_height=target_height,
                error=f"refusing to overwrite existing file: {out}",
            )

        try:
            with Image.open(input_path) as pil:
                pil.load()
                image_bgr = _pil_to_bgr(pil)
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
            return CropResult(
                input_path=input_path,
                output_path=None,
                strategy_used="",
                elapsed_ms=elapsed_ms,
                target_width=target_width,
                target_height=target_height,
                error=str(exc),
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
        )
