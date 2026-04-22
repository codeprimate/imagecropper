"""GFPGAN face enhancement (optional ``[enhance]`` extra)."""

from __future__ import annotations

import importlib.util
import sys
import types
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import requests

from imagecropper.models import ensure_model_directory

GFPGAN_V14_FILENAME = "GFPGANv1.4.pth"
GFPGAN_V14_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"

_gfpganer_by_dir: dict[str, Any] = {}

_TORCHVISION_MODEL_UTILS = r"torchvision\.models\._utils"


def _suppress_torchvision_pretrained_deprecation_warnings() -> None:
    """facexlib/basicsr still pass ``pretrained=``; silence torchvision 0.13+ UserWarnings."""
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=_TORCHVISION_MODEL_UTILS,
    )


def _compat_torchvision_for_basicsr() -> None:
    """Shim removed ``torchvision.transforms.functional_tensor`` (basicsr 1.4.x)."""
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    import torchvision.transforms.functional as tv_functional

    mod = types.ModuleType("torchvision.transforms.functional_tensor")
    mod.rgb_to_grayscale = tv_functional.rgb_to_grayscale  # type: ignore[attr-defined]
    sys.modules["torchvision.transforms.functional_tensor"] = mod


def _ensure_gfpgan_weights(model_dir: Path) -> Path:
    ensure_model_directory(model_dir)
    dest = model_dir / GFPGAN_V14_FILENAME
    if not dest.exists():
        response = requests.get(GFPGAN_V14_URL, timeout=300)
        response.raise_for_status()
        dest.write_bytes(response.content)
    return dest


def _resolve_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_or_create_gfpganer(model_dir: Path) -> Any:
    key = str(model_dir.resolve())
    if key in _gfpganer_by_dir:
        return _gfpganer_by_dir[key]

    _compat_torchvision_for_basicsr()
    _suppress_torchvision_pretrained_deprecation_warnings()
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    from gfpgan import GFPGANer

    model_path = _ensure_gfpgan_weights(model_dir)
    aux_dir = model_dir / "gfpgan_auxiliary"
    aux_dir.mkdir(parents=True, exist_ok=True)

    original_init = FaceRestoreHelper.__init__

    def patched_init(
        self: Any,
        upscale_factor: int,
        face_size: int = 512,
        crop_ratio: tuple[int, int] = (1, 1),
        det_model: str = "retinaface_resnet50",
        save_ext: str = "png",
        template_3points: bool = False,
        pad_blur: bool = False,
        use_parse: bool = False,
        device: str | None = None,
        model_rootpath: str | None = None,
    ) -> None:
        root = str(aux_dir) if model_rootpath in (None, "gfpgan/weights") else model_rootpath
        original_init(
            self,
            upscale_factor,
            face_size,
            crop_ratio,
            det_model,
            save_ext,
            template_3points,
            pad_blur,
            use_parse,
            device,
            root,
        )

    FaceRestoreHelper.__init__ = patched_init
    try:
        device = _resolve_device()
        restorer = GFPGANer(
            model_path=str(model_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )
    finally:
        FaceRestoreHelper.__init__ = original_init

    _gfpganer_by_dir[key] = restorer
    return restorer


def enhance_bgr_gfpgan(image_bgr: npt.NDArray[np.uint8], model_dir: Path) -> npt.NDArray[np.uint8]:
    """Run GFPGAN on a BGR uint8 image; return BGR uint8 of the same shape.

    Raises:
        ImportError: if the ``enhance`` extra is not installed.
        OSError, RuntimeError: on model load / inference failure (caller may fall back).
    """
    _compat_torchvision_for_basicsr()
    if importlib.util.find_spec("gfpgan") is None:
        raise ImportError("GFPGAN is not installed. Install with: uv sync --extra enhance")

    if image_bgr.size == 0:
        raise ValueError("empty image")

    restorer = _get_or_create_gfpganer(model_dir)
    _c, _r, restored_bgr = restorer.enhance(
        image_bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
    )
    if restored_bgr.dtype != np.uint8:
        restored_bgr = np.clip(restored_bgr, 0, 255).astype(np.uint8)
    return cast(npt.NDArray[np.uint8], restored_bgr)
