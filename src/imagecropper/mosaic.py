"""Tight mosaic of already-cropped tiles.

Rows share one width. No tile is enlarged past its own crop size, which is one
shared maximum when every crop uses the same ``--width`` and ``--height``.
A short row keeps larger tiles. Fuller rows shrink. The canvas is scaled down
when either side would pass ``MAX_MOSAIC_SIDE``. There is no padding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isqrt
from pathlib import Path

from PIL import Image

from imagecropper.crop import (
    OutputFormat,
    _apply_format_to_explicit_path,
    _save_pil_image,
)

MAX_MOSAIC_SIDE = 4096
_FIT_ATTEMPTS = 16

_NO_TIGHT_LAYOUT = "tiles cannot form a tight mosaic within the shared max extent"
_EXCEEDS_SIDE = "mosaic exceeds the maximum side length"


@dataclass(frozen=True)
class TileBox:
    """Pixel box of one tile on the mosaic canvas. ``x`` and ``y`` are the top-left."""

    x: int
    y: int
    width: int
    height: int


def column_count(tile_count: int) -> int:
    """Smallest column count ``c`` with ``c * c >= tile_count``."""
    if tile_count < 1:
        msg = "tile_count must be positive"
        raise ValueError(msg)
    root = isqrt(tile_count)
    if root * root < tile_count:
        return root + 1
    return root


def resolve_mosaic_path(path: Path, fmt: OutputFormat) -> Path:
    """Force ``path``'s suffix to ``fmt``, with the same ``.jpeg`` rule as ``--output``."""
    return _apply_format_to_explicit_path(path, fmt)


def layout_tiles(
    sizes: Sequence[tuple[int, int]],
    *,
    max_side: int = MAX_MOSAIC_SIDE,
) -> tuple[tuple[TileBox, ...], tuple[int, int]]:
    """Return each tile's box and the ``(width, height)`` of the tight canvas.

    ``sizes`` is ``(width, height)`` in placement order. Column count is
    :func:`column_count`. Tiles fill each row left to right, then the next row.
    """
    if max_side < 1:
        msg = "max_side must be positive"
        raise ValueError(msg)
    if len(sizes) < 1:
        msg = "mosaic requires at least one tile"
        raise ValueError(msg)
    for width, height in sizes:
        if width < 1 or height < 1:
            msg = "tile width and height must be positive"
            raise ValueError(msg)

    columns = column_count(len(sizes))
    row_sizes = _row_groups(sizes, columns)
    caps = [min(height for _width, height in row) for row in row_sizes]
    heights = _fit_row_heights(row_sizes, caps, max_side)
    return _place(row_sizes, heights)


def compose_mosaic(
    tiles: Sequence[Image.Image],
    *,
    max_side: int = MAX_MOSAIC_SIDE,
) -> Image.Image:
    """Join ``tiles`` into one RGB image. See :func:`layout_tiles`."""
    boxes, canvas_size = layout_tiles([tile.size for tile in tiles], max_side=max_side)
    canvas = Image.new("RGB", canvas_size)
    for tile, box in zip(tiles, boxes, strict=True):
        rgb = tile.convert("RGB")
        try:
            if (box.width, box.height) == rgb.size:
                canvas.paste(rgb, (box.x, box.y))
                continue
            resized = rgb.resize((box.width, box.height), Image.Resampling.LANCZOS)
            try:
                canvas.paste(resized, (box.x, box.y))
            finally:
                resized.close()
        finally:
            if rgb is not tile:
                rgb.close()
    return canvas


def write_mosaic(
    tiles: Sequence[Image.Image],
    path: Path,
    fmt: OutputFormat,
    quality: int | None,
    *,
    max_side: int = MAX_MOSAIC_SIDE,
) -> Image.Image:
    """Compose ``tiles`` and write ``path`` with the crop encoder (DATA-008)."""
    composed = compose_mosaic(tiles, max_side=max_side)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_pil_image(composed, path, fmt, quality)
    return composed


def _row_groups(
    sizes: Sequence[tuple[int, int]],
    columns: int,
) -> list[list[tuple[int, int]]]:
    return [list(sizes[start : start + columns]) for start in range(0, len(sizes), columns)]


def _scaled_width(row_height: int, tile_width: int, tile_height: int) -> int:
    """Half-up width for ``row_height`` while keeping ``tile_width / tile_height``."""
    return (row_height * tile_width + tile_height // 2) // tile_height


def _row_pixel_width(row_height: int, tiles: Sequence[tuple[int, int]]) -> int:
    return sum(_scaled_width(row_height, width, height) for width, height in tiles)


def _width_to_height(
    tiles: Sequence[tuple[int, int]],
    height_cap: int,
) -> dict[int, int]:
    """Largest row height at or under ``height_cap`` for each achievable row width."""
    found: dict[int, int] = {}
    for height in range(1, height_cap + 1):
        found[_row_pixel_width(height, tiles)] = height
    return found


def _solve_row_heights(
    row_sizes: Sequence[Sequence[tuple[int, int]]],
    height_caps: Sequence[int],
) -> tuple[list[int], int] | None:
    """Row heights that share one width, or ``None`` when the caps cannot meet."""
    if any(cap < 1 for cap in height_caps):
        return None
    maps = [_width_to_height(row, cap) for row, cap in zip(row_sizes, height_caps, strict=True)]
    common = set(maps[0])
    for item in maps[1:]:
        common &= set(item)
    if not common:
        return None
    width = max(common)
    return [item[width] for item in maps], width


def _fit_row_heights(
    row_sizes: Sequence[Sequence[tuple[int, int]]],
    height_caps: Sequence[int],
    max_side: int,
) -> list[int]:
    """Shrink row-height caps until the tight canvas fits in ``max_side``."""
    caps = list(height_caps)
    for _ in range(_FIT_ATTEMPTS):
        solved = _solve_row_heights(row_sizes, caps)
        if solved is None:
            raise ValueError(_NO_TIGHT_LAYOUT)
        heights, width = solved
        total_height = sum(heights)
        if width <= max_side and total_height <= max_side:
            return heights
        scale = min(max_side / width, max_side / total_height)
        shrunk = [max(1, int(height * scale)) for height in heights]
        if shrunk == heights:
            break
        caps = [min(old, new) for old, new in zip(caps, shrunk, strict=True)]
    raise ValueError(_EXCEEDS_SIDE)


def _place(
    row_sizes: Sequence[Sequence[tuple[int, int]]],
    heights: Sequence[int],
) -> tuple[tuple[TileBox, ...], tuple[int, int]]:
    boxes: list[TileBox] = []
    y = 0
    canvas_width = 0
    for row, row_height in zip(row_sizes, heights, strict=True):
        x = 0
        for tile_width, tile_height in row:
            if row_height > tile_height:
                raise ValueError(_NO_TIGHT_LAYOUT)
            width = _scaled_width(row_height, tile_width, tile_height)
            if width > tile_width:
                raise ValueError(_NO_TIGHT_LAYOUT)
            boxes.append(TileBox(x, y, width, row_height))
            x += width
        if canvas_width == 0:
            canvas_width = x
        elif x != canvas_width:
            raise ValueError(_NO_TIGHT_LAYOUT)
        y += row_height
    if canvas_width < 1 or y < 1:
        raise ValueError(_NO_TIGHT_LAYOUT)
    return tuple(boxes), (canvas_width, y)
