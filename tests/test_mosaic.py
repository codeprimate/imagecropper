"""Tests for the tight crop mosaic."""

import pytest
from PIL import Image

from imagecropper.mosaic import (
    MAX_MOSAIC_SIDE,
    TileBox,
    column_count,
    compose_mosaic,
    layout_tiles,
)


def test_column_count_is_ceil_sqrt() -> None:
    assert column_count(1) == 1
    assert column_count(2) == 2
    assert column_count(3) == 2
    assert column_count(4) == 2
    assert column_count(5) == 3


def test_column_count_rejects_zero() -> None:
    with pytest.raises(ValueError):
        column_count(0)


def test_layout_rejects_empty_and_bad_limits() -> None:
    with pytest.raises(ValueError):
        layout_tiles(())
    with pytest.raises(ValueError):
        layout_tiles([(4, 4)], max_side=0)
    with pytest.raises(ValueError):
        layout_tiles([(0, 4)])


def test_single_tile_keeps_its_size() -> None:
    boxes, canvas = layout_tiles([(40, 20)])
    assert canvas == (40, 20)
    assert boxes == (boxes[0],)
    assert boxes[0].width == 40
    assert boxes[0].height == 20
    assert boxes[0].x == 0
    assert boxes[0].y == 0


def test_two_by_two_stays_at_crop_size() -> None:
    boxes, canvas = layout_tiles([(10, 10)] * 4)
    assert canvas == (20, 20)
    assert [(box.width, box.height) for box in boxes] == [(10, 10)] * 4
    assert [(box.x, box.y) for box in boxes] == [(0, 0), (10, 0), (0, 10), (10, 10)]


def test_short_row_uses_shared_max_extent_and_fuller_row_shrinks() -> None:
    boxes, canvas = layout_tiles([(32, 32)] * 3)
    assert canvas[0] == 32
    assert canvas[1] == 48
    assert [(box.width, box.height) for box in boxes] == [(16, 16), (16, 16), (32, 32)]
    assert all(box.width <= 32 and box.height <= 32 for box in boxes)
    assert _row_widths(boxes) == [32, 32]
    assert _covers_canvas(boxes, canvas)


def test_wide_tiles_stay_inside_source_extent() -> None:
    boxes, canvas = layout_tiles([(30, 20)] * 3)
    assert all(box.width <= 30 and box.height <= 20 for box in boxes)
    assert len(_row_widths(boxes)) == 2
    assert len(set(_row_widths(boxes))) == 1
    assert _covers_canvas(boxes, canvas)
    assert canvas[0] <= MAX_MOSAIC_SIDE
    assert canvas[1] <= MAX_MOSAIC_SIDE
    short = boxes[2]
    assert short.width * short.height > boxes[0].width * boxes[0].height


def test_canvas_shrinks_to_max_side() -> None:
    boxes, canvas = layout_tiles([(40, 40), (40, 40)], max_side=30)
    assert canvas[0] <= 30
    assert canvas[1] <= 30
    assert all(box.width <= 40 and box.height <= 40 for box in boxes)
    assert _covers_canvas(boxes, canvas)
    assert boxes[0].height == boxes[1].height


def test_large_square_is_capped_at_max_side() -> None:
    _boxes, canvas = layout_tiles([(5000, 100)])
    assert canvas[0] <= MAX_MOSAIC_SIDE
    assert canvas[1] <= MAX_MOSAIC_SIDE
    assert canvas[0] > canvas[1]


def test_impossible_max_side_is_rejected() -> None:
    with pytest.raises(ValueError):
        layout_tiles([(10, 10), (10, 10)], max_side=1)


def test_one_pixel_short_row_cannot_match_widths() -> None:
    with pytest.raises(ValueError):
        layout_tiles([(1, 1), (1, 1), (1, 1)])


def test_compose_shrinks_fuller_row_and_accepts_rgba() -> None:
    tiles = [
        Image.new("RGB", (32, 32), (255, 0, 0)),
        Image.new("RGB", (32, 32), (0, 255, 0)),
        Image.new("RGBA", (32, 32), (0, 0, 255, 255)),
    ]
    mosaic = compose_mosaic(tiles)
    assert mosaic.size == (32, 48)
    assert mosaic.getpixel((0, 0)) == (255, 0, 0)
    assert mosaic.getpixel((16, 0)) == (0, 255, 0)
    assert mosaic.getpixel((0, 16)) == (0, 0, 255)


def test_compose_places_colors_in_order_without_gaps() -> None:
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    tiles = [Image.new("RGB", (10, 10), color) for color in colors]
    mosaic = compose_mosaic(tiles)
    assert mosaic.size == (20, 20)
    assert mosaic.getpixel((0, 0)) == colors[0]
    assert mosaic.getpixel((10, 0)) == colors[1]
    assert mosaic.getpixel((0, 10)) == colors[2]
    assert mosaic.getpixel((10, 10)) == colors[3]
    seen = {mosaic.getpixel((x, y)) for x in range(20) for y in range(20)}
    assert seen == set(colors)


def _row_widths(boxes: tuple[TileBox, ...]) -> list[int]:
    rows: dict[int, int] = {}
    for box in boxes:
        rows[box.y] = rows.get(box.y, 0) + box.width
    return [rows[key] for key in sorted(rows)]


def _covers_canvas(boxes: tuple[TileBox, ...], canvas: tuple[int, int]) -> bool:
    area = sum(box.width * box.height for box in boxes)
    return area == canvas[0] * canvas[1]
