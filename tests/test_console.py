"""Tests for console formatting."""

from pathlib import Path

import click

from imagecropper.console import (
    banner_line1,
    banner_separator,
    footer,
    format_path_cell,
    progress_line,
    truncate_middle,
)
from imagecropper.crop import CropResult


def test_truncate_middle_short_unchanged() -> None:
    assert truncate_middle("abc", 10) == "abc"


def test_truncate_middle_long() -> None:
    s = "abcdefghijklmnopqrstuvwxyz"
    out = truncate_middle(s, 12)
    assert "…" in out
    assert len(out) == 12


def test_truncate_middle_tiny_max() -> None:
    assert len(truncate_middle("abcdef", 2)) == 2


def test_truncate_middle_non_positive_max() -> None:
    assert truncate_middle("abcdef", 0) == ""


def test_format_path_cell_none() -> None:
    cell = format_path_cell(None, 10)
    assert len(cell) == 10


def test_format_path_cell_path() -> None:
    cell = format_path_cell(Path("/tmp/x.jpg"), 40)
    assert "x.jpg" in click.unstyle(cell) or "tmp" in cell


def test_banner_and_separator() -> None:
    line = banner_line1("auto", Path("/tmp/models"), 1024, 1024)
    plain = click.unstyle(line)
    assert "imagecropper" in plain
    assert "out=1024x1024" in plain
    assert "strategy=auto" in plain
    assert "anon=on" not in plain
    line_anon = banner_line1("auto", Path("/tmp/models"), 800, 600, anon=True)
    assert "out=800x600" in click.unstyle(line_anon)
    assert "anon=on" in click.unstyle(line_anon)
    line_no_enh = banner_line1("center", Path("/tmp/m"), 64, 64, enhance=False)
    assert "enhance=off" in click.unstyle(line_no_enh)
    assert len(click.unstyle(banner_separator())) >= 60


def test_banner_native_anon_command() -> None:
    line = banner_line1("auto", Path("/tmp/models"), 0, 0, command_label="anon", native_output=True)
    plain = click.unstyle(line)
    assert "anon" in plain
    assert "out=native" in plain
    assert "strategy=auto" not in plain
    assert "enhance=off" in plain


def test_banner_anon_crop_command_label() -> None:
    line = banner_line1(
        "center",
        Path("/tmp/m"),
        8,
        8,
        anon=True,
        enhance=True,
        command_label="anon",
    )
    plain = click.unstyle(line)
    assert "anon" in plain
    assert "out=8x8" in plain
    assert "strategy=center" in plain
    assert "anon=on" in plain


def test_progress_line_success() -> None:
    r = CropResult(
        input_path=Path("/in/a.jpg"),
        output_path=Path("/out/a-cropped.jpg"),
        strategy_used="center",
        elapsed_ms=12,
        target_width=80,
        target_height=60,
        error=None,
    )
    row = click.unstyle(progress_line(r))
    assert "a.jpg" in row
    assert "a-cropped.jpg" in row
    assert "center" in row
    assert "80x60" in row
    assert "ok" in row
    assert "12ms" in row


def test_progress_line_success_multi_outputs_shows_more_suffix() -> None:
    outs = (
        Path("/out/a-cropped-01.jpg"),
        Path("/out/a-cropped-02.jpg"),
        Path("/out/a-cropped-03.jpg"),
    )
    r = CropResult(
        input_path=Path("/in/a.jpg"),
        output_path=outs[0],
        strategy_used="human ×3",
        elapsed_ms=9,
        target_width=64,
        target_height=64,
        error=None,
        output_paths=outs,
    )
    row = click.unstyle(progress_line(r))
    assert "a-cropped-01.jpg (+2 more)" in row


def test_progress_line_success_includes_debug_basename() -> None:
    r = CropResult(
        input_path=Path("/in/a.jpg"),
        output_path=Path("/out/a-cropped.jpg"),
        strategy_used="center",
        elapsed_ms=3,
        target_width=10,
        target_height=10,
        error=None,
        debug_output_path=Path("/out/a-debug.jpg"),
    )
    row = click.unstyle(progress_line(r))
    assert "dbg=" in row
    assert "a-debug.jpg" in row


def test_progress_line_error() -> None:
    r = CropResult(
        input_path=Path("/in/a.jpg"),
        output_path=None,
        strategy_used="",
        elapsed_ms=5,
        target_width=10,
        target_height=10,
        error="something failed",
    )
    row = click.unstyle(progress_line(r))
    assert "fail" in row
    assert "a.jpg" in row
    assert "something" in row
    assert "10x10" in row


def test_footer() -> None:
    text = click.unstyle(footer(2, 1, 1.25, Path("/tmp")))
    assert "2 ok" in text
    assert "1 failed" in text
    assert "1.25s" in text
    assert "/tmp" in text
