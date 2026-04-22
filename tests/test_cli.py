"""Tests for the imagecropper CLI."""

import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from PIL import Image

from imagecropper import __version__
from imagecropper.cli import _footer_output_dir, cli, main
from imagecropper.crop import CropResult, ImageCropper


def test_version_option() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_option() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "imagecropper" in result.output


def test_no_subcommand_prints_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_main_dispatches_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["imagecropper", "--version"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


def test_crop_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["crop", "--help"])
    assert result.exit_code == 0
    assert "WIDTH" in result.output or "--width" in result.output
    assert "--anon" in result.output
    assert "--enhance" in result.output
    assert "--no-enhance" in result.output
    assert "--debug" in result.output


def test_crop_debug_writes_debug_jpeg(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (20, 30)).save(img)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", str(img), "-W", "5", "-H", "5", "-s", "center", "--debug", "--quiet"],
    )
    assert result.exit_code == 0
    dbg = tmp_path / "a-debug.jpg"
    assert dbg.exists()
    with Image.open(dbg) as pil:
        assert pil.size == (20, 30)


def test_crop_requires_positive_dimensions(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(img)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", str(img), "--width", "0", "--height", "4", "--strategy", "center"],
    )
    assert result.exit_code != 0


def test_crop_output_requires_single_input(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.new("RGB", (4, 4)).save(a)
    Image.new("RGB", (4, 4)).save(b)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", str(a), str(b), "--width", "2", "--height", "2", "-o", str(tmp_path / "o.jpg")],
    )
    assert result.exit_code != 0


def test_crop_defaults_to_1024_without_dimensions(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (100, 100)).save(img)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", str(img), "--strategy", "center", "--quiet"],
    )
    assert result.exit_code == 0
    out = tmp_path / "a-cropped.jpg"
    assert out.exists()
    with Image.open(out) as cropped:
        assert cropped.size == (1024, 1024)


def test_crop_center_quiet(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (20, 30)).save(img)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", str(img), "-W", "5", "-H", "5", "--strategy", "center", "--quiet"],
    )
    assert result.exit_code == 0
    assert (tmp_path / "a-cropped.jpg").exists()


def test_crop_strategy_short_option(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (20, 30)).save(img)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", str(img), "-W", "5", "-H", "5", "-s", "center", "--quiet"],
    )
    assert result.exit_code == 0
    assert (tmp_path / "a-cropped.jpg").exists()


def test_crop_center_verbose_shows_progress(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(img)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", str(img), "--width", "2", "--height", "2", "--strategy", "center"],
    )
    assert result.exit_code == 0
    err = result.stderr or ""
    assert "a.png" in err
    assert "a-cropped.jpg" in err
    assert "ok" in err
    assert "center" in err


def test_crop_missing_file_exits_nonzero() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", "/nonexistent/nope.jpg", "--width", "2", "--height", "2", "--strategy", "center"],
    )
    assert result.exit_code != 0


def test_footer_output_dir_single_parent() -> None:
    results = [
        CropResult(
            input_path=Path("/in/a.jpg"),
            output_path=Path("/out/x.jpg"),
            strategy_used="center",
            elapsed_ms=1,
            target_width=2,
            target_height=2,
            error=None,
        )
    ]
    assert _footer_output_dir(results) == Path("/out")


def test_footer_output_dir_mixed_parents() -> None:
    results = [
        CropResult(
            input_path=Path("/a"),
            output_path=Path("/out1/x.jpg"),
            strategy_used="center",
            elapsed_ms=1,
            target_width=1,
            target_height=1,
            error=None,
        ),
        CropResult(
            input_path=Path("/b"),
            output_path=Path("/out2/y.jpg"),
            strategy_used="center",
            elapsed_ms=1,
            target_width=1,
            target_height=1,
            error=None,
        ),
    ]
    assert _footer_output_dir(results) == Path.cwd()


def test_crop_cli_failure_exit_code(tmp_path: Path, mocker: Any) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(img)
    bad = CropResult(
        input_path=img,
        output_path=None,
        strategy_used="",
        elapsed_ms=1,
        target_width=2,
        target_height=2,
        error="simulated failure",
    )
    mocker.patch.object(ImageCropper, "crop_one", return_value=bad)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", str(img), "--width", "2", "--height", "2", "--strategy", "center"],
    )
    assert result.exit_code == 1
    assert "simulated failure" in (result.stderr or "")


def test_crop_quiet_with_error_still_prints_stderr(tmp_path: Path, mocker: Any) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(img)
    bad = CropResult(
        input_path=img,
        output_path=None,
        strategy_used="",
        elapsed_ms=1,
        target_width=2,
        target_height=2,
        error="quiet error",
    )
    mocker.patch.object(ImageCropper, "crop_one", return_value=bad)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", str(img), "--width", "2", "--height", "2", "--strategy", "center", "--quiet"],
    )
    assert result.exit_code == 1
    assert "quiet error" in (result.stderr or "")


def test_footer_output_dir_all_failed() -> None:
    results = [
        CropResult(
            input_path=Path("/a"),
            output_path=None,
            strategy_used="",
            elapsed_ms=1,
            target_width=1,
            target_height=1,
            error="err",
        )
    ]
    assert _footer_output_dir(results) == Path.cwd()
