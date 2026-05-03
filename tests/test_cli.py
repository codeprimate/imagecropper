"""Tests for the imagecropper CLI."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
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
    assert "--output-dir" in result.output


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


def test_crop_output_and_output_dir_mutually_exclusive(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(img)
    out = tmp_path / "o.jpg"
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "crop",
            str(img),
            "--width",
            "2",
            "--height",
            "2",
            "-s",
            "center",
            "-o",
            str(out),
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code != 0
    err = result.stderr or ""
    assert "output" in err.lower() and "output-dir" in err.lower()


def test_crop_output_dir_writes_beside_out_dir_not_input(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (20, 30)).save(img)
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "crop",
            str(img),
            "-W",
            "5",
            "-H",
            "5",
            "-s",
            "center",
            "--output-dir",
            str(out_dir),
            "--quiet",
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "a-cropped.jpg").exists()
    assert not (tmp_path / "a-cropped.jpg").exists()


def test_crop_output_dir_two_inputs(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.new("RGB", (10, 10)).save(a)
    Image.new("RGB", (10, 10)).save(b)
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "crop",
            str(a),
            str(b),
            "-W",
            "4",
            "-H",
            "4",
            "-s",
            "center",
            "--output-dir",
            str(out_dir),
            "--quiet",
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "a-cropped.jpg").exists()
    assert (out_dir / "b-cropped.jpg").exists()


def test_crop_output_dir_same_stem_collision_refuses_overwrite(
    tmp_path: Path,
) -> None:
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    a1 = d1 / "x.png"
    a2 = d2 / "x.png"
    Image.new("RGB", (10, 10)).save(a1)
    Image.new("RGB", (10, 10)).save(a2)
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "crop",
            str(a1),
            str(a2),
            "-W",
            "4",
            "-H",
            "4",
            "-s",
            "center",
            "--output-dir",
            str(out_dir),
            "--quiet",
        ],
    )
    assert result.exit_code == 1
    assert (out_dir / "x-cropped.jpg").exists()
    err = result.stderr or ""
    assert "overwrite" in err.lower()


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


def test_crop_explicit_output_multi_subjects_error(tmp_path: Path, mocker: Any) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (20, 20)).save(img)
    out = tmp_path / "single.jpg"
    p1 = np.ones((6, 6, 3), dtype=np.uint8)
    p2 = p1 * 2
    mocker.patch.object(
        ImageCropper,
        "_select_regions",
        return_value=[(p1, "human"), (p2, "human")],
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", str(img), "-W", "4", "-H", "4", "-s", "human", "-o", str(out), "--quiet"],
    )
    assert result.exit_code == 1
    err = result.stderr or ""
    assert "multiple subjects" in err.lower()
    assert not out.exists()


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


def test_anon_help_lists_crop() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["anon", "--help"])
    assert result.exit_code == 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "--crop" in combined


def test_anon_native_rejects_strategy_without_crop(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(img)
    runner = CliRunner()
    result = runner.invoke(cli, ["anon", str(img), "-s", "center", "--quiet"])
    assert result.exit_code != 0
    assert "--crop" in (result.stderr or "")


def test_anon_native_writes_sidecar(tmp_path: Path, mocker: Any) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (16, 16)).save(img)
    mocker.patch.object(ImageCropper, "detect_face_bbox", return_value=None)
    runner = CliRunner()
    result = runner.invoke(cli, ["anon", str(img), "--quiet"])
    assert result.exit_code == 0
    assert (tmp_path / "a-anon.jpg").exists()


def test_anon_crop_matches_crop_anon_bytes(tmp_path: Path, mocker: Any) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (24, 24)).save(img)
    mocker.patch.object(ImageCropper, "detect_face_bbox", return_value=(4, 4, 16, 16))
    mocker.patch(
        "imagecropper.crop.anonymize_face_inpaint",
        side_effect=lambda im, bb: im,
    )
    runner = CliRunner()
    r1 = runner.invoke(
        cli,
        ["crop", str(img), "-W", "9", "-H", "9", "-s", "center", "--anon", "--force", "--quiet"],
    )
    assert r1.exit_code == 0
    out = tmp_path / "a-cropped.jpg"
    b1 = out.read_bytes()
    r2 = runner.invoke(
        cli,
        ["anon", str(img), "--crop", "-W", "9", "-H", "9", "-s", "center", "--force", "--quiet"],
    )
    assert r2.exit_code == 0
    assert out.read_bytes() == b1


def test_crop_format_webp_writes_webp(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (20, 30)).save(img)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", str(img), "-W", "5", "-H", "5", "-s", "center", "--format", "webp", "--quiet"],
    )
    assert result.exit_code == 0
    out = tmp_path / "a-cropped.webp"
    assert out.exists()
    with Image.open(out) as pil:
        assert pil.format == "WEBP"


def test_crop_format_png_writes_png(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (20, 30)).save(img)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", str(img), "-W", "5", "-H", "5", "-s", "center", "-f", "png", "--quiet"],
    )
    assert result.exit_code == 0
    out = tmp_path / "a-cropped.png"
    assert out.exists()
    with Image.open(out) as pil:
        assert pil.format == "PNG"


def test_crop_explicit_output_suffix_rewritten_via_cli(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (20, 30)).save(img)
    requested = tmp_path / "out.jpg"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "crop",
            str(img),
            "-W",
            "5",
            "-H",
            "5",
            "-s",
            "center",
            "-o",
            str(requested),
            "--format",
            "png",
            "--quiet",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "out.png").exists()
    assert not requested.exists()


def test_crop_quality_below_range_rejected(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(img)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", str(img), "-W", "2", "-H", "2", "-s", "center", "--quality", "0"],
    )
    assert result.exit_code != 0


def test_crop_quality_above_range_rejected(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(img)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["crop", str(img), "-W", "2", "-H", "2", "-s", "center", "--quality", "101"],
    )
    assert result.exit_code != 0


def test_crop_png_with_quality_is_accepted(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(img)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "crop",
            str(img),
            "-W",
            "4",
            "-H",
            "4",
            "-s",
            "center",
            "-f",
            "png",
            "-q",
            "50",
            "--quiet",
        ],
    )
    assert result.exit_code == 0
    out = tmp_path / "a-cropped.png"
    assert out.exists()
    with Image.open(out) as pil:
        assert pil.format == "PNG"


def test_anon_native_format_writes_extension(tmp_path: Path, mocker: Any) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (16, 16)).save(img)
    mocker.patch.object(ImageCropper, "detect_face_bbox", return_value=None)
    runner = CliRunner()
    result = runner.invoke(cli, ["anon", str(img), "-f", "webp", "--quiet"])
    assert result.exit_code == 0
    out = tmp_path / "a-anon.webp"
    assert out.exists()
    with Image.open(out) as pil:
        assert pil.format == "WEBP"


def test_anon_crop_format_writes_extension(tmp_path: Path, mocker: Any) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (24, 24)).save(img)
    mocker.patch.object(ImageCropper, "detect_face_bbox", return_value=None)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "anon",
            str(img),
            "--crop",
            "-W",
            "8",
            "-H",
            "8",
            "-s",
            "center",
            "-f",
            "png",
            "--quiet",
        ],
    )
    assert result.exit_code == 0
    out = tmp_path / "a-cropped.png"
    assert out.exists()
    with Image.open(out) as pil:
        assert pil.format == "PNG"


def test_crop_debug_with_format_writes_debug_extension(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    Image.new("RGB", (20, 30)).save(img)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "crop",
            str(img),
            "-W",
            "5",
            "-H",
            "5",
            "-s",
            "center",
            "--debug",
            "-f",
            "webp",
            "--quiet",
        ],
    )
    assert result.exit_code == 0
    dbg = tmp_path / "a-debug.webp"
    assert dbg.exists()
    with Image.open(dbg) as pil:
        assert pil.format == "WEBP"
