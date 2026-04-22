"""Command-line interface for imagecropper."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from time import perf_counter
from typing import cast

import click

from imagecropper import __version__
from imagecropper.console import (
    banner_line1,
    banner_separator,
    footer,
    progress_line,
)
from imagecropper.crop import CropResult, ImageCropper, StrategyName


def _echo_err(message: str = "", *, nl: bool = True) -> None:
    """Write to stderr and flush so progress is visible under captured/piped stderr."""
    click.echo(message, err=True, nl=nl)
    with contextlib.suppress(AttributeError, OSError):
        sys.stderr.flush()


def _crop_progress(msg: str) -> None:
    """Interim status while ``crop_one`` runs (per detection / per output file)."""
    _echo_err(click.style(f"  ...  {msg}", fg="bright_black"))


@click.group(name="imagecropper", invoke_without_command=True)
@click.version_option(version=__version__, prog_name="imagecropper")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Image cropping and related image utilities."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help(), err=False)


@cli.command("crop")
@click.argument(
    "inputs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--width",
    "-W",
    type=int,
    default=1024,
    show_default=True,
    help="Output width in pixels.",
)
@click.option(
    "--height",
    "-H",
    type=int,
    default=1024,
    show_default=True,
    help="Output height in pixels.",
)
@click.option(
    "-s",
    "--strategy",
    type=click.Choice(["auto", "human", "face", "center"]),
    default="auto",
    show_default=True,
    help="How to choose the crop region before resize.",
)
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for cached detector files (default: ~/.imagecropper).",
)
@click.option(
    "--output",
    "-o",
    "output",
    type=click.Path(path_type=Path),
    default=None,
    help="Explicit output path (only when a single INPUT is given).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite an existing output file.",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Suppress banner, progress lines, and summary; still print errors.",
)
@click.option(
    "--anon",
    is_flag=True,
    default=False,
    help="After crop and resize, blur inside an expanded SSD face oval (inpaint + Gaussian).",
)
@click.option(
    "--enhance/--no-enhance",
    default=True,
    show_default=True,
    help="After resize, GFPGAN-enhance if a face is detected (default: on).",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Write {stem}-debug.jpg beside each input: original with detector/crop boxes and corner (x,y) labels.",
)
def crop_command(
    inputs: tuple[Path, ...],
    width: int,
    height: int,
    strategy: str,
    model_dir: Path | None,
    output: Path | None,
    force: bool,
    quiet: bool,
    anon: bool,
    enhance: bool,
    debug: bool,
) -> None:
    """Crop each INPUT to WIDTH×HEIGHT (defaults 1024×1024) using person / face / center heuristics."""
    if width <= 0 or height <= 0:
        raise click.BadParameter("width and height must be positive integers")

    input_paths = list(inputs)
    if output is not None and len(input_paths) != 1:
        raise click.UsageError("--output requires exactly one INPUT")

    cropper = ImageCropper(model_dir=model_dir)
    resolved_model_dir = cropper.model_dir

    if not quiet:
        _echo_err(
            banner_line1(
                strategy,
                resolved_model_dir,
                width,
                height,
                anon=anon,
                enhance=enhance,
            ),
        )
        _echo_err(banner_separator())
        _echo_err("")

    results: list[CropResult] = []
    t0 = perf_counter()
    for path in input_paths:
        out_path: Path | None = output if len(input_paths) == 1 else None
        result = cropper.crop_one(
            path,
            width,
            height,
            cast(StrategyName, strategy),
            out_path,
            force,
            anon=anon,
            enhance=enhance,
            debug=debug,
            progress=_crop_progress if not quiet else None,
        )
        results.append(result)
        if result.error or not quiet:
            _echo_err(progress_line(result))

    total_s = perf_counter() - t0
    ok = sum(1 for r in results if not r.error)
    failed = len(results) - ok

    if not quiet:
        _echo_err("")
        out_dir = _footer_output_dir(results)
        _echo_err(footer(ok, failed, total_s, out_dir))

    if failed:
        raise click.exceptions.Exit(1)


def _footer_output_dir(results: list[CropResult]) -> Path:
    outs = [r.output_path for r in results if r.output_path is not None and r.error is None]
    if not outs:
        return Path.cwd()
    parents = {p.parent for p in outs}
    if len(parents) == 1:
        return next(iter(parents))
    return Path.cwd()


def main() -> None:
    """Console script entrypoint."""
    # Line-buffer stderr when supported so progress lines appear under captured/piped stderr
    # (e.g. IDE tasks) instead of only after the process exits.
    _reconfigure = getattr(sys.stderr, "reconfigure", None)
    if callable(_reconfigure):
        with contextlib.suppress(OSError, ValueError):
            _reconfigure(line_buffering=True)
    cli()
