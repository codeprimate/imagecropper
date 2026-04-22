from __future__ import annotations

import shutil
from pathlib import Path

import click

from imagecropper import __version__
from imagecropper.crop import CropResult


def _term_width() -> int:
    return max(60, shutil.get_terminal_size(fallback=(100, 24)).columns)


def truncate_middle(text: str, max_len: int) -> str:
    """Truncate with middle ellipsis when longer than ``max_len``."""
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    head = (max_len - 1) // 2
    tail = max_len - 1 - head
    return f"{text[:head]}…{text[-tail:]}"


def format_path_cell(path: Path | None, max_len: int) -> str:
    if path is None:
        return truncate_middle("", max_len).ljust(max_len)
    s = path.expanduser().as_posix()
    return truncate_middle(s, max_len).ljust(max_len)


def banner_line1(
    strategy: str,
    model_dir: Path,
    out_width: int,
    out_height: int,
    *,
    anon: bool = False,
    enhance: bool = True,
) -> str:
    model_display = model_dir.expanduser().as_posix()
    anon_part = " anon=on" if anon else ""
    enhance_part = " enhance=off" if not enhance else ""
    logical_prefix = (
        f"imagecropper {__version__}  crop  out={out_width}x{out_height}  "
        f"strategy={strategy}{anon_part}{enhance_part}  model_dir="
    )
    avail = max(12, _term_width() - len(logical_prefix))
    tail = truncate_middle(model_display, avail)
    meta_plain = logical_prefix[len(f"imagecropper {__version__}") :]
    head = click.style(f"imagecropper {__version__}", fg="cyan", bold=True)
    meta = click.style(meta_plain, fg="bright_black")
    return head + meta + tail + click.style("", reset=True)


def banner_separator() -> str:
    n = min(_term_width(), 120)
    return click.style("─" * n, fg="bright_black")


def progress_line(result: CropResult) -> str:
    """One stderr line per input: status, timing, basenames, strategy, output size."""
    wxh = f"{result.target_width}x{result.target_height}"
    in_base = result.input_path.name
    ms = click.style(f"{result.elapsed_ms}ms", fg="bright_black")

    if result.error:
        status = click.style("fail", fg="red", bold=True)
        out_base = "—"
        strat = truncate_middle(result.strategy_used or "—", 24)
        err = truncate_middle(result.error, 48)
        body = (
            f"{status}  {ms}  {click.style(in_base, bold=True)}"
            f"  {click.style(out_base, fg='bright_black')}  "
            f"{click.style(strat, fg='yellow')}  "
            f"{click.style(wxh, fg='bright_black')}  "
            f"{click.style(err, fg='red')}"
        )
        return body + click.style("", reset=True)

    assert result.output_path is not None
    out_base = result.output_path.name
    status = click.style("ok", fg="green", bold=True)
    strat = truncate_middle(result.strategy_used, 24)
    arrow = click.style("  →  ", fg="bright_black")
    dbg = ""
    if result.debug_output_path is not None:
        dbg = f"  {click.style('dbg=', fg='bright_black')}{click.style(result.debug_output_path.name, fg='cyan')}"
    line = (
        f"{status}  {ms}  {click.style(in_base, bold=True)}"
        f"{arrow}{click.style(out_base, bold=True)}  "
        f"{click.style(strat, fg='yellow')}  "
        f"{click.style(wxh, fg='bright_black')}"
        f"{dbg}"
    )
    return line + click.style("", reset=True)


def footer(ok: int, failed: int, total_seconds: float, output_dir: Path) -> str:
    d = output_dir.expanduser().as_posix()
    d_show = truncate_middle(d, min(88, max(24, _term_width() - 36)))
    dot = click.style(" · ", fg="bright_black")
    time_s = click.style(f"{total_seconds:.2f}s", fg="bright_black")

    ok_seg = click.style(f"{ok} ok", fg="green")
    if failed:
        fail_seg = click.style(f"{failed} failed", fg="red", bold=True)
    else:
        fail_seg = click.style("0 failed", fg="bright_black")

    arrow = click.style(" → ", fg="bright_black")
    path = click.style(d_show, fg="cyan")
    line = f"  {ok_seg}{dot}{fail_seg}{dot}{time_s}{arrow}{path}"
    return line + click.style("", reset=True)
