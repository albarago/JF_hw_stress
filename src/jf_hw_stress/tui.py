"""Interactive setup wizard and live dashboard (needs `rich`)."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape as esc
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from . import APP_TITLE, __version__
from . import platforms as plat
from . import report as reports
from . import scenarios as scn
from . import tools as tl
from .command import CommandError, Options
from .controller import Thresholds
from .engine import MODE_LABELS, Engine, RunConfig, make_work_dir, remove_work_dir
from .probe import ProbeError, SourceInfo, dovi_notes, probe

console = Console()
VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".ts", ".m2ts", ".mpg", ".wmv", ".webm"}


def clean_path(raw: str) -> str:
    """Accept a dragged-and-dropped path: strip quotes and shell escapes."""
    p = raw.strip()
    if len(p) >= 2 and p[0] == p[-1] and p[0] in "'\"":
        p = p[1:-1]
    return p.replace("\\ ", " ").strip()


def rule(title: str) -> None:
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    console.print()


def countdown_choice(header: str, options: List[Tuple[str, str]], default: str, timeout: float = 60.0) -> str:
    """One keypress with a countdown (raw terminal on macOS/Linux, a normal prompt elsewhere)."""
    console.print(f"\n  [bold]{header}[/bold]\n")
    for key, desc in options:
        style = "bold yellow" if key == default else "bold"
        console.print(f"  [{style}]{key.upper()}[/{style}]  {desc}")
    console.print()
    valid = {k for k, _ in options}
    if os.name == "posix" and sys.stdin.isatty():
        import select
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            deadline = time.monotonic() + timeout
            while True:
                left = deadline - time.monotonic()
                keys = " / ".join(k.upper() for k in valid)
                sys.stdout.write(f"\r  Press {keys} — {default.upper()} in {max(0, left):.0f} s   ")
                sys.stdout.flush()
                if left <= 0:
                    return default
                ready, _, _ = select.select([sys.stdin], [], [], min(0.5, left))
                if ready:
                    ch = sys.stdin.read(1).lower()
                    if ch in valid:
                        return ch
                    if ch in ("\r", "\n"):
                        return default
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            sys.stdout.write("\n\n")
    return Prompt.ask("  Choice", choices=sorted(valid), default=default)


# --- setup steps -----------------------------------------------------------------------------

def choose_tools(path: Optional[str]) -> tl.Tools:
    if path:
        return tl.load(tl.from_path(path))
    rule("ffmpeg")
    found = tl.candidates()
    for i, t in enumerate(found, 1):
        console.print(f"  [bold]{i}[/bold]  {esc(t.origin)}  [dim]{esc(t.ffmpeg)}[/dim]")
    console.print(f"  [bold]{len(found) + 1}[/bold]  Another ffmpeg (binary or folder)\n")
    pick = int(Prompt.ask("  Use", choices=[str(i) for i in range(1, len(found) + 2)], default="1"))
    while True:
        try:
            tools = found[pick - 1] if pick <= len(found) else tl.from_path(clean_path(Prompt.ask("  Path")))
            tools = tl.load(tools)
            warn = "" if tools.is_jellyfin_build else (
                "  [yellow](not jellyfin-ffmpeg: HDR tone mapping may be unavailable)[/yellow]")
            console.print(f"  [green]✓[/green] ffmpeg {esc(tools.version)}{warn}")
            return tools
        except tl.ToolError as exc:
            console.print(f"  [red]✗ {esc(str(exc))}[/red]")
            pick = len(found) + 1


def choose_platforms(tools: tl.Tools, ids: Optional[str]) -> List[plat.Platform]:
    if ids:
        return plat.choose([x.strip() for x in ids.split(",") if x.strip()], tools.caps)
    console.print("\n  [dim]Detecting hardware …[/dim]")
    found = plat.detect(tools.caps)
    if len(found) == 1:
        p = found[0]
        if p.is_hardware:
            console.print(f"  [green]✓[/green] [bold cyan]{esc(p.name)}[/bold cyan]  {esc(p.device)}"
                          + ("" if p.tested else "  [yellow](untested backend)[/yellow]"))
        else:
            console.print("  [yellow]No hardware backend is usable with this ffmpeg build: software encoding.[/yellow]")
        return found
    console.print(f"\n  [bold]{len(found)} hardware backends:[/bold]")
    for i, p in enumerate(found, 1):
        console.print(f"    [bold]{i}[/bold]  {esc(p.name)}  [dim]{esc(p.device)}[/dim]")
    console.print("\n  One number, or several separated by commas to spread streams across them.\n")
    while True:
        parts = [x.strip() for x in Prompt.ask("  Backend(s)", default="1").split(",") if x.strip()]
        if parts and all(x.isdigit() and 1 <= int(x) <= len(found) for x in parts):
            return [found[int(x) - 1] for x in parts]
        console.print(f"  [red]Enter numbers from 1 to {len(found)}.[/red]")


def choose_source(tools: tl.Tools, preset: Optional[str]) -> SourceInfo:
    if preset:
        return probe(tools.ffprobe, str(Path(preset).expanduser()))
    rule("Source file")
    try:
        here = sorted(p for p in Path.cwd().iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)[:9]
    except OSError:
        here = []
    for i, p in enumerate(here, 1):
        console.print(f"  [bold]{i}[/bold]  {esc(p.name)}")
    console.print("\n  Enter a number, or type / drag a file path. A high-bitrate 4K HDR file is the best test.\n"
                  if here else "  Type or drag a file path. A high-bitrate 4K HDR file is the best test.\n")
    while True:
        raw = clean_path(Prompt.ask("  Source"))
        path = here[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(here) else Path(raw).expanduser()
        if path.is_file():
            try:
                return probe(tools.ffprobe, str(path))
            except ProbeError as exc:
                console.print(f"  [red]✗ {esc(str(exc))}[/red]")
                continue
        console.print(f"  [red]✗ not a file: {esc(str(path))}[/red]")


def show_source(source: SourceInfo) -> None:
    rule("Source")
    hdr = f"[bold magenta]{esc(source.hdr_type)}[/bold magenta]" if source.is_hdr else "SDR"
    if source.dovi_profile:
        hdr += f" profile {source.dovi_profile}"
    console.print(f"  Video    {source.width}×{source.height}  {esc(source.codec.upper())} {esc(source.profile)}  "
                  f"{source.bit_depth}-bit  {source.fps:.3f} fps  {source.bitrate_mbps:.1f} Mbps  {hdr}")
    console.print(f"  Audio    {esc(source.audio_label())}")
    if source.duration_secs:
        console.print(f"  Length   {source.duration_secs / 60:.1f} min  [dim](looped during the test)[/dim]")
    for note in dovi_notes(source.dovi_profile):
        console.print(f"  [yellow]⚠[/yellow] [dim]{esc(note)}[/dim]")


def choose_work_dir(preset: Optional[str]) -> str:
    if preset:
        return preset
    from .cli import default_work_base
    rule("Work folder")
    console.print("  Each stream writes a rolling set of HLS segments (old ones are deleted). A run folder is\n"
                  "  created inside the folder you choose and removed afterwards.\n")
    default = default_work_base()
    if sys.platform == "darwin":
        console.print("  [dim]macOS has no RAM disk by default; the temp folder on the internal SSD is fine for\n"
                      "  normal runs. For very long runs a RAM disk avoids SSD wear (see README).[/dim]\n")
    return clean_path(Prompt.ask("  Folder", default=default))


def setup(source: SourceInfo, platforms: List[plat.Platform], tools: tl.Tools, args: argparse.Namespace) -> RunConfig:
    rule("Test mode")
    console.print("  [bold]1[/bold]  Fixed       — a set number of streams for a set time")
    console.print("  [bold]2[/bold]  Escalating  — add streams until one can't keep up with real time")
    console.print("  [bold]3[/bold]  Hybrid      — escalate on hardware, then add software streams")
    console.print("  [bold]4[/bold]  Mixed       — escalate with a random client-like target per stream\n")
    mode = {"1": "fixed", "2": "escalating", "3": "hybrid", "4": "mixed"}[
        Prompt.ask("  Mode", choices=["1", "2", "3", "4"], default="2")]
    streams, duration = 4, None
    thresholds = Thresholds()
    if mode == "fixed":
        streams = IntPrompt.ask("  Concurrent streams", default=4)
        duration = FloatPrompt.ask("  Run for (seconds)", default=120.0)
    else:
        console.print("\n  [dim]Press Enter to keep the defaults.[/dim]")
        ramp = FloatPrompt.ask("  Seconds stable before adding a stream", default=12.0)
        fail = FloatPrompt.ask("  Seconds below real time that count as failure", default=5.0)
        while True:
            ratio = FloatPrompt.ask("  Fraction of the source frame rate each stream must keep", default=0.90)
            if 0.5 <= ratio <= 1.0:
                break
            console.print("  [red]Enter a value from 0.50 to 1.00.[/red]")
        thresholds = Thresholds(fps_ratio=ratio, ramp_interval=ramp, fail_secs=fail)

    pool: List[scn.Scenario] = []
    if mode == "mixed":
        pool = scn.mixed_pool(source)
        if not pool:
            raise CommandError("no mixed-mode scenario fits this source")
        console.print(f"\n  [dim]Mixed pool: {len(pool)} targets that fit this source (no upscaling"
                      f"{', no tone mapping for SDR' if not source.is_hdr else ''}).[/dim]")
        scenario = pool[0]
    else:
        rule("Scenario")
        table = Table(box=box.SIMPLE, header_style="bold dim", show_edge=False)
        for col in ("#", "Scenario", "Description", "Bitrate", "Tone map", ""):
            table.add_column(col)
        usable = []
        for s in scn.SCENARIOS:
            ok, why = scn.applicable(s, source)
            usable.append(ok)
            table.add_row(str(len(usable)), s.label, s.description, s.bitrate, "✓" if s.tonemap else "–",
                          "" if ok else f"[dim]{esc(why)}[/dim]")
        console.print(table)
        first_ok = str(usable.index(True) + 1) if True in usable else "1"
        while True:
            pick = int(Prompt.ask("  Scenario", choices=[str(i) for i in range(1, len(usable) + 1)], default=first_ok))
            if usable[pick - 1]:
                break
            console.print("  [red]That scenario does not fit this source.[/red]")
        scenario = scn.SCENARIOS[pick - 1]
        codec = Prompt.ask("  Output codec", choices=["keep", "h264", "hevc"], default="keep")
        if codec != "keep":
            scenario = scenario.with_codec(codec)

    hw = [p for p in platforms if p.is_hardware]
    hw_decode = False
    if hw:
        supported = any(source.codec in p.decode_codecs for p in hw)
        console.print(f"\n  Hardware decode of {esc(source.codec.upper())}: "
                      + ("[green]supported[/green]" if supported else "[yellow]not supported, software decode[/yellow]"))
        hw_decode = supported and Confirm.ask("  Decode on the hardware?", default=True)
    audio = Prompt.ask("  Audio", choices=["aac", "copy", "none"], default="aac")
    return RunConfig(mode=mode, scenario=scenario, platforms=platforms, pool=pool, streams=streams,
                     hw_decode=hw_decode, options=Options(hls_segment_secs=args.hls_segment, audio=audio,
                                                          realtime=args.realtime, loop=not args.no_loop),
                     thresholds=thresholds, duration=duration, max_streams=args.max_streams, seed=args.seed)


# --- dashboard -------------------------------------------------------------------------------

def dashboard(engine: Engine, note: str = "") -> Panel:
    src, cfg = engine.source, engine.cfg
    head = Text()
    names = " + ".join(f"{p.name} ({p.device})" for p in cfg.platforms)
    head.append("  Hardware  ", style="dim")
    head.append(names, style="bold cyan")
    head.append("\n  Source    ", style="dim")
    head.append(f"{src.width}×{src.height} {src.codec.upper()} {src.bit_depth}-bit {src.fps:.3f} fps "
                f"{src.bitrate_mbps:.0f} Mbps {src.hdr_type}", style="bold")
    head.append("\n  Target    ", style="dim")
    head.append(f"mixed pool of {len(cfg.pool)}" if cfg.mode == "mixed" else cfg.scenario.label, style="bold green")
    if cfg.mode != "mixed" and engine.streams:
        head.append(f"  ({_output_label(engine.streams[0].info.output)})", style="green")
    head.append(f"   {MODE_LABELS[cfg.mode]}", style="cyan")

    now = time.monotonic()
    threshold = engine.controller.threshold if engine.controller else src.fps * cfg.thresholds.fps_ratio
    table = Table(box=box.SIMPLE, header_style="bold dim", expand=True)
    for col, just in (("#", "right"), ("Target", "left"), ("Platform", "left"), ("Status", "left"), ("fps", "right"),
                      ("× real time", "right"), ("Median", "right"), ("Frames", "right"), ("Mbps out", "right"),
                      ("Enc", "center"), ("Dec", "center")):
        table.add_column(col, justify=just)
    for s in engine.streams[-24:]:
        fps = s.fps(now)
        style = "dim" if fps is None else "bold red" if fps < threshold else "yellow" if fps < threshold * 1.1 else "green"
        status = {"starting": "[yellow]starting[/yellow]", "running": "[green]running[/green]",
                  "error": "[bold red]error[/bold red]", "stopped": "[dim]stopped[/dim]"}.get(s.status, s.status)
        med = s.fps_summary.percentile(0.5)
        table.add_row(str(s.info.id), esc(s.info.scenario_label), s.info.platform_id, status,
                      f"[{style}]{fps:.1f}[/{style}]" if fps is not None else "[dim]–[/dim]",
                      f"{fps / src.fps:.2f}" if fps and src.fps else "–",
                      f"{med:.1f}" if med else "–", f"{s.frames:,}", _mbps_cell(engine.output_mbps(s.info.id), s.info.output),
                      "[green]HW[/green]" if s.info.encode_hw else "[yellow]SW[/yellow]",
                      "[green]HW[/green]" if s.info.decode_hw else "[yellow]SW[/yellow]")
    if len(engine.streams) > 24:
        table.caption = f"showing the newest 24 of {len(engine.streams)} streams"

    status = Text()
    avg_out = engine.avg_output_mbps()
    status.append(f"\n  {len(engine.active)} running   combined {engine.combined_fps():.0f} fps   "
                  + (f"output avg {avg_out:.1f} Mbps/stream   " if avg_out is not None else "")
                  + f"reads ~{engine.read_mbps_estimate():.0f} Mbps   writes {engine.io.last_rate / 1e6:.1f} MB/s   "
                  f"elapsed {engine.elapsed():.0f} s", style="bold")
    c = engine.controller
    if c and not engine.continuous and engine.state == "running":
        bar = 30
        if c.phase == "warmup":
            left = c.warmup_left()
            done = int(bar * (1 - left / max(c.t.warmup_secs, 0.1)))
            status.append(f"\n  warming up  {'█' * done}{'░' * (bar - done)}  {left:.0f} s", style="yellow")
        elif c.phase == "stable":
            left = c.next_add_in()
            done = int(bar * (1 - left / max(c.t.ramp_interval, 0.1)))
            status.append(f"\n  stable      {'█' * done}{'░' * (bar - done)}  +1 stream in {left:.0f} s", style="green")
        status.append(f"\n  threshold {c.threshold:.1f} fps per stream · max stable so far {c.max_stable}"
                      + (" · hardware limit found, adding software streams" if engine.saturated else ""), style="dim")
    events = Text("\n".join(f"  {e.t:6.1f} s  {e.message}" for e in engine.events[-6:]), style="dim")
    footer = Text(f"\n  {note or 'Ctrl+C to stop'}", style="dim")
    return Panel(Group(head, table, status, Text(""), events, footer),
                 title=f"[bold]{APP_TITLE}  v{__version__}[/bold]", border_style="bright_blue")


def _output_label(output: str) -> str:
    """Turn '1920x1080 hevc 8M' into '1920×1080 HEVC 8 Mbps'."""
    parts = output.split()
    if len(parts) != 3:
        return output
    size, codec, rate = parts
    return f"{size.replace('x', '×')} {codec.upper()} {rate.rstrip('M')} Mbps"


def _mbps_cell(measured: Optional[float], output: str) -> str:
    target = output.split()[-1].rstrip("M") if output else ""
    shown = f"{measured:.1f}" if measured is not None else "–"
    return f"{shown} [dim]/ {target}[/dim]" if target else shown


def live_loop(engine: Engine, stop_flag: dict) -> None:
    with Live(dashboard(engine), console=console, refresh_per_second=4, screen=True) as live:
        while engine.state == "running":
            if stop_flag["stop"]:
                engine.finish("user")
                break
            engine.tick()
            live.update(dashboard(engine))
            time.sleep(0.25)


def summary(engine: Engine, data: dict) -> None:
    res = data["result"]
    rule("Result")
    if res["max_stable_streams"] is not None:
        hw_max = res["hw_max_stable_streams"]
        console.print(f"  [bold green]{res['max_stable_streams']}[/bold green] stable concurrent stream(s)"
                      + (f", {hw_max} on hardware" if hw_max is not None else ""))
    else:
        console.print(f"  [bold green]{res['streams_keeping_up']}[/bold green] of {engine.cfg.streams} "
                      "stream(s) kept up with real time")
    if res.get("failure"):
        console.print(f"  [dim]{esc(res['failure']['reason'])}[/dim]")
    console.print(f"  Peak {res['peak_concurrent_streams']} streams, {res['peak_combined_fps']:.0f} fps combined, "
                  f"stopped: {esc(str(res['stop_reason']))}")
    if res.get("avg_output_mbps") is not None and engine.streams:
        console.print(f"  Output {res['avg_output_mbps']:.1f} Mbps per stream on average "
                      f"(target {_output_label(engine.streams[0].info.output)}, video + audio)")
    for w in data["warnings"]:
        console.print(f"  [yellow]⚠ {esc(w)}[/yellow]")


def run_interactive(args: argparse.Namespace) -> int:
    console.print()
    console.rule(f"[bold cyan]{APP_TITLE}  v{__version__}[/bold cyan]")
    engine: Optional[Engine] = None
    work_dir = ""
    stop_flag = {"stop": False}
    previous = signal.getsignal(signal.SIGINT)
    try:
        tools = choose_tools(args.ffmpeg)
        platforms = choose_platforms(tools, args.platform)
        source = choose_source(tools, args.source)
        show_source(source)
        if not source.fps:
            console.print("  [red]The source frame rate is unknown, so there is no real-time threshold.[/red]")
            return 2
        base = choose_work_dir(args.work_dir)
        cfg = setup(source, platforms, tools, args)
        work_dir = make_work_dir(base)
        engine = Engine(cfg, tools, source, work_dir)
        engine.start()
        signal.signal(signal.SIGINT, lambda *_: stop_flag.__setitem__("stop", True))
        while True:
            live_loop(engine, stop_flag)
            if engine.state != "limit":
                break
            console.print(dashboard(engine, note="limit reached"))
            choice = countdown_choice("Limit reached. What next?",
                                      [("c", f"keep {len(engine.active)} stream(s) running until Ctrl+C"),
                                       ("s", "stop and show the result")], default="s")
            if choice == "c":
                engine.continue_running()
            else:
                engine.finish()
                break
        if engine.state != "done":
            engine.finish()
        signal.signal(signal.SIGINT, previous)
        console.print(dashboard(engine, note="finished"))
        data = reports.build(engine)
        summary(engine, data)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if Confirm.ask("\n  Save HTML and JSON reports here?", default=True):
            for name, text in ((f"jf-hw-stress-{stamp}.html", reports.render_html(data)),
                               (f"jf-hw-stress-{stamp}.json", json.dumps(data, indent=2) + "\n")):
                Path(name).write_text(text, encoding="utf-8")
                console.print(f"  [green]✓[/green] {esc(str(Path(name).resolve()))}")
        return 0
    except KeyboardInterrupt:
        console.print("\n  [yellow]Stopped.[/yellow]")
        return 130
    except (tl.ToolError, plat.PlatformError, ProbeError, CommandError, OSError) as exc:
        console.print(f"\n  [red]✗ {esc(str(exc))}[/red]")
        return 2
    finally:
        signal.signal(signal.SIGINT, previous)
        if engine is not None and engine.state != "done":
            engine.finish("error")
        if work_dir and not args.keep_files:
            remove_work_dir(work_dir)
        console.show_cursor(True)
