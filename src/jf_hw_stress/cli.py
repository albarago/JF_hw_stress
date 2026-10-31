"""Command line: `jf-hw-stress` (interactive) and `jf-hw-stress --headless --source FILE …`."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from . import APP_TITLE, __version__
from . import platforms as plat
from . import scenarios as scn
from . import tools as tl
from .command import AUDIO_MODES, CommandError, Options
from .controller import Thresholds
from .engine import MODES, RunConfig
from .probe import ProbeError, probe

EPILOG = """\
examples:
  jf-hw-stress                                          interactive setup and live dashboard
  jf-hw-stress --headless --source movie.mkv --json     escalate until the limit, JSON on stdout
  jf-hw-stress --headless --source movie.mkv --mode fixed --streams 6 --duration 120 --html-report r.html
  jf-hw-stress --list-platforms                         what this machine and ffmpeg build support

exit codes: 0 result produced · 1 the run failed (a stream crashed, or none kept up) · 2 setup error · 130 interrupted
"""


def default_work_base() -> str:
    shm = Path("/dev/shm")
    return str(shm) if shm.is_dir() else tempfile.gettempdir()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jf-hw-stress", description=APP_TITLE, epilog=EPILOG,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--headless", action="store_true", help="run without prompts (needs --source)")
    p.add_argument("--source", metavar="FILE", help="video file to transcode")
    p.add_argument("--ffmpeg", "--ffmpeg-dir", dest="ffmpeg", metavar="PATH",
                   help="ffmpeg binary, or a directory with ffmpeg and ffprobe (default: Jellyfin's build, then PATH)")
    p.add_argument("--platform", "--force-platform", dest="platform", metavar="ID[,ID]",
                   help=f"hardware backend(s) to use: {', '.join(plat.BY_ID)} (default: auto-detect); "
                        "several are used round-robin")
    p.add_argument("--mode", choices=MODES, default="escalating", help="test mode (default: escalating)")
    p.add_argument("--streams", type=int, default=4, help="concurrent streams in fixed mode (default: 4)")
    p.add_argument("--scenario", default="1080p_compat", help="scenario id (default: 1080p_compat); see --list-scenarios")
    p.add_argument("--no-hw-decode", action="store_true", help="decode in software, encode on the hardware")
    p.add_argument("--audio", choices=AUDIO_MODES, default="aac",
                   help="aac: re-encode to stereo AAC like most clients (default) · copy · none")
    p.add_argument("--realtime", action="store_true", help="read the source at playback speed (-re)")
    p.add_argument("--no-loop", action="store_true", help="do not loop the source (streams end with the file)")
    p.add_argument("--duration", type=float, metavar="SECS", help="stop after this many seconds (default: at the limit)")
    p.add_argument("--fps-ratio", type=float, default=0.90,
                   help="fraction of the source frame rate each stream must keep (default: 0.90)")
    p.add_argument("--ramp-interval", type=float, default=12.0, help="seconds stable before adding a stream (default: 12)")
    p.add_argument("--fail-secs", type=float, default=5.0, help="seconds below the threshold that count as failure (default: 5)")
    p.add_argument("--warmup-secs", type=float, default=6.0, help="seconds after a launch before judging (default: 6)")
    p.add_argument("--max-streams", type=int, default=64, help="never run more than this many streams (default: 64)")
    p.add_argument("--hls-segment", type=int, default=4, metavar="SECS",
                   help="HLS segment length (default: 4; Jellyfin uses 3–6)")
    p.add_argument("--seed", type=int, help="random seed for mixed mode, to repeat a run")
    p.add_argument("--work-dir", "--cache-dir", dest="work_dir", metavar="DIR",
                   help="where segments are written; a run folder is created inside and removed afterwards "
                        f"(default: {default_work_base()})")
    p.add_argument("--keep-files", action="store_true", help="keep the run folder (segments, playlists, ffmpeg logs)")
    p.add_argument("--json", action="store_true", help="print the JSON report to stdout")
    p.add_argument("--json-file", metavar="FILE", help="write the JSON report to FILE")
    p.add_argument("--html-report", metavar="FILE", help="write the HTML report to FILE")
    p.add_argument("--list-scenarios", action="store_true", help="list scenarios and exit")
    p.add_argument("--list-platforms", action="store_true", help="show which backends this machine and ffmpeg can use, and exit")
    return p


def validate(args: argparse.Namespace) -> List[str]:
    errors = []
    if not 0.5 <= args.fps_ratio <= 1.0:
        errors.append("--fps-ratio must be between 0.5 and 1.0")
    for name in ("ramp_interval", "fail_secs", "warmup_secs"):
        if getattr(args, name) <= 0:
            errors.append(f"--{name.replace('_', '-')} must be positive")
    if args.streams < 1 or args.max_streams < 1:
        errors.append("--streams and --max-streams must be at least 1")
    if args.duration is not None and args.duration <= 0:
        errors.append("--duration must be positive")
    if not 1 <= args.hls_segment <= 30:
        errors.append("--hls-segment must be between 1 and 30")
    return errors


def find_tools(path: Optional[str]) -> tl.Tools:
    if path:
        return tl.load(tl.from_path(path))
    found = tl.candidates()
    if not found:
        raise tl.ToolError("ffmpeg and ffprobe were not found; install them or pass --ffmpeg PATH")
    return tl.load(found[0])


def list_scenarios() -> None:
    for n, (group, items) in enumerate((("Scenarios", scn.SCENARIOS), ("Mixed-mode pool", scn.MIXED))):
        print(f"{chr(10) if n else ''}{group}:")
        print(f"{'id':<18} {'label':<22} {'codec':<5} {'width':>5} {'bitrate':>8}  tone map")
        for s in items:
            print(f"{s.id:<18} {s.label:<22} {s.codec:<5} {s.width:>5} {s.bitrate:>8}  {'yes' if s.tonemap else '-'}")


def list_platforms(tools: tl.Tools) -> None:
    print(f"ffmpeg {tools.version} ({tools.origin})\n  {tools.ffmpeg}\n")
    for p, reason in plat.describe_support(tools.caps):
        state = "available" if reason is None else reason
        note = "" if p.tested else "  (untested backend)"
        tonemap = ""
        if reason is None and p.is_hardware:
            tonemap = "  HDR tone mapping: " + ("hardware" if p.hw_tonemap(tools.caps) else
                                                "software (zscale)" if tools.caps.has_filter("zscale") else "unavailable")
        print(f"  {p.id:<6} {p.name:<28} {state}{tonemap}{note}")


def headless(args: argparse.Namespace) -> int:
    from . import headless as hl
    if not args.source:
        hl.say("--headless needs --source FILE")
        return hl.EXIT_SETUP
    source_path = Path(args.source).expanduser()
    if not source_path.is_file():
        hl.say(f"source not found: {source_path}")
        return hl.EXIT_SETUP
    try:
        tools = find_tools(args.ffmpeg)
        hl.say(f"ffmpeg {tools.version} from {tools.origin}")
        if args.platform:
            chosen = plat.choose([x.strip() for x in args.platform.split(",") if x.strip()], tools.caps)
        else:
            chosen = plat.detect(tools.caps)
        warnings = []
        hw = [p for p in chosen if p.is_hardware]
        if not args.platform and len(hw) > 1:
            warnings.append(f"several hardware backends found ({', '.join(p.id for p in hw)}); using {hw[0].id}. "
                            "Pass --platform to choose, or a comma list to use several.")
            chosen = hw[:1]
        if not hw:
            warnings.append("no hardware backend is usable with this ffmpeg build; running software encodes")
        for p in chosen:
            if p.is_hardware and not p.tested:
                warnings.append(f"{p.name} support has not been verified on real hardware yet")
        for w in warnings:
            hl.say("warning: " + w)
        hl.say("platform: " + ", ".join(f"{p.name} ({p.device})" for p in chosen))
        source = probe(tools.ffprobe, str(source_path))
        hl.say(f"source: {source.width}x{source.height} {source.codec} {source.bit_depth}-bit {source.fps} fps "
               f"{source.bitrate_mbps} Mbps {source.hdr_type}, audio {source.audio_label()}")
        if not source.fps:
            hl.say("the source frame rate is unknown, so there is no real-time threshold")
            return hl.EXIT_SETUP
        if args.mode == "mixed":
            pool = scn.mixed_pool(source)
            if not pool:
                hl.say("no mixed-mode scenario fits this source")
                return hl.EXIT_SETUP
            scenario = pool[0]
        else:
            scenario = scn.by_id(args.scenario)
            if scenario is None:
                hl.say(f"unknown scenario {args.scenario!r}; see --list-scenarios")
                return hl.EXIT_SETUP
            ok, why = scn.applicable(scenario, source)
            if not ok:
                hl.say(f"scenario {scenario.id} does not fit this source: {why}")
                return hl.EXIT_SETUP
            pool = []
        cfg = RunConfig(
            mode=args.mode, scenario=scenario, platforms=chosen, pool=pool, streams=args.streams,
            hw_decode=not args.no_hw_decode,
            options=Options(hls_segment_secs=args.hls_segment, audio=args.audio, loop=not args.no_loop,
                            realtime=args.realtime),
            thresholds=Thresholds(fps_ratio=args.fps_ratio, ramp_interval=args.ramp_interval,
                                  fail_secs=args.fail_secs, warmup_secs=args.warmup_secs),
            duration=args.duration, max_streams=args.max_streams, seed=args.seed)
        return hl.run(cfg, tools, source, args.work_dir or default_work_base(), keep_files=args.keep_files,
                      json_stdout=args.json, json_file=args.json_file, html_file=args.html_report, warnings=warnings)
    except (tl.ToolError, plat.PlatformError, ProbeError, CommandError, ValueError, OSError) as exc:
        hl.say(f"error: {exc}")
        return hl.EXIT_SETUP


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    problems = validate(args)
    if problems:
        for problem in problems:
            print(f"jf-hw-stress: {problem}", file=sys.stderr)
        return 2
    if args.list_scenarios:
        list_scenarios()
        return 0
    if args.list_platforms:
        try:
            list_platforms(find_tools(args.ffmpeg))
        except tl.ToolError as exc:
            print(f"jf-hw-stress: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.headless:
        return headless(args)
    try:
        from .tui import run_interactive
    except ImportError as exc:  # rich missing
        print(f"jf-hw-stress: the interactive mode needs the 'rich' package ({exc}); "
              "install it, or use --headless", file=sys.stderr)
        return 2
    return run_interactive(args)
