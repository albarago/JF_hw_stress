"""Non-interactive runs for scripts, CI and Kubernetes jobs: progress on stderr, results as files or stdout."""
from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

from . import report as reports
from .engine import Engine, RunConfig, make_work_dir, remove_work_dir
from .probe import SourceInfo
from .tools import Tools

EXIT_OK, EXIT_RESULT, EXIT_SETUP, EXIT_INTERRUPTED = 0, 1, 2, 130
STATUS_EVERY = 10.0


def say(message: str) -> None:
    print(f"[jf-hw-stress] {message}", file=sys.stderr, flush=True)


def run(cfg: RunConfig, tools: Tools, source: SourceInfo, work_base: str, keep_files: bool = False,
        json_stdout: bool = False, json_file: Optional[str] = None, html_file: Optional[str] = None,
        warnings: Optional[List[str]] = None) -> int:
    work_dir = make_work_dir(work_base)
    engine = Engine(cfg, tools, source, work_dir)
    interrupted = {"flag": False}

    def on_signal(signum, frame):
        interrupted["flag"] = True
    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    shown = 0   # events already printed
    try:
        engine.start()
        say(f"work directory {work_dir}")
        next_status = time.monotonic() + STATUS_EVERY
        while engine.state == "running":
            if interrupted["flag"]:
                engine.finish("user")
                break
            engine.tick()
            for event in engine.events[shown:]:
                say(f"{event.t:6.1f} s  {event.message}")
            shown = len(engine.events)
            if engine.state == "limit":
                engine.finish()
                break
            if time.monotonic() >= next_status:
                fps = [f for f in (s.fps() for s in engine.active) if f is not None]
                avg_out = engine.avg_output_mbps()
                say(f"{engine.elapsed():6.0f} s  {len(engine.active)} stream(s)  combined {sum(fps):.0f} fps"
                    + (f"  slowest {min(fps):.1f} fps" if fps else "")
                    + (f"  output {avg_out:.1f} Mbps/stream" if avg_out is not None else "")
                    + (f"  max stable {engine.max_stable()}" if engine.controller else ""))
                next_status += STATUS_EVERY
            time.sleep(0.2)
        if engine.state != "done":
            engine.finish()
    finally:
        if engine.state != "done":
            engine.finish("error")
        for s, handler in previous.items():
            signal.signal(s, handler)
        if not keep_files:
            remove_work_dir(work_dir)

    data = reports.build(engine, warnings)
    for event in engine.events[shown:]:
        say(f"{event.t:6.1f} s  {event.message}")
    res = data["result"]
    if res["max_stable_streams"] is not None:
        say(f"result: {res['max_stable_streams']} stable stream(s)"
            + (f", {res['hw_max_stable_streams']} on hardware" if res["hw_max_stable_streams"] is not None else "")
            + f" (stopped: {res['stop_reason']})")
    else:
        say(f"result: {res['streams_keeping_up']} of {cfg.streams} stream(s) kept up with real time")
    if res.get("avg_output_mbps") is not None:
        say(f"output: {res['avg_output_mbps']:.1f} Mbps per stream on average "
            f"(target {cfg.scenario.bitrate_mbps} Mbps video, plus audio)")
    text = json.dumps(data, indent=2)
    if json_file:
        Path(json_file).write_text(text + "\n", encoding="utf-8")
        say(f"JSON report: {json_file}")
    if html_file:
        Path(html_file).write_text(reports.render_html(data), encoding="utf-8")
        say(f"HTML report: {html_file}")
    if json_stdout:
        print(text)
    if interrupted["flag"]:
        return EXIT_INTERRUPTED
    return exit_code(data)


def exit_code(data: dict) -> int:
    """0: a valid result. 1: the run itself failed (a stream crashed or stalled, or not even one kept up)."""
    res = data["result"]
    crashed = any(s["status"] == "error" for s in data["streams"])
    if res["max_stable_streams"] is not None:   # escalating modes
        if res["max_stable_streams"] == 0 or (res.get("failure") or {}).get("kind") in ("error", "stall"):
            return EXIT_RESULT
        return EXIT_OK
    return EXIT_RESULT if crashed else EXIT_OK
