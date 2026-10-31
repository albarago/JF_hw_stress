"""The test itself, shared by the interactive dashboard and headless mode.

The caller drives it: `start()`, then `tick()` a few times a second until `state` is "limit" or "done".
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import command
from .controller import Controller, Reading, Thresholds
from .platforms import SOFTWARE, Platform
from .probe import SourceInfo
from .scenarios import Scenario
from .stream import Stream, StreamInfo
from .tools import Tools

MODES = ("fixed", "escalating", "hybrid", "mixed")
MODE_LABELS = {"fixed": "Fixed", "escalating": "Escalating", "hybrid": "Hybrid HW → SW", "mixed": "Mixed"}
STALL_RETRIES = 1     # a stream that never produces a frame is replaced this many times before the run fails


@dataclass
class RunConfig:
    mode: str
    scenario: Scenario                        # the scenario (fixed/escalating/hybrid), or the first draw (mixed)
    platforms: List[Platform]                 # streams are spread round-robin over these
    pool: List[Scenario] = field(default_factory=list)   # mixed mode only
    streams: int = 4                          # fixed mode only
    hw_decode: bool = True
    options: command.Options = field(default_factory=command.Options)
    thresholds: Thresholds = field(default_factory=Thresholds)
    duration: Optional[float] = None          # seconds; None = until the limit or a stop
    max_streams: int = 64
    seed: Optional[int] = None


class IoMeter:
    """Bytes the transcodes write as HLS segments, and the source read rate they imply."""
    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self._sizes: Dict[str, int] = {}
        self.total_bytes = 0
        self.last_rate = 0.0
        self.peak_rate = 0.0
        self._last: Optional[float] = None

    def sample(self, now: float) -> None:
        delta, sizes = 0, {}
        try:
            for entry in os.scandir(self.directory):
                if entry.name.endswith(".ts"):
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        continue
                    sizes[entry.name] = size
                    delta += max(0, size - self._sizes.get(entry.name, 0))
        except OSError:
            return
        self._sizes = sizes
        self.total_bytes += delta
        if self._last is not None and now > self._last:
            self.last_rate = delta / (now - self._last)
            self.peak_rate = max(self.peak_rate, self.last_rate)
        self._last = now


class SegmentMeter:
    """A stream's average output bitrate: bytes of its finished HLS segments over their playlist durations.

    ffmpeg reports no size or bitrate for HLS output, and streams run faster than real time, so
    bytes per wall-clock second would overstate it. Segments are counted once, by sequence number;
    one that was deleted before a scan is skipped, which leaves the average unbiased.
    """
    def __init__(self, playlist: str) -> None:
        self.playlist = Path(playlist)
        self.bytes = 0
        self.secs = 0.0
        self.segments = 0
        self._last_seq = -1

    def scan(self) -> None:
        try:
            lines = self.playlist.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return
        seq, duration = 0, None
        for line in (ln.strip() for ln in lines):
            try:
                if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                    seq = int(line.split(":", 1)[1])
                elif line.startswith("#EXTINF:"):
                    duration = float(line.split(":", 1)[1].split(",")[0])
                elif line and not line.startswith("#"):
                    if duration and seq > self._last_seq:
                        self._last_seq = seq
                        try:
                            size = (self.playlist.parent / line).stat().st_size
                        except OSError:
                            size = 0
                        if size:
                            self.bytes += size
                            self.secs += duration
                            self.segments += 1
                    seq, duration = seq + 1, None
            except ValueError:
                return   # a half-written playlist: try again next time

    @property
    def mbps(self) -> Optional[float]:
        return self.bytes * 8 / self.secs / 1e6 if self.secs >= 1.0 else None


@dataclass
class Event:
    t: float
    message: str


class Engine:
    def __init__(self, config: RunConfig, tools: Tools, source: SourceInfo, work_dir: str,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 stream_factory: Callable[..., Stream] = Stream) -> None:
        if config.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        self.cfg = config
        self.tools = tools
        self.source = source
        self.work_dir = work_dir
        self._clock, self._wall, self._factory = clock, wall, stream_factory
        self._rng = random.Random(config.seed)
        self.streams: List[Stream] = []
        self.output: Dict[int, SegmentMeter] = {}   # stream id → output bitrate meter
        self.events: List[Event] = []
        self.timeline: List[Dict] = []
        self._timeline_stride, self._next_sample = 1, 0.0
        self.io = IoMeter(work_dir)
        self.state = "idle"              # idle | running | limit | done
        self.stop_reason = ""
        self.failure: Optional[Dict] = None
        self.continuous = False
        self.saturated = False           # hybrid: hardware limit found, now adding software streams
        self.hw_max_stable: Optional[int] = None
        self.peak_concurrent = 0
        self.peak_combined_fps = 0.0
        self.warnings: List[str] = []
        self.started_at = 0.0
        self.ended_at: Optional[float] = None
        self._start_mono = 0.0
        self._hw_platforms = [p for p in config.platforms if p.is_hardware] or [SOFTWARE]
        self._rr = 0
        self._stall_retries = 0
        self.controller: Optional[Controller] = None
        if config.mode != "fixed":
            self.controller = Controller(source.fps, config.thresholds, clock)

    # --- helpers ------------------------------------------------------------------------------
    def elapsed(self) -> float:
        return self._clock() - self._start_mono if self._start_mono else 0.0

    def log(self, message: str) -> None:
        self.events.append(Event(round(self.elapsed(), 1), message))

    @property
    def active(self) -> List[Stream]:
        return [s for s in self.streams if s.active]

    def _pick_scenario(self) -> Scenario:
        if self.cfg.mode == "mixed" and self.cfg.pool:
            return self._rng.choice(self.cfg.pool)
        return self.cfg.scenario

    def plan(self, scenario: Scenario, platform: Platform, software: bool, stream_id: int = 0) -> command.Plan:
        return command.build(platform, scenario, self.source, self.tools.caps, self.tools.ffmpeg, stream_id,
                             self.work_dir, self.cfg.options,
                             hw_decode=self.cfg.hw_decode and not software, hw_encode=not software)

    def check(self) -> None:
        """Build every command this run can use before starting anything, so errors come first."""
        scenarios = self.cfg.pool if self.cfg.mode == "mixed" else [self.cfg.scenario]
        for scenario in scenarios:
            for platform in self._hw_platforms:
                self.plan(scenario, platform, software=not platform.is_hardware)
            if self.cfg.mode == "hybrid":
                self.plan(scenario, SOFTWARE, software=True)

    def launch(self, software: bool = False) -> Stream:
        sid = len(self.streams) + 1
        scenario = self._pick_scenario()
        if software:
            platform = SOFTWARE
        else:
            platform = self._hw_platforms[self._rr % len(self._hw_platforms)]
            self._rr += 1
        plan = self.plan(scenario, platform, software or not platform.is_hardware, sid)
        info = StreamInfo(id=sid, scenario_id=scenario.id, scenario_label=scenario.label, platform_id=platform.id,
                          platform_name=platform.name, encoder=plan.encoder, encode_hw=plan.encode_hw,
                          decode_hw=plan.decode_hw, tonemap=plan.tonemap, output=plan.output)
        stream = self._factory(info, plan.argv, os.path.join(self.work_dir, f"stream{sid}.log"), self.source.fps,
                               clock=self._clock, wall=self._wall)
        self.streams.append(stream)
        self.output[sid] = SegmentMeter(plan.playlist)
        stream.start()
        self.peak_concurrent = max(self.peak_concurrent, len(self.active))
        tag = "HW" if plan.encode_hw else "SW"
        self.log(f"stream {sid} started: {scenario.label} [{tag} {platform.id}] ({len(self.active)} running)")
        return stream

    # --- run ----------------------------------------------------------------------------------
    def start(self) -> None:
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        self.check()
        self.started_at, self._start_mono = self._wall(), self._clock()
        self.state = "running"
        first = self.cfg.streams if self.cfg.mode == "fixed" else 1
        for _ in range(max(1, first)):
            self.launch()
        if self.controller:
            self.log(f"threshold {self.controller.threshold:.1f} fps "
                     f"({self.cfg.thresholds.fps_ratio:.0%} of {self.source.fps:.3f} fps)")

    def tick(self) -> str:
        if self.state != "running":
            return self.state
        now = self._clock()
        for s in self.streams:
            s.poll()
        if now >= self._next_sample:
            self._sample(now)
            self._next_sample = now + 1.0
        if self.cfg.duration is not None and self.elapsed() >= self.cfg.duration:
            self.finish("duration")
            return self.state
        if not self.active:
            self.finish("all-streams-ended")
            return self.state
        if self.controller and not self.continuous:
            self._escalate()
        return self.state

    def _escalate(self) -> None:
        assert self.controller is not None
        decision = self.controller.tick(self.readings())
        if decision.action == "add":
            if len(self.active) >= self.cfg.max_streams:
                self.log(f"stream cap of {self.cfg.max_streams} reached while still stable")
                self._limit("max-streams")
                return
            self.launch(software=self.saturated)
        elif decision.action == "fail":
            self.log(decision.reason)
            failed = self._by_id(decision.stream_id)
            if decision.kind == "stall" and failed is not None and self._stall_retries < STALL_RETRIES:
                # A stream that never starts says nothing about capacity (seen once on VideoToolbox
                # with 6 streams running fine): replace it and keep going.
                self._stall_retries += 1
                self.controller.ignore(failed.info.id)
                failed.stop()
                self.warnings.append(f"stream {failed.info.id} produced no frames and was replaced")
                self.log(f"stream {failed.info.id} stopped; starting a replacement")
                self.controller.rearm()
                self.launch(software=not failed.info.encode_hw)
                return
            if decision.kind == "capacity":
                self._shed_newest()   # back off to the last load that kept up
                if self.cfg.mode == "hybrid" and not self.saturated:
                    self.saturated = True
                    self.hw_max_stable = self.controller.max_stable
                    self.log(f"hardware limit: {self.hw_max_stable} stream(s); adding software streams")
                    self.controller.rearm()
                    self.launch(software=True)
                    return
            elif failed is not None and failed.active:
                self.controller.ignore(failed.info.id)
                failed.stop()
            self.failure = {"stream": decision.stream_id, "reason": decision.reason, "kind": decision.kind}
            self._limit("limit-reached" if decision.kind == "capacity" else "stream-failed")

    def _by_id(self, sid: Optional[int]) -> Optional[Stream]:
        return next((s for s in self.streams if s.info.id == sid), None)

    def _shed_newest(self) -> None:
        live = self.active
        if len(live) > 1:
            newest = max(live, key=lambda s: s.info.id)
            assert self.controller is not None
            self.controller.ignore(newest.info.id)
            newest.stop()
            self.log(f"stream {newest.info.id} stopped to return to the last stable load")

    def _limit(self, reason: str) -> None:
        self.stop_reason = reason
        self.state = "limit"
        if self.controller:
            self.log(f"max stable: {self.controller.max_stable} stream(s)")

    def continue_running(self) -> None:
        """After the limit: keep the current load running until stopped or the duration ends."""
        self.continuous = True
        self.state = "running"
        self.log(f"continuing at {len(self.active)} stream(s)")

    def finish(self, reason: str = "") -> None:
        if self.state == "done":
            return
        if reason and (not self.stop_reason or self.state == "running"):
            self.stop_reason = reason
        self._sample(self._clock())
        for s in self.streams:
            s.signal_stop()     # all at once, so shutdown takes one grace period, not one per stream
        for s in self.streams:
            s.stop()
        for s in self.streams:
            s.poll()
        for meter in self.output.values():
            meter.scan()        # each ffmpeg closes its last segment on the way out
        self.ended_at = self._wall()
        self.state = "done"
        self.log(f"finished: {self.stop_reason or reason}")

    # --- readings -----------------------------------------------------------------------------
    def readings(self) -> List[Reading]:
        now = self._clock()
        return [Reading(s.info.id, s.status, s.fps(now), s.age(), s.error) for s in self.streams]

    def combined_fps(self) -> float:
        now = self._clock()
        return sum(f for f in (s.fps(now) for s in self.active) if f)

    def read_mbps_estimate(self) -> float:
        """Source megabits per second the running streams must read (speed × file bitrate)."""
        now = self._clock()
        return sum(sp for sp in (s.speed(now) for s in self.active) if sp) * self.source.bitrate_mbps

    def output_mbps(self, stream_id: int) -> Optional[float]:
        meter = self.output.get(stream_id)
        return meter.mbps if meter else None

    def avg_output_mbps(self) -> Optional[float]:
        """All streams' segments together; None in mixed mode, where targets differ per stream."""
        if self.cfg.mode == "mixed":
            return None
        secs = sum(m.secs for m in self.output.values())
        return sum(m.bytes for m in self.output.values()) * 8 / secs / 1e6 if secs >= 1.0 else None

    def _sample(self, now: float) -> None:
        self.io.sample(now)
        for s in self.active:
            meter = self.output.get(s.info.id)
            if meter:
                meter.scan()
        fps_values = [f for f in (s.fps(now) for s in self.active) if f is not None]
        combined = sum(fps_values)
        self.peak_combined_fps = max(self.peak_combined_fps, combined)
        if not self._start_mono:
            return
        self.timeline.append({"t": round(now - self._start_mono, 1), "streams": len(self.active),
                              "combined_fps": round(combined, 1),
                              "min_fps": round(min(fps_values), 1) if fps_values else None,
                              "read_mbps": round(self.read_mbps_estimate(), 1)})
        if len(self.timeline) > 3600:   # keep long runs bounded: halve the resolution
            self.timeline = self.timeline[::2]
            self._timeline_stride *= 2

    def max_stable(self) -> Optional[int]:
        return self.controller.max_stable if self.controller else None


def make_work_dir(base: str) -> str:
    """A fresh directory for this run inside `base`; nothing else in `base` is touched."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(base).expanduser() / f"jf-hw-stress-{stamp}-{os.getpid()}"
    path.mkdir(parents=True, exist_ok=False)
    return str(path)


def remove_work_dir(path: str) -> Tuple[int, int]:
    """Delete the run directory's files (only those this tool writes) and the directory itself."""
    removed = left = 0
    p = Path(path)
    if not p.is_dir():
        return 0, 0
    for entry in p.iterdir():
        if entry.is_file() and entry.name.startswith("stream") and entry.suffix in (".ts", ".m3u8", ".log", ".tmp"):
            try:
                entry.unlink()
                removed += 1
            except OSError:
                left += 1
        else:
            left += 1
    if not left:
        try:
            p.rmdir()
        except OSError:
            left += 1
    return removed, left
