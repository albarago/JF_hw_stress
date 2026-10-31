"""When to add a stream and when the machine has reached its limit."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass(frozen=True)
class Thresholds:
    fps_ratio: float = 0.90       # each stream must keep this fraction of the source frame rate
    ramp_interval: float = 12.0   # seconds every stream must stay above it before one more is added
    fail_secs: float = 5.0        # seconds below it that count as failure
    warmup_secs: float = 6.0      # seconds after a launch before streams are judged
    start_timeout: float = 30.0   # a stream that produces no frames for this long has failed


@dataclass(frozen=True)
class Reading:
    """What the controller needs from a stream at one moment."""
    id: int
    status: str            # starting | running | error | stopped
    fps: Optional[float]   # live frame rate, None until measurable
    age: float             # seconds since launch
    error: str = ""


@dataclass(frozen=True)
class Decision:
    action: str            # wait | add | fail
    reason: str = ""
    stream_id: Optional[int] = None


class Controller:
    def __init__(self, source_fps: float, thresholds: Thresholds, clock: Callable[[], float] = time.monotonic):
        if source_fps <= 0:
            raise ValueError("the source frame rate is unknown, so there is no real-time threshold")
        self.t = thresholds
        self.threshold = source_fps * thresholds.fps_ratio
        self._clock = clock
        self.max_stable = 0
        self.phase = "warmup"           # warmup | stable | failed
        self._phase_start = clock()
        self._below: Dict[int, float] = {}
        self._seen_errors: set = set()

    def rearm(self) -> None:
        """Start judging again (after a launch or the hybrid switch to software)."""
        self.phase, self._phase_start = "warmup", self._clock()
        self._below.clear()

    def ignore(self, stream_id: int) -> None:
        """A stream the engine stopped on purpose no longer counts."""
        self._below.pop(stream_id, None)
        self._seen_errors.add(stream_id)

    def next_add_in(self) -> float:
        return max(0.0, self.t.ramp_interval - (self._clock() - self._phase_start)) if self.phase == "stable" else 0.0

    def warmup_left(self) -> float:
        return max(0.0, self.t.warmup_secs - (self._clock() - self._phase_start)) if self.phase == "warmup" else 0.0

    def tick(self, readings: List[Reading]) -> Decision:
        now = self._clock()
        for r in readings:
            if r.status == "error" and r.id not in self._seen_errors:
                self._seen_errors.add(r.id)
                return self._fail(f"stream {r.id} stopped with an error: {r.error or 'unknown error'}", r.id)
        live = [r for r in readings if r.status in ("starting", "running")]
        if not live:
            return Decision("wait")
        for r in live:
            if r.fps is None and r.age > self.t.start_timeout:
                return self._fail(f"stream {r.id} produced no frames in {self.t.start_timeout:.0f} s", r.id)

        if self.phase == "warmup":
            if now - self._phase_start < self.t.warmup_secs or any(r.fps is None for r in live):
                return Decision("wait")
            self.phase, self._phase_start = "stable", now
            self._below.clear()
            return Decision("wait")

        for r in live:
            if r.fps is None:
                continue
            if r.fps < self.threshold:
                since = self._below.setdefault(r.id, now)
                if now - since >= self.t.fail_secs:
                    return self._fail(f"stream {r.id} fell to {r.fps:.1f} fps, under {self.threshold:.1f} fps "
                                      f"for {self.t.fail_secs:.0f} s", r.id)
            else:
                self._below.pop(r.id, None)

        if self._below:
            self._phase_start = now     # not all keeping up: the stability interval starts over
            return Decision("wait")
        if now - self._phase_start >= self.t.ramp_interval:
            self.max_stable = max(self.max_stable, len(live))
            self.rearm()
            return Decision("add")
        return Decision("wait")

    def _fail(self, reason: str, stream_id: Optional[int]) -> Decision:
        self.phase = "failed"
        return Decision("fail", reason, stream_id)
