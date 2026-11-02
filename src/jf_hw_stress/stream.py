"""One running ffmpeg transcode and its live statistics.

ffmpeg writes `-progress` key=value blocks to stdout; a reader thread consumes them as they arrive,
so nothing grows on disk and parsing cost stays constant however long a test runs. ffmpeg's own
`fps=` is an average since the stream started, so the live frame rate is computed here from frame
counts over a short window.
"""
from __future__ import annotations

import collections
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

WINDOW_SECS = 3.0          # live fps is measured over this many seconds
HISTORY = 7200             # one-per-second fps samples kept for percentiles (two hours)
STOP_GRACE = 3.0


class ProgressParser:
    """Turns `-progress` lines into one dict per block (each block ends with progress=…)."""
    def __init__(self) -> None:
        self._block: Dict[str, str] = {}

    def feed(self, line: str) -> Optional[Dict[str, str]]:
        key, sep, value = line.strip().partition("=")
        if not sep:
            return None
        key, value = key.strip(), value.strip()
        if key == "progress":
            block, self._block = self._block, {}
            block["progress"] = value
            return block
        self._block[key] = value
        return None


def _int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value not in (None, "", "N/A") else None
    except ValueError:
        return None


class RateWindow:
    """Frames per second over the last `span` seconds of (time, frame) points."""
    def __init__(self, span: float = WINDOW_SECS) -> None:
        self.span = span
        self._points: Deque[Tuple[float, int]] = collections.deque()

    def add(self, t: float, frame: int) -> None:
        if self._points and frame < self._points[-1][1]:
            self._points.clear()  # counter went backwards: start over rather than report a negative rate
        self._points.append((t, frame))
        while len(self._points) > 2 and t - self._points[1][0] >= self.span:
            self._points.popleft()

    def rate(self, now: float) -> Optional[float]:
        """None until the window covers at least half its span, or when data has stopped arriving."""
        if len(self._points) < 2:
            return None
        (t0, f0), (t1, f1) = self._points[0], self._points[-1]
        if t1 - t0 < self.span / 2 or now - t1 > max(self.span, 2.0):
            return None
        return (f1 - f0) / (t1 - t0)


@dataclass
class FpsSummary:
    count: int = 0
    total: float = 0.0
    low: Optional[float] = None
    high: Optional[float] = None
    recent: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=HISTORY))

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.low = value if self.low is None else min(self.low, value)
        self.high = value if self.high is None else max(self.high, value)
        self.recent.append(value)

    def percentile(self, q: float) -> Optional[float]:
        if not self.recent:
            return None
        values = sorted(self.recent)
        return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]

    def to_dict(self) -> Dict[str, Optional[float]]:
        r = lambda v: None if v is None else round(v, 2)
        return {"samples": self.count, "min": r(self.low), "p5": r(self.percentile(0.05)),
                "p50": r(self.percentile(0.5)), "avg": r(self.total / self.count) if self.count else None,
                "max": r(self.high)}


@dataclass
class StreamInfo:
    """Fixed facts about a stream, set at launch."""
    id: int
    scenario_id: str
    scenario_label: str
    platform_id: str
    platform_name: str
    encoder: str
    encode_hw: bool
    decode_hw: bool
    tonemap: str
    output: str
    burn_in: str = ""            # how subtitles are drawn into this stream, "" = none


class Stream:
    def __init__(self, info: StreamInfo, argv: List[str], log_path: str, source_fps: float,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 popen: Callable[..., subprocess.Popen] = subprocess.Popen) -> None:
        self.info = info
        self.argv = argv
        self.log_path = log_path
        self.source_fps = source_fps
        self._clock, self._wall, self._popen = clock, wall, popen
        self._lock = threading.Lock()
        self._window = RateWindow()
        self.fps_summary = FpsSummary()
        self._next_sample = 0.0
        self.status = "starting"     # starting | running | error | stopped
        self.frames = 0
        self.started_at = 0.0        # wall clock
        self.started_mono = 0.0
        self.ended_at: Optional[float] = None
        self.error = ""
        self.exit_code: Optional[int] = None
        self._stopping = False
        self._proc: Optional[subprocess.Popen] = None
        self._log = None
        self._reader: Optional[threading.Thread] = None
        # A looped source is a finite list of copies; when ffmpeg gets to the end of it (exit 0), the same
        # command starts again and the frame count carries on, so the load stays the same.
        self.restart_on_end = False
        self.restarts = 0
        self._frame_base = 0

    # --- lifecycle ----------------------------------------------------------------------------
    def start(self) -> None:
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.started_at, self.started_mono = self._wall(), self._clock()
        self._spawn()

    def _spawn(self) -> None:
        kwargs = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True     # Ctrl+C reaches us, not every ffmpeg directly
        else:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            self._proc = self._popen(self.argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=self._log,
                                     text=True, errors="replace", bufsize=1, **kwargs)
        except OSError as exc:
            self._fail(f"could not start ffmpeg: {exc}")
            return
        self._reader = threading.Thread(target=self._read, name=f"stream-{self.info.id}", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        parser = ProgressParser()
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        base = self._frame_base
        for line in proc.stdout:
            block = parser.feed(line)
            if block is None:
                continue
            frame = _int(block.get("frame"))
            if frame is None:
                continue
            frame += base
            now = self._clock()
            with self._lock:
                self.frames = frame
                self._window.add(now, frame)
                if frame > 0 and self.status == "starting":
                    self.status = "running"

    def poll(self) -> None:
        """Update status from the process and take the once-per-second fps sample."""
        now = self._clock()
        with self._lock:
            if self._proc is not None and self.status in ("starting", "running"):
                code = self._proc.poll()
                if code == 0 and self.restart_on_end and not self._stopping and self.frames > self._frame_base:
                    self._restart()
                    code = None
                if code is not None:
                    self.exit_code = code
                    self.ended_at = self._wall()
                    if self._stopping:
                        self.status = "stopped"
                    else:
                        self.status = "error"
                        self.error = self._tail_error() or f"ffmpeg exited with code {code}"
            if self.status == "running" and now >= self._next_sample:
                rate = self._window.rate(now)
                if rate is not None:
                    self.fps_summary.add(rate)
                self._next_sample = now + 1.0

    def _restart(self) -> None:
        """ffmpeg finished the looped source: run the same command again (called with the lock held)."""
        if self._reader is not None:
            self._lock.release()
            try:
                self._reader.join(timeout=STOP_GRACE)   # the last progress lines are counted first
            finally:
                self._lock.acquire()
        self._frame_base = self.frames
        self.restarts += 1
        if self._log is not None:
            self._log.write(f"\n[jf-hw-stress] end of the looped source; restart {self.restarts}\n")
            self._log.flush()
        self._spawn()

    def signal_stop(self) -> None:
        """Ask ffmpeg to finish (it flushes and exits); `stop()` then waits for it.

        Only one signal is sent: ffmpeg treats a second one as "exit immediately" and leaves a
        truncated playlist and empty segments behind.
        """
        if self._stopping:
            return
        self._stopping = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                if os.name == "posix":
                    proc.send_signal(signal.SIGTERM)
                else:
                    proc.terminate()
            except OSError:
                pass

    def stop(self) -> None:
        self.signal_stop()
        proc = self._proc
        if proc is None or proc.poll() is not None:
            self._close()
            with self._lock:
                if self.status in ("starting", "running"):
                    self.status, self.ended_at = "stopped", self._wall()
            return
        try:
            proc.wait(timeout=STOP_GRACE)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=STOP_GRACE)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass
        self._close()
        with self._lock:
            if self.status in ("starting", "running"):
                self.status = "stopped"
                self.ended_at = self._wall()

    def _close(self) -> None:
        if self._reader is not None:
            self._reader.join(timeout=STOP_GRACE)
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None

    def _fail(self, message: str) -> None:
        self.status, self.error, self.ended_at = "error", message, self._wall()

    def _tail_error(self) -> str:
        try:
            if self._log is not None:
                self._log.flush()
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                lines = [ln.strip() for ln in fh.readlines()[-6:] if ln.strip()]
            return " | ".join(lines)[-500:]
        except OSError:
            return ""

    # --- readings -----------------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.status in ("starting", "running")

    def fps(self, now: Optional[float] = None) -> Optional[float]:
        with self._lock:
            return self._window.rate(self._clock() if now is None else now) if self.status == "running" else None

    def speed(self, now: Optional[float] = None) -> Optional[float]:
        rate = self.fps(now)
        return None if rate is None or not self.source_fps else rate / self.source_fps

    def age(self) -> float:
        return self._clock() - self.started_mono if self.started_mono else 0.0
