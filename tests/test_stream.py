import sys
import time

from jf_hw_stress.stream import FpsSummary, ProgressParser, RateWindow, Stream, StreamInfo

INFO = StreamInfo(1, "1080p_compat", "1080p", "sw", "Software", "libx264", False, False, "", "1920x1080 h264 8M")


def test_progress_blocks():
    parser = ProgressParser()
    lines = ["frame=10", "fps=0.00", "speed=N/A", "progress=continue", "frame=20", "progress=end"]
    blocks = [b for b in (parser.feed(line) for line in lines) if b]
    assert blocks == [{"frame": "10", "fps": "0.00", "speed": "N/A", "progress": "continue"},
                      {"frame": "20", "progress": "end"}]
    assert parser.feed("garbage") is None


def test_rate_window_is_live_not_cumulative():
    w = RateWindow(span=3)
    for t in range(0, 11):        # 100 fps for 10 s …
        w.add(float(t), t * 100)
    for t in range(11, 15):       # … then 20 fps
        w.add(float(t), 1000 + (t - 10) * 20)
    assert w.rate(14.0) == 20.0   # ffmpeg's own average would still say about 74 fps


def test_rate_window_needs_data_and_freshness():
    w = RateWindow(span=3)
    w.add(0.0, 0)
    assert w.rate(0.5) is None
    w.add(2.0, 50)
    assert w.rate(2.0) == 25.0
    assert w.rate(10.0) is None   # no progress for 8 s: unknown, not stale numbers
    w.add(11.0, 10)               # counter reset
    assert w.rate(11.0) is None


def test_fps_summary():
    s = FpsSummary()
    for v in [10, 20, 30, 40, 50]:
        s.add(v)
    assert s.to_dict() == {"samples": 5, "min": 10, "p5": 10, "p50": 30, "avg": 30.0, "max": 50}


FAKE_FFMPEG = r"""
import sys, time
frame = 0
for i in range(200):
    frame += 25
    print(f"frame={frame}\nfps=0.0\nprogress=continue", flush=True)
    time.sleep(0.05)
"""


def test_stream_reads_progress_from_a_real_process(tmp_path):
    s = Stream(INFO, [sys.executable, "-c", FAKE_FFMPEG], str(tmp_path / "s.log"), 24.0)
    s.start()
    time.sleep(2.2)
    s.poll()
    assert s.status == "running" and s.frames > 500
    # 25 frames every 50 ms is 500 fps at most; slow CI runners oversleep, so only the upper bound is tight
    assert 50 < s.fps() < 600
    s.stop()
    assert s.status == "stopped" and s.ended_at


def test_stream_crash_is_reported_with_its_log(tmp_path):
    script = "import sys; sys.stderr.write('Invalid data found when processing input\\n'); sys.exit(1)"
    s = Stream(INFO, [sys.executable, "-c", script], str(tmp_path / "s.log"), 24.0)
    s.start()
    time.sleep(0.5)
    s.poll()
    assert s.status == "error" and "Invalid data" in s.error and s.exit_code == 1
    s.stop()
    assert s.status == "error"


def test_missing_binary(tmp_path):
    s = Stream(INFO, [str(tmp_path / "nope")], str(tmp_path / "s.log"), 24.0)
    s.start()
    assert s.status == "error" and "could not start" in s.error
