import json

import pytest

from conftest import JELLYFIN_MAC, make_source
from jf_hw_stress import platforms
from jf_hw_stress.controller import Thresholds
from jf_hw_stress.engine import Engine, RunConfig, make_work_dir, remove_work_dir
from jf_hw_stress.headless import exit_code
from jf_hw_stress.report import build, render_html
from jf_hw_stress.scenarios import ALL, mixed_pool
from jf_hw_stress.stream import FpsSummary
from jf_hw_stress.tools import Tools

FAST = Thresholds(fps_ratio=0.9, ramp_interval=4, fail_secs=2, warmup_secs=2, start_timeout=20)


class Machine:
    """A media engine with a fixed total frame rate shared by the running streams (software streams get their own)."""
    def __init__(self, clock, hw_total=200.0, sw_total=60.0, crash=None, stall=None):
        self.clock, self.hw_total, self.sw_total, self.crash = clock, hw_total, sw_total, crash or set()
        self.stall = stall or set()   # stream ids that start but never produce a frame
        self.streams = []

    def factory(self, info, argv, log_path, source_fps, clock, wall):
        stream = FakeStream(self, info, source_fps, clock)
        self.streams.append(stream)
        return stream


class FakeStream:
    def __init__(self, machine, info, source_fps, clock):
        self.m, self.info, self.source_fps, self._clock = machine, info, source_fps, clock
        self.status, self.frames, self.error, self.started_at, self.ended_at = "starting", 0, "", 1.0, None
        self.fps_summary, self.argv, self._start = FpsSummary(), None, 0.0

    def start(self):
        self._start = self._clock()
        if self.info.id in self.m.crash:
            self.status, self.error = "error", "Invalid data found when processing input"

    def _rate(self):
        peers = [s for s in self.m.streams if s.status in ("starting", "running") and s.info.encode_hw == self.info.encode_hw]
        total = self.m.hw_total if self.info.encode_hw else self.m.sw_total
        return total / max(1, len(peers))

    def poll(self):
        if self.status == "starting" and self._clock() - self._start >= 1 and self.info.id not in self.m.stall:
            self.status = "running"
        if self.status == "running":
            self.frames += int(self._rate())
            self.fps_summary.add(self._rate())

    def signal_stop(self):
        pass

    def stop(self):
        if self.status in ("starting", "running"):
            self.status = "stopped"

    @property
    def active(self):
        return self.status in ("starting", "running")

    def fps(self, now=None):
        return self._rate() if self.status == "running" else None

    def speed(self, now=None):
        rate = self.fps()
        return None if rate is None else rate / self.source_fps

    def age(self):
        return self._clock() - self._start


def run(cfg, clock, machine, tmp_path, limit=2000):
    tools = Tools("/bin/ffmpeg", "/bin/ffprobe", "test", "8.1.3-Jellyfin", JELLYFIN_MAC)
    engine = Engine(cfg, tools, make_source(), str(tmp_path), clock=clock, wall=clock, stream_factory=machine.factory)
    engine.start()
    for _ in range(limit):
        clock.advance(0.5)
        if engine.tick() != "running":
            break
    if engine.state == "limit":
        engine.finish()
    return engine


def config(mode="escalating", **kw):
    vt = platforms.VIDEOTOOLBOX
    return RunConfig(mode=mode, scenario=ALL["4k_hdr_sdr"], platforms=[vt], thresholds=FAST, **kw)


def test_escalation_finds_the_capacity(clock, tmp_path):
    # 200 fps shared: 9 streams get 22.2 fps (≥ 21.6), 10 get 20 fps.
    engine = run(config(), clock, Machine(clock), tmp_path)
    assert engine.max_stable() == 9 and engine.peak_concurrent == 10
    assert engine.failure["kind"] == "capacity" and engine.stop_reason == "limit-reached"
    assert len([s for s in engine.streams if s.status == "stopped"]) == 10
    report = build(engine)
    assert report["schema_version"] == 2 and report["result"]["max_stable_streams"] == 9
    assert exit_code(report) == 0


def test_hybrid_finds_hardware_then_software(clock, tmp_path):
    engine = run(config("hybrid"), clock, Machine(clock, sw_total=50.0), tmp_path)
    # hardware: 9; software 50 fps: 2 streams at 25 fps keep up, a third does not
    assert engine.hw_max_stable == 9 and engine.max_stable() == 11
    sw = [s for s in engine.streams if not s.info.encode_hw]
    assert len(sw) == 3 and all(s.info.encoder == "libx265" for s in sw)


def test_a_crash_ends_the_run_as_an_error(clock, tmp_path):
    engine = run(config(), clock, Machine(clock, crash={3}), tmp_path)
    assert engine.failure["kind"] == "error" and engine.max_stable() == 2 and engine.stop_reason == "stream-failed"
    assert exit_code(build(engine)) == 1


def test_a_stalled_stream_is_replaced_once(clock, tmp_path):
    engine = run(config(), clock, Machine(clock, stall={4}), tmp_path)
    assert engine.max_stable() == 9 and engine.failure["kind"] == "capacity"
    assert engine.streams[3].status == "stopped" and engine.streams[4].info.id == 5
    report = build(engine)
    assert any("stream 4 produced no frames" in w for w in report["warnings"]) and exit_code(report) == 0


def test_a_second_stall_ends_the_run_without_a_hybrid_pivot(clock, tmp_path):
    engine = run(config("hybrid"), clock, Machine(clock, stall={4, 5}), tmp_path)
    assert engine.failure["kind"] == "stall" and engine.stop_reason == "stream-failed"
    assert engine.hw_max_stable is None and all(s.info.encode_hw for s in engine.streams)
    report = build(engine)
    assert report["result"]["limit_reached"] is False and exit_code(report) == 1


def test_fixed_mode_counts_streams_keeping_up(clock, tmp_path):
    engine = run(config("fixed", streams=8, duration=30), clock, Machine(clock), tmp_path)
    report = build(engine)
    assert report["result"]["streams_keeping_up"] == 8 and report["result"]["stop_reason"] == "duration"
    engine = run(config("fixed", streams=12, duration=30), clock, Machine(clock), tmp_path)
    assert build(engine)["result"]["streams_keeping_up"] == 0


def test_stream_cap(clock, tmp_path):
    engine = run(config(max_streams=3), clock, Machine(clock), tmp_path)
    assert engine.stop_reason == "max-streams" and engine.max_stable() == 3 and engine.failure is None


def test_mixed_is_repeatable_with_a_seed(clock, tmp_path):
    pool = mixed_pool(make_source())
    first = run(config("mixed", pool=pool, seed=7, duration=40), clock, Machine(clock, hw_total=1000), tmp_path)
    again = run(config("mixed", pool=pool, seed=7, duration=40), clock, Machine(clock, hw_total=1000), tmp_path)
    assert [s.info.scenario_id for s in first.streams] == [s.info.scenario_id for s in again.streams]
    assert len({s.info.scenario_id for s in first.streams}) > 1


def test_round_robin_over_platforms(clock, tmp_path):
    caps_vt = platforms.VIDEOTOOLBOX
    cfg = config("fixed", streams=4, duration=5)
    cfg.platforms = [caps_vt, platforms.SOFTWARE]
    engine = run(cfg, clock, Machine(clock), tmp_path)
    assert [s.info.platform_id for s in engine.streams] == ["vt", "vt", "vt", "vt"]  # software is only the hybrid pivot


def test_report_html_escapes_media_metadata(clock, tmp_path):
    engine = run(config("fixed", streams=1, duration=3), clock, Machine(clock), tmp_path)
    engine.source.path = "/m/<script>alert(1)</script>.mkv"
    engine.source.audio_codec = "<img src=x onerror=alert(1)>"
    page = render_html(build(engine))
    assert "<script>alert" not in page and "<img src=x" not in page and "&lt;script&gt;" in page
    json.dumps(build(engine))  # serialisable


def test_work_dir_removes_only_its_own_files(tmp_path):
    work = make_work_dir(str(tmp_path))
    for name in ("stream1.m3u8", "stream1_00001.ts", "stream1.log"):
        (tmp_path / work / name).write_text("x")
    removed, left = remove_work_dir(work)
    assert (removed, left) == (3, 0) and not (tmp_path / work).exists()
    work = make_work_dir(str(tmp_path / "other"))
    (tmp_path / "other" / work / "keep.txt").write_text("mine")
    assert remove_work_dir(work)[1] == 1 and (tmp_path / "other" / work / "keep.txt").exists()


def test_check_refuses_impossible_runs_before_starting(clock, tmp_path):
    from jf_hw_stress.command import CommandError
    stock = Tools("/bin/ffmpeg", "/bin/ffprobe", "test", "9.0", platforms.Capabilities(
        hwaccels={"videotoolbox"}, encoders={"h264_videotoolbox", "hevc_videotoolbox", "libx264", "libx265"},
        filters={"scale", "scale_vt"}))
    machine = Machine(clock)
    engine = Engine(config(), stock, make_source(), str(tmp_path), clock=clock, wall=clock, stream_factory=machine.factory)
    with pytest.raises(CommandError):
        engine.start()
    assert machine.streams == []
