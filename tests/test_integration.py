"""Real ffmpeg runs on a generated clip (software encode, so it works on any CI runner).

Run with:  pytest -m integration
"""
import json
import shutil
import subprocess

import pytest

from jf_hw_stress.cli import main

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed")]


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "testsrc.mkv"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=24",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "4", "-c:v", "libx264",
                    "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def test_fixed_software_run_produces_a_valid_report(clip, tmp_path, capsys):
    out = tmp_path / "r.json"
    code = main(["--headless", "--source", str(clip), "--platform", "sw", "--mode", "fixed", "--streams", "1",
                 "--scenario", "480p_remote", "--duration", "8", "--work-dir", str(tmp_path), "--json-file", str(out),
                 "--html-report", str(tmp_path / "r.html")])
    assert code == 0
    report = json.loads(out.read_text())
    stream = report["streams"][0]
    assert report["schema_version"] == 2 and stream["encode"]["encoder"] == "libx264"
    assert stream["frames"] > 48            # longer than the 4 s clip: the source looped
    assert stream["fps"]["samples"] >= 3 and stream["status"] == "stopped"
    assert (tmp_path / "r.html").read_text().startswith("<!DOCTYPE html>")
    assert not any(p.name.startswith("jf-hw-stress-") for p in tmp_path.iterdir())   # run folder removed


def _has_libass() -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return any(line.split()[1:2] == ["subtitles"] for line in out.splitlines() if line.strip())


@pytest.mark.skipif(not _has_libass(), reason="this ffmpeg has no libass (subtitles filter)")
@pytest.mark.parametrize("spec", ["auto", "sample"])
def test_burn_in_run_draws_subtitles_past_the_loop(clip, tmp_path, spec):
    from jf_hw_stress.subtitles import sample_ass
    subbed = tmp_path / "subbed.mkv"
    (tmp_path / "s.ass").write_text(sample_ass(4), encoding="utf-8")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(clip), "-i", str(tmp_path / "s.ass"),
                    "-map", "0", "-map", "1", "-c", "copy", str(subbed)], check=True)
    out = tmp_path / "r.json"
    code = main(["--headless", "--source", str(subbed), "--platform", "sw", "--mode", "fixed", "--streams", "1",
                 "--scenario", "480p_remote", "--duration", "8", "--burn-in", spec, "--work-dir", str(tmp_path),
                 "--json-file", str(out)])
    assert code == 0
    report = json.loads(out.read_text())
    stream = report["streams"][0]
    assert stream["burn_in"] == "CPU, libass" and stream["frames"] > 48 and stream["status"] == "stopped"
    assert report["config"]["burn_in"]["origin"] == ("track" if spec == "auto" else "sample")
    assert report["source"]["subtitles"][0]["kind"] == "text"
    assert not any(p.name.startswith("jf-hw-stress-") for p in tmp_path.iterdir())   # run folder removed


def test_escalating_software_run_reaches_a_limit_or_the_duration(clip, tmp_path):
    out = tmp_path / "r.json"
    code = main(["--headless", "--source", str(clip), "--platform", "sw", "--scenario", "480p_remote",
                 "--ramp-interval", "3", "--warmup-secs", "3", "--fail-secs", "2", "--max-streams", "3",
                 "--duration", "40", "--work-dir", str(tmp_path), "--json-file", str(out)])
    report = json.loads(out.read_text())
    assert code in (0, 1)
    assert report["result"]["stop_reason"] in ("limit-reached", "max-streams", "duration")
    assert report["result"]["max_stable_streams"] is not None
