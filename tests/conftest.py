import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jf_hw_stress.probe import SourceInfo  # noqa: E402
from jf_hw_stress.tools import Capabilities, Tools  # noqa: E402


def make_source(**over) -> SourceInfo:
    base = dict(path="/media/film.mkv", width=3840, height=2160, fps=23.976, codec="hevc", profile="Main 10",
                pix_fmt="yuv420p10le", bit_depth=10, bitrate_mbps=44.0, duration_secs=300.0,
                color_transfer="smpte2084", color_primaries="bt2020", hdr_type="HDR10", dovi_profile=None,
                audio_codec="dts", audio_profile="DTS-HD MA", audio_channels=6, audio_sample_rate=48000,
                audio_language="eng")
    base.update(over)
    return SourceInfo(**base)


JELLYFIN_MAC = Capabilities(
    hwaccels={"videotoolbox", "opencl"},
    encoders={"h264_videotoolbox", "hevc_videotoolbox", "libx264", "libx265", "aac"},
    decoders={"hevc", "h264", "av1"},
    filters={"scale", "scale_vt", "tonemap_videotoolbox", "zscale", "tonemap", "hwdownload", "hwupload", "format"},
)
STOCK_MAC = Capabilities(
    hwaccels={"videotoolbox"},
    encoders={"h264_videotoolbox", "hevc_videotoolbox", "libx264", "libx265", "aac"},
    decoders={"hevc", "h264"},
    filters={"scale", "scale_vt", "tonemap", "hwdownload", "hwupload", "format"},
)
LINUX_ALL = Capabilities(
    hwaccels={"cuda", "qsv", "vaapi"},
    encoders={"h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv", "h264_vaapi", "hevc_vaapi", "libx264", "libx265", "aac"},
    decoders={"h264", "hevc", "h264_qsv", "hevc_qsv"},
    filters={"scale", "scale_cuda", "scale_qsv", "scale_vaapi", "zscale", "hwdownload", "hwupload", "format", "tonemap_vaapi"},
)


@pytest.fixture
def source():
    return make_source()


@pytest.fixture
def mac_tools():
    return Tools("/usr/bin/ffmpeg", "/usr/bin/ffprobe", "test", "8.1.3-Jellyfin", JELLYFIN_MAC)


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()
