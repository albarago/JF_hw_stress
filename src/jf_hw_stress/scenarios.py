"""Transcode targets that mirror what Jellyfin produces for common clients."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .probe import SourceInfo


@dataclass(frozen=True)
class Scenario:
    id: str
    label: str
    description: str
    codec: str          # "h264" | "hevc"
    width: int          # output width; height follows the source aspect ratio
    bitrate_mbps: int
    tonemap: bool       # HDR → SDR tone mapping

    @property
    def bitrate(self) -> str:
        return f"{self.bitrate_mbps}M"

    def with_codec(self, codec: str) -> Scenario:
        from dataclasses import replace
        return replace(self, codec=codec)


SCENARIOS: List[Scenario] = [
    Scenario("4k_hdr_sdr", "4K HDR → 1080p SDR", "HDR tone-mapped to 1080p SDR; the heaviest common transcode",
             "hevc", 1920, 8, True),
    Scenario("4k_1080p", "4K → 1080p HEVC", "4K downscale without tone mapping", "hevc", 1920, 8, False),
    Scenario("4k_720p", "4K → 720p H.264", "4K to a mobile-friendly stream", "h264", 1280, 4, False),
    Scenario("1080p_compat", "1080p → H.264", "Codec compatibility for older clients (Chromecast, older Apple TV)",
             "h264", 1920, 8, False),
    Scenario("1080p_webdl", "1080p HEVC 6M", "Typical 1080p web-delivery quality", "hevc", 1920, 6, False),
    Scenario("1080p_bluray", "1080p HEVC 25M", "High-bitrate 1080p that pushes the encoder", "hevc", 1920, 25, False),
    Scenario("720p_mobile", "720p H.264 4M", "Low-bandwidth mobile stream", "h264", 1280, 4, False),
    Scenario("480p_remote", "480p H.264 2M", "Thin remote connection", "h264", 854, 2, False),
]

# Mixed mode draws one of these per stream, to mimic a mix of Jellyfin clients.
MIXED: List[Scenario] = [
    Scenario("mix_4k_hevc_20", "4K HEVC 20M", "4K HEVC high bitrate", "hevc", 3840, 20, False),
    Scenario("mix_4k_hevc_8", "4K HEVC 8M", "4K HEVC medium bitrate", "hevc", 3840, 8, False),
    Scenario("mix_4k_h264_15", "4K H.264 15M", "4K H.264", "h264", 3840, 15, False),
    Scenario("mix_1080_hevc_25", "1080p HEVC 25M", "1080p HEVC Blu-ray-like", "hevc", 1920, 25, False),
    Scenario("mix_1080_hevc_8", "1080p HEVC 8M", "1080p HEVC web-DL", "hevc", 1920, 8, False),
    Scenario("mix_1080_h264_12", "1080p H.264 12M", "1080p H.264 high", "h264", 1920, 12, False),
    Scenario("mix_1080_h264_8", "1080p H.264 8M", "1080p H.264 compatibility", "h264", 1920, 8, False),
    Scenario("mix_720_hevc_4", "720p HEVC 4M", "720p HEVC", "hevc", 1280, 4, False),
    Scenario("mix_720_h264_4", "720p H.264 4M", "720p H.264 mobile", "h264", 1280, 4, False),
    Scenario("mix_720_h264_2", "720p H.264 2M", "720p H.264 low bandwidth", "h264", 1280, 2, False),
    Scenario("mix_480_h264_2", "480p H.264 2M", "480p H.264 remote", "h264", 854, 2, False),
    Scenario("mix_480_h264_1", "480p H.264 1M", "480p H.264 thin connection", "h264", 854, 1, False),
    Scenario("mix_hdr_1080_8", "HDR → 1080p SDR 8M", "HDR tone-mapped to 1080p", "hevc", 1920, 8, True),
    Scenario("mix_hdr_720_4", "HDR → 720p SDR 4M", "HDR tone-mapped to 720p", "h264", 1280, 4, True),
]

ALL = {s.id: s for s in SCENARIOS + MIXED}


def by_id(scenario_id: str) -> Optional[Scenario]:
    return ALL.get(scenario_id)


def applicable(scenario: Scenario, source: SourceInfo) -> Tuple[bool, str]:
    """Whether a scenario makes sense for this source, and why not.

    No upscaling (a 1080p source cannot exercise a 4K output) and tone mapping only for HDR.
    """
    if source.width and scenario.width > source.width:
        return False, f"output {scenario.width} px is wider than the {source.width} px source (no upscaling)"
    if scenario.tonemap and not source.is_hdr:
        return False, "tone mapping needs an HDR source"
    return True, ""


def mixed_pool(source: SourceInfo) -> List[Scenario]:
    return [s for s in MIXED if applicable(s, source)[0]]


def output_height(scenario: Scenario, source: SourceInfo) -> int:
    """Height for the scenario width at the source aspect ratio, rounded to an even number."""
    if source.width and source.height:
        return max(2, int(round(scenario.width * source.height / source.width / 2)) * 2)
    return max(2, int(round(scenario.width * 9 / 16 / 2)) * 2)
