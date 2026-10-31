"""Hardware backends: one table used by auto-detection and by --platform.

Each entry says how ffmpeg decodes, scales, tone-maps and encodes on that backend. `tested` records
whether the filter chains have been run on real hardware; untested backends are built from the
ffmpeg documentation and Jellyfin's own command lines, and are reported as such.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from .tools import SYSTEM, Capabilities, run_text

SW_TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable,"
              "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


class PlatformError(RuntimeError):
    pass


@dataclass(frozen=True)
class Platform:
    id: str
    name: str
    systems: Tuple[str, ...]
    encoders: Dict[str, str]                       # target codec → encoder
    hwaccel: str = ""                              # "" = software decode
    hwaccel_device: str = ""                       # named device from init_args, if any
    hwaccel_output_format: str = ""                # "" = decoded frames land in system memory
    init_args: Tuple[str, ...] = ()                # e.g. -init_hw_device …
    decoders: Dict[str, str] = field(default_factory=dict)  # codec → explicit decoder (QSV)
    decode_codecs: Tuple[str, ...] = ()
    encoder_args: Dict[str, Tuple[str, ...]] = field(default_factory=dict)  # per target codec, "*" for all
    scale_filter: str = "scale"                    # runs on frames in hwaccel_output_format
    scale_format: str = ""                         # extra scale option to output 8-bit, e.g. ":format=nv12"
    upload: str = ""                               # system frames → encoder frames ("" = encoder takes system frames)
    tonemap_filter: str = ""                       # on hardware frames
    tonemap_requires: str = ""                     # filter that must exist for tonemap_filter
    tested: bool = False
    device: str = ""                               # GPU name, filled in by detection

    @property
    def is_hardware(self) -> bool:
        return self.id != "sw"

    def args_for(self, codec: str) -> Tuple[str, ...]:
        return self.encoder_args.get("*", ()) + self.encoder_args.get(codec, ())

    def missing(self, caps: Capabilities) -> List[str]:
        """What this ffmpeg build lacks for the platform (empty when usable)."""
        gaps = []
        if self.hwaccel and self.hwaccel not in caps.hwaccels:
            gaps.append(f"hwaccel {self.hwaccel}")
        gaps += [f"encoder {enc}" for enc in self.encoders.values() if enc not in caps.encoders]
        if self.scale_filter not in caps.filters:
            gaps.append(f"filter {self.scale_filter}")
        return gaps

    def hw_tonemap(self, caps: Capabilities) -> bool:
        return bool(self.tonemap_filter) and caps.has_filter(self.tonemap_requires)


VIDEOTOOLBOX = Platform(
    id="vt", name="Apple VideoToolbox", systems=("Darwin",),
    encoders={"h264": "h264_videotoolbox", "hevc": "hevc_videotoolbox"},
    hwaccel="videotoolbox", hwaccel_output_format="videotoolbox_vld",
    decode_codecs=("h264", "hevc"),
    encoder_args={"hevc": ("-tag:v", "hvc1")},
    scale_filter="scale_vt",
    tonemap_filter="tonemap_videotoolbox=p=bt709:t=bt709:m=bt709:tonemap=bt2390",
    tonemap_requires="tonemap_videotoolbox",
    tested=True,
)
NVENC = Platform(
    id="nvenc", name="NVIDIA NVENC", systems=("Linux", "Windows"),
    encoders={"h264": "h264_nvenc", "hevc": "hevc_nvenc"},
    hwaccel="cuda", hwaccel_output_format="cuda",
    decode_codecs=("h264", "hevc", "av1", "vp9"),
    encoder_args={"*": ("-preset", "p4", "-rc", "cbr")},
    scale_filter="scale_cuda", scale_format=":format=yuv420p",
    tonemap_filter="tonemap_cuda=format=yuv420p:p=bt709:t=bt709:m=bt709:tonemap=bt2390",
    tonemap_requires="tonemap_cuda",
)
QSV = Platform(
    id="qsv", name="Intel Quick Sync", systems=("Linux", "Windows"),
    encoders={"h264": "h264_qsv", "hevc": "hevc_qsv"},
    hwaccel="qsv", hwaccel_device="hw", hwaccel_output_format="qsv",
    init_args=("-init_hw_device", "qsv=hw", "-filter_hw_device", "hw"),
    decoders={"h264": "h264_qsv", "hevc": "hevc_qsv", "av1": "av1_qsv"},
    decode_codecs=("h264", "hevc", "av1"),
    encoder_args={"*": ("-preset", "medium")},
    scale_filter="scale_qsv", scale_format=":format=nv12",
    upload="format=nv12,hwupload=extra_hw_frames=64",
)
VAAPI = Platform(
    id="vaapi", name="VA-API (Intel / AMD)", systems=("Linux",),
    encoders={"h264": "h264_vaapi", "hevc": "hevc_vaapi"},
    hwaccel="vaapi", hwaccel_device="va", hwaccel_output_format="vaapi",
    init_args=("-init_hw_device", "vaapi=va:/dev/dri/renderD128", "-filter_hw_device", "va"),
    decode_codecs=("h264", "hevc", "av1", "vp9"),
    scale_filter="scale_vaapi", scale_format=":format=nv12",
    upload="format=nv12,hwupload",
    tonemap_filter="tonemap_vaapi=format=nv12:p=bt709:t=bt709:m=bt709",
    tonemap_requires="tonemap_vaapi",
)
AMF = Platform(
    id="amf", name="AMD AMF", systems=("Windows",),
    encoders={"h264": "h264_amf", "hevc": "hevc_amf"},
    # Decoded frames are downloaded to system memory: AMF encoders accept them, and stock ffmpeg
    # has no scale filter for D3D11 frames.
    hwaccel="d3d11va",
    decode_codecs=("h264", "hevc", "av1"),
    encoder_args={"*": ("-quality", "speed", "-rc", "cbr")},
)
SOFTWARE = Platform(
    id="sw", name="Software (libx264 / libx265)", systems=("Darwin", "Linux", "Windows"),
    encoders={"h264": "libx264", "hevc": "libx265"},
    encoder_args={"*": ("-preset", "fast")},
    tested=True, device="CPU",
)

ALL: List[Platform] = [VIDEOTOOLBOX, NVENC, QSV, VAAPI, AMF, SOFTWARE]
BY_ID: Dict[str, Platform] = {p.id: p for p in ALL}


# --- device names --------------------------------------------------------------------------------

def _apple_gpu() -> str:
    for line in run_text(["system_profiler", "SPDisplaysDataType"]).splitlines():
        if "Chipset Model" in line:
            return line.split(":", 1)[-1].strip()
    return "Apple GPU"


def _nvidia_gpu() -> str:
    lines = [ln.strip() for ln in run_text(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()]
    return next((ln for ln in lines if ln), "NVIDIA GPU")


def _linux_gpu() -> str:
    for line in run_text(["lspci"]).splitlines():
        if re.search(r"vga|3d|display", line, re.I) and ":" in line:
            return line.split(":", 2)[-1].strip()
    return "GPU"


def _windows_gpu() -> str:
    out = run_text(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_VideoController).Name"], timeout=20)
    names = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return names[0] if names else "GPU"


def device_name(platform: Platform) -> str:
    if platform.id == "vt":
        return _apple_gpu()
    if platform.id == "nvenc" and shutil.which("nvidia-smi"):
        return _nvidia_gpu()
    if SYSTEM == "Windows":
        return _windows_gpu()
    if SYSTEM == "Linux":
        return _linux_gpu()
    return "GPU"


def apple_av1_decode(chip: str) -> bool:
    """Apple added AV1 hardware decode with the M3 and A17 Pro generations."""
    match = re.search(r"\bM(\d+)\b", chip)
    if match:
        return int(match.group(1)) >= 3
    match = re.search(r"\bA(\d+)\b", chip)
    return bool(match) and int(match.group(1)) >= 17


def _finish(platform: Platform) -> Platform:
    if not platform.is_hardware:
        return platform
    device = device_name(platform)
    decode = platform.decode_codecs
    if platform.id == "vt" and apple_av1_decode(device):
        decode = decode + ("av1",)
    return replace(platform, device=device, decode_codecs=decode)


# --- selection ---------------------------------------------------------------------------------

def detect(caps: Capabilities, system: str = SYSTEM) -> List[Platform]:
    """Usable hardware backends in priority order, or software alone when there are none."""
    found = [_finish(p) for p in ALL if p.is_hardware and system in p.systems and not p.missing(caps)]
    return found or [SOFTWARE]


def choose(ids: List[str], caps: Capabilities, system: str = SYSTEM) -> List[Platform]:
    """The named backends, refusing any this ffmpeg build or OS cannot run (never a silent fallback)."""
    chosen = []
    for pid in ids:
        platform = BY_ID.get(pid)
        if platform is None:
            raise PlatformError(f"unknown platform {pid!r}; choose from {', '.join(BY_ID)}")
        if system not in platform.systems:
            raise PlatformError(f"{platform.name} is not available on {system}")
        gaps = platform.missing(caps)
        if gaps:
            raise PlatformError(f"this ffmpeg build cannot use {platform.name}: missing {', '.join(gaps)}")
        chosen.append(_finish(platform))
    if not chosen:
        raise PlatformError("no platform chosen")
    return chosen


def describe_support(caps: Capabilities, system: str = SYSTEM) -> List[Tuple[Platform, Optional[str]]]:
    """Every platform with the reason it is unavailable here (None when usable)."""
    rows = []
    for p in ALL:
        if system not in p.systems:
            rows.append((p, f"not available on {system}"))
        else:
            gaps = p.missing(caps)
            rows.append((p, ("missing " + ", ".join(gaps)) if gaps else None))
    return rows
