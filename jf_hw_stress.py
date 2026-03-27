#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   Jellyfin Hardware Transcode Stress Tester   v1.5              ║
║   Apple VideoToolbox · NVIDIA NVENC · Intel QSV · AMD AMF       ║
╚══════════════════════════════════════════════════════════════════╝

Stress-tests hardware transcoding the way Jellyfin actually uses it.

Usage:
    python3 jf_hw_stress.py                             # interactive TUI
    python3 jf_hw_stress.py --headless --source /path/to/movie.mkv  # headless

Headless mode runs without interactive prompts, outputting JSON results
to stdout. Useful for CI, Kubernetes Jobs, and automated benchmarking.

Requirements:
    pip install rich
"""

from __future__ import annotations

import argparse
import atexit
import datetime
import html as html_module
import json
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.markup import escape as esc
    from rich.panel import Panel
    from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("\n  This tool requires the 'rich' library.\n  Install it with:  pip install rich\n")
    sys.exit(1)

console = Console()


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

VERSION   = "1.5"
APP_TITLE = "Jellyfin Transcode Stress Tester"
SYSTEM    = platform.system()   # "Darwin" | "Linux" | "Windows"

JELLYFIN_HINTS: Dict[str, List[Path]] = {
    "Darwin":  [Path("/Applications/Jellyfin.app/Contents/MacOS")],
    "Linux":   [Path("/usr/lib/jellyfin-ffmpeg"),
                Path("/usr/local/bin/jellyfin-ffmpeg"),
                Path("/opt/jellyfin")],
    "Windows": [Path(r"C:\Program Files\Jellyfin\Server"),
                Path(r"C:\Program Files (x86)\Jellyfin\Server")],
}

VIDEO_EXTS  = {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".ts", ".mpg", ".wmv"}
_KILL_GRACE = 0.5

# Test modes
MODE_FIXED      = "fixed"
MODE_ESCALATING = "escalating"
MODE_HYBRID     = "hybrid"    # HW escalate → when HW saturated pivot to SW
MODE_MIXED      = "mixed"     # escalating with random scenario per stream

# Audio codec display names
_AUDIO_NAMES: Dict[str, str] = {
    "aac": "AAC", "ac3": "Dolby Digital AC3", "eac3": "Dolby Digital+ EAC3",
    "dts": "DTS", "truehd": "Dolby TrueHD", "mlp": "MLP (TrueHD)",
    "flac": "FLAC", "mp3": "MP3", "opus": "Opus", "vorbis": "Vorbis",
    "pcm_s16le": "PCM 16-bit", "pcm_s24le": "PCM 24-bit",
}
_CH_NAMES: Dict[int, str] = {1: "Mono", 2: "Stereo", 4: "4.0", 6: "5.1", 8: "7.1"}

# Software HDR→SDR tonemap chains
_SW_TM_FROM_HW = (          # HW frames on GPU → download then tonemap in SW
    "hwdownload,format=p010le,"
    "zscale=t=linear:npl=100,format=gbrpf32le,"
    "zscale=p=bt709,tonemap=hable,"
    "zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)
_SW_TM_PURE = (             # SW decode path → no download needed
    "zscale=t=linear:npl=100,format=gbrpf32le,"
    "zscale=p=bt709,tonemap=hable,"
    "zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolPaths:
    ffmpeg:  str
    ffprobe: str
    source:  str   # human-readable, e.g. "Jellyfin bundle" | "system PATH"


@dataclass
class SourceInfo:
    path:              str
    width:             int
    height:            int
    fps:               float
    codec:             str
    profile:           str
    bitrate_mbps:      float
    is_hdr:            bool
    color_transfer:    str
    # extended fields
    hdr_type:          str = "SDR"    # SDR | HDR10 | HDR10+ | HLG | Dolby Vision
    dovi_profile:      Optional[int] = None
    bit_depth:         int = 8
    pix_fmt:           str = "unknown"
    audio_codec:       str = "unknown"
    audio_channels:    int = 0
    audio_sample_rate: int = 0
    audio_bitrate_kbps: float = 0.0
    audio_language:    str = ""


@dataclass
class HardwarePlatform:
    name:               str
    short:              str
    hwaccel:            str
    hwaccel_output:     str
    h264_encoder:       str
    hevc_encoder:       str
    scale_filter:       str
    tonemap_filter:     Optional[str]
    gpu_name:           str           = "Unknown"
    extra_encode_flags: List[str]     = field(default_factory=list)
    hw_decode_codecs:   List[str]     = field(default_factory=list)


@dataclass
class Scenario:
    id:           str
    label:        str
    description:  str
    target_codec: str   # "h264" | "hevc"
    width:        int
    bitrate:      str   # e.g. "8M"
    tonemap:      bool


@dataclass
class TestConfig:
    hls_segment_secs: int   = 4
    hls_list_size:    int   = 3
    ramp_interval:    float = 12.0
    fail_secs:        float = 5.0
    fps_ratio:        float = 0.90
    warmup_secs:      float = 6.0


@dataclass
class StreamStats:
    stream_id:   int
    label:       str
    speed:       float = 0.0
    fps:         float = 0.0
    bitrate_str: str   = "..."
    frames:      int   = 0
    status:      str   = "starting"  # starting | running | error | done
    pid:         Optional[int] = None
    enc_hw:      bool = True
    dec_hw:      bool = True
    scenario_id: str  = ""
    started_at:  float = 0.0
    ended_at:    float = 0.0
    fps_history: List[float] = field(default_factory=list)
    error_msg:   str  = ""


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS: List[Scenario] = [
    Scenario("4k_hdr_sdr",   "4K HDR → 1080p SDR",     "HDR tone-map to 1080p — heaviest transcode",           "hevc", 1920, "8M",  True),
    Scenario("4k_1080p",     "4K → 1080p HEVC",         "4K downscale, no tone mapping",                        "hevc", 1920, "8M",  False),
    Scenario("4k_720p",      "4K → 720p H.264",          "4K to mobile — broad device compat",                   "h264", 1280, "4M",  False),
    Scenario("1080p_compat", "1080p HEVC → H.264",       "Codec-compat (Apple TV, Chromecast, older clients)",   "h264", 1920, "8M",  False),
    Scenario("1080p_webdl",  "1080p HEVC WebDL",         "Standard 1080p web-delivery quality",                  "hevc", 1920, "6M",  False),
    Scenario("1080p_bluray", "1080p HEVC Bluray",        "High-bitrate 1080p — pushing encoder limits",          "hevc", 1920, "25M", False),
    Scenario("720p_mobile",  "720p H.264 Mobile",        "Low-bandwidth mobile stream",                          "h264", 1280, "4M",  False),
    Scenario("480p_remote",  "480p H.264 Remote",        "Thin-pipe remote access — lowest bitrate",             "h264", 854,  "2M",  False),
]

# ── Mixed-bag scenario pool ───────────────────────────────────────────────────
# Uses only standard display resolutions (4K / 1080p / 720p / 480p) with a
# spread of codecs and bitrates that mirror real Jellyfin client diversity.
# HDR→SDR entries are appended at runtime only when the source is HDR.
MIXED_SCENARIOS: List[Scenario] = [
    # 4K outputs
    Scenario("mix_4k_hevc_hi",  "4K HEVC 20M",    "4K HEVC high bitrate",       "hevc", 3840, "20M", False),
    Scenario("mix_4k_hevc_med", "4K HEVC 8M",     "4K HEVC medium",             "hevc", 3840, "8M",  False),
    Scenario("mix_4k_h264",     "4K H.264 15M",   "4K H.264",                   "h264", 3840, "15M", False),
    # 1080p outputs
    Scenario("mix_1080_hevc_hi","1080p HEVC 25M", "1080p HEVC Bluray",          "hevc", 1920, "25M", False),
    Scenario("mix_1080_hevc_md","1080p HEVC 8M",  "1080p HEVC WebDL",           "hevc", 1920, "8M",  False),
    Scenario("mix_1080_hevc_lo","1080p HEVC 6M",  "1080p HEVC web",             "hevc", 1920, "6M",  False),
    Scenario("mix_1080_h264_hi","1080p H.264 12M","1080p H.264 high",           "h264", 1920, "12M", False),
    Scenario("mix_1080_h264_md","1080p H.264 8M", "1080p H.264 compat",         "h264", 1920, "8M",  False),
    # 720p outputs
    Scenario("mix_720_hevc",    "720p HEVC 4M",   "720p HEVC",                  "hevc", 1280, "4M",  False),
    Scenario("mix_720_h264",    "720p H.264 4M",  "720p H.264 mobile",          "h264", 1280, "4M",  False),
    Scenario("mix_720_h264_lo", "720p H.264 2M",  "720p H.264 low bandwidth",   "h264", 1280, "2M",  False),
    # 480p outputs
    Scenario("mix_480_h264",    "480p H.264 2M",  "480p H.264 remote",          "h264",  854, "2M",  False),
    Scenario("mix_480_h264_lo", "480p H.264 1M",  "480p H.264 thin pipe",       "h264",  854, "1M",  False),
]

# HDR→SDR variants added to pool at runtime when source is HDR
MIXED_HDR_SCENARIOS: List[Scenario] = [
    Scenario("mix_4k_hdr_1080", "4K HDR→1080p 8M", "4K HDR tone-map to 1080p", "hevc", 1920, "8M",  True),
    Scenario("mix_4k_hdr_720",  "4K HDR→720p 4M",  "4K HDR tone-map to 720p",  "h264", 1280, "4M",  True),
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _run_silent(cmd: List[str], timeout: int = 8) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, errors="replace")
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def _clean_path(raw: str) -> str:
    """
    Sanitise a path that may originate from macOS drag-and-drop.
    Terminal wraps space-containing paths in single quotes; shells
    may also use backslash-escaped spaces.
    """
    p = raw.strip()
    # Strip surrounding quote pair added by macOS Terminal
    if len(p) >= 2 and ((p[0] == "'" and p[-1] == "'") or
                        (p[0] == '"' and p[-1] == '"')):
        p = p[1:-1]
    p = p.replace("\\ ", " ")   # un-escape backslash-spaces
    p = p.rstrip("/").rstrip("\\").strip()
    return p


def _fmt_elapsed(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


def _atexit_restore_terminal():
    """Best-effort terminal restore on any exit path."""
    try:
        if SYSTEM != "Windows" and sys.stdin.isatty():
            os.system("stty sane 2>/dev/null")
    except Exception:
        pass
    try:
        console.show_cursor(True)
    except Exception:
        pass

atexit.register(_atexit_restore_terminal)


def _countdown_choice(
    header: str,
    options: List[Tuple[str, str]],   # (key_char, description)
    default: str,
    timeout: float = 60.0,
) -> str:
    """
    Single-keypress menu with live countdown.
    Uses terminal raw mode on Unix; falls back to Prompt on Windows.
    ONLY call this when NOT inside a Rich Live context.
    Returns the pressed key (lowercased) or default on timeout / Enter.
    """
    console.print()
    console.print(f"  [bold]{header}[/bold]")
    console.print()
    for key, desc in options:
        hi = "bold yellow" if key.lower() == default.lower() else "bold"
        console.print(f"  [{hi}]{key.upper()}[/{hi}]  {desc}")
    console.print()

    valid = {k.lower() for k, _ in options}

    if SYSTEM != "Windows":
        try:
            import select as _sel
            import termios, tty
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            tty.setraw(fd)
            result = default.lower()
            deadline = time.time() + timeout
            try:
                while True:
                    rem = max(0.0, deadline - time.time())
                    keys_hint = " / ".join(k.upper() for k, _ in options)
                    line = (f"\r  Press {keys_hint}"
                            f"  —  auto→{default.upper()} in {rem:.0f}s   ")
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    if rem <= 0:
                        break
                    ready, _, _ = _sel.select([sys.stdin], [], [], min(0.5, rem))
                    if ready:
                        ch = sys.stdin.read(1).lower()
                        if ch in valid:
                            result = ch
                            break
                        elif ch in ("\r", "\n") and default:
                            result = default.lower()
                            break
                        elif ch == "\x03":      # Ctrl+C
                            raise KeyboardInterrupt
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                sys.stdout.write("\n\n")
                sys.stdout.flush()
            return result
        except KeyboardInterrupt:
            raise
        except Exception:
            pass  # fall through to Prompt

    # Windows / fallback
    keys = [k for k, _ in options]
    return Prompt.ask(
        f"  Choice (auto: {default} in {timeout:.0f}s)",
        choices=keys, default=default,
    ).lower()


# ══════════════════════════════════════════════════════════════════════════════
# TOOL DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def _find_tools_in(directory: Path) -> Optional[ToolPaths]:
    suffix = ".exe" if SYSTEM == "Windows" else ""
    ff  = directory / f"ffmpeg{suffix}"
    ffp = directory / f"ffprobe{suffix}"
    if ff.is_file() and ffp.is_file():
        return ToolPaths(ffmpeg=str(ff), ffprobe=str(ffp), source=str(directory))
    if ff.is_file():
        ffp_w = shutil.which("ffprobe")
        if ffp_w:
            return ToolPaths(ffmpeg=str(ff), ffprobe=ffp_w, source=str(directory))
    return None


def _tools_from_path() -> Optional[ToolPaths]:
    ff  = shutil.which("ffmpeg")
    ffp = shutil.which("ffprobe")
    if ff and ffp:
        return ToolPaths(ffmpeg=ff, ffprobe=ffp, source="system PATH")
    return None


def discover_tools() -> ToolPaths:
    console.print()
    console.rule("[bold cyan]Jellyfin / FFmpeg Location[/bold cyan]")
    console.print()

    candidates: List[Tuple[str, ToolPaths]] = []
    for hint in JELLYFIN_HINTS.get(SYSTEM, []):
        if hint.is_dir():
            t = _find_tools_in(hint)
            if t:
                candidates.append((f"Jellyfin bundle  {hint}", t))
    pt = _tools_from_path()
    if pt:
        candidates.append(("System PATH  (ffmpeg + ffprobe)", pt))

    if candidates:
        console.print("  Found the following ffmpeg installations:\n")
        for idx, (label, _) in enumerate(candidates, 1):
            console.print(f"    [bold]{idx}[/bold]  {esc(label)}")
        console.print(f"    [bold]{len(candidates)+1}[/bold]  Enter a custom path\n")

        choices  = [str(i) for i in range(1, len(candidates) + 2)]
        pick     = Prompt.ask("  Select", choices=choices, default="1")
        pick_idx = int(pick) - 1
        if pick_idx < len(candidates):
            _, tools = candidates[pick_idx]
            console.print(
                f"\n  [green]✓[/green]  ffmpeg   [dim]{esc(tools.ffmpeg)}[/dim]\n"
                f"     ffprobe  [dim]{esc(tools.ffprobe)}[/dim]"
            )
            return tools

    # Manual entry
    console.print(
        "  [yellow]No Jellyfin bundle auto-detected.[/yellow]\n"
        "  Enter the directory containing [bold]ffmpeg[/bold] and [bold]ffprobe[/bold].\n"
        "  Leave blank to try system PATH.\n"
    )
    while True:
        raw = Prompt.ask("  Directory (or blank for PATH)").strip()
        if not raw:
            t = _tools_from_path()
            if t:
                console.print(f"  [green]✓[/green]  Using system PATH: {esc(t.ffmpeg)}")
                return t
            console.print("  [red]✗  ffmpeg / ffprobe not found in PATH.[/red]")
            continue
        p = Path(_clean_path(raw)).expanduser()
        if not p.is_dir():
            console.print(f"  [red]✗  Not a directory: {esc(str(p))}[/red]")
            continue
        t = _find_tools_in(p)
        if t:
            console.print(
                f"  [green]✓[/green]  ffmpeg   [dim]{esc(t.ffmpeg)}[/dim]\n"
                f"     ffprobe  [dim]{esc(t.ffprobe)}[/dim]"
            )
            return t
        console.print(f"  [red]✗  ffmpeg / ffprobe not found in: {esc(str(p))}[/red]")


# ══════════════════════════════════════════════════════════════════════════════
# HARDWARE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _mac_gpu() -> str:
    out = _run_silent(["system_profiler", "SPDisplaysDataType"])
    for line in out.splitlines():
        if "Chipset Model" in line:
            return line.split(":", 1)[-1].strip()
    return "Apple GPU"

def _mac_av1_hw(gpu: str) -> bool:
    return any(c in gpu for c in ("M3", "M4", "M5", "A17", "A18"))

def _nvidia_gpu() -> str:
    out   = _run_silent(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    return lines[0] if lines else "NVIDIA GPU"

def _gpu_windows() -> str:
    out = _run_silent(["wmic", "path", "Win32_VideoController",
                       "get", "Name", "/value"], timeout=10)
    for line in out.splitlines():
        if line.lower().startswith("name="):
            n = line.split("=", 1)[-1].strip()
            if n:
                return n
    return "GPU"

def _gpu_linux() -> str:
    out = _run_silent(["lspci"])
    for line in out.splitlines():
        lo = line.lower()
        if "vga" in lo or "3d" in lo or "display" in lo:
            if ":" in line:
                return line.split(":", 2)[-1].strip()
    return "GPU"


def detect_hardware(tools: ToolPaths) -> HardwarePlatform:
    hwaccels = _run_silent([tools.ffmpeg, "-hwaccels"])
    encoders = _run_silent([tools.ffmpeg, "-encoders"])

    if SYSTEM == "Darwin" and "videotoolbox" in hwaccels:
        gpu = _mac_gpu()
        av1 = _mac_av1_hw(gpu)
        return HardwarePlatform(
            name="Apple VideoToolbox", short="vt",
            hwaccel="videotoolbox", hwaccel_output="videotoolbox_vld",
            h264_encoder="h264_videotoolbox", hevc_encoder="hevc_videotoolbox",
            scale_filter="scale_vt",
            tonemap_filter="tonemap_videotoolbox=p=bt709:t=bt709:m=bt709:tonemap=bt2390",
            gpu_name=gpu,
            extra_encode_flags=["-tag:v", "hvc1"],
            hw_decode_codecs=["h264", "hevc"] + (["av1"] if av1 else []),
        )

    if "cuda" in hwaccels and "h264_nvenc" in encoders:
        gpu = (_nvidia_gpu() if shutil.which("nvidia-smi")
               else _gpu_windows() if SYSTEM == "Windows"
               else _gpu_linux())
        return HardwarePlatform(
            name="NVIDIA NVENC", short="nvenc",
            hwaccel="cuda", hwaccel_output="cuda",
            h264_encoder="h264_nvenc", hevc_encoder="hevc_nvenc",
            scale_filter="scale_cuda", tonemap_filter=None,
            gpu_name=gpu,
            extra_encode_flags=["-rc:v", "cbr", "-preset", "p4"],
            hw_decode_codecs=["h264", "hevc", "av1"],
        )

    if "qsv" in hwaccels and "h264_qsv" in encoders:
        gpu = _gpu_windows() if SYSTEM == "Windows" else _gpu_linux()
        return HardwarePlatform(
            name="Intel QuickSync (QSV)", short="qsv",
            hwaccel="qsv", hwaccel_output="qsv",
            h264_encoder="h264_qsv", hevc_encoder="hevc_qsv",
            scale_filter="scale_qsv", tonemap_filter=None,
            gpu_name=gpu,
            extra_encode_flags=["-preset", "medium"],
            hw_decode_codecs=["h264", "hevc"],
        )

    if SYSTEM == "Windows" and "amf" in hwaccels and "h264_amf" in encoders:
        return HardwarePlatform(
            name="AMD AMF", short="amf",
            hwaccel="d3d11va", hwaccel_output="d3d11",
            h264_encoder="h264_amf", hevc_encoder="hevc_amf",
            scale_filter="scale", tonemap_filter=None,
            gpu_name=_gpu_windows(),
            extra_encode_flags=["-quality", "speed", "-rc", "cbr"],
            hw_decode_codecs=["h264", "hevc"],
        )

    if SYSTEM == "Linux" and "vaapi" in hwaccels and "h264_vaapi" in encoders:
        hevc_enc = "hevc_vaapi" if "hevc_vaapi" in encoders else "libx265"
        if hevc_enc == "libx265":
            console.print("  [yellow]⚠  hevc_vaapi unavailable — HEVC falls back to libx265.[/yellow]")
        return HardwarePlatform(
            name="AMD / Intel VAAPI", short="vaapi",
            hwaccel="vaapi", hwaccel_output="vaapi",
            h264_encoder="h264_vaapi", hevc_encoder=hevc_enc,
            scale_filter="scale_vaapi", tonemap_filter=None,
            gpu_name=_gpu_linux(),
            hw_decode_codecs=["h264", "hevc"],
        )

    console.print("  [yellow]⚠  No hardware acceleration detected — software fallback.[/yellow]")
    return HardwarePlatform(
        name="Software (libx264/libx265)", short="sw",
        hwaccel="", hwaccel_output="",
        h264_encoder="libx264", hevc_encoder="libx265",
        scale_filter="scale", tonemap_filter=_SW_TM_PURE,
        gpu_name="CPU",
        extra_encode_flags=["-preset", "fast"],
        hw_decode_codecs=[],
    )


def _force_hardware_platform(tools: ToolPaths, platform: str) -> HardwarePlatform:
    """Force a specific hardware platform, bypassing auto-detection order."""
    encoders = _run_silent([tools.ffmpeg, "-encoders"])

    if platform == "vaapi":
        hevc_enc = "hevc_vaapi" if "hevc_vaapi" in encoders else "libx265"
        return HardwarePlatform(
            name="AMD / Intel VAAPI", short="vaapi",
            hwaccel="vaapi", hwaccel_output="vaapi",
            h264_encoder="h264_vaapi", hevc_encoder=hevc_enc,
            scale_filter="scale_vaapi", tonemap_filter=None,
            gpu_name=_gpu_linux() if SYSTEM == "Linux" else "GPU",
            hw_decode_codecs=["h264", "hevc"],
        )
    elif platform == "qsv":
        return HardwarePlatform(
            name="Intel QuickSync (QSV)", short="qsv",
            hwaccel="qsv", hwaccel_output="qsv",
            h264_encoder="h264_qsv", hevc_encoder="hevc_qsv",
            scale_filter="scale_qsv", tonemap_filter=None,
            gpu_name=_gpu_linux() if SYSTEM == "Linux" else "GPU",
            extra_encode_flags=["-preset", "medium"],
            hw_decode_codecs=["h264", "hevc"],
        )
    elif platform == "nvenc":
        gpu = (_nvidia_gpu() if shutil.which("nvidia-smi") else "NVIDIA GPU")
        return HardwarePlatform(
            name="NVIDIA NVENC", short="nvenc",
            hwaccel="cuda", hwaccel_output="cuda",
            h264_encoder="h264_nvenc", hevc_encoder="hevc_nvenc",
            scale_filter="scale_cuda", tonemap_filter=None,
            gpu_name=gpu,
            extra_encode_flags=["-rc:v", "cbr", "-preset", "p4"],
            hw_decode_codecs=["h264", "hevc", "av1"],
        )
    elif platform == "amf":
        return HardwarePlatform(
            name="AMD AMF", short="amf",
            hwaccel="d3d11va", hwaccel_output="d3d11",
            h264_encoder="h264_amf", hevc_encoder="hevc_amf",
            scale_filter="scale", tonemap_filter=None,
            gpu_name="AMD GPU",
            extra_encode_flags=["-quality", "speed", "-rc", "cbr"],
            hw_decode_codecs=["h264", "hevc"],
        )
    elif platform == "vt":
        return HardwarePlatform(
            name="Apple VideoToolbox", short="vt",
            hwaccel="videotoolbox", hwaccel_output="videotoolbox_vld",
            h264_encoder="h264_videotoolbox", hevc_encoder="hevc_videotoolbox",
            scale_filter="scale_vt",
            tonemap_filter="tonemap_videotoolbox=p=bt709:t=bt709:m=bt709:tonemap=bt2390",
            gpu_name="Apple GPU",
            extra_encode_flags=["-tag:v", "hvc1"],
            hw_decode_codecs=["h264", "hevc"],
        )
    else:  # sw
        return HardwarePlatform(
            name="Software (libx264/libx265)", short="sw",
            hwaccel="", hwaccel_output="",
            h264_encoder="libx264", hevc_encoder="libx265",
            scale_filter="scale", tonemap_filter=_SW_TM_PURE,
            gpu_name="CPU",
            extra_encode_flags=["-preset", "fast"],
            hw_decode_codecs=[],
        )


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE FILE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def pick_source_file() -> str:
    """
    Numbered shortcuts for CWD video files.
    Accepts: a number, a typed path, or a drag-and-drop path.
    Does NOT use Rich choices= so free-text always works.
    """
    console.print()
    console.rule("[bold cyan]Source File[/bold cyan]")
    console.print()

    cwd_vids = sorted(
        p for p in Path.cwd().iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    shown = cwd_vids[:8]

    if shown:
        console.print("  Video files in current directory:\n")
        for idx, p in enumerate(shown, 1):
            console.print(f"    [bold]{idx}[/bold]  {esc(p.name)}")
        console.print("\n  Enter a number, or drag-and-drop / type the full path.\n")
    else:
        console.print("  No video files found in current directory.")
        console.print("  Drag-and-drop a file here, or type the full path.\n")

    while True:
        raw     = Prompt.ask("  Source file")
        cleaned = _clean_path(raw)

        if shown and cleaned.isdigit():
            idx = int(cleaned) - 1
            if 0 <= idx < len(shown):
                path = str(shown[idx])
                console.print(f"  [green]✓[/green]  {esc(path)}")
                return path
            console.print(f"  [red]✗  Enter 1–{len(shown)} or a file path.[/red]")
            continue

        p = Path(cleaned).expanduser()
        if p.is_file():
            return str(p)
        if p.is_dir():
            console.print(
                f"  [yellow]⚠  That is a folder, not a file: {esc(str(p))}[/yellow]\n"
                "     Drop the actual video file (e.g. movie.mkv), not its folder."
            )
            continue
        console.print(f"  [red]✗  File not found:[/red] [dim]{esc(str(p))}[/dim]")


# ══════════════════════════════════════════════════════════════════════════════
# CACHE DIRECTORY SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _ramdisk_path() -> Optional[str]:
    if SYSTEM == "Linux":
        shm = Path("/dev/shm")
        if shm.is_dir():
            return str(shm / "transcode_stress")
    return None

def _disk_free_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / 1024 ** 3
    except Exception:
        return 0.0


def pick_cache_dir() -> str:
    console.print()
    console.rule("[bold cyan]Scratch / Cache Directory[/bold cyan]")
    console.print()
    console.print(
        "  HLS stress testing writes a [bold]continuous stream of .ts segments[/bold].\n"
        "  Targeting your boot SSD risks unnecessary TBW consumption.\n\n"
        "  Preference order:\n"
        "  [green]1st[/green]  RAM disk  [dim](zero wear — Linux: /dev/shm | macOS: Disk Utility)[/dim]\n"
        "  [yellow]2nd[/yellow]  External SSD / secondary drive\n"
        "  [red]3rd[/red]  System temp [dim](quick tests only)[/dim]\n"
    )

    suggestions: List[Tuple[str, str]] = []

    ram = _ramdisk_path()
    if ram:
        suggestions.append((f"RAM disk  {ram}", ram))

    if SYSTEM == "Darwin":
        try:
            for vol in Path("/Volumes").iterdir():
                if vol.name != "Macintosh HD" and vol.is_dir() and os.path.ismount(str(vol)):
                    free = _disk_free_gb(str(vol))
                    suggestions.append((f"External volume  {vol}  ({free:.0f} GB free)",
                                        str(vol / "transcode_stress")))
        except Exception:
            pass
    elif SYSTEM == "Linux":
        try:
            for mnt in Path("/mnt").iterdir():
                if mnt.is_dir():
                    free = _disk_free_gb(str(mnt))
                    suggestions.append((f"Mount  {mnt}  ({free:.0f} GB free)",
                                        str(mnt / "transcode_stress")))
        except Exception:
            pass
    elif SYSTEM == "Windows":
        import string
        boot = Path(sys.executable).anchor
        for letter in string.ascii_uppercase:
            d = Path(f"{letter}:\\")
            if d.is_dir() and str(d) != boot:
                free = _disk_free_gb(str(d))
                suggestions.append((f"Drive {letter}:\\  ({free:.0f} GB free)",
                                     str(d / "transcode_stress")))

    tmp = str(Path(tempfile.gettempdir()) / "transcode_stress")
    suggestions.append((f"System temp  {tmp}  (internal drive)", tmp))
    suggestions.append(("Enter a custom path", ""))

    for idx, (label, _) in enumerate(suggestions, 1):
        console.print(f"  [bold]{idx}[/bold]  {esc(label)}")
    console.print()

    default_pick = str(max(1, len(suggestions) - 1))
    choices      = [str(i) for i in range(1, len(suggestions) + 1)]
    pick         = Prompt.ask("  Select", choices=choices, default=default_pick)
    _, chosen    = suggestions[int(pick) - 1]

    if chosen:
        if SYSTEM == "Darwin" and "/tmp" in chosen:
            console.print("  [yellow]⚠  /tmp is on your internal SSD — fine for quick tests.[/yellow]")
        return chosen

    while True:
        raw  = Prompt.ask("  Custom path")
        p    = Path(_clean_path(raw)).expanduser()
        free = _disk_free_gb(str(p.anchor))
        console.print(f"  [dim]Free on that volume: {free:.1f} GB[/dim]")
        return str(p)


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE PROBE  (enhanced: bit depth, HDR type, DoVi, audio)
# ══════════════════════════════════════════════════════════════════════════════

def probe_source(tools: ToolPaths, path: str) -> SourceInfo:
    """Full JSON probe for maximum detail including DoVi, HDR10+, audio."""
    r = subprocess.run(
        [tools.ffprobe, "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True, errors="replace",
    )

    # ── Parse JSON ────────────────────────────────────────────────────────
    vs: Dict[str, Any] = {}
    as_: Dict[str, Any] = {}
    fmt: Dict[str, Any] = {}
    try:
        data    = json.loads(r.stdout)
        streams = data.get("streams", [])
        fmt     = data.get("format", {})
        vids    = [s for s in streams if s.get("codec_type") == "video"]
        auds    = [s for s in streams if s.get("codec_type") == "audio"]
        vs      = vids[0] if vids else {}
        as_     = auds[0] if auds else {}
    except Exception:
        pass

    width   = int(vs.get("width",  0) or 0)
    height  = int(vs.get("height", 0) or 0)
    codec   = vs.get("codec_name", "unknown")
    profile = vs.get("profile",    "")
    color_transfer  = vs.get("color_transfer",  "unknown")
    color_primaries = vs.get("color_primaries", "unknown")
    pix_fmt = vs.get("pix_fmt", "unknown")

    # ── FPS ───────────────────────────────────────────────────────────────
    fps = 24.0
    for key in ("r_frame_rate", "avg_frame_rate"):
        val = vs.get(key, "")
        if val and "/" in val:
            try:
                n, d = val.split("/")
                n, d = float(n), float(d)
                if d > 0 and n > 0:
                    fps = n / d
                    break
            except Exception:
                pass

    # ── Bitrate ───────────────────────────────────────────────────────────
    bitrate_mbps = 0.0
    for br_src in (vs.get("bit_rate"), fmt.get("bit_rate")):
        try:
            bitrate_mbps = int(br_src) / 1_000_000
            if bitrate_mbps > 0:
                break
        except Exception:
            pass

    # ── Bit depth ─────────────────────────────────────────────────────────
    bit_depth = int(vs.get("bits_per_raw_sample", 0) or 0)
    if bit_depth == 0:
        for marker, depth in (("12", 12), ("10", 10), ("8", 8)):
            if marker in pix_fmt:
                bit_depth = depth
                break
        else:
            bit_depth = 8

    # ── DoVi detection ────────────────────────────────────────────────────
    dovi_profile: Optional[int] = None
    side_data: List[Dict] = vs.get("side_data_list", [])
    for sd in side_data:
        sdt = sd.get("side_data_type", "").lower()
        if "dovi" in sdt or "dolby vision" in sdt:
            dovi_profile = sd.get("dv_profile", -1)
            break
    # Fallback: codec tag (dvh1 / dvhe = DoVi HEVC)
    if dovi_profile is None:
        ctag = vs.get("codec_tag_string", "").lower()
        if ctag in ("dvh1", "dvhe", "dva1", "dvav"):
            dovi_profile = -1

    # ── HDR10+ detection ──────────────────────────────────────────────────
    hdr10plus = any(
        "hdr10+" in sd.get("side_data_type", "").lower() or
        "smpte2094" in sd.get("side_data_type", "").lower()
        for sd in side_data
    )

    # ── HDR type ─────────────────────────────────────────────────────────
    ct = color_transfer.lower()
    cp = color_primaries.lower()
    if dovi_profile is not None:
        p_str  = str(dovi_profile) if dovi_profile and dovi_profile > 0 else "?"
        hdr_type = f"Dolby Vision (Profile {p_str})"
    elif hdr10plus:
        hdr_type = "HDR10+"
    elif "smpte2084" in ct or "bt2020" in cp:
        hdr_type = "HDR10"
    elif "arib-std-b67" in ct:
        hdr_type = "HLG"
    else:
        hdr_type = "SDR"

    is_hdr = hdr_type != "SDR"

    # ── Audio ─────────────────────────────────────────────────────────────
    a_codec   = as_.get("codec_name", "unknown")
    a_ch      = int(as_.get("channels", 0) or 0)
    a_sr      = int(as_.get("sample_rate", 0) or 0)
    a_lang    = as_.get("tags", {}).get("language", "")
    try:
        a_br_kbps = int(as_.get("bit_rate", 0) or 0) / 1000
    except Exception:
        a_br_kbps = 0.0

    return SourceInfo(
        path=path, width=width, height=height,
        fps=fps, codec=codec, profile=profile,
        bitrate_mbps=bitrate_mbps,
        is_hdr=is_hdr, color_transfer=color_transfer,
        hdr_type=hdr_type, dovi_profile=dovi_profile,
        bit_depth=bit_depth, pix_fmt=pix_fmt,
        audio_codec=a_codec, audio_channels=a_ch,
        audio_sample_rate=a_sr, audio_bitrate_kbps=a_br_kbps,
        audio_language=a_lang,
    )


def display_source_info(source: SourceInfo):
    """Print rich source description with HDR/DoVi/audio detail and warnings."""
    console.print()
    console.rule("[bold cyan]Source Analysis[/bold cyan]")
    console.print()

    # Video
    console.print("  [bold]Video[/bold]")
    console.print(f"    Resolution   {source.width}×{source.height}")
    console.print(f"    Codec        {esc(source.codec.upper())}  {esc(source.profile)}")
    console.print(f"    Bit depth    {source.bit_depth}-bit  ({esc(source.pix_fmt)})")
    console.print(f"    Frame rate   {source.fps:.3f} fps")
    console.print(f"    Bitrate      {source.bitrate_mbps:.1f} Mbps")

    hdr_style = "bold magenta" if source.is_hdr else "dim"
    console.print(f"    HDR          [{hdr_style}]{esc(source.hdr_type)}[/{hdr_style}]")
    if source.is_hdr:
        console.print(f"    Transfer     {esc(source.color_transfer)}")

    # Audio
    a_display = _AUDIO_NAMES.get(source.audio_codec, source.audio_codec.upper())
    a_ch_name = _CH_NAMES.get(source.audio_channels, f"{source.audio_channels}ch")
    console.print()
    console.print("  [bold]Audio[/bold]")
    if source.audio_codec and source.audio_codec != "unknown":
        console.print(f"    Codec        {esc(a_display)}")
        console.print(f"    Channels     {a_ch_name}")
        if source.audio_sample_rate:
            console.print(f"    Sample rate  {source.audio_sample_rate} Hz")
        if source.audio_bitrate_kbps > 0:
            console.print(f"    Bitrate      {source.audio_bitrate_kbps:.0f} kbps")
        if source.audio_language:
            console.print(f"    Language     {esc(source.audio_language)}")
    else:
        console.print("    [dim]No audio track detected[/dim]")

    # Warnings
    console.print()
    warnings: List[Tuple[str, str, str]] = []   # (title, detail, severity)

    if source.dovi_profile is not None:
        p = source.dovi_profile
        if p == 5:
            warnings.append((
                "Dolby Vision Profile 5 — Full Enhancement Layer (FEL)",
                "FEL requires dual-layer decoding. Most hardware decoders and ffmpeg "
                "builds will strip the EL and process the BL only. Tone-mapping "
                "accuracy may be significantly reduced.",
                "red",
            ))
        elif p in (7,):
            warnings.append((
                "Dolby Vision Profile 7 — Minimal Enhancement Layer (MEL)",
                "MEL is supported by most hardware decoders. ffmpeg will process "
                "the base HEVC layer; the enhancement layer metadata is used where supported.",
                "yellow",
            ))
        elif p == 8:
            warnings.append((
                "Dolby Vision Profile 8 — Cross-Compatible (base HEVC)",
                "Profile 8 is the most compatible profile. The base layer is "
                "standard HDR10-compatible HEVC. Tone mapping works correctly.",
                "yellow",
            ))
        else:
            warnings.append((
                f"Dolby Vision Profile {p if p and p > 0 else '?'} detected",
                "Verify your Jellyfin ffmpeg build supports this DoVi profile. "
                "Tone mapping behaviour may differ from standard HDR10.",
                "yellow",
            ))

        warnings.append((
            "Dolby Vision — encoder re-mux note",
            "Transcoding DoVi will strip RPU metadata. Output will be HDR10 or SDR, "
            "not Dolby Vision.",
            "dim",
        ))

    for title, detail, sev in warnings:
        console.print(f"  [bold {sev}]⚠  {title}[/bold {sev}]")
        console.print(f"     [dim]{detail}[/dim]")
        console.print()


# ══════════════════════════════════════════════════════════════════════════════
# FFMPEG COMMAND BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_command(
    tools:            ToolPaths,
    hw:               HardwarePlatform,
    scenario:         Scenario,
    source:           SourceInfo,
    stream_id:        int,
    cache_dir:        str,
    cfg:              TestConfig,
    use_hw_decode:    bool = True,
    use_hw_encode:    bool = True,
) -> Tuple[List[str], str, bool, bool]:
    """Returns (cmd, log_path, enc_hw_actual, dec_hw_actual)."""
    dec_hw = use_hw_decode and bool(hw.hwaccel) and (source.codec in hw.hw_decode_codecs)
    enc_hw = use_hw_encode and bool(hw.hwaccel)

    encoder = (hw.hevc_encoder if scenario.target_codec == "hevc" else hw.h264_encoder) \
              if enc_hw else \
              ("libx265" if scenario.target_codec == "hevc" else "libx264")

    bitrate_n = int(scenario.bitrate.rstrip("M"))
    bufsize   = f"{bitrate_n * 2}M"

    # Preserve aspect ratio with explicit even height
    if source.height and source.width:
        ar = source.width / source.height
        th = int(round(scenario.width / ar))
        th += th % 2
    else:
        th = 1080

    scale_f = (f"{hw.scale_filter}=w={scenario.width}:h={th}"
               if (dec_hw and enc_hw)
               else f"scale=w={scenario.width}:h={th}")

    if scenario.tonemap:
        if dec_hw and enc_hw and hw.tonemap_filter:
            tm = hw.tonemap_filter
        elif dec_hw:
            tm = _SW_TM_FROM_HW
        else:
            tm = _SW_TM_PURE
        vf = f"{scale_f},{tm}"
    else:
        vf = scale_f

    seg_path = os.path.join(cache_dir, f"st_{stream_id}_%d.ts")
    playlist = os.path.join(cache_dir, f"st_{stream_id}.m3u8")
    log_path = os.path.join(cache_dir, f"st_{stream_id}.log")

    cmd: List[str] = [
        tools.ffmpeg, "-hide_banner", "-loglevel", "error",
        "-progress", "pipe:1",
    ]
    if dec_hw:
        cmd += ["-hwaccel", hw.hwaccel, "-hwaccel_output_format", hw.hwaccel_output]

    cmd += ["-i", source.path, "-map", "0:v:0", "-map", "0:a:0?", "-vf", vf]
    cmd += ["-c:v", encoder, "-b:v", scenario.bitrate, "-maxrate", scenario.bitrate,
            "-bufsize", bufsize, "-g", "48"]
    cmd += hw.extra_encode_flags if enc_hw else ["-preset", "fast"]
    cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += [
        "-f", "hls",
        "-hls_time", str(cfg.hls_segment_secs),
        "-hls_list_size", str(cfg.hls_list_size),
        "-hls_flags", "delete_segments",
        "-hls_segment_filename", seg_path,
        playlist,
    ]
    return cmd, log_path, enc_hw, dec_hw


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS LOG PARSER
# ══════════════════════════════════════════════════════════════════════════════

_PROGRESS_KEYS = {"speed", "fps", "frame", "bitrate"}

def parse_log(log_path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    try:
        with open(log_path, "r", errors="replace") as fh:
            lines = fh.readlines()
        for line in reversed(lines[-80:]):
            if "=" in line:
                k, _, v = line.partition("=")
                k = k.strip()
                if k in _PROGRESS_KEYS and k not in data:
                    data[k] = v.strip()
            if len(data) == len(_PROGRESS_KEYS):
                break
    except Exception:
        pass
    return data


def _tail_log_error(log_path: str, lines: int = 10) -> str:
    """Read the last N lines of an ffmpeg log to surface error messages."""
    try:
        with open(log_path, "r", errors="replace") as fh:
            all_lines = fh.readlines()
        tail = all_lines[-lines:] if len(all_lines) >= lines else all_lines
        # Filter to lines that look like errors (not progress key=value)
        err_lines = [ln.strip() for ln in tail if "=" not in ln or "Error" in ln or "error" in ln.lower()]
        return "\n".join(err_lines[-5:]) if err_lines else "\n".join(ln.strip() for ln in tail[-3:])
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# I/O MONITOR  (write throughput to cache dir; estimated read rate)
# ══════════════════════════════════════════════════════════════════════════════

class IOMonitor:
    """Tracks bytes written to cache_dir per second (actual) and estimates reads."""
    def __init__(self, cache_dir: str, source: SourceInfo):
        self._cache       = Path(cache_dir)
        self._source      = source
        self._lock        = threading.Lock()
        self._file_bytes: Dict[str, int] = {}
        self._write_mbs   = 0.0
        self._read_mbs    = 0.0
        self._total_mb    = 0.0
        self._running     = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while self._running:
            time.sleep(1.0)
            self._tick()

    def _tick(self):
        delta = 0
        new_sizes: Dict[str, int] = {}
        try:
            for f in self._cache.iterdir():
                if not f.is_file():
                    continue
                try:
                    sz = f.stat().st_size
                except Exception:
                    continue
                new_sizes[f.name] = sz
                prev = self._file_bytes.get(f.name, 0)
                if sz > prev:
                    delta += sz - prev
        except Exception:
            pass
        with self._lock:
            self._file_bytes  = new_sizes
            self._write_mbs   = delta / 1024 ** 2
            self._total_mb   += delta / 1024 ** 2

    def update_read(self, active_speed_sum: float):
        """Call from dashboard render with combined speed across running streams."""
        with self._lock:
            self._read_mbs = active_speed_sum * self._source.bitrate_mbps

    @property
    def write_mbs(self) -> float:
        with self._lock:
            return self._write_mbs

    @property
    def read_mbs(self) -> float:
        with self._lock:
            return self._read_mbs

    @property
    def total_written_mb(self) -> float:
        with self._lock:
            return self._total_mb


# ══════════════════════════════════════════════════════════════════════════════
# STREAM MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class StreamManager:
    def __init__(
        self,
        tools:     ToolPaths,
        hw:        HardwarePlatform,
        scenario:  Scenario,
        source:    SourceInfo,
        cache_dir: str,
        cfg:       TestConfig,
        mixed_pool: Optional[List[Scenario]] = None,
    ):
        self.tools      = tools
        self.hw         = hw
        self.scenario   = scenario
        self.source     = source
        self.cache_dir  = cache_dir
        self.cfg        = cfg
        self.mixed_pool = mixed_pool   # if set, pick random scenario per launch

        self._procs:     Dict[int, subprocess.Popen] = {}
        self._log_paths: Dict[int, str]              = {}
        self._handles:   Dict[int, Any]              = {}
        self.stats:      Dict[int, StreamStats]      = {}
        self._lock       = threading.Lock()
        self._next_id    = 1

    def launch(
        self,
        use_hw_decode:    bool = True,
        use_hw_encode:    bool = True,
        scenario_override: Optional[Scenario] = None,
    ) -> int:
        sid = self._next_id
        self._next_id += 1

        scn = (scenario_override
               or (random.choice(self.mixed_pool) if self.mixed_pool else None)
               or self.scenario)

        cmd, log_path, enc_hw, dec_hw = build_command(
            self.tools, self.hw, scn, self.source,
            sid, self.cache_dir, self.cfg,
            use_hw_decode, use_hw_encode,
        )
        fh   = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=fh)

        enc_tag  = "HW" if enc_hw else "SW"
        label    = f"{scn.target_codec.upper()}-{scn.width}p [{enc_tag}]"

        with self._lock:
            self._procs[sid]     = proc
            self._log_paths[sid] = log_path
            self._handles[sid]   = fh
            self.stats[sid]      = StreamStats(
                stream_id=sid, label=label, pid=proc.pid,
                enc_hw=enc_hw, dec_hw=dec_hw, scenario_id=scn.id,
                started_at=time.time(),
            )
        return sid

    def refresh(self):
        with self._lock:
            for sid, proc in list(self._procs.items()):
                s   = self.stats[sid]
                ret = proc.poll()
                if ret is not None:
                    if s.status not in ("error", "done"):
                        s.ended_at = time.time()
                    if ret != 0:
                        s.status = "error"
                        if not s.error_msg:
                            s.error_msg = _tail_log_error(self._log_paths[sid])
                    else:
                        s.status = "done"
                    continue
                raw      = parse_log(self._log_paths[sid])
                s.status = "running"
                try:
                    s.speed = float(raw.get("speed", "0").rstrip("x"))
                except ValueError:
                    s.speed = 0.0
                try:
                    s.fps = float(raw.get("fps", "0"))
                except ValueError:
                    s.fps = 0.0
                if s.fps > 0 and s.frames > 10:
                    s.fps_history.append(s.fps)
                s.bitrate_str = raw.get("bitrate", "...")
                try:
                    s.frames = int(raw.get("frame", "0"))
                except ValueError:
                    s.frames = 0

    def kill_all(self):
        with self._lock:
            for proc in self._procs.values():
                try:
                    proc.terminate()
                except Exception:
                    pass
            time.sleep(_KILL_GRACE)
            for proc in self._procs.values():
                try:
                    if proc.poll() is None:
                        proc.kill()
                except Exception:
                    pass
            for fh in self._handles.values():
                try:
                    fh.close()
                except Exception:
                    pass

    @property
    def count(self) -> int:
        return len(self._procs)

    @property
    def active_count(self) -> int:
        return sum(1 for s in self.stats.values() if s.status == "running")

    def combined_speed(self) -> float:
        return sum(s.speed for s in self.stats.values() if s.status == "running")


# ══════════════════════════════════════════════════════════════════════════════
# ESCALATING DIFFICULTY CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class EscalatingController:
    def __init__(self, cfg: TestConfig, source_fps: float):
        self.threshold     = source_fps * cfg.fps_ratio
        self.ramp_interval = cfg.ramp_interval
        self.fail_secs     = cfg.fail_secs
        self.fps_ratio     = cfg.fps_ratio
        self.warmup_secs   = cfg.warmup_secs
        self.source_fps    = source_fps

        self._phase:       str              = "warmup"
        self._phase_start: float            = time.time()
        self._below_since: Dict[int, float] = {}

        self.max_stable:     int = 0
        self.failure_reason: str = ""
        self.failed_stream:  Optional[int] = None

    def reset_warmup(self):
        self._phase         = "warmup"
        self._phase_start   = time.time()
        self._below_since.clear()
        self.failure_reason = ""
        self.failed_stream  = None

    def tick(self, stats: Dict[int, StreamStats]) -> str:
        """Returns 'wait' | 'add' | 'fail'."""
        now    = time.time()
        active = {sid: s for sid, s in stats.items()
                  if s.status == "running" and s.frames > 10}
        if not active:
            return "wait"

        if self._phase == "warmup":
            if now - self._phase_start >= self.warmup_secs:
                self._phase = "stable"
                self._phase_start = now
                self._below_since.clear()
            return "wait"

        for sid, s in active.items():
            if s.fps < self.threshold:
                if sid not in self._below_since:
                    self._below_since[sid] = now
                elif now - self._below_since[sid] >= self.fail_secs:
                    self._phase         = "failed"
                    self.failed_stream  = sid
                    self.failure_reason = (
                        f"Stream #{sid} dropped to {s.fps:.1f} fps "
                        f"(threshold {self.threshold:.1f} fps) "
                        f"for {self.fail_secs:.0f}s"
                    )
                    return "fail"
            else:
                self._below_since.pop(sid, None)

        all_stable = all(s.fps >= self.threshold for s in active.values())
        if all_stable:
            if now - self._phase_start >= self.ramp_interval:
                self.max_stable   = max(self.max_stable, len(active))
                self._phase       = "warmup"
                self._phase_start = now
                return "add"
        else:
            self._phase_start = now
        return "wait"

    @property
    def phase(self) -> str:
        return self._phase

    def next_ramp_in(self) -> float:
        return max(0.0, self.ramp_interval - (time.time() - self._phase_start))

    def warmup_remaining(self) -> float:
        return max(0.0, self.warmup_secs - (time.time() - self._phase_start))


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def render_dashboard(
    hw:            HardwarePlatform,
    source:        SourceInfo,
    scenario:      Scenario,
    manager:       StreamManager,
    escalator:     Optional[EscalatingController],
    io_mon:        Optional[IOMonitor],
    elapsed:       float,
    events:        List[str],
    test_mode:     str  = MODE_FIXED,
    use_hw_decode: bool = True,
    hw_saturated:  bool = False,
    continuous:    bool = False,
) -> Panel:

    # ── Info lines ────────────────────────────────────────────────────────
    gpu_line = Text()
    gpu_line.append("  GPU     ", style="dim")
    gpu_line.append(hw.gpu_name, style="bold cyan")
    gpu_line.append(f"  [{hw.name}]", style="dim")

    src_line = Text()
    src_line.append("  SOURCE  ", style="dim")
    src_line.append(f"{source.width}×{source.height}", style="bold white")
    src_line.append(f"  {source.codec.upper()}", style="bold yellow")
    if source.bit_depth:
        src_line.append(f"  {source.bit_depth}-bit", style="dim")
    if source.is_hdr:
        src_line.append(f"  {source.hdr_type}", style="bold magenta")
    src_line.append(f"  {source.fps:.3f} fps  {source.bitrate_mbps:.1f} Mbps", style="dim")

    scn_line = Text()
    scn_line.append("  SCENARIO ", style="dim")
    scn_line.append(scenario.label, style="bold green")
    scn_line.append(f"  {scenario.bitrate}", style="dim")
    if scenario.tonemap:
        scn_line.append("  Tone Map", style="magenta")
    if test_mode == MODE_MIXED:
        scn_line.append("  [Mixed Bag]", style="bold cyan")

    # ── Encode/Decode indicator ───────────────────────────────────────────
    ed_line = Text()
    ed_line.append("  ENCODE  ", style="dim")
    if hw.hwaccel:
        ed_line.append("HW ", style="bold green")
        ed_line.append(hw.name, style="dim")
    else:
        ed_line.append("SW ", style="bold yellow")
        ed_line.append("libx264/libx265", style="dim")
    ed_line.append("   DECODE  ", style="dim")
    if use_hw_decode and hw.hw_decode_codecs and source.codec in hw.hw_decode_codecs:
        ed_line.append("HW ", style="bold green")
        ed_line.append(f"{source.codec.upper()} via {hw.short}", style="dim")
    else:
        ed_line.append("SW ", style="bold yellow")
        ed_line.append("software", style="dim")
    if hw_saturated:
        ed_line.append("   [bold red]HW SATURATED — new streams: SW[/bold red]")
    if continuous:
        ed_line.append("   [bold yellow]RUNNING CONTINUOUSLY[/bold yellow]")

    # ── Stream table ──────────────────────────────────────────────────────
    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold dim", expand=True)
    tbl.add_column("  #",    width=5,  justify="right")
    tbl.add_column("Mode",   width=17)
    tbl.add_column("Status", width=9)
    tbl.add_column("FPS",    width=7,  justify="right")
    tbl.add_column("vs src", width=7,  justify="right")
    tbl.add_column("Speed",  width=7,  justify="right")
    tbl.add_column("Bitrate",width=13, justify="right")
    tbl.add_column("Frames", width=9,  justify="right")
    tbl.add_column("ENC",    width=5,  justify="center")
    tbl.add_column("DEC",    width=5,  justify="center")

    threshold   = escalator.threshold if escalator else 0.0
    total_speed = 0.0

    for sid in sorted(manager.stats):
        s = manager.stats[sid]
        total_speed += s.speed

        status_m = {"starting": "[yellow]start[/yellow]",
                    "running":  "[green]run[/green]",
                    "error":    "[bold red]error[/bold red]",
                    "done":     "[dim]done[/dim]"}.get(s.status, s.status)

        if s.status == "running" and threshold > 0 and s.frames > 10:
            ratio = s.fps / threshold
            fps_style = ("bold red" if ratio < 1.0 else
                         "yellow"   if ratio < 1.1 else "green")
        else:
            fps_style = "dim"
        fps_cell = f"[{fps_style}]{s.fps:.1f}[/{fps_style}]"

        if source.fps > 0 and s.frames > 10 and s.status == "running":
            pct     = (s.fps / source.fps) * 100
            ps      = "green" if pct >= 90 else ("yellow" if pct >= 70 else "red")
            vs_cell = f"[{ps}]{pct:.0f}%[/{ps}]"
        else:
            vs_cell = "[dim]—[/dim]"

        enc_cell = "[bold green]HW[/bold green]" if s.enc_hw else "[yellow]SW[/yellow]"
        dec_cell = "[bold green]HW[/bold green]" if s.dec_hw else "[yellow]SW[/yellow]"

        tbl.add_row(f"#{sid}", s.label, status_m, fps_cell, vs_cell,
                    f"{s.speed:.2f}×", s.bitrate_str, str(s.frames),
                    enc_cell, dec_cell)

    tbl.add_section()
    tbl.add_row("", "[dim]COMBINED[/dim]", "", "", "",
                f"[bold]{total_speed:.2f}×[/bold]", "", "", "", "")

    # ── I/O bar ───────────────────────────────────────────────────────────
    io_lines: List[Text] = []
    if io_mon:
        io_mon.update_read(total_speed)
        w = io_mon.write_mbs
        r = io_mon.read_mbs
        total = io_mon.total_written_mb
        t = Text()
        t.append("  I/O  ", style="dim")
        t.append(f"Write {w:.1f} MB/s", style="yellow")
        t.append("   ", style="dim")
        t.append(f"Read ~{r:.1f} MB/s (est)", style="cyan")
        t.append(f"   Total written {total:.0f} MB", style="dim")
        io_lines.append(t)

    # ── Escalator bar ─────────────────────────────────────────────────────
    esc_lines: List[Text] = []
    if escalator and test_mode in (MODE_ESCALATING, MODE_HYBRID, MODE_MIXED) and not continuous:
        BAR = 30
        ph  = escalator.phase
        if ph == "warmup":
            rem    = escalator.warmup_remaining()
            filled = int(max(0.0, 1.0 - rem / escalator.warmup_secs) * BAR)
            t = Text()
            t.append("  ⟳  Warming up   ", style="yellow")
            t.append("█" * filled, style="yellow")
            t.append("░" * (BAR - filled), style="dim")
            t.append(f"  {rem:.0f}s remaining", style="yellow")
            esc_lines.append(t)
        elif ph == "stable":
            rem    = escalator.next_ramp_in()
            filled = int(max(0.0, 1.0 - rem / escalator.ramp_interval) * BAR)
            t = Text()
            t.append("  ✓  Stable       ", style="green")
            t.append("█" * filled, style="green")
            t.append("░" * (BAR - filled), style="dim")
            t.append(f"  +1 stream in {rem:.0f}s", style="green")
            esc_lines.append(t)
        elif ph == "failed":
            t = Text()
            t.append("  ✗  LIMIT REACHED", style="bold red")
            if hw_saturated:
                t.append("  (HW saturated, adding SW streams)", style="red")
            esc_lines.append(t)
        t2 = Text()
        t2.append(
            f"  Threshold {escalator.threshold:.1f} fps "
            f"({escalator.fps_ratio*100:.0f}% of {source.fps:.1f} fps)  │  "
            f"Max stable: {escalator.max_stable}",
            style="dim",
        )
        esc_lines.append(t2)

    # ── Event log — all event strings are escaped ─────────────────────────
    event_lines = [
        Text.assemble(("  ", "dim"), (esc(e), "dim"))
        for e in events[-6:]
    ]

    # ── Footer ────────────────────────────────────────────────────────────
    mode_labels = {
        MODE_FIXED: "Fixed", MODE_ESCALATING: "Escalating",
        MODE_HYBRID: "Hybrid HW→SW", MODE_MIXED: "Mixed Bag",
    }
    footer = Text()
    footer.append(f"  {manager.active_count}", style="bold")
    footer.append(" active  /  ", style="dim")
    footer.append(f"{manager.count}", style="bold")
    footer.append(" total  │  ", style="dim")
    footer.append(mode_labels.get(test_mode, test_mode), style="cyan")
    footer.append("  │  Elapsed  ", style="dim")
    footer.append(_fmt_elapsed(elapsed), style="bold")
    footer.append("  │  Ctrl+C to stop", style="dim")

    body = Group(
        Text(""),
        gpu_line, src_line, scn_line, ed_line,
        Text(""),
        tbl,
        *io_lines,
        *esc_lines,
        Text(""),
        *event_lines,
        Text(""),
        footer,
    )

    return Panel(
        body,
        title=f"[bold white]{APP_TITLE}  v{VERSION}[/bold white]",
        border_style="bright_blue",
        padding=(0, 1),
    )


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(
    hw:            HardwarePlatform,
    tools:         ToolPaths,
    source:        SourceInfo,
    scenario:      Scenario,
    manager:       StreamManager,
    escalator:     Optional[EscalatingController],
    io_mon:        Optional[IOMonitor],
    cfg:           TestConfig,
    test_mode:     str,
    use_hw_decode: bool,
    hw_saturated:  bool,
    total_elapsed: float,
    cache_dir:     str,
) -> str:
    ts       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname    = f"jf_stress_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    out_path = Path.cwd() / fname

    mode_labels = {
        MODE_FIXED: "Fixed Streams", MODE_ESCALATING: "Escalating",
        MODE_HYBRID: "Hybrid HW→SW", MODE_MIXED: "Mixed Bag",
    }
    a_display = _AUDIO_NAMES.get(source.audio_codec, source.audio_codec.upper())
    a_ch_name = _CH_NAMES.get(source.audio_channels, f"{source.audio_channels}ch")
    enc_label = "Hardware" if hw.hwaccel else "Software"
    dec_label = ("Hardware" if use_hw_decode and source.codec in hw.hw_decode_codecs
                 else "Software")

    _h = html_module.escape  # safe HTML escaping for user-provided values

    # ── Stream rows ──────────────────────────────────────────────────────
    stream_rows = ""
    for sid in sorted(manager.stats):
        s = manager.stats[sid]
        fps_class  = ("fps-good" if s.fps >= (escalator.threshold if escalator else 0)
                      else "fps-bad") if s.status == "running" else ""
        enc_badge  = '<span class="badge hw">HW</span>' if s.enc_hw else '<span class="badge sw">SW</span>'
        dec_badge  = '<span class="badge hw">HW</span>' if s.dec_hw else '<span class="badge sw">SW</span>'
        stat_badge = {'running':'<span class="badge run">run</span>',
                      'error':  '<span class="badge err">error</span>',
                      'done':   '<span class="badge done">done</span>',
                      'starting':'<span class="badge start">start</span>'}.get(s.status, s.status)
        stream_rows += (
            f"<tr>"
            f"<td>#{sid}</td><td>{_h(s.label)}</td><td>{stat_badge}</td>"
            f'<td class="{fps_class}">{s.fps:.1f}</td>'
            f"<td>{s.speed:.2f}×</td><td>{_h(s.bitrate_str)}</td>"
            f"<td>{s.frames:,}</td><td>{enc_badge}</td><td>{dec_badge}</td>"
            f"</tr>\n"
        )

    # ── Escalator block ───────────────────────────────────────────────────
    esc_html = ""
    if escalator:
        sat_note = ""
        if hw_saturated:
            sat_note = '<p class="warn">⚠ HW engines reached saturation — SW streams were added after HW limit.</p>'
        fail_note = ""
        if escalator.failure_reason:
            fail_note = f'<p class="err-note">Final failure: {escalator.failure_reason}</p>'
        esc_html = f"""
        <div class="section">
          <h2>Escalating Results</h2>
          <div class="kv-grid">
            <div class="kv"><span>Max stable concurrent streams</span><strong>{escalator.max_stable}</strong></div>
            <div class="kv"><span>FPS threshold</span><strong>{escalator.threshold:.1f} fps
              ({escalator.fps_ratio*100:.0f}% of {source.fps:.3f} fps)</strong></div>
            <div class="kv"><span>HW saturation reached</span><strong>{'Yes' if hw_saturated else 'No'}</strong></div>
          </div>
          {sat_note}{fail_note}
        </div>"""

    # ── I/O block ─────────────────────────────────────────────────────────
    io_html = ""
    if io_mon:
        io_html = f"""
        <div class="section">
          <h2>I / O</h2>
          <div class="kv-grid">
            <div class="kv"><span>Total written to cache</span><strong>{io_mon.total_written_mb:.0f} MB</strong></div>
            <div class="kv"><span>Write rate (last tick)</span><strong>{io_mon.write_mbs:.1f} MB/s</strong></div>
            <div class="kv"><span>Est. read rate</span><strong>{io_mon.read_mbs:.1f} MB/s</strong></div>
            <div class="kv"><span>Cache directory</span><code>{_h(cache_dir)}</code></div>
          </div>
        </div>"""

    # ── DoVi note ─────────────────────────────────────────────────────────
    dovi_html = ""
    if source.dovi_profile is not None:
        p = source.dovi_profile
        label = f"Profile {p}" if p and p > 0 else "Profile unknown"
        dovi_html = f'<span class="badge hdr">Dolby Vision {label}</span>'

    hdr_badge = (f'<span class="badge hdr">{source.hdr_type}</span>'
                 if source.is_hdr else '<span class="badge sdr">SDR</span>')

    # ── Ramp config table (only for escalating modes) ─────────────────────
    ramp_rows = ""
    if test_mode != MODE_FIXED:
        ramp_rows = f"""
            <div class="kv"><span>Ramp interval</span><strong>{cfg.ramp_interval:.0f}s</strong></div>
            <div class="kv"><span>Fail threshold</span><strong>{cfg.fail_secs:.0f}s below
              {cfg.fps_ratio*100:.0f}% of source FPS</strong></div>"""

    # ── Full HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Jellyfin Stress Test Report — {ts}</title>
<style>
  :root {{
    --bg:       #0d1117;
    --surface:  #161b22;
    --border:   #30363d;
    --text:     #e6edf3;
    --muted:    #7d8590;
    --accent:   #58a6ff;
    --green:    #3fb950;
    --yellow:   #d29922;
    --red:      #f85149;
    --purple:   #bc8cff;
    --cyan:     #39d353;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px; line-height: 1.6; padding: 32px 24px;
  }}
  .header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 24px; margin-bottom: 32px;
  }}
  .header h1 {{
    font-size: 22px; font-weight: 700; color: var(--accent);
    letter-spacing: -0.3px;
  }}
  .header .meta {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}
  .section {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;
  }}
  .section h2 {{
    font-size: 13px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.8px; color: var(--muted);
    margin-bottom: 16px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .kv-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 10px 24px;
  }}
  .kv {{ display: flex; flex-direction: column; gap: 2px; }}
  .kv span {{ font-size: 11px; color: var(--muted); text-transform: uppercase;
              letter-spacing: 0.5px; }}
  .kv strong {{ font-size: 14px; color: var(--text); }}
  .kv code {{
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px; color: var(--accent);
    word-break: break-all;
  }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 4px;
  }}
  thead tr {{ border-bottom: 1px solid var(--border); }}
  th {{
    text-align: left; padding: 6px 10px;
    font-size: 11px; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.5px;
  }}
  td {{ padding: 7px 10px; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(88,166,255,0.04); }}
  .fps-good {{ color: var(--green); font-weight: 600; }}
  .fps-bad  {{ color: var(--red);   font-weight: 600; }}
  .badge {{
    display: inline-block; padding: 2px 7px; border-radius: 4px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.3px;
  }}
  .badge.hw   {{ background: rgba(63,185,80,0.18);  color: var(--green); }}
  .badge.sw   {{ background: rgba(210,153,34,0.18); color: var(--yellow); }}
  .badge.hdr  {{ background: rgba(188,140,255,0.18);color: var(--purple); }}
  .badge.sdr  {{ background: rgba(125,133,144,0.14);color: var(--muted); }}
  .badge.run  {{ background: rgba(63,185,80,0.15);  color: var(--green); }}
  .badge.err  {{ background: rgba(248,81,73,0.15);  color: var(--red); }}
  .badge.done {{ background: rgba(125,133,144,0.14);color: var(--muted); }}
  .badge.start{{ background: rgba(210,153,34,0.14); color: var(--yellow); }}
  .result-highlight {{
    background: rgba(63,185,80,0.08); border: 1px solid rgba(63,185,80,0.3);
    border-radius: 8px; padding: 14px 20px; margin-bottom: 16px;
    display: flex; align-items: baseline; gap: 12px;
  }}
  .result-highlight .num {{
    font-size: 40px; font-weight: 700; color: var(--green); line-height: 1;
  }}
  .result-highlight .label {{ color: var(--muted); font-size: 13px; }}
  .warn {{ color: var(--yellow); font-size: 13px; margin-top: 10px; }}
  .err-note {{ color: var(--red); font-size: 13px; margin-top: 6px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 700px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<div class="header">
  <h1>Jellyfin Transcode Stress Test</h1>
  <div class="meta">Generated {ts} &nbsp;·&nbsp; {APP_TITLE} v{VERSION}</div>
</div>

<div class="two-col">

  <div class="section">
    <h2>Hardware</h2>
    <div class="kv-grid">
      <div class="kv"><span>Platform</span><strong>{hw.name}</strong></div>
      <div class="kv"><span>GPU</span><strong>{_h(hw.gpu_name)}</strong></div>
      <div class="kv"><span>ffmpeg</span><code>{_h(tools.ffmpeg)}</code></div>
    </div>
  </div>

  <div class="section">
    <h2>Test Configuration</h2>
    <div class="kv-grid">
      <div class="kv"><span>Mode</span><strong>{mode_labels.get(test_mode, test_mode)}</strong></div>
      <div class="kv"><span>Scenario</span><strong>{scenario.label}</strong></div>
      <div class="kv"><span>Target bitrate</span><strong>{scenario.bitrate}</strong></div>
      <div class="kv"><span>Output codec</span><strong>{scenario.target_codec.upper()}</strong></div>
      <div class="kv"><span>Tone mapping</span><strong>{'Yes' if scenario.tonemap else 'No'}</strong></div>
      <div class="kv"><span>Encode</span><strong>{enc_label}</strong></div>
      <div class="kv"><span>Decode</span><strong>{dec_label}</strong></div>
      <div class="kv"><span>HLS segment</span><strong>{cfg.hls_segment_secs}s</strong></div>
      <div class="kv"><span>Total elapsed</span><strong>{_fmt_elapsed(total_elapsed)}</strong></div>
      {ramp_rows}
    </div>
  </div>

</div>

<div class="section">
  <h2>Source File</h2>
  <div class="kv-grid">
    <div class="kv"><span>Path</span><code>{_h(source.path)}</code></div>
    <div class="kv"><span>Resolution</span><strong>{source.width}×{source.height}</strong></div>
    <div class="kv"><span>Codec</span><strong>{source.codec.upper()}  {source.profile}</strong></div>
    <div class="kv"><span>Bit depth</span><strong>{source.bit_depth}-bit  ({source.pix_fmt})</strong></div>
    <div class="kv"><span>Frame rate</span><strong>{source.fps:.3f} fps</strong></div>
    <div class="kv"><span>Bitrate</span><strong>{source.bitrate_mbps:.1f} Mbps</strong></div>
    <div class="kv"><span>HDR</span><strong>{hdr_badge} {dovi_html}</strong></div>
    <div class="kv"><span>Color transfer</span><strong>{source.color_transfer}</strong></div>
    <div class="kv"><span>Audio codec</span><strong>{a_display}</strong></div>
    <div class="kv"><span>Channels</span><strong>{a_ch_name}</strong></div>
    <div class="kv"><span>Sample rate</span><strong>{source.audio_sample_rate} Hz</strong></div>
    {f'<div class="kv"><span>Audio bitrate</span><strong>{source.audio_bitrate_kbps:.0f} kbps</strong></div>' if source.audio_bitrate_kbps else ''}
    {f'<div class="kv"><span>Language</span><strong>{source.audio_language}</strong></div>' if source.audio_language else ''}
  </div>
</div>

{esc_html}

<div class="section">
  <h2>Stream Summary</h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Label</th><th>Status</th>
        <th>FPS</th><th>Speed</th><th>Bitrate</th>
        <th>Frames</th><th>ENC</th><th>DEC</th>
      </tr>
    </thead>
    <tbody>
{stream_rows}    </tbody>
  </table>
</div>

{io_html}

</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

def run_setup(
    hw: HardwarePlatform, source: SourceInfo
) -> Tuple[Scenario, str, int, bool, TestConfig, Optional[List[Scenario]]]:
    """
    Setup wizard — logical order:
      1. Test mode (Fixed / Escalating / Hybrid / Mixed)
      2. Stream count  (Fixed only)
      3. Escalating parameters  (non-Fixed modes)
      4. Scenario + target output  (Fixed / Escalating / Hybrid)
      5. Codec override
      6. Hardware decode
      7. HLS segment size
    Returns: scenario, test_mode, fixed_streams, use_hw_decode, cfg, mixed_pool
    """

    # ── 1. Test mode ──────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Test Mode[/bold cyan]")
    console.print()
    console.print("  [bold]1[/bold]  Fixed Streams   — run a set number of concurrent streams")
    console.print("  [bold]2[/bold]  Escalating      — auto-ramp HW streams until framerate fails")
    console.print("  [bold]3[/bold]  Hybrid HW→SW    — escalate HW; when saturated, add SW streams")
    console.print(
        "  [bold]4[/bold]  Mixed Bag       — escalating with random codec/resolution/bitrate\n"
        "                    per stream across 4K / 1080p / 720p / 480p "
        "(simulates real Jellyfin load)\n"
    )
    mp = Prompt.ask("  Mode", choices=["1", "2", "3", "4"], default="2")
    test_mode = {
        "1": MODE_FIXED, "2": MODE_ESCALATING,
        "3": MODE_HYBRID, "4": MODE_MIXED,
    }[mp]

    # ── 2. Stream count (Fixed only) ──────────────────────────────────────
    fixed_streams = 1
    if test_mode == MODE_FIXED:
        fixed_streams = IntPrompt.ask("  Number of concurrent streams", default=4)

    # ── 3. Escalating / HLS parameters ────────────────────────────────────
    cfg = TestConfig()

    console.print()
    console.rule("[bold cyan]HLS Segment Size[/bold cyan]")
    console.print()
    console.print(
        "  Controls seconds of video per HLS chunk.\n"
        "  Jellyfin default is [bold]3[/bold]s. Larger values reduce I/O, increase seek latency.\n"
    )
    hls_secs = IntPrompt.ask("  Segment duration (seconds)", default=4)
    cfg.hls_segment_secs = max(1, min(hls_secs, 30))

    if test_mode != MODE_FIXED:
        console.print()
        console.rule("[bold cyan]Escalating Parameters[/bold cyan]")
        console.print("  Press Enter to accept defaults.\n")
        cfg.ramp_interval = FloatPrompt.ask(
            "  Seconds of stability before adding a stream", default=12.0)
        cfg.fail_secs = FloatPrompt.ask(
            "  Seconds below threshold before declaring failure", default=5.0)
        while True:
            cfg.fps_ratio = FloatPrompt.ask(
                "  Minimum FPS ratio to maintain (0.50 – 1.00)", default=0.90)
            if 0.50 <= cfg.fps_ratio <= 1.00:
                break
            console.print("  [red]Please enter a value between 0.50 and 1.00[/red]")

    # ── 4a. Mixed bag pool (no manual scenario selection) ─────────────────
    mixed_pool: Optional[List[Scenario]] = None
    if test_mode == MODE_MIXED:
        pool = list(MIXED_SCENARIOS)
        if source.is_hdr:
            pool += MIXED_HDR_SCENARIOS
        else:
            console.print()
            console.print(
                "  [dim]HDR→SDR variants excluded from mix (source is not HDR).[/dim]"
            )
        mixed_pool = pool
        resolutions = sorted({f"{s.width}p" if s.width < 3840 else "4K" for s in pool})
        console.print(
            f"  [dim]Mix pool: {len(mixed_pool)} scenarios across "
            f"{', '.join(resolutions)}[/dim]"
        )
        # For Mixed mode we still need a "default" scenario for the manager constructor;
        # the pool will override it on every actual launch.
        scenario = MIXED_SCENARIOS[0]

    # ── 4b. Scenario selection (non-Mixed modes) ──────────────────────────
    else:
        console.print()
        console.rule("[bold cyan]Scenario[/bold cyan]")
        console.print()

        tbl = Table(box=box.SIMPLE, header_style="bold dim", show_edge=False)
        tbl.add_column("  #",  width=4)
        tbl.add_column("Label",       width=26)
        tbl.add_column("Description", width=46)
        tbl.add_column("Bitrate",     width=9,  justify="right")
        tbl.add_column("TM",          width=4,  justify="center")
        for i, s in enumerate(SCENARIOS, 1):
            tbl.add_row(f"  {i}", s.label, s.description, s.bitrate,
                        "[magenta]✓[/magenta]" if s.tonemap else "[dim]–[/dim]")
        console.print(tbl)

        choice   = Prompt.ask("  Choose scenario",
                              choices=[str(i) for i in range(1, len(SCENARIOS) + 1)],
                              default="1")
        scenario = SCENARIOS[int(choice) - 1]

        # HDR / DoVi warnings for tone-map scenarios
        if scenario.tonemap:
            if not source.is_hdr:
                console.print()
                console.print(
                    f"  [red]⚠  Source is not HDR (detected: {esc(source.hdr_type)}).[/red]\n"
                    "     Tone mapping has no effect on SDR content.\n"
                    "     Consider a non-tone-mapping scenario instead."
                )
                if not Confirm.ask("  Enable tone mapping anyway?", default=False):
                    scenario = replace(scenario, tonemap=False)
            elif source.dovi_profile is not None:
                console.print()
                p = source.dovi_profile
                if p == 5:
                    console.print(
                        "  [bold red]⚠  Dolby Vision Profile 5 (FEL) — tone mapping "
                        "accuracy reduced.[/bold red]\n"
                        "     Enhancement layer stripped; processing base layer only."
                    )
                else:
                    console.print(
                        "  [yellow]⚠  Dolby Vision source — output will be HDR10/SDR, "
                        "not Dolby Vision.[/yellow]"
                    )

        # ── 5. Output codec override ───────────────────────────────────────
        console.print()
        console.rule("[bold cyan]Output Codec[/bold cyan]")
        console.print()
        console.print(f"  Scenario default: [bold]{scenario.target_codec.upper()}[/bold]\n")
        console.print("  [bold]1[/bold]  Keep scenario default")
        console.print("  [bold]2[/bold]  Force H.264  (broadest client compat)")
        console.print("  [bold]3[/bold]  Force HEVC   (better compression)\n")
        cp = Prompt.ask("  Codec", choices=["1", "2", "3"], default="1")
        if cp == "2":
            scenario = replace(scenario, target_codec="h264")
            console.print("  [dim]→ H.264 output[/dim]")
        elif cp == "3":
            scenario = replace(scenario, target_codec="hevc")
            console.print("  [dim]→ HEVC output[/dim]")

    # ── 6. Hardware decode ────────────────────────────────────────────────
    use_hw_decode = False
    if hw.hw_decode_codecs:
        console.print()
        console.rule("[bold cyan]Hardware Decode[/bold cyan]")
        console.print()
        supported = ", ".join(c.upper() for c in hw.hw_decode_codecs)
        console.print(f"  [{esc(hw.name)}] supports HW decode for: [bold]{esc(supported)}[/bold]\n")
        src_ok = source.codec in hw.hw_decode_codecs
        if src_ok:
            console.print(
                f"  Source codec [bold]{esc(source.codec.upper())}[/bold] is supported. "
                "HW decode recommended.\n"
            )
        else:
            console.print(
                f"  [yellow]⚠  {esc(source.codec.upper())} not in the HW decode list — "
                "will fall back to software decode.[/yellow]\n"
            )
        console.print("  [bold]1[/bold]  Yes — enable hardware decode")
        console.print("  [bold]2[/bold]  No  — software decode + hardware encode\n")
        use_hw_decode = Prompt.ask("  Hardware decode", choices=["1", "2"], default="1") == "1"
    else:
        console.print()
        console.print("  [dim]Hardware decode not available on this platform.[/dim]")

    return scenario, test_mode, fixed_streams, use_hw_decode, cfg, mixed_pool


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP RUNNER  (shared between initial and continuous runs)
# ══════════════════════════════════════════════════════════════════════════════

def run_loop(
    hw:            HardwarePlatform,
    source:        SourceInfo,
    scenario:      Scenario,
    manager:       StreamManager,
    escalator:     Optional[EscalatingController],
    io_mon:        Optional[IOMonitor],
    cfg:           TestConfig,
    test_mode:     str,
    use_hw_decode: bool,
    hw_saturated:  bool,
    events:        List[str],
    start_time:    float,
    continuous:    bool = False,
) -> Tuple[bool, bool, bool]:
    """
    Run the Live dashboard loop.
    Returns (done, hw_saturated, user_stopped).
    done = True means we should stop everything.
    """
    done         = False
    user_stopped = False

    def ts() -> str:
        return _fmt_elapsed(time.time() - start_time)

    def shutdown(sig=None, frame=None):
        nonlocal done, user_stopped
        done         = True
        user_stopped = True

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    with Live(console=console, refresh_per_second=5, screen=True) as live:
        while not done:
            manager.refresh()
            elapsed = time.time() - start_time

            if not continuous and escalator and test_mode in (
                MODE_ESCALATING, MODE_HYBRID, MODE_MIXED
            ):
                action = escalator.tick(manager.stats)

                if action == "add":
                    use_enc = not hw_saturated
                    use_dec = use_hw_decode and not hw_saturated
                    sid     = manager.launch(
                        use_hw_decode=use_dec, use_hw_encode=use_enc
                    )
                    enc_tag = "HW" if use_enc else "SW"
                    events.append(f"{ts()}  ↑ Stream #{sid} [{enc_tag}] ({manager.count} total)")

                elif action == "fail":
                    if test_mode == MODE_HYBRID and not hw_saturated:
                        hw_saturated = True
                        events.append(
                            f"{ts()}  HW engines saturated at {manager.count} streams — "
                            "pivoting to SW encode"
                        )
                        escalator.reset_warmup()
                        sid = manager.launch(use_hw_decode=False, use_hw_encode=False)
                        events.append(f"{ts()}  ↑ Stream #{sid} [SW] ({manager.count} total)")
                    else:
                        if hw_saturated:
                            events.append(f"{ts()}  SW also exhausted — full system limit reached")
                        else:
                            events.append(f"{ts()}  {escalator.failure_reason}")
                        events.append(f"{ts()}  Max stable: {escalator.max_stable} stream(s)")

                        # Show final panel then break out
                        panel = render_dashboard(
                            hw, source, scenario, manager, escalator, io_mon,
                            elapsed, events, test_mode, use_hw_decode, hw_saturated,
                        )
                        live.update(panel)
                        time.sleep(1.0)
                        done = True
                        break

            panel = render_dashboard(
                hw, source, scenario, manager, escalator, io_mon,
                elapsed, events, test_mode, use_hw_decode, hw_saturated, continuous,
            )
            live.update(panel)
            time.sleep(0.2)

    return done, hw_saturated, user_stopped


# ══════════════════════════════════════════════════════════════════════════════
# JSON REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_json_report(
    hw:            HardwarePlatform,
    tools:         ToolPaths,
    source:        SourceInfo,
    scenario:      Scenario,
    manager:       StreamManager,
    escalator:     Optional[EscalatingController],
    io_mon:        Optional[IOMonitor],
    cfg:           TestConfig,
    test_mode:     str,
    use_hw_decode: bool,
    hw_saturated:  bool,
    total_elapsed: float,
    cache_dir:     str,
) -> Dict[str, Any]:
    """Generate a machine-readable JSON report for automated consumption."""
    streams = []
    for sid in sorted(manager.stats):
        s = manager.stats[sid]
        fps_hist = s.fps_history
        stream_data: Dict[str, Any] = {
            "id": sid,
            "label": s.label,
            "scenario_id": s.scenario_id,
            "status": s.status,
            "fps": round(s.fps, 2),
            "speed": round(s.speed, 3),
            "bitrate": s.bitrate_str,
            "frames": s.frames,
            "encode_hw": s.enc_hw,
            "decode_hw": s.dec_hw,
            "started_at": s.started_at,
            "ended_at": s.ended_at if s.ended_at else None,
            "duration_secs": round(s.ended_at - s.started_at, 2) if s.ended_at else None,
        }
        if fps_hist:
            stream_data["fps_stats"] = {
                "min": round(min(fps_hist), 2),
                "max": round(max(fps_hist), 2),
                "avg": round(sum(fps_hist) / len(fps_hist), 2),
                "p50": round(sorted(fps_hist)[len(fps_hist) // 2], 2),
                "p95": round(sorted(fps_hist)[int(len(fps_hist) * 0.95)], 2) if len(fps_hist) >= 20 else None,
                "samples": len(fps_hist),
            }
        if s.error_msg:
            stream_data["error"] = s.error_msg
        streams.append(stream_data)

    mode_labels = {
        MODE_FIXED: "fixed", MODE_ESCALATING: "escalating",
        MODE_HYBRID: "hybrid", MODE_MIXED: "mixed",
    }

    report: Dict[str, Any] = {
        "version": VERSION,
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware": {
            "platform": hw.name,
            "gpu": hw.gpu_name,
            "ffmpeg_path": tools.ffmpeg,
            "ffprobe_path": tools.ffprobe,
        },
        "source": {
            "path": source.path,
            "width": source.width,
            "height": source.height,
            "codec": source.codec,
            "profile": source.profile,
            "bit_depth": source.bit_depth,
            "pix_fmt": source.pix_fmt,
            "fps": round(source.fps, 3),
            "bitrate_mbps": round(source.bitrate_mbps, 2),
            "hdr_type": source.hdr_type,
            "audio_codec": source.audio_codec,
            "audio_channels": source.audio_channels,
        },
        "config": {
            "mode": mode_labels.get(test_mode, test_mode),
            "scenario": scenario.label,
            "target_codec": scenario.target_codec,
            "target_bitrate": scenario.bitrate,
            "tonemap": scenario.tonemap,
            "hls_segment_secs": cfg.hls_segment_secs,
            "encode_hw": bool(hw.hwaccel),
            "decode_hw": use_hw_decode,
        },
        "results": {
            "total_elapsed_secs": round(total_elapsed, 2),
            "total_streams": manager.count,
            "active_streams": manager.active_count,
            "combined_speed": round(manager.combined_speed(), 3),
            "hw_saturated": hw_saturated,
            "streams": streams,
        },
    }

    if escalator:
        report["results"]["max_stable_streams"] = escalator.max_stable
        report["results"]["fps_threshold"] = round(escalator.threshold, 2)
        if escalator.failure_reason:
            report["results"]["failure_reason"] = escalator.failure_reason

    if io_mon:
        report["io"] = {
            "total_written_mb": round(io_mon.total_written_mb, 1),
            "last_write_mbs": round(io_mon.write_mbs, 2),
            "last_read_mbs_est": round(io_mon.read_mbs, 2),
            "cache_dir": cache_dir,
        }

    return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="jf_hw_stress",
        description="Jellyfin Hardware Transcode Stress Tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive TUI (default)
  python3 jf_hw_stress.py

  # Headless escalating test with JSON output
  python3 jf_hw_stress.py --headless --source /media/movies/movie.mkv \\
      --mode escalating --json --duration 120

  # Fixed 4-stream test for CI
  python3 jf_hw_stress.py --headless --source /data/test.mkv \\
      --mode fixed --streams 4 --duration 60 --json

  # Kubernetes Job (auto-detect ffmpeg from PATH)
  python3 jf_hw_stress.py --headless --source /media/movies/movie.mkv \\
      --cache-dir /dev/shm/stress --mode escalating --json --duration 180
""",
    )
    p.add_argument("--headless", action="store_true",
                    help="Run without interactive prompts (requires --source)")
    p.add_argument("--source", type=str, metavar="PATH",
                    help="Path to source video file")
    p.add_argument("--ffmpeg-dir", type=str, metavar="DIR",
                    help="Directory containing ffmpeg and ffprobe (default: system PATH)")
    p.add_argument("--cache-dir", type=str, metavar="DIR",
                    help="Scratch directory for HLS segments (default: /dev/shm/transcode_stress or /tmp)")
    p.add_argument("--mode", type=str, default="escalating",
                    choices=["fixed", "escalating", "hybrid", "mixed"],
                    help="Test mode (default: escalating)")
    p.add_argument("--streams", type=int, default=4,
                    help="Number of streams for fixed mode (default: 4)")
    p.add_argument("--scenario", type=str, default="1080p_compat",
                    help="Scenario ID (default: 1080p_compat). Use --list-scenarios to see options")
    p.add_argument("--hw-decode", action="store_true", default=True,
                    help="Enable hardware decode (default: True)")
    p.add_argument("--no-hw-decode", action="store_true",
                    help="Disable hardware decode")
    p.add_argument("--duration", type=int, default=300, metavar="SECS",
                    help="Max test duration in seconds (default: 300)")
    p.add_argument("--ramp-interval", type=float, default=12.0,
                    help="Seconds of stability before adding a stream (default: 12)")
    p.add_argument("--fail-secs", type=float, default=5.0,
                    help="Seconds below threshold before failure (default: 5)")
    p.add_argument("--fps-ratio", type=float, default=0.90,
                    help="Min FPS ratio to maintain 0.5-1.0 (default: 0.90)")
    p.add_argument("--json", action="store_true",
                    help="Output JSON report to stdout")
    p.add_argument("--json-file", type=str, metavar="PATH",
                    help="Write JSON report to file")
    p.add_argument("--html-report", type=str, metavar="PATH",
                    help="Write HTML report to file")
    p.add_argument("--force-platform", type=str, metavar="PLATFORM",
                    choices=["vaapi", "qsv", "nvenc", "amf", "vt", "sw"],
                    help="Force a specific hardware platform instead of auto-detect "
                         "(useful when ffmpeg is compiled with multiple hwaccels)")
    p.add_argument("--list-scenarios", action="store_true",
                    help="List available scenarios and exit")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# HEADLESS RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_headless(args: argparse.Namespace):
    """Non-interactive runner for CI, Kubernetes Jobs, and scripted benchmarks."""
    import sys as _sys

    # Validate required args
    if not args.source:
        print("ERROR: --headless requires --source", file=_sys.stderr)
        _sys.exit(1)
    if not Path(args.source).is_file():
        print(f"ERROR: Source file not found: {args.source}", file=_sys.stderr)
        _sys.exit(1)

    # Tool discovery
    if args.ffmpeg_dir:
        tools = _find_tools_in(Path(args.ffmpeg_dir))
        if not tools:
            print(f"ERROR: ffmpeg/ffprobe not found in {args.ffmpeg_dir}", file=_sys.stderr)
            _sys.exit(1)
    else:
        tools = _tools_from_path()
        if not tools:
            print("ERROR: ffmpeg/ffprobe not found in PATH", file=_sys.stderr)
            _sys.exit(1)

    print(f"[headless] ffmpeg: {tools.ffmpeg}", file=_sys.stderr)
    print(f"[headless] ffprobe: {tools.ffprobe}", file=_sys.stderr)

    # Hardware detection
    if args.force_platform:
        hw = _force_hardware_platform(tools, args.force_platform)
    else:
        hw = detect_hardware(tools)
    print(f"[headless] Hardware: {hw.name} — {hw.gpu_name}", file=_sys.stderr)

    # Probe source
    source = probe_source(tools, args.source)
    print(f"[headless] Source: {source.width}x{source.height} {source.codec.upper()} "
          f"{source.fps:.3f}fps {source.bitrate_mbps:.1f}Mbps {source.hdr_type}",
          file=_sys.stderr)

    # Source duration check
    try:
        r = subprocess.run(
            [tools.ffprobe, "-v", "quiet", "-print_format", "json",
             "-show_format", args.source],
            capture_output=True, text=True, errors="replace",
        )
        fmt = json.loads(r.stdout).get("format", {})
        dur = float(fmt.get("duration", 0))
        if 0 < dur < 30:
            print(f"WARNING: Source file is only {dur:.0f}s long — "
                  "streams may end before test completes", file=_sys.stderr)
    except Exception:
        pass

    # Cache directory
    cache_dir = args.cache_dir
    if not cache_dir:
        shm = Path("/dev/shm")
        cache_dir = str(shm / "transcode_stress") if shm.is_dir() else str(
            Path(tempfile.gettempdir()) / "transcode_stress")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for old in Path(cache_dir).glob("st_*"):
        try:
            old.unlink()
        except Exception:
            pass
    print(f"[headless] Cache: {cache_dir}", file=_sys.stderr)

    # Map mode
    mode_map = {
        "fixed": MODE_FIXED, "escalating": MODE_ESCALATING,
        "hybrid": MODE_HYBRID, "mixed": MODE_MIXED,
    }
    test_mode = mode_map[args.mode]

    # Scenario
    scenario_map = {s.id: s for s in SCENARIOS + MIXED_SCENARIOS}
    if args.scenario in scenario_map:
        scenario = scenario_map[args.scenario]
    else:
        scenario = SCENARIOS[3]  # 1080p_compat default
        print(f"[headless] Scenario '{args.scenario}' not found, using {scenario.label}",
              file=_sys.stderr)

    use_hw_decode = args.hw_decode and not args.no_hw_decode

    # Config
    cfg = TestConfig(
        ramp_interval=args.ramp_interval,
        fail_secs=args.fail_secs,
        fps_ratio=args.fps_ratio,
    )

    # Mixed pool
    mixed_pool: Optional[List[Scenario]] = None
    if test_mode == MODE_MIXED:
        mixed_pool = list(MIXED_SCENARIOS)
        if source.is_hdr:
            mixed_pool += MIXED_HDR_SCENARIOS

    # Build objects
    manager = StreamManager(tools, hw, scenario, source, cache_dir, cfg, mixed_pool)
    io_mon = IOMonitor(cache_dir, source)
    escalator: Optional[EscalatingController] = None
    if test_mode != MODE_FIXED:
        escalator = EscalatingController(cfg, source.fps)

    events: List[str] = []
    start_time = time.time()
    hw_saturated = False

    def ts() -> str:
        return _fmt_elapsed(time.time() - start_time)

    # Initial launches
    if test_mode == MODE_FIXED:
        for _ in range(args.streams):
            manager.launch(use_hw_decode=use_hw_decode, use_hw_encode=True)
        print(f"[headless] Launched {args.streams} fixed streams", file=_sys.stderr)
    else:
        manager.launch(use_hw_decode=use_hw_decode, use_hw_encode=True)
        thr = source.fps * cfg.fps_ratio
        print(f"[headless] Stream #1 launched — threshold {thr:.1f} fps", file=_sys.stderr)

    io_mon.start()

    # Headless main loop — no TUI, just poll and escalate
    done = False
    def shutdown(sig=None, frame=None):
        nonlocal done
        done = True
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    tick_interval = 1.0
    last_status = 0.0

    while not done:
        time.sleep(tick_interval)
        manager.refresh()
        elapsed = time.time() - start_time

        # Duration limit
        if elapsed >= args.duration:
            print(f"[headless] Duration limit ({args.duration}s) reached", file=_sys.stderr)
            done = True
            break

        # Escalation logic
        if escalator and test_mode in (MODE_ESCALATING, MODE_HYBRID, MODE_MIXED):
            action = escalator.tick(manager.stats)
            if action == "add":
                use_enc = not hw_saturated
                use_dec = use_hw_decode and not hw_saturated
                sid = manager.launch(use_hw_decode=use_dec, use_hw_encode=use_enc)
                enc_tag = "HW" if use_enc else "SW"
                msg = f"{ts()}  ↑ Stream #{sid} [{enc_tag}] ({manager.count} total)"
                events.append(msg)
                print(f"[headless] {msg}", file=_sys.stderr)
            elif action == "fail":
                if test_mode == MODE_HYBRID and not hw_saturated:
                    hw_saturated = True
                    msg = f"{ts()}  HW saturated at {manager.count} streams — pivoting to SW"
                    events.append(msg)
                    print(f"[headless] {msg}", file=_sys.stderr)
                    escalator.reset_warmup()
                    sid = manager.launch(use_hw_decode=False, use_hw_encode=False)
                    events.append(f"{ts()}  ↑ Stream #{sid} [SW]")
                else:
                    msg = f"{ts()}  Max stable: {escalator.max_stable} stream(s)"
                    events.append(msg)
                    if escalator.failure_reason:
                        events.append(escalator.failure_reason)
                    print(f"[headless] SATURATION: {msg}", file=_sys.stderr)
                    done = True

        # Periodic status
        if elapsed - last_status >= 10:
            running = [s for s in manager.stats.values() if s.status == "running"]
            if running:
                avg_fps = sum(s.fps for s in running) / len(running)
                combined = manager.combined_speed()
                print(f"[headless] {ts()} | {len(running)} streams | "
                      f"avg FPS {avg_fps:.1f} | speed {combined:.2f}x",
                      file=_sys.stderr)
            last_status = elapsed

        # Check for all streams dead
        if all(s.status in ("error", "done") for s in manager.stats.values()):
            print("[headless] All streams ended", file=_sys.stderr)
            done = True

    io_mon.stop()
    total_elapsed = time.time() - start_time

    # Generate reports
    report = generate_json_report(
        hw, tools, source, scenario, manager, escalator, io_mon,
        cfg, test_mode, use_hw_decode, hw_saturated, total_elapsed, cache_dir,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))

    if args.json_file:
        Path(args.json_file).write_text(json.dumps(report, indent=2, default=str))
        print(f"[headless] JSON report saved: {args.json_file}", file=_sys.stderr)

    if args.html_report:
        rpath = generate_report(
            hw, tools, source, scenario, manager, escalator, io_mon,
            cfg, test_mode, use_hw_decode, hw_saturated, total_elapsed, cache_dir,
        )
        # Move to requested path
        import shutil as _shutil
        _shutil.move(rpath, args.html_report)
        print(f"[headless] HTML report saved: {args.html_report}", file=_sys.stderr)

    # Print summary to stderr
    print(f"\n[headless] === RESULTS ===", file=_sys.stderr)
    print(f"[headless] Hardware: {hw.name} — {hw.gpu_name}", file=_sys.stderr)
    print(f"[headless] Mode: {args.mode}", file=_sys.stderr)
    print(f"[headless] Streams: {manager.count} total, {manager.active_count} active", file=_sys.stderr)
    print(f"[headless] Combined speed: {manager.combined_speed():.2f}x", file=_sys.stderr)
    if escalator:
        print(f"[headless] Max stable: {escalator.max_stable}", file=_sys.stderr)
    print(f"[headless] Elapsed: {_fmt_elapsed(total_elapsed)}", file=_sys.stderr)

    # Surface errors
    for sid, s in manager.stats.items():
        if s.error_msg:
            print(f"[headless] Stream #{sid} ERROR: {s.error_msg}", file=_sys.stderr)

    manager.kill_all()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    console.print()
    console.rule(f"[bold cyan]{APP_TITLE}  v{VERSION}[/bold cyan]")
    console.print(
        "  [dim]Apple VideoToolbox · NVIDIA NVENC · Intel QSV · AMD AMF/VAAPI[/dim]\n"
    )

    manager:   Optional[StreamManager]       = None
    io_mon:    Optional[IOMonitor]           = None
    escalator: Optional[EscalatingController] = None

    try:
        # ── Tool discovery ─────────────────────────────────────────────────
        tools = discover_tools()

        # ── Hardware detection ─────────────────────────────────────────────
        console.print()
        console.print("[dim]  Detecting hardware acceleration …[/dim]", end="  ")
        hw = detect_hardware(tools)
        console.print(f"[green]✓[/green]  [bold cyan]{esc(hw.name)}[/bold cyan]  {esc(hw.gpu_name)}")

        # ── Source file ────────────────────────────────────────────────────
        input_path = pick_source_file()
        console.print()
        console.print("[dim]  Probing source …[/dim]", end="  ")
        source  = probe_source(tools, input_path)
        hdr_tag = f"  [bold magenta]{esc(source.hdr_type)}[/bold magenta]" if source.is_hdr else ""
        console.print(
            f"[green]✓[/green]  [dim]{source.width}×{source.height}  "
            f"{esc(source.codec.upper())}  {source.bit_depth}-bit  "
            f"{source.fps:.3f} fps  {source.bitrate_mbps:.1f} Mbps[/dim]{hdr_tag}"
        )

        # Show detailed source info
        display_source_info(source)

        # ── Cache directory ────────────────────────────────────────────────
        cache_dir = pick_cache_dir()

        # ── Setup ──────────────────────────────────────────────────────────
        scenario, test_mode, fixed_streams, use_hw_decode, cfg, mixed_pool = \
            run_setup(hw, source)

        # ── Prepare cache ──────────────────────────────────────────────────
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        for old in Path(cache_dir).glob("st_*"):
            try:
                old.unlink()
            except Exception:
                pass

        # ── Build objects ──────────────────────────────────────────────────
        manager = StreamManager(tools, hw, scenario, source, cache_dir, cfg, mixed_pool)
        io_mon  = IOMonitor(cache_dir, source)

        if test_mode != MODE_FIXED:
            escalator = EscalatingController(cfg, source.fps)

        events:     List[str] = []
        start_time: float     = time.time()
        hw_saturated: bool    = False

        def ts() -> str:
            return _fmt_elapsed(time.time() - start_time)

        # ── Initial launches ───────────────────────────────────────────────
        console.print()
        if test_mode == MODE_FIXED:
            for _ in range(fixed_streams):
                manager.launch(use_hw_decode=use_hw_decode, use_hw_encode=True)
            events.append(f"{ts()}  {fixed_streams} stream(s) launched")
        else:
            manager.launch(use_hw_decode=use_hw_decode, use_hw_encode=True)
            thr = source.fps * cfg.fps_ratio
            events.append(
                f"{ts()}  Stream #1 launched — "
                f"threshold {thr:.1f} fps  ramp {cfg.ramp_interval:.0f}s  "
                f"fail {cfg.fail_secs:.0f}s"
            )

        io_mon.start()

        # ════════════════════════════════════════════════════
        # MAIN RUN LOOP + SATURATION DECISION
        # ════════════════════════════════════════════════════

        continuous    = False
        session_done  = False

        while not session_done:
            done, hw_saturated, user_stopped = run_loop(
                hw, source, scenario, manager, escalator, io_mon,
                cfg, test_mode, use_hw_decode, hw_saturated,
                events, start_time, continuous=continuous,
            )

            if user_stopped or test_mode == MODE_FIXED:
                session_done = True
                break

            if continuous:
                # User Ctrl+C'd during continuous run → stop
                session_done = True
                break

            # ── Saturation reached (escalating modes) ──────────────────────
            # Print the final Live state as a static snapshot so the user
            # can read/screenshot it while we wait for their decision.
            elapsed   = time.time() - start_time
            snap_panel = render_dashboard(
                hw, source, scenario, manager, escalator, io_mon,
                elapsed, events, test_mode, use_hw_decode, hw_saturated,
            )
            console.print(snap_panel)
            console.print()
            console.print(
                f"  [bold yellow]Saturation reached[/bold yellow]  —  "
                f"{manager.active_count} streams  │  "
                f"Max stable: {escalator.max_stable if escalator else '?'}"
            )

            choice = _countdown_choice(
                header="What would you like to do?",
                options=[
                    ("c", "Continue running indefinitely at current load"),
                    ("s", "Stop, keep results visible, then optionally save report"),
                ],
                default="s",
                timeout=60.0,
            )

            if choice == "c":
                continuous = True
                events.append(f"{ts()}  Continuing indefinitely at saturation capacity")
                # Continue the loop — run_loop will run with continuous=True
                # and only stop on Ctrl+C
            else:
                session_done = True

        # ════════════════════════════════════════════════════
        # FINAL DISPLAY — keep dashboard visible for screenshot
        # ════════════════════════════════════════════════════

        io_mon.stop()
        total_elapsed = time.time() - start_time
        elapsed       = total_elapsed

        final_panel = render_dashboard(
            hw, source, scenario, manager, escalator, io_mon,
            elapsed, events, test_mode, use_hw_decode, hw_saturated,
        )
        console.print()
        console.print(final_panel)
        console.print()
        console.print("  [dim]Screenshot now if needed, then press Enter to see summary.[/dim]")
        try:
            input()
        except Exception:
            pass

        # ── Summary ────────────────────────────────────────────────────────
        console.rule("[bold]Session Complete[/bold]")
        console.print()
        console.print(f"  Hardware    [bold cyan]{esc(hw.name)}[/bold cyan]  {esc(hw.gpu_name)}")
        console.print(f"  Scenario    [bold green]{esc(scenario.label)}[/bold green]  {scenario.bitrate}")
        console.print(
            f"  Codec       {scenario.target_codec.upper()}  │  "
            f"Encode: {'HW' if hw.hwaccel else 'SW'}  │  "
            f"Decode: {'HW' if use_hw_decode and source.codec in hw.hw_decode_codecs else 'SW'}"
        )
        console.print(
            f"  Source      {source.width}×{source.height}  "
            f"{esc(source.codec.upper())}  {source.fps:.3f} fps  "
            f"{esc(source.hdr_type)}"
        )
        console.print(f"  Cache       [dim]{esc(cache_dir)}[/dim]")

        if escalator and test_mode != MODE_FIXED:
            console.print()
            console.print(
                f"  Max stable concurrent streams:  "
                f"[bold green]{escalator.max_stable}[/bold green]"
            )
            if hw_saturated:
                console.print(
                    "  HW saturation:  yes  "
                    "[dim](SW streams were added after HW engines filled)[/dim]"
                )
            if escalator.failure_reason:
                console.print(f"  Final failure:  [red]{esc(escalator.failure_reason)}[/red]")

        if io_mon:
            console.print(
                f"  I/O written:    {io_mon.total_written_mb:.0f} MB to cache"
            )

        console.print(f"\n  Total elapsed:  {_fmt_elapsed(total_elapsed)}\n")

        # ── Report prompt ──────────────────────────────────────────────────
        if Confirm.ask("  Save an HTML report?", default=False):
            rpath = generate_report(
                hw, tools, source, scenario, manager, escalator, io_mon,
                cfg, test_mode, use_hw_decode, hw_saturated, total_elapsed, cache_dir,
            )
            console.print(f"  [green]✓[/green]  Report saved: [dim]{esc(rpath)}[/dim]")

        console.print()

    except KeyboardInterrupt:
        console.print("\n  [yellow]Interrupted.[/yellow]")

    finally:
        if io_mon:
            io_mon.stop()
        if manager:
            manager.kill_all()
        console.show_cursor(True)


if __name__ == "__main__":
    args = parse_args()

    if args.list_scenarios:
        print("Available scenarios:")
        print(f"  {'ID':<20} {'Label':<26} {'Bitrate':<8} {'TM'}")
        print(f"  {'─'*20} {'─'*26} {'─'*8} {'─'*4}")
        for s in SCENARIOS:
            print(f"  {s.id:<20} {s.label:<26} {s.bitrate:<8} {'✓' if s.tonemap else '–'}")
        print("\nMixed-bag pool:")
        for s in MIXED_SCENARIOS:
            print(f"  {s.id:<20} {s.label:<26} {s.bitrate:<8} {'✓' if s.tonemap else '–'}")
        sys.exit(0)

    if args.headless:
        run_headless(args)
    else:
        main()
