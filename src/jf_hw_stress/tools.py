"""Find ffmpeg/ffprobe and read what the build supports."""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

SYSTEM = platform.system()  # "Darwin" | "Linux" | "Windows"
EXE = ".exe" if SYSTEM == "Windows" else ""

# Where Jellyfin installs its own ffmpeg build (jellyfin-ffmpeg).
JELLYFIN_DIRS = {
    "Darwin": ["/Applications/Jellyfin.app/Contents/MacOS"],
    "Linux": ["/usr/lib/jellyfin-ffmpeg", "/usr/share/jellyfin-ffmpeg", "/opt/jellyfin-ffmpeg"],
    "Windows": [r"C:\Program Files\Jellyfin\Server", r"C:\Program Files (x86)\Jellyfin\Server"],
}


class ToolError(RuntimeError):
    pass


def run_text(cmd: List[str], timeout: float = 15) -> str:
    """stdout+stderr of a short command, or '' when it cannot run."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                             timeout=timeout, stdin=subprocess.DEVNULL)
        return (out.stdout or "") + (out.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        return ""


@dataclass
class Capabilities:
    hwaccels: Set[str] = field(default_factory=set)
    encoders: Set[str] = field(default_factory=set)
    decoders: Set[str] = field(default_factory=set)
    filters: Set[str] = field(default_factory=set)

    def has_filter(self, name: str) -> bool:
        return name in self.filters


@dataclass
class Tools:
    ffmpeg: str
    ffprobe: str
    origin: str                 # "jellyfin-ffmpeg (…)" | "PATH" | "--ffmpeg"
    version: str = ""
    caps: Capabilities = field(default_factory=Capabilities)

    @property
    def is_jellyfin_build(self) -> bool:
        return "jellyfin" in self.version.lower()


def _codec_names(text: str) -> Set[str]:
    """Names from `ffmpeg -encoders/-decoders/-filters`: the second column of each listing line."""
    names = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r"[A-Z.|]{2,}", parts[0].replace("-", ".")) and parts[1] != "=":
            names.add(parts[1])
    return names


def _hwaccels(text: str) -> Set[str]:
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.lower().startswith("hardware acceleration methods"))
    except StopIteration:
        return set()
    return {line.strip() for line in lines[start + 1:] if line.strip()}


def read_capabilities(ffmpeg: str) -> Capabilities:
    base = [ffmpeg, "-hide_banner"]
    return Capabilities(
        hwaccels=_hwaccels(run_text(base + ["-hwaccels"])),
        encoders=_codec_names(run_text(base + ["-encoders"])),
        decoders=_codec_names(run_text(base + ["-decoders"])),
        filters=_codec_names(run_text(base + ["-filters"])),
    )


def read_version(ffmpeg: str) -> str:
    first = run_text([ffmpeg, "-hide_banner", "-version"]).splitlines()
    match = re.match(r"ffmpeg version (\S+)", first[0]) if first else None
    return match.group(1) if match else ""


def _pair_in(directory: Path) -> Optional[tuple]:
    ffmpeg, ffprobe = directory / f"ffmpeg{EXE}", directory / f"ffprobe{EXE}"
    if ffmpeg.is_file() and ffprobe.is_file():
        return str(ffmpeg), str(ffprobe)
    return None


def candidates() -> List[Tools]:
    """Every ffmpeg/ffprobe pair found: Jellyfin's bundled build first, then PATH."""
    found: List[Tools] = []
    for directory in JELLYFIN_DIRS.get(SYSTEM, []):
        pair = _pair_in(Path(directory))
        if pair:
            found.append(Tools(pair[0], pair[1], f"jellyfin-ffmpeg ({directory})"))
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg and ffprobe and all(os.path.realpath(ffmpeg) != os.path.realpath(t.ffmpeg) for t in found):
        found.append(Tools(ffmpeg, ffprobe, "PATH"))
    return found


def from_path(path: str) -> Tools:
    """An ffmpeg binary or a directory containing ffmpeg and ffprobe."""
    p = Path(path).expanduser()
    if p.is_dir():
        pair = _pair_in(p)
        if not pair:
            raise ToolError(f"no ffmpeg and ffprobe in {p}")
        return Tools(pair[0], pair[1], str(p))
    if p.is_file():
        probe = p.with_name(p.name.replace("ffmpeg", "ffprobe"))
        if not probe.is_file():
            raise ToolError(f"found {p} but no {probe.name} next to it")
        return Tools(str(p), str(probe), str(p.parent))
    raise ToolError(f"not found: {p}")


def load(tools: Tools) -> Tools:
    """Fill in the version and capabilities (a few short ffmpeg calls)."""
    tools.version = read_version(tools.ffmpeg)
    if not tools.version:
        raise ToolError(f"{tools.ffmpeg} did not report a version; is it ffmpeg?")
    tools.caps = read_capabilities(tools.ffmpeg)
    return tools
