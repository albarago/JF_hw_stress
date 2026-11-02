"""Subtitle burn-in: which subtitles to draw into the video, and the files ffmpeg needs, made once per run.

Jellyfin burns subtitles in when a client can't show them itself (common for PGS and styled ASS). It does
it in one of two ways (EncodingHelper):

- text subtitles (ASS, SSA, SRT, WebVTT…): the track is first extracted to a file, and libass draws it
  (`subtitles=f=…`), with the file's embedded fonts as `fontsdir`;
- picture subtitles (PGS, VobSub, DVB): the subtitle stream goes into the filter graph, is scaled to the
  output size and laid over the video.

The command lines are built in `command.py`. This module picks the subtitles and prepares their files.
"""
from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .probe import SourceInfo, SubtitleTrack

SAMPLE = "sample"
AUTO = "auto"
TEXT_EXTS = {".ass": "ass", ".ssa": "ass", ".srt": "subrip", ".vtt": "webvtt"}
IMAGE_EXTS = {".sup": "hdmv_pgs_subtitle"}
COVER_SECS = 6 * 3600           # looped text subtitles are repeated to cover this much playback
MAX_REPEATED_BYTES = 64_000_000  # …but a repeated file never grows past this (each stream parses it at start)
PREP_TIMEOUT = 300
BURN_DIR = "burnin"             # inside the run folder; removed with it


class BurnInError(ValueError):
    pass


@dataclass(frozen=True)
class BurnIn:
    """The subtitles chosen for burn-in."""
    kind: str                    # "text" | "image"
    origin: str                  # "track" | "file" | "sample"
    codec: str                   # ffprobe codec name ("ass", "subrip", "hdmv_pgs_subtitle"…)
    label: str                   # for people, e.g. "track 1: PGS eng (default)"
    track: Optional[SubtitleTrack] = None
    path: str = ""               # origin "file"

    @property
    def is_text(self) -> bool:
        return self.kind == "text"

    @property
    def is_ass(self) -> bool:
        return self.codec in ("ass", "ssa")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "origin": self.origin, "codec": self.codec, "label": self.label,
                "track": self.track.number if self.track else None,
                "stream_index": self.track.index if self.track else None,
                "language": (self.track.language or None) if self.track else None,
                "file": self.path or None}


@dataclass(frozen=True)
class Prepared:
    """Burn-in ready for ffmpeg: the files it reads and where the subtitle stream enters the graph."""
    burn: BurnIn
    text_file: str = ""          # text: the file the subtitles filter reads
    fonts_dir: str = ""          # text: fonts extracted from the source, for libass
    image_input: str = ""        # image from a file: a second ffmpeg input
    stream_spec: str = ""        # image: the graph input, "0:N" (the source) or "1:0" (image_input)
    canvas: Tuple[int, int] = (0, 0)   # image: size the pictures were authored for (-canvas_size)
    repeats: int = 1             # text: copies of the subtitles laid end to end to follow the looped source
    events: int = 0              # text: subtitle lines in one copy


# --- choosing ----------------------------------------------------------------------------------

def resolve(spec: str, source: SourceInfo) -> BurnIn:
    """Turn --burn-in's value into a choice: auto, a subtitle track number, sample, or a file."""
    spec = (spec or "").strip()
    burnable = [t for t in source.subtitles if t.burnable]
    if spec.lower() == AUTO:
        if not burnable:
            raise BurnInError("the source has no subtitle track that can be burned in; "
                              "use --burn-in sample or a subtitle file")
        return from_track(next((t for t in burnable if t.default), burnable[0]))
    if spec.lower() == SAMPLE:
        return BurnIn("text", "sample", "ass", "built-in sample (styled ASS, signs and karaoke)")
    if re.fullmatch(r"(s:)?\d+", spec):
        number = int(spec.split(":")[-1])
        track = next((t for t in source.subtitles if t.number == number), None)
        if track is None:
            have = ", ".join(str(t.number) for t in source.subtitles) or "none"
            raise BurnInError(f"no subtitle track {number} in the source (tracks: {have})")
        if not track.burnable:
            raise BurnInError(f"subtitle track {number} ({track.codec}) has nothing ffmpeg can draw")
        return from_track(track)
    path = Path(spec).expanduser()
    if not path.is_file():
        raise BurnInError(f"--burn-in: {spec!r} is not auto, sample, a subtitle track number or a file")
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return BurnIn("text", "file", TEXT_EXTS[ext], f"file {path.name}", path=str(path.resolve()))
    if ext in IMAGE_EXTS:
        return BurnIn("image", "file", IMAGE_EXTS[ext], f"file {path.name} (PGS)", path=str(path.resolve()))
    raise BurnInError(f"--burn-in: unsupported subtitle file type {ext or '(none)'}; "
                      f"use {', '.join(sorted(TEXT_EXTS) + sorted(IMAGE_EXTS))}")


def from_track(track: SubtitleTrack) -> BurnIn:
    return BurnIn("image" if track.is_image else "text", "track", track.codec,
                  f"track {track.number}: {track.label()}", track=track)


def subtitle_lines(source: SourceInfo) -> List[str]:
    """One line per subtitle track, numbered as --burn-in takes them."""
    lines = []
    for t in source.subtitles:
        kind = "picture" if t.is_image else "text" if t.is_text else "cannot be burned in"
        lines.append(f"{t.number:>3}  {t.label():<40} {kind}  (stream {t.index})")
    if source.subtitles and source.font_attachments:
        lines.append(f"     {source.font_attachments} embedded font(s), used for ASS subtitles")
    return lines


# --- preparing ---------------------------------------------------------------------------------

def prepare(burn: BurnIn, ffmpeg: str, ffprobe: str, source: SourceInfo, work_dir: str, loop: bool,
            cover_secs: float = COVER_SECS) -> Prepared:
    """Make the files ffmpeg reads during the run (in the run folder) and say how to use them."""
    if not burn.is_text:
        if burn.origin == "file":
            return Prepared(burn, image_input=burn.path, stream_spec="1:0", canvas=_image_canvas(ffprobe, burn.path))
        track = burn.track
        assert track is not None
        canvas = (track.width, track.height) if track.width and track.height and "dvb" not in track.codec else (0, 0)
        return Prepared(burn, stream_spec=f"0:{track.index}", canvas=canvas)

    folder = Path(work_dir) / BURN_DIR
    folder.mkdir(parents=True, exist_ok=True)
    fonts_dir = ""
    if burn.origin == "sample":
        length = cover_secs if loop else max(source.duration_secs, 60.0)
        path = folder / "sample.ass"
        path.write_text(sample_ass(length, source.width or 1920, source.height or 1080), encoding="utf-8")
        return Prepared(burn, text_file=str(path), events=count_events(path.read_text(encoding="utf-8"), "ass"))

    ext = ".ass" if burn.is_ass else ".srt"
    raw = folder / f"subs-once{ext}"
    if burn.origin == "track":
        assert burn.track is not None
        codec = "copy" if burn.codec in ("ass", "subrip") else ("ass" if burn.is_ass else "srt")
        _ffmpeg(ffmpeg, ["-i", source.path, "-map", f"0:{burn.track.index}", "-c:s", codec, str(raw)],
                f"could not extract subtitle track {burn.track.number}")
        if source.font_attachments:
            fonts = folder / "fonts"
            fonts.mkdir(exist_ok=True)
            # ffmpeg writes every attachment into the working folder; it exits non-zero without an output.
            _ffmpeg(ffmpeg, ["-dump_attachment:t", "", "-i", source.path, "-t", "0", "-f", "null", "-"],
                    "", cwd=str(fonts), check=False)
            if any(fonts.iterdir()):
                fonts_dir = str(fonts)
    elif burn.codec in ("ass", "subrip"):
        raw.write_bytes(Path(burn.path).read_bytes())
    else:
        _ffmpeg(ffmpeg, ["-i", burn.path, "-c:s", "srt", str(raw)], f"could not read {burn.path}")

    text = raw.read_text(encoding="utf-8-sig", errors="replace")
    kind = "ass" if burn.is_ass else "srt"
    events = count_events(text, kind)
    repeats = 1
    if loop and source.duration_secs > 0 and events:
        repeats = max(1, math.ceil(cover_secs / source.duration_secs))
        repeats = max(1, min(repeats, MAX_REPEATED_BYTES // max(1, len(text.encode("utf-8")))))
    final = folder / f"subs{ext}"
    final.write_text(repeat(text, kind, source.duration_secs, repeats) if repeats > 1 else text, encoding="utf-8")
    raw.unlink()
    return Prepared(burn, text_file=str(final), fonts_dir=fonts_dir, repeats=repeats, events=events)


def _ffmpeg(ffmpeg: str, args: List[str], error: str, cwd: Optional[str] = None, check: bool = True) -> None:
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-y"] + args
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=PREP_TIMEOUT,
                             stdin=subprocess.DEVNULL, cwd=cwd)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BurnInError(f"{error}: {exc}") from None
    if check and out.returncode != 0:
        detail = (out.stderr or "").strip().splitlines()
        raise BurnInError(f"{error}: {detail[-1][:300] if detail else f'ffmpeg exited {out.returncode}'}")


def _image_canvas(ffprobe: str, path: str) -> Tuple[int, int]:
    try:
        out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "s:0", "-show_entries",
                              "stream=width,height", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
        w, h = (int(x) for x in out.stdout.strip().split(",")[:2])
        return (w, h) if w > 0 and h > 0 else (0, 0)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return (0, 0)


# --- following the looped source ---------------------------------------------------------------
#
# The source is looped (-stream_loop -1), and ffmpeg keeps timestamps rising across loops. A subtitles
# file covers one pass, so after the first loop nothing would be drawn and the load would drop. Copies of
# the subtitles, each shifted by the source's length, keep them on screen for the whole run. Picture
# subtitles come from the looped input itself and need nothing. (Each loop can drift by a few
# milliseconds against the video; that changes nothing about the load.)

ASS_TIME = re.compile(r"^(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")
SRT_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


def _ass_secs(value: str) -> Optional[float]:
    m = ASS_TIME.match(value.strip())
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None


def _ass_time(secs: float) -> str:
    cs = int(round(max(0.0, secs) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _srt_time(secs: float) -> str:
    ms = int(round(max(0.0, secs) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def count_events(text: str, kind: str) -> int:
    if kind == "ass":
        return sum(1 for line in text.splitlines() if line.startswith("Dialogue:"))
    return sum(1 for line in text.splitlines() if "-->" in line)


def repeat(text: str, kind: str, period: float, copies: int) -> str:
    return repeat_ass(text, period, copies) if kind == "ass" else repeat_srt(text, period, copies)


def repeat_ass(text: str, period: float, copies: int) -> str:
    """The script with its Dialogue lines repeated `copies` times, copy k shifted by k × period."""
    lines = text.splitlines()
    fields: List[str] = []
    section, dialogue = "", []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.lower()
        elif section == "[events]" and stripped.lower().startswith("format:"):
            fields = [f.strip().lower() for f in stripped.split(":", 1)[1].split(",")]
        elif section == "[events]" and line.startswith("Dialogue:"):
            dialogue.append(i)
    if not dialogue or "start" not in fields or "end" not in fields:
        return text
    start_i, end_i = fields.index("start"), fields.index("end")
    extra = []
    for k in range(1, copies):
        for i in dialogue:
            head, body = lines[i].split(":", 1)
            parts = body.split(",", len(fields) - 1)
            start, end = _ass_secs(parts[start_i]), _ass_secs(parts[end_i])
            if start is None or end is None:
                continue
            parts[start_i] = _ass_time(start + k * period)
            parts[end_i] = _ass_time(end + k * period)
            extra.append(f"{head}:" + ",".join(parts))
    last = dialogue[-1]
    return "\n".join(lines[:last + 1] + extra + lines[last + 1:]) + "\n"


def repeat_srt(text: str, period: float, copies: int) -> str:
    """The cues repeated `copies` times, copy k shifted by k × period, renumbered."""
    blocks = [b for b in re.split(r"\r?\n\s*\r?\n", text.strip()) if "-->" in b]
    out, n = [], 0
    for k in range(copies):
        offset = k * period
        for block in blocks:
            lines = block.splitlines()
            at = next(i for i, ln in enumerate(lines) if "-->" in ln)
            n += 1
            timing = SRT_TIME.sub(lambda m, off=offset: _srt_time(_srt_secs(m) + off), lines[at])
            out.append("\n".join([str(n), timing] + lines[at + 1:]))
    return "\n\n".join(out) + "\n"


def _srt_secs(m: re.Match) -> float:
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4).ljust(3, "0")) / 1000


# --- the built-in sample -----------------------------------------------------------------------

STYLE_FIELDS = ", ".join([
    "Name", "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour", "OutlineColour", "BackColour", "Bold", "Italic",
    "Underline", "StrikeOut", "ScaleX", "ScaleY", "Spacing", "Angle", "BorderStyle", "Outline", "Shadow", "Alignment",
    "MarginL", "MarginR", "MarginV", "Encoding"])


def sample_ass(length_secs: float, width: int = 1920, height: int = 1080) -> str:
    """Styled ASS like a typeset anime episode: two lines of dialogue always on screen, a moving, rotating,
    blurred sign, and a karaoke line. Animated tags make libass redraw every frame, as real typesetting does."""
    head = f"""[Script Info]
; Generated by jf-hw-stress for subtitle burn-in tests
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: {STYLE_FIELDS}
Style: Default,Arial,{height * 64 // 1080},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,2,2,120,120,60,1
Style: Sign,Arial,{height * 84 // 1080},&H0000E5FF,&H000000FF,&H00301000,&H00000000,-1,0,0,0,100,100,2,0,1,5,0,8,60,60,80,1
Style: Karaoke,Arial,{height * 56 // 1080},&H00FFFFFF,&H00FF8800,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,8,60,60,220,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    cycle, n = 4.0, int(math.ceil(length_secs / 4.0))
    for i in range(n):
        t0, t1 = i * cycle, (i + 1) * cycle
        a, b = _ass_time(t0), _ass_time(t1)
        lines.append(f"Dialogue: 0,{a},{b},Default,,0,0,0,,{{\\fad(150,150)}}Line {i + 1}: subtitles burned into "
                     f"the video\\Nwith an outline, a shadow and a fade, like most dialogue")
        x0, x1 = (300, width - 300) if i % 2 else (width - 300, 300)
        lines.append(f"Dialogue: 1,{a},{b},Sign,,0,0,0,,{{\\move({x0},{height // 5},{x1},{height // 4})"
                     f"\\t(\\frz{360 if i % 2 else -360}\\blur4\\fscx130\\fscy130)\\3c&H402000&}}SIGN {i + 1}")
        kara = "".join(f"{{\\k{25 + (j * 7) % 20}}}{w} " for j, w in enumerate(
            "every syllable lights up in turn while the line keeps moving".split()))
        lines.append(f"Dialogue: 2,{a},{b},Karaoke,,0,0,0,,{{\\fad(200,0)\\t(0,{int(cycle * 1000)},\\fscx110)}}{kara.strip()}")
    return head + "\n".join(lines) + "\n"
