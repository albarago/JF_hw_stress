"""Read the source file's properties with ffprobe."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

PROBE_TIMEOUT = 30

AUDIO_NAMES: Dict[str, str] = {
    "aac": "AAC", "ac3": "Dolby Digital", "eac3": "Dolby Digital Plus", "dts": "DTS",
    "truehd": "Dolby TrueHD", "mlp": "MLP", "flac": "FLAC", "mp3": "MP3", "opus": "Opus",
    "vorbis": "Vorbis", "pcm_s16le": "PCM 16-bit", "pcm_s24le": "PCM 24-bit",
}
CHANNEL_NAMES: Dict[int, str] = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}

# Subtitle formats Jellyfin draws as pictures (MediaStream.IsTextFormat: pgs, dvdsub, vobsub, dvbsub, sup, sub),
# under the names ffprobe gives them.
IMAGE_SUBTITLES = {"hdmv_pgs_subtitle", "pgssub", "dvd_subtitle", "dvdsub", "vobsub", "dvb_subtitle", "dvbsub",
                   "xsub"}
# Subtitle streams that carry no drawable text or pictures.
UNBURNABLE_SUBTITLES = {"dvb_teletext", "eia_608", "arib_caption", "hdmv_text_subtitle"}
SUBTITLE_NAMES: Dict[str, str] = {
    "ass": "ASS", "ssa": "SSA", "subrip": "SRT", "srt": "SRT", "webvtt": "WebVTT", "mov_text": "MP4 text",
    "text": "text", "hdmv_pgs_subtitle": "PGS", "dvd_subtitle": "VobSub", "dvb_subtitle": "DVB", "xsub": "XSUB",
}


class ProbeError(RuntimeError):
    pass


@dataclass
class SubtitleTrack:
    index: int                   # ffmpeg stream index in the file (the `0:N` in a filter graph)
    number: int                  # position among the subtitle tracks, from 0 (the `0:s:N` specifier)
    codec: str
    language: str = ""
    title: str = ""
    default: bool = False
    forced: bool = False
    width: int = 0               # picture subtitles: the canvas they were authored for
    height: int = 0

    @property
    def is_image(self) -> bool:
        return self.codec in IMAGE_SUBTITLES

    @property
    def is_text(self) -> bool:
        return not self.is_image and self.codec not in UNBURNABLE_SUBTITLES

    @property
    def burnable(self) -> bool:
        return self.is_image or self.is_text

    @property
    def is_ass(self) -> bool:
        return self.codec in ("ass", "ssa")

    def label(self) -> str:
        parts = [SUBTITLE_NAMES.get(self.codec, self.codec.upper())]
        if self.language and self.language != "und":
            parts.append(self.language)
        if self.title:
            parts.append(f"“{self.title}”")
        flags = [f for f, on in (("default", self.default), ("forced", self.forced)) if on]
        if flags:
            parts.append(f"({', '.join(flags)})")
        return " ".join(parts)


@dataclass
class SourceInfo:
    path: str
    width: int
    height: int
    fps: float
    codec: str
    profile: str
    pix_fmt: str
    bit_depth: int
    bitrate_mbps: float          # whole file (container), the rate a reader must sustain
    duration_secs: float
    color_transfer: str
    color_primaries: str
    hdr_type: str                # SDR | HDR10 | HDR10+ | HLG | Dolby Vision
    dovi_profile: Optional[int]  # None when not Dolby Vision; 0 when the profile is unknown
    audio_codec: str
    audio_profile: str
    audio_channels: int
    audio_sample_rate: int
    audio_language: str
    subtitles: List[SubtitleTrack] = field(default_factory=list)
    font_attachments: int = 0    # fonts embedded in the file (MKV attachments), used by ASS subtitles

    @property
    def is_hdr(self) -> bool:
        return self.hdr_type != "SDR"

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_codec)

    def audio_label(self) -> str:
        if not self.audio_codec:
            return "none"
        name = AUDIO_NAMES.get(self.audio_codec, self.audio_codec.upper())
        if self.audio_profile and self.audio_profile.lower() not in name.lower():
            name += f" ({self.audio_profile})"
        return f"{name} {CHANNEL_NAMES.get(self.audio_channels, f'{self.audio_channels} ch')}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["is_hdr"] = self.is_hdr
        for track, entry in zip(self.subtitles, data["subtitles"]):
            entry["kind"] = "image" if track.is_image else "text" if track.is_text else "unsupported"
        return data


def _num(value: Any, kind=float, default=0):
    try:
        return kind(value)
    except (TypeError, ValueError):
        return default


def _rate(value: str) -> float:
    if not value or "/" not in str(value):
        return _num(value, float, 0.0)
    num, _, den = str(value).partition("/")
    n, d = _num(num), _num(den)
    return n / d if n > 0 and d > 0 else 0.0


def classify_hdr(stream: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    """HDR type and Dolby Vision profile, checked in order of specificity.

    BT.2020 primaries alone are not HDR (wide-gamut SDR exists); the transfer function decides.
    """
    side_data: List[Dict[str, Any]] = stream.get("side_data_list") or []
    dovi: Optional[int] = None
    for item in side_data:
        kind = str(item.get("side_data_type", "")).lower()
        if "dovi" in kind or "dolby vision" in kind:
            dovi = _num(item.get("dv_profile"), int, 0)
            break
    if dovi is None and str(stream.get("codec_tag_string", "")).lower() in ("dvh1", "dvhe", "dva1", "dvav"):
        dovi = 0
    if dovi is not None:
        return "Dolby Vision", dovi
    if any("2094" in str(i.get("side_data_type", "")) or "hdr10+" in str(i.get("side_data_type", "")).lower()
           for i in side_data):
        return "HDR10+", None
    transfer = str(stream.get("color_transfer", "")).lower()
    if transfer == "smpte2084":
        return "HDR10", None
    if transfer == "arib-std-b67":
        return "HLG", None
    return "SDR", None


def parse(data: Dict[str, Any], path: str) -> SourceInfo:
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
    if video is None:
        raise ProbeError("no video stream found")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fmt = data.get("format") or {}

    fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    pix_fmt = str(video.get("pix_fmt") or "unknown")
    depth = _num(video.get("bits_per_raw_sample"), int, 0)
    if not depth:
        depth = 12 if "12" in pix_fmt else 10 if "10" in pix_fmt else 8
    bitrate = _num(fmt.get("bit_rate")) or _num(video.get("bit_rate"))
    hdr_type, dovi = classify_hdr(video)
    subtitles = []
    for s in (s for s in streams if s.get("codec_type") == "subtitle"):
        tags, disp = s.get("tags") or {}, s.get("disposition") or {}
        subtitles.append(SubtitleTrack(
            index=_num(s.get("index"), int, 0), number=len(subtitles), codec=str(s.get("codec_name") or "unknown"),
            language=str(tags.get("language") or ""), title=str(tags.get("title") or ""),
            default=bool(disp.get("default")), forced=bool(disp.get("forced")),
            width=_num(s.get("width"), int, 0), height=_num(s.get("height"), int, 0)))
    fonts = sum(1 for s in streams if s.get("codec_type") == "attachment"
                and (str(s.get("codec_name") or "") in ("ttf", "otf")
                     or str((s.get("tags") or {}).get("mimetype") or "").startswith(("font/", "application/x-truetype",
                                                                                      "application/vnd.ms-opentype",
                                                                                      "application/x-font"))))
    return SourceInfo(
        path=path,
        width=_num(video.get("width"), int, 0), height=_num(video.get("height"), int, 0),
        fps=round(fps, 3) if fps else 0.0,
        codec=str(video.get("codec_name") or "unknown"), profile=str(video.get("profile") or ""),
        pix_fmt=pix_fmt, bit_depth=depth,
        bitrate_mbps=round(bitrate / 1e6, 2), duration_secs=round(_num(fmt.get("duration")), 3),
        color_transfer=str(video.get("color_transfer") or "unknown"),
        color_primaries=str(video.get("color_primaries") or "unknown"),
        hdr_type=hdr_type, dovi_profile=dovi,
        audio_codec=str(audio.get("codec_name") or ""), audio_profile=str(audio.get("profile") or ""),
        audio_channels=_num(audio.get("channels"), int, 0), audio_sample_rate=_num(audio.get("sample_rate"), int, 0),
        audio_language=str((audio.get("tags") or {}).get("language") or ""),
        subtitles=subtitles, font_attachments=fonts,
    )


def probe(ffprobe: str, path: str) -> SourceInfo:
    cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                             timeout=PROBE_TIMEOUT, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise ProbeError(f"ffprobe did not finish within {PROBE_TIMEOUT} s") from None
    except OSError as exc:
        raise ProbeError(f"could not run ffprobe: {exc}") from None
    if out.returncode != 0:
        raise ProbeError((out.stderr or "ffprobe failed").strip().splitlines()[-1][:300])
    try:
        data = json.loads(out.stdout)
    except ValueError:
        raise ProbeError("ffprobe returned unreadable output") from None
    return parse(data, path)


def dovi_notes(profile: Optional[int]) -> List[str]:
    """What a Dolby Vision profile means for a transcode."""
    if profile is None:
        return []
    notes = {
        5: "Dolby Vision profile 5: single layer with no HDR10 fallback. Colours are only correct with a "
           "Dolby Vision-aware tone mapper (jellyfin-ffmpeg has one); other builds give wrong colours.",
        7: "Dolby Vision profile 7: HDR10 base layer plus an enhancement layer (MEL or FEL). ffmpeg decodes "
           "the base layer and ignores the enhancement layer.",
        8: "Dolby Vision profile 8: single layer with an HDR10, HLG or SDR compatible base, decoded as that base.",
    }
    first = notes.get(profile, f"Dolby Vision profile {profile or 'unknown'}: check that your ffmpeg build supports it.")
    return [first, "Transcoding drops Dolby Vision metadata: the output is SDR when tone-mapped, otherwise the base layer."]
