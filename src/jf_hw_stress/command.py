"""Build one ffmpeg command line for one stream (a pure function, so it can be tested without ffmpeg)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .platforms import SW_TONEMAP, Platform
from .probe import SourceInfo
from .scenarios import Scenario, output_height
from .subtitles import Prepared
from .tools import Capabilities

AUDIO_MODES = ("aac", "copy", "none")
CPU_OVERLAY = "overlay=eof_action=pass:repeatlast=0"


class CommandError(ValueError):
    pass


@dataclass(frozen=True)
class Options:
    hls_segment_secs: int = 4
    hls_list_size: int = 3
    audio: str = "aac"           # aac: stereo AAC 192k, like most Jellyfin clients get | copy | none
    loop: bool = True            # loop the source so short files never end a test
    realtime: bool = False       # read the source at playback speed (-re) instead of as fast as possible


@dataclass(frozen=True)
class Plan:
    """What a stream will actually do, worked out before it starts."""
    argv: List[str]
    encoder: str
    encode_hw: bool
    decode_hw: bool
    frames_on_gpu: bool
    tonemap: str                 # "" | "hardware" | "software"
    output: str                  # e.g. "1920x1080 h264 8M"
    playlist: str
    burn_in: str = ""            # "" | how subtitles are drawn, e.g. "GPU (overlay_videotoolbox)"


@dataclass(frozen=True)
class Filters:
    """The video filters: one chain (-vf), or a graph whose subtitle branch is laid over the main chain."""
    main: str
    frames_on_gpu: bool
    tonemap: str
    sub: str = ""                # subtitle branch; empty = no overlay, `main` is the whole chain
    sub_input: str = ""          # its input pad, e.g. "[0:3]" ("" when the branch starts with alphasrc)
    overlay: str = ""            # [main][sub] → output, plus anything after the overlay
    burn_in: str = ""


# --- subtitle pieces, as Jellyfin builds them (EncodingHelper) ---------------------------------

def escape_filter_path(path: str) -> str:
    """A path inside a quoted filter option (MediaEncoder.EscapeSubtitleFilterPath, minus the shell quoting)."""
    return path.replace("\\", "/").replace(":", "\\:").replace("'", "'\\\\\\''")


def subtitles_filter(prep: Prepared, alpha: bool) -> str:
    """libass drawing a text subtitle file (GetTextSubtitlesFilter). `alpha` draws onto a transparent canvas."""
    f = f"subtitles=f='{escape_filter_path(prep.text_file)}'"
    if alpha:
        f += ":alpha=1:sub2video=1"
    if prep.fonts_dir:
        f += f":fontsdir='{escape_filter_path(prep.fonts_dir)}'"
    return f


def image_sub_scale(prep: Prepared, source: SourceInfo, width: int, height: int) -> str:
    """Picture subtitles scaled to the output (GetGraphicalSubPreProcessFilters): stretched when the canvas has
    the video's shape, otherwise fitted by height and padded or cropped to the output."""
    cw, ch = prep.canvas
    video_dar = source.width / source.height if source.width and source.height else 0
    if not (cw and ch and video_dar) or abs(video_dar - cw / ch) < 0.01:
        return f"scale,scale={width}:{height}:fast_bilinear"
    return (f"scale,scale=-1:{height}:fast_bilinear,crop,pad=max({width}\\,iw):max({height}\\,ih):"
            f"(ow-iw)/2:(oh-ih)/2:black@0,crop={width}:{height}")


def sub_canvas(platform: Platform, width: int, height: int) -> Tuple[int, int]:
    """QSV and VA-API draw subtitles at most 1080 lines high and let the overlay scale them."""
    limit = platform.sub_max_height
    if limit and height > limit:
        return max(2, int(round(width * limit / height / 2)) * 2), limit
    return width, height


def alphasrc(prep: Prepared, source: SourceInfo, width: int, height: int) -> str:
    """The transparent canvas libass draws on for a GPU overlay (GetAlphaSrcFilter): full frame rate for ASS
    (animated), 10 fps for plain text."""
    rate = min(source.fps or 25, 60) if prep.burn.is_ass else 10
    return f"alphasrc=s={width}x{height}:r={rate:g}:start=0"


def _gpu_sub_branch(platform: Platform, prep: Prepared, source: SourceInfo, width: int, height: int) -> str:
    w, h = sub_canvas(platform, width, height)
    if prep.burn.is_text:
        first = [alphasrc(prep, source, w, h), f"format={platform.sub_format}", subtitles_filter(prep, alpha=True)]
    else:
        first = [image_sub_scale(prep, source, w, h), f"format={platform.sub_format}"]
    return ",".join(first + [platform.sub_upload])


def _gpu_overlay(platform: Platform, prep: Prepared, caps: Capabilities, width: int, height: int) -> str:
    f = f"{platform.overlay_filter}=eof_action=pass:repeatlast=0"
    if platform.sub_max_height:
        f += f":w={width}:h={height}"
    if platform.id == "nvenc" and prep.burn.is_text and caps.jellyfin_build:
        f += ":alpha_format=premultiplied"
    return f


# --- the filters -------------------------------------------------------------------------------

def _filters(platform: Platform, scenario: Scenario, source: SourceInfo, caps: Capabilities, decode_hw: bool,
             encode_hw: bool, width: int, height: int, prep: Optional[Prepared] = None) -> Filters:
    frames_on_gpu = decode_hw and bool(platform.hwaccel_output_format)
    tonemap = ""
    if scenario.tonemap and not source.is_hdr:
        raise CommandError("tone mapping needs an HDR source")
    can_sw_tonemap = caps.has_filter("zscale")
    if frames_on_gpu and encode_hw:
        chain = [f"{platform.scale_filter}=w={width}:h={height}{platform.scale_format}"]
        still_on_gpu = True
        if scenario.tonemap:
            if platform.hw_tonemap(caps):
                chain.append(platform.tonemap_filter)
                tonemap = "hardware"
            elif can_sw_tonemap:
                chain += ["hwdownload", "format=p010le", SW_TONEMAP]
                if platform.upload:
                    chain.append(platform.upload)
                else:
                    still_on_gpu = False   # the encoder takes the tone-mapped frames from system memory
                tonemap = "software"
            else:
                raise CommandError(_no_tonemap(platform))
        if prep is None:
            return Filters(",".join(chain), True, tonemap)
        if still_on_gpu and platform.hw_overlay(caps, prep.burn.is_text):
            return Filters(",".join(chain), True, tonemap,
                           sub=_gpu_sub_branch(platform, prep, source, width, height),
                           sub_input="" if prep.burn.is_text else f"[{prep.stream_spec}]",
                           overlay=_gpu_overlay(platform, prep, caps, width, height),
                           burn_in=f"GPU ({platform.overlay_filter})")
        # No overlay filter for these frames: bring them back, draw the subtitles, and hand them on.
        if still_on_gpu:
            chain += ["hwdownload", "format=nv12"]
        return _cpu_burn_in(chain, platform, prep, source, width, height, encode_hw, tonemap,
                            "CPU (frames copied from the GPU and back)")

    chain = []
    if frames_on_gpu:  # decoded on the GPU but encoded in software: bring frames back first
        chain += ["hwdownload", f"format={'p010le' if source.bit_depth > 8 else 'nv12'}"]
    chain.append(f"scale=w={width}:h={height}")
    if scenario.tonemap:
        if not can_sw_tonemap:
            raise CommandError(_no_tonemap(platform))
        chain.append(SW_TONEMAP)
        tonemap = "software"
    else:
        chain.append("format=yuv420p")
    if prep is not None:
        return _cpu_burn_in(chain, platform, prep, source, width, height, encode_hw, tonemap, "CPU")
    if encode_hw and platform.upload:
        chain.append(platform.upload)
    return Filters(",".join(chain), False, tonemap)


def _cpu_burn_in(chain: List[str], platform: Platform, prep: Prepared, source: SourceInfo, width: int, height: int,
                 encode_hw: bool, tonemap: str, where: str) -> Filters:
    """Subtitles drawn on frames in system memory: libass in the chain, or a CPU overlay for pictures."""
    upload = [platform.upload] if encode_hw and platform.upload else []
    if prep.burn.is_text:
        return Filters(",".join(chain + [subtitles_filter(prep, alpha=False)] + upload), False, tonemap,
                       burn_in=f"{where}, libass")
    return Filters(",".join(chain), False, tonemap, sub=image_sub_scale(prep, source, width, height),
                   sub_input=f"[{prep.stream_spec}]", overlay=",".join([CPU_OVERLAY] + upload),
                   burn_in=f"{where}, overlay")


def _no_tonemap(platform: Platform) -> str:
    return (f"this ffmpeg build cannot tone-map HDR for {platform.name}: it has neither "
            f"{platform.tonemap_requires or 'a hardware tone-mapping filter'} nor zscale. "
            "Use jellyfin-ffmpeg, or a scenario without tone mapping.")


LOOP_FILE = "loop.ffconcat"


def loop_list(path: str, duration_secs: float, copies: int) -> str:
    """An ffconcat list that plays the source `copies` times back to back.

    Looping this way instead of with -stream_loop: -stream_loop resets the decoder at every loop, and with
    VideoToolbox H.264 the reset brings frames back in a different format, so the filters fail ("hwaccel
    changed") and the stream ends after one pass. The concat demuxer keeps one decoder for the whole run.
    Timestamps keep rising across copies, as with -stream_loop.
    """
    quoted = "'" + path.replace("'", "'\\''") + "'"
    entry = f"file {quoted}\nduration {duration_secs:.3f}\n"
    return "ffconcat version 1.0\n" + entry * max(1, copies)


def build(platform: Platform, scenario: Scenario, source: SourceInfo, caps: Capabilities, ffmpeg: str,
          stream_id: int, work_dir: str, options: Options, hw_decode: bool = True, hw_encode: bool = True,
          burn: Optional[Prepared] = None, loop_input: str = "") -> Plan:
    """One stream's command. `loop_input` is a `loop_list` file; without one, a looped source uses -stream_loop."""
    if options.audio not in AUDIO_MODES:
        raise CommandError(f"audio must be one of {', '.join(AUDIO_MODES)}")
    encode_hw = hw_encode and platform.is_hardware
    if encode_hw:
        encoder = platform.encoders[scenario.codec]
        extra = list(platform.args_for(scenario.codec))
    else:
        encoder = "libx265" if scenario.codec == "hevc" else "libx264"
        extra = ["-preset", "fast"]
    if encoder not in caps.encoders:
        raise CommandError(f"this ffmpeg build has no {encoder} encoder")
    if burn is not None and burn.burn.is_text and not caps.has_filter("subtitles"):
        raise CommandError("this ffmpeg build has no subtitles filter (libass), so it cannot burn in text subtitles")
    decode_hw = hw_decode and bool(platform.hwaccel) and source.codec in platform.decode_codecs
    width = scenario.width
    height = output_height(scenario, source)
    f = _filters(platform, scenario, source, caps, decode_hw, encode_hw, width, height, burn)
    seg = max(1, int(options.hls_segment_secs))
    gop = max(1, int(round((source.fps or 24) * seg)))
    playlist = os.path.join(work_dir, f"stream{stream_id}.m3u8")
    argv: List[str] = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-nostats",
                       "-progress", "pipe:1", "-stats_period", "0.5"]
    if decode_hw or (encode_hw and platform.upload):
        argv += list(platform.init_args)
    if f.burn_in.startswith("GPU") and platform.overlay_init:
        argv += list(platform.overlay_init)
    if decode_hw:
        argv += ["-hwaccel", platform.hwaccel]
        if platform.hwaccel_device:
            argv += ["-hwaccel_device", platform.hwaccel_device]
        if platform.hwaccel_output_format:
            argv += ["-hwaccel_output_format", platform.hwaccel_output_format]
        if source.codec in platform.decoders:
            argv += ["-c:v", platform.decoders[source.codec]]
    if burn is not None and not burn.burn.is_text and burn.canvas[0] and not burn.image_input:
        argv += ["-canvas_size", f"{burn.canvas[0]}x{burn.canvas[1]}"]
    if options.loop and not loop_input:
        argv += ["-stream_loop", "-1"]
    if options.realtime:
        argv += ["-re"]
    argv += ["-f", "concat", "-safe", "0", "-i", loop_input] if options.loop and loop_input else ["-i", source.path]
    if burn is not None and burn.image_input:
        if options.loop:
            argv += ["-stream_loop", "-1"]
        if burn.canvas[0]:
            argv += ["-canvas_size", f"{burn.canvas[0]}x{burn.canvas[1]}"]
        argv += ["-i", burn.image_input]
    if f.overlay:
        graph = f"{f.sub_input}{f.sub}[sub];[0:v:0]{f.main}[main];[main][sub]{f.overlay}[vout]"
        argv += ["-filter_complex", graph, "-map", "[vout]"]
    else:
        argv += ["-map", "0:v:0"]
    if options.audio != "none" and source.has_audio:
        argv += ["-map", "0:a:0"]
    if not f.overlay:
        argv += ["-vf", f.main]
    argv += ["-c:v", encoder, "-b:v", scenario.bitrate, "-maxrate", scenario.bitrate,
             "-bufsize", f"{scenario.bitrate_mbps * 2}M", "-g", str(gop),
             "-force_key_frames", f"expr:gte(t,n_forced*{seg})"] + extra
    if options.audio == "none" or not source.has_audio:
        argv += ["-an"]
    elif options.audio == "copy":
        argv += ["-c:a", "copy"]
    else:
        argv += ["-c:a", "aac", "-b:a", "192k", "-ac", "2"]
    argv += ["-max_muxing_queue_size", "4096",
             "-f", "hls", "-hls_time", str(seg), "-hls_list_size", str(max(1, options.hls_list_size)),
             "-hls_flags", "delete_segments", "-hls_segment_type", "mpegts",
             "-hls_segment_filename", os.path.join(work_dir, f"stream{stream_id}_%05d.ts"), playlist]
    output = f"{width}x{height} {scenario.codec} {scenario.bitrate}"
    return Plan(argv=argv, encoder=encoder, encode_hw=encode_hw, decode_hw=decode_hw,
                frames_on_gpu=f.frames_on_gpu, tonemap=f.tonemap, output=output, playlist=playlist,
                burn_in=f.burn_in)
