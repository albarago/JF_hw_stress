"""Build one ffmpeg command line for one stream (a pure function, so it can be tested without ffmpeg)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

from .platforms import SW_TONEMAP, Platform
from .probe import SourceInfo
from .scenarios import Scenario, output_height
from .tools import Capabilities

AUDIO_MODES = ("aac", "copy", "none")


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


def _vf(platform: Platform, scenario: Scenario, source: SourceInfo, caps: Capabilities,
        decode_hw: bool, encode_hw: bool, width: int, height: int) -> Tuple[str, bool, str]:
    """The filter chain, whether frames stay on the GPU, and how tone mapping is done."""
    frames_on_gpu = decode_hw and bool(platform.hwaccel_output_format)
    tonemap = ""
    if scenario.tonemap and not source.is_hdr:
        raise CommandError("tone mapping needs an HDR source")
    can_sw_tonemap = caps.has_filter("zscale")

    if frames_on_gpu and encode_hw:
        chain = [f"{platform.scale_filter}=w={width}:h={height}{platform.scale_format}"]
        if scenario.tonemap:
            if platform.hw_tonemap(caps):
                chain.append(platform.tonemap_filter)
                tonemap = "hardware"
            elif can_sw_tonemap:
                chain += ["hwdownload", "format=p010le", SW_TONEMAP]
                if platform.upload:
                    chain.append(platform.upload)
                tonemap = "software"
            else:
                raise CommandError(_no_tonemap(platform))
        return ",".join(chain), True, tonemap

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
    if encode_hw and platform.upload:
        chain.append(platform.upload)
    return ",".join(chain), False, tonemap


def _no_tonemap(platform: Platform) -> str:
    return (f"this ffmpeg build cannot tone-map HDR for {platform.name}: it has neither "
            f"{platform.tonemap_requires or 'a hardware tone-mapping filter'} nor zscale. "
            "Use jellyfin-ffmpeg, or a scenario without tone mapping.")


def build(platform: Platform, scenario: Scenario, source: SourceInfo, caps: Capabilities, ffmpeg: str,
          stream_id: int, work_dir: str, options: Options, hw_decode: bool = True, hw_encode: bool = True) -> Plan:
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
    decode_hw = hw_decode and bool(platform.hwaccel) and source.codec in platform.decode_codecs
    width = scenario.width
    height = output_height(scenario, source)
    vf, frames_on_gpu, tonemap = _vf(platform, scenario, source, caps, decode_hw, encode_hw, width, height)

    seg = max(1, int(options.hls_segment_secs))
    gop = max(1, int(round((source.fps or 24) * seg)))
    playlist = os.path.join(work_dir, f"stream{stream_id}.m3u8")
    argv: List[str] = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-nostats",
                       "-progress", "pipe:1", "-stats_period", "0.5"]
    if decode_hw or (encode_hw and platform.upload):
        argv += list(platform.init_args)
    if decode_hw:
        argv += ["-hwaccel", platform.hwaccel]
        if platform.hwaccel_device:
            argv += ["-hwaccel_device", platform.hwaccel_device]
        if platform.hwaccel_output_format:
            argv += ["-hwaccel_output_format", platform.hwaccel_output_format]
        if source.codec in platform.decoders:
            argv += ["-c:v", platform.decoders[source.codec]]
    if options.loop:
        argv += ["-stream_loop", "-1"]
    if options.realtime:
        argv += ["-re"]
    argv += ["-i", source.path, "-map", "0:v:0"]
    if options.audio != "none" and source.has_audio:
        argv += ["-map", "0:a:0"]
    argv += ["-vf", vf, "-c:v", encoder, "-b:v", scenario.bitrate, "-maxrate", scenario.bitrate,
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
                frames_on_gpu=frames_on_gpu, tonemap=tonemap, output=output, playlist=playlist)
