# JSON report, schema version 2

`--json` (stdout), `--json-file FILE` and the interactive "save reports" option all write this document. The HTML
report is built from the same data.

`schema_version` changes only when a field is renamed, removed or changes meaning. New fields can be added within a
version, so ignore fields you don't know. The changes from v1 are in [CHANGELOG.md](../CHANGELOG.md).

Times are ISO 8601 strings with the local time zone. Frame rates are frames per second. A value that wasn't
measured is `null`.

## Top level

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | `2` |
| `tool` | object | `name` (`"jf-hw-stress"`) and `version` |
| `started_at`, `ended_at` | string | When the first stream started and the run ended |
| `duration_secs` | number | Length of the run |
| `host` | object | `os` (`Darwin`, `Linux`, `Windows`), `os_version`, `machine` (`arm64`, `x86_64`, …), `python` |
| `ffmpeg` | object | `path`, `ffprobe`, `version` (for example `8.1.3-Jellyfin`) and `origin` (where it was found) |
| `platforms` | array | Backends used, in round-robin order: `id`, `name`, `device` (GPU name), `tested` (whether this backend has been verified on real hardware) |
| `source` | object | The input file, below |
| `config` | object | The settings, below |
| `result` | object | The outcome, below |
| `io` | object | Disk writes, below |
| `streams` | array | One entry per stream started, below |
| `timeline` | array | One sample per second, below |
| `events` | array | `{t, message}`: what happened, with `t` in seconds from the start |
| `warnings` | array of strings | Anything that may affect the result, such as an untested backend or a replaced stream |

## `source`

| Field | Meaning |
|---|---|
| `path` | Path of the file as given. Remove it before sharing if it's private. |
| `width`, `height`, `fps`, `codec`, `profile`, `pix_fmt`, `bit_depth` | Video stream properties |
| `bitrate_mbps` | Overall bitrate in megabits per second |
| `duration_secs` | Length of the file |
| `color_transfer`, `color_primaries` | As reported by ffprobe, for example `smpte2084` and `bt2020` |
| `hdr_type` | `SDR`, `HDR10`, `HDR10+`, `HLG` or `Dolby Vision` |
| `dovi_profile` | Dolby Vision profile number, or `null` |
| `is_hdr` | `true` for any HDR type |
| `audio_codec`, `audio_profile`, `audio_channels`, `audio_sample_rate`, `audio_language` | First audio stream, `null` or empty when there is none |

## `config`

| Field | Meaning |
|---|---|
| `mode` | `fixed`, `escalating`, `hybrid` or `mixed` |
| `scenario` | Scenario id, `null` in mixed mode |
| `scenario_pool` | Mixed mode: the scenario ids streams were drawn from; otherwise `null` |
| `fixed_streams` | Fixed mode: the number of streams; otherwise `null` |
| `hw_decode` | Whether hardware decoding was requested (see `streams[].decode.hw` for what each stream did) |
| `audio` | `aac`, `copy` or `none` |
| `loop`, `realtime` | Whether the source was looped, and read at playback speed (`-re`) |
| `hls` | `segment_secs` and `list_size` |
| `thresholds` | `fps_ratio`, `threshold_fps` (the frame rate each stream had to keep), `ramp_interval_secs`, `fail_secs`, `warmup_secs` |
| `duration_limit_secs` | `--duration`, or `null` |
| `max_streams` | Stream cap |
| `seed` | Mixed-mode random seed, or `null` |

## `result`

| Field | Meaning |
|---|---|
| `stop_reason` | `limit-reached`, `max-streams`, `duration`, `stream-failed`, `all-streams-ended`, `user` or `error` (see the README) |
| `limit_reached` | `true` when the run found the machine's limit (`limit-reached`) or the stream cap (`max-streams`) |
| `max_stable_streams` | Escalating, hybrid and mixed modes: the most streams that all kept up for a full ramp interval. `null` in fixed mode. |
| `hw_max_stable_streams` | Hybrid mode: the count on hardware alone, before software streams were added; otherwise `null` |
| `streams_keeping_up` | Fixed mode: how many streams had a median frame rate at or above the threshold; otherwise `null` |
| `peak_concurrent_streams` | Most streams running at once |
| `peak_combined_fps` | Highest per-second sum of all streams' frame rates |
| `failure` | `null`, or `{stream, reason, kind}` for what ended the run. `kind` is `capacity` (a stream fell behind), `stall` (a stream produced no frames, twice) or `error` (ffmpeg exited with an error). |

## `io`

| Field | Meaning |
|---|---|
| `segments_written_mb` | Megabytes of HLS segments written during the run |
| `peak_write_mb_s` | Highest per-second write rate, in megabytes per second |
| `work_dir` | The run folder (removed after the run unless `--keep-files`) |

## `streams[]`

| Field | Meaning |
|---|---|
| `id` | 1, 2, 3, … in launch order |
| `scenario`, `scenario_label` | Scenario id and its label |
| `platform` | Backend id (`vt`, `nvenc`, `qsv`, `vaapi`, `amf`, `sw`) |
| `output` | For example `1920x1080 hevc 8M` |
| `encode` | `hw` (bool) and `encoder` (for example `hevc_videotoolbox`) |
| `decode` | `hw` (bool) |
| `tonemap` | `hardware`, `software` or `null` |
| `status` | `stopped` (ended by the tool) or `error` (ffmpeg exited on its own) |
| `started_at`, `ended_at` | Times |
| `frames` | Frames encoded |
| `fps` | Once-per-second samples of the 3-second frame rate: `samples`, `min`, `p5`, `p50`, `avg`, `max`. Early samples are high because a stream runs alone or with fewer peers at first; `p50` and `p5` describe the shared load better than `avg`. |
| `error` | The end of ffmpeg's log when it failed, otherwise `null` |

## `timeline[]`

One entry per second (halved in resolution every 3,600 entries, so long runs stay small).

| Field | Meaning |
|---|---|
| `t` | Seconds from the start |
| `streams` | Streams running |
| `combined_fps` | Sum of their frame rates |
| `min_fps` | The slowest stream's frame rate, or `null` before any was measurable |
| `read_mbps` | Estimated source read rate in megabits per second (each stream's speed × the source bitrate) |

## Example

From a MacBook Air (Apple M5), trimmed to one stream, two timeline entries and two events, with the
file paths replaced by typical ones.

```json
{
  "schema_version": 2,
  "tool": {"name": "jf-hw-stress", "version": "2.0.0"},
  "started_at": "2026-10-31T08:38:13-06:00",
  "ended_at": "2026-10-31T08:41:48-06:00",
  "duration_secs": 214.5,
  "host": {"os": "Darwin", "os_version": "25.6.0", "machine": "arm64", "python": "3.13.17"},
  "ffmpeg": {"path": "/Applications/Jellyfin.app/Contents/MacOS/ffmpeg",
             "ffprobe": "/Applications/Jellyfin.app/Contents/MacOS/ffprobe",
             "version": "8.1.3-Jellyfin", "origin": "jellyfin-ffmpeg (/Applications/Jellyfin.app/Contents/MacOS)"},
  "platforms": [{"id": "vt", "name": "Apple VideoToolbox", "device": "Apple M5", "tested": true}],
  "source": {"path": "/Users/me/clips/avatar-2160p-hdr-5min.mkv", "width": 3840, "height": 2160, "fps": 23.976,
             "codec": "hevc", "profile": "Main 10", "pix_fmt": "yuv420p10le", "bit_depth": 10,
             "bitrate_mbps": 43.8, "duration_secs": 300.716, "color_transfer": "smpte2084",
             "color_primaries": "bt2020", "hdr_type": "Dolby Vision", "dovi_profile": 7,
             "audio_codec": "dts", "audio_profile": "DTS-HD MA", "audio_channels": 6,
             "audio_sample_rate": 48000, "audio_language": "eng", "is_hdr": true},
  "config": {"mode": "escalating", "scenario": "4k_hdr_sdr", "scenario_pool": null, "fixed_streams": null,
             "hw_decode": true, "audio": "aac", "loop": true, "realtime": false,
             "hls": {"segment_secs": 4, "list_size": 3},
             "thresholds": {"fps_ratio": 0.9, "threshold_fps": 21.58, "ramp_interval_secs": 12.0,
                            "fail_secs": 5.0, "warmup_secs": 6.0},
             "duration_limit_secs": null, "max_streams": 64, "seed": null},
  "result": {"stop_reason": "limit-reached", "limit_reached": true, "max_stable_streams": 11,
             "hw_max_stable_streams": null, "streams_keeping_up": null, "peak_concurrent_streams": 12,
             "peak_combined_fps": 308.7,
             "failure": {"stream": 1, "reason": "stream 1 fell to 20.9 fps, under 21.6 fps for 5 s",
                         "kind": "capacity"}},
  "io": {"segments_written_mb": 1652.7, "peak_write_mb_s": 17.63,
         "work_dir": "/var/folders/xy/T/jf-hw-stress-20261031-083813-67175"},
  "streams": [
    {"id": 1, "scenario": "4k_hdr_sdr", "scenario_label": "4K HDR → 1080p SDR", "platform": "vt",
     "output": "1920x1080 hevc 8M", "encode": {"hw": true, "encoder": "hevc_videotoolbox"},
     "decode": {"hw": true}, "tonemap": "hardware", "status": "stopped",
     "started_at": "2026-10-31T08:38:13-06:00", "ended_at": "2026-10-31T08:41:47-06:00", "frames": 14171,
     "fps": {"samples": 205, "min": 20.83, "p5": 22.5, "p50": 42.02, "avg": 67.51, "max": 255.82},
     "error": null}
  ],
  "timeline": [
    {"t": 0.0, "streams": 1, "combined_fps": 0, "min_fps": null, "read_mbps": 0.0},
    {"t": 199.8, "streams": 12, "combined_fps": 250.2, "min_fps": 22.5, "read_mbps": 457.2}
  ],
  "events": [
    {"t": 0.0, "message": "stream 1 started: 4K HDR → 1080p SDR [HW vt] (1 running)"},
    {"t": 0.0, "message": "threshold 21.6 fps (90% of 23.976 fps)"}
  ],
  "warnings": []
}
```
