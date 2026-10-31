# Changelog

## 2.1.0 (unreleased)

### Added

- Output bitrate: each stream's measured bitrate next to its target in the dashboard ("Mbps out"), the average in
  the summary, headless output and HTML report, and `streams[].output_mbps` and `result.avg_output_mbps` in the JSON
  report.

### Fixed

- Stopping a run sent ffmpeg a second stop signal, which made it exit at once and leave an empty last segment and an
  empty playlist behind (seen with `--keep-files`). It now gets one signal and finishes cleanly.

## 2.0.0 (2026-10-31)

A rewrite. Several v1 results were wrong (see [the audit](docs/AUDIT-2026-10-31.md)), so v2 numbers are not
comparable with v1 numbers, and the JSON report has a new layout (`schema_version: 2`).

### Fixed

- **Pass/fail used the wrong frame rate.** v1 judged streams by ffmpeg's `fps=`, an average since the stream
  started, so a stream that slowed down kept passing and "max stable streams" was overstated. v2 measures each
  stream over the last 3 seconds.
- After a failure, the newest stream is stopped so the remaining load is the stable count.
- A stream that crashes or never starts ends the run as a failure. v1 kept adding streams indefinitely.
- The source is looped, so short files no longer end a test early.
- Progress is read from a pipe as it arrives. v1 re-read every stream's growing log five times a second, which
  got slower over a run and competed with the encoders it measured.
- ffmpeg no longer reads the keyboard (`-nostdin`), and Ctrl+C stops every stream cleanly.
- The program no longer runs twice, and the installed `jf-hw-stress` command now reads its options.
- HLG sources are no longer reported as HDR10. Dolby Vision profile notes are corrected (profile 5 is the single-layer
  one with no HDR10 fallback; profile 7 is dual-layer).
- The estimated read rate was 8 times too high (megabits shown as megabytes).
- Hybrid and mixed modes report each stream's real encoder and scenario.
- Values from the media file are escaped in the HTML report.
- Output is always 8-bit, and keyframes line up with HLS segments for any frame rate.
- The Windows GPU name no longer relies on `wmic`, which current Windows 11 removed.

### Changed

- `--platform` (alias `--force-platform`) refuses a backend this machine or ffmpeg build can't run, instead of
  starting streams that fail. Several backends can be given, comma-separated.
- An unknown `--scenario`, or one that doesn't fit the source (upscaling, tone mapping an SDR file), is an error.
  Mixed mode leaves out scenarios that don't fit the source.
- HDR scenarios on an ffmpeg build without tone mapping are refused with an explanation.
- Each run uses its own work folder, removed afterwards. v1 left segments behind and deleted any `st_*` file in the
  chosen folder. `--cache-dir` is now `--work-dir` (the old name still works).
- Exit codes: 0 result, 1 run failed, 2 setup error, 130 interrupted.
- Installable as a package (`pipx install …`, `python -m jf_hw_stress`), with a single-file `jf-hw-stress.pyz`
  attached to each release.
- MIT license.

### Added

- `--realtime` (read the source at playback speed), `--no-loop`, `--keep-files`, `--seed`, `--max-streams`,
  `--warmup-secs`, `--hls-segment`, `--audio`, `--list-platforms`, and `--version`.
- `--ffmpeg` accepts a binary or a folder (`--ffmpeg-dir` still works). `--hw-decode`, which did nothing, is
  removed; hardware decoding is the default and `--no-hw-decode` turns it off.
- A chart of streams and frame rate over time in the HTML report.
- Unit tests, a software integration test, and CI on Linux, macOS and Windows.

### JSON report: v1 → v2

The full v2 layout is in [docs/json-schema.md](docs/json-schema.md). Timestamps are now ISO 8601 strings with a
time zone instead of Unix seconds. Modes and scenarios are ids (`escalating`, `4k_hdr_sdr`) instead of labels.

| v1 field | v2 field |
|---|---|
| `version` | `tool.version` |
| `timestamp` | `started_at` (and `ended_at`) |
| `hardware.platform` | `platforms[].name` (and `platforms[].id`) |
| `hardware.gpu` | `platforms[].device` |
| `hardware.ffmpeg_path`, `hardware.ffprobe_path` | `ffmpeg.path`, `ffmpeg.ffprobe` |
| `source.*` | `source.*` (same names, plus `duration_secs`, `color_*`, `dovi_profile`, `audio_*`, `is_hdr`) |
| `config.mode` (label) | `config.mode` (id) |
| `config.scenario` (label) | `config.scenario` (id; `null` in mixed mode, see `config.scenario_pool`) |
| `config.target_codec`, `config.target_bitrate`, `config.tonemap` | `streams[].output`, `streams[].tonemap` |
| `config.hls_segment_secs` | `config.hls.segment_secs` |
| `config.encode_hw` | `streams[].encode.hw` (per stream) |
| `config.decode_hw` | `config.hw_decode` (requested), `streams[].decode.hw` (actual) |
| `results` | `result` |
| `results.streams` | `streams` (top level) |
| `results.total_elapsed_secs` | `duration_secs` |
| `results.total_streams` | length of `streams` |
| `results.active_streams` | removed (every stream is stopped when the report is written) |
| `results.combined_speed` | `result.peak_combined_fps`, `timeline[].combined_fps` |
| `results.hw_saturated` | `result.hw_max_stable_streams` is not `null` |
| `results.max_stable_streams` | `result.max_stable_streams` |
| `results.fps_threshold` | `config.thresholds.threshold_fps` |
| `results.failure_reason` | `result.failure.reason` (with `stream` and `kind`) |
| `streams[].label`, `streams[].scenario_id` | `streams[].scenario_label`, `streams[].scenario` |
| `streams[].fps` (average since start) | `streams[].fps.p50` and the rest of `streams[].fps` |
| `streams[].fps_stats.*` | `streams[].fps.*` (`p95` replaced by `p5`, the slow end) |
| `streams[].speed` | removed: `streams[].fps.p50 / source.fps` |
| `streams[].bitrate` | `streams[].output` |
| `streams[].encode_hw`, `streams[].decode_hw` | `streams[].encode.hw`, `streams[].decode.hw` |
| `streams[].duration_secs` | removed: `ended_at - started_at` |
| `io.total_written_mb` | `io.segments_written_mb` (segments only; v1 included logs) |
| `io.last_write_mbs` | `io.peak_write_mb_s` |
| `io.last_read_mbs_est` | `timeline[].read_mbps` (megabits per second, now correct) |
| `io.cache_dir` | `io.work_dir` |

New in v2: `schema_version`, `host`, `config.thresholds`, `config.audio`, `config.loop`, `config.realtime`,
`config.seed`, `result.stop_reason`, `result.limit_reached`, `result.streams_keeping_up`,
`result.peak_concurrent_streams`, `streams[].encode.encoder`, `streams[].platform`, `timeline`, `events`,
`warnings`.

## 1.5.1 and earlier

See the git history.
