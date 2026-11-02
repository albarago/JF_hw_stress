# jf-hw-stress

Find out how many transcodes your Jellyfin machine can run at once.

jf-hw-stress starts Jellyfin-style HLS transcodes of a video file you choose, adds one stream at a time, and
watches each stream's frame rate. When a stream can no longer keep up with real time, it reports the last stream
count that did. It uses the same ffmpeg that Jellyfin uses and similar settings, so the result is close to what
your server can actually deliver.

```text
$ jf-hw-stress --headless --source avatar-2160p-hdr.mkv --scenario 4k_hdr_sdr
…
result: 11 stable stream(s) (stopped: limit-reached)
```

That example is a MacBook Air (Apple M5) converting a 4K Dolby Vision film (HEVC, 44 Mbps) to 1080p SDR with
hardware tone mapping: 11 streams at the same time, each at 90% of real time or better.

- [What it measures](#what-it-measures)
- [Install](#install)
- [Quick start](#quick-start)
- [Test modes](#test-modes)
- [Scenarios](#scenarios)
- [Hardware support](#hardware-support)
- [Which ffmpeg to use](#which-ffmpeg-to-use)
- [Choosing a source file](#choosing-a-source-file)
- [Subtitle burn-in](#subtitle-burn-in)
- [Work folder and disk writes](#work-folder-and-disk-writes)
- [Reading the results](#reading-the-results)
- [Automation (CI, Kubernetes, scripts)](#automation-ci-kubernetes-scripts)
- [Before you run it](#before-you-run-it)
- [Contributing and license](#contributing-and-license)

## What it measures

Each stream is one ffmpeg process doing what Jellyfin does when a client can't play a file directly: decode the
source, scale it (and tone-map HDR to SDR if asked), encode H.264 or HEVC at a fixed bitrate, re-encode the audio
to stereo AAC, and write rolling HLS segments.

Streams run as fast as the hardware allows, and they share it. The tool measures each stream's frame rate over
the last 3 seconds. A stream passes while it stays at or above the **threshold**: 90% of the source frame rate
by default (21.6 fps for a 23.976 fps film). Use `--fps-ratio 1.0` for a strict real-time threshold.

With `--burn-in`, every stream also draws subtitles into the video, as Jellyfin does for clients that can't show
them (see [Subtitle burn-in](#subtitle-burn-in)).

It does **not** measure:

- Jellyfin's own overhead, network delivery to clients, or other server work (library scans, trickplay);
- direct play, which costs almost nothing;
- reading many different files from disk. Every stream reads the same file, so after the first pass it usually
  comes from the operating system's cache.

The result is a capacity estimate for one kind of transcode, so test the kinds your users actually trigger.

## Install

You need **Python 3.9 or newer** and **ffmpeg**. On a Jellyfin server, use the ffmpeg that comes with Jellyfin
(see [Which ffmpeg to use](#which-ffmpeg-to-use)).

**pipx** (recommended):

```sh
pipx install git+https://github.com/albarago/JF_hw_stress.git
jf-hw-stress --version
```

**Single file**: download `jf-hw-stress.pyz` from the
[latest release](https://github.com/albarago/JF_hw_stress/releases/latest). It includes its one dependency
([rich](https://github.com/Textualize/rich)) and runs with any Python 3.9+, including the `python3` that ships
with macOS:

```sh
curl -LO https://github.com/albarago/JF_hw_stress/releases/latest/download/jf-hw-stress.pyz
python3 jf-hw-stress.pyz --version
```

On Windows, run it with `py jf-hw-stress.pyz`.

**From source**:

```sh
git clone https://github.com/albarago/JF_hw_stress.git
cd JF_hw_stress
python3 -m pip install .
```

## Quick start

Run it with no arguments for a guided setup and a live dashboard:

```sh
jf-hw-stress
```

It asks which ffmpeg to use, which hardware backend, the source file, the work folder, the test mode and the
scenario, then shows every stream's frame rate as it runs. When it finds the limit, it offers to keep that load
running until you press Ctrl+C, or to stop and show the result. It can then save HTML and JSON reports.

To run without prompts:

```sh
# Find the limit for 4K HDR → 1080p SDR, write both reports
jf-hw-stress --headless --source movie.mkv --scenario 4k_hdr_sdr \
  --json-file result.json --html-report result.html

# Check that 6 streams of 1080p → H.264 hold up for 10 minutes
jf-hw-stress --headless --source movie.mkv --mode fixed --streams 6 --duration 600

# The same limit test with the file's subtitles burned in
jf-hw-stress --headless --source anime.mkv --burn-in auto

# See what this machine and ffmpeg build can use
jf-hw-stress --list-platforms
jf-hw-stress --list-scenarios
```

`jf-hw-stress --help` lists every option. Press Ctrl+C at any time; all ffmpeg processes are stopped and the work
folder is removed.

## Test modes

| Mode | `--mode` | What it does |
|---|---|---|
| Escalating | `escalating` (default) | Starts one stream and adds another every 12 s while all of them keep up. When one falls below the threshold for 5 s, it stops the newest stream and reports the last count that was stable. |
| Fixed | `fixed` | Runs `--streams N` for `--duration` seconds and reports how many kept up, judged by each stream's median frame rate. Use it to confirm a number, or for a soak test. |
| Hybrid | `hybrid` | Escalates on the hardware first. At the hardware limit it adds software (CPU) streams on top, and reports both counts. This shows how much the CPU adds when the GPU is full. |
| Mixed | `mixed` | Escalates with a random scenario for each new stream, drawn from a pool of 14 resolutions, codecs and bitrates, as a household with mixed clients would. `--seed` repeats the same sequence. |

The timing is adjustable: `--ramp-interval` (seconds stable before adding a stream), `--fail-secs` (seconds below
the threshold that count as failure), `--warmup-secs` (seconds after a launch before streams are judged) and
`--max-streams` (default 64).

## Scenarios

| `--scenario` | Output | Notes |
|---|---|---|
| `4k_hdr_sdr` | 1080p HEVC 8 Mbps, tone-mapped to SDR | The heaviest common transcode. Needs an HDR source. |
| `4k_1080p` | 1080p HEVC 8 Mbps | Downscale without tone mapping |
| `4k_720p` | 720p H.264 4 Mbps | |
| `1080p_compat` (default) | 1080p H.264 8 Mbps | Older clients that can't play the source codec |
| `1080p_webdl` | 1080p HEVC 6 Mbps | |
| `1080p_bluray` | 1080p HEVC 25 Mbps | High bitrate, more work for the encoder |
| `720p_mobile` | 720p H.264 4 Mbps | |
| `480p_remote` | 480p H.264 2 Mbps | A slow remote connection |

The output height follows the source's aspect ratio. Scenarios never upscale. A scenario wider than the source is
refused, and mixed mode leaves such scenarios out of its pool. Tone-mapping scenarios need an HDR source
(HDR10, HDR10+, HLG or Dolby Vision).

Output is always 8-bit, as Jellyfin produces for clients that need a transcode. Audio is re-encoded to stereo AAC
at 192 kbps by default, which is what most clients receive. `--audio copy` and `--audio none` remove that CPU work
from the test.

## Hardware support

| Backend | `--platform` | Systems | Status | HDR → SDR tone mapping |
|---|---|---|---|---|
| Apple VideoToolbox | `vt` | macOS | **Tested** (Apple M5) | On the GPU with jellyfin-ffmpeg (`tonemap_videotoolbox`) |
| NVIDIA NVENC | `nvenc` | Linux, Windows | Untested | On the GPU if ffmpeg has `tonemap_cuda`, otherwise on the CPU |
| Intel Quick Sync | `qsv` | Linux, Windows | Untested | On the CPU |
| VA-API (Intel, AMD) | `vaapi` | Linux | Untested | On the GPU if ffmpeg has `tonemap_vaapi`, otherwise on the CPU |
| AMD AMF | `amf` | Windows | Untested | On the CPU |
| Software | `sw` | all | **Tested** | On the CPU (`zscale`) |

The backend is detected automatically. `--platform` picks one (or several, comma-separated, used in turn). A
backend this machine or ffmpeg build can't run is refused with the reason; the tool never falls back silently.

**Untested** means the ffmpeg commands are built from the ffmpeg documentation and Jellyfin's own command lines
but haven't been run on that hardware yet. If you have it, a test run and a report would help a lot (see
[CONTRIBUTING.md](CONTRIBUTING.md)).

CPU tone mapping (`zscale`) is slow: about 112 fps for 4K on an M5 against 233 fps on its GPU. When the report says
`"tonemap": "software"`, the CPU may be the limit, not the GPU.

Subtitle burn-in on the GPU uses `overlay_videotoolbox` (tested), `overlay_cuda`, `overlay_qsv` or `overlay_vaapi`
(untested, built from Jellyfin's command lines). See [Subtitle burn-in](#subtitle-burn-in).

Hardware decoding is used when the backend supports the source codec. `--no-hw-decode` decodes on the CPU and
encodes on the GPU.

## Which ffmpeg to use

Use **jellyfin-ffmpeg**, the build Jellyfin ships, so you test what your server runs. The tool looks for it first:

- macOS: `/Applications/Jellyfin.app/Contents/MacOS`
- Linux and Docker: `/usr/lib/jellyfin-ffmpeg`, `/usr/share/jellyfin-ffmpeg`, `/opt/jellyfin-ffmpeg`
- Windows: `C:\Program Files\Jellyfin\Server`

If it isn't there, the tool uses the `ffmpeg` on your `PATH`. Point it elsewhere with `--ffmpeg`, giving a binary
or a folder that contains `ffmpeg` and `ffprobe`.

Stock ffmpeg builds (Homebrew, most Linux distributions) usually lack the hardware tone-mapping filters, and some
lack `zscale`. On those builds the HDR scenario uses CPU tone mapping, or is refused with an explanation if neither
is available. The other scenarios work with any build that has the hardware encoders.

## Choosing a source file

Use a real file of the kind your users play, for example a 4K HDR remux to test the worst case. A few minutes is
enough, because the tool loops the source.

Put the file on a **local disk**. The transcodes read it much faster than real time: 11 streams of a 44 Mbps file
at the limit read about 460 Mbit/s, and on a Wi-Fi link that becomes the bottleneck. To cut a short clip from a
larger file without re-encoding:

```sh
ffmpeg -ss 1800 -t 300 -i movie.mkv -map 0 -c copy clip.mkv
```

`-map 0` keeps the subtitle tracks and embedded fonts, which [subtitle burn-in](#subtitle-burn-in) needs.

Dolby Vision profile 7 (UHD Blu-ray) and 8 files are decoded as their HDR10 base layer, as Jellyfin does. Profile 5
has no HDR10 base layer; its colours are only correct with a Dolby Vision-aware tone mapper, and the tool warns
about it.

## Subtitle burn-in

Jellyfin burns subtitles into the video when the client can't show them itself. This is common with PGS
subtitles from Blu-rays and with styled ASS subtitles from anime, and it makes Jellyfin transcode files the
client could otherwise play directly. `--burn-in` adds the same work to every stream:

```sh
jf-hw-stress --headless --source anime.mkv --list-subtitles        # the tracks, numbered from 0
jf-hw-stress --headless --source anime.mkv --burn-in auto          # the default (or first) track
jf-hw-stress --headless --source film.mkv --burn-in 1              # subtitle track 1
jf-hw-stress --headless --source film.mkv --burn-in film.en.srt    # an external file (.ass .ssa .srt .vtt .sup)
jf-hw-stress --headless --source film.mkv --burn-in sample         # built-in styled ASS, for files without subtitles
```

The interactive setup asks after the audio question. It works the way Jellyfin's `EncodingHelper` does:

- **Text subtitles** (ASS, SSA, SRT, WebVTT) are extracted to a file once, before the run, together with the
  file's embedded fonts. libass then draws them (`subtitles=`).
- **Picture subtitles** (PGS, VobSub, DVB) go into the filter graph, are scaled to the output size and laid
  over the video.
- When decoding and encoding are both on the GPU, the subtitles are drawn on a transparent layer, uploaded and
  overlaid on the GPU: `overlay_videotoolbox`, `overlay_cuda`, `overlay_qsv` or `overlay_vaapi`. Otherwise they
  are drawn on the CPU (`subtitles=` in the chain, or `overlay`), which is also what happens when the ffmpeg build
  lacks the GPU overlay filter. Each stream's report says which (`streams[].burn_in`).
- Subtitles are drawn after scaling and tone mapping, as in Jellyfin.

The source is looped, so text subtitles are repeated with shifted times to keep them on screen after the first
pass. Picture subtitles loop with the video.

`sample` is a worst case: two lines of dialogue on screen at all times, plus a moving, rotating, blurred sign
and a karaoke line, so libass redraws every frame. Real dialogue subtitles cost much less.

What it measured on a MacBook Air (Apple M5, VideoToolbox, jellyfin-ffmpeg 8.1), 1080p → H.264 8 Mbps:

| Source | Without burn-in | With burn-in |
|---|---|---|
| H.264 anime episode, ASS dialogue with 21 embedded fonts | 12 streams | 12 streams |
| HEVC 10-bit film, PGS | 12 streams | 12 streams |
| AV1 10-bit anime film, typeset ASS | 12 streams | 11 streams |
| H.264, `sample` | — | 12 streams (one stream alone: 154 fps instead of 272 fps) |

On this machine the media engine is the limit, and libass runs on the CPU cores beside it, so burn-in barely
changes the count. On a machine with a weaker CPU, or a GPU that also does the subtitle work (an Intel iGPU), the
difference can be larger. That's worth measuring on your own hardware.

Not covered: DVB teletext and closed captions (nothing to draw), and Jellyfin's Vulkan overlay for AMD on Linux
(VA-API here uses `overlay_vaapi`, Jellyfin's Intel path). AMF burns in on the CPU; Jellyfin uses OpenCL.

## Work folder and disk writes

Each stream writes HLS segments, keeping the last few and deleting older ones. A run folder named
`jf-hw-stress-<date>-<time>-<pid>` is created inside the work folder and removed at the end. Only the files the
tool wrote are deleted. `--keep-files` keeps the run folder, with each stream's ffmpeg log.

The default work folder is `/dev/shm` (memory) on Linux and the temp folder elsewhere. `--work-dir` chooses
another.

The writes add up: the 11-stream example wrote 1.7 GB in 3.5 minutes, at up to 18 MB/s. That's fine for an
occasional test on an SSD. For long soak tests on macOS, use a RAM disk:

```sh
diskutil erasevolume HFS+ JFStress $(hdiutil attach -nomount ram://4194304)   # 2 GB
jf-hw-stress --work-dir /Volumes/JFStress
diskutil eject /Volumes/JFStress
```

## Reading the results

The headline number is **max stable streams**: the most streams that all stayed above the threshold for a full
ramp interval. In fixed mode it is **streams keeping up**: how many of the N streams had a median frame rate at or
above the threshold.

**Mbps out** is the bitrate each stream actually produced, measured from its finished segments, next to the target
(for example `6.0 / 8`). It includes the audio and container overhead, so it is what a client would download.
Hardware encoders often come in under the target: on an Apple M5, the 8 Mbps HEVC scenario averages about 6 Mbps.
Jellyfin's dashboard shows the target, not this measured value.

The run ends for one of these reasons (`result.stop_reason` in the JSON):

| Reason | Meaning |
|---|---|
| `limit-reached` | A stream fell behind; the previous count is the result. |
| `max-streams` | `--max-streams` was reached while every stream still kept up. The real limit is higher. |
| `duration` | `--duration` ended the run. |
| `stream-failed` | ffmpeg failed or stalled (see `result.failure.reason` and the ffmpeg log). This isn't a capacity result. |
| `all-streams-ended` | Every stream reached the end of the file (only with `--no-loop`). |
| `user` | Stopped with Ctrl+C, a stop signal, or from the dashboard. |
| `error` | The tool itself hit an unexpected error. |

A stream that produces no frames for 30 s is replaced once, with a warning in the report, because a stall says
nothing about capacity. A second stall ends the run as `stream-failed`.

The HTML report has a summary, the source and settings, a chart of streams and combined frame rate over time, and a
table of every stream. The JSON report has the same data and more; its fields are documented in
[docs/json-schema.md](docs/json-schema.md).

## Automation (CI, Kubernetes, scripts)

`--headless` never prompts. `--json` prints the report to stdout; progress goes to stderr.

```sh
jf-hw-stress --headless --source /media/sample.mkv --scenario 4k_hdr_sdr --json > result.json
jq '.result.max_stable_streams' result.json
```

| Exit code | Meaning |
|---|---|
| 0 | A result was produced. |
| 1 | The run failed: a stream crashed or stalled, or not even one stream kept up. |
| 2 | Setup error: bad option, missing file, no usable ffmpeg or backend, or a scenario this build can't run. |
| 130 | Interrupted. |

For a Kubernetes node, run the single-file build in a pod that has Python 3, jellyfin-ffmpeg, and the GPU device
your Jellyfin pod uses (for example `/dev/dri` for VA-API and Quick Sync, or the NVIDIA device plugin). Set
`--duration` so a job can't run forever, and use `--work-dir` on an `emptyDir` with `medium: Memory`.

The JSON includes the source file's path. Remove it before sharing a report if the path is private.

## Before you run it

- The test loads your GPU and CPU fully, on purpose. On a live Jellyfin server, run it when nobody is watching:
  it competes with real streams, and its result is lower while they run.
- Laptops get hot and fans spin up. An Apple M5 MacBook Air ran the full 4K HDR test without throttling, but
  check your own machine.
- The source file is only read, never modified. Nothing is sent over the network.

## Contributing and license

Bug reports, test results from untested hardware and pull requests are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

MIT licensed. See [LICENSE](LICENSE).
