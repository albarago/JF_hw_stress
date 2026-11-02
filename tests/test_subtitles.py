"""Subtitle burn-in: probing, choosing, preparing, and the command lines for each backend."""
from dataclasses import replace

import pytest

from conftest import make_source
from jf_hw_stress import command, platforms, subtitles
from jf_hw_stress.command import CommandError, Options
from jf_hw_stress.engine import Engine, RunConfig, remove_work_dir
from jf_hw_stress.probe import SubtitleTrack, parse
from jf_hw_stress.scenarios import ALL
from jf_hw_stress.subtitles import BurnIn, BurnInError, Prepared
from jf_hw_stress.tools import Capabilities, Tools

COMPAT = ALL["1080p_compat"]          # 1920 wide H.264
HDR = ALL["4k_hdr_sdr"]
OVERLAY_FILTERS = {"overlay_videotoolbox", "overlay_cuda", "overlay_qsv", "overlay_vaapi", "alphasrc", "subtitles",
                   "overlay"}
MAC = Capabilities(hwaccels={"videotoolbox"},
                   encoders={"h264_videotoolbox", "hevc_videotoolbox", "libx264", "libx265"},
                   filters={"scale", "scale_vt", "tonemap_videotoolbox", "zscale", "hwdownload", "hwupload", "format"}
                   | OVERLAY_FILTERS, jellyfin_build=True)
MAC_NO_OVERLAY = replace(MAC, filters=MAC.filters - {"overlay_videotoolbox"})
LINUX = Capabilities(hwaccels={"cuda", "qsv", "vaapi"},
                     encoders={"h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv", "h264_vaapi", "hevc_vaapi",
                               "libx264", "libx265"},
                     filters={"scale", "scale_cuda", "scale_qsv", "scale_vaapi", "zscale", "hwdownload", "hwupload",
                              "format"} | OVERLAY_FILTERS, jellyfin_build=True)

PGS = SubtitleTrack(index=3, number=0, codec="hdmv_pgs_subtitle", language="eng", default=True, width=1920, height=1080)
ASS = SubtitleTrack(index=4, number=1, codec="ass", language="eng", title="Signs & Songs")
SDR_1080 = make_source(width=1920, height=1080, codec="h264", profile="High", pix_fmt="yuv420p", bit_depth=8,
                       color_transfer="bt709", color_primaries="bt709", hdr_type="SDR", subtitles=[PGS, ASS],
                       font_attachments=3)


@pytest.fixture(autouse=True)
def no_device_lookup(monkeypatch):
    monkeypatch.setattr(platforms, "device_name", lambda p: "Test GPU")


def text_prep(**over) -> Prepared:
    base = dict(burn=BurnIn("text", "track", "ass", "track 1", track=ASS), text_file="/run/burnin/subs.ass",
                fonts_dir="/run/burnin/fonts")
    base.update(over)
    return Prepared(**base)


def image_prep(**over) -> Prepared:
    base = dict(burn=BurnIn("image", "track", PGS.codec, "track 0", track=PGS), stream_spec="0:3", canvas=(1920, 1080))
    base.update(over)
    return Prepared(**base)


def build(platform, prep, caps=MAC, scenario=COMPAT, source=SDR_1080, **kw):
    return command.build(platform, scenario, source, caps, "/bin/ffmpeg", 1, "/work", kw.pop("options", Options()),
                         burn=prep, **kw)


def arg(plan, flag):
    return plan.argv[plan.argv.index(flag) + 1]


# --- probing -----------------------------------------------------------------------------------

def test_probe_lists_subtitle_tracks_and_fonts():
    data = {"format": {"duration": "60"}, "streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, "avg_frame_rate": "24/1"},
        {"index": 1, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle", "width": 1920, "height": 1080,
         "tags": {"language": "eng"}, "disposition": {"default": 1}},
        {"index": 2, "codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "spa", "title": "Forced"},
         "disposition": {"forced": 1}},
        {"index": 3, "codec_type": "subtitle", "codec_name": "dvb_teletext"},
        {"index": 4, "codec_type": "attachment", "codec_name": "ttf"},
        {"index": 5, "codec_type": "attachment", "tags": {"mimetype": "font/otf", "filename": "x.otf"}},
        {"index": 6, "codec_type": "attachment", "tags": {"mimetype": "image/jpeg"}},
    ]}
    src = parse(data, "/m.mkv")
    assert [(t.index, t.number, t.codec) for t in src.subtitles] == [
        (1, 0, "hdmv_pgs_subtitle"), (2, 1, "subrip"), (3, 2, "dvb_teletext")]
    pgs, srt, ttx = src.subtitles
    assert pgs.is_image and pgs.default and (pgs.width, pgs.height) == (1920, 1080)
    assert srt.is_text and srt.forced and srt.label() == "SRT spa “Forced” (forced)"
    assert not ttx.burnable
    assert src.font_attachments == 2
    assert [t["kind"] for t in src.to_dict()["subtitles"]] == ["image", "text", "unsupported"]


# --- choosing ----------------------------------------------------------------------------------

def test_resolve_auto_prefers_the_default_track():
    assert subtitles.resolve("auto", SDR_1080).track is PGS
    no_default = replace(SDR_1080, subtitles=[replace(PGS, default=False), ASS])
    assert subtitles.resolve("auto", no_default).track.number == 0


def test_resolve_by_number_sample_and_errors(tmp_path):
    burn = subtitles.resolve("1", SDR_1080)
    assert (burn.kind, burn.origin, burn.is_ass) == ("text", "track", True)
    assert subtitles.resolve("s:0", SDR_1080).kind == "image"
    assert subtitles.resolve("sample", SDR_1080).origin == "sample"
    with pytest.raises(BurnInError, match="no subtitle track 7"):
        subtitles.resolve("7", SDR_1080)
    with pytest.raises(BurnInError, match="no subtitle track that can be burned in"):
        subtitles.resolve("auto", make_source())
    with pytest.raises(BurnInError, match="nothing ffmpeg can draw"):
        subtitles.resolve("0", replace(SDR_1080, subtitles=[SubtitleTrack(2, 0, "dvb_teletext")]))
    srt, sup, txt = tmp_path / "a.srt", tmp_path / "b.sup", tmp_path / "c.txt"
    for f in (srt, sup, txt):
        f.write_text("x")
    assert (subtitles.resolve(str(srt), SDR_1080).kind, subtitles.resolve(str(sup), SDR_1080).kind) == ("text", "image")
    with pytest.raises(BurnInError, match="unsupported subtitle file type .txt"):
        subtitles.resolve(str(txt), SDR_1080)
    with pytest.raises(BurnInError, match="is not auto, sample"):
        subtitles.resolve("/no/such.ass", SDR_1080)


# --- following the loop ------------------------------------------------------------------------

ASS_TEXT = """[Script Info]
PlayResX: 1920

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Comment: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,not drawn
Dialogue: 0,0:00:01.50,0:00:03.25,Default,,0,0,0,,Hello, world
Dialogue: 0,0:59:59.90,1:00:00.95,Default,,0,0,0,,{\\an8}Late, line

[Fonts]
fontname: x.ttf
"""


def test_repeat_ass_shifts_each_copy_and_keeps_other_sections():
    out = subtitles.repeat_ass(ASS_TEXT, 3600.5, 3)
    lines = out.splitlines()
    dialogue = [ln for ln in lines if ln.startswith("Dialogue:")]
    assert dialogue == [
        "Dialogue: 0,0:00:01.50,0:00:03.25,Default,,0,0,0,,Hello, world",
        "Dialogue: 0,0:59:59.90,1:00:00.95,Default,,0,0,0,,{\\an8}Late, line",
        "Dialogue: 0,1:00:02.00,1:00:03.75,Default,,0,0,0,,Hello, world",
        "Dialogue: 0,2:00:00.40,2:00:01.45,Default,,0,0,0,,{\\an8}Late, line",
        "Dialogue: 0,2:00:02.50,2:00:04.25,Default,,0,0,0,,Hello, world",
        "Dialogue: 0,3:00:00.90,3:00:01.95,Default,,0,0,0,,{\\an8}Late, line"]
    assert lines.index("[Fonts]") > lines.index(dialogue[-1])     # copies go inside [Events]
    assert sum(ln.startswith("Comment:") for ln in lines) == 1
    assert subtitles.count_events(out, "ass") == 6


def test_repeat_srt_shifts_and_renumbers():
    srt = "1\n00:00:03,480 --> 00:00:05,551\nNu pakken\nde moffen ons.\n\n2\n00:59:59,999 --> 01:00:01,000\nWij\n"
    out = subtitles.repeat_srt(srt, 150.4, 2)
    assert out == ("1\n00:00:03,480 --> 00:00:05,551\nNu pakken\nde moffen ons.\n\n"
                   "2\n00:59:59,999 --> 01:00:01,000\nWij\n\n"
                   "3\n00:02:33,880 --> 00:02:35,951\nNu pakken\nde moffen ons.\n\n"
                   "4\n01:02:30,399 --> 01:02:31,400\nWij\n")


def test_sample_covers_the_length_with_animated_lines():
    text = subtitles.sample_ass(60, 1920, 1080)
    assert subtitles.count_events(text, "ass") == 45          # 15 cycles of 4 s, three lines each
    assert "\\move(" in text and "\\t(" in text and "\\k" in text and "PlayResY: 1080" in text
    last = [ln for ln in text.splitlines() if ln.startswith("Dialogue:")][-1]
    assert last.split(",")[2] == "0:01:00.00"


def test_loop_list_repeats_the_source():
    text = command.loop_list("/media/it's.mkv", 151.735, 3)
    assert text.splitlines() == ["ffconcat version 1.0"] + ["file '/media/it'\\''s.mkv'", "duration 151.735"] * 3


# --- command lines -----------------------------------------------------------------------------

def test_videotoolbox_text_overlay_matches_jellyfin():
    plan = build(platforms.VIDEOTOOLBOX, text_prep())
    assert arg(plan, "-filter_complex") == (
        "alphasrc=s=1920x1080:r=23.976:start=0,format=bgra,"
        "subtitles=f='/run/burnin/subs.ass':alpha=1:sub2video=1:fontsdir='/run/burnin/fonts',hwupload[sub];"
        "[0:v:0]scale_vt=w=1920:h=1080[main];[main][sub]overlay_videotoolbox=eof_action=pass:repeatlast=0[vout]")
    assert arg(plan, "-map") == "[vout]" and "-vf" not in plan.argv
    assert arg(plan, "-init_hw_device") == "videotoolbox=vt"
    assert plan.burn_in == "GPU (overlay_videotoolbox)"


def test_videotoolbox_picture_overlay_with_canvas_size():
    plan = build(platforms.VIDEOTOOLBOX, image_prep())
    assert arg(plan, "-filter_complex") == (
        "[0:3]scale,scale=1920:1080:fast_bilinear,format=bgra,hwupload[sub];"
        "[0:v:0]scale_vt=w=1920:h=1080[main];[main][sub]overlay_videotoolbox=eof_action=pass:repeatlast=0[vout]")
    assert plan.argv.index("-canvas_size") < plan.argv.index("-i") and arg(plan, "-canvas_size") == "1920x1080"


def test_plain_text_subtitles_draw_at_10_fps_on_the_gpu():
    srt = text_prep(burn=BurnIn("text", "file", "subrip", "file x.srt"), text_file="/s/x.srt", fonts_dir="")
    assert arg(build(platforms.VIDEOTOOLBOX, srt), "-filter_complex").startswith(
        "alphasrc=s=1920x1080:r=10:start=0,format=bgra,subtitles=f='/s/x.srt':alpha=1:sub2video=1,hwupload[sub]")


def test_software_burns_text_into_the_chain_and_overlays_pictures():
    text = build(platforms.SOFTWARE, text_prep())
    assert arg(text, "-vf") == ("scale=w=1920:h=1080,format=yuv420p,"
                                "subtitles=f='/run/burnin/subs.ass':fontsdir='/run/burnin/fonts'")
    assert text.burn_in == "CPU, libass" and "-filter_complex" not in text.argv
    pic = build(platforms.SOFTWARE, image_prep())
    assert arg(pic, "-filter_complex") == ("[0:3]scale,scale=1920:1080:fast_bilinear[sub];"
                                           "[0:v:0]scale=w=1920:h=1080,format=yuv420p[main];"
                                           "[main][sub]overlay=eof_action=pass:repeatlast=0[vout]")
    assert pic.burn_in == "CPU, overlay"


def test_nvidia_text_overlay_uses_cuda_upload_and_premultiplied_alpha():
    plan = build(platforms.NVENC, text_prep(), caps=LINUX)
    graph = arg(plan, "-filter_complex")
    assert ",format=yuva420p,subtitles=" in graph and graph.split("[sub]")[0].endswith(",hwupload=derive_device=cuda")
    assert graph.endswith("overlay_cuda=eof_action=pass:repeatlast=0:alpha_format=premultiplied[vout]")
    assert plan.argv[plan.argv.index("-init_hw_device"):][:4] == ["-init_hw_device", "cuda=cu:0", "-filter_hw_device", "cu"]
    stock = build(platforms.NVENC, text_prep(), caps=replace(LINUX, jellyfin_build=False))
    assert "alpha_format" not in arg(stock, "-filter_complex")


def test_qsv_draws_subtitles_at_most_1080_high_and_scales_them_in_the_overlay():
    uhd = replace(SDR_1080, width=3840, height=2160)
    plan = build(platforms.QSV, image_prep(canvas=(3840, 2160)), caps=LINUX, scenario=ALL["mix_4k_hevc_20"],
                 source=uhd)
    graph = arg(plan, "-filter_complex")
    assert graph.startswith("[0:3]scale,scale=1920:1080:fast_bilinear,format=bgra,"
                            "hwupload=derive_device=qsv:extra_hw_frames=64[sub]")
    assert graph.endswith("[main][sub]overlay_qsv=eof_action=pass:repeatlast=0:w=3840:h=2160[vout]")


def test_vaapi_overlay():
    graph = arg(build(platforms.VAAPI, text_prep(), caps=LINUX), "-filter_complex")
    assert "hwupload=derive_device=vaapi[sub]" in graph and "overlay_vaapi=eof_action=pass:repeatlast=0:w=1920:h=1080" in graph


def test_without_a_gpu_overlay_frames_come_back_to_the_cpu():
    plan = build(platforms.VIDEOTOOLBOX, text_prep(), caps=MAC_NO_OVERLAY)
    assert arg(plan, "-vf") == ("scale_vt=w=1920:h=1080,hwdownload,format=nv12,"
                                "subtitles=f='/run/burnin/subs.ass':fontsdir='/run/burnin/fonts'")
    assert plan.burn_in == "CPU (frames copied from the GPU and back), libass"
    assert "-init_hw_device" not in plan.argv


def test_qsv_cpu_overlay_uploads_after_the_overlay():
    caps = replace(LINUX, filters=LINUX.filters - {"overlay_qsv"})
    graph = arg(build(platforms.QSV, image_prep(), caps=caps), "-filter_complex")
    assert graph.endswith("[main][sub]overlay=eof_action=pass:repeatlast=0,format=nv12,hwupload=extra_hw_frames=64[vout]")
    assert "scale_qsv=w=1920:h=1080:format=nv12,hwdownload,format=nv12[main]" in graph


def test_hardware_decode_with_software_encode_burns_in_on_the_cpu():
    plan = build(platforms.VIDEOTOOLBOX, text_prep(), hw_encode=False)
    assert arg(plan, "-vf").startswith("hwdownload,format=nv12,scale=w=1920:h=1080,format=yuv420p,subtitles=")
    assert plan.burn_in == "CPU, libass"


def test_software_tone_mapping_on_videotoolbox_burns_in_after_it():
    hdr_caps = replace(MAC, filters=MAC.filters - {"tonemap_videotoolbox"})
    plan = build(platforms.VIDEOTOOLBOX, text_prep(), caps=hdr_caps, scenario=HDR, source=make_source(subtitles=[ASS]))
    vf = arg(plan, "-vf")
    assert vf.index("tonemap=hable") < vf.index("subtitles=") and "overlay" not in vf
    assert plan.tonemap == "software" and plan.burn_in.startswith("CPU (frames copied")


def test_picture_subtitles_with_another_shape_are_fitted_by_height():
    scope = replace(SDR_1080, width=1920, height=800)
    plan = build(platforms.SOFTWARE, image_prep(), source=scope)
    assert arg(plan, "-filter_complex").startswith(
        "[0:3]scale,scale=-1:800:fast_bilinear,crop,pad=max(1920\\,iw):max(800\\,ih):(ow-iw)/2:(oh-ih)/2:black@0,"
        "crop=1920:800[sub]")


def test_external_picture_file_is_a_second_looped_input():
    prep = image_prep(burn=BurnIn("image", "file", PGS.codec, "file x.sup", path="/s/x.sup"), image_input="/s/x.sup",
                      stream_spec="1:0")
    plan = build(platforms.VIDEOTOOLBOX, prep, loop_input="/work/loop.ffconcat")
    inputs = [i for i, a in enumerate(plan.argv) if a == "-i"]
    assert [plan.argv[i + 1] for i in inputs] == ["/work/loop.ffconcat", "/s/x.sup"]
    assert plan.argv[inputs[1] - 4:inputs[1]] == ["-stream_loop", "-1", "-canvas_size", "1920x1080"]
    assert arg(plan, "-filter_complex").startswith("[1:0]scale,")


def test_looping_uses_the_concat_list_when_there_is_one():
    plan = build(platforms.VIDEOTOOLBOX, None, loop_input="/work/loop.ffconcat")
    i = plan.argv.index("-i")
    assert plan.argv[i - 4:i + 2] == ["-f", "concat", "-safe", "0", "-i", "/work/loop.ffconcat"]
    assert "-stream_loop" not in plan.argv
    assert "-stream_loop" in build(platforms.VIDEOTOOLBOX, None).argv


def test_streams_restart_when_they_reach_the_end_of_the_loop_list(tmp_path):
    from jf_hw_stress.stream import Stream, StreamInfo
    info = StreamInfo(1, "x", "x", "sw", "Software", "libx264", False, False, "", "640x360 h264 2M")
    script = tmp_path / "fake-ffmpeg.sh"
    script.write_text("#!/bin/sh\nprintf 'frame=50\\nprogress=continue\\nframe=100\\nprogress=end\\n'\n")
    script.chmod(0o755)
    stream = Stream(info, [str(script)], str(tmp_path / "s.log"), 24.0)
    stream.restart_on_end = True
    stream.start()
    import time
    deadline = time.monotonic() + 10
    while stream.restarts < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
        stream.poll()
    stream.stop()
    assert stream.restarts >= 2 and stream.frames >= 100 * stream.restarts and stream.status == "stopped"
    assert "restart 1" in (tmp_path / "s.log").read_text()


def test_escaping_paths_for_the_filter_graph():
    assert command.escape_filter_path("C:\\Subs\\it's.ass") == "C\\:/Subs/it'\\\\\\''s.ass"


def test_text_burn_in_needs_libass():
    caps = replace(MAC, filters=MAC.filters - {"subtitles"})
    with pytest.raises(CommandError, match="no subtitles filter"):
        build(platforms.SOFTWARE, text_prep(), caps=caps)


# --- engine ------------------------------------------------------------------------------------

def test_engine_prepares_the_sample_once_and_cleans_up(tmp_path):
    tools = Tools("/bin/ffmpeg", "/bin/ffprobe", "test", "8.1-Jellyfin", MAC)
    cfg = RunConfig(mode="fixed", scenario=COMPAT, platforms=[platforms.VIDEOTOOLBOX], streams=1, duration=60,
                    burn_in=subtitles.resolve("sample", SDR_1080))
    engine = Engine(cfg, tools, SDR_1080, str(tmp_path), stream_factory=lambda *a, **k: None)
    engine.write_loop_list()
    engine.prepare_burn_in()
    assert engine.burn.text_file.endswith("burnin/sample.ass") and engine.burn.events == 16200   # 6 h of source
    assert (tmp_path / "loop.ffconcat").is_file() and engine.loop_input.endswith("loop.ffconcat")
    assert "burning in subtitles: built-in sample" in engine.events[-1].message
    plan = engine.plan(COMPAT, platforms.VIDEOTOOLBOX, software=False, stream_id=1)
    assert "subtitles=f=" in arg(plan, "-filter_complex") and "-stream_loop" not in plan.argv
    (tmp_path / "keep.txt").write_text("not ours")
    remove_work_dir(str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]
