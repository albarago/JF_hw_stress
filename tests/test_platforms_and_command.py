import pytest

from conftest import JELLYFIN_MAC, LINUX_ALL, STOCK_MAC, make_source
from jf_hw_stress import command, platforms
from jf_hw_stress.command import CommandError, Options
from jf_hw_stress.scenarios import ALL, SCENARIOS, applicable, mixed_pool, output_height

HDR = ALL["4k_hdr_sdr"]
COMPAT = ALL["1080p_compat"]


@pytest.fixture(autouse=True)
def no_device_lookup(monkeypatch):
    monkeypatch.setattr(platforms, "device_name", lambda p: "Test GPU M5")


def build(platform, scenario, caps, **kw):
    return command.build(platform, scenario, kw.pop("source", make_source()), caps, "/bin/ffmpeg", 1, "/work",
                         kw.pop("options", Options()), **kw)


def test_detect_prefers_hardware_and_adds_apple_av1():
    found = platforms.detect(JELLYFIN_MAC, system="Darwin")
    assert [p.id for p in found] == ["vt"]
    assert "av1" in found[0].decode_codecs and found[0].device == "Test GPU M5"


def test_detect_falls_back_to_software():
    caps = platforms.Capabilities(encoders={"libx264", "libx265"}, filters={"scale"})
    assert [p.id for p in platforms.detect(caps, system="Linux")] == ["sw"]


def test_choose_refuses_instead_of_silently_falling_back():
    with pytest.raises(platforms.PlatformError, match="not available on Darwin"):
        platforms.choose(["nvenc"], JELLYFIN_MAC, system="Darwin")
    with pytest.raises(platforms.PlatformError, match="missing"):
        platforms.choose(["amf"], LINUX_ALL, system="Windows")
    with pytest.raises(platforms.PlatformError, match="unknown"):
        platforms.choose(["gpu9000"], LINUX_ALL, system="Linux")
    assert [p.id for p in platforms.choose(["nvenc", "qsv"], LINUX_ALL, system="Linux")] == ["nvenc", "qsv"]


@pytest.mark.parametrize("chip,expected", [
    ("Apple M1 Pro", False), ("Apple M3", True), ("Apple M5", True), ("A17 Pro", True), ("Intel UHD", False),
])
def test_apple_av1(chip, expected):
    assert platforms.apple_av1_decode(chip) is expected


def test_videotoolbox_hdr_stays_on_the_gpu_with_jellyfin_ffmpeg():
    plan = build(platforms.VIDEOTOOLBOX, HDR, JELLYFIN_MAC)
    vf = plan.argv[plan.argv.index("-vf") + 1]
    assert vf == "scale_vt=w=1920:h=1080,tonemap_videotoolbox=p=bt709:t=bt709:m=bt709:tonemap=bt2390"
    assert (plan.encoder, plan.encode_hw, plan.decode_hw, plan.tonemap) == ("hevc_videotoolbox", True, True, "hardware")
    assert plan.argv[plan.argv.index("-hwaccel_output_format") + 1] == "videotoolbox_vld"


def test_stock_ffmpeg_without_tone_mapping_is_refused_clearly():
    with pytest.raises(CommandError, match="cannot tone-map"):
        build(platforms.VIDEOTOOLBOX, HDR, STOCK_MAC)
    assert build(platforms.VIDEOTOOLBOX, COMPAT, STOCK_MAC).encoder == "h264_videotoolbox"


def test_every_command_is_safe_for_a_terminal_and_loops():
    argv = build(platforms.VIDEOTOOLBOX, COMPAT, JELLYFIN_MAC).argv
    assert "-nostdin" in argv and argv[argv.index("-progress") + 1] == "pipe:1"
    assert argv[argv.index("-stream_loop") + 1] == "-1" and argv.index("-stream_loop") < argv.index("-i")
    assert "-re" not in argv
    assert "-re" in build(platforms.VIDEOTOOLBOX, COMPAT, JELLYFIN_MAC, options=Options(realtime=True, loop=False)).argv


def test_keyframes_follow_the_segment_length():
    argv = build(platforms.VIDEOTOOLBOX, COMPAT, JELLYFIN_MAC, options=Options(hls_segment_secs=6)).argv
    assert argv[argv.index("-g") + 1] == "144" and "expr:gte(t,n_forced*6)" in argv


def test_audio_modes():
    def audio(mode, **src):
        argv = build(platforms.VIDEOTOOLBOX, COMPAT, JELLYFIN_MAC, options=Options(audio=mode), source=make_source(**src)).argv
        return argv[argv.index("-force_key_frames") + 2:argv.index("-max_muxing_queue_size")], "0:a:0" in argv
    assert audio("aac") == (["-c:a", "aac", "-b:a", "192k", "-ac", "2"], True)
    assert audio("copy") == (["-c:a", "copy"], True)
    assert audio("none") == (["-an"], False)
    assert audio("aac", audio_codec="") == (["-an"], False)


def test_hevc_tag_only_on_hevc():
    assert "hvc1" in build(platforms.VIDEOTOOLBOX, HDR, JELLYFIN_MAC).argv
    assert "hvc1" not in build(platforms.VIDEOTOOLBOX, COMPAT, JELLYFIN_MAC).argv


def test_software_decode_with_hardware_encode():
    plan = build(platforms.VIDEOTOOLBOX, COMPAT, JELLYFIN_MAC, hw_decode=False)
    assert "-hwaccel" not in plan.argv and plan.argv[plan.argv.index("-vf") + 1] == "scale=w=1920:h=1080,format=yuv420p"
    assert (plan.decode_hw, plan.encode_hw) == (False, True)


def test_software_platform_and_hybrid_pivot():
    plan = build(platforms.VIDEOTOOLBOX, COMPAT, JELLYFIN_MAC, hw_decode=False, hw_encode=False)
    assert plan.encoder == "libx264" and "-hwaccel" not in plan.argv
    sw = build(platforms.SOFTWARE, HDR, JELLYFIN_MAC)
    assert sw.encoder == "libx265" and sw.tonemap == "software" and "zscale" in sw.argv[sw.argv.index("-vf") + 1]


def test_qsv_uses_its_decoder_and_device():
    plan = build(platforms.QSV, COMPAT, LINUX_ALL)
    argv = plan.argv
    assert argv[argv.index("-init_hw_device") + 1] == "qsv=hw" and argv[argv.index("-c:v") + 1] == "hevc_qsv"
    assert argv[argv.index("-vf") + 1] == "scale_qsv=w=1920:h=1080:format=nv12"
    sw_dec = build(platforms.QSV, COMPAT, LINUX_ALL, hw_decode=False)
    assert sw_dec.argv[sw_dec.argv.index("-vf") + 1].endswith("format=nv12,hwupload=extra_hw_frames=64")


def test_vaapi_hardware_tonemap_and_nvenc_software_tonemap():
    vaapi = build(platforms.VAAPI, HDR, LINUX_ALL)
    assert vaapi.tonemap == "hardware" and "tonemap_vaapi" in vaapi.argv[vaapi.argv.index("-vf") + 1]
    nvenc = build(platforms.NVENC, HDR, LINUX_ALL)
    vf = nvenc.argv[nvenc.argv.index("-vf") + 1]
    assert nvenc.tonemap == "software" and vf.startswith("scale_cuda=w=1920:h=1080:format=yuv420p,hwdownload")


def test_amf_frames_come_back_to_system_memory():
    caps = platforms.Capabilities(hwaccels={"d3d11va"}, encoders={"h264_amf", "hevc_amf", "libx264", "libx265"},
                                  filters={"scale", "format"})
    plan = build(platforms.AMF, COMPAT, caps)
    assert "-hwaccel_output_format" not in plan.argv and plan.argv[plan.argv.index("-vf") + 1].startswith("scale=")


def test_missing_encoder_is_refused():
    caps = platforms.Capabilities(hwaccels={"videotoolbox"}, encoders={"h264_videotoolbox"}, filters={"scale_vt", "scale"})
    with pytest.raises(CommandError, match="hevc_videotoolbox"):
        build(platforms.VIDEOTOOLBOX, ALL["4k_1080p"], caps)


def test_scenarios_do_not_upscale_or_tonemap_sdr():
    hd_sdr = make_source(width=1920, height=1080, hdr_type="SDR", color_transfer="bt709")
    assert applicable(ALL["mix_4k_hevc_20"], hd_sdr)[0] is False
    assert applicable(HDR, hd_sdr) == (False, "tone mapping needs an HDR source")
    assert all(s.width <= 1920 and not s.tonemap for s in mixed_pool(hd_sdr))
    assert len(mixed_pool(make_source())) == 14
    assert output_height(ALL["1080p_compat"], make_source(width=3840, height=1600)) == 800
    assert all(output_height(s, make_source(width=1998, height=1080)) % 2 == 0 for s in SCENARIOS)
