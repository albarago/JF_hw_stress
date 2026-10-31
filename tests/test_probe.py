import pytest

from jf_hw_stress.probe import ProbeError, classify_hdr, dovi_notes, parse


def video(**over):
    base = {"codec_type": "video", "codec_name": "hevc", "profile": "Main 10", "width": 3840, "height": 2160,
            "pix_fmt": "yuv420p10le", "avg_frame_rate": "24000/1001", "r_frame_rate": "24000/1001",
            "color_transfer": "smpte2084", "color_primaries": "bt2020"}
    base.update(over)
    return base


AUDIO = {"codec_type": "audio", "codec_name": "dts", "profile": "DTS-HD MA", "channels": 6, "sample_rate": "48000",
         "tags": {"language": "eng"}}


def test_hdr10_source():
    info = parse({"streams": [video(), AUDIO], "format": {"bit_rate": "43797424", "duration": "300.716"}}, "/f.mkv")
    assert (info.hdr_type, info.is_hdr, info.bit_depth) == ("HDR10", True, 10)
    assert info.fps == pytest.approx(23.976, abs=0.001)
    assert info.bitrate_mbps == 43.8
    assert info.audio_label() == "DTS (DTS-HD MA) 5.1"


def test_hlg_is_not_reported_as_hdr10():
    # HLG always uses BT.2020 primaries; v1 checked primaries first and called it HDR10.
    assert classify_hdr(video(color_transfer="arib-std-b67")) == ("HLG", None)


def test_bt2020_primaries_alone_are_sdr():
    assert classify_hdr(video(color_transfer="bt709")) == ("SDR", None)


@pytest.mark.parametrize("profile", [5, 7, 8])
def test_dolby_vision_profiles(profile):
    stream = video(side_data_list=[{"side_data_type": "DOVI configuration record", "dv_profile": profile}])
    assert classify_hdr(stream) == ("Dolby Vision", profile)


def test_dolby_vision_from_codec_tag_has_unknown_profile():
    assert classify_hdr(video(codec_tag_string="dvh1")) == ("Dolby Vision", 0)


def test_dovi_notes_describe_layers_correctly():
    assert "single layer with no HDR10 fallback" in dovi_notes(5)[0]
    assert "enhancement layer (MEL or FEL)" in dovi_notes(7)[0]
    assert dovi_notes(None) == []


def test_cover_art_is_not_the_video_stream():
    art = {"codec_type": "video", "codec_name": "mjpeg", "width": 600, "height": 900, "disposition": {"attached_pic": 1}}
    info = parse({"streams": [art, video(codec_name="h264", pix_fmt="yuv420p", color_transfer="bt709")], "format": {}}, "/f")
    assert (info.codec, info.width, info.bit_depth, info.hdr_type) == ("h264", 3840, 8, "SDR")


def test_no_video_is_an_error():
    with pytest.raises(ProbeError):
        parse({"streams": [AUDIO], "format": {}}, "/f")


def test_missing_audio_and_odd_values():
    info = parse({"streams": [video(avg_frame_rate="0/0", r_frame_rate="25/1", width="x")], "format": {"bit_rate": "N/A"}}, "/f")
    assert (info.fps, info.width, info.bitrate_mbps, info.has_audio, info.audio_label()) == (25.0, 0, 0.0, False, "none")
