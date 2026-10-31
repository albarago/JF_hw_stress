from jf_hw_stress import __version__
from jf_hw_stress.cli import main


def test_list_scenarios(capsys):
    assert main(["--list-scenarios"]) == 0
    out = capsys.readouterr().out
    assert "4k_hdr_sdr" in out and "mix_480_h264_1" in out


def test_version(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert __version__ in capsys.readouterr().out


def test_invalid_values_are_setup_errors(capsys):
    assert main(["--headless", "--source", "x.mkv", "--fps-ratio", "1.5"]) == 2
    assert "--fps-ratio" in capsys.readouterr().err


def test_headless_needs_an_existing_source(capsys, tmp_path):
    assert main(["--headless"]) == 2
    assert main(["--headless", "--source", str(tmp_path / "missing.mkv")]) == 2
    assert "source not found" in capsys.readouterr().err
