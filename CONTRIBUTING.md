# Contributing

Bug reports, test results and pull requests are welcome.

## Test results from untested hardware

NVENC, Quick Sync, VA-API and AMF are built from documentation and haven't been run on real hardware yet. If you
have one of them, a report helps more than anything else:

1. Run `jf-hw-stress --list-platforms` and copy the output.
2. Run a short test with each scenario you can, for example:

   ```sh
   jf-hw-stress --headless --source movie.mkv --scenario 1080p_compat --keep-files --json-file result.json
   jf-hw-stress --headless --source hdr-movie.mkv --scenario 4k_hdr_sdr --keep-files --json-file result-hdr.json
   ```

3. Open an issue with the `--list-platforms` output, the JSON reports, and, if a stream failed, its log
   (`stream<N>.log` in the run folder that `--keep-files` keeps). Remove private paths first.

## Development setup

```sh
git clone https://github.com/albarago/JF_hw_stress.git
cd JF_hw_stress
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Run the checks that CI runs:

```sh
.venv/bin/ruff check src tests scripts
.venv/bin/pytest                    # unit tests, no ffmpeg needed
.venv/bin/pytest -m integration     # real software encodes; needs ffmpeg with libx264 and libx265 on PATH
.venv/bin/python scripts/build_pyz.py && python3 dist/jf-hw-stress.pyz --version
```

Run the tool from the checkout with `.venv/bin/jf-hw-stress` or `.venv/bin/python -m jf_hw_stress`.

## Code layout

Everything is in `src/jf_hw_stress/`:

| Module | Does |
|---|---|
| `cli.py` | Options, validation, headless setup |
| `tui.py` | Interactive setup and live dashboard (uses `rich`) |
| `headless.py` | The run loop without prompts, exit codes |
| `engine.py` | The test: launches streams, asks the controller, records the timeline |
| `controller.py` | When to add a stream and when the limit is reached (pure logic, no processes) |
| `stream.py` | One ffmpeg process: reads `-progress` from a pipe, measures the live frame rate |
| `command.py` | Builds each ffmpeg command line (a pure function of platform, scenario and source) |
| `platforms.py` | The backend table used by detection and `--platform` |
| `scenarios.py` | Output targets |
| `probe.py` | Reads the source with ffprobe, classifies HDR |
| `tools.py` | Finds ffmpeg and ffprobe, reads what the build supports |
| `report.py` | JSON report (schema 2) and the HTML page built from it |

To add or fix a backend, change its entry in `platforms.py` and add a test in
`tests/test_platforms_and_command.py` that checks the command line. Mark it `tested=True` only after a run on real
hardware.

The only runtime dependency is `rich`, and only the interactive mode needs it. Please keep it that way. Code must run
on Python 3.9.

## Pull requests

- Keep changes focused, and add or update tests for behaviour changes.
- If the JSON report changes, update `docs/json-schema.md`. Renaming, removing or changing the meaning of a field
  needs a new `schema_version`.
- Add a line to `CHANGELOG.md` under the next version.

## Releases (maintainers)

1. Update `__version__` in `src/jf_hw_stress/__init__.py` and `version` in `pyproject.toml`, and date the
   `CHANGELOG.md` entry.
2. Tag and push: `git tag v2.0.0 && git push origin v2.0.0`.
3. The release workflow checks that the tag matches the version, builds the wheel, sdist and `jf-hw-stress.pyz`,
   and creates a GitHub release with them attached.

Publishing to PyPI is off until it is set up:

1. On PyPI, add a [trusted publisher](https://docs.pypi.org/trusted-publishers/) for this repository with workflow
   `release.yml` and environment `pypi`.
2. In the GitHub repository settings, create an environment named `pypi`, and set the repository variable
   `PUBLISH_PYPI` to `true`.

After that, every release tag also publishes to PyPI, and `pipx install jf-hw-stress` works.
