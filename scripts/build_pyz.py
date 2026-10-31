#!/usr/bin/env python3
"""Build dist/jf-hw-stress.pyz: one file with the package and its dependencies, run with any Python 3.9+.

    python3 scripts/build_pyz.py            # needs pip and network access to fetch rich
    python3 dist/jf-hw-stress.pyz --version

Dependencies are resolved for the oldest supported Python (3.9), so packages that only older
versions need are included, and only pure-Python wheels are accepted, so the file works on every OS.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipapp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "jf_hw_stress"
MIN_PYTHON = "3.9"
MAIN = """\
import sys

from jf_hw_stress.cli import main

sys.exit(main())
"""


def requirements() -> list:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.S | re.M)
    if not block:
        raise SystemExit("no dependencies list in pyproject.toml")
    return re.findall(r'"([^"]+)"', block.group(1))


def version() -> str:
    match = re.search(r'__version__\s*=\s*"([^"]+)"', (PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("no __version__ in jf_hw_stress/__init__.py")
    return match.group(1)


def build(out: Path) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "app"
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                        "--target", str(stage), "--python-version", MIN_PYTHON, "--only-binary=:all:",
                        "--implementation", "py", "--abi", "none", "--platform", "any", *requirements()],
                       check=True)
        shutil.rmtree(stage / "bin", ignore_errors=True)
        shutil.copytree(PACKAGE, stage / "jf_hw_stress", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (stage / "__main__.py").write_text(MAIN, encoding="utf-8")
        for cache in list(stage.rglob("__pycache__")):
            shutil.rmtree(cache, ignore_errors=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        zipapp.create_archive(stage, out, interpreter="/usr/bin/env python3", compressed=True)
    out.chmod(0o755)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "jf-hw-stress.pyz", help="output file")
    args = ap.parse_args()
    path = build(args.out)
    print(f"built {path} (jf-hw-stress {version()}, {path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
