"""Build the project-local Festival/UniSyn crossover runtime in WSL."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "festvox_festival.cc"
OUTPUT = ROOT / "build" / "festvox-festival"


def windows_to_wsl(path: Path) -> str:
    value = str(path.resolve())
    drive, rest = value[0], value[2:]
    return f"/mnt/{drive.lower()}/" + rest.replace("\\", "/").lstrip("/")


def build(distro: str = "Ubuntu") -> Path:
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        raise RuntimeError("wsl.exe was not found")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    source = windows_to_wsl(SOURCE)
    output = windows_to_wsl(OUTPUT)
    command = [
        wsl, "-d", distro, "--", "g++",
        "-std=c++11", "-O2", "-Wall", "-Wextra",
        "-I/usr/include/festival",
        "-I/usr/include/speech_tools",
        source,
        "/usr/lib/libFestival.a",
        "-lestools", "-lestbase", "-leststring",
        "-lsystemd", "-ltinfo", "-ldl", "-lpthread", "-lm",
        "-o", output,
    ]
    process = subprocess.run(
        command, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180)
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or
                  "compiler produced no diagnostic").strip()
        raise RuntimeError(
            "Could not build the native UniSyn runtime.\n"
            "Install its WSL build dependencies with:\n"
            "  sudo apt install g++ festival-dev libestools-dev "
            "libsystemd-dev libncurses-dev\n\n"
            + detail[-4000:])
    if not OUTPUT.is_file() or OUTPUT.stat().st_size < 1024:
        raise RuntimeError("compiler reported success but produced no runtime")
    return OUTPUT


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distro", default="Ubuntu")
    args = parser.parse_args(argv)
    print(build(args.distro))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
