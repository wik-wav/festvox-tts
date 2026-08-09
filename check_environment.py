"""Read-only FestVox installation and external-provider check."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "src" / "festvox_tts"
REQUIRED_FILES = (
    "requirements.txt",
    "festvox.example.json",
    "build_festival_voice.py",
    "run_gui.py",
    "src/festvox_tts/cache_support.py",
    "src/festvox_tts/synth_diphone.py",
    "src/festvox_tts/build_festival_voice.py",
    "src/festvox_tts/utau2festvox.py",
    "src/festvox_tts/arpasing_profile.py",
    "src/festvox_tts/speaker_pitch.py",
    "src/festvox_tts/source_timing.py",
    "src/festvox_tts/voice_manifest.py",
    "src/festvox_tts/voice_paths.py",
    "src/festvox_tts/pitch_domain.py",
    "src/festvox_tts/japanese_models.py",
    "src/festvox_tts/japanese_frontend.py",
    "src/festvox_tts/japanese_kana_frontend.py",
    "src/festvox_tts/japanese_openjtalk.py",
    "src/festvox_tts/japanese_profiles.py",
    "src/festvox_tts/japanese_candidates.py",
    "src/festvox_tts/japanese_utau.py",
    "src/festvox_tts/japanese_festival.py",
    "src/festvox_tts/japanese_duration.py",
    "src/festvox_tts/japanese_pitch.py",
    "src/festvox_tts/japanese_assembly.py",
    "src/festvox_tts/japanese_editing.py",
    "src/festvox_tts/japanese_devoicing.py",
    "src/festvox_tts/japanese_synthesis.py",
    "src/festvox_tts/source_filter_voicing.py",
    "src/festvox_tts/diphone_loudness.py",
    "src/festvox_tts/join_discontinuity.py",
    "src/festvox_tts/join_spectrogram.py",
    "src/festvox_tts/formant_analysis.py",
    "src/festvox_tts/formant_plots.py",
    "src/festvox_tts/rendered_formant_diagnostic.py",
    "src/festvox_tts/vocal_tract.py",
    "src/festvox_tts/vocal_tract_validation.py",
    "src/festvox_tts/festvox_gui/festvox_gui.py",
    "src/festvox_tts/festvox_gui/festvox_core.py",
    "src/festvox_tts/festvox_gui/requirements.txt",
    "src/festvox_tts/profiles/en-jap-mapping.yaml",
    "src/festvox_tts/profiles/japanese_duration_priors_v1.json",
    "src/festvox_tts/profiles/japanese_pitch_model_v1.json",
    "src/festvox_tts/profiles/reference_voice_space_v1.json",
)
REQUIRED_MODULES = ("numpy", "PyQt5", "pyqtgraph", "cmudict")
OPTIONAL_MODULES = (
    "sounddevice", "librosa", "scipy", "pyopenjtalk", "pyworld", "torch",
)


def decode_process_output(data):
    """Decode command output, including WSL's UTF-16 Windows diagnostics."""
    if not data:
        return ""
    if isinstance(data, str):
        return data.strip()
    encoding = "utf-16-le" if b"\x00" in data[:64] else "utf-8"
    return data.decode(encoding, errors="replace").lstrip("\ufeff").strip()


def module_status(name):
    try:
        available = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        available = False
    return {"name": name, "available": available}


def executable_status(name):
    path = shutil.which(name)
    return {"name": name, "available": bool(path), "path": path or ""}


def wsl_festival_status():
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        return {"available": False, "detail": "WSL executable unavailable"}
    try:
        process = subprocess.run(
            [wsl, "sh", "-lc", "command -v festival"],
            capture_output=True, timeout=8,
            creationflags=(subprocess.CREATE_NO_WINDOW
                           if sys.platform == "win32" else 0),
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "detail": str(error)}
    detail = decode_process_output(process.stdout or process.stderr)
    return {"available": process.returncode == 0, "detail": detail}


def inspect_environment(check_wsl=True):
    files = [
        {"path": relative, "available": (ROOT / relative).is_file()}
        for relative in REQUIRED_FILES
    ]
    required = [module_status(name) for name in REQUIRED_MODULES]
    optional = [module_status(name) for name in OPTIONAL_MODULES]
    commands = [executable_status(name) for name in
                ("ffmpeg", "nvidia-smi", "wsl.exe")]
    config = ROOT / "festvox.json"
    voices = ROOT / "generated_voices"
    report = {
        "schema_version": 1,
        "python": sys.version,
        "project_root": str(ROOT),
        "required_files": files,
        "required_python_modules": required,
        "optional_python_modules": optional,
        "optional_commands": commands,
        "festival_wsl": (wsl_festival_status() if check_wsl else
                         {"available": None, "detail": "not checked"}),
        "local_config": {
            "available": config.is_file(),
            "template": "festvox.example.json",
        },
        "generated_voice_count": sum(
            1 for path in voices.glob("*/dic/diphone_index.json"))
            if voices.is_dir() else 0,
    }
    missing_files = [row["path"] for row in files if not row["available"]]
    missing_modules = [row["name"] for row in required
                       if not row["available"]]
    report["ready"] = not missing_files and not missing_modules
    report["missing_required_files"] = missing_files
    report["missing_required_modules"] = missing_modules
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="emit the complete machine-readable report")
    parser.add_argument("--skip-wsl", action="store_true",
                        help="do not query Festival inside WSL")
    args = parser.parse_args(argv)
    report = inspect_environment(check_wsl=not args.skip_wsl)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("FestVox environment: %s" %
              ("READY" if report["ready"] else "INCOMPLETE"))
        if report["missing_required_files"]:
            print("Missing files: " +
                  ", ".join(report["missing_required_files"]))
        if report["missing_required_modules"]:
            print("Missing Python modules: " +
                  ", ".join(report["missing_required_modules"]))
        print("Generated voices: %d" % report["generated_voice_count"])
        print("Festival/WSL: %s" %
              ("available" if report["festival_wsl"]["available"] else
               report["festival_wsl"]["detail"]))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
