"""Build and describe Festival UniSyn grouped runtime audio.

Festival's development-oriented ``grouped=false`` database opens separate
signal and pitchmark files while rendering.  A grouped database contains the
same indexed units in one deterministic file, allowing UniSyn to seek directly
to the requested unit data.  The original WAV and pitchmark files remain in
the generated voice for diagnostics and as an explicit fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable, Mapping, Optional

from voice_paths import windows_to_wsl_path


RUNTIME_AUDIO_STORAGE_MODES = ("grouped", "separate")
RUNTIME_AUDIO_STORAGE_SCHEMA_VERSION = 1
_SAFE_SYMBOL = re.compile(r"^[A-Za-z0-9_]+$")


def _scheme_string(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _runtime_path(path: Path) -> str:
    return windows_to_wsl_path(path) if os.name == "nt" else str(path)


def grouped_audio_relative_path(voice_name: str) -> str:
    name = str(voice_name)
    if not _SAFE_SYMBOL.fullmatch(name):
        raise ValueError(f"unsafe Festival voice name: {voice_name!r}")
    return f"group/{name}_diphone.group"


def separate_runtime_metadata(*, requested: str = "separate") -> dict:
    if requested not in RUNTIME_AUDIO_STORAGE_MODES:
        raise ValueError(f"unsupported runtime audio storage: {requested}")
    return {
        "schema_version": RUNTIME_AUDIO_STORAGE_SCHEMA_VERSION,
        "requested": requested,
        "effective": "separate",
        "access_model": "individual-wav-and-pitchmark-files",
        "source_wav_files_retained": True,
        "pitchmark_files_retained": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest().upper()


def build_grouped_runtime(
    voice_root: Path | str,
    *,
    voice_name: str,
    scheme_path: Path | str,
    voice_entry_point: str,
    festival_bin: str,
    run_external: Callable[..., object],
    wsl_distro: Optional[str] = None,
    timeout: float = 900.0,
) -> dict:
    """Pack the selected separate PSOLA database into one group file.

    The build process always forces the generated Scheme to select its
    separate database.  This matters during ``--overwrite`` builds: an older
    group file must never become the source of a newly rebuilt cache.
    Festival writes a fixed staging name and the completed file is atomically
    installed, so a failed rebuild cannot replace a working runtime cache.
    """
    root = Path(voice_root).expanduser().resolve()
    scheme = Path(scheme_path).expanduser().resolve()
    entry_point = str(voice_entry_point)
    if not root.is_dir():
        raise FileNotFoundError(f"generated voice root not found: {root}")
    try:
        scheme.relative_to(root)
    except ValueError as error:
        raise ValueError("voice Scheme must be inside the generated voice") from error
    if not scheme.is_file():
        raise FileNotFoundError(f"voice Scheme not found: {scheme}")
    if not _SAFE_SYMBOL.fullmatch(entry_point):
        raise ValueError(f"unsafe Festival voice entry point: {entry_point!r}")

    relative = grouped_audio_relative_path(voice_name)
    target = root / Path(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.building")
    script = root / ".build_grouped_runtime.scm"
    staging.unlink(missing_ok=True)

    runtime_root = _runtime_path(root).rstrip("/")
    runtime_scheme = _runtime_path(scheme)
    runtime_staging = _runtime_path(staging)
    source = (
        ";; Temporary builder script. Written only inside generated output.\n"
        "(define festvox_gui_force_separate_database t)\n"
        f"(set! load-path (cons {_scheme_string(runtime_root + '/festvox')} "
        "load-path))\n"
        f"(set! load-path (cons {_scheme_string(runtime_root)} load-path))\n"
        f"(load {_scheme_string(runtime_scheme)})\n"
        f"({entry_point})\n"
        f"(us_make_group_file {_scheme_string(runtime_staging)}\n"
        " '((sig_file_format riff) (sig_sample_format short)))\n"
        "(print 'FESTVOX-GROUP-RUNTIME-OK)\n"
    )
    script.write_text(source, encoding="utf-8", newline="\n")
    try:
        result = run_external(
            [festival_bin, "-b", str(script)],
            wsl_distro=wsl_distro,
            timeout=timeout,
        )
        stdout = str(getattr(result, "stdout", "") or "")
        stderr = str(getattr(result, "stderr", "") or "")
        returncode = int(getattr(result, "returncode", 1))
        if (
            returncode != 0
            or "FESTVOX-GROUP-RUNTIME-OK" not in stdout
            or not staging.is_file()
            or staging.stat().st_size <= 64
        ):
            detail = (stderr or stdout or "Festival produced no diagnostic").strip()
            raise RuntimeError(
                "Festival could not build the grouped UniSyn runtime "
                f"(exit {returncode}):\n{detail[-3000:]}"
            )
        staging.replace(target)
    finally:
        script.unlink(missing_ok=True)
        staging.unlink(missing_ok=True)

    return {
        "schema_version": RUNTIME_AUDIO_STORAGE_SCHEMA_VERSION,
        "requested": "grouped",
        "effective": "grouped",
        "access_model": "unisyn-indexed-group-file",
        "group_file": relative,
        "group_file_bytes": target.stat().st_size,
        "group_file_sha256": _sha256(target),
        "signal_file_format": "riff",
        "signal_sample_format": "short",
        "source_wav_files_retained": True,
        "pitchmark_files_retained": True,
        "contextual_unit_selection_preserved": True,
    }


def apply_runtime_audio_metadata(
    voice_root: Path | str,
    runtime_audio_storage: Mapping[str, object],
) -> None:
    """Record runtime storage in every existing generated metadata view."""
    root = Path(voice_root).expanduser().resolve()
    storage = dict(runtime_audio_storage)
    paths = (
        root / "dic" / "diphone_index.json",
        root / "dic" / "unit_alternatives.json",
        root / "dic" / "voice_manifest.json",
        root / "dic" / "japanese_build_report.json",
    )
    group_file = str(storage.get("group_file") or "")
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        payload["runtime_audio_storage"] = storage
        if group_file and isinstance(payload.get("output_relative_files"), list):
            files = {str(item) for item in payload["output_relative_files"]}
            files.add(group_file.replace("\\", "/"))
            payload["output_relative_files"] = sorted(files)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
