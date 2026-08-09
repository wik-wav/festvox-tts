"""Shared Windows/WSL paths and generated-voice registration records.

User-facing builder and GUI paths are canonical Windows paths.  Festival's
WSL path is derived at the boundary and retained only as inspectable runtime
metadata.  Existing WSL-only registrations remain readable as legacy entries;
they are never moved or deleted during migration.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
from typing import Mapping, Optional


VOICE_REGISTRATION_SCHEMA_VERSION = 1
UNIFIED_BUILDER_VERSION = "unified-festival-builder-v1"


class VoicePathError(ValueError):
    """A source/output path would make a build ambiguous or unsafe."""


def windows_to_wsl_path(value: object) -> str:
    """Translate a drive-letter Windows path; preserve POSIX paths."""
    text = str(value or "").strip()
    if not text or text.startswith("/"):
        return text.replace("\\", "/")
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if match:
        tail = match.group(2).replace("\\", "/")
        return f"/mnt/{match.group(1).lower()}/{tail}"
    return text.replace("\\", "/")


def wsl_to_windows_path(value: object) -> Optional[str]:
    """Return a Windows drive path for a WSL /mnt path, else ``None``."""
    text = str(value or "").strip().replace("\\", "/")
    match = re.match(r"^/mnt/([A-Za-z])(?:/(.*))?$", text)
    if not match:
        return None
    tail = (match.group(2) or "").replace("/", "\\")
    if not tail:
        return f"{match.group(1).upper()}:\\"
    return f"{match.group(1).upper()}:\\{tail}".rstrip("\\")


def canonical_windows_path(value: object) -> str:
    """Normalize a user-visible local path without requiring it to exist."""
    text = str(value or "").strip()
    mapped = wsl_to_windows_path(text)
    if mapped:
        text = mapped
    if not text or text.startswith("/"):
        raise VoicePathError("A Windows drive path is required.")
    return os.path.normpath(os.path.abspath(os.path.expanduser(text)))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_build_layout(
    samples: object,
    output: object,
    *,
    oto: object = "",
    overwrite: bool = False,
) -> tuple[Path, Path, Optional[Path]]:
    """Validate a non-destructive source/output layout for either builder."""
    source = Path(canonical_windows_path(samples)).resolve()
    destination = Path(canonical_windows_path(output)).resolve()
    oto_path = (
        Path(canonical_windows_path(oto)).resolve() if str(oto or "").strip()
        else None
    )
    if not source.is_dir():
        raise VoicePathError(f"Source sample folder not found: {source}")
    if oto_path is not None:
        if not oto_path.exists():
            raise VoicePathError(f"OTO source not found: {oto_path}")
        if not _is_within(oto_path, source):
            raise VoicePathError(
                "The selected OTO must be inside the selected source sample "
                "folder. This keeps source provenance and WAV resolution "
                "unambiguous."
            )
    if destination == source or _is_within(destination, source):
        raise VoicePathError(
            "Generated output cannot be the source UTAU folder or one of its "
            "children."
        )
    if _is_within(source, destination):
        raise VoicePathError(
            "Generated output cannot contain the source UTAU folder."
        )
    if (destination / "oto.ini").is_file():
        raise VoicePathError(
            f"Refusing a generated output folder that contains oto.ini: "
            f"{destination}"
        )
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise VoicePathError(
            f"Generated output is not empty: {destination}. Pass --overwrite "
            "to update known generated files without deleting unrelated data."
        )
    return source, destination, oto_path


def metadata_registration_fields(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Extract stable compatibility identity from generated metadata."""
    data = dict(metadata or {})
    primary = str(data.get("primary_language") or data.get("language") or "")
    entries = {
        str(key): str(value)
        for key, value in dict(data.get("voice_entry_points") or {}).items()
        if value
    }
    entry_point = entries.get(primary) or str(
        data.get("voice_entry_point") or ""
    )
    return {
        "source_bundle_id": str(data.get("source_bundle_id") or ""),
        "configuration_id": str(data.get("configuration_id") or ""),
        "language": primary,
        "alias_system": str(data.get("alias_system") or ""),
        "entry_point": entry_point,
        "builder_version": str(
            data.get("front_door_builder_version")
            or data.get("builder_version") or data.get("version") or ""
        ),
    }


def make_voice_registration(
    *,
    windows_path: object,
    name: str,
    scm: Optional[str],
    voice: str,
    voice_en: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    """Create the single canonical registration written by the GUI."""
    local = canonical_windows_path(windows_path)
    fields = metadata_registration_fields(metadata or {})
    return {
        "registration_schema_version": VOICE_REGISTRATION_SCHEMA_VERSION,
        "name": str(name),
        "dir": local,
        "windows_path": local,
        "runtime_path": windows_to_wsl_path(local),
        "path_status": "current",
        "voice": fields["entry_point"] or str(voice),
        "voice_en": (str(voice_en) if voice_en else None),
        "scm": (str(scm) if scm else None),
        **fields,
    }


def migrate_voice_registration(
    value: Mapping[str, object],
    *,
    metadata: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    """Migrate an old registration in memory without touching its folder."""
    result = dict(value or {})
    raw = str(
        result.get("windows_path") or result.get("dir") or ""
    ).strip()
    windows_path = None
    if raw and not raw.startswith("/"):
        windows_path = canonical_windows_path(raw)
    elif raw:
        windows_path = wsl_to_windows_path(raw)
        if windows_path:
            windows_path = canonical_windows_path(windows_path)

    if windows_path:
        result["dir"] = windows_path
        result["windows_path"] = windows_path
        result["runtime_path"] = windows_to_wsl_path(windows_path)
        result["path_status"] = "current"
    else:
        result["runtime_path"] = str(
            result.get("runtime_path") or raw
        ).replace("\\", "/")
        result["windows_path"] = ""
        result["path_status"] = "legacy_wsl_only"

    fields = metadata_registration_fields(metadata or {})
    for key, item in fields.items():
        if item:
            result[key] = item
    if fields.get("entry_point"):
        result["voice"] = fields["entry_point"]
    result["registration_schema_version"] = (
        VOICE_REGISTRATION_SCHEMA_VERSION
    )
    return result
