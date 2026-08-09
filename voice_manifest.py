"""Stable generated-voice identity and compatibility metadata.

A source recording bundle describes immutable recordings and OTO evidence.
A voice configuration describes one concrete interpretation of that evidence.
Keeping those identities separate prevents a CV build, a VCV build, and a
future language build from accidentally sharing aliases or canonical phones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence


VOICE_MANIFEST_SCHEMA_VERSION = 1
JAPANESE_BUILD_TYPES = ("cv", "vcv", "cvvc")

# One completed-phrase gain is deliberately shared by every language routed
# through a generated Festival voice. The peak ceiling remains the final
# safety bound; the wider positive range is needed for quiet phone inventories
# (notably Japanese subsets) without changing levels between their units.
DEFAULT_GENERATED_VOICE_OUTPUT_CALIBRATION = {
    "schema_version": 1,
    "method": "active_speech_rms",
    "target_dbfs": -20.0,
    "minimum_gain_db": -6.0,
    "maximum_gain_db": 12.0,
    "minimum_active_seconds": 0.08,
    "peak_ceiling": 0.98,
}


def generated_voice_output_calibration(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Return the explicit or legacy-compatible generated-voice policy.

    An explicit metadata field is always authoritative, including an empty
    mapping used to opt out. Older voices from this repository predate the
    field, so their stable generated-voice identity and builder marker opt
    into the current shared default. Built-in and unknown external Festival
    voices have neither marker and remain untouched.
    """
    data = dict(metadata or {})
    if "output_calibration" in data:
        raw = data.get("output_calibration")
        if not isinstance(raw, Mapping):
            return {}
        policy = dict(raw)
        if policy:
            policy.setdefault("policy_source", "voice_metadata")
        return policy

    generated_identity = all(data.get(key) for key in (
        "voice_manifest_schema_version",
        "source_bundle_id",
        "configuration_id",
    ))
    generated_runtime = str(data.get("kind") or "") in {
        "festival_unisyn_runtime_index",
        "generated_festival_voice_manifest",
    }
    generated_builder = bool(
        data.get("front_door_builder_version") or data.get("builder_version")
    )
    if not (generated_identity and (generated_runtime or generated_builder)):
        return {}
    policy = dict(DEFAULT_GENERATED_VOICE_OUTPUT_CALIBRATION)
    policy["policy_source"] = "legacy_generated_voice_default"
    return policy


def _stable_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_language_code(value: object) -> str:
    code = str(value or "").strip().casefold().replace("_", "-")
    return {"jp": "ja", "japanese": "ja", "english": "en"}.get(
        code, code
    )


@dataclass(frozen=True)
class SourceRecordingBundle:
    """Identity of source recordings, independent of a build policy."""

    source_bundle_id: str
    inventory_sha256: str
    source_scope: str = "."
    source_kind: str = "utau_recording_bundle"
    oto_files: tuple[Mapping[str, object], ...] = ()
    recording_files: tuple[Mapping[str, object], ...] = ()
    metadata_files: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )
    speaker_pitch_analysis: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_source_manifest(
        cls, source: Mapping[str, object]
    ) -> "SourceRecordingBundle":
        recordings = [
            dict(item) for item in (source.get("recording_files") or ())
        ]
        # OTO files define linguistic configurations, not recording identity.
        # When recording evidence exists, derive the source bundle solely from
        # its path-neutral sample inventory so English and Japanese OTOs over
        # the same recordings retain one source ID.
        fingerprint = (
            _stable_digest({
                "source_kind": "utau_recording_bundle",
                "recording_files": sorted(
                    recordings,
                    key=lambda item: str(item.get("path") or "").casefold(),
                ),
            })
            if recordings else str(source.get("fingerprint_sha256") or "")
        )
        if not fingerprint:
            fingerprint = _stable_digest({
                "source_scope": str(source.get("source_scope") or "."),
                "oto_files": list(source.get("oto_files") or ()),
                "recording_files": list(
                    source.get("recording_files") or ()
                ),
                "metadata_files": dict(source.get("metadata_files") or {}),
            })
        return cls(
            source_bundle_id="srb_" + fingerprint[:24],
            inventory_sha256=fingerprint,
            source_scope=str(source.get("source_scope") or "."),
            oto_files=tuple(dict(item) for item in (
                source.get("oto_files") or ()
            )),
            recording_files=tuple(recordings),
            metadata_files={
                str(key): dict(item)
                for key, item in dict(
                    source.get("metadata_files") or {}
                ).items()
            },
            speaker_pitch_analysis=dict(
                source.get("speaker_pitch_analysis") or {}
            ),
        )

    def to_dict(self) -> dict[str, object]:
        result = {
            "source_bundle_id": self.source_bundle_id,
            "source_kind": self.source_kind,
            "inventory_sha256": self.inventory_sha256,
            "source_scope": self.source_scope,
            "oto_files": [dict(item) for item in self.oto_files],
            "recording_files": [
                dict(item) for item in self.recording_files
            ],
            "metadata_files": {
                key: dict(self.metadata_files[key])
                for key in sorted(self.metadata_files)
            },
        }
        if self.speaker_pitch_analysis:
            result["speaker_pitch_analysis"] = dict(
                self.speaker_pitch_analysis
            )
        return result


def source_recording_bundle_from_paths(
    sample_root: Path,
    *,
    oto_files: Sequence[Path],
    recording_files: Sequence[Path],
    metadata_files: Sequence[Path] = (),
    speaker_pitch_analysis: Optional[Mapping[str, object]] = None,
) -> SourceRecordingBundle:
    """Build path-neutral source identity from an explicitly selected scope."""
    root = Path(sample_root).expanduser().resolve()

    def record(path: Path) -> dict[str, object]:
        resolved = Path(path).expanduser().resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"source inventory path is outside the sample root: {path}"
            ) from exc
        content = resolved.read_bytes()
        return {
            "path": relative,
            "sha256": hashlib.sha256(content).hexdigest(),
            "byte_length": len(content),
        }

    oto_rows = sorted(
        (record(path) for path in oto_files),
        key=lambda item: str(item["path"]).casefold(),
    )
    recording_rows = sorted(
        (record(path) for path in recording_files),
        key=lambda item: str(item["path"]).casefold(),
    )
    metadata_rows = sorted(
        (record(path) for path in metadata_files),
        key=lambda item: str(item["path"]).casefold(),
    )
    metadata = {
        str(item["path"]): item for item in metadata_rows
    }
    fingerprint_payload = {
        "source_scope": ".",
        "oto_files": oto_rows,
        "recording_files": recording_rows,
        "metadata_files": metadata,
    }
    return SourceRecordingBundle.from_source_manifest({
        **fingerprint_payload,
        "fingerprint_sha256": _stable_digest(fingerprint_payload),
        "speaker_pitch_analysis": dict(speaker_pitch_analysis or {}),
    })


@dataclass(frozen=True)
class VoiceConfiguration:
    """One explicit language/alias interpretation of a recording bundle."""

    configuration_id: str
    source_bundle_id: str
    primary_language: str
    supported_languages: tuple[str, ...]
    alias_system: str
    canonical_phone_namespace: str
    bank_type: str
    voice_entry_points: Mapping[str, str] = field(default_factory=dict)
    frontend: str = "japanese_frontend_v1"
    duration_model: str = "japanese_baseline_v1"
    prosody_model: str = "japanese_accent_v1"
    selected_subbank_id: Optional[str] = None
    selected_voice_color: Optional[str] = None
    selection_status: str = "explicit"

    def __post_init__(self) -> None:
        primary = normalize_language_code(self.primary_language)
        supported = tuple(
            dict.fromkeys(normalize_language_code(item)
                          for item in self.supported_languages)
        )
        if not primary or primary not in supported:
            raise ValueError(
                "primary_language must be present in supported_languages"
            )
        if primary == "ja" and self.bank_type not in JAPANESE_BUILD_TYPES:
            raise ValueError(
                "Japanese bank_type must be one of cv, vcv, or cvvc"
            )
        if not self.bank_type:
            raise ValueError("bank_type must not be empty")
        if self.selection_status not in {"explicit", "analysis-proposal"}:
            raise ValueError(
                "selection_status must be explicit or analysis-proposal"
            )

    @classmethod
    def japanese(
        cls,
        *,
        source_bundle_id: str,
        bank_type: str,
        configuration_policy: Mapping[str, object],
        voice_entry_point: str = "",
        selected_subbank_id: Optional[str] = None,
        selected_voice_color: Optional[str] = None,
        selection_status: str = "explicit",
    ) -> "VoiceConfiguration":
        if bank_type not in JAPANESE_BUILD_TYPES:
            raise ValueError(
                "A build requires an explicit Japanese bank type: "
                "cv, vcv, or cvvc. Auto and mixed are analysis modes only."
            )
        identity = {
            "source_bundle_id": source_bundle_id,
            "primary_language": "ja",
            "bank_type": bank_type,
            "policy": dict(configuration_policy),
            "selected_subbank_id": selected_subbank_id,
            "selected_voice_color": selected_voice_color,
            "selection_status": selection_status,
        }
        configuration_id = "vcfg_" + _stable_digest(identity)[:24]
        alias_system = f"utau-japanese-{bank_type}-v1"
        namespace = f"{configuration_id}.ja"
        entries = ({"ja": voice_entry_point} if voice_entry_point else {})
        return cls(
            configuration_id=configuration_id,
            source_bundle_id=source_bundle_id,
            primary_language="ja",
            supported_languages=("ja",),
            alias_system=alias_system,
            canonical_phone_namespace=namespace + ".phones",
            bank_type=bank_type,
            voice_entry_points=entries,
            selected_subbank_id=selected_subbank_id,
            selected_voice_color=selected_voice_color,
            selection_status=selection_status,
        )

    @classmethod
    def single_language(
        cls,
        *,
        source_bundle_id: str,
        language: str,
        bank_type: str,
        alias_system: str,
        configuration_policy: Mapping[str, object],
        voice_entry_point: str,
        frontend: str,
        duration_model: str,
        prosody_model: str,
        selected_subbank_id: Optional[str] = None,
        selected_voice_color: Optional[str] = None,
    ) -> "VoiceConfiguration":
        """Create one explicit non-Japanese linguistic configuration."""
        language = normalize_language_code(language)
        if language == "ja":
            raise ValueError("Use VoiceConfiguration.japanese for Japanese")
        if not language or not bank_type or not alias_system:
            raise ValueError(
                "language, bank_type, and alias_system are required"
            )
        identity = {
            "source_bundle_id": source_bundle_id,
            "primary_language": language,
            "bank_type": bank_type,
            "alias_system": alias_system,
            "policy": dict(configuration_policy),
            "selected_subbank_id": selected_subbank_id,
            "selected_voice_color": selected_voice_color,
        }
        configuration_id = "vcfg_" + _stable_digest(identity)[:24]
        namespace = f"{configuration_id}.{language}"
        return cls(
            configuration_id=configuration_id,
            source_bundle_id=source_bundle_id,
            primary_language=language,
            supported_languages=(language,),
            alias_system=alias_system,
            canonical_phone_namespace=namespace + ".phones",
            bank_type=bank_type,
            voice_entry_points={language: str(voice_entry_point)},
            frontend=frontend,
            duration_model=duration_model,
            prosody_model=prosody_model,
            selected_subbank_id=selected_subbank_id,
            selected_voice_color=selected_voice_color,
            selection_status="explicit",
        )

    @classmethod
    def arpasing(
        cls,
        *,
        source_bundle_id: str,
        primary_language: str,
        supported_languages: Sequence[str],
        configuration_policy: Mapping[str, object],
        voice_entry_points: Mapping[str, str],
        selected_voice_color: Optional[str] = None,
    ) -> "VoiceConfiguration":
        """Create one shared ARPAsing unit database with explicit frontends."""
        primary = normalize_language_code(primary_language)
        supported = tuple(dict.fromkeys(
            normalize_language_code(item) for item in supported_languages
        ))
        if primary not in {"en", "asaxi"}:
            raise ValueError("ARPAsing primary language must be en or asaxi")
        if not supported or primary not in supported:
            raise ValueError("ARPAsing supported languages must include primary")
        invalid = sorted(set(supported) - {"en", "asaxi", "ja"})
        if invalid:
            raise ValueError("Unsupported ARPAsing frontend: " + ", ".join(invalid))
        entries = {
            normalize_language_code(key): str(value)
            for key, value in voice_entry_points.items()
            if normalize_language_code(key) in supported and str(value)
        }
        if set(entries) != set(supported):
            raise ValueError("Every supported ARPAsing language needs an entry point")
        identity = {
            "source_bundle_id": source_bundle_id,
            "primary_language": primary,
            "supported_languages": list(supported),
            "bank_type": "arpasing",
            "alias_system": "utau-arpasing-profile-v1",
            "policy": dict(configuration_policy),
            "selected_voice_color": selected_voice_color,
        }
        configuration_id = "vcfg_" + _stable_digest(identity)[:24]
        return cls(
            configuration_id=configuration_id,
            source_bundle_id=source_bundle_id,
            primary_language=primary,
            supported_languages=supported,
            alias_system="utau-arpasing-profile-v1",
            canonical_phone_namespace=f"{configuration_id}.arpasing.phones",
            bank_type="arpasing",
            voice_entry_points=entries,
            frontend="festival-arpasing-multilingual-v1",
            duration_model="language-scoped-shared-units-v1",
            prosody_model="language-scoped-shared-units-v1",
            selected_voice_color=selected_voice_color,
            selection_status="explicit",
        )

    @property
    def alias_namespace(self) -> str:
        return f"{self.configuration_id}.aliases"

    def with_entry_point(self, language: str, entry_point: str):
        entries = dict(self.voice_entry_points)
        entries[normalize_language_code(language)] = str(entry_point)
        return VoiceConfiguration(
            configuration_id=self.configuration_id,
            source_bundle_id=self.source_bundle_id,
            primary_language=self.primary_language,
            supported_languages=self.supported_languages,
            alias_system=self.alias_system,
            canonical_phone_namespace=self.canonical_phone_namespace,
            bank_type=self.bank_type,
            voice_entry_points=entries,
            frontend=self.frontend,
            duration_model=self.duration_model,
            prosody_model=self.prosody_model,
            selected_subbank_id=self.selected_subbank_id,
            selected_voice_color=self.selected_voice_color,
            selection_status=self.selection_status,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration_id": self.configuration_id,
            "source_bundle_id": self.source_bundle_id,
            "primary_language": self.primary_language,
            "supported_languages": list(self.supported_languages),
            "alias_system": self.alias_system,
            "alias_namespace": self.alias_namespace,
            "canonical_phone_namespace": self.canonical_phone_namespace,
            "bank_type": self.bank_type,
            "voice_entry_points": {
                key: self.voice_entry_points[key]
                for key in sorted(self.voice_entry_points)
            },
            "frontend": self.frontend,
            "duration_model": self.duration_model,
            "prosody_model": self.prosody_model,
            "selected_subbank_id": self.selected_subbank_id,
            "selected_voice_color": self.selected_voice_color,
            "selection_status": self.selection_status,
        }


@dataclass(frozen=True)
class VoiceCompatibility:
    metadata_status: str
    primary_language: Optional[str]
    supported_languages: tuple[str, ...]
    voice_entry_points: Mapping[str, str]
    phones: tuple[str, ...] = ()
    configuration_id: Optional[str] = None
    reason: str = ""

    @property
    def is_legacy(self) -> bool:
        return self.metadata_status == "legacy"

    @property
    def is_current(self) -> bool:
        return self.metadata_status == "current"

    def supports(self, language: str) -> bool:
        return normalize_language_code(language) in self.supported_languages

    def to_dict(self) -> dict[str, object]:
        return {
            "metadata_status": self.metadata_status,
            "primary_language": self.primary_language,
            "supported_languages": list(self.supported_languages),
            "voice_entry_points": dict(self.voice_entry_points),
            "phones": list(self.phones),
            "configuration_id": self.configuration_id,
            "reason": self.reason,
        }


def read_voice_compatibility(
    metadata: Mapping[str, object],
    *,
    configured_entry_points: Optional[Mapping[str, object]] = None,
) -> VoiceCompatibility:
    """Interpret current metadata and label older generated voices honestly."""
    data = dict(metadata or {})
    configured = {
        normalize_language_code(key): str(value)
        for key, value in dict(configured_entry_points or {}).items()
        if value
    }
    primary = normalize_language_code(data.get("primary_language"))
    supported = tuple(dict.fromkeys(
        normalize_language_code(item)
        for item in (data.get("supported_languages") or ())
        if normalize_language_code(item)
    ))
    entries = {
        normalize_language_code(key): str(value)
        for key, value in dict(data.get("voice_entry_points") or {}).items()
        if value
    }
    entries.update({key: value for key, value in configured.items()
                    if key not in entries})
    if (
        data.get("source_bundle_id")
        and data.get("configuration_id")
        and primary
        and supported
        and data.get("alias_system")
        and entries
    ):
        return VoiceCompatibility(
            metadata_status="current",
            primary_language=primary,
            supported_languages=supported,
            voice_entry_points=entries,
            phones=tuple(str(item) for item in (data.get("phones") or ())),
            configuration_id=str(data.get("configuration_id")),
        )

    legacy_language = normalize_language_code(data.get("language"))
    legacy_entry = str(data.get("voice_entry_point") or "")
    if legacy_language and legacy_entry:
        entries.setdefault(legacy_language, legacy_entry)
        return VoiceCompatibility(
            metadata_status="legacy",
            primary_language=legacy_language,
            supported_languages=(legacy_language,),
            voice_entry_points=entries,
            phones=tuple(str(item) for item in (data.get("phones") or ())),
            reason=(
                "Legacy generated metadata: rebuild this voice to record its "
                "source bundle and explicit voice configuration."
            ),
        )

    if configured:
        languages = tuple(sorted(configured))
        return VoiceCompatibility(
            metadata_status="legacy",
            primary_language=(languages[0] if len(languages) == 1 else None),
            supported_languages=languages,
            voice_entry_points=configured,
            reason=(
                "Legacy registration without a generated voice manifest; "
                "language compatibility is inferred from registered entry "
                "points. Rebuild when practical."
            ),
        )
    return VoiceCompatibility(
        metadata_status="unknown",
        primary_language=None,
        supported_languages=(),
        voice_entry_points={},
        reason=(
            "No generated voice compatibility metadata was found. The GUI "
            "cannot safely infer this voice's languages or phoneset."
        ),
    )


def generated_voice_fields(
    bundle: SourceRecordingBundle,
    configuration: VoiceConfiguration,
) -> dict[str, object]:
    """Required top-level fields plus inspectable nested records."""
    result = {
        "voice_manifest_schema_version": VOICE_MANIFEST_SCHEMA_VERSION,
        "source_bundle_id": bundle.source_bundle_id,
        "configuration_id": configuration.configuration_id,
        "primary_language": configuration.primary_language,
        "supported_languages": list(configuration.supported_languages),
        "alias_system": configuration.alias_system,
        "alias_namespace": configuration.alias_namespace,
        "canonical_phone_namespace": (
            configuration.canonical_phone_namespace
        ),
        "voice_entry_points": dict(configuration.voice_entry_points),
        "source_recording_bundle": bundle.to_dict(),
        "voice_configuration": configuration.to_dict(),
    }
    if bundle.speaker_pitch_analysis:
        result["speaker_pitch_analysis"] = dict(
            bundle.speaker_pitch_analysis
        )
    return result
