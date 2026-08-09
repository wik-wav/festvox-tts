"""Japanese UTAU bank profiles for the Phase 2 candidate compiler.

Profiles are proposals and explicit user policy, not converted voicebanks.  The
module reads source metadata without writing to it and deliberately does not
import the production English converter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import japanese_utau as ju


PROFILE_SCHEMA_VERSION = 1
PROFILE_SCHEMA_STATUS = "phase2-provisional"
BANK_CONFIGURATIONS = ("auto", "cv", "vcv", "cvvc", "mixed")
CANDIDATE_FAMILIES = ("cv", "vcv", "cvvc", "extra")
PROFILE_ROLES = (
    "mora_cv",
    "phrase_start_cv",
    "vowel_blend",
    "vcv_mora",
    "vc_transition",
    "release",
    "special_mora",
    "silence",
    "breath",
    "extra",
    "unresolved",
)


@dataclass(frozen=True)
class ProfileDiagnostic:
    code: str
    message: str
    severity: str = "warning"
    action: Optional[str] = None
    source_path: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.action:
            result["action"] = self.action
        if self.source_path:
            result["source_path"] = self.source_path
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProfileDiagnostic":
        return cls(
            code=str(value.get("code", "profile_diagnostic")),
            message=str(value.get("message", "")),
            severity=str(value.get("severity", "warning")),
            action=(str(value["action"]) if value.get("action") else None),
            source_path=(
                str(value["source_path"])
                if value.get("source_path") else None
            ),
        )


@dataclass(frozen=True)
class JapaneseSubbank:
    subbank_id: str
    color: str = ""
    prefix: str = ""
    suffix: str = ""
    tone_ranges: tuple[str, ...] = ()
    source: str = "profile"
    order: int = 0

    def to_dict(self) -> dict:
        return {
            "subbank_id": self.subbank_id,
            "color": self.color,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "tone_ranges": list(self.tone_ranges),
            "source": self.source,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "JapaneseSubbank":
        return cls(
            subbank_id=str(value.get("subbank_id", "")),
            color=str(value.get("color", "")),
            prefix=str(value.get("prefix", "")),
            suffix=str(value.get("suffix", "")),
            tone_ranges=tuple(str(item) for item in (
                value.get("tone_ranges") or ()
            )),
            source=str(value.get("source", "profile")),
            order=int(value.get("order", 0)),
        )


@dataclass(frozen=True)
class JapaneseAliasOverride:
    """An exact alias or source-location interpretation supplied by the user."""

    role: str
    family: Optional[str] = None
    mora: Optional[str] = None
    left_context: Optional[str] = None
    right_context: Optional[str] = None
    enabled: bool = True
    priority: int = 0
    note: str = ""

    def __post_init__(self) -> None:
        if self.role not in PROFILE_ROLES:
            raise ValueError(
                f"alias override role must be one of {PROFILE_ROLES}, "
                f"got {self.role!r}"
            )
        if self.family is not None and self.family not in CANDIDATE_FAMILIES:
            raise ValueError(
                f"alias override family must be one of {CANDIDATE_FAMILIES}, "
                f"got {self.family!r}"
            )

    def to_dict(self) -> dict:
        result = {
            "role": self.role,
            "enabled": self.enabled,
            "priority": self.priority,
        }
        for key in ("family", "mora", "left_context", "right_context"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.note:
            result["note"] = self.note
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "JapaneseAliasOverride":
        return cls(
            role=str(value.get("role", "unresolved")),
            family=(str(value["family"]) if value.get("family") else None),
            mora=(str(value["mora"]) if value.get("mora") else None),
            left_context=(
                str(value["left_context"])
                if value.get("left_context") else None
            ),
            right_context=(
                str(value["right_context"])
                if value.get("right_context") else None
            ),
            enabled=bool(value.get("enabled", True)),
            priority=int(value.get("priority", 0)),
            note=str(value.get("note", "")),
        )


@dataclass(frozen=True)
class JapaneseMoraicNasalAllophone:
    """Bank-specific source labels for one acoustic realization of /N/.

    These labels are deliberately profile data.  UTAU banks do not share a
    standard meaning for aliases such as ``n``, ``nn``, ``ng``, or numbered
    kana, while the canonical linguistic phone remains Japanese ``N``.
    """

    mora_aliases: tuple[str, ...] = ()
    context_aliases: tuple[str, ...] = ()
    following_phones: tuple[str, ...] = ()
    default: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "mora_aliases", "context_aliases", "following_phones"
        ):
            values = getattr(self, field_name)
            if any(not str(value).strip() for value in values):
                raise ValueError(f"{field_name} cannot contain empty values")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} cannot contain duplicates")

    def to_dict(self) -> dict:
        result = {
            "mora_aliases": list(self.mora_aliases),
            "context_aliases": list(self.context_aliases),
            "following_phones": list(self.following_phones),
            "default": self.default,
        }
        if self.note:
            result["note"] = self.note
        return result

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "JapaneseMoraicNasalAllophone":
        return cls(
            mora_aliases=tuple(str(item) for item in (
                value.get("mora_aliases") or ()
            )),
            context_aliases=tuple(str(item) for item in (
                value.get("context_aliases") or ()
            )),
            following_phones=tuple(str(item) for item in (
                value.get("following_phones") or ()
            )),
            default=bool(value.get("default", False)),
            note=str(value.get("note", "")),
        )


@dataclass(frozen=True)
class BankContext:
    bank_root: Path
    source_scope: Path

    @property
    def source_scope_relative(self) -> str:
        return _relative_path(self.source_scope, self.bank_root)


@dataclass(frozen=True)
class JapaneseBankProfile:
    bank_configuration: str = "auto"
    inferred_configuration: str = "unknown"
    inference_confidence: float = 0.0
    default_encoding: Optional[str] = None
    encoding_overrides: dict[str, str] = field(default_factory=dict)
    alias_prefixes: tuple[str, ...] = ()
    alias_suffixes: tuple[str, ...] = ()
    voice_color: Optional[str] = None
    enabled_families: tuple[str, ...] = CANDIDATE_FAMILIES
    alias_overrides: dict[str, JapaneseAliasOverride] = field(
        default_factory=dict
    )
    moraic_nasal_allophones: dict[
        str, JapaneseMoraicNasalAllophone
    ] = field(default_factory=dict)
    unknown_alias_policy: str = "preserve"
    subbanks: tuple[JapaneseSubbank, ...] = ()
    source_scope: str = "."
    metadata_files: dict[str, dict] = field(default_factory=dict)
    diagnostics: tuple[ProfileDiagnostic, ...] = ()
    schema_version: int = PROFILE_SCHEMA_VERSION
    schema_status: str = PROFILE_SCHEMA_STATUS
    language: str = "ja"
    source_bundle_id: str = ""
    configuration_id: str = ""
    alias_system: str = ""
    alias_namespace: str = ""
    canonical_phone_namespace: str = ""
    _bank_root: Optional[Path] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.bank_configuration not in BANK_CONFIGURATIONS:
            raise ValueError(
                f"bank_configuration must be one of {BANK_CONFIGURATIONS}, "
                f"got {self.bank_configuration!r}"
            )
        invalid_families = set(self.enabled_families) - set(
            CANDIDATE_FAMILIES
        )
        if invalid_families:
            raise ValueError(
                f"unknown candidate families: {sorted(invalid_families)}"
            )
        if self.unknown_alias_policy != "preserve":
            raise ValueError(
                "Phase 2 only permits unknown_alias_policy='preserve'"
            )
        defaults = [
            name for name, rule in self.moraic_nasal_allophones.items()
            if rule.default
        ]
        if len(defaults) > 1:
            raise ValueError(
                "moraic_nasal_allophones permits at most one default"
            )
        for name in self.moraic_nasal_allophones:
            if not name.strip():
                raise ValueError("moraic nasal allophone IDs cannot be empty")
        for attribute in (
            "mora_aliases", "context_aliases", "following_phones"
        ):
            owners: dict[str, str] = {}
            for name, rule in self.moraic_nasal_allophones.items():
                for value in getattr(rule, attribute):
                    if value in owners:
                        raise ValueError(
                            f"{attribute} value {value!r} is assigned to both "
                            f"{owners[value]!r} and {name!r}"
                        )
                    owners[value] = name

    @property
    def effective_configuration(self) -> str:
        if self.bank_configuration != "auto":
            return self.bank_configuration
        return self.inferred_configuration

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "schema_status": self.schema_status,
            "kind": "japanese_utau_bank_profile",
            "language": self.language,
            "source_bundle_id": self.source_bundle_id,
            "configuration_id": self.configuration_id,
            "alias_system": self.alias_system,
            "alias_namespace": self.alias_namespace,
            "canonical_phone_namespace": self.canonical_phone_namespace,
            "bank_configuration": self.bank_configuration,
            "inferred_configuration": self.inferred_configuration,
            "effective_configuration": self.effective_configuration,
            "inference_confidence": self.inference_confidence,
            "default_encoding": self.default_encoding,
            "encoding_overrides": dict(sorted(
                self.encoding_overrides.items()
            )),
            "alias_prefixes": list(self.alias_prefixes),
            "alias_suffixes": list(self.alias_suffixes),
            "voice_color": self.voice_color,
            "enabled_families": list(self.enabled_families),
            "alias_overrides": {
                key: self.alias_overrides[key].to_dict()
                for key in sorted(self.alias_overrides)
            },
            "moraic_nasal_allophones": {
                key: self.moraic_nasal_allophones[key].to_dict()
                for key in sorted(self.moraic_nasal_allophones)
            },
            "unknown_alias_policy": self.unknown_alias_policy,
            "subbanks": [item.to_dict() for item in self.subbanks],
            "source_scope": self.source_scope,
            "metadata_files": {
                key: dict(self.metadata_files[key])
                for key in sorted(self.metadata_files)
            },
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "JapaneseBankProfile":
        if value.get("language", "ja") != "ja":
            raise ValueError("Japanese bank profiles must use language='ja'")
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            schema_status=str(value.get(
                "schema_status", PROFILE_SCHEMA_STATUS
            )),
            language="ja",
            source_bundle_id=str(value.get("source_bundle_id") or ""),
            configuration_id=str(value.get("configuration_id") or ""),
            alias_system=str(value.get("alias_system") or ""),
            alias_namespace=str(value.get("alias_namespace") or ""),
            canonical_phone_namespace=str(
                value.get("canonical_phone_namespace") or ""
            ),
            bank_configuration=str(value.get(
                "bank_configuration", "auto"
            )),
            inferred_configuration=str(value.get(
                "inferred_configuration", "unknown"
            )),
            inference_confidence=float(value.get(
                "inference_confidence", 0.0
            )),
            default_encoding=(
                str(value["default_encoding"])
                if value.get("default_encoding") else None
            ),
            encoding_overrides={
                str(key): str(item)
                for key, item in dict(
                    value.get("encoding_overrides") or {}
                ).items()
            },
            alias_prefixes=tuple(str(item) for item in (
                value.get("alias_prefixes") or ()
            )),
            alias_suffixes=tuple(str(item) for item in (
                value.get("alias_suffixes") or ()
            )),
            voice_color=(
                str(value["voice_color"])
                if value.get("voice_color") is not None else None
            ),
            enabled_families=tuple(str(item) for item in (
                value.get("enabled_families") or CANDIDATE_FAMILIES
            )),
            alias_overrides={
                str(key): JapaneseAliasOverride.from_dict(item)
                for key, item in dict(
                    value.get("alias_overrides") or {}
                ).items()
            },
            moraic_nasal_allophones={
                str(key): JapaneseMoraicNasalAllophone.from_dict(item)
                for key, item in dict(
                    value.get("moraic_nasal_allophones") or {}
                ).items()
            },
            unknown_alias_policy=str(value.get(
                "unknown_alias_policy", "preserve"
            )),
            subbanks=tuple(JapaneseSubbank.from_dict(item) for item in (
                value.get("subbanks") or ()
            )),
            source_scope=str(value.get("source_scope", ".")),
            metadata_files={
                str(key): dict(item)
                for key, item in dict(
                    value.get("metadata_files") or {}
                ).items()
            },
            diagnostics=tuple(ProfileDiagnostic.from_dict(item) for item in (
                value.get("diagnostics") or ()
            )),
        )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _relative_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path is outside the source bank: {path}") from exc
    value = relative.as_posix()
    return value or "."


def resolve_bank_context(source: Path) -> BankContext:
    """Find a nearby bank metadata root while retaining the selected scope."""
    selected = Path(source).expanduser().resolve()
    if not selected.exists():
        raise FileNotFoundError(f"Japanese UTAU source not found: {selected}")
    start = selected.parent if selected.is_file() else selected
    bank_root = start
    current = start
    for _ in range(4):
        if (current / "character.yaml").is_file() \
                or (current / "prefix.map").is_file():
            bank_root = current
            break
        if current.parent == current:
            break
        current = current.parent
    return BankContext(bank_root=bank_root, source_scope=selected)


def _strip_yaml_comment(value: str) -> str:
    quoted = None
    escaped = False
    output = []
    for character in value:
        if escaped:
            output.append(character)
            escaped = False
            continue
        if character == "\\" and quoted == '"':
            output.append(character)
            escaped = True
            continue
        if character in {"'", '"'}:
            if quoted is None:
                quoted = character
            elif quoted == character:
                quoted = None
            output.append(character)
            continue
        if character == "#" and quoted is None and (
            not output or output[-1].isspace()
        ):
            break
        output.append(character)
    return "".join(output)


def _yaml_scalar(value: str) -> str:
    value = _strip_yaml_comment(value).strip()
    if value.startswith('"') and value.endswith('"'):
        try:
            return str(json.loads(value))
        except json.JSONDecodeError:
            return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    return value


def _yaml_list(value: str) -> list[str]:
    value = _strip_yaml_comment(value).strip()
    if not (value.startswith("[") and value.endswith("]")):
        return []
    return [
        item for item in (
            _yaml_scalar(part) for part in value[1:-1].split(",")
        ) if item
    ]


def _read_metadata_text(path: Path) -> tuple[str, str, str, int]:
    decoded = ju.decode_text_file(path)
    return (
        decoded.text,
        decoded.encoding,
        decoded.sha256,
        decoded.byte_length,
    )


def _parse_character_yaml(path: Path) -> tuple[dict, dict]:
    text, encoding, digest, byte_length = _read_metadata_text(path)
    result = {"subbanks": [], "text_file_encoding": None}
    current = None
    in_subbanks = False
    in_tone_ranges = False
    for raw_line in text.splitlines():
        without_comment = _strip_yaml_comment(raw_line)
        if not without_comment.strip():
            continue
        indent = len(without_comment) - len(without_comment.lstrip())
        line = without_comment.strip()
        if not in_subbanks:
            if line.startswith("text_file_encoding:"):
                result["text_file_encoding"] = _yaml_scalar(
                    line.split(":", 1)[1]
                ) or None
            if line == "subbanks:":
                in_subbanks = True
            continue
        if indent == 0 and not line.startswith("-") and ":" in line:
            break
        if line.startswith("- "):
            body = line[2:].strip()
            if ":" in body:
                if current is not None:
                    result["subbanks"].append(current)
                current = {
                    "color": "",
                    "prefix": "",
                    "suffix": "",
                    "tone_ranges": [],
                }
                key, value = body.split(":", 1)
                key = key.strip()
                if key in {"color", "prefix", "suffix"}:
                    current[key] = _yaml_scalar(value)
                elif key == "tone_ranges":
                    current["tone_ranges"].extend(_yaml_list(value))
                in_tone_ranges = key == "tone_ranges"
            elif current is not None and in_tone_ranges:
                tone_range = _yaml_scalar(body)
                if tone_range:
                    current["tone_ranges"].append(tone_range)
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"color", "prefix", "suffix"}:
            current[key] = _yaml_scalar(value)
            in_tone_ranges = False
        elif key == "tone_ranges":
            current["tone_ranges"].extend(_yaml_list(value))
            in_tone_ranges = True
        else:
            in_tone_ranges = False
    if current is not None:
        result["subbanks"].append(current)
    provenance = {
        "path": "character.yaml",
        "sha256": digest,
        "byte_length": byte_length,
        "encoding": encoding,
    }
    return result, provenance


def _parse_prefix_map(path: Path) -> tuple[list[dict], dict]:
    text, encoding, digest, byte_length = _read_metadata_text(path)
    grouped: dict[tuple[str, str], list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        if not line.strip() or line.lstrip().startswith(("#", ";")):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        tone = fields[0].strip()
        prefix = fields[1]
        suffix = "\t".join(fields[2:]).strip()
        grouped.setdefault((prefix, suffix), []).append(tone)
    rows = [
        {
            "color": "",
            "prefix": prefix,
            "suffix": suffix,
            "tone_ranges": tones,
        }
        for (prefix, suffix), tones in sorted(grouped.items())
    ]
    provenance = {
        "path": "prefix.map",
        "sha256": digest,
        "byte_length": byte_length,
        "encoding": encoding,
    }
    return rows, provenance


def _subbank_id(row: Mapping[str, object], source: str, order: int) -> str:
    payload = {
        "color": str(row.get("color", "")),
        "prefix": str(row.get("prefix", "")),
        "suffix": str(row.get("suffix", "")),
        "tone_ranges": [str(item) for item in (
            row.get("tone_ranges") or ()
        )],
        "source": source,
        "order": order,
    }
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "jsb_" + hashlib.sha256(raw).hexdigest()[:20]


def _make_subbank(
    row: Mapping[str, object],
    source: str,
    order: int,
) -> JapaneseSubbank:
    return JapaneseSubbank(
        subbank_id=_subbank_id(row, source, order),
        color=str(row.get("color", "")),
        prefix=str(row.get("prefix", "")),
        suffix=str(row.get("suffix", "")),
        tone_ranges=tuple(str(item) for item in (
            row.get("tone_ranges") or ()
        )),
        source=source,
        order=order,
    )


def _normalize_encoding(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    folded = value.strip().casefold().replace("-", "_")
    if folded in {"shift_jis", "shiftjis", "sjis", "windows_31j"}:
        return "cp932"
    return value.strip()


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in values if item))


def infer_bank_profile(
    source: Path,
    *,
    bank_configuration: str = "auto",
    default_encoding: Optional[str] = None,
    encoding_overrides: Optional[Mapping[str, str]] = None,
    alias_prefixes: Sequence[str] = (),
    alias_suffixes: Sequence[str] = (),
    voice_color: Optional[str] = None,
    enabled_families: Sequence[str] = CANDIDATE_FAMILIES,
    alias_overrides: Optional[
        Mapping[str, JapaneseAliasOverride | Mapping[str, object]]
    ] = None,
    moraic_nasal_allophones: Optional[
        Mapping[
            str,
            JapaneseMoraicNasalAllophone | Mapping[str, object],
        ]
    ] = None,
    oto_files: Optional[Sequence[Path]] = None,
) -> JapaneseBankProfile:
    """Infer a read-only profile proposal and merge explicit policy."""
    if bank_configuration not in BANK_CONFIGURATIONS:
        raise ValueError(
            f"bank_configuration must be one of {BANK_CONFIGURATIONS}"
        )
    context = resolve_bank_context(source)
    metadata_files: dict[str, dict] = {}
    diagnostics: list[ProfileDiagnostic] = []
    subbanks: list[JapaneseSubbank] = []
    metadata_encoding = None

    character_yaml = context.bank_root / "character.yaml"
    if character_yaml.is_file():
        parsed, provenance = _parse_character_yaml(character_yaml)
        metadata_files["character.yaml"] = provenance
        metadata_encoding = _normalize_encoding(
            parsed.get("text_file_encoding")
        )
        for row in parsed["subbanks"]:
            subbanks.append(_make_subbank(
                row, "character.yaml", len(subbanks)
            ))
        if not parsed["subbanks"]:
            diagnostics.append(ProfileDiagnostic(
                code="character_yaml_no_subbanks",
                message=(
                    "character.yaml contains no readable OpenUtau subbanks."
                ),
                action=(
                    "Set alias prefixes/suffixes explicitly if aliases carry "
                    "pitch or voice-color affixes."
                ),
                source_path="character.yaml",
            ))

    prefix_map = context.bank_root / "prefix.map"
    if prefix_map.is_file():
        rows, provenance = _parse_prefix_map(prefix_map)
        metadata_files["prefix.map"] = provenance
        for row in rows:
            duplicate = any(
                item.color == str(row.get("color", ""))
                and item.prefix == str(row.get("prefix", ""))
                and item.suffix == str(row.get("suffix", ""))
                for item in subbanks
            )
            if not duplicate:
                subbanks.append(_make_subbank(
                    row, "prefix.map", len(subbanks)
                ))

    prefixes = _unique(
        tuple(item.prefix for item in subbanks) + tuple(alias_prefixes)
    )
    suffixes = _unique(
        tuple(item.suffix for item in subbanks) + tuple(alias_suffixes)
    )
    effective_encoding = _normalize_encoding(
        default_encoding
    ) or metadata_encoding

    analysis = ju.analyze_bank(
        context.source_scope,
        encoding_override=effective_encoding,
        bank_type="auto",
        alias_prefixes=prefixes,
        alias_suffixes=suffixes,
        oto_files=oto_files,
    )
    overrides: dict[str, JapaneseAliasOverride] = {}
    for key, value in (alias_overrides or {}).items():
        overrides[str(key)] = (
            value if isinstance(value, JapaneseAliasOverride)
            else JapaneseAliasOverride.from_dict(value)
        )
    nasal_allophones: dict[str, JapaneseMoraicNasalAllophone] = {}
    for key, value in (moraic_nasal_allophones or {}).items():
        nasal_allophones[str(key)] = (
            value if isinstance(value, JapaneseMoraicNasalAllophone)
            else JapaneseMoraicNasalAllophone.from_dict(value)
        )

    return JapaneseBankProfile(
        bank_configuration=bank_configuration,
        inferred_configuration=analysis.bank_type,
        inference_confidence=analysis.confidence,
        default_encoding=effective_encoding,
        encoding_overrides={
            str(key).replace("\\", "/"): str(value)
            for key, value in (encoding_overrides or {}).items()
        },
        alias_prefixes=prefixes,
        alias_suffixes=suffixes,
        voice_color=voice_color,
        enabled_families=tuple(enabled_families),
        alias_overrides=overrides,
        moraic_nasal_allophones=nasal_allophones,
        unknown_alias_policy="preserve",
        subbanks=tuple(subbanks),
        source_scope=context.source_scope_relative,
        metadata_files=metadata_files,
        diagnostics=tuple(diagnostics),
        _bank_root=context.bank_root,
    )


def profile_json_bytes(profile: JapaneseBankProfile) -> bytes:
    return (
        json.dumps(
            profile.to_dict(), ensure_ascii=False, sort_keys=True,
            indent=2,
        ) + "\n"
    ).encode("utf-8")


def write_profile(
    profile: JapaneseBankProfile,
    output: Path,
    *,
    source_root: Optional[Path] = None,
) -> None:
    root = Path(source_root).resolve() if source_root else profile._bank_root
    target = Path(output).expanduser().resolve()
    if root is not None and _is_within(target, root):
        raise ValueError(
            "refusing to write a Japanese profile inside the source voicebank"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(profile_json_bytes(profile))


def load_profile(path: Path) -> JapaneseBankProfile:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Japanese bank profile JSON must contain an object")
    return JapaneseBankProfile.from_dict(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Infer a read-only Phase 2 Japanese UTAU bank profile."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--bank-configuration", choices=BANK_CONFIGURATIONS,
                        default="auto")
    parser.add_argument("--encoding", default=None)
    parser.add_argument("--alias-prefix", action="append", default=[])
    parser.add_argument("--alias-suffix", action="append", default=[])
    parser.add_argument("--voice-color", default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    profile = infer_bank_profile(
        args.source,
        bank_configuration=args.bank_configuration,
        default_encoding=args.encoding,
        alias_prefixes=args.alias_prefix,
        alias_suffixes=args.alias_suffix,
        voice_color=args.voice_color,
    )
    data = profile_json_bytes(profile)
    if args.output:
        write_profile(profile, args.output)
    else:
        print(data.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
