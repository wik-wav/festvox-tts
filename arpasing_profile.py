"""Dependency-free ARPAsing profile and Japanese phoneme-map support."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Optional, Sequence


PROFILE_SCHEMA_VERSION = 1
DEFAULT_PHONEME_MAP_PATH = (
    Path(__file__).resolve().parent / "profiles" / "en-jap-mapping.yaml"
)
_PHONE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ArpasingProfileDiagnostic:
    code: str
    message: str
    severity: str = "warning"
    line: Optional[int] = None
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.line is not None:
            result["line"] = self.line
        if self.details:
            result["details"] = dict(self.details)
        return result


@dataclass(frozen=True)
class ArpasingPhonemeMapEntry:
    grapheme: str
    phonemes: tuple[str, ...]
    line: int

    def to_dict(self) -> dict[str, object]:
        return {
            "grapheme": self.grapheme,
            "phonemes": list(self.phonemes),
            "line": self.line,
        }


@dataclass(frozen=True)
class ArpasingMapResolution:
    grapheme: str
    phonemes: tuple[str, ...]
    line: int
    alternatives: tuple[tuple[str, ...], ...] = ()

    @property
    def ambiguous(self) -> bool:
        return len(self.alternatives) > 1


@dataclass(frozen=True)
class ArpasingVoiceProfile:
    source_name: str
    source_sha256: str
    symbols: Mapping[str, str]
    timings: Mapping[str, float]
    replacements: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    entries: tuple[ArpasingPhonemeMapEntry, ...]
    diagnostics: tuple[ArpasingProfileDiagnostic, ...] = ()
    schema_version: int = PROFILE_SCHEMA_VERSION

    def candidates(self, grapheme: str) -> tuple[ArpasingPhonemeMapEntry, ...]:
        return tuple(item for item in self.entries if item.grapheme == grapheme)

    def resolve(
        self,
        grapheme: str,
        *,
        max_phones: Optional[int] = None,
    ) -> Optional[ArpasingMapResolution]:
        candidates = list(self.candidates(str(grapheme)))
        for source, target in self.replacements:
            if tuple([str(grapheme)]) == source:
                candidates.append(ArpasingPhonemeMapEntry(
                    str(grapheme), target, 0
                ))
        unique: list[ArpasingPhonemeMapEntry] = []
        seen: set[tuple[str, ...]] = set()
        for item in candidates:
            if not item.phonemes or item.phonemes in seen:
                continue
            if max_phones is not None and len(item.phonemes) > max_phones:
                continue
            if not all(
                phone in self.symbols or phone in {"pau", "sil", "sp"}
                for phone in item.phonemes
            ):
                continue
            unique.append(item)
            seen.add(item.phonemes)
        if not unique:
            return None
        # Rich CV/palatal mappings outrank one-phone shorthand. For mappings of
        # equal length the source file's first declaration remains stable.
        chosen = max(unique, key=lambda item: (len(item.phonemes), -item.line))
        return ArpasingMapResolution(
            grapheme=str(grapheme),
            phonemes=chosen.phonemes,
            line=chosen.line,
            alternatives=tuple(item.phonemes for item in unique),
        )

    def resolved_map(self, *, max_phones: Optional[int] = None) \
            -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for grapheme in dict.fromkeys(item.grapheme for item in self.entries):
            resolved = self.resolve(grapheme, max_phones=max_phones)
            if resolved is not None:
                result[grapheme] = list(resolved.phonemes)
        return result

    def runtime_map(self) -> dict[str, object]:
        routes = {}
        for route, grapheme in (
            ("default", "ん"),
            ("labial", "んm"),
            ("velar", "んng"),
            ("uvular", "んn"),
        ):
            resolved = self.resolve(grapheme, max_phones=1)
            if resolved is not None:
                routes[route] = resolved.phonemes[0]
        return {
            "schema_version": self.schema_version,
            "kind": "arpasing_japanese_phoneme_map",
            "source": self.source_name,
            "source_sha256": self.source_sha256,
            "grapheme_to_phones": self.resolved_map(max_phones=3),
            "timing_multipliers": {
                key: self.timings[key] for key in sorted(self.timings)
            },
            "moraic_nasal_routes": routes,
            "canonical_fallbacks": {
                "N": routes.get("default", "nn"),
                "j": "jh",
            },
        }

    def metadata(self) -> dict[str, object]:
        warning_count = sum(
            1 for item in self.diagnostics if item.severity in {"warning", "error"}
        )
        return {
            "schema_version": self.schema_version,
            "kind": "arpasing_voice_profile",
            "phoneme_map": {
                "source": self.source_name,
                "source_sha256": self.source_sha256,
                "symbol_count": len(self.symbols),
                "timing_count": len(self.timings),
                "replacement_count": len(self.replacements),
                "entry_count": len(self.entries),
                "resolved_entry_count": len(self.resolved_map(max_phones=3)),
                "diagnostic_count": len(self.diagnostics),
                "warning_count": warning_count,
            },
            "symbol_types": {
                key: self.symbols[key] for key in sorted(self.symbols)
            },
            "timing_multipliers": {
                key: self.timings[key] for key in sorted(self.timings)
            },
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def _strip_comment(value: str) -> str:
    quote = None
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if quote == '"' and character == "\\":
            escaped = True
            continue
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if character == "#" and quote is None:
            return value[:index].rstrip()
    return value.rstrip()


def _split_top_level(value: str, delimiter: str = ",") -> list[str]:
    result = []
    start = 0
    quote = None
    escaped = False
    depth = 0
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if quote == '"' and character == "\\":
            escaped = True
            continue
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if quote is not None:
            continue
        if character in "[{(":
            depth += 1
        elif character in "]})":
            depth = max(0, depth - 1)
        elif character == delimiter and depth == 0:
            result.append(value[start:index].strip())
            start = index + 1
    result.append(value[start:].strip())
    return result


def _scalar(value: str) -> str:
    value = _strip_comment(value).strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            return str(json.loads(value))
        except json.JSONDecodeError:
            return value[1:-1]
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def _list(value: str) -> tuple[str, ...]:
    value = _strip_comment(value).strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError("expected an inline YAML list")
    body = value[1:-1].strip()
    if not body:
        return ()
    return tuple(_scalar(item) for item in _split_top_level(body))


def _inline_map(value: str) -> dict[str, str]:
    value = _strip_comment(value).strip()
    if not (value.startswith("{") and value.endswith("}")):
        raise ValueError("expected an inline YAML mapping")
    result = {}
    for item in _split_top_level(value[1:-1]):
        if ":" not in item:
            raise ValueError("inline YAML mapping item has no colon")
        key, raw = item.split(":", 1)
        result[_scalar(key)] = raw.strip()
    return result


def load_arpasing_profile(path: Path | str | None = None) \
        -> ArpasingVoiceProfile:
    source = Path(path) if path is not None else DEFAULT_PHONEME_MAP_PATH
    payload = source.read_bytes()
    text = payload.decode("utf-8-sig")
    source_name = (
        "bundled:profiles/en-jap-mapping.yaml"
        if source.resolve() == DEFAULT_PHONEME_MAP_PATH.resolve()
        else source.name
    )
    symbols: dict[str, str] = {}
    timings: dict[str, float] = {}
    replacements = []
    entries = []
    diagnostics: list[ArpasingProfileDiagnostic] = []
    section = ""
    pending_grapheme = None
    pending_line = None

    for line_number, raw_line in enumerate(text.splitlines(), 1):
        clean = _strip_comment(raw_line).rstrip()
        stripped = clean.strip()
        if not stripped or stripped.startswith("%") or stripped == "---":
            continue
        if not raw_line[:1].isspace() and stripped.endswith(":"):
            section = stripped[:-1].strip()
            continue
        try:
            if section in {"symbols", "timings", "replacements"} \
                    and stripped.startswith("- {"):
                values = _inline_map(stripped[2:].strip())
                if section == "symbols":
                    symbol = _scalar(values.get("symbol", ""))
                    phone_type = _scalar(values.get("type", ""))
                    if not symbol or not phone_type:
                        raise ValueError("symbol row needs symbol and type")
                    if symbol in symbols and symbols[symbol] != phone_type:
                        diagnostics.append(ArpasingProfileDiagnostic(
                            "conflicting_symbol_type",
                            f"{symbol!r} has conflicting symbol types.",
                            line=line_number,
                            details={"first": symbols[symbol], "next": phone_type},
                        ))
                    symbols.setdefault(symbol, phone_type)
                elif section == "timings":
                    symbol = _scalar(values.get("symbol", ""))
                    timings[symbol] = float(_scalar(values.get("value", "")))
                else:
                    replacements.append((
                        _list(values.get("from", "[]")),
                        _list(values.get("to", "[]")),
                    ))
                continue
            if section == "entries" and stripped.startswith("- grapheme:"):
                if pending_grapheme is not None:
                    diagnostics.append(ArpasingProfileDiagnostic(
                        "entry_missing_phonemes",
                        f"Mapping {pending_grapheme!r} has no phoneme list.",
                        line=pending_line,
                    ))
                pending_grapheme = _scalar(stripped.split(":", 1)[1])
                pending_line = line_number
                continue
            if section == "entries" and stripped.startswith("phonemes:"):
                if pending_grapheme is None:
                    raise ValueError("phonemes row has no preceding grapheme")
                entries.append(ArpasingPhonemeMapEntry(
                    pending_grapheme,
                    _list(stripped.split(":", 1)[1]),
                    int(pending_line or line_number),
                ))
                pending_grapheme = None
                pending_line = None
                continue
        except (KeyError, TypeError, ValueError) as error:
            diagnostics.append(ArpasingProfileDiagnostic(
                "profile_row_invalid",
                str(error),
                severity="error",
                line=line_number,
                details={"section": section},
            ))

    if pending_grapheme is not None:
        diagnostics.append(ArpasingProfileDiagnostic(
            "entry_missing_phonemes",
            f"Mapping {pending_grapheme!r} has no phoneme list.",
            line=pending_line,
        ))

    grouped: dict[str, list[ArpasingPhonemeMapEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.grapheme, []).append(entry)
        invalid = [phone for phone in entry.phonemes
                   if not _PHONE.fullmatch(phone)]
        if invalid:
            diagnostics.append(ArpasingProfileDiagnostic(
                "invalid_phone_spelling",
                f"Mapping {entry.grapheme!r} contains invalid phone spelling.",
                line=entry.line,
                details={"phones": invalid},
            ))
    for grapheme, rows in grouped.items():
        variants = tuple(dict.fromkeys(row.phonemes for row in rows))
        if len(variants) > 1:
            diagnostics.append(ArpasingProfileDiagnostic(
                "conflicting_grapheme_mapping",
                f"Mapping {grapheme!r} has {len(variants)} alternatives; "
                "the longest valid sequence wins and all alternatives remain "
                "recorded.",
                line=rows[0].line,
                details={"alternatives": [list(item) for item in variants]},
            ))
    for symbol in timings:
        if symbol not in symbols:
            diagnostics.append(ArpasingProfileDiagnostic(
                "timing_symbol_undeclared",
                f"Timing multiplier refers to undeclared symbol {symbol!r}.",
                severity="info",
            ))

    return ArpasingVoiceProfile(
        source_name=source_name,
        source_sha256=hashlib.sha256(payload).hexdigest().upper(),
        symbols=dict(symbols),
        timings=dict(timings),
        replacements=tuple(replacements),
        entries=tuple(entries),
        diagnostics=tuple(diagnostics),
    )


def map_japanese_mora(
    reading: str,
    canonical_phones: Sequence[str],
    runtime_map: Mapping[str, object],
    *,
    following_phone: Optional[str] = None,
    available_phones: Sequence[str] = (),
) -> tuple[tuple[str, ...], Optional[str]]:
    """Map one canonical Japanese mora into an ARPAsing phone namespace."""
    canonical = tuple(str(item) for item in canonical_phones)
    available = set(str(item) for item in available_phones)
    grapheme_map = dict(runtime_map.get("grapheme_to_phones") or {})
    nasal_routes = dict(runtime_map.get("moraic_nasal_routes") or {})
    fallbacks = dict(runtime_map.get("canonical_fallbacks") or {})

    if canonical == ("N",):
        following = str(following_phone or "")
        if following in {"m", "b", "p", "my", "by", "py"}:
            route = "labial"
        elif following in {"k", "g", "ky", "gy", "ng", "ngy"}:
            route = "velar"
        elif not following or following in {
            "s", "sh", "z", "zh", "j", "jh", "ch", "ts", "dz",
        }:
            route = "uvular"
        else:
            route = "default"
        target = str(
            nasal_routes.get(route)
            or nasal_routes.get("default")
            or fallbacks.get("N")
            or "nn"
        )
        mapped = (target,)
        reason = f"moraic_nasal_{route}"
    else:
        row = grapheme_map.get(str(reading))
        reason = "profile_grapheme_map"
        if not (isinstance(row, (list, tuple)) and row):
            # Open JTalk can occasionally reject its all-kana reading when
            # non-spoken punctuation makes the reading/label counts differ.
            # Its label-derived mora then has a value such as ``ra`` rather
            # than ``ら``. Recover the lookup key from the canonical linguistic
            # phones, but continue to use this voice profile's target phones.
            from japanese_kana_frontend import (
                canonical_mora_reading,
                normalize_kana_reading,
            )

            normalized_reading = normalize_kana_reading(str(reading))
            if normalized_reading != str(reading):
                row = grapheme_map.get(normalized_reading)
                if isinstance(row, (list, tuple)) and row:
                    reason = "profile_normalized_grapheme_map"
            if not (isinstance(row, (list, tuple)) and row):
                canonical_reading = canonical_mora_reading(canonical)
                if canonical_reading:
                    row = grapheme_map.get(canonical_reading)
                    if isinstance(row, (list, tuple)) and row:
                        reason = "profile_canonical_mora_map"
        if isinstance(row, (list, tuple)) and row:
            mapped = tuple(str(item) for item in row)
        else:
            mapped = tuple(str(fallbacks.get(phone) or phone)
                           for phone in canonical)
            reason = "canonical_phone_fallback"

    if available and any(phone not in available for phone in mapped):
        missing = [phone for phone in mapped if phone not in available]
        fallback = tuple(str(fallbacks.get(phone) or phone)
                         for phone in canonical)
        if fallback and all(phone in available for phone in fallback):
            return fallback, "profile_target_missing_used_canonical_fallback"
        return mapped, "profile_target_missing:" + ",".join(missing)
    return mapped, reason
