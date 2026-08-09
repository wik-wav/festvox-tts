"""Language-neutral rendering policy for non-lexical phone symbols.

Frontends keep symbols such as ``cl`` in the canonical utterance because they
own timing and remain editable.  Generated UTAU voices must not, however,
interpret a coincidentally named OTO alias as the acoustic realization of that
symbol.  This module resolves canonical phones to source-selection phones
without changing the linguistic sequence exposed to the user.

The generated-voice realization of ``cl`` anticipates the following
consonant. A voice with a genuine, authored ``cl`` phone can expose it under a
distinct canonical token such as ``cl_literal`` while retaining structural
``cl``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Optional, Sequence


SPECIAL_PHONE_REALIZATION_SCHEMA_VERSION = 1
STRUCTURAL_CL_MODE = "anticipatory_consonant"
LITERAL_MODE = "literal"
LITERAL_ALIAS_MODE = "literal_alias"
DEFAULT_LITERAL_CL_DISPLAY_PHONE = "cl_literal"
PHONE_NAME_RE = re.compile(r"^[A-Za-z0-9_@:~#]+$")

BOUNDARY_PHONES = frozenset({"pau", "sil", "sp", "#", "*"})
VOWEL_PHONES = frozenset({
    # Japanese / Asaxi
    "a", "i", "u", "e", "o",
    # ARPAbet / ARPAsing
    "aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey",
    "ih", "iy", "ow", "oy", "uh", "uw",
})


DEFAULT_GENERATED_VOICE_POLICY = {
    "schema_version": SPECIAL_PHONE_REALIZATION_SCHEMA_VERSION,
    "phones": {
        "cl": {
            "mode": STRUCTURAL_CL_MODE,
            "description": (
                "Keep cl as one editable phone while sourcing its interval "
                "from a held following-consonant transition."
            ),
        },
        "pau": {"mode": LITERAL_MODE},
        "sil": {"mode": LITERAL_MODE},
        "sp": {"mode": LITERAL_MODE},
        "q": {"mode": LITERAL_MODE},
    },
}

SUPPORTED_PHONE_MODES = {
    "cl": frozenset({STRUCTURAL_CL_MODE, LITERAL_MODE}),
    "pau": frozenset({LITERAL_MODE}),
    "sil": frozenset({LITERAL_MODE}),
    "sp": frozenset({LITERAL_MODE}),
    "q": frozenset({LITERAL_MODE}),
}


@dataclass(frozen=True)
class SpecialPhoneRealization:
    """One canonical phone whose source-selection identity was considered."""

    index: int
    phone: str
    mode: str
    source_phone: str
    status: str
    following_phone: str = ""
    required_diphones: tuple[str, ...] = ()
    missing_diphones: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "index": int(self.index),
            "phone": self.phone,
            "mode": self.mode,
            "source_phone": self.source_phone,
            "status": self.status,
            "following_phone": self.following_phone,
            "required_diphones": list(self.required_diphones),
            "missing_diphones": list(self.missing_diphones),
        }


@dataclass(frozen=True)
class SpecialPhoneResolution:
    """Canonical and source-selection views of one phone sequence."""

    display_phones: tuple[str, ...]
    render_phones: tuple[str, ...]
    realizations: tuple[SpecialPhoneRealization, ...]

    @property
    def unresolved(self) -> tuple[SpecialPhoneRealization, ...]:
        return tuple(
            row for row in self.realizations
            if row.status not in {"resolved", "literal"}
        )


def generated_voice_policy(
    overrides: Optional[Mapping[str, str]] = None,
    literal_phone_mappings: Optional[Mapping[str, str]] = None,
) -> dict[str, object]:
    """Return a validated, independent policy for builder metadata.

    Generated voices use structural ``cl`` by default in every language.
    A bank with a genuine linguistic /cl/ may explicitly expose its source
    phone under a distinct canonical token. Structural ``cl`` remains
    available, so both meanings can coexist without an ambiguous symbol.

    ``cl=literal`` is retained as a shorthand for adding the canonical
    ``cl_literal`` -> source ``cl`` mapping; it does not replace structural
    ``cl``.
    """
    policy = {
        "schema_version": SPECIAL_PHONE_REALIZATION_SCHEMA_VERSION,
        "phones": {
            phone: dict(settings)
            for phone, settings in DEFAULT_GENERATED_VOICE_POLICY[
                "phones"
            ].items()
        },
        "literal_phone_mappings": {},
    }
    for raw_phone, raw_mode in dict(overrides or {}).items():
        phone = str(raw_phone or "").strip()
        mode = str(raw_mode or "").strip()
        supported = SUPPORTED_PHONE_MODES.get(phone)
        if supported is None:
            raise ValueError(
                f"unknown special phone {phone!r}; expected one of "
                + ", ".join(sorted(SUPPORTED_PHONE_MODES))
            )
        if mode not in supported:
            raise ValueError(
                f"unsupported realization {phone}={mode!r}; expected "
                + " or ".join(sorted(supported))
            )
        if phone == "cl" and mode == LITERAL_MODE:
            policy["literal_phone_mappings"][
                DEFAULT_LITERAL_CL_DISPLAY_PHONE
            ] = {
                "source_phone": "cl",
                "mode": LITERAL_ALIAS_MODE,
                "description": (
                    "Expose the creator-declared literal /cl/ source while "
                    "retaining structural cl."
                ),
            }
            continue
        settings = {"mode": mode}
        if phone == "cl":
            settings["description"] = (
                "Keep cl as one editable phone while sourcing its interval "
                "from a held following-consonant transition."
            )
        policy["phones"][phone] = settings
    for raw_display, raw_source in dict(
            literal_phone_mappings or {}).items():
        display = str(raw_display or "").strip()
        source = str(raw_source or "").strip()
        if not display or not PHONE_NAME_RE.fullmatch(display):
            raise ValueError(
                f"invalid literal display phone {display!r}"
            )
        if not source or not PHONE_NAME_RE.fullmatch(source):
            raise ValueError(
                f"invalid literal source phone {source!r}"
            )
        if source not in SUPPORTED_PHONE_MODES:
            raise ValueError(
                f"literal source phone {source!r} is not a registered "
                "special phone"
            )
        if display in SUPPORTED_PHONE_MODES or display == source:
            raise ValueError(
                f"literal display phone {display!r} must be distinct from "
                f"the structural source symbol {source!r}"
            )
        existing = policy["literal_phone_mappings"].get(display)
        if existing and existing.get("source_phone") != source:
            raise ValueError(
                f"conflicting literal phone mappings for {display!r}"
            )
        policy["literal_phone_mappings"][display] = {
            "source_phone": source,
            "mode": LITERAL_ALIAS_MODE,
            "description": (
                f"Creator-declared literal {source} source exposed as "
                f"{display} without replacing structural {source}."
            ),
        }
    return policy


def parse_special_phone_mode_specs(
    values: Optional[Iterable[str]],
) -> dict[str, str]:
    """Parse repeatable ``PHONE=MODE`` command-line declarations."""
    result: dict[str, str] = {}
    for raw in values or ():
        text = str(raw or "").strip()
        if "=" not in text:
            raise ValueError(
                f"invalid special-phone mode {text!r}; use PHONE=MODE"
            )
        phone, mode = (part.strip() for part in text.split("=", 1))
        # Validation and actionable supported-mode diagnostics live in one
        # place, so CLI and future profile readers cannot drift.
        generated_voice_policy({phone: mode})
        if phone in result and result[phone] != mode:
            raise ValueError(
                f"conflicting special-phone modes for {phone!r}: "
                f"{result[phone]!r} and {mode!r}"
            )
        result[phone] = mode
    return result


def parse_literal_phone_map_specs(
    values: Optional[Iterable[str]],
) -> dict[str, str]:
    """Parse repeatable ``DISPLAY=SOURCE`` literal-phone declarations."""
    result: dict[str, str] = {}
    for raw in values or ():
        text = str(raw or "").strip()
        if "=" not in text:
            raise ValueError(
                f"invalid literal phone map {text!r}; use DISPLAY=SOURCE"
            )
        display, source = (part.strip() for part in text.split("=", 1))
        generated_voice_policy(
            literal_phone_mappings={display: source}
        )
        if display in result and result[display] != source:
            raise ValueError(
                f"conflicting literal phone mappings for {display!r}: "
                f"{result[display]!r} and {source!r}"
            )
        result[display] = source
    return result


def _looks_like_generated_utau_voice(metadata: Mapping[str, object]) -> bool:
    # Generic names such as ``builder_version`` and ``configuration_id`` are
    # not proof that a third-party Festival voice follows this contract.
    # Structural defaults are inherited only from markers emitted by this
    # converter; every other voice remains literal unless it declares an
    # explicit policy.
    return bool(
        metadata.get("context_model") == "oto_directional_v1"
        or metadata.get("kind") in {
            "festival_unisyn_runtime_index",
            "japanese_festival_runtime_index",
            "generated_festival_voice_manifest",
        }
    )


def _phone_modes(
    metadata: Optional[Mapping[str, object]],
) -> dict[str, str]:
    data = dict(metadata or {})
    raw = data.get("special_phone_realizations")
    modes: dict[str, str] = {}
    if isinstance(raw, Mapping):
        phones = raw.get("phones", raw)
        if isinstance(phones, Mapping):
            for phone, value in phones.items():
                if isinstance(value, Mapping):
                    mode = value.get("mode")
                else:
                    mode = value
                if str(mode or "").strip():
                    modes[str(phone)] = str(mode).strip()
    if modes.get("cl") == LITERAL_MODE and isinstance(raw, Mapping):
        # Compatibility with the first explicit-policy draft: preserve its
        # creator opt-in, but do not make structural cl unavailable.
        modes["cl"] = STRUCTURAL_CL_MODE
    if "cl" not in modes:
        modes["cl"] = (
            STRUCTURAL_CL_MODE
            if _looks_like_generated_utau_voice(data)
            else LITERAL_MODE
        )
    return modes


def _literal_phone_mappings(
    metadata: Optional[Mapping[str, object]],
) -> dict[str, str]:
    data = dict(metadata or {})
    raw_policy = data.get("special_phone_realizations")
    if not isinstance(raw_policy, Mapping):
        return {}
    raw_mappings = raw_policy.get("literal_phone_mappings")
    result: dict[str, str] = {}
    if isinstance(raw_mappings, Mapping):
        for display, value in raw_mappings.items():
            source = (
                value.get("source_phone")
                if isinstance(value, Mapping) else value
            )
            display_name = str(display or "").strip()
            source_name = str(source or "").strip()
            if display_name and source_name:
                result[display_name] = source_name
    raw_phones = raw_policy.get("phones")
    raw_cl = (
        raw_phones.get("cl")
        if isinstance(raw_phones, Mapping) else None
    )
    raw_cl_mode = (
        raw_cl.get("mode") if isinstance(raw_cl, Mapping) else raw_cl
    )
    if str(raw_cl_mode or "") == LITERAL_MODE:
        result.setdefault(DEFAULT_LITERAL_CL_DISPLAY_PHONE, "cl")
    return result


def declared_display_phones(
    source_phones: Iterable[str],
    policy: Optional[Mapping[str, object]],
) -> list[str]:
    """Return source phones plus creator-declared literal display aliases."""
    phones = {str(phone) for phone in source_phones}
    metadata = {"special_phone_realizations": dict(policy or {})}
    phones.update(_literal_phone_mappings(metadata))
    return sorted(phones)


def _is_following_consonant(phone: str) -> bool:
    base = str(phone or "").split("__", 1)[0].rstrip("_").casefold()
    return bool(
        base
        and base not in BOUNDARY_PHONES
        and base not in VOWEL_PHONES
        and base != "cl"
    )


def _base_diphones(values: Optional[Iterable[str]]) -> Optional[set[str]]:
    if values is None:
        return None
    return {str(value) for value in values}


def resolve_special_phone_sequence(
    phones: Sequence[str],
    *,
    metadata: Optional[Mapping[str, object]] = None,
    available_diphones: Optional[Iterable[str]] = None,
    allow_unverified_inventory: bool = False,
) -> SpecialPhoneResolution:
    """Resolve structural phones while preserving one-to-one timing indexes.

    ``available_diphones`` is optional for authored Festival voices whose
    inventory is not exposed to Python.  Generated voices pass their runtime
    index, allowing an old bank that lacks a consonant hold unit to fail
    visibly instead of silently selecting a literal ``cl`` OTO alias.
    """
    display = tuple(str(phone).strip() for phone in phones)
    render = list(display)
    modes = _phone_modes(metadata)
    literal_mappings = _literal_phone_mappings(metadata)
    inventory = _base_diphones(available_diphones)
    records: list[SpecialPhoneRealization] = []

    for index, phone in enumerate(display):
        source = literal_mappings.get(phone)
        if source:
            render[index] = source
            records.append(SpecialPhoneRealization(
                index=index,
                phone=phone,
                mode=LITERAL_ALIAS_MODE,
                source_phone=source,
                status="literal",
            ))

    for index, phone in enumerate(display):
        if phone in literal_mappings:
            continue
        mode = modes.get(phone, LITERAL_MODE)
        if phone != "cl" or mode == LITERAL_MODE:
            # Ordinary literal boundary phones do not need a per-occurrence
            # record. Keep a record for cl itself so an authored literal /cl/
            # remains inspectable, but avoid filling every project with
            # no-op pau/sil/sp/q provenance rows.
            if phone == "cl" and phone in modes:
                records.append(SpecialPhoneRealization(
                    index=index,
                    phone=phone,
                    mode=mode,
                    source_phone=phone,
                    status="literal",
                ))
            continue
        if mode != STRUCTURAL_CL_MODE:
            records.append(SpecialPhoneRealization(
                index=index,
                phone=phone,
                mode=mode,
                source_phone=phone,
                status="unsupported_mode",
            ))
            continue

        # Literal display aliases were resolved in the first pass, so a
        # structural closure anticipates the actual source phone even when
        # the following canonical token has a different name.
        following = render[index + 1] if index + 1 < len(render) else ""
        if not _is_following_consonant(following):
            # An orphan closure has no articulatory target.  Preserve its
            # editable interval but source silence rather than a coincidental
            # literal OTO alias.
            render[index] = "pau"
            records.append(SpecialPhoneRealization(
                index=index,
                phone=phone,
                mode=mode,
                source_phone="pau",
                status="resolved_silence_fallback",
                following_phone=following,
            ))
            continue

        previous = render[index - 1] if index else "pau"
        required = (
            f"{previous}-{following}",
            f"{following}-{following}",
        )
        if inventory is None and not allow_unverified_inventory:
            records.append(SpecialPhoneRealization(
                index=index,
                phone=phone,
                mode=mode,
                source_phone=following,
                status="inventory_unavailable",
                following_phone=following,
                required_diphones=required,
                missing_diphones=required,
            ))
            continue
        missing = (
            tuple(item for item in required if item not in inventory)
            if inventory is not None else ()
        )
        if missing:
            records.append(SpecialPhoneRealization(
                index=index,
                phone=phone,
                mode=mode,
                source_phone=following,
                status="missing_source_diphones",
                following_phone=following,
                required_diphones=required,
                missing_diphones=missing,
            ))
            continue

        render[index] = following
        records.append(SpecialPhoneRealization(
            index=index,
            phone=phone,
            mode=mode,
            source_phone=following,
            status="resolved",
            following_phone=following,
            required_diphones=required,
        ))

    return SpecialPhoneResolution(
        display_phones=display,
        render_phones=tuple(render),
        realizations=tuple(records),
    )


def source_pair(
    render_phones: Sequence[str],
    boundary_index: int,
) -> str:
    """Return the actual source-selection pair for a displayed boundary."""
    index = int(boundary_index)
    if index < 0 or index + 1 >= len(render_phones):
        return ""
    return f"{render_phones[index]}-{render_phones[index + 1]}"
