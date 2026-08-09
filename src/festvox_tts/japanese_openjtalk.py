"""Optional pyopenjtalk adapter and Open JTalk full-context parser.

Phase 1 uses Open JTalk only for linguistic analysis.  It does not request or
consume synthesized audio, HTS durations, or an F0 trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import importlib
import importlib.util
from importlib import metadata
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from japanese_kana_frontend import (
    KanaMoraSpec,
    normalize_kana_reading,
    segment_kana_reading,
)
from japanese_models import (
    JapaneseAccentPhrase,
    JapaneseFrontendDiagnostic,
    JapaneseMora,
    JapanesePhone,
    JapanesePhrase,
    JapaneseUtterance,
)


_PLACEHOLDERS = {"", "x", "xx"}
_VOWELS = {"a", "i", "u", "e", "o"}
_KNOWN_CONSONANTS = {
    "b", "by", "ch", "d", "dy", "f", "g", "gy", "h", "hy",
    "j", "k", "ky", "m", "my", "n", "ny", "p", "py", "r",
    "ry", "s", "sh", "t", "ts", "v", "w", "y", "z",
}
_SOURCE_BOUNDARIES = {
    "、": 1, ",": 1, "，": 1, "・": 1,
    ":": 2, "：": 2, ";": 2, "；": 2,
    "▽": 2,
    "。": 3, ".": 3, "．": 3,
    "!": 3, "！": 3, "?": 3, "？": 3,
}
_INLINE_BRACKETS = {
    "「", "」", "『", "』", "【", "】", "〈", "〉", "《", "》",
    "（", "）", "(", ")", "［", "］", "[", "]", "“", "”", '"',
}


@dataclass(frozen=True)
class QuinphoneContext:
    two_before: Optional[str]
    previous: Optional[str]
    current: Optional[str]
    following: Optional[str]
    two_after: Optional[str]
    raw: str


@dataclass(frozen=True)
class MoraPositionContext:
    """Open JTalk A fields; forward/backward positions are one-based."""

    relative_to_accent_nucleus: Optional[int]
    position_forward: Optional[int]
    position_backward: Optional[int]
    raw: str


@dataclass(frozen=True)
class AccentPhraseLabelContext:
    """Open JTalk F fields for the current accent phrase.

    `accent_nucleus` is the label's one-based accent type.  Zero means
    unaccented.  It is converted to zero-based only in JapaneseAccentPhrase.
    """

    mora_count: Optional[int]
    accent_nucleus: Optional[int]
    interrogative: Optional[bool]
    emotion: Optional[str]
    position_forward_in_breath_group: Optional[int]
    position_backward_in_breath_group: Optional[int]
    mora_start_forward_in_breath_group: Optional[int]
    mora_end_backward_in_breath_group: Optional[int]
    raw: str


@dataclass(frozen=True)
class BreathGroupLabelContext:
    """Named Open JTalk I fields retained without using them for F0."""

    accent_phrase_count: Optional[int]
    mora_count: Optional[int]
    position_forward_in_utterance: Optional[int]
    position_backward_in_utterance: Optional[int]
    accent_phrase_start_forward: Optional[int]
    accent_phrase_end_backward: Optional[int]
    mora_start_forward: Optional[int]
    mora_end_backward: Optional[int]
    raw: str


@dataclass(frozen=True)
class UtteranceLabelContext:
    breath_group_count: Optional[int]
    accent_phrase_count: Optional[int]
    mora_count: Optional[int]
    raw: str


@dataclass(frozen=True)
class ParsedOpenJTalkLabel:
    raw: str
    quinphone: QuinphoneContext
    mora: Optional[MoraPositionContext]
    accent_phrase: Optional[AccentPhraseLabelContext]
    breath_group: Optional[BreathGroupLabelContext]
    utterance: Optional[UtteranceLabelContext]
    raw_groups: dict[str, str]
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class CanonicalPhoneIdentity:
    symbol: str
    phone_type: str
    devoiced: Optional[bool] = None
    unknown: bool = False


@dataclass
class _MoraBucket:
    index: int
    phrase_index: int
    accent_phrase_index: int
    position_in_accent_phrase: int
    phones: list[JapanesePhone] = field(default_factory=list)
    labels: list[ParsedOpenJTalkLabel] = field(default_factory=list)
    label_indices: list[int] = field(default_factory=list)


@dataclass
class _AccentBucket:
    index: int
    phrase_index: int
    local_position: int
    moras: list[_MoraBucket] = field(default_factory=list)
    context: Optional[AccentPhraseLabelContext] = None


@dataclass(frozen=True)
class _SourcePhrase:
    surface: str
    punctuation: str
    boundary_strength: int
    interrogative: bool


class OpenJTalkFrontendError(RuntimeError):
    def __init__(self, diagnostic: JapaneseFrontendDiagnostic):
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class OpenJTalkUnavailableError(OpenJTalkFrontendError):
    pass


class OpenJTalkAnalysisError(OpenJTalkFrontendError):
    pass


def pyopenjtalk_status() -> tuple[bool, str]:
    """Return whether pyopenjtalk and its runtime dictionary are usable."""
    if importlib.util.find_spec("pyopenjtalk") is None:
        return False, "pyopenjtalk is not installed"
    try:
        module = importlib.import_module("pyopenjtalk")
    except (ImportError, OSError) as error:
        return False, f"pyopenjtalk could not be imported: {error}"
    if not callable(getattr(module, "g2p", None)) or not callable(
            getattr(module, "extract_fullcontext", None)):
        return False, "pyopenjtalk lacks the required analysis functions"
    dictionary = getattr(module, "OPEN_JTALK_DICT_DIR", None)
    if dictionary and not Path(os.fsdecode(dictionary)).is_dir():
        return False, "the Open JTalk MeCab dictionary directory is missing"
    return True, ""


def is_pyopenjtalk_available() -> bool:
    return pyopenjtalk_status()[0]


def canonicalize_openjtalk_phone(raw_symbol: Optional[str]) \
        -> CanonicalPhoneIdentity:
    raw = raw_symbol or ""
    if raw == "I":
        return CanonicalPhoneIdentity("i", "vowel", devoiced=True)
    if raw == "U":
        return CanonicalPhoneIdentity("u", "vowel", devoiced=True)
    if raw in _VOWELS:
        return CanonicalPhoneIdentity(raw, "vowel", devoiced=False)
    if raw in _KNOWN_CONSONANTS:
        return CanonicalPhoneIdentity(raw, "consonant")
    if raw == "N":
        return CanonicalPhoneIdentity("N", "special")
    if raw == "cl":
        return CanonicalPhoneIdentity("cl", "special")
    if raw == "pau":
        return CanonicalPhoneIdentity("pau", "special")
    if raw == "sil":
        return CanonicalPhoneIdentity("sil", "special")
    if raw == "br":
        return CanonicalPhoneIdentity("br", "special")
    return CanonicalPhoneIdentity(raw or "unk", "unknown", unknown=True)


def _clean_phone(token: str) -> Optional[str]:
    return None if token.casefold() in _PLACEHOLDERS else token


def _split_once(
    value: str,
    delimiter: str,
    field_name: str,
    issues: list[str],
) -> tuple[str, str]:
    if delimiter not in value:
        issues.append(f"{field_name} is missing {delimiter!r}")
        return value, ""
    return value.split(delimiter, 1)


def _optional_int(
    token: str,
    field_name: str,
    issues: list[str],
) -> Optional[int]:
    if token.casefold() in _PLACEHOLDERS:
        return None
    try:
        return int(token)
    except ValueError:
        issues.append(f"{field_name} is not an integer: {token!r}")
        return None


def _optional_text(token: str) -> Optional[str]:
    return None if token.casefold() in _PLACEHOLDERS else token


def _pair(
    value: str,
    delimiter: str,
    field_name: str,
    issues: list[str],
) -> tuple[Optional[int], Optional[int]]:
    left, right = _split_once(value, delimiter, field_name, issues)
    return (
        _optional_int(left, f"{field_name}.first", issues),
        _optional_int(right, f"{field_name}.second", issues),
    )


def _parse_quinphone(raw: str, issues: list[str]) -> QuinphoneContext:
    two_before, remainder = _split_once(raw, "^", "quinphone", issues)
    previous, remainder = _split_once(remainder, "-", "quinphone", issues)
    current, remainder = _split_once(remainder, "+", "quinphone", issues)
    following, two_after = _split_once(remainder, "=", "quinphone", issues)
    return QuinphoneContext(
        two_before=_clean_phone(two_before),
        previous=_clean_phone(previous),
        current=_clean_phone(current),
        following=_clean_phone(following),
        two_after=_clean_phone(two_after),
        raw=raw,
    )


def _parse_mora_context(
    raw: Optional[str], issues: list[str]
) -> Optional[MoraPositionContext]:
    if raw is None:
        return None
    parts = raw.split("+")
    if len(parts) != 3:
        issues.append("A must contain three + separated fields")
        parts = (parts + [""] * 3)[:3]
    return MoraPositionContext(
        relative_to_accent_nucleus=_optional_int(
            parts[0], "A.relative_to_accent_nucleus", issues
        ),
        position_forward=_optional_int(parts[1], "A.position_forward", issues),
        position_backward=_optional_int(parts[2], "A.position_backward", issues),
        raw=raw,
    )


def _parse_accent_context(
    raw: Optional[str], issues: list[str]
) -> Optional[AccentPhraseLabelContext]:
    if raw is None:
        return None
    mora_accent, remainder = _split_once(raw, "#", "F", issues)
    question_emotion, remainder = _split_once(remainder, "@", "F", issues)
    phrase_positions, mora_span = _split_once(remainder, "|", "F", issues)
    mora_count, accent_nucleus = _pair(
        mora_accent, "_", "F.mora_and_accent", issues
    )
    question_token, emotion_token = _split_once(
        question_emotion, "_", "F.question_and_emotion", issues
    )
    question_value = _optional_int(question_token, "F.interrogative", issues)
    interrogative: Optional[bool]
    if question_value is None:
        interrogative = None
    elif question_value in {0, 1}:
        interrogative = bool(question_value)
    else:
        issues.append(
            f"F.interrogative must be 0 or 1, got {question_value}"
        )
        interrogative = None
    phrase_forward, phrase_backward = _pair(
        phrase_positions, "_", "F.phrase_positions", issues
    )
    mora_start, mora_end = _pair(mora_span, "_", "F.mora_span", issues)
    return AccentPhraseLabelContext(
        mora_count=mora_count,
        accent_nucleus=accent_nucleus,
        interrogative=interrogative,
        emotion=_optional_text(emotion_token),
        position_forward_in_breath_group=phrase_forward,
        position_backward_in_breath_group=phrase_backward,
        mora_start_forward_in_breath_group=mora_start,
        mora_end_backward_in_breath_group=mora_end,
        raw=raw,
    )


def _parse_breath_group_context(
    raw: Optional[str], issues: list[str]
) -> Optional[BreathGroupLabelContext]:
    if raw is None:
        return None
    counts, remainder = _split_once(raw, "@", "I", issues)
    utterance_positions, remainder = _split_once(remainder, "&", "I", issues)
    accent_span, mora_span = _split_once(remainder, "|", "I", issues)
    accent_count, mora_count = _pair(counts, "-", "I.counts", issues)
    forward, backward = _pair(
        utterance_positions, "+", "I.utterance_positions", issues
    )
    accent_start, accent_end = _pair(
        accent_span, "-", "I.accent_phrase_span", issues
    )
    mora_start, mora_end = _pair(mora_span, "+", "I.mora_span", issues)
    return BreathGroupLabelContext(
        accent_phrase_count=accent_count,
        mora_count=mora_count,
        position_forward_in_utterance=forward,
        position_backward_in_utterance=backward,
        accent_phrase_start_forward=accent_start,
        accent_phrase_end_backward=accent_end,
        mora_start_forward=mora_start,
        mora_end_backward=mora_end,
        raw=raw,
    )


def _parse_utterance_context(
    raw: Optional[str], issues: list[str]
) -> Optional[UtteranceLabelContext]:
    if raw is None:
        return None
    breath_groups, remainder = _split_once(raw, "+", "K", issues)
    accent_phrases, moras = _split_once(remainder, "-", "K", issues)
    return UtteranceLabelContext(
        breath_group_count=_optional_int(
            breath_groups, "K.breath_group_count", issues
        ),
        accent_phrase_count=_optional_int(
            accent_phrases, "K.accent_phrase_count", issues
        ),
        mora_count=_optional_int(moras, "K.mora_count", issues),
        raw=raw,
    )


def parse_full_context_label(raw_label: str) -> ParsedOpenJTalkLabel:
    """Parse one label with named stages and preserve every raw group."""
    issues: list[str] = []
    sections = raw_label.strip().split("/")
    quinphone = _parse_quinphone(sections[0] if sections else "", issues)
    groups: dict[str, str] = {}
    for section in sections[1:]:
        if ":" not in section:
            issues.append(f"context group lacks ':': {section!r}")
            continue
        name, value = section.split(":", 1)
        if name in groups:
            issues.append(f"duplicate context group {name!r}")
        groups[name] = value
    return ParsedOpenJTalkLabel(
        raw=raw_label,
        quinphone=quinphone,
        mora=_parse_mora_context(groups.get("A"), issues),
        accent_phrase=_parse_accent_context(groups.get("F"), issues),
        breath_group=_parse_breath_group_context(groups.get("I"), issues),
        utterance=_parse_utterance_context(groups.get("K"), issues),
        raw_groups=groups,
        issues=tuple(issues),
    )


def _split_source_phrases(text: str) -> list[_SourcePhrase]:
    phrases: list[_SourcePhrase] = []
    buffer: list[str] = []
    index = 0
    while index < len(text):
        if text[index:index + 5].casefold() == "[pau]":
            surface = "".join(buffer).strip()
            if surface:
                phrases.append(_SourcePhrase(surface, "[pau]", 2, False))
            buffer = []
            index += 5
            continue
        character = text[index]
        if character in {".", "．"} \
                and index > 0 and index + 1 < len(text) \
                and text[index - 1].isdigit() and text[index + 1].isdigit():
            # Open JTalk pronounces a decimal point as テン; it is part of the
            # number and must not be mistaken for sentence punctuation.
            buffer.append(character)
            index += 1
            continue
        if character in _SOURCE_BOUNDARIES:
            surface = "".join(buffer).strip()
            if surface:
                phrases.append(_SourcePhrase(
                    surface,
                    character,
                    _SOURCE_BOUNDARIES[character],
                    character in {"?", "？"},
                ))
            elif phrases:
                previous = phrases[-1]
                phrases[-1] = replace(
                    previous,
                    punctuation=previous.punctuation + character,
                    boundary_strength=max(
                        previous.boundary_strength,
                        _SOURCE_BOUNDARIES[character],
                    ),
                    interrogative=(
                        previous.interrogative or character in {"?", "？"}
                    ),
                )
            buffer = []
        else:
            buffer.append(character)
        index += 1
    surface = "".join(buffer).strip()
    if surface:
        phrases.append(_SourcePhrase(surface, "", 3, False))
    return phrases


def _source_pause_marks_by_mora(
    normalized_reading: str,
    morphology_nodes: Sequence[Mapping[str, Any]] | None,
) -> dict[int, tuple[str, ...]]:
    """Return punctuation attached to each boundary between spoken morae.

    Open JTalk emits the same ``pau`` phone for commas and for non-spoken
    quotation/bracket marks.  The latter must not automatically become a
    rendered phrase gap.  NJD morphology gives the most reliable zero-mora
    punctuation positions; the normalized reading is a deterministic fallback
    for static label fixtures and pyopenjtalk variants without ``run_frontend``.
    """
    marks: dict[int, list[str]] = {}

    def add(position: int, value: str) -> None:
        bucket = marks.setdefault(max(0, position), [])
        if value not in bucket:
            bucket.append(value)

    mora_cursor = 0
    for raw_node in morphology_nodes or ():
        node = dict(raw_node)
        try:
            count = max(0, int(node.get("mora_size") or 0))
        except (TypeError, ValueError):
            count = 0
        if count:
            mora_cursor += count
            continue
        surface = str(node.get("string") or node.get("orig") or "")
        for character in surface:
            if character in _SOURCE_BOUNDARIES or character in _INLINE_BRACKETS:
                add(mora_cursor, character)

    reading = normalize_kana_reading(normalized_reading)
    chunk_start = 0
    reading_cursor = 0
    for index, character in enumerate(reading):
        if character not in _SOURCE_BOUNDARIES \
                and character not in _INLINE_BRACKETS:
            continue
        chunk, _ = segment_kana_reading(reading[chunk_start:index])
        reading_cursor += len(chunk)
        add(reading_cursor, character)
        chunk_start = index + 1

    return {
        position: tuple(values)
        for position, values in marks.items()
    }


def _is_bracket_only_pause(
    mora_position: int,
    pause_marks: Mapping[int, Sequence[str]],
) -> bool:
    marks = tuple(pause_marks.get(mora_position, ()))
    return bool(marks) and all(mark in _INLINE_BRACKETS for mark in marks)


def _grammatical_role(node: Mapping[str, Any]) -> str:
    """Return a stable coarse role while retaining all original NJD fields."""
    part_of_speech = str(node.get("pos") or "")
    surface = str(node.get("string") or node.get("orig") or "")
    conjugation = str(node.get("ctype") or "")
    if part_of_speech == "助詞":
        return "particle"
    if part_of_speech == "助動詞":
        if "デス" in conjugation or surface == "です":
            return "polite_copula"
        if "マス" in conjugation or surface == "ます":
            return "polite_auxiliary"
        if surface in {"ない", "ぬ", "ん"}:
            return "negative_auxiliary"
        return "auxiliary"
    if part_of_speech in {"接続詞", "連体詞"}:
        return "function_word"
    if part_of_speech in {
            "名詞", "動詞", "形容詞", "副詞", "感動詞", "フィラー"}:
        return "content_word"
    return "other"


def _morphology_by_mora(
    nodes: Sequence[Mapping[str, Any]] | None,
    mora_count: int,
) -> tuple[dict[int, dict[str, object]], tuple[str, ...]]:
    """Map NJD nodes onto the label-derived mora sequence by mora counts."""
    if not nodes:
        return {}, ()
    result: dict[int, dict[str, object]] = {}
    cursor = 0
    diagnostics = []
    retained_fields = (
        "string", "orig", "read", "pron", "pos", "pos_group1",
        "pos_group2", "pos_group3", "ctype", "cform", "chain_rule",
        "chain_flag", "acc", "mora_size",
    )
    for node_index, raw_node in enumerate(nodes):
        node = dict(raw_node)
        try:
            count = max(0, int(node.get("mora_size") or 0))
        except (TypeError, ValueError):
            count = 0
        if not count:
            continue
        if cursor + count > mora_count:
            diagnostics.append(
                "Open JTalk morphology exceeds the label-derived mora count"
            )
            break
        retained = {
            key: node.get(key) for key in retained_fields if key in node
        }
        role = _grammatical_role(node)
        for local_index in range(count):
            result[cursor + local_index] = {
                **retained,
                "node_index": node_index,
                "mora_position_in_node_zero_based": local_index,
                "mora_count_in_node": count,
                "grammatical_role": role,
                "function_word": role in {
                    "particle", "polite_copula", "polite_auxiliary",
                    "negative_auxiliary", "auxiliary", "function_word",
                },
            }
        cursor += count
    if cursor != mora_count:
        diagnostics.append(
            "Open JTalk morphology covers "
            f"{cursor} of {mora_count} label-derived morae"
        )
    return result, tuple(diagnostics)


def _derived_mora_spec(
    bucket: _MoraBucket,
    previous_vowel: Optional[str],
) -> KanaMoraSpec:
    symbols = tuple(phone.symbol for phone in bucket.phones)
    if symbols == ("N",):
        return KanaMoraSpec("ん", "ん", None, None, symbols, "moraic_nasal")
    if symbols == ("cl",):
        return KanaMoraSpec("っ", "っ", None, None, symbols, "geminate")
    vowel = next((symbol for symbol in reversed(symbols) if symbol in _VOWELS), None)
    consonants = tuple(symbol for symbol in symbols if symbol not in _VOWELS)
    special = None
    if vowel is not None and not consonants and vowel == previous_vowel:
        special = "long_vowel"
    reading = "".join(symbols) or "�"
    return KanaMoraSpec(
        surface=reading,
        reading=reading,
        consonant="".join(consonants) or None,
        vowel=vowel,
        phones=symbols or ("unk",),
        special_mora=special,
        unknown=any(phone.unknown for phone in bucket.phones),
        confidence=0.5,
        provenance={"derived_from_openjtalk_phones": True},
    )


def parse_openjtalk_labels(
    labels: Sequence[str],
    *,
    source_text: str,
    normalized_reading: str = "",
    frontend_name: str = "openjtalk",
    frontend_version: Optional[str] = None,
    morphology_nodes: Sequence[Mapping[str, Any]] | None = None,
) -> JapaneseUtterance:
    parsed = [parse_full_context_label(label) for label in labels]
    diagnostics: list[JapaneseFrontendDiagnostic] = []
    pause_marks = _source_pause_marks_by_mora(
        normalized_reading, morphology_nodes)
    inline_pause_labels: list[int] = []
    for label_index, item in enumerate(parsed):
        for issue in item.issues:
            diagnostics.append(JapaneseFrontendDiagnostic(
                code="openjtalk_label_parse_issue",
                message=f"Open JTalk label {label_index}: {issue}.",
                severity="warning",
                action=(
                    "Keep the raw label and verify this field against the "
                    "installed Open JTalk version."
                ),
                frontend=frontend_name,
                confidence=0.4,
                raw_data={
                    "label_index": label_index,
                    "raw_label": item.raw,
                    "issue": issue,
                },
            ))

    phones: list[JapanesePhone] = []
    accent_buckets: list[_AccentBucket] = []
    mora_buckets: list[_MoraBucket] = []
    phrase_boundaries: dict[int, int] = {}
    phrase_has_speech: set[int] = set()
    current_phrase = 0
    current_accent_local: Optional[int] = None
    current_accent: Optional[_AccentBucket] = None
    current_mora_key: Optional[tuple[int, int, int]] = None
    current_mora: Optional[_MoraBucket] = None
    last_mora_position: Optional[int] = None
    after_final_sil = False

    for label_index, item in enumerate(parsed):
        raw_phone = item.quinphone.current
        identity = canonicalize_openjtalk_phone(raw_phone)
        if identity.unknown:
            diagnostics.append(JapaneseFrontendDiagnostic(
                code="unknown_openjtalk_phone",
                message=f"Unknown Open JTalk phone {raw_phone!r} was preserved.",
                severity="warning",
                action="Add an explicit canonical-phone mapping if it is valid.",
                frontend=frontend_name,
                confidence=0.0,
                raw_data={
                    "label_index": label_index,
                    "raw_phone": raw_phone,
                    "raw_label": item.raw,
                },
            ))

        if identity.symbol == "sil":
            owner = current_phrase if current_phrase in phrase_has_speech else None
            phones.append(JapanesePhone(
                index=len(phones),
                symbol="sil",
                raw_symbol=raw_phone,
                phone_type="special",
                phrase_index=owner,
                is_silence=True,
                raw_label=item.raw,
            ))
            if owner is not None:
                phrase_boundaries[owner] = 3
                after_final_sil = True
            continue

        if identity.symbol == "pau":
            owner = current_phrase if current_phrase in phrase_has_speech else None
            inline_bracket_pause = (
                owner is not None
                and _is_bracket_only_pause(len(mora_buckets), pause_marks)
            )
            phones.append(JapanesePhone(
                index=len(phones),
                symbol="pau",
                raw_symbol=raw_phone,
                phone_type="special",
                phrase_index=owner,
                is_pause=True,
                raw_label=item.raw,
            ))
            if inline_bracket_pause:
                inline_pause_labels.append(label_index)
                diagnostics.append(JapaneseFrontendDiagnostic(
                    code="openjtalk_inline_bracket_pause",
                    message=(
                        "Open JTalk inserted a pause at a non-spoken "
                        "quotation or bracket boundary; it was retained in "
                        "provenance but not promoted to a rendered phrase gap."
                    ),
                    severity="info",
                    action=(
                        "Add explicit comma or sentence punctuation when an "
                        "audible phrase pause is intended."
                    ),
                    frontend=frontend_name,
                    confidence=0.95,
                    raw_data={
                        "label_index": label_index,
                        "mora_boundary": len(mora_buckets),
                        "source_marks": list(
                            pause_marks.get(len(mora_buckets), ())
                        ),
                        "raw_label": item.raw,
                    },
                ))
            elif owner is not None:
                phrase_boundaries[owner] = 2
                current_phrase += 1
            current_accent_local = None
            current_accent = None
            current_mora_key = None
            current_mora = None
            last_mora_position = None
            after_final_sil = False
            continue

        if after_final_sil:
            current_phrase += 1
            current_accent_local = None
            current_accent = None
            current_mora_key = None
            current_mora = None
            last_mora_position = None
            after_final_sil = False

        phrase_has_speech.add(current_phrase)
        label_mora_position = (
            item.mora.position_forward if item.mora is not None else None
        )
        label_accent_position = (
            item.accent_phrase.position_forward_in_breath_group
            if item.accent_phrase is not None else None
        )
        if label_accent_position is None:
            if current_accent_local is None:
                label_accent_position = 1
            elif label_mora_position == 1 and last_mora_position not in {None, 1}:
                label_accent_position = current_accent_local + 1
            else:
                label_accent_position = current_accent_local

        if label_accent_position != current_accent_local:
            current_accent = _AccentBucket(
                index=len(accent_buckets),
                phrase_index=current_phrase,
                local_position=label_accent_position,
                context=item.accent_phrase,
            )
            accent_buckets.append(current_accent)
            current_accent_local = label_accent_position
            current_mora_key = None
            current_mora = None
        elif current_accent is not None and current_accent.context is None:
            current_accent.context = item.accent_phrase

        if label_mora_position is None:
            if current_mora is None:
                label_mora_position = 1
            elif identity.phone_type in {"vowel", "special"}:
                label_mora_position = current_mora.position_in_accent_phrase
            else:
                label_mora_position = current_mora.position_in_accent_phrase + 1

        mora_key = (
            current_phrase, current_accent_local, label_mora_position
        )
        if mora_key != current_mora_key:
            if current_accent is None:
                current_accent = _AccentBucket(
                    index=len(accent_buckets),
                    phrase_index=current_phrase,
                    local_position=current_accent_local or 1,
                    context=item.accent_phrase,
                )
                accent_buckets.append(current_accent)
            current_mora = _MoraBucket(
                index=len(mora_buckets),
                phrase_index=current_phrase,
                accent_phrase_index=current_accent.index,
                position_in_accent_phrase=label_mora_position,
            )
            mora_buckets.append(current_mora)
            current_accent.moras.append(current_mora)
            current_mora_key = mora_key

        phone = JapanesePhone(
            index=len(phones),
            symbol=identity.symbol,
            raw_symbol=raw_phone,
            phone_type=identity.phone_type,
            phrase_index=current_phrase,
            accent_phrase_index=current_accent.index,
            mora_index=current_mora.index,
            devoiced=identity.devoiced,
            unknown=identity.unknown,
            raw_label=item.raw,
            confidence=0.0 if identity.unknown else 1.0,
        )
        phones.append(phone)
        current_mora.phones.append(phone)
        current_mora.labels.append(item)
        current_mora.label_indices.append(label_index)
        last_mora_position = label_mora_position

    if phrase_has_speech:
        phrase_boundaries.setdefault(max(phrase_has_speech), 3)

    reading = normalize_kana_reading(normalized_reading)
    reading_specs: tuple[KanaMoraSpec, ...] = ()
    if reading:
        reading_specs, reading_diagnostics = segment_kana_reading(reading)
        diagnostics.extend(
            replace(item, frontend=frontend_name)
            for item in reading_diagnostics
        )
        if len(reading_specs) != len(mora_buckets):
            diagnostics.append(JapaneseFrontendDiagnostic(
                code="openjtalk_reading_mora_mismatch",
                message=(
                    f"Open JTalk reading has {len(reading_specs)} morae but "
                    f"the labels describe {len(mora_buckets)}."
                ),
                severity="warning",
                action=(
                    "Keep label phones authoritative and inspect the raw labels."
                ),
                frontend=frontend_name,
                confidence=0.5,
                raw_data={"normalized_reading": reading},
            ))
            reading_specs = ()

    morphology, morphology_issues = _morphology_by_mora(
        morphology_nodes, len(mora_buckets))
    for issue in morphology_issues:
        diagnostics.append(JapaneseFrontendDiagnostic(
            code="openjtalk_morphology_mora_mismatch",
            message=issue + ".",
            severity="warning",
            action=(
                "Keep label-derived morae authoritative and inspect the "
                "preserved morphology fields."
            ),
            frontend=frontend_name,
            confidence=0.5,
        ))

    moras: list[JapaneseMora] = []
    previous_vowel: Optional[str] = None
    for bucket in mora_buckets:
        spec = (
            reading_specs[bucket.index]
            if reading_specs
            else _derived_mora_spec(bucket, previous_vowel)
        )
        devoiced_values = {
            phone.devoiced for phone in bucket.phones
            if phone.devoiced is not None
        }
        devoiced = True if True in devoiced_values else (
            False if False in devoiced_values else None
        )
        mora = JapaneseMora(
            index=bucket.index,
            phrase_index=bucket.phrase_index,
            accent_phrase_index=bucket.accent_phrase_index,
            surface=spec.surface,
            reading=spec.reading,
            phones=tuple(bucket.phones),
            consonant=spec.consonant,
            vowel=spec.vowel,
            special_mora=spec.special_mora,
            devoiced=devoiced,
            confidence=min(
                [spec.confidence] + [phone.confidence for phone in bucket.phones]
            ),
            provenance={
                **spec.provenance,
                "openjtalk_mora_position_one_based": (
                    bucket.position_in_accent_phrase
                ),
                "raw_label_indices": list(bucket.label_indices),
                "morphology": morphology.get(bucket.index),
            },
        )
        moras.append(mora)
        previous_vowel = mora.vowel

    source_phrases = _split_source_phrases(source_text)
    phrase_objects: list[JapanesePhrase] = []
    accent_objects: dict[int, JapaneseAccentPhrase] = {}
    for accent_bucket in accent_buckets:
        accent_moras = tuple(moras[item.index] for item in accent_bucket.moras)
        context = accent_bucket.context
        label_nucleus = context.accent_nucleus if context is not None else None
        accent_state = "unknown"
        nucleus = None
        if label_nucleus == 0:
            accent_state = "unaccented"
        elif label_nucleus is not None and 1 <= label_nucleus <= len(accent_moras):
            accent_state = "accented"
            nucleus = label_nucleus - 1
        elif label_nucleus is not None:
            diagnostics.append(JapaneseFrontendDiagnostic(
                code="openjtalk_accent_nucleus_out_of_range",
                message=(
                    f"Accent nucleus {label_nucleus} is outside a "
                    f"{len(accent_moras)}-mora accent phrase."
                ),
                severity="warning",
                action="Inspect the raw F context before applying an accent.",
                frontend=frontend_name,
                confidence=0.2,
                raw_data={"raw_F": context.raw if context else None},
            ))
        if context is not None and context.mora_count is not None \
                and context.mora_count != len(accent_moras):
            diagnostics.append(JapaneseFrontendDiagnostic(
                code="openjtalk_accent_phrase_length_mismatch",
                message=(
                    f"F reports {context.mora_count} morae but the grouped "
                    f"accent phrase contains {len(accent_moras)}."
                ),
                severity="warning",
                action="Inspect A and F fields in the preserved raw labels.",
                frontend=frontend_name,
                confidence=0.4,
                raw_data={"raw_F": context.raw},
            ))
        same_phrase = [
            item for item in accent_buckets
            if item.phrase_index == accent_bucket.phrase_index
        ]
        is_last = same_phrase[-1].index == accent_bucket.index
        boundary = (
            phrase_boundaries.get(accent_bucket.phrase_index, 3)
            if is_last else 1
        )
        accent_objects[accent_bucket.index] = JapaneseAccentPhrase(
            index=accent_bucket.index,
            phrase_index=accent_bucket.phrase_index,
            moras=accent_moras,
            accent_state=accent_state,
            accent_nucleus=nucleus,
            interrogative=(context.interrogative is True if context else False),
            boundary_strength=boundary,
            confidence=1.0 if context is not None else 0.5,
            provenance={
                "raw_F": context.raw if context else None,
                "label_accent_nucleus_one_based": label_nucleus,
                "internal_accent_nucleus_zero_based": nucleus,
                "position_in_breath_group_one_based": (
                    context.position_forward_in_breath_group
                    if context else None
                ),
            },
        )

    for phrase_index in sorted(phrase_has_speech):
        phrase_accents = tuple(
            accent_objects[item.index]
            for item in accent_buckets
            if item.phrase_index == phrase_index
        )
        phrase_moras = tuple(
            mora for accent in phrase_accents for mora in accent.moras
        )
        source = (
            source_phrases[phrase_index]
            if phrase_index < len(source_phrases)
            else _SourcePhrase("", "", phrase_boundaries.get(phrase_index, 3), False)
        )
        phone_indices = tuple(
            phone.index for phone in phones
            if phone.phrase_index == phrase_index
            and not phone.is_pause and not phone.is_silence
        )
        phrase_objects.append(JapanesePhrase(
            index=phrase_index,
            surface=source.surface,
            normalized_reading="".join(mora.reading for mora in phrase_moras),
            accent_phrases=phrase_accents,
            punctuation_after=source.punctuation,
            # Open JTalk emits the same generic ``pau`` class for several
            # punctuation strengths.  An explicit source mark is therefore
            # more specific: in particular, Japanese comma ``、`` must remain
            # a minor boundary instead of being promoted to a major pause.
            boundary_strength=(
                source.boundary_strength if source.punctuation
                else phrase_boundaries.get(
                    phrase_index, source.boundary_strength)
            ),
            interrogative=(
                source.interrogative
                or any(accent.interrogative for accent in phrase_accents)
            ),
            phone_indices=phone_indices,
            confidence=min(
                (mora.confidence for mora in phrase_moras), default=1.0
            ),
            provenance={
                "source_phrase_matched": phrase_index < len(source_phrases),
                "raw_label_phone_indices": list(phone_indices),
            },
        ))

    if not labels:
        diagnostics.append(JapaneseFrontendDiagnostic(
            code="openjtalk_no_labels",
            message="Open JTalk returned no full-context labels.",
            severity="error",
            action="Verify the Open JTalk dictionary and input text.",
            frontend=frontend_name,
            confidence=0.0,
        ))
    if len(source_phrases) != len(phrase_objects) and source_text.strip():
        diagnostics.append(JapaneseFrontendDiagnostic(
            code="openjtalk_source_phrase_mismatch",
            message=(
                f"Source punctuation suggests {len(source_phrases)} phrases "
                f"but labels contain {len(phrase_objects)}."
            ),
            severity="info",
            action="Use label pauses as authoritative and keep the source text.",
            frontend=frontend_name,
            confidence=0.7,
        ))

    effective_reading = reading or "".join(
        phrase.normalized_reading for phrase in phrase_objects
    )
    confidence = 1.0
    if any(item.severity == "error" for item in diagnostics):
        confidence = 0.3
    elif diagnostics:
        confidence = 0.75
    return JapaneseUtterance(
        source_text=source_text,
        normalized_reading=effective_reading,
        phrases=tuple(phrase_objects),
        phones=tuple(phones),
        diagnostics=tuple(diagnostics),
        frontend_name=frontend_name,
        frontend_version=frontend_version,
        confidence=confidence,
        provenance={
            "raw_labels": list(labels),
            "raw_label_count": len(labels),
            "label_format": "Open JTalk JPCommon A-K full-context",
            "accent_label_indexing": "one-based; zero means unaccented",
            "canonical_accent_indexing": "zero-based within accent phrase",
            "duration_authority": False,
            "f0_authority": False,
            "morphology_available": bool(morphology),
            "inline_bracket_pause_label_indices": inline_pause_labels,
        },
    )


def _module_version(module: Any) -> Optional[str]:
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        return metadata.version("pyopenjtalk")
    except metadata.PackageNotFoundError:
        return None


class OpenJTalkJapaneseFrontend:
    name = "openjtalk"

    def __init__(self, module: Any = None):
        self._module = module

    def _load(self) -> Any:
        if self._module is not None:
            return self._module
        available, reason = pyopenjtalk_status()
        if not available:
            raise OpenJTalkUnavailableError(JapaneseFrontendDiagnostic(
                code="pyopenjtalk_unavailable",
                message=(
                    "The Open JTalk frontend was requested, but it is not "
                    f"operational: {reason}."
                ),
                severity="error",
                action=(
                    "Repair or reinstall pyopenjtalk and its dictionary, or "
                    "select the kana frontend."
                ),
                frontend=self.name,
                confidence=1.0,
            ))
        try:
            self._module = importlib.import_module("pyopenjtalk")
        except (ImportError, OSError) as error:
            raise OpenJTalkUnavailableError(JapaneseFrontendDiagnostic(
                code="pyopenjtalk_import_failed",
                message=f"pyopenjtalk could not be imported: {error}",
                severity="error",
                action=(
                    "Repair the local pyopenjtalk installation or select kana."
                ),
                frontend=self.name,
                confidence=1.0,
                raw_data={"exception": repr(error)},
            )) from error
        return self._module

    def analyze(self, text: str) -> JapaneseUtterance:
        module = self._load()
        try:
            reading = module.g2p(text, kana=True)
            labels = module.extract_fullcontext(text)
            morphology_nodes = (
                module.run_frontend(text)
                if callable(getattr(module, "run_frontend", None)) else None
            )
        except Exception as error:
            raise OpenJTalkAnalysisError(JapaneseFrontendDiagnostic(
                code="pyopenjtalk_analysis_failed",
                message=f"Open JTalk could not analyze the text: {error}",
                severity="error",
                action=(
                    "Check the local dictionary/input, or select kana for a "
                    "dependency-free fallback."
                ),
                frontend=self.name,
                confidence=1.0,
                raw_data={"exception": repr(error)},
            )) from error
        return parse_openjtalk_labels(
            labels,
            source_text=text,
            normalized_reading=reading,
            frontend_name=self.name,
            frontend_version=_module_version(module),
            morphology_nodes=morphology_nodes,
        )
