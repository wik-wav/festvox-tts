# -*- coding: utf-8 -*-
"""Dictionary-driven Asaxi utterance and F0 planning.

The lexical dictionary supplies canonical phones and one H/L value per mora.
This module composes those word-level records into an utterance contour while
keeping Asaxi's boundary tones and question deaccenting separate from the
English and Japanese frontends.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Iterable, Mapping, Optional, Sequence

import asaxi_frontend as af
import asaxi_pitch as asaxi_pitch_domain
import english_syllables as english_syllable_domain


DEFAULT_DICTIONARY_PATH = (
    Path(__file__).resolve().parent
    / "dictionaries"
    / "asaxi_lexicon.json"
)
WH_WORDS = frozenset({
    "kjo", "kvå", "ksá", "kshá", "ksi", "ksè", "kăgo",
})
QUESTION_PARTICLES = frozenset({"kè", "kkè"})
APPEAL_PARTICLES = frozenset({"kă", "në"})
INSISTENT_TAILS = frozenset({"wő", "ő"})
ATONAL_PREFIXES = ("zè", "pa", "na", "no")
DOMINANT_MORPHEMES = ("ná", "xă", "ă")
PLURAL_GRAMMAR_SOURCE = (
    "01_Worldbuilding/Asaxi/Grammar_Structure/"
    "10_Nominal Pluralization in Asaxi.md"
)
COMPOUND_GRAMMAR_SOURCE = (
    "01_Worldbuilding/Asaxi/Grammar_Structure/"
    "16_Adjectives_Constitution vs Simile.md"
)
COMPOUND_PROSODY_SOURCE = (
    "01_Worldbuilding/Asaxi/Grammar_Structure/"
    "61_Prosody, Stress & Intonation.md"
)
PLURAL_DIPHTHONG_ENDINGS = frozenset({"ă", "å", "ë", "ỏ", "ő", "ů"})
COMPOUND_LEXICAL_TYPES = frozenset({
    "adjective",
    "noun",
    "number",
    "root",
    "root word",
    "verb",
})
COMPOUND_BRIDGES = frozenset({"w"})
DOCUMENTED_COMPOUND_SEGMENTATIONS = {
    "gaviŕoŕo": ("gavi", "ŕoŕo"),
}
PRODUCTIVE_SUFFIX_MORPHEMES = (
    (
        "ů",
        "-ů",
        "verbalizer",
        "01_Worldbuilding/Asaxi/Grammar_Structure/06_Verbs in Asaxi.md",
    ),
    (
        "nă",
        "-nă",
        "adjectival-suffix",
        "01_Worldbuilding/Asaxi/Grammar_Structure/"
        "09_Adjectives_Forming Adjectives in Asaxi.md",
    ),
    (
        "nýj",
        "-nýj",
        "adjectival-suffix",
        "01_Worldbuilding/Asaxi/Grammar_Structure/"
        "09_Adjectives_Forming Adjectives in Asaxi.md",
    ),
    (
        "ŕa",
        "-ŕa",
        "stative-suffix",
        "01_Worldbuilding/Asaxi/Grammar_Structure/"
        "21_Existential Logic & The Validity System.md",
    ),
)
MORPHOLOGY_INVENTORY_CACHE_LIMIT = 8
MORPHOLOGY_ANALYSIS_CACHE_LIMIT = 4096


@dataclass(frozen=True)
class AsaxiProsodyDiagnostic:
    code: str
    message: str
    severity: str = "warning"
    word_index: Optional[int] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "word_index": self.word_index,
        }


@dataclass(frozen=True)
class AsaxiProsodyMora:
    index: int
    word_index: int
    word: str
    text: str
    phones: tuple[str, ...]
    phone_start: int
    phone_end: int
    lexical_pitch: str
    pitch: str
    accentable: bool
    kind: str

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "word_index": self.word_index,
            "word": self.word,
            "text": self.text,
            "phones": list(self.phones),
            "phone_start": self.phone_start,
            "phone_end": self.phone_end,
            "lexical_pitch": self.lexical_pitch,
            "pitch": self.pitch,
            "accentable": self.accentable,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class AsaxiProsodyMorpheme:
    surface: str
    lemma: str
    role: str
    pitch_accent_class: str
    source_note: str = ""
    rule: str = ""
    children: tuple[AsaxiProsodyMorpheme, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "lemma": self.lemma,
            "role": self.role,
            "pitch_accent_class": self.pitch_accent_class,
            "source_note": self.source_note,
            "rule": self.rule,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True)
class AsaxiProsodyWord:
    index: int
    surface: str
    lexical_type: str
    phones: tuple[str, ...]
    pitch_accent: str
    pitch_accent_class: str
    mora_start: int
    mora_end: int
    dictionary_source: str
    phrase_expression: str = ""
    phrase_source: str = ""
    morphemes: tuple[AsaxiProsodyMorpheme, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "surface": self.surface,
            "lexical_type": self.lexical_type,
            "phones": list(self.phones),
            "pitch_accent": self.pitch_accent,
            "pitch_accent_class": self.pitch_accent_class,
            "mora_start": self.mora_start,
            "mora_end": self.mora_end,
            "dictionary_source": self.dictionary_source,
            "phrase_expression": self.phrase_expression,
            "phrase_source": self.phrase_source,
            "morphemes": [
                morpheme.to_dict() for morpheme in self.morphemes
            ],
        }


def format_morpheme_analysis(
    morphemes: Sequence[AsaxiProsodyMorpheme],
    *,
    markdown: bool = False,
) -> str:
    """Format a recursive morpheme tree without discarding its hierarchy."""

    def render(morpheme: AsaxiProsodyMorpheme) -> str:
        lemma = (
            f"`{morpheme.lemma}`"
            if markdown
            else morpheme.lemma
        )
        role = morpheme.role.replace("-", " ")
        if not morpheme.children:
            return f"{lemma} ({role})"
        children = " + ".join(render(child) for child in morpheme.children)
        return f"{lemma} ({role}: {children})"

    return " + ".join(render(morpheme) for morpheme in morphemes)


@dataclass(frozen=True)
class AsaxiProsodyPlan:
    source_text: str
    normalized_text: str
    words: tuple[AsaxiProsodyWord, ...]
    moras: tuple[AsaxiProsodyMora, ...]
    phones: tuple[str, ...]
    boundary_mark: str
    boundary_tone: str
    interrogative: bool
    directive: bool
    dictionary_ruleset: str
    diagnostics: tuple[AsaxiProsodyDiagnostic, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_text": self.source_text,
            "normalized_text": self.normalized_text,
            "words": [word.to_dict() for word in self.words],
            "moras": [mora.to_dict() for mora in self.moras],
            "phones": list(self.phones),
            "boundary_mark": self.boundary_mark,
            "boundary_tone": self.boundary_tone,
            "interrogative": self.interrogative,
            "directive": self.directive,
            "dictionary_ruleset": self.dictionary_ruleset,
            "diagnostics": [
                diagnostic.to_dict() for diagnostic in self.diagnostics
            ],
        }


@dataclass(frozen=True)
class AsaxiRenderedMora:
    """One sentence-level mora aligned to Festival's rendered segments."""

    index: int
    phrase_index: int
    local_mora_index: int
    word_index: int
    word: str
    text: str
    phones: tuple[str, ...]
    pitch: str
    lexical_pitch: str
    accentable: bool
    kind: str
    segment_indices: tuple[int, ...]
    start: Optional[float]
    end: Optional[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "mora_index": self.index,
            "phrase_index": self.phrase_index,
            "local_mora_index": self.local_mora_index,
            "word_index": self.word_index,
            "word": self.word,
            "text": self.text,
            "phones": list(self.phones),
            "pitch": self.pitch,
            "lexical_pitch": self.lexical_pitch,
            "accentable": self.accentable,
            "kind": self.kind,
            "segment_indices": list(self.segment_indices),
            "start": self.start,
            "end": self.end,
        }


_DICTIONARY_CACHE: dict[Path, tuple[tuple[int, int], af.AsaxiSynthesisDictionary]] = {}


def load_dictionary(
    path: str | Path = DEFAULT_DICTIONARY_PATH,
) -> af.AsaxiSynthesisDictionary:
    """Load a dictionary once per stable file size/mtime pair."""

    source = Path(path).resolve()
    stat = source.stat()
    token = (int(stat.st_mtime_ns), int(stat.st_size))
    cached = _DICTIONARY_CACHE.get(source)
    if cached is None or cached[0] != token:
        cached = (token, af.load_synthesis_dictionary(source))
        _DICTIONARY_CACHE[source] = cached
    return cached[1]


def _boundary_mark(text: str) -> str:
    match = re.search(
        r"([.?!,:;])[\s”’\"')\]]*$",
        str(text or ""),
    )
    return match.group(1) if match else "."


def _variant_payload(
    entry: af.AsaxiLexiconEntry,
    lexical_type: str = "",
) -> tuple[Mapping[str, object], bool]:
    requested = str(lexical_type or "").strip().casefold()
    if requested:
        for variant in entry.variants:
            if str(variant.get("lexical_type") or "").casefold() == requested:
                return variant, True
        raise ValueError(
            f"{entry.word!r} has no {requested!r} dictionary variant"
        )
    return {
        "lexical_type": "",
        "phones": entry.phones,
        "pitch_accent": entry.pitch_accent,
        "pitch_accent_class": entry.pitch_accent_class,
        "g2p_override": entry.g2p_override,
        "source_note": entry.source_note,
    }, False


def _partition_phones(
    moras: Sequence[af.AsaxiMora],
    phones: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    expected = tuple(
        phone for mora in moras for phone in mora.phones
    )
    actual = tuple(str(phone) for phone in phones)
    if actual == expected:
        return tuple(tuple(mora.phones) for mora in moras)
    if not moras:
        return ()
    weights = [len(mora.phones) for mora in moras]
    if not any(weights):
        weights[-1] = 1
    total = sum(weights)
    boundaries = [0]
    consumed = 0
    for weight in weights[:-1]:
        consumed += weight
        boundaries.append(round(len(actual) * consumed / total))
    boundaries.append(len(actual))
    return tuple(
        actual[boundaries[index]:boundaries[index + 1]]
        for index in range(len(moras))
    )


def _borrowed_term_moras(
    surface: str,
    phones: Sequence[str],
) -> tuple[af.AsaxiMora, ...]:
    """Represent an English-routed term as syllable-sized Asaxi beats.

    Capitalized terms participate in the surrounding Asaxi contour, but their
    internal grouping comes from the English phone stream rather than from
    Asaxi spelling. Character spans are display approximations only; phone
    membership is supplied by the deterministic English syllabifier.
    """

    pronunciation = tuple(str(phone) for phone in phones)
    syllabification = english_syllable_domain.syllabify_english(
        pronunciation
    )
    syllables = syllabification.syllables
    if not syllables:
        return (
            af.AsaxiMora(
                0,
                surface,
                pronunciation,
                0,
                len(surface),
                False,
                "borrowed_nonvocalic",
            ),
        )
    result = []
    for index, syllable in enumerate(syllables):
        start = round(len(surface) * index / len(syllables))
        end = round(len(surface) * (index + 1) / len(syllables))
        label = surface[start:end] or surface
        result.append(af.AsaxiMora(
            index=index,
            text=label,
            phones=tuple(syllable.phones),
            start=start,
            end=end,
            accentable=bool(syllable.nucleus),
            kind="borrowed_syllable",
        ))
    return tuple(result)


def _phrase_overrides(
    dictionary: af.AsaxiSynthesisDictionary,
    words: Sequence[str],
) -> dict[int, tuple[tuple[str, ...], str, str]]:
    """Return non-overlapping leftmost-longest expression accent matches.

    An idiom can occur inside a larger utterance, so exact whole-utterance
    lookup is insufficient. Each result maps one word index to that word's
    accent chunk plus the expression and source note that supplied it.
    """

    candidates: dict[int, list[tuple[int, str, Mapping[str, object]]]] = {}
    normalized_words = tuple(af.normalize_word(word) for word in words)
    for expression, raw_record in dictionary.phrases.items():
        if not isinstance(raw_record, Mapping):
            continue
        expression_words = af.words_in_text(
            expression,
            reject_unsupported_letters=True,
        )
        if len(expression_words) < 2:
            continue
        width = len(expression_words)
        for start in range(0, len(normalized_words) - width + 1):
            if normalized_words[start:start + width] == expression_words:
                candidates.setdefault(start, []).append(
                    (width, expression, raw_record)
                )

    matched: dict[int, tuple[tuple[str, ...], str, str]] = {}
    cursor = 0
    while cursor < len(normalized_words):
        options = sorted(
            candidates.get(cursor, ()),
            key=lambda row: (-row[0], row[1]),
        )
        selected = None
        for width, expression, record in options:
            parsed = af.parse_pitch_pattern(
                str(record.get("pitch_accent") or "")
            )
            if len(parsed) == width:
                selected = (width, expression, record, parsed)
                break
        if selected is None:
            cursor += 1
            continue
        width, expression, record, parsed = selected
        source = str(record.get("source_note") or "")
        for local_index, values in enumerate(parsed):
            matched[cursor + local_index] = (
                tuple(values),
                expression,
                source,
            )
        cursor += width
    return matched


@dataclass(frozen=True)
class _MorphologicalPitch:
    values: tuple[str, ...]
    lexical_type: str
    source_note: str
    components: tuple[AsaxiProsodyMorpheme, ...]
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class _MorphemeSpec:
    surface: str
    lemma: str
    role: str
    values: tuple[str, ...]
    pitch_accent_class: str
    host_lexical_types: tuple[str, ...] = ()
    source_note: str = ""
    rule: str = "concatenative"

    def public(self) -> AsaxiProsodyMorpheme:
        return AsaxiProsodyMorpheme(
            surface=self.surface,
            lemma=self.lemma,
            role=self.role,
            pitch_accent_class=self.pitch_accent_class,
            source_note=self.source_note,
            rule=self.rule,
        )


@dataclass(frozen=True)
class _StemRealization:
    surface: str
    root_text: str
    root_values: tuple[str, ...]
    root_moras: tuple[af.AsaxiMora, ...]
    lexical_type: str
    source_notes: tuple[str, ...]
    values: tuple[str, ...]
    components: tuple[AsaxiProsodyMorpheme, ...]
    rule_priority: int = 0
    rule: str = "lexical-stem"


@dataclass(frozen=True)
class _CompoundPart:
    surface: str
    stem: Optional[_StemRealization] = None

    @property
    def is_bridge(self) -> bool:
        return self.stem is None


@dataclass(frozen=True)
class _AttestedPart:
    surface: str
    lemma: str
    role: str
    values: tuple[str, ...]
    pitch_accent_class: str
    lexical_type: str = ""
    source_note: str = ""
    is_content: bool = False


@dataclass
class _MorphologyInventory:
    dictionary: af.AsaxiSynthesisDictionary
    prefixes: dict[str, _MorphemeSpec]
    suffixes: dict[str, _MorphemeSpec]
    stems: tuple[tuple[str, _StemRealization], ...]
    compound_units: dict[str, tuple[_StemRealization, ...]]
    analyses: dict[str, Optional[_MorphologicalPitch]]


_MORPHOLOGY_INVENTORIES: dict[int, _MorphologyInventory] = {}


def _segment_morphemes(
    value: str,
    candidates: Sequence[str],
) -> Optional[tuple[str, ...]]:
    """Return a deterministic longest-first segmentation, if one exists."""

    normalized = af.normalize_word(value)
    ordered = tuple(sorted(
        {
            af.normalize_word(candidate)
            for candidate in candidates
            if af.normalize_word(candidate)
        },
        key=lambda item: (-len(item), item),
    ))
    memo: dict[int, Optional[tuple[str, ...]]] = {}

    def visit(offset: int) -> Optional[tuple[str, ...]]:
        if offset == len(normalized):
            return ()
        if offset in memo:
            return memo[offset]
        for candidate in ordered:
            if not normalized.startswith(candidate, offset):
                continue
            tail = visit(offset + len(candidate))
            if tail is not None:
                memo[offset] = (candidate,) + tail
                return memo[offset]
        memo[offset] = None
        return None

    return visit(0)


def _mora_signature(value: str) -> tuple[str, ...]:
    return tuple(mora.text for mora in af.split_morae(value))


def _entry_lexical_type(entry: af.AsaxiLexiconEntry) -> str:
    if entry.variants:
        return str(entry.variants[0].get("lexical_type") or "")
    return ""


def _variant_for_type(
    entry: af.AsaxiLexiconEntry,
    lexical_type: str,
) -> Optional[Mapping[str, object]]:
    requested = str(lexical_type or "").casefold()
    for variant in entry.variants:
        if str(variant.get("lexical_type") or "").casefold() == requested:
            return variant
    return None


def _variant_pitch_values(
    entry: af.AsaxiLexiconEntry,
    variant: Mapping[str, object],
) -> Optional[tuple[str, ...]]:
    parsed = af.parse_pitch_pattern(str(variant.get("pitch_accent") or ""))
    if len(parsed) != 1 or len(parsed[0]) != len(entry.moras):
        return None
    return tuple(parsed[0])


def _entry_morpheme_values(
    entry: af.AsaxiLexiconEntry,
) -> tuple[str, ...]:
    values = tuple(entry.pitch_values)
    if values:
        return values
    # Consonantal bridges such as -n- have no independent pitch-bearing
    # mora. They merge with the following vowel in the surface word.
    if all(not mora.accentable for mora in entry.moras):
        return ()
    return tuple(
        "H"
        if entry.pitch_accent_class == "dominant" and mora.accentable
        else "L"
        for mora in entry.moras
    )


def _indexed_morpheme_values(
    record: Mapping[str, object],
    surface: str,
) -> tuple[str, ...]:
    parsed = af.parse_pitch_pattern(
        str(record.get("pitch_accent") or "")
    )
    moras = af.split_morae(surface)
    if len(parsed) == 1 and len(parsed[0]) == len(moras):
        return tuple(parsed[0])
    if all(not mora.accentable for mora in moras):
        return ()
    return tuple(
        "H"
        if (
            str(record.get("pitch_accent_class") or "")
            == "dominant"
            and mora.accentable
        )
        else "L"
        for mora in moras
    )


def _bound_morpheme_inventories(
    dictionary: af.AsaxiSynthesisDictionary,
) -> tuple[dict[str, _MorphemeSpec], dict[str, _MorphemeSpec]]:
    """Build positional affix inventories from dictionary notation.

    A trailing hyphen marks a prefix, a leading hyphen marks a suffix, and a
    morpheme with both is a bound bridge in the post-root chain. Free particles
    are not treated as affixes merely because their spelling happens to match
    the end of a word.
    """

    prefixes: dict[str, tuple[int, _MorphemeSpec]] = {}
    suffixes: dict[str, tuple[int, _MorphemeSpec]] = {}

    def register(
        target: dict[str, tuple[int, _MorphemeSpec]],
        spec: _MorphemeSpec,
        priority: int,
    ) -> None:
        previous = target.get(spec.surface)
        if previous is None or priority > previous[0]:
            target[spec.surface] = (priority, spec)

    for key, entry in dictionary.entries.items():
        if _entry_lexical_type(entry) != "particle":
            continue
        leading = key.startswith("-")
        trailing = key.endswith("-")
        surface = af.normalize_word(key.strip("-"))
        if not surface:
            continue
        values = _entry_morpheme_values(entry)
        if trailing and not leading:
            register(
                prefixes,
                _MorphemeSpec(
                    surface=surface,
                    lemma=key,
                    role="prefix",
                    values=values,
                    pitch_accent_class=entry.pitch_accent_class,
                    source_note=entry.source_note,
                    rule="dictionary-bound-prefix",
                ),
                4,
            )
        if leading:
            register(
                suffixes,
                _MorphemeSpec(
                    surface=surface,
                    lemma=key,
                    role=("interfix" if trailing else "suffix"),
                    values=values,
                    pitch_accent_class=entry.pitch_accent_class,
                    source_note=entry.source_note,
                    rule="dictionary-bound-suffix",
                ),
                4,
            )
    for key, record in dictionary.morphemes.items():
        attachment = str(record.get("attachment") or "").casefold()
        if attachment not in {"prefix", "suffix", "infix"}:
            continue
        surface = af.normalize_word(
            str(record.get("surface") or key.strip("-"))
        )
        if not surface:
            continue
        source_notes = tuple(
            str(note)
            for note in record.get("source_notes") or ()
            if str(note)
        )
        spec = _MorphemeSpec(
            surface=surface,
            lemma=str(record.get("canonical_form") or key),
            role=str(record.get("role") or attachment),
            values=_indexed_morpheme_values(record, surface),
            pitch_accent_class=str(
                record.get("pitch_accent_class") or "atonal"
            ),
            host_lexical_types=tuple(
                str(item).casefold()
                for item in record.get("host_lexical_types") or ()
                if str(item)
            ),
            source_note=(source_notes[0] if source_notes else ""),
            rule=(
                "canonical-grammar-morpheme"
                if record.get("registry_source")
                else "dictionary-morpheme-index"
            ),
        )
        priority = (
            8 if record.get("override_source")
            else (7 if record.get("registry_source") else 5)
        )
        if attachment == "prefix":
            register(prefixes, spec, priority)
        else:
            register(suffixes, spec, priority)

    for surface in ATONAL_PREFIXES:
        source_entry = (
            dictionary.entries.get(surface + "-")
            or dictionary.lookup(surface)
        )
        register(
            prefixes,
            _MorphemeSpec(
                surface=surface,
                lemma=surface + "-",
                role="prefix",
                values=tuple("L" for _ in af.split_morae(surface)),
                pitch_accent_class="atonal",
                source_note=(
                    source_entry.source_note if source_entry else ""
                ),
                rule="documented-atonal-prefix",
            ),
            5,
        )

    for surface in DOMINANT_MORPHEMES:
        source_entry = dictionary.lookup(surface)
        values = (
            _entry_morpheme_values(source_entry)
            if source_entry is not None
            else tuple(
                "H" if mora.accentable else "L"
                for mora in af.split_morae(surface)
            )
        )
        for target, role in (
            (prefixes, "dominant-prefix"),
            (suffixes, "dominant-suffix"),
        ):
            register(
                target,
                _MorphemeSpec(
                    surface=surface,
                    lemma=surface,
                    role=role,
                    values=values,
                    pitch_accent_class="dominant",
                    source_note=(
                        source_entry.source_note if source_entry else ""
                    ),
                    rule="mobile-dominant-morpheme",
                ),
                3,
            )

    for surface, lemma, role, source_note in PRODUCTIVE_SUFFIX_MORPHEMES:
        register(
            suffixes,
            _MorphemeSpec(
                surface=surface,
                lemma=lemma,
                role=role,
                values=tuple(
                    "L" for _ in af.split_morae(surface)
                ),
                pitch_accent_class="atonal",
                source_note=source_note,
                rule="documented-productive-suffix",
            ),
            3,
        )

    return (
        {key: row[1] for key, row in prefixes.items()},
        {key: row[1] for key, row in suffixes.items()},
    )


def _plural_surface_form(root: str) -> tuple[str, str, str]:
    """Return surface form, allomorph, and documented rule identifier."""

    value = af.normalize_word(root)
    moras = af.split_morae(value)
    if value.endswith("o"):
        return value[:-1] + "a", "-a", "plural-o-replacement"
    if value.endswith(("a", "á")):
        return value + "ma", "-ma", "plural-a-ma"
    if value.endswith(("mm", "nn")):
        return value[:-1] + "a", "-a", "plural-syllabic-resolution"
    if (
        len(moras) >= 2
        and moras[-1].text == moras[-2].text
        and value[-1:] in PLURAL_DIPHTHONG_ENDINGS
    ):
        return (
            value[:-1] + "a",
            "-a",
            "plural-reduplicated-diphthong-reduction",
        )
    if value[-1:] in af.VOWEL_GRAPHEMES - PLURAL_DIPHTHONG_ENDINGS:
        return value + "wa", "-wa", "plural-vowel-bridge"
    return value + "a", "-a", "plural-simple-suffix"


def _plural_stem_realization(
    root_text: str,
    root_entry: af.AsaxiLexiconEntry,
) -> Optional[_StemRealization]:
    noun_variant = _variant_for_type(root_entry, "noun")
    if noun_variant is None:
        return None
    root_values = _variant_pitch_values(root_entry, noun_variant)
    if root_values is None:
        return None
    surface, allomorph, rule = _plural_surface_form(root_text)
    surface_moras = af.split_morae(surface)
    if len(surface_moras) < len(root_values):
        return None

    values = list(root_values)
    added = len(surface_moras) - len(root_values)
    root_accentable = sum(mora.accentable for mora in root_entry.moras)
    spread_root_high = (
        root_accentable == 1 and any(value == "H" for value in root_values)
    )
    if added:
        extension = surface_moras[-added:]
        values.extend(
            "H" if spread_root_high and mora.accentable else "L"
            for mora in extension
        )
    if len(values) != len(surface_moras):
        return None

    source_note = str(
        noun_variant.get("source_note") or root_entry.source_note
    )
    plural_surface = {
        "-ma": "ma",
        "-wa": "wa",
    }.get(allomorph, "a")
    return _StemRealization(
        surface=surface,
        root_text=root_text,
        root_values=root_values,
        root_moras=root_entry.moras,
        lexical_type="noun",
        source_notes=tuple(
            item for item in (source_note, PLURAL_GRAMMAR_SOURCE) if item
        ),
        values=tuple(values),
        components=(
            AsaxiProsodyMorpheme(
                surface=root_text,
                lemma=root_text,
                role="root",
                pitch_accent_class=str(
                    noun_variant.get("pitch_accent_class")
                    or root_entry.pitch_accent_class
                ),
                source_note=source_note,
                rule="dictionary-stem",
            ),
            AsaxiProsodyMorpheme(
                surface=plural_surface,
                lemma=allomorph,
                role="plural",
                pitch_accent_class="atonal",
                source_note=PLURAL_GRAMMAR_SOURCE,
                rule=rule,
            ),
        ),
        # Prefer additive allomorphs when both an additive and a reconstructed
        # replacement analysis are possible. For example, documented dă+a
        # must beat the hypothetical inverse dăo->dăa.
        rule_priority=(
            3
            if rule in {
                "plural-a-ma",
                "plural-vowel-bridge",
                "plural-simple-suffix",
            }
            else 2
        ),
        rule=rule,
    )


def _stem_realizations(
    root_text: str,
    root_entry: af.AsaxiLexiconEntry,
) -> tuple[_StemRealization, ...]:
    source_note = root_entry.source_note
    identity = _StemRealization(
        surface=root_text,
        root_text=root_text,
        root_values=tuple(root_entry.pitch_values),
        root_moras=root_entry.moras,
        lexical_type=_entry_lexical_type(root_entry),
        source_notes=(source_note,) if source_note else (),
        values=tuple(root_entry.pitch_values),
        components=(
            AsaxiProsodyMorpheme(
                surface=root_text,
                lemma=root_text,
                role="root",
                pitch_accent_class=root_entry.pitch_accent_class,
                source_note=source_note,
                rule="dictionary-stem",
            ),
        ),
    )
    plural = _plural_stem_realization(root_text, root_entry)
    return (identity,) + ((plural,) if plural is not None else ())


def _morphology_inventory(
    dictionary: af.AsaxiSynthesisDictionary,
) -> _MorphologyInventory:
    cache_key = id(dictionary)
    cached = _MORPHOLOGY_INVENTORIES.get(cache_key)
    if cached is not None and cached.dictionary is dictionary:
        return cached

    prefixes, suffixes = _bound_morpheme_inventories(dictionary)
    stems: list[tuple[str, _StemRealization]] = []
    compound_units: dict[str, list[_StemRealization]] = {}
    for root_text, root_entry in dictionary.entries.items():
        if (
            not root_text
            or any(character in root_text for character in "-. ")
            or root_entry.pitch_accent_class not in {"lexical", "mixed"}
        ):
            continue
        realizations = _stem_realizations(root_text, root_entry)
        stems.extend((root_text, stem) for stem in realizations)
        for stem in realizations:
            if (
                stem.lexical_type.casefold() in COMPOUND_LEXICAL_TYPES
                and stem.surface
            ):
                compound_units.setdefault(stem.surface[0], []).append(stem)
    cached = _MorphologyInventory(
        dictionary=dictionary,
        prefixes=prefixes,
        suffixes=suffixes,
        stems=tuple(stems),
        compound_units={
            initial: tuple(sorted(
                units,
                key=lambda stem: (
                    -len(stem.surface),
                    -stem.rule_priority,
                    stem.root_text,
                    stem.rule,
                ),
            ))
            for initial, units in compound_units.items()
        },
        analyses={},
    )
    while (
        len(_MORPHOLOGY_INVENTORIES) >= MORPHOLOGY_INVENTORY_CACHE_LIMIT
        and cache_key not in _MORPHOLOGY_INVENTORIES
    ):
        _MORPHOLOGY_INVENTORIES.pop(next(iter(_MORPHOLOGY_INVENTORIES)))
    _MORPHOLOGY_INVENTORIES[cache_key] = cached
    return cached


def _cache_morphology_analysis(
    inventory: _MorphologyInventory,
    word: str,
    result: Optional[_MorphologicalPitch],
) -> None:
    while (
        len(inventory.analyses) >= MORPHOLOGY_ANALYSIS_CACHE_LIMIT
        and word not in inventory.analyses
    ):
        inventory.analyses.pop(next(iter(inventory.analyses)))
    inventory.analyses[word] = result


def _component_values(
    spec: _MorphemeSpec,
    *,
    spread_root_high: bool,
) -> tuple[str, ...]:
    if (
        spread_root_high
        and spec.values
        and spec.pitch_accent_class == "atonal"
    ):
        return tuple(
            "H" if mora.accentable else "L"
            for mora in af.split_morae(spec.surface)
        )
    return spec.values


def _morpheme_accepts_host(
    spec: _MorphemeSpec,
    lexical_type: str,
) -> bool:
    return (
        not spec.host_lexical_types
        or str(lexical_type or "").casefold() in spec.host_lexical_types
    )


def _compound_segmentations(
    word: str,
    start: int,
    inventory: _MorphologyInventory,
) -> tuple[tuple[_CompoundPart, ...], ...]:
    """Segment a compound body into typed lexical units and optional bridges."""

    stop = len(word)
    memo: dict[int, tuple[tuple[_CompoundPart, ...], ...]] = {}

    def visit(offset: int) -> tuple[tuple[_CompoundPart, ...], ...]:
        if offset == stop:
            return ((),)
        if offset in memo:
            return memo[offset]

        found: dict[
            tuple[tuple[str, str, str, str], ...],
            tuple[_CompoundPart, ...],
        ] = {}
        for stem in inventory.compound_units.get(word[offset], ()):
            if not word.startswith(stem.surface, offset):
                continue
            end = offset + len(stem.surface)
            # Number belongs to the compound head. Accepting an inflected
            # modifier would over-segment many ordinary lexical forms.
            if stem.rule != "lexical-stem" and end != stop:
                continue

            tails = list(visit(end))
            if end < stop and word[end] in COMPOUND_BRIDGES:
                for tail in visit(end + 1):
                    if not tail:
                        continue
                    tails.append(
                        (_CompoundPart(surface=word[end]),) + tail
                    )
            for tail in tails:
                row = (_CompoundPart(surface=stem.surface, stem=stem),) + tail
                key = tuple(
                    (
                        "bridge",
                        part.surface,
                        "",
                        "",
                    )
                    if part.is_bridge
                    else (
                        "stem",
                        part.surface,
                        part.stem.root_text,
                        part.stem.rule,
                    )
                    for part in row
                )
                found[key] = row

        def order(parts: tuple[_CompoundPart, ...]) -> tuple[object, ...]:
            members = tuple(
                part.stem for part in parts if part.stem is not None
            )
            return (
                len(members),
                -sum(stem.rule_priority for stem in members),
                -sum(len(stem.surface) ** 2 for stem in members),
                sum(part.is_bridge for part in parts),
                tuple(
                    (
                        part.surface
                        if part.is_bridge
                        else f"{part.stem.root_text}:{part.stem.rule}"
                    )
                    for part in parts
                ),
            )

        memo[offset] = tuple(sorted(found.values(), key=order)[:64])
        return memo[offset]

    return visit(start)


def _attested_compound_segmentations(
    word: str,
    dictionary: af.AsaxiSynthesisDictionary,
) -> tuple[tuple[str, ...], ...]:
    record = dictionary.lookup_morphology(word)
    rows = []
    if record is not None:
        for analysis in record.get("analyses") or ():
            if not isinstance(analysis, Mapping):
                continue
            segments = tuple(
                af.normalize_word(segment)
                for segment in analysis.get("segments") or ()
                if af.normalize_word(segment)
            )
            if len(segments) >= 2 and segments not in rows:
                rows.append(segments)
    documented = DOCUMENTED_COMPOUND_SEGMENTATIONS.get(word)
    if documented and documented not in rows:
        rows.append(documented)
    return tuple(rows)


def _flatten_compound_components(
    components: Sequence[AsaxiProsodyMorpheme],
) -> tuple[str, ...]:
    result = []
    for component in components:
        if component.children:
            result.extend(_flatten_compound_components(component.children))
        else:
            result.append(af.normalize_word(component.surface))
    return tuple(item for item in result if item)


def _content_part(
    surface: str,
    dictionary: af.AsaxiSynthesisDictionary,
) -> Optional[_AttestedPart]:
    entry = dictionary.lookup(surface)
    if entry is None:
        return None
    for variant in entry.variants:
        lexical_type = str(
            variant.get("lexical_type") or ""
        ).casefold()
        if lexical_type not in COMPOUND_LEXICAL_TYPES:
            continue
        values = _variant_pitch_values(entry, variant)
        if values is None:
            continue
        return _AttestedPart(
            surface=surface,
            lemma=surface,
            role="lexical-unit",
            values=values,
            pitch_accent_class=str(
                variant.get("pitch_accent_class") or "lexical"
            ),
            lexical_type=lexical_type,
            source_note=str(
                variant.get("source_note") or entry.source_note
            ),
            is_content=True,
        )
    return None


def _free_morpheme_part(
    surface: str,
    dictionary: af.AsaxiSynthesisDictionary,
) -> Optional[_AttestedPart]:
    entry = dictionary.lookup(surface)
    if entry is not None:
        return _AttestedPart(
            surface=surface,
            lemma=surface,
            role="function-morpheme",
            values=_entry_morpheme_values(entry),
            pitch_accent_class=entry.pitch_accent_class,
            lexical_type=_entry_lexical_type(entry),
            source_note=entry.source_note,
        )
    for key, record in dictionary.morphemes.items():
        indexed_surface = af.normalize_word(
            str(record.get("surface") or key.strip("-"))
        )
        if (
            indexed_surface != surface
            or str(record.get("attachment") or "free") != "free"
        ):
            continue
        source_notes = tuple(
            str(note)
            for note in record.get("source_notes") or ()
            if str(note)
        )
        return _AttestedPart(
            surface=surface,
            lemma=str(record.get("canonical_form") or key),
            role=str(record.get("role") or "function-morpheme"),
            values=_indexed_morpheme_values(record, surface),
            pitch_accent_class=str(
                record.get("pitch_accent_class") or "atonal"
            ),
            lexical_type=str(record.get("lexical_type") or ""),
            source_note=(source_notes[0] if source_notes else ""),
        )
    return None


def _classify_attested_parts(
    segments: Sequence[str],
    inventory: _MorphologyInventory,
) -> Optional[tuple[_AttestedPart, ...]]:
    dictionary = inventory.dictionary
    content = [
        _content_part(segment, dictionary)
        for segment in segments
    ]
    plural_indices: set[int] = set()
    for index in range(1, len(segments)):
        previous = content[index - 1]
        if (
            previous is None
            or previous.lexical_type != "noun"
            or segments[index] not in {"a", "ma", "wa"}
        ):
            continue
        _surface, allomorph, _rule = _plural_surface_form(
            previous.surface
        )
        if allomorph.strip("-") == segments[index]:
            plural_indices.add(index)
            content[index] = None

    content_indices = tuple(
        index for index, part in enumerate(content)
        if part is not None
    )
    parts: list[_AttestedPart] = []
    for index, segment in enumerate(segments):
        if index in plural_indices:
            root = content[index - 1]
            if root is None:
                return None
            spread = (
                len(root.values) == 1
                and root.values[0] == "H"
            )
            parts.append(_AttestedPart(
                surface=segment,
                lemma=f"-{segment}",
                role="plural",
                values=tuple(
                    "H" if spread and mora.accentable else "L"
                    for mora in af.split_morae(segment)
                ),
                pitch_accent_class="atonal",
                source_note=PLURAL_GRAMMAR_SOURCE,
            ))
            continue

        if segment == "ga" and index == 0:
            source = dictionary.lookup_morpheme("ga")
            source_notes = (
                source.get("source_notes") if source else ()
            )
            parts.append(_AttestedPart(
                surface=segment,
                lemma="ga-",
                role="compound-prefix",
                values=tuple("L" for _ in af.split_morae(segment)),
                pitch_accent_class="atonal",
                source_note=(
                    str(source_notes[0])
                    if source_notes
                    else COMPOUND_GRAMMAR_SOURCE
                ),
            ))
            continue

        if (
            segment in COMPOUND_BRIDGES
            and any(item < index for item in content_indices)
            and any(item > index for item in content_indices)
        ):
            parts.append(_AttestedPart(
                surface=segment,
                lemma=f"-{segment}-",
                role="compound-bridge",
                values=tuple("L" for _ in af.split_morae(segment)),
                pitch_accent_class="atonal",
                source_note=COMPOUND_GRAMMAR_SOURCE,
            ))
            continue

        prefix = inventory.prefixes.get(segment)
        suffix = inventory.suffixes.get(segment)
        if prefix is not None and (
            index == 0
            or any(item > index for item in content_indices)
        ):
            parts.append(_AttestedPart(
                surface=segment,
                lemma=prefix.lemma,
                role=prefix.role,
                values=prefix.values,
                pitch_accent_class=prefix.pitch_accent_class,
                source_note=prefix.source_note,
            ))
            continue
        if suffix is not None and (
            index > 0
            or (
                not suffix.values
                and any(item > index for item in content_indices)
            )
        ):
            host_type = next(
                (
                    content[item].lexical_type
                    for item in reversed(content_indices)
                    if item < index and content[item] is not None
                ),
                "",
            )
            if not _morpheme_accepts_host(suffix, host_type):
                return None
            parts.append(_AttestedPart(
                surface=segment,
                lemma=suffix.lemma,
                role=suffix.role,
                values=suffix.values,
                pitch_accent_class=suffix.pitch_accent_class,
                source_note=suffix.source_note,
            ))
            continue
        if content[index] is not None:
            parts.append(content[index])
            continue
        free = _free_morpheme_part(segment, dictionary)
        if free is None:
            return None
        parts.append(free)
    return tuple(parts)


def _align_attested_pitch(
    parts: Sequence[_AttestedPart],
    surface_moras: Sequence[af.AsaxiMora],
) -> tuple[str, ...]:
    underlying_labels: list[str] = []
    underlying_values: list[str] = []
    for part in parts:
        if not part.values:
            continue
        moras = af.split_morae(part.surface)
        if not moras:
            continue
        for index, mora in enumerate(moras):
            underlying_labels.append(mora.text)
            underlying_values.append(
                part.values[min(index, len(part.values) - 1)]
            )

    surface_labels = [mora.text for mora in surface_moras]
    assigned: list[list[str]] = [[] for _ in surface_labels]
    matcher = SequenceMatcher(
        None,
        underlying_labels,
        surface_labels,
        autojunk=False,
    )
    for operation, left_start, left_end, right_start, right_end in (
        matcher.get_opcodes()
    ):
        left_count = left_end - left_start
        right_count = right_end - right_start
        if operation == "equal":
            for offset in range(left_count):
                assigned[right_start + offset].append(
                    underlying_values[left_start + offset]
                )
            continue
        if operation == "replace" and right_count:
            for offset in range(left_count):
                target = right_start + min(
                    (offset * right_count) // max(1, left_count),
                    right_count - 1,
                )
                assigned[target].append(
                    underlying_values[left_start + offset]
                )
            continue
        if operation == "delete" and surface_labels:
            target = (
                right_start - 1
                if right_start > 0
                else min(right_start, len(surface_labels) - 1)
            )
            assigned[target].extend(
                underlying_values[left_start:left_end]
            )

    return tuple(
        "H" if "H" in values else "L"
        for values in assigned
    )


def _attested_morphological_pitch(
    word: str,
    full_moras: Sequence[af.AsaxiMora],
    inventory: _MorphologyInventory,
) -> Optional[_MorphologicalPitch]:
    record = inventory.dictionary.lookup_morphology(word)
    if record is None:
        return None
    candidates = []
    for analysis in record.get("analyses") or ():
        if not isinstance(analysis, Mapping):
            continue
        segments = tuple(
            af.normalize_word(segment)
            for segment in analysis.get("segments") or ()
            if af.normalize_word(segment)
        )
        if len(segments) < 2:
            continue
        parts = _classify_attested_parts(segments, inventory)
        if parts is None:
            continue
        content_indices = tuple(
            index for index, part in enumerate(parts)
            if part.is_content
        )
        is_compound = (
            bool(parts and parts[0].role == "compound-prefix")
            or len(content_indices) >= 2
        )
        accent_bearer = (
            content_indices[0]
            if is_compound and content_indices
            else (content_indices[-1] if content_indices else None)
        )
        adjusted = []
        for index, part in enumerate(parts):
            values = part.values
            role = part.role
            if is_compound and part.is_content:
                role = (
                    "compound-head"
                    if index == content_indices[-1]
                    else "compound-modifier"
                )
                if index != accent_bearer:
                    values = tuple("L" for _ in values)
            adjusted.append(_AttestedPart(
                surface=part.surface,
                lemma=part.lemma,
                role=role,
                values=values,
                pitch_accent_class=part.pitch_accent_class,
                lexical_type=part.lexical_type,
                source_note=part.source_note,
                is_content=part.is_content,
            ))

        values = _align_attested_pitch(adjusted, full_moras)
        if len(values) != len(full_moras):
            continue
        source_notes = list(analysis.get("source_notes") or ())
        source_notes.extend(
            part.source_note for part in adjusted if part.source_note
        )
        candidates.append(_MorphologicalPitch(
            values=values,
            lexical_type=(
                adjusted[content_indices[-1]].lexical_type
                if content_indices
                else ""
            ),
            source_note="; ".join(dict.fromkeys(source_notes)),
            components=tuple(
                AsaxiProsodyMorpheme(
                    surface=part.surface,
                    lemma=part.lemma,
                    role=part.role,
                    pitch_accent_class=part.pitch_accent_class,
                    source_note=part.source_note,
                    rule="attested-interlinear-segmentation",
                )
                for part in adjusted
            ),
        ))

    signatures = {
        (
            candidate.values,
            tuple(
                (part.surface, part.lemma, part.role)
                for part in candidate.components
            ),
        )
        for candidate in candidates
    }
    if len(signatures) != 1:
        return None
    return candidates[0]


def _compound_pitch_candidates(
    word: str,
    full_signature: Sequence[str],
    inventory: _MorphologyInventory,
) -> list[
    tuple[
        tuple[int, int, int, int],
        tuple[str, str],
        _MorphologicalPitch,
    ]
]:
    """Return conservative recursive compound analyses for an unknown word."""

    attested = _attested_compound_segmentations(
        word,
        inventory.dictionary,
    )
    modes = [(False, 0)]
    if word.startswith("ga") and len(word) > len("ga"):
        modes.append((True, len("ga")))

    candidates: list[
        tuple[
            tuple[int, int, int, int],
            tuple[str, str],
            _MorphologicalPitch,
        ]
    ] = []
    for has_ga_prefix, start in modes:
        for parts in _compound_segmentations(word, start, inventory):
            members = tuple(
                part.stem for part in parts if part.stem is not None
            )
            if len(members) < (1 if has_ga_prefix else 2):
                continue
            if (
                not has_ga_prefix
                and len({stem.root_text for stem in members}) < 2
            ):
                # Productive reduplication has its own rules. Do not turn an
                # arbitrary repeated lexical form into a compound.
                continue

            values: list[str] = []
            components: list[AsaxiProsodyMorpheme] = []
            source_notes = [
                COMPOUND_GRAMMAR_SOURCE,
                COMPOUND_PROSODY_SOURCE,
            ]
            if has_ga_prefix:
                values.extend("L" for _ in af.split_morae("ga"))
                components.append(AsaxiProsodyMorpheme(
                    surface="ga",
                    lemma="ga-",
                    role="compound-prefix",
                    pitch_accent_class="atonal",
                    source_note=COMPOUND_GRAMMAR_SOURCE,
                    rule="ga-fusing-compound-prefix",
                ))

            member_index = 0
            for part in parts:
                if part.is_bridge:
                    components.append(AsaxiProsodyMorpheme(
                        surface=part.surface,
                        lemma=f"-{part.surface}-",
                        role="compound-bridge",
                        pitch_accent_class="atonal",
                        source_note=COMPOUND_GRAMMAR_SOURCE,
                        rule="documented-compound-bridge",
                    ))
                    continue

                stem = part.stem
                keep_accent = member_index == 0
                values.extend(
                    stem.values
                    if keep_accent
                    else tuple("L" for _ in stem.values)
                )
                is_head = member_index == len(members) - 1
                components.append(AsaxiProsodyMorpheme(
                    surface=stem.surface,
                    lemma=stem.surface,
                    role=("compound-head" if is_head else "compound-modifier"),
                    pitch_accent_class=(
                        stem.components[0].pitch_accent_class
                        if stem.components
                        else "lexical"
                    ),
                    source_note="; ".join(stem.source_notes),
                    rule=(
                        "compound-accent-bearing-member"
                        if keep_accent
                        else "compound-deaccented-member"
                    ),
                    children=(
                        stem.components
                        if stem.rule != "lexical-stem"
                        else ()
                    ),
                ))
                source_notes.extend(stem.source_notes)
                member_index += 1

            if len(values) != len(full_signature):
                continue
            candidate_segments = _flatten_compound_components(components)
            if attested:
                if candidate_segments not in attested:
                    continue
            elif not has_ga_prefix:
                # An arbitrary unknown string being tileable with two
                # dictionary words is not evidence that it is a compound.
                # Non-ga compounds require an attested interlinear or
                # explicitly documented segmentation.
                continue
            lexical_type = members[-1].lexical_type
            member_key = "|".join(
                f"{stem.root_text}:{stem.rule}" for stem in members
            )
            rank = (
                1,
                -len(members),
                sum(len(stem.surface) ** 2 for stem in members),
                sum(stem.rule_priority for stem in members),
            )
            candidates.append((
                rank,
                (
                    "ga-compound" if has_ga_prefix else "compound",
                    member_key,
                ),
                _MorphologicalPitch(
                    values=tuple(values),
                    lexical_type=lexical_type,
                    source_note="; ".join(dict.fromkeys(source_notes)),
                    components=tuple(components),
                ),
            ))
    return candidates


def _infer_morphological_pitch(
    surface: str,
    dictionary: af.AsaxiSynthesisDictionary,
) -> Optional[_MorphologicalPitch]:
    """Compose an unknown written form from traceable dictionary pieces.

    The inference is deliberately conservative. It searches typed lexical
    stems and their documented surface realizations, then requires every
    remaining character to be covered by an explicitly bound morpheme.
    Surface allomorph rules remain traceable to their grammar source, and
    free particles are never guessed from a coincidental word-edge match.
    """

    word = af.normalize_word(surface)
    full_signature = _mora_signature(word)
    if not full_signature:
        return None

    inventory = _morphology_inventory(dictionary)
    if word in inventory.analyses:
        return inventory.analyses[word]
    attested_segmentations = _attested_compound_segmentations(
        word,
        dictionary,
    )
    prefix_entries = inventory.prefixes
    suffix_entries = inventory.suffixes
    prefix_tokens = tuple(prefix_entries)
    suffix_tokens = tuple(suffix_entries)

    candidates: list[
        tuple[
            tuple[int, int, int, int],
            tuple[str, str],
            _MorphologicalPitch,
        ]
    ] = []
    for root_text, stem in inventory.stems:
        search_from = 0
        while True:
            root_start = word.find(stem.surface, search_from)
            if root_start < 0:
                break
            root_end = root_start + len(stem.surface)
            search_from = root_start + 1
            prefix_text = word[:root_start]
            suffix_text = word[root_end:]
            if (
                not prefix_text
                and not suffix_text
                and stem.rule == "lexical-stem"
            ):
                continue
            prefix = _segment_morphemes(prefix_text, prefix_tokens)
            suffix = _segment_morphemes(suffix_text, suffix_tokens)
            if prefix is None or suffix is None:
                continue
            if any(
                not _morpheme_accepts_host(
                    prefix_entries[component],
                    stem.lexical_type,
                )
                for component in prefix
            ) or any(
                not _morpheme_accepts_host(
                    suffix_entries[component],
                    stem.lexical_type,
                )
                for component in suffix
            ):
                continue

            root_accentable = sum(
                mora.accentable for mora in stem.root_moras
            )
            spread_root_high = (
                root_accentable == 1
                and any(value == "H" for value in stem.root_values)
            )
            values: list[str] = []
            components: list[AsaxiProsodyMorpheme] = []
            source_notes: list[str] = []
            for component in prefix:
                spec = prefix_entries[component]
                values.extend(_component_values(
                    spec,
                    spread_root_high=False,
                ))
                components.append(spec.public())
                if spec.source_note:
                    source_notes.append(spec.source_note)

            values.extend(stem.values)
            components.extend(stem.components)
            source_notes.extend(stem.source_notes)

            for component in suffix:
                spec = suffix_entries[component]
                values.extend(_component_values(
                    spec,
                    spread_root_high=spread_root_high,
                ))
                components.append(spec.public())
                if spec.source_note:
                    source_notes.append(spec.source_note)
            if len(values) != len(full_signature):
                continue
            if (
                attested_segmentations
                and _flatten_compound_components(components)
                not in attested_segmentations
            ):
                continue

            rank = (
                10 + stem.rule_priority,
                len(stem.root_moras),
                len(root_text),
                -len(components),
            )
            tie_break = (root_text, stem.rule)
            candidates.append((
                rank,
                tie_break,
                _MorphologicalPitch(
                    values=tuple(values),
                    lexical_type=stem.lexical_type,
                    source_note="; ".join(dict.fromkeys(source_notes)),
                    components=tuple(components),
                ),
            ))

    candidates.extend(
        _compound_pitch_candidates(word, full_signature, inventory)
    )

    if not candidates:
        attested = _attested_morphological_pitch(
            word,
            af.split_morae(word),
            inventory,
        )
        _cache_morphology_analysis(inventory, word, attested)
        return attested
    best_rank = max(row[0] for row in candidates)
    finalists = sorted(
        (row for row in candidates if row[0] == best_rank),
        key=lambda row: row[1],
    )
    selected = finalists[0][2]
    descriptions = tuple(dict.fromkeys(
        format_morpheme_analysis(row[2].components)
        for row in finalists[1:]
        if row[2].components != selected.components
    ))
    result = _MorphologicalPitch(
        values=selected.values,
        lexical_type=selected.lexical_type,
        source_note=selected.source_note,
        components=selected.components,
        alternatives=descriptions,
    )
    _cache_morphology_analysis(inventory, word, result)
    return result


def _standalone_morpheme_record(
    surface: str,
    dictionary: af.AsaxiSynthesisDictionary,
) -> Optional[Mapping[str, object]]:
    """Resolve a written bound form when the grammar licenses it alone."""

    candidates = dictionary.lookup_morpheme_surfaces(surface)
    if not candidates:
        return None
    signatures = {
        (
            tuple(str(phone) for phone in row.get("phones") or ()),
            str(row.get("pitch_accent") or ""),
            str(row.get("pitch_accent_class") or ""),
        )
        for row in candidates
    }
    if len(signatures) != 1:
        return None
    return candidates[0]


def analyze_utterance(
    text: str,
    dictionary: Optional[af.AsaxiSynthesisDictionary] = None,
    *,
    lexical_type_hints: Optional[Mapping[int | str, str]] = None,
    phone_overrides: Optional[Mapping[str, Sequence[str]]] = None,
    capitalized_pronunciations: Optional[
        Mapping[str, Sequence[str]]
    ] = None,
) -> AsaxiProsodyPlan:
    """Compose lexical accent, deaccenting, and boundary-tone state."""

    source = str(text or "")
    normalized = source.lower()
    dictionary = dictionary or load_dictionary()
    hints = dict(lexical_type_hints or {})
    overrides = {
        af.normalize_word(word): tuple(
            str(phone).strip()
            for phone in phones
            if str(phone).strip()
        )
        for word, phones in dict(phone_overrides or {}).items()
    }
    capitalized_phones = {
        af.normalize_word(word): tuple(
            str(phone).strip()
            for phone in phones
            if str(phone).strip()
        )
        for word, phones in dict(
            capitalized_pronunciations or {}
        ).items()
    }
    written_surfaces = af.words_in_text(
        source,
        reject_unsupported_letters=True,
        preserve_case=True,
    )
    surfaces = tuple(
        af.normalize_word(surface) for surface in written_surfaces
    )
    if not surfaces:
        raise ValueError("No Asaxi words were found.")

    diagnostics: list[AsaxiProsodyDiagnostic] = []
    resolved: list[dict[str, object]] = []
    phrase_overrides = _phrase_overrides(dictionary, surfaces)
    reported_expressions: set[str] = set()
    for word_index, surface in enumerate(surfaces):
        written_surface = written_surfaces[word_index]
        capitalized = af.is_capitalized_term(written_surface)
        entry = dictionary.lookup(surface)
        hint = hints.get(word_index, hints.get(surface, ""))
        morphemes: tuple[AsaxiProsodyMorpheme, ...] = ()
        if capitalized:
            phones = (
                capitalized_phones.get(surface)
                or overrides.get(surface)
            )
            if not phones:
                raise ValueError(
                    f"Capitalized term {written_surface!r} has no English "
                    "pronunciation. Add a Dictionary pronunciation override."
                )
            moras = _borrowed_term_moras(written_surface, phones)
            values = [
                "L" for _mora in moras
            ]
            selected = next(
                (
                    mora.index for mora in moras
                    if mora.accentable
                ),
                None,
            )
            if selected is not None:
                values[selected] = "H"
            pattern = af.canonical_pitch_pattern(values)
            lexical_type = "borrowed_term"
            accent_class = "borrowed_default"
            source_note = (
                "Grammar_Structure/"
                "00_Asaxi Orthography & Punctuation Standard.md"
            )
            if surface in capitalized_phones:
                diagnostics.append(AsaxiProsodyDiagnostic(
                    "capitalized_english_g2p",
                    (
                        f"{written_surface!r} used the English frontend "
                        "because it follows the full-cap term convention."
                    ),
                    "info",
                    word_index,
                ))
        elif entry is None:
            analysis = af.analyze_word(surface)
            if analysis.unknown_graphemes:
                raise ValueError(
                    f"Unsupported Asaxi graphemes in {surface!r}: "
                    + ", ".join(analysis.unknown_graphemes)
                )
            moras = analysis.moras
            indexed_morpheme = _standalone_morpheme_record(
                surface,
                dictionary,
            )
            if indexed_morpheme is not None:
                phones = tuple(
                    str(phone)
                    for phone in indexed_morpheme.get("phones") or ()
                )
                pattern = str(
                    indexed_morpheme.get("pitch_accent")
                    or af.default_pitch_pattern(surface, atonal=True)
                )
                lexical_type = str(
                    indexed_morpheme.get("lexical_type") or "particle"
                )
                accent_class = str(
                    indexed_morpheme.get("pitch_accent_class")
                    or "atonal"
                )
                source_notes = tuple(
                    str(note)
                    for note in indexed_morpheme.get("source_notes") or ()
                    if str(note)
                )
                source_note = source_notes[0] if source_notes else ""
                canonical = str(
                    indexed_morpheme.get("canonical_form") or surface
                )
                role = str(
                    indexed_morpheme.get("role")
                    or indexed_morpheme.get("attachment")
                    or "function-morpheme"
                )
                morphemes = (
                    AsaxiProsodyMorpheme(
                        surface=surface,
                        lemma=canonical,
                        role=role,
                        pitch_accent_class=accent_class,
                        source_note=source_note,
                        rule="standalone-grammar-morpheme",
                    ),
                )
                diagnostics.append(AsaxiProsodyDiagnostic(
                    "standalone_morpheme_resolution",
                    (
                        f"{surface!r} matched documented grammatical form "
                        f"{canonical!r}."
                    ),
                    "info",
                    word_index,
                ))
            else:
                phones = analysis.phones
                morphology = _infer_morphological_pitch(
                    surface,
                    dictionary,
                )
                if morphology is None:
                    pattern = af.default_pitch_pattern(surface)
                    lexical_type = ""
                    accent_class = "lexical"
                    source_note = ""
                    diagnostics.append(AsaxiProsodyDiagnostic(
                        "no_matching_lexical_units",
                        (
                            f"{surface!r} could not be fully segmented into "
                            "matching lexical units and bound morphemes; "
                            "regular G2P and default lexical accent were used."
                        ),
                        "warning",
                        word_index,
                    ))
                else:
                    pattern = af.canonical_pitch_pattern(morphology.values)
                    lexical_type = morphology.lexical_type
                    accent_class = "morphological"
                    source_note = morphology.source_note
                    morphemes = morphology.components
                    component_text = format_morpheme_analysis(
                        morphology.components
                    )
                    diagnostics.append(AsaxiProsodyDiagnostic(
                        "morphological_pitch_inference",
                        (
                            f"{surface!r} was composed transparently as "
                            f"{component_text}."
                        ),
                        "info",
                        word_index,
                    ))
                    if morphology.alternatives:
                        diagnostics.append(AsaxiProsodyDiagnostic(
                            "ambiguous_morphological_analysis",
                            (
                                f"{surface!r} also permits equally ranked "
                                "analysis: "
                                + "; ".join(morphology.alternatives)
                            ),
                            "warning",
                            word_index,
                        ))
        else:
            variant, explicitly_selected = _variant_payload(entry, str(hint))
            phones = tuple(str(phone) for phone in variant["phones"])
            pattern = str(variant["pitch_accent"])
            lexical_type = str(variant.get("lexical_type") or "")
            accent_class = str(
                variant.get("pitch_accent_class") or "lexical"
            )
            source_note = str(
                variant.get("source_note") or entry.source_note
            )
            moras = entry.moras
            if len(entry.variants) > 1 and not explicitly_selected:
                diagnostics.append(AsaxiProsodyDiagnostic(
                    "ambiguous_homograph_default",
                    (
                        f"{surface!r} has {len(entry.variants)} typed variants; "
                        f"the dictionary default {entry.pitch_accent!r} was used."
                    ),
                    "warning",
                    word_index,
                ))
        if surface in overrides:
            if not overrides[surface]:
                raise ValueError(
                    f"Phone override for {surface!r} is empty."
                )
            phones = overrides[surface]
            diagnostics.append(AsaxiProsodyDiagnostic(
                "user_g2p_override",
                (
                    f"{written_surface!r} used the active pronunciation "
                    "override."
                ),
                "info",
                word_index,
            ))
        parsed = af.parse_pitch_pattern(pattern)
        if len(parsed) != 1:
            raise ValueError(f"{surface!r} has a non-word pitch pattern")
        values = parsed[0]
        if values and len(values) != len(moras):
            raise ValueError(
                f"{surface!r} has {len(moras)} morae but pattern {pattern!r}"
            )
        phrase_expression = ""
        phrase_source = ""
        if not capitalized and word_index in phrase_overrides:
            candidate, phrase_expression, phrase_source = (
                phrase_overrides[word_index]
            )
            if len(candidate) != len(moras):
                raise ValueError(
                    f"Phrase accent for {surface!r} has the wrong mora count"
                )
            values = candidate
            pattern = af.canonical_pitch_pattern(values)
            accent_class = "phrase"
            if phrase_expression not in reported_expressions:
                diagnostics.append(AsaxiProsodyDiagnostic(
                    "phrase_dictionary_override",
                    (
                        f"Expression {phrase_expression!r} used its "
                        "dictionary accent."
                    ),
                    "info",
                    word_index,
                ))
                reported_expressions.add(phrase_expression)
        resolved.append({
            "surface": written_surface,
            "lexical_type": lexical_type,
            "phones": phones,
            "moras": moras,
            "mora_phones": _partition_phones(moras, phones),
            "values": list(values),
            "pitch_accent": pattern,
            "pitch_accent_class": accent_class,
            "source_note": source_note,
            "phrase_expression": phrase_expression,
            "phrase_source": phrase_source,
            "morphemes": morphemes,
        })

    mark = _boundary_mark(normalized)
    interrogative = mark == "?"
    directive = mark == "!" or surfaces[-1] in APPEAL_PARTICLES
    insistent = surfaces[-1] in INSISTENT_TAILS
    if insistent:
        boundary_tone = "H%"
    elif interrogative or directive:
        boundary_tone = "LH%"
    elif mark in {",", ":", ";"}:
        boundary_tone = "H-"
    else:
        boundary_tone = "L%"
    flat_question = (
        interrogative
        and not any(word in QUESTION_PARTICLES for word in surfaces)
    )
    wh_index = next(
        (index for index, word in enumerate(surfaces) if word in WH_WORDS),
        None,
    )
    if flat_question:
        for item in resolved:
            item["values"] = ["L"] * len(item["moras"])
        if wh_index is not None:
            wh_values = resolved[wh_index]["values"]
            wh_moras = resolved[wh_index]["moras"]
            selected = next(
                (
                    index for index, mora in enumerate(wh_moras)
                    if mora.accentable
                ),
                None,
            )
            if selected is not None:
                wh_values[selected] = "H"

    accentable_positions = [
        (word_index, mora_index)
        for word_index, item in enumerate(resolved)
        for mora_index, mora in enumerate(item["moras"])
        if mora.accentable
    ]

    # Ordinary assertions receive an onset H. If that H replaces an atonal L,
    # the immediately following mora is downstepped, including a dominant one.
    if not interrogative and not directive and not insistent:
        selected = next(
            (
                (word_index, mora_index)
                for word_index, item in enumerate(resolved)
                for mora_index, mora in enumerate(item["moras"])
                if mora.accentable
            ),
            None,
        )
        if selected is not None:
            boundary_high_was_inserted = (
                resolved[selected[0]]["values"][selected[1]] != "H"
            )
            resolved[selected[0]]["values"][selected[1]] = "H"
            if boundary_high_was_inserted:
                all_positions = [
                    (word_index, mora_index)
                    for word_index, item in enumerate(resolved)
                    for mora_index, _ in enumerate(item["moras"])
                ]
                selected_index = all_positions.index(selected)
                if selected_index + 1 < len(all_positions):
                    downstepped = all_positions[selected_index + 1]
                    resolved[downstepped[0]]["values"][downstepped[1]] = "L"
    elif interrogative:
        for item in resolved:
            for mora_index, mora in enumerate(item["moras"]):
                if mora.accentable:
                    item["values"][mora_index] = "L"
                    break
            else:
                continue
            break
        if wh_index is not None:
            selected = next(
                (
                    index
                    for index, mora in enumerate(resolved[wh_index]["moras"])
                    if mora.accentable
                ),
                None,
            )
            if selected is not None:
                resolved[wh_index]["values"][selected] = "H"

    # The initial vocative particle carries the call; names before the first
    # comma are deaccented. The following clause retains its own contour.
    comma_index = normalized.find(",")
    vocative_word_count = (
        len(af.words_in_text(normalized[:comma_index]))
        if comma_index >= 0
        else 0
    )
    if surfaces[0] == "ăjo" and vocative_word_count >= 2:
        lexical_vocative = af.parse_pitch_pattern(
            resolved[0]["pitch_accent"]
        )[0]
        resolved[0]["values"] = list(lexical_vocative)
        for word_index in range(1, vocative_word_count):
            resolved[word_index]["values"] = [
                "L" for _ in resolved[word_index]["moras"]
            ]
        diagnostics.append(AsaxiProsodyDiagnostic(
            "vocative_name_deaccenting",
            "The initial ăjo phrase retained its call contour and deaccented "
            "the following name.",
            "info",
            0,
        ))

    if insistent:
        for index in range(len(resolved[-1]["values"])):
            if resolved[-1]["moras"][index].accentable:
                resolved[-1]["values"][index] = "H"
    elif boundary_tone == "L%" and len(accentable_positions) > 1:
        current_accentable = [
            str(resolved[word_index]["values"][mora_index])
            for word_index, mora_index in accentable_positions
        ]
        # A lexical or dominant H plateau falls at the boundary after its
        # final mora. Isolated phrase-final H does not form such a plateau.
        if current_accentable[-2:] != ["H", "H"]:
            final_word, final_mora = accentable_positions[-1]
            resolved[final_word]["values"][final_mora] = "L"

    words: list[AsaxiProsodyWord] = []
    moras_out: list[AsaxiProsodyMora] = []
    phones_out: list[str] = []
    for word_index, item in enumerate(resolved):
        mora_start = len(moras_out)
        phone_cursor = len(phones_out)
        word_phones = tuple(item["phones"])
        phones_out.extend(word_phones)
        local_cursor = phone_cursor
        for mora, mora_phones, lexical_pitch, surface_pitch in zip(
            item["moras"],
            item["mora_phones"],
            af.parse_pitch_pattern(item["pitch_accent"])[0],
            item["values"],
        ):
            local_end = local_cursor + len(mora_phones)
            moras_out.append(AsaxiProsodyMora(
                index=len(moras_out),
                word_index=word_index,
                word=str(item["surface"]),
                text=mora.text,
                phones=tuple(mora_phones),
                phone_start=local_cursor,
                phone_end=local_end,
                lexical_pitch=lexical_pitch,
                pitch=str(surface_pitch),
                accentable=mora.accentable,
                kind=mora.kind,
            ))
            local_cursor = local_end
        if local_cursor != phone_cursor + len(word_phones):
            raise ValueError(
                f"Phone partition for {item['surface']!r} is inconsistent"
            )
        words.append(AsaxiProsodyWord(
            index=word_index,
            surface=str(item["surface"]),
            lexical_type=str(item["lexical_type"]),
            phones=word_phones,
            pitch_accent=str(item["pitch_accent"]),
            pitch_accent_class=str(item["pitch_accent_class"]),
            mora_start=mora_start,
            mora_end=len(moras_out),
            dictionary_source=str(item["source_note"]),
            phrase_expression=str(item["phrase_expression"]),
            phrase_source=str(item["phrase_source"]),
            morphemes=tuple(item["morphemes"]),
        ))

    return AsaxiProsodyPlan(
        source_text=source,
        normalized_text=normalized,
        words=tuple(words),
        moras=tuple(moras_out),
        phones=tuple(phones_out),
        boundary_mark=mark,
        boundary_tone=boundary_tone,
        interrogative=interrogative,
        directive=directive,
        dictionary_ruleset=dictionary.ruleset,
        diagnostics=tuple(diagnostics),
    )


def _segment_row(segment) -> tuple[str, float, float]:
    if hasattr(segment, "phone"):
        return (
            str(segment.phone),
            float(segment.start),
            float(segment.end),
        )
    return str(segment[0]), float(segment[1]), float(segment[2])


def _expected_to_rendered(
    expected: Sequence[str],
    rendered: Sequence[str],
) -> tuple[dict[int, int], float]:
    if tuple(expected) == tuple(rendered):
        return {index: index for index in range(len(expected))}, 1.0
    matcher = SequenceMatcher(
        None, tuple(expected), tuple(rendered), autojunk=False
    )
    mapping: dict[int, int] = {}
    matched = 0
    for left, right, size in matcher.get_matching_blocks():
        for offset in range(size):
            mapping[left + offset] = right + offset
        matched += size
    denominator = max(1, len(expected), len(rendered))
    return mapping, matched / denominator


def rendered_morae(
    plans: AsaxiProsodyPlan | Sequence[AsaxiProsodyPlan],
    segments: Iterable,
) -> tuple[
    tuple[AsaxiRenderedMora, ...],
    tuple[AsaxiProsodyDiagnostic, ...],
]:
    """Align one or more phrase plans to a final Festival segment timeline.

    Phrase-local mora indexes are deliberately replaced with sentence-level
    indexes. Pause segments remain outside the phone matcher, so the mapping
    remains stable when a phrase boundary uses one, two, or four ``pau``
    regions.
    """

    plan_rows = (
        (plans,) if isinstance(plans, AsaxiProsodyPlan)
        else tuple(plans)
    )
    rows = [_segment_row(segment) for segment in segments]
    spoken = [
        (index, row)
        for index, row in enumerate(rows)
        if row[0] not in {"pau", "sil", "#"}
    ]
    expected: list[str] = []
    phrase_phone_offsets: list[int] = []
    diagnostics: list[AsaxiProsodyDiagnostic] = []
    for plan in plan_rows:
        phrase_phone_offsets.append(len(expected))
        expected.extend(plan.phones)
        diagnostics.extend(plan.diagnostics)

    mapping, coverage = _expected_to_rendered(
        expected, [row[1][0] for row in spoken]
    )
    if coverage < 0.999:
        diagnostics.append(AsaxiProsodyDiagnostic(
            "festival_phone_alignment",
            (
                "Dictionary/Festival phone alignment covered "
                f"{coverage:.1%} of the utterance."
            ),
            "warning" if coverage >= 0.85 else "error",
        ))
    if coverage < 0.85:
        raise ValueError(
            "Asaxi dictionary phones do not align with Festival's returned "
            f"segments ({coverage:.1%} coverage)."
        )

    aligned: list[AsaxiRenderedMora] = []
    for phrase_index, plan in enumerate(plan_rows):
        phone_offset = phrase_phone_offsets[phrase_index]
        for mora in plan.moras:
            spoken_indices = [
                mapping[phone_offset + phone_index]
                for phone_index in range(mora.phone_start, mora.phone_end)
                if phone_offset + phone_index in mapping
            ]
            segment_indices = tuple(
                spoken[index][0] for index in spoken_indices
            )
            if segment_indices:
                start = min(rows[index][1] for index in segment_indices)
                end = max(rows[index][2] for index in segment_indices)
            else:
                start = end = None
                if mora.phone_end > mora.phone_start:
                    diagnostics.append(AsaxiProsodyDiagnostic(
                        "mora_without_rendered_phone",
                        f"Mora {mora.text!r} has no aligned Festival phone.",
                        "warning",
                        mora.word_index,
                    ))
            aligned.append(AsaxiRenderedMora(
                index=len(aligned),
                phrase_index=phrase_index,
                local_mora_index=mora.index,
                word_index=mora.word_index,
                word=mora.word,
                text=mora.text,
                phones=mora.phones,
                pitch=mora.pitch,
                lexical_pitch=mora.lexical_pitch,
                accentable=mora.accentable,
                kind=mora.kind,
                segment_indices=segment_indices,
                start=start,
                end=end,
            ))
    return tuple(aligned), tuple(diagnostics)


def targets_for_plans(
    plans: AsaxiProsodyPlan | Sequence[AsaxiProsodyPlan],
    segments: Iterable,
    *,
    base_pitch_hz: float = 160.0,
    fall_percent: float = 18.0,
    mora_tone_overrides: Optional[Mapping[object, object]] = None,
    mora_pitch_offsets_cents: Optional[Mapping[object, object]] = None,
) -> tuple[list[tuple[float, float]], tuple[AsaxiProsodyDiagnostic, ...]]:
    """Map phrase plans onto one duration-sensitive sentence contour.

    This compatibility API returns only Festival targets and diagnostics.
    Call :func:`realize_pitch_for_plans` when the diagnostic trace is also
    required.
    """
    realization, diagnostics = realize_pitch_for_plans(
        plans,
        segments,
        base_pitch_hz=base_pitch_hz,
        fall_percent=fall_percent,
        mora_tone_overrides=mora_tone_overrides,
        mora_pitch_offsets_cents=mora_pitch_offsets_cents,
    )
    return list(realization.targets), diagnostics


def realize_pitch_for_plans(
    plans: AsaxiProsodyPlan | Sequence[AsaxiProsodyPlan],
    segments: Iterable,
    *,
    base_pitch_hz: float = 160.0,
    fall_percent: float = 18.0,
    mora_tone_overrides: Optional[Mapping[object, object]] = None,
    mora_pitch_offsets_cents: Optional[Mapping[object, object]] = None,
    model: Optional[asaxi_pitch_domain.AsaxiPitchModel] = None,
) -> tuple[
    asaxi_pitch_domain.AsaxiPitchRealization,
    tuple[AsaxiProsodyDiagnostic, ...],
]:
    """Realize one final segment timeline and retain its prosody trace."""
    plan_rows = (
        (plans,) if isinstance(plans, AsaxiProsodyPlan)
        else tuple(plans)
    )
    rows = [_segment_row(segment) for segment in segments]
    aligned, diagnostics = rendered_morae(plan_rows, rows)
    realization = asaxi_pitch_domain.realize_pitch(
        plan_rows,
        aligned,
        base_pitch_hz=base_pitch_hz,
        fall_percent=fall_percent,
        mora_tone_overrides=mora_tone_overrides,
        mora_pitch_offsets_cents=mora_pitch_offsets_cents,
        model=model,
    )
    return realization, diagnostics


def targets_for_segments(
    plan: AsaxiProsodyPlan,
    segments: Iterable,
    *,
    base_pitch_hz: float = 160.0,
    fall_percent: float = 18.0,
    mora_tone_overrides: Optional[Mapping[object, object]] = None,
    mora_pitch_offsets_cents: Optional[Mapping[object, object]] = None,
) -> tuple[list[tuple[float, float]], tuple[AsaxiProsodyDiagnostic, ...]]:
    """Backward-compatible one-plan target helper."""

    return targets_for_plans(
        plan,
        segments,
        base_pitch_hz=base_pitch_hz,
        fall_percent=fall_percent,
        mora_tone_overrides=mora_tone_overrides,
        mora_pitch_offsets_cents=mora_pitch_offsets_cents,
    )
