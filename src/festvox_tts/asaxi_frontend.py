# -*- coding: utf-8 -*-
"""Canonical Asaxi word analysis shared by the vault and FestVox runtime.

The module deliberately keeps three identities separate:

* romanized graphemes, as stored in the Markdown vault;
* morae, which carry lexical pitch targets;
* bank phones, which are the existing ARPAsing-compatible synthesis symbols.

Pitch patterns use one ``H`` or ``L`` value per mora, separated by dots.
Multiword expressions may separate word patterns with ``|``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


ASAXI_G2P_RULES = (
    ("nŋ", ("nng",)), ("nn", ("nn",)), ("mm", ("mm",)),
    ("ch", ("ch",)), ("sh", ("sh",)), ("dh", ("dh",)),
    ("jh", ("jh",)), ("zh", ("zh",)), ("th", ("th",)),
    ("dz", ("dz",)), ("si", ("sh", "i")), ("ni", ("ny", "i")),
    ("å", ("a", "w")), ("ă", ("a", "y")), ("ë", ("e", "y")),
    ("ỏ", ("o", "w")), ("ő", ("o", "y")), ("ů", ("u", "w")),
    ("è", ("ax",)), ("ě", ("er",)), ("ý", ("ih",)),
    ("ù", ("u",)), ("á", ("ao",)), ("a", ("a",)),
    ("e", ("e",)), ("i", ("i",)), ("o", ("o",)),
    ("u", ("u",)), ("ŕ", ("dx",)), ("ń", ("ny",)),
    ("ś", ("sh",)), ("ŋ", ("ng",)), ("'", ("q",)),
    ("x", ("hh",)), ("c", ("ts",)), ("j", ("y",)),
    ("b", ("b",)), ("d", ("d",)), ("f", ("f",)),
    ("g", ("g",)), ("h", ("h",)), ("k", ("k",)),
    ("l", ("l",)), ("m", ("m",)), ("n", ("n",)),
    ("p", ("p",)), ("r", ("r",)), ("s", ("s",)),
    ("t", ("t",)), ("v", ("v",)), ("w", ("w",)),
    ("y", ("y",)), ("z", ("z",)),
)

_RULE_MAP = dict(ASAXI_G2P_RULES)
_GRAPHEMES = tuple(
    sorted(_RULE_MAP, key=lambda item: (-len(item), item))
)
VOWEL_GRAPHEMES = frozenset(
    {"a", "e", "i", "o", "u", "á", "ă", "å", "è", "ë", "ě",
     "ỏ", "ő", "ù", "ů", "ý"}
)
MORA_NUCLEUS_GRAPHEMES = frozenset(
    grapheme
    for grapheme in _GRAPHEMES
    if grapheme[-1:] in VOWEL_GRAPHEMES
)
SYLLABIC_NASALS = frozenset({"mm", "nn", "nŋ"})
DOTTED_GEMINATE_NASALS = frozenset({"m", "n"})
STOP_GRAPHEMES = frozenset(
    {"p", "t", "k", "b", "d", "g", "ch", "c", "dz", "jh"}
)
PALATAL_PHONES = frozenset(
    consonant + "y"
    for consonant in
    "b d g k m n p r t h l v f ng dx".split()
)
ASAXI_WORD_RE = re.compile(
    r"[a-záăåèëěỏőùůýŋŕśń']+(?:\.[a-záăåèëěỏőùůýŋŕśń']+)*",
    re.IGNORECASE,
)
_PITCH_TOKEN_RE = re.compile(r"^[HL](?:\.[HL])*$")

# Full-cap tokens are an explicit orthographic language switch inside Asaxi
# text. Most terms use the ordinary English frontend; this small table records
# attested project pronunciations whose dialectal phone choice differs from
# the default Festival/CMU entry. Project/user dictionary entries still have
# final authority at synthesis time.
CAPITALIZED_ENGLISH_PRONUNCIATION_OVERRIDES = {
    "JOHN": ("jh", "ao", "n"),
}


@dataclass(frozen=True)
class AsaxiGrapheme:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class AsaxiMora:
    """One phonological beat.

    ``accentable`` is false for syllabic nasals and geminate closures. Those
    morae still occupy time, but the lexical H target skips to a voiced mora.
    """

    index: int
    text: str
    phones: tuple[str, ...]
    start: int
    end: int
    accentable: bool = True
    kind: str = "ordinary"


@dataclass(frozen=True)
class AsaxiWordAnalysis:
    surface: str
    phones: tuple[str, ...]
    moras: tuple[AsaxiMora, ...]
    unknown_graphemes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AsaxiLexiconEntry:
    word: str
    phones: tuple[str, ...]
    moras: tuple[AsaxiMora, ...]
    pitch_accent: str
    pitch_accent_class: str
    g2p_override: bool = False
    source_note: str = ""
    source_notes: tuple[str, ...] = ()
    variants: tuple[Mapping[str, object], ...] = ()

    @property
    def pitch_values(self) -> tuple[str, ...]:
        words = parse_pitch_pattern(self.pitch_accent)
        if len(words) != 1:
            raise ValueError(
                f"{self.word!r} is a word entry but has a phrase pattern"
            )
        return words[0]


@dataclass(frozen=True)
class AsaxiSynthesisDictionary:
    schema_version: int
    ruleset: str
    entries: Mapping[str, AsaxiLexiconEntry]
    phrases: Mapping[str, Mapping[str, object]]
    source_summary: Mapping[str, object]
    morphemes: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )
    morphological_analyses: Mapping[
        str, Mapping[str, object]
    ] = field(default_factory=dict)

    def lookup(self, word: str) -> Optional[AsaxiLexiconEntry]:
        return self.entries.get(normalize_word(word))

    def lookup_variants(
        self,
        word: str,
    ) -> tuple[Mapping[str, object], ...]:
        entry = self.lookup(word)
        return entry.variants if entry is not None else ()

    def lookup_morpheme(
        self,
        form: str,
    ) -> Optional[Mapping[str, object]]:
        return self.morphemes.get(normalize_word(form))

    def lookup_morpheme_surfaces(
        self,
        surface: str,
    ) -> tuple[Mapping[str, object], ...]:
        """Return canonical grammar records sharing a written surface form."""

        requested = normalize_word(surface).strip("-")
        return tuple(
            record
            for _form, record in sorted(self.morphemes.items())
            if normalize_word(
                str(record.get("surface") or _form.strip("-"))
            ) == requested
        )

    def lookup_morphology(
        self,
        word: str,
    ) -> Optional[Mapping[str, object]]:
        return self.morphological_analyses.get(normalize_word(word))

    def pronunciations(self) -> dict[str, list[str]]:
        return {
            word: list(entry.phones)
            for word, entry in self.entries.items()
        }


def normalize_word(value: str) -> str:
    return unicodedata.normalize("NFC", str(value or "").strip().lower())


def is_capitalized_term(value: str) -> bool:
    """Return whether a written token uses Asaxi's full-cap term convention."""

    token = unicodedata.normalize("NFC", str(value or "").strip())
    letters = [character for character in token if character.isalpha()]
    return (
        bool(letters)
        and any(character.isupper() for character in letters)
        and not any(character.islower() for character in letters)
    )


def words_in_text(
    value: str,
    *,
    reject_unsupported_letters: bool = False,
    preserve_case: bool = False,
) -> tuple[str, ...]:
    """Return Asaxi word tokens without silently dropping foreign graphemes.

    Punctuation and whitespace may occur between words. In strict mode, any
    Unicode letter or combining mark outside a recognized token is rejected;
    this prevents a typo such as ``ū`` from turning one word into two partial
    dictionary lookups. ``preserve_case`` exposes the original token spelling
    to mixed-language frontends; dictionary callers retain the historical
    lowercase result by default.
    """

    normalized = unicodedata.normalize("NFC", str(value or "").strip())
    if not preserve_case:
        normalized = normalized.lower()
    matches = tuple(ASAXI_WORD_RE.finditer(normalized))
    if reject_unsupported_letters:
        covered = [False] * len(normalized)
        for match in matches:
            for index in range(match.start(), match.end()):
                covered[index] = True
        unsupported = []
        for index, character in enumerate(normalized):
            category = unicodedata.category(character)
            if (
                not covered[index]
                and (character.isalpha() or category.startswith("M"))
                and character not in unsupported
            ):
                unsupported.append(character)
        if unsupported:
            raise ValueError(
                "Unsupported Asaxi grapheme(s): " + ", ".join(unsupported)
            )
    return tuple(match.group(0) for match in matches)


def capitalized_terms_in_text(value: str) -> tuple[str, ...]:
    """Return distinct full-cap terms in source order, preserving spelling."""

    result = []
    seen = set()
    for token in words_in_text(value, preserve_case=True):
        key = unicodedata.normalize("NFC", token)
        if is_capitalized_term(key) and key not in seen:
            result.append(key)
            seen.add(key)
    return tuple(result)


def tokenize_graphemes(word: str) -> tuple[AsaxiGrapheme, ...]:
    """Longest-match tokenization retaining source character offsets."""

    value = normalize_word(word)
    result: list[AsaxiGrapheme] = []
    cursor = 0
    while cursor < len(value):
        if value[cursor] in {"-", ".", " "}:
            result.append(AsaxiGrapheme(
                value[cursor], cursor, cursor + 1))
            cursor += 1
            continue
        match = next(
            (item for item in _GRAPHEMES
             if value.startswith(item.lower(), cursor)),
            None,
        )
        if match is None:
            result.append(AsaxiGrapheme(
                value[cursor], cursor, cursor + 1))
            cursor += 1
            continue
        result.append(AsaxiGrapheme(
            value[cursor:cursor + len(match)], cursor, cursor + len(match)))
        cursor += len(match)
    return tuple(result)


def _is_dotted_nasal_separator(
    tokens: Sequence[AsaxiGrapheme],
    index: int,
) -> bool:
    """Return whether ``tokens[index]`` separates an Asaxi nasal geminate.

    Asaxi distinguishes syllabic ``mm``/``nn`` from true geminates
    ``m.m``/``n.n``. The dot is therefore phonological structure rather than
    discardable punctuation: the left nasal closes one syllable and the
    right nasal begins the next.
    """

    if index <= 0 or index + 1 >= len(tokens):
        return False
    left = tokens[index - 1].text
    return (
        tokens[index].text == "."
        and left in DOTTED_GEMINATE_NASALS
        and tokens[index + 1].text == left
    )


def _phones_for_tokens(tokens: Iterable[AsaxiGrapheme]) -> tuple[str, ...]:
    phones: list[str] = []
    token_list = list(tokens)
    raw: list[tuple[str, bool]] = []
    for index, token in enumerate(token_list):
        if token.text in {"-", ".", " "}:
            continue
        follows_dotted_nasal = (
            index >= 2
            and _is_dotted_nasal_separator(token_list, index - 1)
        )
        raw.append((token.text, follows_dotted_nasal))
    cursor = 0
    while cursor < len(raw):
        grapheme, preserve_repeat = raw[cursor]
        mapped = list(_RULE_MAP.get(grapheme, ()))
        if not mapped:
            cursor += 1
            continue
        if (phones and mapped[0] == phones[-1]
                and grapheme not in VOWEL_GRAPHEMES
                and grapheme not in SYLLABIC_NASALS
                and not preserve_repeat):
            if grapheme in STOP_GRAPHEMES:
                phones[-1] = "cl"
                phones.extend(mapped)
            cursor += 1
            continue
        phones.extend(mapped)
        cursor += 1

    palatalized: list[str] = []
    cursor = 0
    while cursor < len(phones):
        if (cursor + 2 < len(phones)
                and phones[cursor + 1] == "y"
                and phones[cursor] + "y" in PALATAL_PHONES):
            palatalized.append(phones[cursor] + "y")
            cursor += 2
        else:
            palatalized.append(phones[cursor])
            cursor += 1
    return tuple(palatalized)


def g2p_asaxi(word: str) -> tuple[str, ...]:
    return _phones_for_tokens(tokenize_graphemes(word))


def split_morae(word: str) -> tuple[AsaxiMora, ...]:
    """Split one written word into morae without guessing syllable stress."""

    tokens = list(tokenize_graphemes(word))
    moras: list[AsaxiMora] = []
    pending: list[AsaxiGrapheme] = []

    def append_mora(
        owned: list[AsaxiGrapheme],
        *,
        accentable: bool = True,
        kind: str = "ordinary",
        phones: Optional[tuple[str, ...]] = None,
    ) -> None:
        if not owned:
            return
        start = owned[0].start
        end = owned[-1].end
        moras.append(AsaxiMora(
            index=len(moras),
            text="".join(item.text for item in owned),
            phones=(
                _phones_for_tokens(owned) if phones is None else phones
            ),
            start=start,
            end=end,
            accentable=accentable,
            kind=kind,
        ))

    def extend_previous_mora(owned: list[AsaxiGrapheme]) -> None:
        if not owned:
            return
        if not moras:
            append_mora(
                owned,
                accentable=False,
                kind="nonvocalic",
            )
            return
        previous = moras[-1]
        moras[-1] = AsaxiMora(
            previous.index,
            previous.text + "".join(item.text for item in owned),
            previous.phones + _phones_for_tokens(owned),
            previous.start,
            owned[-1].end,
            previous.accentable,
            previous.kind,
        )

    cursor = 0
    while cursor < len(tokens):
        token = tokens[cursor]
        text = token.text
        if text == "." and _is_dotted_nasal_separator(tokens, cursor):
            # The first nasal is the coda of the preceding syllable; the
            # second remains pending as the onset of the following syllable.
            # Keeping both phones distinguishes true m.m/n.n gemination from
            # the syllabic nuclei written without a dot as mm/nn.
            extend_previous_mora(pending)
            pending = []
            cursor += 1
            continue
        if text in {"-", ".", " "}:
            cursor += 1
            continue
        if text == "'":
            if moras:
                previous = moras[-1]
                moras[-1] = AsaxiMora(
                    previous.index,
                    previous.text + text,
                    previous.phones + ("q",),
                    previous.start,
                    token.end,
                    previous.accentable,
                    previous.kind,
                )
            else:
                pending.append(token)
            cursor += 1
            continue
        if (cursor + 1 < len(tokens)
                and text == tokens[cursor + 1].text
                and text not in VOWEL_GRAPHEMES
                and text not in SYLLABIC_NASALS):
            if pending:
                extend_previous_mora(pending)
                pending = []
            append_mora(
                [token],
                accentable=False,
                kind="geminate",
                phones=("cl",) if text in STOP_GRAPHEMES else (),
            )
            cursor += 1
            continue
        if text in SYLLABIC_NASALS:
            owned = pending + [token]
            pending = []
            append_mora(
                owned, accentable=False, kind="syllabic_nasal")
            cursor += 1
            continue
        # Some atomic G2P graphemes include both an onset and their vowel
        # nucleus (currently ``ni`` and ``si``). They finish a mora just as a
        # one-character vowel does; otherwise the next onset is incorrectly
        # absorbed into the same block (for example ``nihè``).
        if text in MORA_NUCLEUS_GRAPHEMES:
            owned = pending + [token]
            pending = []
            append_mora(owned)
            cursor += 1
            continue
        pending.append(token)
        cursor += 1

    if pending:
        if moras:
            extend_previous_mora(pending)
        else:
            append_mora(pending, accentable=False, kind="nonvocalic")
    return tuple(moras)


def analyze_word(word: str) -> AsaxiWordAnalysis:
    value = normalize_word(word)
    tokens = tokenize_graphemes(value)
    unknown = tuple(sorted({
        token.text for token in tokens
        if token.text not in _RULE_MAP
        and token.text not in {"-", ".", " "}
    }))
    return AsaxiWordAnalysis(
        surface=value,
        phones=g2p_asaxi(value),
        moras=split_morae(value),
        unknown_graphemes=unknown,
    )


def canonical_pitch_pattern(values: Iterable[str]) -> str:
    normalized = tuple(
        str(value).strip().upper() for value in values)
    if not normalized:
        return "none"
    pattern = ".".join(normalized)
    if not _PITCH_TOKEN_RE.fullmatch(pattern):
        raise ValueError(f"invalid Asaxi pitch pattern: {pattern!r}")
    return pattern


def parse_pitch_pattern(pattern: str) -> tuple[tuple[str, ...], ...]:
    """Parse canonical word or phrase notation into H/L values."""

    chunks = [chunk.strip() for chunk in str(pattern or "").split("|")]
    if not chunks or any(not chunk for chunk in chunks):
        raise ValueError(f"invalid Asaxi pitch pattern: {pattern!r}")
    parsed = []
    for chunk in chunks:
        normalized = chunk.replace(" ", "").upper()
        if normalized == "NONE":
            parsed.append(())
            continue
        if not _PITCH_TOKEN_RE.fullmatch(normalized):
            raise ValueError(f"invalid Asaxi pitch pattern: {pattern!r}")
        parsed.append(tuple(normalized.split(".")))
    return tuple(parsed)


def default_pitch_pattern(
    word: str,
    *,
    atonal: bool = False,
    accent_mora: Optional[int] = None,
) -> str:
    """Return a conservative rule-default pattern for one word.

    ``accent_mora`` is zero-based. When omitted, the first accentable mora
    receives H. Atonal words are entirely L.
    """

    moras = split_morae(word)
    if not moras:
        raise ValueError(f"no Asaxi morae found in {word!r}")
    values = ["L"] * len(moras)
    if not atonal:
        selected = accent_mora
        if selected is None:
            selected = next(
                (mora.index for mora in moras if mora.accentable),
                None,
            )
        if selected is not None and 0 <= int(selected) < len(values):
            values[int(selected)] = "H"
    return canonical_pitch_pattern(values)


def _decode_mora(index: int, payload: Mapping[str, object]) -> AsaxiMora:
    return AsaxiMora(
        index=index,
        text=str(payload.get("text") or ""),
        phones=tuple(str(phone) for phone in payload.get("phones") or ()),
        start=int(payload.get("start") or 0),
        end=int(payload.get("end") or 0),
        accentable=bool(payload.get("accentable", True)),
        kind=str(payload.get("kind") or "ordinary"),
    )


def _pitch_fits_moras(
    parsed: tuple[tuple[str, ...], ...],
    moras: tuple[AsaxiMora, ...],
) -> bool:
    if len(parsed) != 1:
        return False
    if parsed[0]:
        return len(parsed[0]) == len(moras)
    return bool(moras) and all(not mora.accentable for mora in moras)


def _decode_dictionary_variant(
    word: str,
    payload: Mapping[str, object],
    moras: tuple[AsaxiMora, ...],
) -> dict[str, object]:
    lexical_type = str(payload.get("lexical_type") or "").strip().casefold()
    if not lexical_type:
        raise ValueError(
            f"dictionary variant for {word!r} has no lexical type"
        )
    phones = tuple(
        str(phone).strip()
        for phone in payload.get("phones") or ()
        if str(phone).strip()
    )
    if not phones:
        raise ValueError(
            f"dictionary variant for {word!r} has no synthesis phones"
        )
    g2p_override = bool(
        payload.get("g2p_override", False)
        or str(payload.get("g2p_source") or "").casefold() == "override"
    )
    if not g2p_override and phones != g2p_asaxi(word):
        raise ValueError(
            f"dictionary variant for {word!r} has inconsistent G2P data"
        )
    pattern = str(payload.get("pitch_accent") or "")
    parsed = parse_pitch_pattern(pattern)
    if not _pitch_fits_moras(parsed, moras):
        raise ValueError(
            f"dictionary variant for {word!r} has {len(moras)} morae but "
            f"pitch pattern {pattern!r}"
        )
    source_notes = tuple(
        str(note)
        for note in payload.get("source_notes") or ()
        if str(note)
    )
    return {
        **dict(payload),
        "lexical_type": lexical_type,
        "phones": phones,
        "g2p_source": "override" if g2p_override else "rules",
        "g2p_override": g2p_override,
        "pitch_accent": canonical_pitch_pattern(parsed[0]),
        "pitch_accent_class": str(
            payload.get("pitch_accent_class") or "lexical"
        ),
        "source_notes": source_notes,
        "source_note": str(payload.get("source_note") or ""),
    }


def load_synthesis_dictionary(
    path: str | Path,
) -> AsaxiSynthesisDictionary:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if int(payload.get("schema_version") or 0) != 1:
        raise ValueError("unsupported Asaxi synthesis dictionary schema")
    if str(payload.get("language") or "") != "asaxi":
        raise ValueError("Asaxi dictionary language must be 'asaxi'")
    entries: dict[str, AsaxiLexiconEntry] = {}
    for raw_word, raw_entry in dict(payload.get("entries") or {}).items():
        word = normalize_word(raw_word)
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"dictionary entry {word!r} is not an object")
        moras = tuple(
            _decode_mora(index, mora)
            for index, mora in enumerate(raw_entry.get("moras") or ())
        )
        pattern = str(raw_entry.get("pitch_accent") or "")
        parsed = parse_pitch_pattern(pattern)
        if not _pitch_fits_moras(parsed, moras):
            raise ValueError(
                f"dictionary entry {word!r} has {len(moras)} morae but "
                f"pitch pattern {pattern!r}"
            )
        phones = tuple(
            str(phone).strip()
            for phone in raw_entry.get("phones") or ()
            if str(phone).strip()
        )
        g2p_override = bool(
            raw_entry.get("g2p_override", False)
            or str(raw_entry.get("g2p_source") or "").casefold()
            == "override"
        )
        if not phones:
            raise ValueError(
                f"dictionary entry {word!r} has no synthesis phones"
            )
        if not g2p_override and phones != g2p_asaxi(word):
            raise ValueError(
                f"dictionary entry {word!r} has inconsistent G2P data"
            )
        source_notes = tuple(
            str(note)
            for note in raw_entry.get("source_notes") or ()
            if str(note)
        )
        source_note = str(raw_entry.get("source_note") or "")
        if not source_notes and source_note:
            source_notes = (source_note,)
        variants = tuple(
            _decode_dictionary_variant(word, variant, moras)
            for variant in raw_entry.get("variants") or ()
            if isinstance(variant, Mapping)
        )
        if variants:
            default_variant = int(raw_entry.get("default_variant") or 0)
            if not 0 <= default_variant < len(variants):
                raise ValueError(
                    f"dictionary entry {word!r} has invalid default variant"
                )
            selected = variants[default_variant]
            top_signature = (
                phones,
                canonical_pitch_pattern(parsed[0]),
                str(raw_entry.get("pitch_accent_class") or "lexical"),
                g2p_override,
            )
            variant_signature = (
                tuple(selected["phones"]),
                selected["pitch_accent"],
                selected["pitch_accent_class"],
                bool(selected["g2p_override"]),
            )
            if top_signature != variant_signature:
                raise ValueError(
                    f"dictionary entry {word!r} disagrees with its "
                    "default variant"
                )
        entries[word] = AsaxiLexiconEntry(
            word=word,
            phones=phones,
            moras=moras,
            pitch_accent=canonical_pitch_pattern(parsed[0]),
            pitch_accent_class=str(
                raw_entry.get("pitch_accent_class") or "lexical"),
            g2p_override=g2p_override,
            source_note=source_note,
            source_notes=source_notes,
            variants=variants,
        )
    return AsaxiSynthesisDictionary(
        schema_version=1,
        ruleset=str(payload.get("ruleset") or ""),
        entries=entries,
        morphemes={
            normalize_word(form): dict(record)
            for form, record in dict(
                payload.get("morphemes") or {}
            ).items()
            if isinstance(record, Mapping)
        },
        morphological_analyses={
            normalize_word(word): dict(record)
            for word, record in dict(
                payload.get("morphological_analyses") or {}
            ).items()
            if isinstance(record, Mapping)
        },
        phrases=dict(payload.get("phrases") or {}),
        source_summary=dict(payload.get("source_summary") or {}),
    )
