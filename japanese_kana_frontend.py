"""Dependency-free kana and supported Hepburn-romaji Japanese frontend."""

from __future__ import annotations

from dataclasses import dataclass, field
import unicodedata
from typing import Iterable, Optional

from japanese_models import (
    JapaneseAccentPhrase,
    JapaneseFrontendDiagnostic,
    JapaneseMora,
    JapanesePhone,
    JapanesePhrase,
    JapaneseUtterance,
)


_VOWELS = {"a", "i", "u", "e", "o"}
_ROMAJI_CHARACTERS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'"
    "āīūēōĀĪŪĒŌâîûêôÂÎÛÊÔ"
)
_MACRONS = {
    "ā": "a~", "ī": "i~", "ū": "u~", "ē": "e~", "ō": "o~",
    "â": "a~", "î": "i~", "û": "u~", "ê": "e~", "ô": "o~",
}
_PUNCTUATION_STRENGTH = {
    "、": 1, ",": 1, "，": 1, "・": 1,
    ":": 2, "：": 2, ";": 2, "；": 2,
    "▽": 2,
    "。": 3, ".": 3, "!": 3, "！": 3, "?": 3, "？": 3,
}
_QUESTION_MARKS = {"?", "？"}
_IGNORABLE_READING_PUNCTUATION = {
    "\u300c", "\u300d", "\u300e", "\u300f",
    "\u3010", "\u3011", "\u3008", "\u3009", "\u300a", "\u300b",
    "\uff08", "\uff09", "(", ")", "\uff3b", "\uff3d", "[", "]",
    "\u201c", "\u201d", '"',
}


@dataclass(frozen=True)
class KanaMoraSpec:
    surface: str
    reading: str
    consonant: Optional[str]
    vowel: Optional[str]
    phones: tuple[str, ...]
    special_mora: Optional[str] = None
    unknown: bool = False
    confidence: float = 1.0
    provenance: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RomajiConversion:
    reading: str
    diagnostics: tuple[JapaneseFrontendDiagnostic, ...] = ()


@dataclass
class _PhraseDraft:
    surface_parts: list[str] = field(default_factory=list)
    mora_specs: list[KanaMoraSpec] = field(default_factory=list)
    punctuation_after: str = ""
    boundary_strength: int = 3
    interrogative: bool = False
    explicit_pause: bool = False


def _base_mora_table() -> dict[str, tuple[Optional[str], str, tuple[str, ...]]]:
    table: dict[str, tuple[Optional[str], str, tuple[str, ...]]] = {}

    def add(row: str, consonant: Optional[str], vowels: str = "aiueo") -> None:
        for kana, vowel in zip(row, vowels):
            phones = (vowel,) if consonant is None else (consonant, vowel)
            table[kana] = (consonant, vowel, phones)

    add("あいうえお", None)
    add("かきくけこ", "k")
    add("がぎぐげご", "g")
    add("さしすせそ", "s")
    table["し"] = ("sh", "i", ("sh", "i"))
    add("ざじずぜぞ", "z")
    table["じ"] = ("j", "i", ("j", "i"))
    add("たちつてと", "t")
    table["ち"] = ("ch", "i", ("ch", "i"))
    table["つ"] = ("ts", "u", ("ts", "u"))
    add("だぢづでど", "d")
    table["ぢ"] = ("j", "i", ("j", "i"))
    table["づ"] = ("z", "u", ("z", "u"))
    add("なにぬねの", "n")
    add("はひふへほ", "h")
    table["ふ"] = ("f", "u", ("f", "u"))
    add("ばびぶべぼ", "b")
    add("ぱぴぷぺぽ", "p")
    add("まみむめも", "m")
    add("らりるれろ", "r")
    table.update({
        "や": ("y", "a", ("y", "a")),
        "ゆ": ("y", "u", ("y", "u")),
        "よ": ("y", "o", ("y", "o")),
        "わ": ("w", "a", ("w", "a")),
        "ゐ": ("w", "i", ("w", "i")),
        "ゑ": ("w", "e", ("w", "e")),
        "を": (None, "o", ("o",)),
        "ゔ": ("v", "u", ("v", "u")),
        "ゎ": ("w", "a", ("w", "a")),
        "ぁ": (None, "a", ("a",)),
        "ぃ": (None, "i", ("i",)),
        "ぅ": (None, "u", ("u",)),
        "ぇ": (None, "e", ("e",)),
        "ぉ": (None, "o", ("o",)),
    })
    return table


_KANA_MORAS = _base_mora_table()


def _add_yoon(base: str, consonant: str) -> None:
    for small, vowel in (("ゃ", "a"), ("ゅ", "u"), ("ょ", "o")):
        _KANA_MORAS[base + small] = (
            consonant, vowel, (consonant, vowel)
        )


for _base, _consonant in (
    ("き", "ky"), ("ぎ", "gy"), ("し", "sh"), ("じ", "j"),
    ("ち", "ch"), ("に", "ny"), ("ひ", "hy"), ("び", "by"),
    ("ぴ", "py"), ("み", "my"), ("り", "ry"),
):
    _add_yoon(_base, _consonant)


_KANA_MORAS.update({
    "いぇ": ("y", "e", ("y", "e")),
    "うぃ": ("w", "i", ("w", "i")),
    "うぇ": ("w", "e", ("w", "e")),
    "うぉ": ("w", "o", ("w", "o")),
    "くぁ": ("kw", "a", ("k", "w", "a")),
    "くぃ": ("kw", "i", ("k", "w", "i")),
    "くぇ": ("kw", "e", ("k", "w", "e")),
    "くぉ": ("kw", "o", ("k", "w", "o")),
    "ぐぁ": ("gw", "a", ("g", "w", "a")),
    "ぐぃ": ("gw", "i", ("g", "w", "i")),
    "ぐぇ": ("gw", "e", ("g", "w", "e")),
    "ぐぉ": ("gw", "o", ("g", "w", "o")),
    "しぇ": ("sh", "e", ("sh", "e")),
    "じぇ": ("j", "e", ("j", "e")),
    "ちぇ": ("ch", "e", ("ch", "e")),
    "つぁ": ("ts", "a", ("ts", "a")),
    "つぃ": ("ts", "i", ("ts", "i")),
    "つぇ": ("ts", "e", ("ts", "e")),
    "つぉ": ("ts", "o", ("ts", "o")),
    "てぃ": ("t", "i", ("t", "i")),
    "とぅ": ("t", "u", ("t", "u")),
    "でぃ": ("d", "i", ("d", "i")),
    "どぅ": ("d", "u", ("d", "u")),
    "ふぁ": ("f", "a", ("f", "a")),
    "ふぃ": ("f", "i", ("f", "i")),
    "ふぇ": ("f", "e", ("f", "e")),
    "ふぉ": ("f", "o", ("f", "o")),
    "ゔぁ": ("v", "a", ("v", "a")),
    "ゔぃ": ("v", "i", ("v", "i")),
    "ゔぇ": ("v", "e", ("v", "e")),
    "ゔぉ": ("v", "o", ("v", "o")),
})


_CANONICAL_MORA_READINGS: dict[tuple[str, ...], str] = {}
for _reading, (_consonant, _vowel, _phones) in _KANA_MORAS.items():
    # The base table declares ordinary hiragana before any alternate spelling,
    # so setdefault gives each canonical mora one stable profile lookup key.
    _CANONICAL_MORA_READINGS.setdefault(tuple(_phones), _reading)


def canonical_mora_reading(phones: Iterable[str]) -> Optional[str]:
    """Return the stable hiragana lookup key for one canonical phone tuple."""
    return _CANONICAL_MORA_READINGS.get(tuple(str(phone) for phone in phones))


_ROMAJI_TO_HIRAGANA = {
    "kya": "きゃ", "kyu": "きゅ", "kyo": "きょ",
    "gya": "ぎゃ", "gyu": "ぎゅ", "gyo": "ぎょ",
    "sha": "しゃ", "shu": "しゅ", "sho": "しょ",
    "sya": "しゃ", "syu": "しゅ", "syo": "しょ",
    "cha": "ちゃ", "chu": "ちゅ", "cho": "ちょ",
    "tya": "ちゃ", "tyu": "ちゅ", "tyo": "ちょ",
    "nya": "にゃ", "nyu": "にゅ", "nyo": "にょ",
    "hya": "ひゃ", "hyu": "ひゅ", "hyo": "ひょ",
    "bya": "びゃ", "byu": "びゅ", "byo": "びょ",
    "pya": "ぴゃ", "pyu": "ぴゅ", "pyo": "ぴょ",
    "mya": "みゃ", "myu": "みゅ", "myo": "みょ",
    "rya": "りゃ", "ryu": "りゅ", "ryo": "りょ",
    "jya": "じゃ", "jyu": "じゅ", "jyo": "じょ",
    "ja": "じゃ", "ji": "じ", "ju": "じゅ", "jo": "じょ",
    "shi": "し", "chi": "ち", "tsu": "つ", "dzu": "づ",
    "she": "しぇ", "je": "じぇ", "che": "ちぇ",
    "tsa": "つぁ", "tsi": "つぃ", "tse": "つぇ", "tso": "つぉ",
    "fa": "ふぁ", "fi": "ふぃ", "fe": "ふぇ", "fo": "ふぉ",
    "va": "ゔぁ", "vi": "ゔぃ", "vu": "ゔ", "ve": "ゔぇ", "vo": "ゔぉ",
    "kwa": "くぁ", "kwi": "くぃ", "kwe": "くぇ", "kwo": "くぉ",
    "gwa": "ぐぁ", "gwi": "ぐぃ", "gwe": "ぐぇ", "gwo": "ぐぉ",
    "ka": "か", "ki": "き", "ku": "く", "ke": "け", "ko": "こ",
    "ga": "が", "gi": "ぎ", "gu": "ぐ", "ge": "げ", "go": "ご",
    "sa": "さ", "si": "し", "su": "す", "se": "せ", "so": "そ",
    "za": "ざ", "zi": "じ", "zu": "ず", "ze": "ぜ", "zo": "ぞ",
    "ta": "た", "ti": "ち", "tu": "つ", "te": "て", "to": "と",
    "da": "だ", "di": "ぢ", "du": "づ", "de": "で", "do": "ど",
    "na": "な", "ni": "に", "nu": "ぬ", "ne": "ね", "no": "の",
    "ha": "は", "hi": "ひ", "hu": "ふ", "fu": "ふ", "he": "へ", "ho": "ほ",
    "ba": "ば", "bi": "び", "bu": "ぶ", "be": "べ", "bo": "ぼ",
    "pa": "ぱ", "pi": "ぴ", "pu": "ぷ", "pe": "ぺ", "po": "ぽ",
    "ma": "ま", "mi": "み", "mu": "む", "me": "め", "mo": "も",
    "ya": "や", "yu": "ゆ", "yo": "よ",
    "ra": "ら", "ri": "り", "ru": "る", "re": "れ", "ro": "ろ",
    "wa": "わ", "wi": "うぃ", "we": "うぇ", "wo": "を",
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お",
}


def katakana_to_hiragana(text: str) -> str:
    converted = []
    for character in unicodedata.normalize("NFKC", text):
        codepoint = ord(character)
        if 0x30A1 <= codepoint <= 0x30F6:
            converted.append(chr(codepoint - 0x60))
        else:
            converted.append(character)
    return "".join(converted)


def normalize_kana_reading(text: str) -> str:
    return "".join(
        character
        for character in katakana_to_hiragana(text)
        if not character.isspace()
    )


def convert_romaji(text: str, *, source_start: int = 0) -> RomajiConversion:
    expanded = "".join(
        _MACRONS.get(character.lower(), character.lower())
        for character in unicodedata.normalize("NFC", text)
    )
    output: list[str] = []
    diagnostics: list[JapaneseFrontendDiagnostic] = []
    index = 0
    while index < len(expanded):
        character = expanded[index]
        if character == "~":
            output.append("ー")
            index += 1
            continue
        if character == "n" and index + 1 < len(expanded) \
                and expanded[index + 1] == "'":
            output.append("ん")
            index += 2
            continue
        if expanded[index:index + 3] == "tch":
            output.append("っ")
            index += 1
            continue
        if character not in "aeioun" and index + 1 < len(expanded) \
                and expanded[index + 1] == character:
            output.append("っ")
            index += 1
            continue
        if character == "n" and (
            index + 1 == len(expanded)
            or expanded[index + 1] not in "aeiouy"
        ):
            output.append("ん")
            index += 1
            continue
        matched = False
        for length in (4, 3, 2, 1):
            token = expanded[index:index + length]
            kana = _ROMAJI_TO_HIRAGANA.get(token)
            if kana is None:
                continue
            output.append(kana)
            index += length
            matched = True
            break
        if matched:
            continue
        diagnostics.append(JapaneseFrontendDiagnostic(
            code="unsupported_romaji",
            message=f"Unsupported romaji sequence begins with {character!r}.",
            severity="warning",
            action="Use supported Hepburn romaji or enter the intended kana.",
            source_start=source_start + index,
            source_end=source_start + index + 1,
            frontend="kana",
            confidence=0.0,
            raw_data={"character": character, "run": text},
        ))
        output.append("�")
        index += 1
    return RomajiConversion("".join(output), tuple(diagnostics))


def romaji_to_hiragana(text: str) -> str:
    """Convert supported Hepburn romaji without silently dropping input."""
    return convert_romaji(text).reading


def segment_kana_reading(
    reading: str,
    *,
    source_start: int = 0,
) -> tuple[tuple[KanaMoraSpec, ...], tuple[JapaneseFrontendDiagnostic, ...]]:
    normalized = normalize_kana_reading(reading)
    moras: list[KanaMoraSpec] = []
    diagnostics: list[JapaneseFrontendDiagnostic] = []
    index = 0
    while index < len(normalized):
        character = normalized[index]
        if (
            character.isspace()
            or character in _PUNCTUATION_STRENGTH
            or character in _IGNORABLE_READING_PUNCTUATION
        ):
            index += 1
            continue
        if character == "ん":
            moras.append(KanaMoraSpec(
                surface=character,
                reading=character,
                consonant=None,
                vowel=None,
                phones=("N",),
                special_mora="moraic_nasal",
            ))
            index += 1
            continue
        if character == "っ":
            moras.append(KanaMoraSpec(
                surface=character,
                reading=character,
                consonant=None,
                vowel=None,
                phones=("cl",),
                special_mora="geminate",
            ))
            index += 1
            continue
        if character in {"ー", "ｰ"}:
            previous_vowel = moras[-1].vowel if moras else None
            if previous_vowel in _VOWELS:
                moras.append(KanaMoraSpec(
                    surface=character,
                    reading="ー",
                    consonant=None,
                    vowel=previous_vowel,
                    phones=(previous_vowel,),
                    special_mora="long_vowel",
                    provenance={"lengthens_mora": len(moras) - 1},
                ))
            else:
                diagnostics.append(JapaneseFrontendDiagnostic(
                    code="orphan_long_vowel_mark",
                    message="A long-vowel mark has no preceding vowel to lengthen.",
                    severity="warning",
                    action="Place ー after a vowel-bearing mora.",
                    source_start=source_start + index,
                    source_end=source_start + index + 1,
                    frontend="kana",
                    confidence=0.0,
                ))
                moras.append(KanaMoraSpec(
                    surface=character,
                    reading="�",
                    consonant=None,
                    vowel=None,
                    phones=("unk",),
                    special_mora="unknown",
                    unknown=True,
                    confidence=0.0,
                ))
            index += 1
            continue

        if character == "�":
            moras.append(KanaMoraSpec(
                surface=character,
                reading=character,
                consonant=None,
                vowel=None,
                phones=("unk",),
                special_mora="unknown",
                unknown=True,
                confidence=0.0,
                provenance={"raw_character": character},
            ))
            index += 1
            continue

        match = None
        for length in (2, 1):
            candidate = normalized[index:index + length]
            if candidate in _KANA_MORAS:
                match = candidate
                break
        if match is not None:
            consonant, vowel, phones = _KANA_MORAS[match]
            moras.append(KanaMoraSpec(
                surface=match,
                reading=match,
                consonant=consonant,
                vowel=vowel,
                phones=phones,
                confidence=0.8 if match in "ぁぃぅぇぉゎ" else 1.0,
            ))
            index += len(match)
            continue

        diagnostics.append(JapaneseFrontendDiagnostic(
            code="unsupported_kana_character",
            message=f"Kana character {character!r} is not in the fallback table.",
            severity="warning",
            action="Enter a supported mora or use the Open JTalk frontend.",
            source_start=source_start + index,
            source_end=source_start + index + 1,
            frontend="kana",
            confidence=0.0,
            raw_data={"character": character},
        ))
        moras.append(KanaMoraSpec(
            surface=character,
            reading="�",
            consonant=None,
            vowel=None,
            phones=("unk",),
            special_mora="unknown",
            unknown=True,
            confidence=0.0,
            provenance={"raw_character": character},
        ))
        index += 1
    return tuple(moras), tuple(diagnostics)


def _is_kana(character: str) -> bool:
    return (
        "\u3040" <= character <= "\u30ff"
        or "\uff66" <= character <= "\uff9f"
    )


def _is_kanji(character: str) -> bool:
    return (
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
        or character in {"々", "〆", "ヶ"}
    )


def _phone_type(symbol: str, unknown: bool = False) -> str:
    if unknown or symbol == "unk":
        return "unknown"
    if symbol in _VOWELS:
        return "vowel"
    if symbol in {"N", "cl", "pau", "sil"}:
        return "special"
    return "consonant"


def _unknown_specs(text: str) -> list[KanaMoraSpec]:
    return [
        KanaMoraSpec(
            surface=character,
            reading="�",
            consonant=None,
            vowel=None,
            phones=("unk",),
            special_mora="unknown",
            unknown=True,
            confidence=0.0,
            provenance={"raw_character": character},
        )
        for character in text
    ]


class KanaJapaneseFrontend:
    name = "kana"
    version = "phase1"

    def analyze(self, text: str) -> JapaneseUtterance:
        source = unicodedata.normalize("NFKC", text)
        drafts: list[_PhraseDraft] = []
        current = _PhraseDraft()
        diagnostics: list[JapaneseFrontendDiagnostic] = []

        def close_phrase(
            punctuation: str = "",
            strength: int = 3,
            interrogative: bool = False,
            explicit_pause: bool = False,
        ) -> None:
            nonlocal current
            if current.mora_specs:
                current.punctuation_after = punctuation
                current.boundary_strength = strength
                current.interrogative = interrogative
                current.explicit_pause = explicit_pause
                drafts.append(current)
                current = _PhraseDraft()
            elif drafts and punctuation:
                drafts[-1].punctuation_after += punctuation
                drafts[-1].boundary_strength = max(
                    drafts[-1].boundary_strength, strength
                )
                drafts[-1].interrogative |= interrogative
                drafts[-1].explicit_pause |= explicit_pause

        index = 0
        while index < len(source):
            if source[index:index + 5].casefold() == "[pau]":
                close_phrase("[pau]", 2, explicit_pause=True)
                index += 5
                continue
            character = source[index]
            if character in _PUNCTUATION_STRENGTH:
                close_phrase(
                    character,
                    _PUNCTUATION_STRENGTH[character],
                    character in _QUESTION_MARKS,
                )
                index += 1
                continue
            if character.isspace():
                current.surface_parts.append(character)
                index += 1
                continue
            if character in _ROMAJI_CHARACTERS:
                end = index + 1
                while end < len(source) and source[end] in _ROMAJI_CHARACTERS:
                    end += 1
                run = source[index:end]
                conversion = convert_romaji(run, source_start=index)
                specs, segment_diagnostics = segment_kana_reading(
                    conversion.reading, source_start=index
                )
                current.surface_parts.append(run)
                current.mora_specs.extend(specs)
                diagnostics.extend(conversion.diagnostics)
                diagnostics.extend(segment_diagnostics)
                index = end
                continue
            if _is_kana(character):
                end = index + 1
                while end < len(source) and _is_kana(source[end]):
                    end += 1
                run = source[index:end]
                specs, segment_diagnostics = segment_kana_reading(
                    run, source_start=index
                )
                current.surface_parts.append(run)
                current.mora_specs.extend(specs)
                diagnostics.extend(segment_diagnostics)
                index = end
                continue
            if _is_kanji(character):
                end = index + 1
                while end < len(source) and _is_kanji(source[end]):
                    end += 1
                run = source[index:end]
                current.surface_parts.append(run)
                current.mora_specs.extend(_unknown_specs(run))
                diagnostics.append(JapaneseFrontendDiagnostic(
                    code="unsupported_kanji",
                    message=(
                        f"The dependency-free frontend cannot determine the "
                        f"reading of {run!r}."
                    ),
                    severity="warning",
                    action=(
                        "Enter the reading in kana/romaji or select Open JTalk."
                    ),
                    source_start=index,
                    source_end=end,
                    frontend=self.name,
                    confidence=0.0,
                    raw_data={"text": run},
                ))
                index = end
                continue

            current.surface_parts.append(character)
            current.mora_specs.extend(_unknown_specs(character))
            diagnostics.append(JapaneseFrontendDiagnostic(
                code="unsupported_character",
                message=f"Unsupported character {character!r} was kept as unknown.",
                severity="warning",
                action="Replace it with supported kana or romaji.",
                source_start=index,
                source_end=index + 1,
                frontend=self.name,
                confidence=0.0,
                raw_data={"character": character},
            ))
            index += 1

        if current.mora_specs:
            current.boundary_strength = 3
            drafts.append(current)

        diagnostics.insert(0, JapaneseFrontendDiagnostic(
            code="lexical_accent_unavailable",
            message=(
                "The kana fallback uses a deterministic neutral accent; it "
                "does not know lexical pitch accent."
            ),
            severity="info",
            action=(
                "Install pyopenjtalk for lexical analysis or edit the accent "
                "structure in a later phase."
            ),
            frontend=self.name,
            confidence=1.0,
        ))

        phones: list[JapanesePhone] = [JapanesePhone(
            index=0,
            symbol="sil",
            raw_symbol="sil",
            phone_type="special",
            is_silence=True,
        )]
        phrases: list[JapanesePhrase] = []
        global_mora_index = 0
        global_accent_index = 0

        for phrase_index, draft in enumerate(drafts):
            phrase_phone_indices: list[int] = []
            moras: list[JapaneseMora] = []
            for spec in draft.mora_specs:
                mora_phones: list[JapanesePhone] = []
                for symbol in spec.phones:
                    phone = JapanesePhone(
                        index=len(phones),
                        symbol=symbol,
                        raw_symbol=(
                            str(spec.provenance.get("raw_character"))
                            if spec.unknown else symbol
                        ),
                        phone_type=_phone_type(symbol, spec.unknown),
                        phrase_index=phrase_index,
                        accent_phrase_index=global_accent_index,
                        mora_index=global_mora_index,
                        unknown=spec.unknown,
                        confidence=spec.confidence,
                    )
                    phones.append(phone)
                    mora_phones.append(phone)
                    phrase_phone_indices.append(phone.index)
                moras.append(JapaneseMora(
                    index=global_mora_index,
                    phrase_index=phrase_index,
                    accent_phrase_index=global_accent_index,
                    surface=spec.surface,
                    reading=spec.reading,
                    phones=tuple(mora_phones),
                    consonant=spec.consonant,
                    vowel=spec.vowel,
                    special_mora=spec.special_mora,
                    devoiced=None,
                    confidence=spec.confidence,
                    provenance=spec.provenance,
                ))
                global_mora_index += 1

            accent_phrase = JapaneseAccentPhrase(
                index=global_accent_index,
                phrase_index=phrase_index,
                moras=tuple(moras),
                accent_state="unavailable",
                accent_nucleus=None,
                interrogative=draft.interrogative,
                boundary_strength=draft.boundary_strength,
                confidence=0.4,
                provenance={"neutral_accent_default": True},
            )
            reading = "".join(mora.reading for mora in moras)
            phrases.append(JapanesePhrase(
                index=phrase_index,
                surface="".join(draft.surface_parts).strip(),
                normalized_reading=reading,
                accent_phrases=(accent_phrase,),
                punctuation_after=draft.punctuation_after,
                boundary_strength=draft.boundary_strength,
                interrogative=draft.interrogative,
                phone_indices=tuple(phrase_phone_indices),
                confidence=min(
                    (mora.confidence for mora in moras), default=1.0
                ),
                provenance={
                    "explicit_pause": draft.explicit_pause,
                    "neutral_accent_default": True,
                },
            ))
            global_accent_index += 1

            if draft.punctuation_after or draft.explicit_pause:
                phones.append(JapanesePhone(
                    index=len(phones),
                    symbol="pau",
                    raw_symbol="pau",
                    phone_type="special",
                    phrase_index=phrase_index,
                    is_pause=True,
                ))

        phones.append(JapanesePhone(
            index=len(phones),
            symbol="sil",
            raw_symbol="sil",
            phone_type="special",
            is_silence=True,
        ))
        normalized_reading = "".join(
            phrase.normalized_reading for phrase in phrases
        )
        confidence = min(
            (phrase.confidence for phrase in phrases), default=1.0
        )
        return JapaneseUtterance(
            source_text=text,
            normalized_reading=normalized_reading,
            phrases=tuple(phrases),
            phones=tuple(phones),
            diagnostics=tuple(diagnostics),
            frontend_name=self.name,
            frontend_version=self.version,
            confidence=min(confidence, 0.7),
            provenance={
                "input_normalization": "NFKC",
                "reading_script": "hiragana",
                "lexical_accent": "unavailable",
                "neutral_accent_default": True,
            },
        )
