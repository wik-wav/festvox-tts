# -*- coding: utf-8 -*-
"""Build a deterministic, source-traceable Asaxi prosody corpus.

This corpus is intended for recording and later acoustic alignment.  It does
not treat the current synthetic contour as ground truth.  Every prompt keeps
its source, the current dictionary/mora hypothesis, and any frontend
diagnostics so recorded evidence can correct the model.

The generated corpus has four complementary strata:

* attested prosody controls from the elicitation record;
* fixed expressions and representative lexical citation forms;
* short, translated examples mined conservatively from grammar notes;
* natural narrative utterances from the interlinear reader.

Only Python's standard library and the adjacent Asaxi frontend are required.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
from typing import Iterable, Mapping, Sequence

import asaxi_frontend as af
import asaxi_prosody


SCHEMA_VERSION = 2
CORPUS_ID = "asaxi-prosody-v1"
GENERATOR_VERSION = "1.1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAULT_ROOT = PROJECT_ROOT / "Lozenge-T-Notes"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "corpora" / CORPUS_ID
INTERLINEAR_GLOB = "*Velveteen Rabbit*.md"
PROSODY_NOTE = (
    "01_Worldbuilding/Asaxi/Grammar_Structure/"
    "61_Prosody, Stress & Intonation.md"
)
GRAMMAR_DIR = "01_Worldbuilding/Asaxi/Grammar_Structure"
DICTIONARY_RELATIVE = "src/festvox_tts/dictionaries/asaxi_lexicon.json"

_TERMINAL_RE = re.compile(r".+?(?:[.!?]+[”\"]*(?=\s|$)|$)", re.DOTALL)
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_QUOTED_ENGLISH_RE = re.compile(r'"([^"]+)"')
_SPACE_RE = re.compile(r"\s+")
_MARKDOWN_LINK_RE = re.compile(r"\[\[([^\]|]+\|)?([^\]]+)\]\]")
_LEADING_LABEL_RE = re.compile(r"^\s*\*{0,2}[^:*]{1,30}:\*{0,2}\s*")
_BAD_GRAMMAR_MARKERS = ("+", "=", "→", "⇒", "{", "}", "<", ">")
_ENGLISH_LINE_OPENERS = frozenset({
    "a", "additionally", "analysis", "because", "example", "here", "if",
    "implication", "meaning", "note", "nuance", "standard", "statement",
    "the", "this", "when", "where", "while",
})


@dataclass(frozen=True)
class PromptSeed:
    text: str
    translation_en: str
    kind: str
    tier: str
    takes: int
    source_note: str
    source_line: int = 0
    source_section: str = ""
    translation_scope: str = "exact"
    annotation_status: str = "model_hypothesis"
    expected_reading: str = ""
    demonstration: str = ""
    extra_tags: tuple[str, ...] = ()
    explicit_id: str = ""


@dataclass(frozen=True)
class CorpusBuild:
    manifest: Mapping[str, object]
    recording_script: str
    prompt_list: str
    reader_corpus: str
    coverage: Mapping[str, object]
    readme: str

    def files(self) -> dict[str, str]:
        return {
            "manifest.json": _json_text(self.manifest),
            "recording_script.tsv": self.recording_script,
            "prompts.txt": self.prompt_list,
            "reader_corpus.md": self.reader_corpus,
            "coverage.json": _json_text(self.coverage),
            "README.md": self.readme,
        }


# These readings are transcribed directly from the 2026-06-12 elicitation
# appendix.  English descriptions are deliberately conservative where the
# item was elicited for morphology/prosody rather than as a free translation.
PROSODY_CONTROLS: tuple[PromptSeed, ...] = (
    PromptSeed(
        "shěsonů.",
        "to read (citation form)",
        "prosody_control",
        "A",
        3,
        PROSODY_NOTE,
        expected_reading="H.L.L",
        demonstration="default root-initial accent",
        extra_tags=("lexical_accent:default", "elicited"),
        annotation_status="attested",
        explicit_id="asx_ctl_001",
    ),
    PromptSeed(
        "zènáshěsonů.",
        "did not read (controlled inflected form)",
        "prosody_control",
        "A",
        3,
        PROSODY_NOTE,
        expected_reading="H.L.H.L.L",
        demonstration="statement onset, downstepped negation, accent recovery",
        extra_tags=("downstep", "dominant:negation", "elicited"),
        annotation_status="attested",
        explicit_id="asx_ctl_002",
    ),
    PromptSeed(
        "pazènáchỏnů.",
        "controlled complex inflected form",
        "prosody_control",
        "A",
        3,
        PROSODY_NOTE,
        expected_reading="H.L.H.H.H",
        demonstration="downstep, dominance recovery, monomoraic plateau",
        extra_tags=("downstep", "dominant:negation", "plateau", "elicited"),
        annotation_status="attested",
        explicit_id="asx_ctl_003",
    ),
    PromptSeed(
        "mmbănă.",
        "happy; joyful (citation form)",
        "prosody_control",
        "A",
        3,
        PROSODY_NOTE,
        expected_reading="L.H.L",
        demonstration="syllabic nasal cannot bear the H target",
        extra_tags=("mora:syllabic_nasal", "accent_skip", "elicited"),
        annotation_status="attested",
        explicit_id="asx_ctl_004",
    ),
    PromptSeed(
        "gaviŕoŕo.",
        "controlled compound citation form",
        "prosody_control",
        "A",
        3,
        PROSODY_NOTE,
        expected_reading="H.L.L.L",
        demonstration="compound accentual unification",
        extra_tags=("compound", "deaccenting", "elicited"),
        annotation_status="attested",
        explicit_id="asx_ctl_005",
    ),
    PromptSeed(
        "kozètètá.",
        "controlled fused-compound citation form",
        "prosody_control",
        "A",
        3,
        PROSODY_NOTE,
        expected_reading="H.L.L.L",
        demonstration="fused compound with one peak",
        extra_tags=("compound", "deaccenting", "elicited"),
        annotation_status="attested",
        explicit_id="asx_ctl_006",
    ),
    PromptSeed(
        "to wo shěso ma.",
        "I have a book.",
        "prosody_control",
        "A",
        3,
        PROSODY_NOTE,
        expected_reading="H.L.H.L.L",
        demonstration="statement onset and terminal fall",
        extra_tags=("boundary:onset_high", "boundary:terminal_fall", "elicited"),
        annotation_status="attested",
        explicit_id="asx_ctl_007",
    ),
    PromptSeed(
        "no xogă?",
        "Are you coming?",
        "prosody_control",
        "A",
        3,
        PROSODY_NOTE,
        expected_reading="L.L.L↗",
        demonstration="particleless question with total deaccenting",
        extra_tags=("question:particleless", "deaccenting", "elicited"),
        annotation_status="attested",
        explicit_id="asx_ctl_008",
    ),
    PromptSeed(
        "no kvå xoxo?",
        "When are you leaving?",
        "prosody_control",
        "A",
        3,
        PROSODY_NOTE,
        expected_reading="L.H.L.L↗",
        demonstration="wh-focus and post-focus deaccenting",
        extra_tags=("question:wh", "post_focus_deaccenting", "elicited"),
        annotation_status="attested",
        explicit_id="asx_ctl_009",
    ),
    PromptSeed(
        "ŕoŕo daohè!",
        "Give the animal!",
        "prosody_control",
        "A",
        3,
        PROSODY_NOTE,
        expected_reading="H.L.H.L↗",
        demonstration="directive with appeal rise",
        extra_tags=("directive", "boundary:terminal_rise", "elicited"),
        annotation_status="attested",
        explicit_id="asx_ctl_010",
    ),
    PromptSeed(
        "haśùnáhè!",
        "Do not run!",
        "prosody_control",
        "A",
        3,
        PROSODY_NOTE,
        expected_reading="L.H.H↗",
        demonstration="non-initial lexical accent and dominant plateau",
        extra_tags=("directive", "dominant:negation", "plateau", "elicited"),
        annotation_status="attested",
        explicit_id="asx_ctl_011",
    ),
    PromptSeed(
        "wo zèxăcè wő.",
        "I assert that it was certainly so.",
        "prosody_control",
        "A",
        3,
        PROSODY_NOTE,
        expected_reading="L.L.H.H.H",
        demonstration="insistent high terminal plateau",
        extra_tags=("dominant:affirmation", "boundary:insistent", "elicited"),
        annotation_status="attested",
        explicit_id="asx_ctl_012",
    ),
    PromptSeed(
        "ăjo lem, måmå natăka!",
        "O Lem, remember tomorrow! (controlled vocative)",
        "prosody_control",
        "A",
        3,
        PROSODY_NOTE,
        expected_reading="H.L | L || H.L | L.H.L↗",
        demonstration="vocative contour, name deaccenting, directive rise",
        extra_tags=("vocative", "deaccenting", "directive", "elicited"),
        annotation_status="attested",
        explicit_id="asx_ctl_013",
    ),
)


def _json_text(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", str(value or ""))


def _clean_markdown(value: str) -> str:
    text = _nfc(value)
    text = _MARKDOWN_LINK_RE.sub(r"\2", text)
    text = text.replace("**", "").replace("__", "").replace("*", "")
    text = text.replace("[", "").replace("]", "")
    text = text.replace("\\", "")
    text = text.replace("«", "„").replace("»", "”")
    text = re.sub(r"(?:\.\s*){2,}", "...", text)
    return _SPACE_RE.sub(" ", text).strip()


def _clean_asaxi(value: str) -> str:
    text = _clean_markdown(value).lower()
    text = text.replace("(", "").replace(")", "")
    return text.strip(" \t—")


def _clean_english(value: str) -> str:
    return _clean_markdown(value).strip()


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"corpus source must be inside the project: {path}") \
            from error


def _stable_id(kind: str, source_note: str, text: str) -> str:
    prefix = {
        "grammar_example": "gra",
        "natural_reader": "nat",
        "lexical_citation": "lex",
        "fixed_expression": "idm",
    }.get(kind, "x")
    material = "\0".join((kind, source_note, _clean_asaxi(text)))
    suffix = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return f"asx_{prefix}_{suffix}"


def _read_frontmatter(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    result: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if not line or line[:1].isspace() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip("\"'")
    return result


def _translation_for_source(path: Path) -> str:
    frontmatter = _read_frontmatter(path)
    translation = (
        frontmatter.get("trnsltion. En")
        or frontmatter.get("translation. En")
        or frontmatter.get("translation_en")
        or ""
    )
    if translation:
        return _clean_english(translation)
    title = frontmatter.get("title", "")
    if " - " in title:
        return _clean_english(title.split(" - ", 1)[1])
    return ""


def _find_interlinear(vault_root: Path) -> Path:
    text_dir = vault_root / "01_Worldbuilding" / "Asaxi" / "Texts"
    candidates = [
        path for path in text_dir.glob(INTERLINEAR_GLOB)
        if "Reader's Text" not in path.name
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Expected exactly one interlinear Velveteen Rabbit note, found "
            f"{len(candidates)} under {text_dir}"
        )
    return candidates[0]


def _split_terminal_utterances(text: str) -> tuple[str, ...]:
    value = _clean_asaxi(text)
    raw_parts = tuple(
        part
        for match in _TERMINAL_RE.finditer(value)
        if (part := _clean_asaxi(match.group(0)))
        and af.words_in_text(part)
    )
    if not raw_parts:
        return (value,) if value else ()
    parts: list[str] = []
    inside_quote = False
    for raw_part in raw_parts:
        started_inside = inside_quote
        for character in raw_part:
            if character == "„":
                inside_quote = True
            elif character == "”":
                inside_quote = False
        part = raw_part
        if started_inside and not part.lstrip().startswith("„"):
            part = "„" + part
        if inside_quote and not part.rstrip().endswith("”"):
            part += "”"
        parts.append(part)
    return tuple(parts)


def _split_english_utterances(text: str) -> tuple[str, ...]:
    value = _clean_english(text)
    raw_parts = tuple(
        part
        for match in _TERMINAL_RE.finditer(value)
        if (part := _clean_english(match.group(0)))
    )
    if not raw_parts:
        return (value,) if value else ()
    parts: list[str] = []
    inside_ascii_quote = False
    inside_curly_quote = False
    for raw_part in raw_parts:
        started_ascii = inside_ascii_quote
        started_curly = inside_curly_quote
        for character in raw_part:
            if character == '"':
                inside_ascii_quote = not inside_ascii_quote
            elif character in {"“", "„"}:
                inside_curly_quote = True
            elif character == "”":
                inside_curly_quote = False
        part = raw_part
        if started_ascii and not part.lstrip().startswith('"'):
            part = '"' + part
        if started_curly and not part.lstrip().startswith(("“", "„")):
            part = "“" + part
        if inside_ascii_quote and not part.rstrip().endswith('"'):
            part += '"'
        if inside_curly_quote and not part.rstrip().endswith("”"):
            part += "”"
        parts.append(part)
    return tuple(parts)


def extract_natural_reader(
    vault_root: Path,
    project_root: Path,
) -> tuple[list[PromptSeed], set[Path]]:
    source = _find_interlinear(vault_root)
    source_note = _relative(source, project_root)
    lines = source.read_text(
        encoding="utf-8-sig",
        errors="replace",
    ).splitlines()
    section = ""
    result: list[PromptSeed] = []
    for index, line in enumerate(lines):
        if line.startswith("## ") and not line.startswith("### "):
            section = _clean_markdown(line[3:])
        if not line.startswith("> "):
            continue
        english = _clean_english(line[2:])
        asaxi_lines: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            candidate = lines[cursor].strip()
            if (
                candidate.startswith("|")
                or candidate.startswith("> ")
                or candidate.startswith("#")
                or candidate == "---"
            ):
                break
            if candidate:
                asaxi_lines.append(candidate)
            cursor += 1
        if not asaxi_lines:
            continue
        utterances = _split_terminal_utterances(" ".join(asaxi_lines))
        english_utterances = _split_english_utterances(english)
        sentence_aligned = len(english_utterances) == len(utterances)
        for utterance_index, utterance in enumerate(utterances):
            if sentence_aligned:
                translation = english_utterances[utterance_index]
                translation_scope = (
                    "exact" if len(utterances) == 1 else "terminal_aligned"
                )
            else:
                translation = english
                translation_scope = "source_context"
            result.append(PromptSeed(
                text=utterance,
                translation_en=translation,
                kind="natural_reader",
                tier="C",
                takes=1,
                source_note=source_note,
                source_line=index + 1,
                source_section=section,
                translation_scope=translation_scope,
                annotation_status="model_hypothesis",
                extra_tags=("natural_speech", "narrative"),
            ))
    return result, {source}


def _looks_like_spoken_asaxi(
    text: str,
    dictionary: af.AsaxiSynthesisDictionary,
) -> bool:
    if not text or any(marker in text for marker in _BAD_GRAMMAR_MARKERS):
        return False
    if any(character.isdigit() for character in text):
        return False
    if not re.search(r"[.!?]\s*$", text):
        return False
    try:
        words = af.words_in_text(text, reject_unsupported_letters=True)
    except ValueError:
        return False
    if not words or len(words) > 24:
        return False
    if words[0] in _ENGLISH_LINE_OPENERS:
        return False
    known = sum(dictionary.lookup(word) is not None for word in words)
    has_asaxi_mark = any(
        ord(character) > 127 and character.isalpha()
        for character in text
    )
    return known >= 2 or (known >= 1 and has_asaxi_mark)


def extract_grammar_examples(
    vault_root: Path,
    project_root: Path,
    dictionary: af.AsaxiSynthesisDictionary,
) -> tuple[list[PromptSeed], set[Path]]:
    grammar_root = vault_root / GRAMMAR_DIR
    result: list[PromptSeed] = []
    sources: set[Path] = set()
    seen: set[str] = set()
    for path in sorted(grammar_root.glob("*.md"), key=lambda item: item.name):
        relative = _relative(path, project_root)
        lines = path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        ).splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.startswith("> "):
                continue
            raw = line[2:].strip()
            translation_matches = [
                match
                for match in _QUOTED_ENGLISH_RE.finditer(raw)
                if _ASCII_LETTER_RE.search(match.group(1))
            ]
            if not translation_matches:
                continue
            translation_match = translation_matches[0]
            translation = translation_match.group(1).strip()
            quote_at = translation_match.start()
            if quote_at <= 0:
                continue
            candidate = raw[:quote_at]
            placeholders = re.findall(r"\\?\[([^\]]+)\\?\]", candidate)
            unresolved_placeholder = False
            for placeholder in placeholders:
                try:
                    placeholder_words = af.words_in_text(
                        placeholder,
                        reject_unsupported_letters=True,
                    )
                except ValueError:
                    unresolved_placeholder = True
                    break
                if any(
                    dictionary.lookup(word) is None
                    for word in placeholder_words
                ):
                    unresolved_placeholder = True
                    break
            if unresolved_placeholder:
                continue
            candidate = _INLINE_CODE_RE.sub(" ", candidate)
            candidate = _LEADING_LABEL_RE.sub("", candidate)
            candidate = re.sub(r"\([^)]*\)\s*$", "", candidate)
            candidate = _clean_asaxi(candidate)
            if not _looks_like_spoken_asaxi(candidate, dictionary):
                continue
            normalized = candidate.casefold()
            if normalized in seen:
                continue
            try:
                asaxi_prosody.analyze_utterance(candidate, dictionary)
            except (TypeError, ValueError):
                continue
            seen.add(normalized)
            sources.add(path)
            result.append(PromptSeed(
                text=candidate,
                translation_en=_clean_english(translation),
                kind="grammar_example",
                tier="B",
                takes=2,
                source_note=relative,
                source_line=line_number,
                source_section=path.stem,
                annotation_status="source_attested_model_accent",
                extra_tags=("controlled_sentence", "grammar_attested"),
            ))
    return result, sources


def _entry_source_path(
    vault_root: Path,
    entry: af.AsaxiLexiconEntry,
) -> Path | None:
    if not entry.source_note:
        return None
    path = vault_root / Path(entry.source_note)
    return path if path.is_file() else None


def _lexical_candidate_features(
    word: str,
    entry: af.AsaxiLexiconEntry,
) -> frozenset[str]:
    features = {
        f"accent_class:{entry.pitch_accent_class}",
        f"pitch_pattern:{entry.pitch_accent}",
        f"mora_count:{min(len(entry.moras), 5)}",
    }
    for mora in entry.moras:
        features.add(f"mora:{mora.kind}")
    if entry.moras and not entry.moras[0].accentable:
        features.add("accent_skip")
    if len(entry.moras) == 1:
        features.add("plateau_candidate")
    first_phone = entry.phones[0] if entry.phones else ""
    features.add(
        "onset:vowel" if first_phone in {
            "a", "e", "i", "o", "u", "ax", "er", "ih", "uw", "ao",
        } else "onset:consonant"
    )
    return frozenset(features)


def select_lexical_citations(
    vault_root: Path,
    project_root: Path,
    dictionary: af.AsaxiSynthesisDictionary,
    *,
    limit: int = 72,
) -> tuple[list[PromptSeed], set[Path]]:
    candidates: list[tuple[str, af.AsaxiLexiconEntry, str, Path, frozenset[str]]] = []
    pattern_counts = Counter(
        entry.pitch_accent for entry in dictionary.entries.values()
    )
    for word, entry in sorted(dictionary.entries.items()):
        if (
            not word
            or " " in word
            or "-" in word
            or entry.pitch_accent in {"", "none"}
            or not entry.phones
        ):
            continue
        path = _entry_source_path(vault_root, entry)
        if path is None:
            continue
        translation = _translation_for_source(path)
        if not translation:
            continue
        candidates.append((
            word,
            entry,
            translation,
            path,
            _lexical_candidate_features(word, entry),
        ))

    frequent_patterns = {
        pattern for pattern, count in pattern_counts.items()
        if count >= 2 and pattern not in {"", "none"}
    }
    required = {
        "accent_class:lexical",
        "accent_class:atonal",
        "accent_class:dominant",
        "accent_class:mixed",
        "mora:syllabic_nasal",
        "mora:geminate",
        "accent_skip",
        "plateau_candidate",
        "onset:vowel",
        "onset:consonant",
    }
    required.update(f"pitch_pattern:{pattern}" for pattern in frequent_patterns)
    required.update(f"mora_count:{count}" for count in range(1, 6))

    selected: list[tuple[str, af.AsaxiLexiconEntry, str, Path, frozenset[str]]] = []
    remaining = candidates[:]
    uncovered = set(required)
    while remaining and uncovered and len(selected) < limit:
        best = min(
            remaining,
            key=lambda item: (
                -len(item[4] & uncovered),
                hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
                item[0],
            ),
        )
        gain = best[4] & uncovered
        if not gain:
            break
        selected.append(best)
        remaining.remove(best)
        uncovered.difference_update(gain)

    # Fill the remainder deterministically while preventing one common
    # H.L class from crowding out the less frequent patterns.
    selected_words = {item[0] for item in selected}
    pattern_selected = Counter(item[1].pitch_accent for item in selected)
    fill = sorted(
        (item for item in remaining if item[0] not in selected_words),
        key=lambda item: (
            pattern_selected[item[1].pitch_accent],
            hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
            item[0],
        ),
    )
    for item in fill:
        if len(selected) >= limit:
            break
        selected.append(item)
        pattern_selected[item[1].pitch_accent] += 1

    result: list[PromptSeed] = []
    sources: set[Path] = set()
    for word, entry, translation, path, features in selected:
        sources.add(path)
        result.append(PromptSeed(
            text=f"{word}.",
            translation_en=f"{translation} (citation form)",
            kind="lexical_citation",
            tier="A",
            takes=2,
            source_note=_relative(path, project_root),
            annotation_status="dictionary",
            expected_reading=entry.pitch_accent,
            demonstration=(
                f"{entry.pitch_accent_class} accent; "
                f"{len(entry.moras)} mora(e)"
            ),
            extra_tags=tuple(sorted(features | {"lexical_citation"})),
        ))
    return result, sources


def fixed_expression_prompts(
    vault_root: Path,
    project_root: Path,
    dictionary: af.AsaxiSynthesisDictionary,
) -> tuple[list[PromptSeed], set[Path]]:
    result: list[PromptSeed] = []
    sources: set[Path] = set()
    for expression, record in sorted(dictionary.phrases.items()):
        source_note = str(record.get("source_note") or "")
        path = vault_root / Path(source_note)
        if not path.is_file():
            continue
        sources.add(path)
        translation = _translation_for_source(path) or "fixed expression"
        result.append(PromptSeed(
            text=f"{expression}.",
            translation_en=translation,
            kind="fixed_expression",
            tier="A",
            takes=3,
            source_note=_relative(path, project_root),
            annotation_status="dictionary",
            expected_reading=str(record.get("pitch_accent") or ""),
            demonstration="multiword dictionary accent",
            extra_tags=("fixed_expression", "phrase_accent"),
        ))
    return result, sources


def _speech_act(plan: asaxi_prosody.AsaxiProsodyPlan) -> str:
    if plan.interrogative:
        return "question"
    if plan.directive:
        return "directive"
    return "statement"


def _phrase_boundaries(text: str) -> list[dict[str, object]]:
    strength = {
        ",": "minor",
        ":": "minor",
        ";": "major",
        "—": "major",
        ".": "terminal",
        "?": "terminal",
        "!": "terminal",
    }
    return [
        {
            "character_index": index,
            "mark": character,
            "strength": strength[character],
        }
        for index, character in enumerate(text)
        if character in strength
    ]


def _length_tag(word_count: int) -> str:
    if word_count <= 2:
        return "length:citation"
    if word_count <= 7:
        return "length:short"
    if word_count <= 16:
        return "length:medium"
    return "length:long"


def _coverage_tags(
    seed: PromptSeed,
    plan: asaxi_prosody.AsaxiProsodyPlan,
) -> tuple[str, ...]:
    tags = set(seed.extra_tags)
    act = _speech_act(plan)
    tags.add(f"speech_act:{act}")
    tags.add(f"boundary:{plan.boundary_mark}")
    tags.add(_length_tag(len(plan.words)))
    if "," in seed.text:
        tags.add("punctuation:comma")
    if ";" in seed.text:
        tags.add("punctuation:semicolon")
    if "—" in seed.text:
        tags.add("punctuation:dash")
    if "„" in seed.text or "”" in seed.text:
        tags.add("punctuation:quotation")
    if any(word.surface in asaxi_prosody.WH_WORDS for word in plan.words):
        tags.add("question:wh")
    if any(
        word.surface in asaxi_prosody.QUESTION_PARTICLES
        for word in plan.words
    ):
        tags.add("question:particle")
    if any(
        word.surface in asaxi_prosody.INSISTENT_TAILS
        for word in plan.words
    ):
        tags.add("boundary:insistent")
    for word in plan.words:
        tags.add(f"accent_class:{word.pitch_accent_class}")
        if word.phrase_expression:
            tags.add("phrase_dictionary_override")
    for mora in plan.moras:
        tags.add(f"mora:{mora.kind}")
    first_phone = plan.phones[0] if plan.phones else ""
    tags.add(
        "onset:vowel" if first_phone in {
            "a", "e", "i", "o", "u", "ax", "er", "ih", "uw", "ao",
        } else "onset:consonant"
    )
    return tuple(sorted(tags))


def _reference_values(reading: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[HL]", str(reading or "").upper()))


def _predicted_reading(
    plan: asaxi_prosody.AsaxiProsodyPlan,
) -> str:
    values = ".".join(mora.pitch for mora in plan.moras)
    return values + ("↗" if plan.boundary_tone == "LH%" else "")


def _reference_authority(seed: PromptSeed) -> str:
    if not seed.expected_reading:
        return "none"
    if seed.annotation_status == "attested":
        return "attested"
    if seed.annotation_status == "dictionary":
        return "dictionary"
    return seed.annotation_status


def _pitch_analysis(
    seed: PromptSeed,
    plan: asaxi_prosody.AsaxiProsodyPlan,
) -> dict[str, object]:
    predicted_values = tuple(mora.pitch for mora in plan.moras)
    dictionary_values = tuple(
        value
        for word in plan.words
        for value in af.parse_pitch_pattern(word.pitch_accent)[0]
    )
    reference_values = _reference_values(seed.expected_reading)
    reference_authority = _reference_authority(seed)
    comparison_scope = (
        "lexical_or_phrase"
        if reference_authority == "dictionary"
        else "utterance"
    )
    comparison_values = (
        dictionary_values
        if comparison_scope == "lexical_or_phrase"
        else predicted_values
    )
    reference_boundary = (
        "LH%" if "↗" in seed.expected_reading else ""
    )

    word_transcript = []
    reference_cursor = 0
    reference_is_word_aligned = (
        bool(reference_values)
        and len(reference_values) == len(predicted_values)
    )
    for word in plan.words:
        predicted = tuple(
            mora.pitch
            for mora in plan.moras[word.mora_start:word.mora_end]
        )
        width = word.mora_end - word.mora_start
        reference = (
            reference_values[reference_cursor:reference_cursor + width]
            if reference_is_word_aligned
            else ()
        )
        reference_cursor += width
        morphemes = [
            morpheme.to_dict() for morpheme in word.morphemes
        ]
        morphology = (
            asaxi_prosody.format_morpheme_analysis(word.morphemes)
            if word.morphemes
            else ""
        )
        word_transcript.append({
            "word": word.surface,
            "dictionary_or_phrase": word.pitch_accent,
            "predicted_utterance": ".".join(predicted) or "none",
            "reference": ".".join(reference) if reference else "",
            "pitch_accent_class": word.pitch_accent_class,
            "phrase_expression": word.phrase_expression,
            "morphemes": morphemes,
            "morphology": morphology,
        })

    dictionary_compact = " | ".join(
        f"{item['word']} [{item['dictionary_or_phrase']}]"
        for item in word_transcript
    )
    predicted_compact = " | ".join(
        f"{item['word']} [{item['predicted_utterance']}]"
        for item in word_transcript
    )
    morphology_compact = " | ".join(
        f"{item['word']} [{item['morphology'] or 'lexical'}]"
        for item in word_transcript
    )
    reference_compact = (
        " | ".join(
            f"{item['word']} [{item['reference']}]"
            for item in word_transcript
        )
        if reference_is_word_aligned
        else ""
    )

    if not reference_values:
        status = "no_reference"
    elif len(reference_values) != len(comparison_values):
        status = "mora_count_mismatch"
    else:
        values_match = reference_values == comparison_values
        boundary_matches = (
            comparison_scope != "utterance"
            or
            not reference_boundary
            or reference_boundary == plan.boundary_tone
        )
        status = "exact" if values_match and boundary_matches else (
            "boundary_mismatch" if values_match else "pitch_mismatch"
        )
    compared = min(len(reference_values), len(comparison_values))
    matching = sum(
        left == right
        for left, right in zip(comparison_values, reference_values)
    )
    boundary_match = (
        plan.boundary_tone == reference_boundary
        if reference_boundary and comparison_scope == "utterance"
        else None
    )
    return {
        "predicted": {
            "authority": "model_prediction",
            "word_transcript": word_transcript,
            "dictionary_pitch_by_word": dictionary_compact,
            "utterance_pitch_by_word": predicted_compact,
            "morphology_by_word": morphology_compact,
            "mora_sequence": ".".join(predicted_values),
            "reading": _predicted_reading(plan),
            "boundary_tone": plan.boundary_tone,
        },
        "reference": {
            "authority": reference_authority,
            "reading": seed.expected_reading,
            "mora_sequence": ".".join(reference_values),
            "boundary_tone": reference_boundary,
            "word_aligned": reference_is_word_aligned,
            "pitch_by_word": reference_compact,
        },
        "agreement": {
            "status": status,
            "comparison_scope": comparison_scope,
            "comparison_mora_count": len(comparison_values),
            "reference_mora_count": len(reference_values),
            "compared_mora_count": compared,
            "matching_mora_count": matching,
            "pitch_match_ratio": (
                round(matching / compared, 6) if compared else None
            ),
            "boundary_match": boundary_match,
        },
    }


def _prompt_payload(
    seed: PromptSeed,
    dictionary: af.AsaxiSynthesisDictionary,
) -> dict[str, object]:
    text = _clean_asaxi(seed.text)
    plan = asaxi_prosody.analyze_utterance(text, dictionary)
    diagnostics = [diagnostic.to_dict() for diagnostic in plan.diagnostics]
    pitch_analysis = _pitch_analysis(seed, plan)
    agreement = pitch_analysis["agreement"]
    if agreement["status"] == "mora_count_mismatch":
        diagnostics.append({
            "code": "reference_mora_count_mismatch",
            "message": (
                f"The preserved {pitch_analysis['reference']['authority']} "
                f"reading has {agreement['reference_mora_count']} H/L values "
                f"for {agreement['comparison_mora_count']} analyzed morae."
            ),
            "severity": "warning",
            "word_index": None,
        })
    elif agreement["status"] in {"pitch_mismatch", "boundary_mismatch"}:
        diagnostics.append({
            "code": "reference_pitch_mismatch",
            "message": (
                "The current model prediction does not yet match the "
                f"preserved {pitch_analysis['reference']['authority']} "
                "reading."
            ),
            "severity": "warning",
            "word_index": None,
        })
    warning_count = sum(
        item["severity"] in {"warning", "error"} for item in diagnostics
    )
    prompt_id = seed.explicit_id or _stable_id(
        seed.kind,
        seed.source_note,
        text,
    )
    return {
        "id": prompt_id,
        "asaxi": text,
        "translation_en": _clean_english(seed.translation_en),
        "translation_scope": seed.translation_scope,
        "kind": seed.kind,
        "tier": seed.tier,
        "recommended_takes": seed.takes,
        "source": {
            "note": seed.source_note,
            "line": seed.source_line,
            "section": seed.source_section,
        },
        "annotation_status": seed.annotation_status,
        "requires_linguistic_review": bool(warning_count),
        "demonstration": seed.demonstration,
        "speech_act": _speech_act(plan),
        "phrase_boundaries": _phrase_boundaries(text),
        "coverage_tags": list(_coverage_tags(seed, plan)),
        "pitch_analysis": pitch_analysis,
        "analysis": plan.to_dict(),
        "diagnostics": diagnostics,
    }


def _deduplicate(seeds: Iterable[PromptSeed]) -> list[PromptSeed]:
    result: list[PromptSeed] = []
    seen: set[tuple[str, str]] = set()
    for seed in seeds:
        key = (seed.kind, _clean_asaxi(seed.text).casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append(seed)
    return result


def _coverage(prompts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    kind_counts = Counter(str(prompt["kind"]) for prompt in prompts)
    tier_counts = Counter(str(prompt["tier"]) for prompt in prompts)
    act_counts = Counter(str(prompt["speech_act"]) for prompt in prompts)
    translation_scope_counts = Counter(
        str(prompt["translation_scope"]) for prompt in prompts
    )
    tag_counts = Counter(
        str(tag)
        for prompt in prompts
        for tag in prompt["coverage_tags"]
    )
    diagnostic_counts = Counter(
        str(item["code"])
        for prompt in prompts
        for item in prompt["diagnostics"]
    )
    agreement_counts = Counter(
        str(prompt["pitch_analysis"]["agreement"]["status"])
        for prompt in prompts
    )
    reference_authority_counts = Counter(
        str(prompt["pitch_analysis"]["reference"]["authority"])
        for prompt in prompts
    )
    take_count = sum(int(prompt["recommended_takes"]) for prompt in prompts)
    words = sum(len(prompt["analysis"]["words"]) for prompt in prompts)
    moras = sum(len(prompt["analysis"]["moras"]) for prompt in prompts)
    phones = sum(len(prompt["analysis"]["phones"]) for prompt in prompts)
    estimated_seconds = sum(
        max(0.8, len(prompt["analysis"]["moras"]) / 5.5 + 0.45)
        * int(prompt["recommended_takes"])
        for prompt in prompts
    )
    required_tags = {
        "speech_act:statement",
        "speech_act:question",
        "speech_act:directive",
        "mora:syllabic_nasal",
        "mora:geminate",
        "onset:vowel",
        "onset:consonant",
        "punctuation:comma",
        "punctuation:quotation",
        "question:wh",
        "question:particle",
        "boundary:insistent",
        "phrase_accent",
        "natural_speech",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": CORPUS_ID,
        "prompt_count": len(prompts),
        "recommended_take_count": take_count,
        "estimated_recording_minutes": round(estimated_seconds / 60.0, 1),
        "word_token_count": words,
        "mora_token_count": moras,
        "phone_token_count": phones,
        "by_kind": dict(sorted(kind_counts.items())),
        "by_tier": dict(sorted(tier_counts.items())),
        "by_speech_act": dict(sorted(act_counts.items())),
        "by_translation_scope": dict(
            sorted(translation_scope_counts.items())
        ),
        "coverage_tags": dict(sorted(tag_counts.items())),
        "diagnostics": dict(sorted(diagnostic_counts.items())),
        "pitch_reference_agreement": dict(
            sorted(agreement_counts.items())
        ),
        "pitch_reference_authority": dict(
            sorted(reference_authority_counts.items())
        ),
        "prompts_requiring_linguistic_review": sum(
            bool(prompt["requires_linguistic_review"]) for prompt in prompts
        ),
        "required_tags": sorted(required_tags),
        "missing_required_tags": sorted(required_tags - tag_counts.keys()),
    }


def _source_records(
    paths: Iterable[Path],
    project_root: Path,
) -> list[dict[str, str]]:
    return [
        {
            "path": _relative(path, project_root),
            "sha256": _sha256(path),
        }
        for path in sorted(
            {path.resolve() for path in paths},
            key=lambda item: _relative(item, project_root),
        )
    ]


def _tsv_cell(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).replace("\t", " ").strip()


def _recording_script(prompts: Sequence[Mapping[str, object]]) -> str:
    rows = [[
        "recording_id",
        "prompt_id",
        "take",
        "tier",
        "kind",
        "asaxi",
        "translation_en",
        "translation_scope",
        "dictionary_pitch_by_word",
        "predicted_utterance_pitch_by_word",
        "morphology_by_word",
        "predicted_boundary_tone",
        "reference_authority",
        "reference_reading",
        "reference_agreement",
        "annotation_status",
    ]]
    for prompt in prompts:
        for take in range(1, int(prompt["recommended_takes"]) + 1):
            rows.append([
                f"{prompt['id']}_t{take:02d}",
                prompt["id"],
                str(take),
                prompt["tier"],
                prompt["kind"],
                prompt["asaxi"],
                prompt["translation_en"],
                prompt["translation_scope"],
                prompt["pitch_analysis"]["predicted"][
                    "dictionary_pitch_by_word"
                ],
                prompt["pitch_analysis"]["predicted"][
                    "utterance_pitch_by_word"
                ],
                prompt["pitch_analysis"]["predicted"][
                    "morphology_by_word"
                ],
                prompt["pitch_analysis"]["predicted"]["boundary_tone"],
                prompt["pitch_analysis"]["reference"]["authority"],
                prompt["pitch_analysis"]["reference"]["reading"],
                prompt["pitch_analysis"]["agreement"]["status"],
                prompt["annotation_status"],
            ])
    return "\n".join(
        "\t".join(_tsv_cell(cell) for cell in row)
        for row in rows
    ) + "\n"


def _prompt_list(prompts: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Asaxi prosody recording prompts",
        "# Format: prompt_id<TAB>recommended_takes<TAB>Asaxi",
        "",
    ]
    for prompt in prompts:
        lines.append(
            f"{prompt['id']}\t{prompt['recommended_takes']}\t"
            f"{_tsv_cell(prompt['asaxi'])}"
        )
    return "\n".join(lines) + "\n"


def _reader_pitch_table(prompt: Mapping[str, object]) -> list[str]:
    transcript = prompt["pitch_analysis"]["predicted"]["word_transcript"]
    lines = [
        (
            "| # | Asaxi word | Dictionary / phrase accent | "
            "Predicted in utterance | Morpheme analysis | "
            "Reference evidence |"
        ),
        "|---:|---|---|---|---|---|",
    ]
    for index, item in enumerate(transcript, start=1):
        reference = (
            f"`{item['reference']}`"
            if item["reference"]
            else "—"
        )
        lines.append(
            f"| {index} | {item['word']} | "
            f"`{item['dictionary_or_phrase']}` | "
            f"`{item['predicted_utterance']}` | "
            f"{item['morphology'] or '—'} | {reference} |"
        )
    return lines


def _agreement_description(prompt: Mapping[str, object]) -> str:
    pitch = prompt["pitch_analysis"]
    agreement = pitch["agreement"]
    status = agreement["status"]
    if status == "no_reference":
        return "No direct reference is available for this prompt."
    if status == "exact":
        if agreement["comparison_scope"] == "lexical_or_phrase":
            return (
                "Exact lexical match: every H/L value agrees with the "
                "preserved dictionary or phrase entry. Utterance-level "
                "boundary rules may still change the predicted column."
            )
        return (
            "Exact: every reference H/L value matches the current model"
            + (
                ", including the marked boundary rise."
                if agreement["boundary_match"] is True
                else "."
            )
        )
    if status == "mora_count_mismatch":
        return (
            "Not directly alignable: the preserved reference contains "
            f"{agreement['reference_mora_count']} H/L values, while the "
            f"analyzed form has {agreement['comparison_mora_count']} morae."
        )
    if status == "boundary_mismatch":
        return (
            "Mora values match, but the marked reference boundary tone does "
            "not match the current model."
        )
    return (
        f"Mismatch: {agreement['matching_mora_count']} of "
        f"{agreement['compared_mora_count']} compared H/L values match."
    )


def _reader_corpus(
    prompts: Sequence[Mapping[str, object]],
    coverage: Mapping[str, object],
) -> str:
    lines = [
        "# Asaxi Prosody Recording Corpus",
        "",
        (
            f"{coverage['prompt_count']} prompts; "
            f"{coverage['recommended_take_count']} recommended takes; "
            f"approximately {coverage['estimated_recording_minutes']} "
            "minutes."
        ),
        "",
        "## How To Read The Transcript",
        "",
        "Each prompt has a table with one row per written word:",
        "",
        (
            "| # | Asaxi word | Dictionary / phrase accent | "
            "Predicted in utterance | Morpheme analysis | "
            "Reference evidence |"
        ),
        "|---:|---|---|---|---|---|",
        (
            "| 1 | sháma | `H.H` | `H.H` | "
            "shá (root) + -ma (plural) | `H.H` |"
        ),
        "",
        "Dictionary / phrase accent is lexical source data. Predicted in "
        "utterance is the current prosody model after morphology, "
        "deaccenting, phrasing, and boundary rules. Reference evidence is "
        "either attested utterance elicitation or a lexical dictionary "
        "reading, and its scope is stated in the agreement line. It is never "
        "filled with a model prediction. H and L are mora-level targets, not "
        "measured frequencies.",
        "",
        "Read only the plain Asaxi line. The English line gives meaning and "
        "context; do not read it aloud.",
        "",
    ]
    tier_titles = {
        "A": "Tier A — Core Elicitation And Lexical Anchors",
        "B": "Tier B — Translated Grammar Sentences",
        "C": "Tier C — Natural Narrative",
    }
    prompt_number = 0
    for tier in ("A", "B", "C"):
        tier_prompts = [
            prompt for prompt in prompts if prompt["tier"] == tier
        ]
        lines.extend([
            f"## {tier_titles[tier]}",
            "",
        ])
        for prompt in tier_prompts:
            prompt_number += 1
            english_label = (
                "English source context"
                if prompt["translation_scope"] == "source_context"
                else "English translation"
            )
            lines.extend([
                f"### {prompt_number}. `{prompt['id']}`",
                "",
                (
                    f'<span class="asaxi-text">{prompt["asaxi"]}</span>'
                ),
                "",
                f"{english_label}: {prompt['translation_en']}",
                "",
                "Word-level pitch analysis:",
                "",
            ])
            lines.extend(_reader_pitch_table(prompt))
            lines.append("")
            pitch = prompt["pitch_analysis"]
            lines.append(
                "Current model prediction: "
                f"`{pitch['predicted']['reading']}` · "
                f"Boundary tone: `{pitch['predicted']['boundary_tone']}`"
            )
            if pitch["reference"]["reading"]:
                label = (
                    "Attested reference"
                    if pitch["reference"]["authority"] == "attested"
                    else "Dictionary reference"
                )
                lines.append(
                    f"{label}: `{pitch['reference']['reading']}`"
                )
            lines.append(
                "Reference agreement: "
                + _agreement_description(prompt)
            )
            lines.extend([
                (
                    f"Record: {prompt['recommended_takes']} take(s) · "
                    f"Speech act: {prompt['speech_act']} · "
                    f"Annotation: `{prompt['annotation_status']}`"
                ),
                (
                    f"Source: `{prompt['source']['note']}`"
                    + (
                        f", line {prompt['source']['line']}"
                        if prompt["source"]["line"]
                        else ""
                    )
                ),
            ])
            if prompt["requires_linguistic_review"]:
                lines.append(
                    "Review note: this prompt has an unresolved dictionary, "
                    "reference-alignment, or model-agreement warning. See "
                    "its manifest diagnostics before fitting."
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _readme(coverage: Mapping[str, object]) -> str:
    kinds = "\n".join(
        f"- `{name}`: {count}"
        for name, count in coverage["by_kind"].items()
    )
    return f"""# Asaxi Prosody Corpus v1

This directory is generated by `asaxi_prosody_corpus.py`. It contains
{coverage['prompt_count']} prompts and {coverage['recommended_take_count']}
recommended recordings (about {coverage['estimated_recording_minutes']}
minutes of speech).

## Contents

- `manifest.json`: canonical prompts, source provenance, phones, morae,
  separately labeled model predictions and direct references, agreement
  results, boundaries, diagnostics, and coverage tags.
- `recording_script.tsv`: one row per requested take.
- `prompts.txt`: compact speaker-facing prompt list.
- `reader_corpus.md`: readable edition with English and clearly separated
  dictionary, predicted-utterance, and direct-reference pitch transcripts.
- `coverage.json`: corpus totals, category counts, and unresolved warnings.

## Strata

{kinds}

Tier A items isolate lexical and grammatical contrasts and should receive all
requested repetitions. Tier B contains translated grammar examples. Tier C is
natural narrative material and normally needs one clean take.

## Recording protocol

1. Record mono WAV at a stable sample rate and bit depth; 48 kHz, 24-bit is
   preferred when the interface supports it.
2. Keep the microphone, distance, room, speaking voice, and input gain fixed.
3. Read the Asaxi column only. English is context, not a line to speak.
4. Use a comfortable neutral pitch and rate. Preserve punctuation, discourse
   grouping, and natural phrase-final behavior; do not chant the H/L labels.
5. Name takes exactly as `recording_id` in `recording_script.tsv`, for example
   `asx_ctl_001_t01.wav`.
6. Do not trim internal pauses or repair disfluencies destructively. Record a
   replacement take and retain rejected takes outside the aligned dataset.
7. Calibrate once before the first session, then include a small Tier A anchor
   set at the start of later sessions to measure session drift.

The H/L values in `manifest.json` are annotations, not generated F0 tracks.
`pitch_analysis.predicted` is always model output.
`pitch_analysis.reference.authority: attested` is direct elicitation evidence;
`dictionary` is lexical documentation. These are never silently substituted
for one another. Predictions must remain editable after alignment and review.
"""


def build_corpus(
    *,
    project_root: Path = PROJECT_ROOT,
    vault_root: Path = DEFAULT_VAULT_ROOT,
) -> CorpusBuild:
    project_root = project_root.resolve()
    vault_root = vault_root.resolve()
    dictionary_path = project_root / DICTIONARY_RELATIVE
    dictionary = asaxi_prosody.load_dictionary(dictionary_path)

    natural, natural_sources = extract_natural_reader(
        vault_root,
        project_root,
    )
    grammar, grammar_sources = extract_grammar_examples(
        vault_root,
        project_root,
        dictionary,
    )
    lexical, lexical_sources = select_lexical_citations(
        vault_root,
        project_root,
        dictionary,
    )
    idioms, idiom_sources = fixed_expression_prompts(
        vault_root,
        project_root,
        dictionary,
    )

    seeds = _deduplicate(
        list(PROSODY_CONTROLS) + idioms + lexical + grammar + natural
    )
    prompts = [_prompt_payload(seed, dictionary) for seed in seeds]
    ids = [str(prompt["id"]) for prompt in prompts]
    if len(ids) != len(set(ids)):
        duplicates = sorted(
            identifier
            for identifier, count in Counter(ids).items()
            if count > 1
        )
        raise ValueError(f"duplicate stable prompt IDs: {duplicates}")

    coverage = _coverage(prompts)
    if coverage["missing_required_tags"]:
        raise ValueError(
            "corpus coverage is incomplete: "
            + ", ".join(coverage["missing_required_tags"])
        )

    source_paths = (
        natural_sources
        | grammar_sources
        | lexical_sources
        | idiom_sources
        | {
            dictionary_path,
            vault_root / PROSODY_NOTE,
        }
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": CORPUS_ID,
        "language": "asaxi",
        "generator_version": GENERATOR_VERSION,
        "dictionary_ruleset": dictionary.ruleset,
        "orthography": {
            "case": "lowercase",
            "quotation_marks": "polish",
            "quoted_open": "„",
            "quoted_close": "”",
        },
        "purpose": (
            "recording, alignment, and empirical fitting of Asaxi timing, "
            "lexical accent, phrasing, and boundary tones"
        ),
        "annotation_policy": {
            "attested": (
                "directly transcribed from the dated elicitation record"
            ),
            "dictionary": (
                "lexical or phrase accent from the synthesis dictionary"
            ),
            "source_attested_model_accent": (
                "sentence and translation are attested; accent is the current "
                "model hypothesis"
            ),
            "model_hypothesis": (
                "natural text is attested; inflected-word accent and phrasing "
                "remain hypotheses until recording review"
            ),
            "prediction_reference_separation": (
                "pitch_analysis.predicted is always model output; "
                "pitch_analysis.reference preserves attested or dictionary "
                "evidence; agreement reports compare them without replacing "
                "either"
            ),
        },
        "summary": coverage,
        "sources": _source_records(source_paths, project_root),
        "prompts": prompts,
    }
    return CorpusBuild(
        manifest=manifest,
        recording_script=_recording_script(prompts),
        prompt_list=_prompt_list(prompts),
        reader_corpus=_reader_corpus(prompts, coverage),
        coverage=coverage,
        readme=_readme(coverage),
    )


def write_corpus(build: CorpusBuild, output_dir: Path) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in build.files().items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8", newline="\n")
        written.append(path)
    return tuple(written)


def check_corpus(build: CorpusBuild, output_dir: Path) -> list[str]:
    differences = []
    for name, expected in build.files().items():
        path = output_dir / name
        if not path.is_file():
            differences.append(f"missing: {name}")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            differences.append(f"stale: {name}")
    return differences


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Lozenge-T-Vault project root",
    )
    parser.add_argument(
        "--vault",
        type=Path,
        default=DEFAULT_VAULT_ROOT,
        help="Asaxi notes vault",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="generated corpus directory",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if committed generated files differ from current sources",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build = build_corpus(
        project_root=args.project_root,
        vault_root=args.vault,
    )
    if args.check:
        differences = check_corpus(build, args.output)
        if differences:
            print("\n".join(differences), file=sys.stderr)
            return 1
        print(
            f"{CORPUS_ID}: {build.coverage['prompt_count']} prompts; current"
        )
        return 0
    written = write_corpus(build, args.output)
    print(json.dumps(build.coverage, ensure_ascii=False, indent=2))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
