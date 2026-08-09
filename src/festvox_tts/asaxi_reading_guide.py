# -*- coding: utf-8 -*-
"""Generate a reader-facing Asaxi pitch-accent guide as Markdown.

The guide uses the canonical Asaxi dictionary and utterance planner. It does
not maintain a second set of accent rules. Input may be plain Asaxi text or a
Markdown document containing Asaxi prose; Markdown extraction is conservative
so navigation, explanatory English, tables, and frontmatter are not spoken.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import sys
import unicodedata
from typing import Iterable, Optional, Sequence

import asaxi_frontend as af
import asaxi_prosody


GENERATOR_VERSION = "1.0"
_TERMINAL_RE = re.compile(
    r".+?(?:[.!?]+[”\"]*(?=\s|$)|$)",
    re.DOTALL,
)
_SPACE_RE = re.compile(r"\s+")
_ASAXI_MARKER_RE = re.compile(r"[áăåèëěỏőùůýŋŕśń]", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+\|)?([^\]]+)\]\]")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_CODE_RE = re.compile(r"`([^`]*)`")
_EMPHASIS_RE = re.compile(r"[*_~]+")


@dataclass(frozen=True)
class GuideUtterance:
    text: str
    section: str = ""
    source_line: int = 0


@dataclass(frozen=True)
class ReadingGuide:
    markdown: str
    utterance_count: int
    warning_count: int


def _strip_markdown(value: str) -> str:
    text = _WIKILINK_RE.sub(lambda match: match.group(2), str(value or ""))
    text = _MARKDOWN_LINK_RE.sub(lambda match: match.group(1), text)
    text = _CODE_RE.sub(lambda match: match.group(1), text)
    text = _HTML_TAG_RE.sub("", text)
    text = _EMPHASIS_RE.sub("", text)
    return text


def _clean_asaxi(value: str) -> str:
    text = unicodedata.normalize("NFC", _strip_markdown(value))
    text = text.replace("«", "„").replace("»", "”")
    return _SPACE_RE.sub(" ", text).strip().lower()


def split_utterances(value: str) -> tuple[str, ...]:
    """Split terminal utterances while retaining balanced Polish quotes."""

    text = _clean_asaxi(value)
    raw_parts = tuple(
        part
        for match in _TERMINAL_RE.finditer(text)
        if (part := _clean_asaxi(match.group(0)))
        and af.words_in_text(part)
    )
    if not raw_parts:
        return (text,) if text else ()

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


def _frontmatter_end(lines: Sequence[str]) -> int:
    if not lines or lines[0].strip() != "---":
        return 0
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return index + 1
    return 0


def _frontmatter_title(lines: Sequence[str]) -> str:
    end = _frontmatter_end(lines)
    for line in lines[1:max(1, end - 1)]:
        if line.casefold().startswith("title:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return ""


def _looks_like_asaxi_paragraph(
    value: str,
    dictionary: af.AsaxiSynthesisDictionary,
) -> bool:
    text = _clean_asaxi(value)
    if not text:
        return False
    try:
        words = af.words_in_text(
            text,
            reject_unsupported_letters=True,
        )
    except ValueError:
        return False
    if not words:
        return False
    if _ASAXI_MARKER_RE.search(text):
        return True
    known = sum(dictionary.lookup(word) is not None for word in words)
    return known / len(words) >= 0.7


def _flush_markdown_paragraph(
    rows: list[GuideUtterance],
    paragraph: list[str],
    *,
    source_line: int,
    section: str,
    dictionary: af.AsaxiSynthesisDictionary,
) -> None:
    if not paragraph:
        return
    value = " ".join(part.strip() for part in paragraph if part.strip())
    if not _looks_like_asaxi_paragraph(value, dictionary):
        return
    for utterance in split_utterances(value):
        rows.append(GuideUtterance(
            text=utterance,
            section=section,
            source_line=source_line,
        ))


def extract_markdown_utterances(
    value: str,
    dictionary: af.AsaxiSynthesisDictionary,
) -> tuple[GuideUtterance, ...]:
    """Extract Asaxi prose from Markdown without reading its documentation."""

    lines = unicodedata.normalize("NFC", str(value or "")).splitlines()
    frontmatter_end = _frontmatter_end(lines)
    divider = next(
        (
            index
            for index in range(frontmatter_end, len(lines))
            if lines[index].strip() == "---"
        ),
        None,
    )
    content_start = divider + 1 if divider is not None else frontmatter_end

    rows: list[GuideUtterance] = []
    paragraph: list[str] = []
    paragraph_line = 0
    section = ""
    fenced = False

    def flush() -> None:
        nonlocal paragraph, paragraph_line
        _flush_markdown_paragraph(
            rows,
            paragraph,
            source_line=paragraph_line,
            section=section,
            dictionary=dictionary,
        )
        paragraph = []
        paragraph_line = 0

    for index in range(content_start, len(lines)):
        raw = lines[index]
        stripped = raw.strip()
        if stripped.startswith("```"):
            flush()
            fenced = not fenced
            continue
        if fenced:
            continue
        heading = re.match(r"^(#{2,6})\s+(.+?)\s*$", stripped)
        if heading:
            flush()
            section = _strip_markdown(heading.group(2)).strip()
            continue
        if (
            not stripped
            or stripped == "---"
            or stripped.startswith(("-", "*", "+", ">", "|", "<!--"))
            or stripped.casefold() == "navigation:"
        ):
            flush()
            continue
        if not paragraph:
            paragraph_line = index + 1
        paragraph.append(stripped)
    flush()
    return tuple(rows)


def extract_plain_utterances(value: str) -> tuple[GuideUtterance, ...]:
    rows: list[GuideUtterance] = []
    paragraphs = re.split(r"\n\s*\n", str(value or ""))
    line_cursor = 1
    for paragraph in paragraphs:
        paragraph_line = line_cursor
        line_cursor += paragraph.count("\n") + 2
        for utterance in split_utterances(paragraph):
            rows.append(GuideUtterance(
                text=utterance,
                source_line=paragraph_line,
            ))
    return tuple(rows)


def extract_utterances(
    value: str,
    dictionary: af.AsaxiSynthesisDictionary,
    *,
    input_format: str = "plain",
) -> tuple[GuideUtterance, ...]:
    mode = str(input_format or "plain").casefold()
    if mode == "markdown":
        rows = extract_markdown_utterances(value, dictionary)
    elif mode == "plain":
        rows = extract_plain_utterances(value)
    else:
        raise ValueError(f"Unsupported input format: {input_format!r}")
    if not rows:
        raise ValueError("No Asaxi utterances were found in the input.")
    return rows


def _speech_act(plan: asaxi_prosody.AsaxiProsodyPlan) -> str:
    if plan.interrogative:
        return "question"
    if plan.directive:
        return "directive"
    return "statement"


def _reading(plan: asaxi_prosody.AsaxiProsodyPlan) -> str:
    values = ".".join(mora.pitch for mora in plan.moras)
    return values + ("↗" if plan.boundary_tone == "LH%" else "")


def _boundary_description(tone: str) -> str:
    return {
        "L%": "closed assertion with a final fall",
        "LH%": "appeal contour with a final rise",
        "H%": "insistent high ending without a rise",
        "H-": "non-final continuation",
    }.get(str(tone), "unspecified boundary contour")


def _yaml_string(value: str) -> str:
    return '"' + str(value or "").replace("\\", "\\\\").replace(
        '"', '\\"'
    ) + '"'


def _warning_lines(
    diagnostics: Iterable[asaxi_prosody.AsaxiProsodyDiagnostic],
) -> list[str]:
    warnings = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.severity in {"warning", "error"}
    ]
    if not warnings:
        return []
    lines = [
        "> [!warning]- Model review notes",
    ]
    for diagnostic in warnings:
        lines.append(
            f"> - `{diagnostic.code}`: {diagnostic.message}"
        )
    return lines


def build_reading_guide(
    value: str,
    *,
    dictionary: Optional[af.AsaxiSynthesisDictionary] = None,
    input_format: str = "plain",
    title: str = "Asaxi Pitch Accent Reading Guide",
    source_label: str = "inline Asaxi text",
) -> ReadingGuide:
    dictionary = dictionary or asaxi_prosody.load_dictionary()
    source_name = Path(source_label).name
    rows = extract_utterances(
        value,
        dictionary,
        input_format=input_format,
    )
    source_digest = hashlib.sha256(
        unicodedata.normalize("NFC", str(value or "")).encode("utf-8")
    ).hexdigest()
    lines = [
        "---",
        f"title: {_yaml_string(title)}",
        "tags:",
        "  - Asaxi",
        "  - language",
        "  - prosody",
        "  - reading-guide",
        f"generated_by: asaxi_reading_guide.py/{GENERATOR_VERSION}",
        f"source_document: {_yaml_string(source_name)}",
        f"source_sha256: {source_digest}",
        "---",
        f"# {title}",
        "",
    ]
    if Path(source_name).suffix.casefold() in {".md", ".markdown"}:
        lines.extend([
            "Navigation:",
            f"- [[{Path(source_name).stem}|Source reading text]]",
            "",
        ])
    lines.extend([
        "## How To Read This Guide",
        "",
        "- `H` is a high mora-level target; `L` is a low target.",
        "- Dots separate morae. Word boundaries are shown by separate rows.",
        "- `↗` and boundary tone `LH%` mark a final appeal rise.",
        "- `L%` is a statement fall, `H%` an insistent high ending, and "
        "`H-` a continuation.",
        "- Dictionary / phrase accent is source data. Predicted in utterance "
        "shows the current model after morphology, phrasing, downstep, "
        "deaccenting, and boundary rules.",
        "- Read the Asaxi line naturally. The labels guide relative melody; "
        "they are not exact musical notes or stress marks.",
        "",
        (
            f"This guide contains {len(rows)} utterances from "
            f"`{source_name}`."
        ),
        "",
    ])
    warning_count = 0
    active_section = None
    for number, row in enumerate(rows, start=1):
        if row.section and row.section != active_section:
            active_section = row.section
            lines.extend([f"## {row.section}", ""])
        plan = asaxi_prosody.analyze_utterance(row.text, dictionary)
        warnings = _warning_lines(plan.diagnostics)
        warning_count += len([
            item for item in plan.diagnostics
            if item.severity in {"warning", "error"}
        ])
        lines.extend([
            f"### Utterance {number}",
            "",
            f'<span class="asaxi-text">{row.text}</span>',
            "",
            f"Predicted sentence reading: `{_reading(plan)}`",
            "",
            (
                f"Boundary tone: `{plan.boundary_tone}` "
                f"({_boundary_description(plan.boundary_tone)})"
            ),
            "",
            (
                f"Speech act: `{_speech_act(plan)}`"
                + (
                    f" · Source line: {row.source_line}"
                    if row.source_line
                    else ""
                )
            ),
            "",
            (
                "| # | Asaxi word | Morae | Dictionary / phrase accent | "
                "Predicted in utterance | Class | Morpheme analysis |"
            ),
            "|---:|---|---|---|---|---|---|",
        ])
        for word_number, word in enumerate(plan.words, start=1):
            word_moras = plan.moras[word.mora_start:word.mora_end]
            mora_text = " · ".join(mora.text for mora in word_moras)
            predicted = ".".join(mora.pitch for mora in word_moras)
            morphology = (
                asaxi_prosody.format_morpheme_analysis(
                    word.morphemes,
                    markdown=True,
                )
                if word.morphemes
                else "—"
            )
            lines.append(
                f"| {word_number} | {word.surface} | {mora_text} | "
                f"`{word.pitch_accent}` | `{predicted}` | "
                f"`{word.pitch_accent_class}` | {morphology} |"
            )
        lines.append("")
        if warnings:
            lines.extend(warnings)
            lines.append("")
    return ReadingGuide(
        markdown="\n".join(lines).rstrip() + "\n",
        utterance_count=len(rows),
        warning_count=warning_count,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="plain-text or Markdown input; omit to read standard input",
    )
    parser.add_argument(
        "--text",
        help="direct Asaxi text instead of an input file",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Markdown guide to write",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "plain", "markdown"),
        default="auto",
        help="input interpretation; auto uses the input file suffix",
    )
    parser.add_argument(
        "--title",
        default="",
        help="guide title; defaults to the source title when available",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=asaxi_prosody.DEFAULT_DICTIONARY_PATH,
        help="canonical Asaxi synthesis dictionary",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the existing output differs; do not write",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = _parser()
    args = parser.parse_args(argv)
    if args.input is not None and args.text is not None:
        parser.error("use either an input file or --text, not both")
    if args.output.suffix.casefold() != ".md":
        parser.error("--output must end in .md")

    if args.input is not None:
        input_path = args.input.resolve()
        output_path = args.output.resolve()
        if input_path == output_path:
            parser.error("input and output paths must be different")
        value = input_path.read_text(encoding="utf-8-sig")
        source_label = input_path.name
    elif args.text is not None:
        value = args.text
        source_label = "inline Asaxi text"
    else:
        value = sys.stdin.read()
        source_label = "standard input"

    mode = args.input_format
    if mode == "auto":
        mode = (
            "markdown"
            if args.input is not None
            and args.input.suffix.casefold() in {".md", ".markdown"}
            else "plain"
        )
    raw_lines = value.splitlines()
    source_title = _frontmatter_title(raw_lines) if mode == "markdown" else ""
    title = args.title or (
        f"{source_title}: Pitch Accent Guide"
        if source_title
        else "Asaxi Pitch Accent Reading Guide"
    )
    guide = build_reading_guide(
        value,
        dictionary=asaxi_prosody.load_dictionary(args.dictionary),
        input_format=mode,
        title=title,
        source_label=source_label,
    )
    output = args.output.resolve()
    if args.check:
        if not output.is_file():
            print(f"missing: {output}", file=sys.stderr)
            return 1
        if output.read_text(encoding="utf-8") != guide.markdown:
            print(f"stale: {output}", file=sys.stderr)
            return 1
        print(
            f"current: {guide.utterance_count} Asaxi utterances ({output})"
        )
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(guide.markdown, encoding="utf-8", newline="\n")
    print(
        f"wrote {guide.utterance_count} Asaxi utterances to {output}"
    )
    if guide.warning_count:
        print(
            f"review warnings retained: {guide.warning_count}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
