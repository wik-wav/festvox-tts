# -*- coding: utf-8 -*-
"""Tests for the standalone Asaxi pitch-accent reading-guide generator."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import asaxi_prosody
import asaxi_reading_guide as guide


PROJECT_ROOT = Path(__file__).resolve().parents[2]
READER_PATH = (
    PROJECT_ROOT
    / "Lozenge-T-Notes"
    / "01_Worldbuilding"
    / "Asaxi"
    / "Texts"
    / "onă gaksamipỏpỏ (Reader's Text).md"
)


class AsaxiReadingGuideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dictionary = asaxi_prosody.load_dictionary()

    def test_plain_text_is_split_and_rendered_per_word(self) -> None:
        result = guide.build_reading_guide(
            "sè no txănýj. no kjo?",
            dictionary=self.dictionary,
            title="Reading Test",
            source_label=r"C:\private\input.txt",
        )

        self.assertEqual(result.utterance_count, 2)
        self.assertEqual(
            result.markdown.count('<span class="asaxi-text">'),
            2,
        )
        self.assertIn(
            "| # | Asaxi word | Morae | Dictionary / phrase accent | "
            "Predicted in utterance | Class | Morpheme analysis |",
            result.markdown,
        )
        self.assertIn("Boundary tone: `LH%`", result.markdown)
        self.assertIn("↗", result.markdown)
        self.assertIn('source_document: "input.txt"', result.markdown)
        self.assertNotIn(r"C:\private", result.markdown)
        self.assertNotIn("**sè no txănýj.**", result.markdown)
        self.assertNotIn("*sè no txănýj.*", result.markdown)

    def test_inflected_word_shows_its_morpheme_parse(self) -> None:
        result = guide.build_reading_guide(
            "sháma xogă.",
            dictionary=self.dictionary,
            source_label="sample.txt",
        )

        self.assertIn("`H.H`", result.markdown)
        self.assertIn(
            "`shá` (root) + `-ma` (plural)",
            result.markdown,
        )
        self.assertNotIn(
            "`no_matching_lexical_units`: 'sháma'",
            result.markdown,
        )

    def test_dotted_nasal_is_displayed_across_adjacent_syllables(self) -> None:
        result = guide.build_reading_guide(
            "găxănă ono kem.ma.",
            dictionary=self.dictionary,
            source_label="sample.txt",
        )

        self.assertIn("| kem.ma | kem · ma |", result.markdown)
        self.assertNotIn("| kem.ma | ke · mma |", result.markdown)

    def test_compound_shows_nested_lexical_units(self) -> None:
        result = guide.build_reading_guide(
            "gapỏbifùbiwa.",
            dictionary=self.dictionary,
            source_label="sample.txt",
        )

        self.assertIn("`L.H.L.L.L.L`", result.markdown)
        self.assertIn(
            "`ga-` (compound prefix) + "
            "`pỏbi` (compound modifier) + "
            "`fùbiwa` (compound head: "
            "`fùbi` (root) + `-wa` (plural))",
            result.markdown,
        )
        self.assertNotIn("no_matching_lexical_units", result.markdown)

    def test_polish_quotes_remain_balanced_when_dialogue_splits(self) -> None:
        parts = guide.split_utterances(
            "„o,” tte ko zëjù, „sè no txănýj! "
            "no gaksamipỏpỏ zèxiŕa?”"
        )

        self.assertEqual(len(parts), 2)
        for part in parts:
            self.assertEqual(part.count("„"), part.count("”"), part)

    def test_markdown_extraction_skips_documentation_and_navigation(self) -> None:
        source = """\
---
title: Example Reader
---
# Example

Navigation:
- [[The Asaxi Language]]

This English introduction must not become an utterance.

---

## I. sample — Sample

sè no txănýj.

- This list is editorial material.

> This blockquote is also editorial.

```text
no kjo?
```

no kjo?
"""
        rows = guide.extract_markdown_utterances(
            source,
            self.dictionary,
        )

        self.assertEqual(
            [row.text for row in rows],
            ["sè no txănýj.", "no kjo?"],
        )
        self.assertEqual(
            {row.section for row in rows},
            {"I. sample — Sample"},
        )

    def test_generation_is_byte_deterministic(self) -> None:
        first = guide.build_reading_guide(
            "xő xăcèna xiŕa.",
            dictionary=self.dictionary,
            source_label="sample.md",
        )
        second = guide.build_reading_guide(
            "xő xăcèna xiŕa.",
            dictionary=self.dictionary,
            source_label="sample.md",
        )

        self.assertEqual(first, second)
        self.assertIn("[[sample|Source reading text]]", first.markdown)
        self.assertRegex(
            first.markdown,
            r"source_sha256: [0-9a-f]{64}\n",
        )

    def test_cli_write_check_and_stale_detection(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.txt"
            output = root / "guide.md"
            source.write_text("sè no txănýj.", encoding="utf-8")
            argv = [str(source), "--output", str(output)]

            with redirect_stdout(io.StringIO()), redirect_stderr(
                io.StringIO()
            ):
                self.assertEqual(guide.main(argv), 0)
                self.assertEqual(guide.main([*argv, "--check"]), 0)
                output.write_text(
                    output.read_text(encoding="utf-8") + "stale\n",
                    encoding="utf-8",
                )
                self.assertEqual(guide.main([*argv, "--check"]), 1)

    def test_cli_refuses_to_overwrite_its_input(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.md"
            source.write_text("sè no txănýj.", encoding="utf-8")
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    guide.main([str(source), "--output", str(source)])

    def test_complete_reader_extracts_all_fifteen_sections(self) -> None:
        self.assertTrue(READER_PATH.is_file(), READER_PATH)
        source = READER_PATH.read_text(encoding="utf-8-sig")
        rows = guide.extract_markdown_utterances(
            source,
            self.dictionary,
        )
        sections = tuple(dict.fromkeys(
            row.section for row in rows if row.section
        ))

        self.assertGreaterEqual(len(rows), 240)
        self.assertEqual(len(sections), 15)
        self.assertTrue(sections[0].startswith("I. "))
        self.assertTrue(sections[-1].startswith("XV. "))
        extracted = "\n".join(row.text for row in rows)
        self.assertNotIn("Clean Asaxi text", extracted)
        self.assertNotIn("The Asaxi Language Index", extracted)


if __name__ == "__main__":
    unittest.main()
