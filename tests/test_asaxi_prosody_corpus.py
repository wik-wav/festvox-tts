# -*- coding: utf-8 -*-
"""Deterministic Asaxi prosody-corpus tests."""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

import asaxi_prosody_corpus as corpus


class AsaxiProsodyCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not corpus.DEFAULT_VAULT_ROOT.is_dir():
            raise unittest.SkipTest(
                "private Asaxi source notes are not part of the public "
                "FestVox repository"
            )
        cls.build = corpus.build_corpus()
        cls.manifest = cls.build.manifest
        cls.prompts = cls.manifest["prompts"]

    def test_schema_and_balanced_strata(self) -> None:
        self.assertEqual(self.manifest["schema_version"], 2)
        self.assertEqual(self.manifest["corpus_id"], "asaxi-prosody-v1")
        self.assertGreaterEqual(len(self.prompts), 450)
        counts = self.build.coverage["by_kind"]
        self.assertEqual(counts["prosody_control"], 13)
        self.assertEqual(counts["fixed_expression"], 10)
        self.assertEqual(counts["lexical_citation"], 72)
        self.assertGreaterEqual(counts["grammar_example"], 130)
        self.assertGreaterEqual(counts["natural_reader"], 200)
        self.assertFalse(self.build.coverage["missing_required_tags"])

    def test_ids_and_recording_rows_are_unique(self) -> None:
        prompt_ids = [prompt["id"] for prompt in self.prompts]
        self.assertEqual(len(prompt_ids), len(set(prompt_ids)))
        rows = self.build.recording_script.splitlines()
        self.assertEqual(
            len(rows) - 1,
            self.build.coverage["recommended_take_count"],
        )
        recording_ids = [row.split("\t", 1)[0] for row in rows[1:]]
        self.assertEqual(len(recording_ids), len(set(recording_ids)))

    def test_asaxi_is_lowercase_clean_and_uses_polish_quotes(self) -> None:
        for prompt in self.prompts:
            text = prompt["asaxi"]
            self.assertEqual(text, text.lower(), prompt["id"])
            self.assertNotIn("«", text, prompt["id"])
            self.assertNotIn("»", text, prompt["id"])
            self.assertNotIn("**", text, prompt["id"])
            self.assertNotIn("[", text, prompt["id"])
            self.assertNotIn("]", text, prompt["id"])
            self.assertNotIn("\\", text, prompt["id"])
            self.assertEqual(
                text.count("„"),
                text.count("”"),
                prompt["id"],
            )

    def test_mined_grammar_examples_have_no_editorial_placeholders(self) -> None:
        grammar = [
            prompt for prompt in self.prompts
            if prompt["kind"] == "grammar_example"
        ]
        forbidden = re.compile(
            r"\b(?:statement|wizard|table|weather)\b",
            re.IGNORECASE,
        )
        self.assertFalse([
            prompt["asaxi"]
            for prompt in grammar
            if forbidden.search(prompt["asaxi"])
        ])

    def test_every_prompt_has_phone_mora_and_provenance_data(self) -> None:
        for prompt in self.prompts:
            analysis = prompt["analysis"]
            pitch = prompt["pitch_analysis"]
            transcript = pitch["predicted"]["word_transcript"]
            self.assertTrue(analysis["words"], prompt["id"])
            self.assertTrue(analysis["moras"], prompt["id"])
            self.assertTrue(analysis["phones"], prompt["id"])
            self.assertEqual(
                len(transcript),
                len(analysis["words"]),
                prompt["id"],
            )
            for item in transcript:
                self.assertTrue(item["word"], prompt["id"])
                self.assertTrue(
                    item["dictionary_or_phrase"],
                    prompt["id"],
                )
                self.assertTrue(
                    item["predicted_utterance"],
                    prompt["id"],
                )
                self.assertIsInstance(item["morphemes"], list)
                self.assertIsInstance(item["morphology"], str)
            self.assertTrue(prompt["source"]["note"], prompt["id"])
            self.assertTrue(prompt["coverage_tags"], prompt["id"])
            self.assertEqual(
                pitch["predicted"]["authority"],
                "model_prediction",
            )
            self.assertIn(pitch["reference"]["authority"], {
                "none",
                "attested",
                "dictionary",
                "source_attested_model_accent",
                "model_hypothesis",
            })
            self.assertIn(pitch["agreement"]["status"], {
                "no_reference",
                "exact",
                "mora_count_mismatch",
                "pitch_mismatch",
                "boundary_mismatch",
            })
            has_warning = any(
                item["severity"] in {"warning", "error"}
                for item in prompt["diagnostics"]
            )
            self.assertEqual(
                prompt["requires_linguistic_review"],
                has_warning,
                prompt["id"],
            )

    def test_attested_controls_are_preserved_exactly(self) -> None:
        controls = {
            prompt["id"]: prompt
            for prompt in self.prompts
            if prompt["kind"] == "prosody_control"
        }
        self.assertEqual(set(controls), {
            f"asx_ctl_{index:03d}" for index in range(1, 14)
        })
        self.assertEqual(
            controls["asx_ctl_002"]["pitch_analysis"]["reference"][
                "reading"
            ],
            "H.L.H.L.L",
        )
        self.assertEqual(
            controls["asx_ctl_008"]["speech_act"],
            "question",
        )
        self.assertEqual(
            controls["asx_ctl_013"]["annotation_status"],
            "attested",
        )
        exact = {
            prompt_id
            for prompt_id, prompt in controls.items()
            if prompt["pitch_analysis"]["agreement"]["status"] == "exact"
        }
        self.assertEqual(exact, {
            f"asx_ctl_{index:03d}"
            for index in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13)
        })
        self.assertEqual(
            {
                prompt_id: prompt["pitch_analysis"]["agreement"]["status"]
                for prompt_id, prompt in controls.items()
                if prompt_id not in exact
            },
            {
                "asx_ctl_010": "mora_count_mismatch",
                "asx_ctl_011": "mora_count_mismatch",
            },
        )

    def test_quoted_boundaries_survive_dialogue_splitting(self) -> None:
        quoted = [
            prompt for prompt in self.prompts
            if "punctuation:quotation" in prompt["coverage_tags"]
        ]
        self.assertGreaterEqual(len(quoted), 50)
        self.assertTrue(any(
            prompt["speech_act"] == "question" for prompt in quoted
        ))
        self.assertTrue(any(
            prompt["speech_act"] == "directive" for prompt in quoted
        ))

    def test_english_is_present_and_scope_is_explicit(self) -> None:
        scopes = {prompt["translation_scope"] for prompt in self.prompts}
        self.assertIn("exact", scopes)
        self.assertIn("terminal_aligned", scopes)
        self.assertIn("source_context", scopes)
        for prompt in self.prompts:
            self.assertTrue(prompt["translation_en"], prompt["id"])
            self.assertRegex(
                prompt["translation_en"],
                r"[A-Za-z]",
                prompt["id"],
            )

    def test_splitter_balances_multiple_dialogue_spans(self) -> None:
        parts = corpus._split_terminal_utterances(
            "„o,” tte ko zëjù, „sè no txănýj! "
            "no gaksamipỏpỏ zèxiŕa?”"
        )
        self.assertEqual(len(parts), 2)
        for part in parts:
            self.assertEqual(part.count("„"), part.count("”"), part)

    def test_manifest_paths_are_project_relative(self) -> None:
        encoded = json.dumps(self.manifest, ensure_ascii=False)
        self.assertNotRegex(encoded, r"[A-Za-z]:[\\/]")
        for source in self.manifest["sources"]:
            self.assertFalse(Path(source["path"]).is_absolute())
            self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")

    def test_reader_edition_is_translated_and_word_separated(self) -> None:
        reader = self.build.reader_corpus
        self.assertIn(
            "| # | Asaxi word | Dictionary / phrase accent | "
            "Predicted in utterance | Morpheme analysis | "
            "Reference evidence |",
            reader,
        )
        self.assertEqual(
            reader.count('<span class="asaxi-text">'),
            len(self.prompts),
        )
        self.assertEqual(
            reader.count("Word-level pitch analysis:"),
            len(self.prompts),
        )
        self.assertEqual(
            reader.count("Current model prediction:"),
            len(self.prompts),
        )
        self.assertEqual(
            reader.count("Reference agreement:"),
            len(self.prompts),
        )
        self.assertNotIn("Attested or dictionary reading:", reader)
        for prompt in self.prompts:
            self.assertIn(
                f"### ",
                reader,
            )
            self.assertIn(
                f"`{prompt['id']}`",
                reader,
            )
            english_label = (
                "English source context"
                if prompt["translation_scope"] == "source_context"
                else "English translation"
            )
            self.assertIn(
                f"{english_label}: {prompt['translation_en']}",
                reader,
            )
            self.assertNotIn(
                f"**{prompt['asaxi']}**",
                reader,
            )
            self.assertNotIn(
                f"*{prompt['asaxi']}*",
                reader,
            )

    def test_inflected_words_retain_structured_morpheme_analysis(self) -> None:
        plural_rows = [
            item
            for prompt in self.prompts
            for item in prompt["pitch_analysis"]["predicted"][
                "word_transcript"
            ]
            if item["word"] == "sháma"
        ]

        self.assertTrue(plural_rows)
        for item in plural_rows:
            self.assertEqual(item["dictionary_or_phrase"], "H.H")
            self.assertEqual(
                [
                    (morpheme["lemma"], morpheme["role"])
                    for morpheme in item["morphemes"]
                ],
                [("shá", "root"), ("-ma", "plural")],
            )
            self.assertEqual(
                item["morphology"],
                "shá (root) + -ma (plural)",
            )

    def test_recording_script_has_readable_pitch_columns(self) -> None:
        header = self.build.recording_script.splitlines()[0].split("\t")
        self.assertIn("translation_en", header)
        self.assertIn("translation_scope", header)
        self.assertIn("dictionary_pitch_by_word", header)
        self.assertIn("predicted_utterance_pitch_by_word", header)
        self.assertIn("morphology_by_word", header)
        self.assertIn("predicted_boundary_tone", header)
        self.assertIn("reference_authority", header)
        self.assertIn("reference_reading", header)
        self.assertIn("reference_agreement", header)

    def test_generation_is_byte_deterministic_and_current(self) -> None:
        second = corpus.build_corpus()
        self.assertEqual(self.build.files(), second.files())
        self.assertEqual(
            corpus.check_corpus(self.build, corpus.DEFAULT_OUTPUT_DIR),
            [],
        )


if __name__ == "__main__":
    unittest.main()
