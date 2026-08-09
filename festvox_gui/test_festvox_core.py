import json
import os
import unittest
import tempfile
import random
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import festvox_core as fc


class EnglishSyllabificationMetadataTests(unittest.TestCase):
    def test_english_synthesis_attaches_diagnostic_syllables(self):
        phones = ["pau", "r", "ae1", "b", "ih0", "t", "pau"]
        synthesis = fc.Synthesis(
            np.zeros(70, np.float32),
            100,
            [fc.Segment(phone, index * .1, (index + 1) * .1)
             for index, phone in enumerate(phones)],
            lang="en",
            phones=phones,
        )

        syllables = synthesis.english_syllabification["syllables"]

        self.assertEqual(
            [row["phones"] for row in syllables],
            [["r", "ae1"], ["b", "ih0", "t"]],
        )
        self.assertEqual(
            synthesis.english_syllabification["pause_indices"], [0, 6])

    def test_segment_phones_are_used_when_phone_list_is_absent(self):
        synthesis = fc.Synthesis(
            np.zeros(30, np.float32),
            100,
            [
                fc.Segment("hh", 0.0, .1),
                fc.Segment("eh1", .1, .2),
                fc.Segment("l", .2, .3),
            ],
            lang="English",
        )

        self.assertEqual(
            synthesis.english_syllabification["syllables"][0]["phones"],
            ["hh", "eh1", "l"],
        )

    def test_non_english_synthesis_does_not_attach_english_metadata(self):
        synthesis = fc.Synthesis(
            np.zeros(20, np.float32),
            100,
            [fc.Segment("a", 0.0, .2)],
            lang="ja",
            phones=["a"],
        )

        self.assertEqual(synthesis.english_syllabification, {})


class PhrasePauseTests(unittest.TestCase):
    def test_structural_target_remap_drops_only_deleted_phone_targets(self):
        old = [
            fc.Segment("a", 0.0, 0.10, uid="phone-a"),
            fc.Segment("t", 0.10, 0.20, uid="phone-t"),
            fc.Segment("i", 0.20, 0.30, uid="phone-i"),
        ]
        new = [
            fc.Segment("a", 0.0, 0.10, uid="phone-a"),
            fc.Segment("i", 0.10, 0.30, uid="phone-i"),
        ]
        targets = [(0.05, 170.0), (0.15, 210.0), (0.25, 185.0)]

        remapped = fc.remap_targets_aligned(targets, old, new)

        np.testing.assert_allclose(
            remapped,
            [(0.05, 170.0), (0.20, 185.0)],
            rtol=0.0, atol=1.0e-12,
        )
        # Whole-utterance scaling would incorrectly move the surviving /i/
        # target to 0.1667 seconds.
        self.assertNotAlmostEqual(remapped[-1][0], 1.0 / 6.0)

    def test_pause_anchor_rebuild_discards_stale_interior_pitch_points(self):
        entries = [
            ("a", 0.20),
            ("pau", 0.05), ("pau", 0.10),
            ("pau", 0.10), ("pau", 0.05),
            ("i", 0.20),
        ]
        stale = [
            (0.00, 190.0), (0.20, 180.0),
            (0.213, 245.0), (0.267, 132.0), (0.341, 231.0),
            (0.419, 118.0), (0.50, 205.0), (0.70, 195.0),
        ]

        first = fc.anchor_phrase_targets(entries, stale, 165.0)
        second = fc.anchor_phrase_targets(entries, first, 165.0)

        self.assertEqual(first, second)
        self.assertFalse(any(
            abs(time - stale_time) < 1.0e-9
            for time, _value in first
            for stale_time in (0.213, 0.267, 0.341, 0.419)
        ))
        interior = [
            (time, value) for time, value in first
            if 0.20 < time < 0.50
        ]
        self.assertEqual(len(interior), 7)
        values = [value for _time, value in interior]
        self.assertEqual(values, sorted(values))

    def test_sentence_line_breaks_are_phrase_boundaries(self):
        text = "first phrase\nsecond phrase\nthird phrase?"

        self.assertEqual(fc.split_sentence_phrases(text), [
            "first phrase", "second phrase", "third phrase?",
        ])
        self.assertEqual(fc.phrase_boundary_marks(text), [".", "."])
        self.assertEqual(
            fc.phrase_boundary_marks("first phrase.\nsecond phrase"),
            ["."])

    def test_single_explicit_pause_stays_inside_logical_phrase(self):
        self.assertEqual(
            fc.split_sentence_phrases(
                "first part [pau] still first. second part"),
            ["first part [pau] still first.", "second part"],
        )
        self.assertEqual(
            fc.split_sentence_phrases(
                "first phrase [pau] [pau] second phrase"),
            ["first phrase", "second phrase"],
        )

    def test_japanese_punctuation_splits_without_following_whitespace(self):
        text = "\u95a2\u4fc2\u306a\u3044\u3067\u3059\u3002\u6b21\u3067\u3059\uff1f\u307e\u3060\u3001\u7d9a\u304f\u3002"

        self.assertEqual(fc.text_phrase_chunks(text), [
            ("\u95a2\u4fc2\u306a\u3044\u3067\u3059\u3002", "."),
            ("\u6b21\u3067\u3059\uff1f", "?"),
            ("\u307e\u3060\u3001", ","),
            ("\u7d9a\u304f\u3002", "."),
        ])
        self.assertEqual(fc.split_sentence_phrases(text), [
            "\u95a2\u4fc2\u306a\u3044\u3067\u3059\u3002",
            "\u6b21\u3067\u3059\uff1f", "\u307e\u3060\u3001", "\u7d9a\u304f\u3002",
        ])

    def test_japanese_inline_quote_pause_is_bounded_but_preserved(self):
        source = [
            ("pau", .10), ("a", .20), ("r", .08),
            ("pau", .80), ("w", .08), ("a", .20), ("pau", .10),
        ]

        result = fc.retime_japanese_inline_pauses(
            source, "\u300c\u30a2\u30e9\u30fc\u30c8\u300d\u306f\u3002")

        self.assertEqual([phone for phone, _duration in result],
                         [phone for phone, _duration in source])
        self.assertAlmostEqual(result[3][1], .06)
        self.assertAlmostEqual(result[0][1], .10)
        self.assertAlmostEqual(result[-1][1], .10)

    def test_japanese_inline_pause_mapping_aborts_on_count_mismatch(self):
        source = [
            ("pau", .10), ("a", .20), ("pau", .80),
            ("w", .08), ("pau", .10),
        ]

        self.assertEqual(
            fc.retime_japanese_inline_pauses(
                source, "\u6587\u306b\u306f\u62ec\u5f27\u304c\u306a\u3044\u3002"),
            source,
        )

    def test_japanese_list_separators_use_semantic_pause_strength(self):
        source = [
            ("pau", .10), ("a", .20), ("pau", .80),
            ("i", .20), ("pau", .10),
        ]

        minor = fc.retime_japanese_inline_pauses(source, "\u672c\u5cf6\u30fb\u5927\u6771\u5cf6")
        major = fc.retime_japanese_inline_pauses(source, "\u898b\u51fa\u3057\u25bd\u9805\u76ee")

        self.assertAlmostEqual(minor[2][1], .12)
        self.assertAlmostEqual(major[2][1], .30)

    def test_explicit_inline_pause_is_not_retimed_as_a_quote(self):
        source = [
            ("pau", .10), ("a", .20), ("pau", .47),
            ("i", .20), ("pau", .10),
        ]

        self.assertEqual(
            fc.retime_japanese_inline_pauses(
                source, "\u524d [pau] \u5f8c"),
            source,
        )

    def test_asaxi_text_lowering_preserves_full_cap_language_switches(self):
        self.assertEqual(
            fc.normalize_synthesis_text("TaKI \u00cbn JOHN", "asaxi"),
            "taki \u00ebn JOHN")
        self.assertEqual(
            fc.normalize_synthesis_text("TaKI", "en"), "TaKI")

    def test_asaxi_wsl_synth_preserves_capitalized_terms_for_english_g2p(self):
        backend = fc.FestivalWSLBackend({})
        expected = fc.Synthesis(
            np.zeros(1, np.float32), 16000, [], lang="asaxi")
        with patch.object(
                backend, "_synth_asaxi",
                return_value=expected) as synth_asaxi:
            rendered = backend.synth(
                "To JOHN an\u0151 apo ch\u1ecfn\u016f.",
                "asaxi",
                "fixture",
            )

        self.assertIs(rendered, expected)
        self.assertEqual(
            synth_asaxi.call_args.args[0],
            "to JOHN an\u0151 apo ch\u1ecfn\u016f.",
        )

    def test_asaxi_synth_routes_mora_tone_and_pitch_to_language_branch(self):
        backend = fc.FestivalWSLBackend({})
        expected = fc.Synthesis(
            np.zeros(1, np.float32), 16000, [], lang="asaxi")
        with patch.object(
                backend, "_synth_asaxi",
                return_value=expected) as synth_asaxi:
            rendered = backend.synth(
                "Shěso",
                "asaxi",
                "fixture",
                asaxi_tone_overrides={"1": "L"},
                asaxi_pitch_offsets_cents={"1": 125},
            )

        self.assertIs(rendered, expected)
        self.assertEqual(synth_asaxi.call_args.args[0], "shěso")
        self.assertEqual(
            synth_asaxi.call_args.kwargs["asaxi_tone_overrides"],
            {"1": "L"},
        )
        self.assertEqual(
            synth_asaxi.call_args.kwargs["asaxi_pitch_offsets_cents"],
            {"1": 125},
        )

    def test_internal_pause_is_split_without_changing_total_duration(self):
        source = [("pau", 0.15), ("t", 0.06), ("pau", 0.48),
                  ("hh", 0.09), ("pau", 0.15)]

        result = fc.split_internal_pauses(source, lead_pause=0.12)

        self.assertEqual([p for p, _ in result],
                         ["pau", "t", "pau", "pau", "pau", "pau",
                          "hh", "pau"])
        self.assertAlmostEqual(result[2][1], 0.12)
        self.assertAlmostEqual(result[3][1], 0.14)
        self.assertAlmostEqual(result[4][1], 0.14)
        self.assertAlmostEqual(result[5][1], 0.08)
        self.assertAlmostEqual(sum(d for _, d in result),
                               sum(d for _, d in source))

    def test_existing_double_pause_becomes_four_without_changing_total(self):
        source = [("pau", 0.15), ("t", 0.06), ("pau", 0.10),
                  ("pau", 0.30), ("hh", 0.09), ("pau", 0.15)]

        result = fc.split_internal_pauses(source)
        self.assertEqual([phone for phone, _duration in result],
                         ["pau", "t", "pau", "pau", "pau", "pau",
                          "hh", "pau"])
        self.assertAlmostEqual(sum(duration for _phone, duration in result),
                               sum(duration for _phone, duration in source))

    def test_existing_three_pause_edit_splits_only_middle_gap(self):
        source = [("a", .1), ("pau", .07), ("pau", .42),
                  ("pau", .05), ("b", .1)]
        result = fc.split_internal_pauses(source)

        self.assertEqual([phone for phone, _duration in result],
                         ["a", "pau", "pau", "pau", "pau", "b"])
        self.assertEqual(result[1][1], .07)
        self.assertEqual(result[2][1], .21)
        self.assertEqual(result[3][1], .21)
        self.assertEqual(result[4][1], .05)
        self.assertAlmostEqual(sum(duration for _phone, duration in result),
                               sum(duration for _phone, duration in source))

    def test_existing_four_pause_edit_is_retained_verbatim(self):
        source = [
            ("a", .1), ("pau", .05), ("pau", .19),
            ("pau", .11), ("pau", .06), ("b", .1),
        ]

        self.assertEqual(fc.split_internal_pauses(source), source)

    def test_utterance_edges_are_paired_without_changing_duration(self):
        source = [("pau", .15), ("a", .20), ("pau", .12)]

        result = fc.split_edge_pauses(source, guard_pause=.08)

        self.assertEqual([phone for phone, _duration in result],
                         ["pau", "pau", "a", "pau", "pau"])
        self.assertAlmostEqual(result[1][1], .08)
        self.assertAlmostEqual(result[-2][1], .08)
        self.assertAlmostEqual(sum(duration for _phone, duration in result),
                               sum(duration for _phone, duration in source))
        self.assertEqual(fc.split_edge_pauses(result), result)

    def test_short_pause_splits_evenly(self):
        result = fc.split_internal_pauses(
            [("a", 0.1), ("pau", 0.08), ("b", 0.1)], lead_pause=0.12)

        self.assertEqual([phone for phone, _duration in result],
                         ["a", "pau", "pau", "pau", "pau", "b"])
        self.assertAlmostEqual(result[1][1], .032)
        self.assertAlmostEqual(result[2][1], .04 / 3.0)
        self.assertAlmostEqual(result[3][1], .04 / 3.0)
        self.assertAlmostEqual(result[4][1], .08 * 4.0 / 15.0)
        self.assertAlmostEqual(sum(duration for _phone, duration in result),
                               .28)

    def test_single_pause_fault_collapses_explicit_pause_runs(self):
        result = fc.collapse_pause_runs([
            ("pau", .1), ("a", .2), ("pau", .12), ("pau", .31),
            ("b", .2), ("pau", .1)])

        self.assertEqual([phone for phone, _dur in result],
                         ["pau", "a", "pau", "b", "pau"])
        self.assertAlmostEqual(result[2][1], .43)

    def test_phrase_combiner_keeps_four_pauses_unless_fault_is_enabled(self):
        first = fc.Synthesis(
            np.zeros(40, np.float32), 100,
            [fc.Segment("pau", 0.0, .1), fc.Segment("t", .1, .3),
             fc.Segment("pau", .3, .4)], voicebank="one")
        second = fc.Synthesis(
            np.zeros(40, np.float32), 100,
            [fc.Segment("pau", 0.0, .1), fc.Segment("h", .1, .3),
             fc.Segment("pau", .3, .4)], voicebank="two")

        normal = fc.combine_syntheses([first, second])
        faulted = fc.combine_syntheses(
            [first, second], single_pause=True)

        self.assertEqual([segment.phone for segment in normal.segments],
                         ["pau", "t", "pau", "pau", "pau", "pau",
                          "h", "pau"])
        self.assertEqual([segment.phone for segment in faulted.segments],
                         ["pau", "t", "pau", "h", "pau"])
        self.assertEqual(len(normal.samples), 80)
        self.assertEqual(len(faulted.samples), 70)

    def test_phrase_combiner_offsets_frame_trajectory_provenance(self):
        first = fc.Synthesis(
            np.zeros(40, np.float32), 100,
            [fc.Segment("a", 0.0, .4)],
            target_pitchmarks=[.01, .02],
            frame_trajectory_records=[{
                "target_index": 1, "time": .02, "segment_index": 0,
                "centre_offset_samples": -3,
            }])
        second = fc.Synthesis(
            np.zeros(40, np.float32), 100,
            [fc.Segment("e", 0.0, .4)],
            target_pitchmarks=[.01, .02],
            frame_trajectory_records=[{
                "target_index": 1, "time": .02, "segment_index": 0,
                "centre_offset_samples": 4,
            }])

        combined = fc.combine_syntheses([first, second])

        self.assertEqual(
            [row["target_index"]
             for row in combined.frame_trajectory_records],
            [1, 3])
        self.assertEqual(
            [row["segment_index"]
             for row in combined.frame_trajectory_records],
            [0, 1])
        self.assertEqual(
            [row["phrase_index"]
             for row in combined.frame_trajectory_records],
            [0, 1])
        self.assertAlmostEqual(
            combined.frame_trajectory_records[1]["time"], .42)

    def test_phrase_combiner_preserves_paired_edges_as_four_at_join(self):
        def phrase(phone):
            return fc.Synthesis(
                np.zeros(40, np.float32), 100,
                [fc.Segment("pau", 0, .05),
                 fc.Segment("pau", .05, .10),
                 fc.Segment(phone, .10, .30),
                 fc.Segment("pau", .30, .35),
                 fc.Segment("pau", .35, .40)])

        combined = fc.combine_syntheses([phrase("a"), phrase("b")])

        self.assertEqual([segment.phone for segment in combined.segments],
                         ["pau", "pau", "a", "pau", "pau", "pau", "pau",
                          "b", "pau", "pau"])
        self.assertEqual(len(combined.samples), 80)

    def test_phrase_combiner_preserves_structural_phone_source_alignment(self):
        first_segments = fc.segments_from_durations([
            ("pau", .05), ("i", .10), ("cl", .08),
            ("s", .10), ("pau", .05),
        ])
        second_segments = fc.segments_from_durations([
            ("pau", .05), ("a", .10), ("pau", .05),
        ])
        first = fc.Synthesis(
            np.zeros(380, np.float32), 1000, first_segments,
            phones=["i", "cl", "s"],
            render_phones=["pau", "i", "s", "s", "pau"],
            special_phone_realizations=[{
                "index": 2,
                "phone": "cl",
                "mode": "anticipatory_consonant",
                "source_phone": "s",
                "status": "resolved",
            }],
        )
        second = fc.Synthesis(
            np.zeros(200, np.float32), 1000, second_segments,
            phones=["a"],
            render_phones=["pau", "a", "pau"],
        )

        combined = fc.combine_syntheses([first, second])

        self.assertEqual(
            len(combined.render_phones), len(combined.segments))
        closure_index = next(
            index for index, segment in enumerate(combined.segments)
            if segment.phone == "cl"
        )
        self.assertEqual(combined.render_phones[closure_index], "s")
        self.assertEqual(
            combined.special_phone_realizations[0]["index"],
            closure_index,
        )

    def test_phrase_combiner_preserves_completed_phrase_calibration(self):
        def phrase(gain):
            return fc.Synthesis(
                np.ones(40, np.float32) * .05, 100,
                [fc.Segment("pau", 0, .1),
                 fc.Segment("a", .1, .3),
                 fc.Segment("pau", .3, .4)],
                voicebank="voice",
                automatic_gain_db=gain,
                output_calibration={
                    "method": "active_speech_rms",
                    "applied": True,
                    "scope": "completed_phrase",
                })

        combined = fc.combine_syntheses([phrase(2.0), phrase(4.0)])

        self.assertEqual(
            combined.output_calibration["method"], "per_phrase")
        self.assertEqual(
            combined.output_calibration["calibrated_phrase_count"], 2)
        self.assertEqual(
            [row["automatic_gain_db"] for row in
             combined.output_calibration["phrases"]], [2.0, 4.0])

    def test_phrase_combiner_preserves_asaxi_word_and_idiom_provenance(self):
        first = fc.Synthesis(
            np.zeros(20, np.float32), 100,
            [fc.Segment("a", 0.0, .2)],
            asaxi_prosody={
                "schema_version": 1,
                "dictionary_ruleset": "asaxi-pitch-v1",
                "phrase_count": 1,
                "word_count": 2,
                "mora_count": 3,
                "rendered_phones": ["a"],
                "moras": [{
                    "mora_index": 0,
                    "segment_indices": [0],
                    "start": 0.0,
                    "end": 0.2,
                }],
                "phrases": [{
                    "source_text": "ga vi",
                    "matched_expression": "ga vi",
                }],
            },
        )
        second = fc.Synthesis(
            np.zeros(20, np.float32), 100,
            [fc.Segment("e", 0.0, .2)],
            asaxi_prosody={
                "schema_version": 1,
                "dictionary_ruleset": "asaxi-pitch-v1",
                "phrase_count": 1,
                "word_count": 1,
                "mora_count": 2,
                "rendered_phones": ["e"],
                "moras": [{
                    "mora_index": 0,
                    "segment_indices": [0],
                    "start": 0.0,
                    "end": 0.2,
                }],
                "phrases": [{"source_text": "ma"}],
            },
        )

        combined = fc.combine_syntheses([first, second], lang="asaxi")

        self.assertEqual(
            combined.asaxi_prosody["kind"],
            "asaxi_phrase_sequence_prosody",
        )
        self.assertEqual(combined.asaxi_prosody["phrase_count"], 2)
        self.assertEqual(combined.asaxi_prosody["word_count"], 3)
        self.assertEqual(combined.asaxi_prosody["mora_count"], 5)
        self.assertEqual(
            combined.asaxi_prosody["phrases"][0]["phrases"][0][
                "matched_expression"
            ],
            "ga vi",
        )
        self.assertEqual(
            [row["mora_index"] for row in
             combined.asaxi_prosody["moras"]],
            [0, 1],
        )
        self.assertEqual(
            [row["segment_indices"] for row in
             combined.asaxi_prosody["moras"]],
            [[0], [1]],
        )

    def test_festival_addenda_accepts_accented_asaxi_and_multiword_text(self):
        addenda = fc.FestivalWSLBackend._dict_addenda(
            "Shěsonů ga vi.",
            {
                "shěsonů": ["sh", "eh", "s", "ao", "n", "uw"],
                "ga": ["g", "a"],
                "vi": ["v", "i"],
            },
        )

        self.assertIn('("shěsonů" nil', addenda)
        self.assertIn('("ga" nil', addenda)
        self.assertIn('("vi" nil', addenda)

    def test_capitalized_asaxi_terms_use_attested_and_english_frontends(self):
        class PronunciationBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {}})
                self.calls = []

            def phonemize(self, words, voicebank, lang=""):
                self.calls.append((tuple(words), voicebank, lang))
                return {"alice": ["ae", "l", "ih", "s"]}

        backend = PronunciationBackend()
        pronunciations = backend._capitalized_asaxi_pronunciations(
            "to JOHN ja ALICE.", "voice"
        )

        self.assertEqual(pronunciations["john"], ("jh", "ao", "n"))
        self.assertEqual(
            pronunciations["alice"], ("ae", "l", "ih", "s"))
        self.assertEqual(
            backend.calls,
            [(("ALICE",), "voice", "en")],
        )

    def test_user_dictionary_remains_final_for_capitalized_asaxi_term(self):
        class NoFrontendBackend(fc.FestivalWSLBackend):
            def phonemize(self, words, voicebank, lang=""):
                raise AssertionError("explicit pronunciation must skip G2P")

        backend = NoFrontendBackend({"festival_wsl": {}})
        pronunciations = backend._capitalized_asaxi_pronunciations(
            "to JOHN.", "voice", {"john": ["jh", "aa", "n"]}
        )

        self.assertEqual(pronunciations, {})

    def test_asaxi_seed_injects_inferred_dotted_geminate_pronunciation(self):
        class RecordingBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {}})
                self.addenda = ""

            def _synth_common(self, body, voicebank, speed, text, lang,
                              **kwargs):
                self.addenda = kwargs.get("addenda", "")
                entries = [
                    ("pau", .05), ("k", .06), ("e", .10),
                    ("m", .06), ("m", .06), ("a", .10), ("pau", .05),
                ]
                return fc.Synthesis(
                    np.zeros(480, np.float32),
                    1000,
                    fc.segments_from_durations(entries),
                    text=text,
                    lang=lang,
                    voicebank=voicebank,
                )

        backend = RecordingBackend()
        dictionary = backend._asaxi_dictionary()

        seed, plan, _diagnostics = backend._asaxi_seed(
            "kem.ma",
            "voice",
            1.0,
            165.0,
            10.0,
            False,
            {},
            dictionary,
            dictionary.pronunciations(),
            {},
        )

        self.assertEqual(
            list(plan.words[0].phones), ["k", "e", "m", "m", "a"])
        self.assertIn(
            '("kem.ma" nil (((k e m m a) 1)))',
            backend.addenda,
        )
        spoken_durations = [
            round(segment.dur, 6)
            for segment in seed.segments
            if segment.phone != "pau"
        ]
        self.assertGreater(len(set(spoken_durations)), 1)
        self.assertLess(sum(spoken_durations[:3]), 0.18)
        self.assertEqual(
            seed.asaxi_prosody["duration_plan"]["model_id"],
            "asaxi-moraic-rules-v1",
        )
        self.assertTrue(seed.targets)
        self.assertLessEqual(
            max(time for time, _frequency in seed.targets),
            seed.segments[-1].end,
        )

    def test_asaxi_seed_repairs_missing_palatalized_bank_transition(self):
        class RecordingBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {}})
                self.addenda = ""

            def voice_metadata(self, voicebank):
                return {
                    "index": {
                        key: {}
                        for key in (
                            "b-o", "o-w", "w-hy",
                            "w-h", "h-y", "y-y", "y-ao",
                        )
                    },
                }

            def _synth_common(self, body, voicebank, speed, text, lang,
                              **kwargs):
                self.addenda = kwargs.get("addenda", "")
                entries = [
                    ("pau", .05), ("b", .04), ("o", .06), ("w", .03),
                    ("h", .04), ("y", .03), ("y", .03), ("ao", .08),
                    ("pau", .05),
                ]
                return fc.Synthesis(
                    np.zeros(500, np.float32),
                    1000,
                    fc.segments_from_durations(entries),
                    text=text,
                    lang=lang,
                    voicebank=voicebank,
                )

        backend = RecordingBackend()
        dictionary = backend._asaxi_dictionary()
        text = "b\u1ecfhj\u00e1"

        seed, plan, diagnostics = backend._asaxi_seed(
            text,
            "voice",
            1.0,
            165.0,
            10.0,
            False,
            {},
            dictionary,
            dictionary.pronunciations(),
            {},
        )

        self.assertEqual(
            plan.phones,
            ("b", "o", "w", "h", "y", "y", "ao"),
        )
        self.assertIn(
            f'("{text}" nil (((b o w h y y ao) 1)))',
            backend.addenda,
        )
        self.assertEqual(
            seed.asaxi_prosody["phone_fallbacks"][0][
                "missing_canonical_diphones"
            ],
            ["hy-ao"],
        )
        self.assertIn(
            "asaxi_phone_fallback_applied",
            {diagnostic.code for diagnostic in diagnostics},
        )

    def test_asaxi_equal_timing_fault_overrides_mora_duration_model(self):
        class RecordingBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {}})
                self.explicit_entries = []
                self.previous_targets = []

            def _synth_common(self, body, voicebank, speed, text, lang,
                              **kwargs):
                entries = [
                    ("pau", .05), ("k", .10), ("e", .10),
                    ("m", .10), ("m", .10), ("a", .10), ("pau", .05),
                ]
                return fc.Synthesis(
                    np.zeros(600, np.float32),
                    1000,
                    fc.segments_from_durations(entries),
                    text=text,
                    lang=lang,
                    voicebank=voicebank,
                )

            def synth_phones(self, phones, voicebank, **kwargs):
                self.explicit_entries = list(kwargs["seg_durs"])
                self.previous_targets = list(kwargs["prev_targets"])
                return fc.Synthesis(
                    np.zeros(1, np.float32),
                    1000,
                    fc.segments_from_durations(self.explicit_entries),
                    text=kwargs.get("text", ""),
                    lang=kwargs.get("lang", ""),
                    voicebank=voicebank,
                )

        backend = RecordingBackend()
        result = backend.synth(
            "kem.ma",
            "asaxi",
            "voice",
            speed=1.0,
            pitch=165.0,
            fall=10.0,
            fault_mode={"disable_phone_timing": True},
        )

        spoken = [
            duration for phone, duration in backend.explicit_entries
            if phone != "pau"
        ]
        self.assertEqual(spoken, [0.10] * 5)
        self.assertEqual(
            result.asaxi_prosody["duration_fault_override"],
            "equal_phone_timing",
        )
        self.assertEqual(
            result.asaxi_prosody["duration_model_id"],
            "asaxi-moraic-rules-v1",
        )
        self.assertEqual(
            result.asaxi_prosody["pitch_model_id"],
            "asaxi-hierarchical-log-f0-v1",
        )
        trace = result.asaxi_prosody["prosody_trace"]
        self.assertEqual(trace["cumulative_frequency_drift"], "disabled")
        self.assertEqual(
            backend.previous_targets,
            [
                (row["time_seconds"], row["final_f0_hz"])
                for row in trace["trajectory"]
            ],
        )

    def test_equal_timing_preserves_pause_lengths(self):
        result = fc.equalize_phone_durations(
            [("pau", 0.2), ("t", 0.06), ("aa", 0.14), ("pau", 0.4)],
            phone_dur=0.1)

        self.assertEqual(result, [("pau", 0.2), ("t", 0.1),
                                  ("aa", 0.1), ("pau", 0.4)])

    def test_moraic_nasal_timing_role_does_not_reclassify_plain_nn(self):
        self.assertTrue(fc.is_timing_nucleus("N", "moraic_nasal"))
        self.assertTrue(fc.is_timing_nucleus("nn", "moraic_nasal"))
        self.assertFalse(fc.is_timing_nucleus("nn"))
        self.assertFalse(fc.is_timing_nucleus("n", "consonant"))

    def test_question_block_rises_and_stays_in_range(self):
        result = fc.intonation_targets(
            [{"start": 0.2, "end": 0.8, "kind": "?"}], 185.0, 10.0)

        self.assertEqual(len(result), 3)
        self.assertGreater(result[-1][1], result[0][1])
        self.assertTrue(all(fc.PITCH_MIN_HZ <= f <= fc.PITCH_MAX_HZ
                            for _, f in result))

    def test_phrase_blocks_treat_double_pause_as_one_boundary(self):
        segs = [fc.Segment("pau", 0.0, 0.1), fc.Segment("a", 0.1, 0.2),
                fc.Segment("pau", 0.2, 0.3), fc.Segment("pau", 0.3, 0.7),
                fc.Segment("b", 0.7, 0.8), fc.Segment("pau", 0.8, 0.9)]

        self.assertEqual(fc.phrase_blocks(segs, "one? two."), [
            {"start": 0.1, "end": 0.2, "kind": "?"},
            {"start": 0.7, "end": 0.8, "kind": "."},
        ])

    def test_phrase_playback_spans_keep_extra_acoustic_phrase_and_pauses(self):
        phones = [
            "pau", "a", "b", "pau", "pau", "pau", "pau",
            "c", "d", "pau", "e", "f", "pau",
        ]
        segments = [
            fc.Segment(phone, index * .1, (index + 1) * .1)
            for index, phone in enumerate(phones)
        ]

        spans = fc.phrase_playback_spans(segments, [3, 7])

        self.assertEqual(len(spans), 2)
        self.assertEqual(
            [(row["start_index"], row["end_index"]) for row in spans],
            [(0, 4), (5, 12)],
        )
        self.assertEqual(
            [(row["spoken_start_index"], row["spoken_end_index"])
             for row in spans],
            [(1, 2), (7, 11)],
        )
        self.assertEqual(
            [index for row in spans
             for index in range(row["start_index"],
                                row["end_index"] + 1)],
            list(range(len(segments))),
        )

    def test_phrase_playback_spans_split_four_pause_boundary_two_and_two(self):
        phones = [
            "pau", "pau", "a", "pau", "pau",
            "pau", "pau", "b", "pau", "pau",
        ]
        segments = [
            fc.Segment(phone, index * .1, (index + 1) * .1)
            for index, phone in enumerate(phones)
        ]

        spans = fc.phrase_playback_spans(segments, [1, 1])

        self.assertEqual(
            [(row["start_index"], row["end_index"]) for row in spans],
            [(0, 4), (5, 9)],
        )
        self.assertEqual(spans[0]["spoken_start_index"], 2)
        self.assertEqual(spans[1]["spoken_start_index"], 7)

    def test_phrase_blocks_normalize_japanese_punctuation(self):
        segs = [fc.Segment("pau", 0.0, 0.1), fc.Segment("a", 0.1, 0.2),
                fc.Segment("pau", 0.2, 0.3), fc.Segment("b", 0.3, 0.4),
                fc.Segment("pau", 0.4, 0.5)]

        self.assertEqual(fc.phrase_blocks(segs, "これは？ つぎ。"), [
            {"start": 0.1, "end": 0.2, "kind": "?"},
            {"start": 0.3, "end": 0.4, "kind": "."},
        ])

    def test_phrase_plan_uses_four_pauses_and_single_fault_keeps_total(self):
        a = fc.Synthesis(np.zeros(1, np.float32), 1, [
            fc.Segment("pau", 0.0, 0.1), fc.Segment("t", 0.1, 0.2),
            fc.Segment("pau", 0.2, 0.3)], targets=[(0.15, 180)])
        b = fc.Synthesis(np.zeros(1, np.float32), 1, [
            fc.Segment("pau", 0.0, 0.1), fc.Segment("hh", 0.1, 0.2),
            fc.Segment("pau", 0.2, 0.3)], targets=[(0.15, 170)])

        double, _, _ = fc.merge_phrase_plans([a, b], [".", "."])
        single, _, _ = fc.merge_phrase_plans(
            [a, b], [".", "."], single_pause=True)

        self.assertEqual([p for p, _ in double],
                         ["pau", "t", "pau", "pau", "pau", "pau",
                          "hh", "pau"])
        self.assertEqual([p for p, _ in single],
                         ["pau", "t", "pau", "hh", "pau"])
        self.assertAlmostEqual(sum(d for _, d in double),
                               sum(d for _, d in single))

    def test_semantic_phrase_pause_settings_control_total_duration(self):
        settings = {"minor": 140, "major": 320, "sentence": 610}
        for mark, expected in ((",", .140), (":", .320), ("?", .610)):
            parts = fc.phrase_pause_durations_with_settings(
                mark, 1.0, settings)
            self.assertEqual(len(parts), 4)
            self.assertAlmostEqual(sum(parts), expected)
            self.assertTrue(all(value >= 0.0 for value in parts))
        self.assertEqual(fc.normalize_phrase_pauses_ms({
            "minor": -5, "major": 9999, "sentence": "bad",
        }), {"minor": 0, "major": 2000, "sentence": 500})

    def test_phrase_pause_retimer_preserves_spoken_phones_and_indexes(self):
        original = [
            ("pau", .05), ("a", .12),
            ("pau", .08), ("pau", .10), ("pau", .06),
            ("k", .07), ("a", .12), ("pau", .05),
        ]
        result = fc.retime_internal_phrase_pauses(
            original, "one, two.", settings={
                "minor": 150, "major": 300, "sentence": 600})

        self.assertEqual([phone for phone, _duration in result],
                         [phone for phone, _duration in original])
        self.assertEqual(result[1], original[1])
        self.assertEqual(result[5:7], original[5:7])
        self.assertAlmostEqual(sum(duration for _phone, duration
                                   in result[2:5]), .150)

    def test_generated_voice_delete_rejects_utau_and_removes_temp_db(self):
        with tempfile.TemporaryDirectory() as root:
            utau = Path(root) / "source"
            utau.mkdir()
            (utau / "oto.ini").write_text("", encoding="utf-8")
            with self.assertRaises(fc.BackendError):
                fc.validate_generated_voice_dir(utau)

            multipitch = Path(root) / "multipitch-source"
            subbank = multipitch / "F3"
            subbank.mkdir(parents=True)
            (subbank / "oto.ini").write_text("", encoding="utf-8")
            with self.assertRaises(fc.BackendError):
                fc.validate_generated_voice_dir(multipitch)

            generated = Path(root) / "generated"
            (generated / "dic").mkdir(parents=True)
            (generated / "dic" / "diphone_index.json").write_text(
                "{}", encoding="utf-8")
            fc.delete_generated_voice_dir(generated)
            self.assertFalse(generated.exists())

    def test_missing_wsl_voice_can_be_unregistered_without_delete(self):
        class MissingVoiceBackend(fc.FestivalWSLBackend):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.commands = []

            def _run(self, args, timeout=None):
                self.commands.append(list(args))
                return SimpleNamespace(returncode=1, stdout="", stderr="")

        cfg = {"festival_wsl": {
            "voices": {"missing": {
                "dir": "/home/test/voices/missing",
                "voice": "voice_missing",
            }},
            "installed_voices": ["missing", "kal_diphone"],
            "default_voice": "missing",
        }}
        backend = MissingVoiceBackend(cfg)
        backend._alternatives["missing"] = {"a-b": []}
        backend._voice_metadata["missing"] = {"language": "ja"}
        backend._sustains[("missing", "a")] = (np.zeros(8), 16000)

        info = backend.voicebank_removal_info("missing")
        self.assertFalse(info["exists"])
        backend.uninstall_voicebank("missing", delete_files=False)

        self.assertNotIn("missing", cfg["festival_wsl"]["voices"])
        self.assertEqual(cfg["festival_wsl"]["installed_voices"],
                         ["kal_diphone"])
        self.assertEqual(cfg["festival_wsl"]["default_voice"], "")
        self.assertNotIn("missing", backend._alternatives)
        self.assertNotIn("missing", backend._voice_metadata)
        self.assertNotIn(("missing", "a"), backend._sustains)
        self.assertFalse(any(command and command[0] == "rm"
                             for command in backend.commands))

    def test_diphone_database_eviction_clears_audio_and_sustains(self):
        class Database:
            cleared = False

            def clear_cache(self):
                self.cleared = True

        backend = object.__new__(fc.DiphoneBackend)
        database = Database()
        backend._dbs = {"old": database}
        backend._sustains = {
            ("old", "a"): (np.ones(8), 16000),
            ("keep", "a"): (np.ones(8), 16000),
        }

        backend._drop_database("old")

        self.assertTrue(database.cleared)
        self.assertNotIn("old", backend._dbs)
        self.assertNotIn(("old", "a"), backend._sustains)
        self.assertIn(("keep", "a"), backend._sustains)

    def test_stale_installed_voice_registration_can_be_removed(self):
        cfg = {"festival_wsl": {
            "installed_voices": ["stale_voice", "kal_diphone"],
            "default_voice": "stale_voice",
        }}
        backend = fc.FestivalWSLBackend(cfg)

        info = backend.voicebank_removal_info("stale_voice")

        self.assertFalse(info["exists"])
        self.assertEqual(info["kind"], "registration")
        self.assertIn("stale_voice", info["path"])
        backend.uninstall_voicebank("stale_voice", delete_files=False)
        self.assertEqual(cfg["festival_wsl"]["installed_voices"],
                         ["kal_diphone"])
        self.assertEqual(cfg["festival_wsl"]["default_voice"], "")

    def test_kal_diphone_is_available_without_a_saved_voice_scan(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {}})

        voices = {voice["name"]: voice for voice in backend.voicebanks()}

        self.assertIn("kal_diphone", voices)
        self.assertEqual(
            voices["kal_diphone"]["dir"],
            "/usr/share/festival/voices/english/kal_diphone",
        )
        self.assertEqual(voices["kal_diphone"]["source"],
                         "Festival built-in")
        self.assertEqual(voices["kal_diphone"]["supported_languages"],
                         ["en"])
        self.assertEqual(backend.default_voicebank(), "kal_diphone")

    def test_stale_kal_mirror_cannot_shadow_or_remove_builtin(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {
            "voices": {"kal_diphone": {
                "dir": r"Z:\missing\kal_diphone",
                "voice": "voice_broken_kal",
                "scm": "festvox/missing.scm",
            }},
        }})

        voices = [voice for voice in backend.voicebanks()
                  if voice["name"] == "kal_diphone"]
        self.assertEqual(len(voices), 1)
        self.assertTrue(voices[0]["ok"])
        self.assertEqual(voices[0]["source"], "Festival built-in")
        self.assertEqual(
            voices[0]["dir"],
            "/usr/share/festival/voices/english/kal_diphone",
        )
        preamble = backend._voice_preamble("kal_diphone", "en")
        self.assertIn("(voice_kal_diphone)", preamble)
        self.assertIn("(set! festvox_gui_legacy_joins nil)", preamble)
        self.assertIn(
            '(Param.set "unisyn.window_name" "hanning")', preamble)
        self.assertIn(
            '(Param.set "unisyn.window_factor" 1.0)', preamble)
        self.assertTrue(preamble.endswith(
            '(Param.set "unisyn.window_symmetric" 1)'))
        self.assertNotIn("missing", preamble)
        with self.assertRaisesRegex(fc.BackendError, "built-in Festival"):
            backend.voicebank_removal_info("kal_diphone")

    def test_kal_is_explicitly_english_only(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {}})

        compatibility = backend.voice_compatibility("kal_diphone")
        self.assertTrue(compatibility.supports("en"))
        self.assertFalse(compatibility.supports("ja"))
        with self.assertRaisesRegex(fc.BackendError, "English only"):
            backend._voice_preamble("kal_diphone", "ja")

    def test_voice_preamble_exposes_legacy_join_fault(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {}})

        preamble = backend._voice_preamble(
            "kal_diphone", "en", legacy_joins=True)

        self.assertIn("(set! festvox_gui_legacy_joins t)", preamble)
        self.assertIn("(voice_kal_diphone)", preamble)
        self.assertIn(
            '(Param.set "unisyn.window_name" "hanning")', preamble)
        self.assertIn(
            '(Param.set "unisyn.window_factor" 1.0)', preamble)
        self.assertTrue(preamble.endswith(
            '(Param.set "unisyn.window_symmetric" 1)'))

    def test_generated_voice_load_is_guarded_for_persistent_runtime(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {
            "voices": {
                "integrated": {
                    "dir": r"D:\generated\integrated",
                    "runtime_path": "/mnt/d/generated/integrated",
                    "voice": "voice_integrated",
                    "voice_en": "voice_integrated_en",
                    "scm": "festvox/integrated.scm",
                },
            },
        }})
        compatibility = fc.VoiceCompatibility(
            metadata_status="current",
            primary_language="en",
            supported_languages=("en", "asaxi", "ja"),
            voice_entry_points={
                "en": "voice_integrated_en",
                "asaxi": "voice_integrated",
                "ja": "voice_integrated_ja",
            },
        )

        with patch.object(
                backend, "voice_compatibility",
                return_value=compatibility):
            first = backend._voice_preamble("integrated", "en")
            second = backend._voice_preamble("integrated", "en")

        self.assertEqual(first, second)
        self.assertEqual(first.count('(load "/mnt/d/generated/integrated/'
                                     'festvox/integrated.scm")'), 1)
        self.assertIn("(defvar festvox_gui_voice_loaded_", first)
        self.assertIn("(if (not festvox_gui_voice_loaded_", first)
        self.assertIn("(voice_integrated_en)", first)

    def test_normal_join_window_stays_asymmetric_for_long_phones(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {}})
        metadata = {"source_window_policy": {
            "mode": "adaptive", "half_window_ms": 120.0,
            "normal_unisyn_window_symmetric": False,
        }, "supported_languages": ["ja"]}

        with patch.object(backend, "voice_metadata", return_value=metadata):
            self.assertFalse(backend._use_symmetric_join_window(
                "fixture", [("pau", 1.0), ("a", 0.239999)]))
            self.assertFalse(backend._use_symmetric_join_window(
                "fixture", [("pau", 1.0), ("a", 0.240000)]))
            self.assertFalse(backend._use_symmetric_join_window(
                "fixture", [("a", 0.650000), ("i", 0.280000)]))
            self.assertTrue(backend._use_symmetric_join_window(
                "fixture", [("a", 0.650000)], legacy_joins=True))

    def test_adaptive_join_window_never_treats_long_pause_as_phone(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {}})
        metadata = {"source_window_policy": {
            "mode": "adaptive", "half_window_ms": 80.0,
            "normal_unisyn_window_symmetric": False,
        }, "supported_languages": ["ja"]}

        with patch.object(backend, "voice_metadata", return_value=metadata):
            self.assertFalse(backend._use_symmetric_join_window(
                "fixture", [("pau", 2.0), ("sil", 1.0), ("a", 0.12)]))

    def test_builtin_kal_keeps_authored_symmetric_join_window(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {}})

        with patch.object(backend, "voice_metadata", return_value={}):
            self.assertTrue(backend._use_symmetric_join_window(
                "kal_diphone", [("a", 0.8)], legacy_joins=True))
            self.assertTrue(backend._use_symmetric_join_window(
                "kal_diphone", [("a", 0.8)], legacy_joins=False))
            self.assertTrue(backend._use_symmetric_join_window(
                "generated_fixture", [("a", 0.8)], legacy_joins=False))
            self.assertTrue(backend._use_symmetric_join_window(
                "generated_fixture", [("a", 0.8)], legacy_joins=True))

    def test_adaptive_metadata_enables_asymmetric_join_window(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {}})
        metadata = {
            "source_window_policy": {
                "mode": "adaptive",
                "normal_unisyn_window_symmetric": False,
            },
            "supported_languages": ["ja"],
        }

        with patch.object(backend, "voice_metadata", return_value=metadata):
            self.assertFalse(backend._use_symmetric_join_window(
                "generated_fixture", [("a", 0.8)], legacy_joins=False))
            self.assertTrue(backend._use_symmetric_join_window(
                "generated_fixture", [("a", 0.8)], legacy_joins=True))

    def test_integrated_adaptive_voice_keeps_symmetric_join_window(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {}})
        metadata = {
            "source_window_policy": {
                "mode": "adaptive",
                "normal_unisyn_window_symmetric": True,
            },
            "supported_languages": ["en", "asaxi", "ja"],
        }

        with patch.object(backend, "voice_metadata", return_value=metadata):
            self.assertTrue(backend._use_symmetric_join_window(
                "integrated_fixture", [("a", 0.8)],
                legacy_joins=False))

    def test_join_window_settings_are_bounded_and_backward_compatible(self):
        base = {
            "mode": "voice",
            "window_factor": 1.0,
            "crossover_ms": 40.0,
            "crossover_overrides": {},
        }
        self.assertEqual(
            fc.FestivalWSLBackend.normalize_join_settings(None),
            base)
        self.assertEqual(
            fc.FestivalWSLBackend.normalize_join_settings({
                "mode": "phase-aligned", "window_factor": 5.0}),
            dict(base, mode="asymmetric", window_factor=1.25))
        self.assertEqual(
            fc.FestivalWSLBackend.normalize_join_settings({
                "mode": "unknown", "window_factor": float("nan")}),
            base)
        self.assertEqual(
            fc.FestivalWSLBackend.normalize_join_settings({
                "mode": "symmetric", "window_factor": 0.2}),
            dict(base, mode="symmetric"))

    def test_join_crossover_settings_use_milliseconds_and_bound_overrides(self):
        normalized = fc.FestivalWSLBackend.normalize_join_settings({
            "crossover_ms": 500,
            "crossover_overrides": {
                "3": {"left_ms": 80, "right_ms": 80},
                "-1": {"left_ms": 2, "right_ms": 2},
                "bad": {"left_ms": "x", "right_ms": 4},
            },
        })

        self.assertEqual(normalized["crossover_ms"], 100.0)
        self.assertEqual(normalized["crossover_overrides"], {
            "3": {"left_ms": 50.0, "right_ms": 50.0},
        })

    def test_join_window_resolution_preserves_render_invariants(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {}})
        metadata = {
            "source_window_policy": {
                "mode": "adaptive",
                "normal_unisyn_window_symmetric": False,
            },
            "supported_languages": ["ja"],
        }
        faults = {"_join_settings": {
            "mode": "symmetric", "window_factor": 1.12}}

        with patch.object(backend, "voice_metadata", return_value=metadata):
            resolved = backend.resolve_join_settings(
                "fixture", fault_mode=faults)

        self.assertTrue(resolved["window_symmetric"])
        self.assertAlmostEqual(resolved["window_factor"], 1.12)
        self.assertEqual(resolved["scope"], "utterance")
        self.assertTrue(resolved["preserves_unit_selection"])
        self.assertTrue(resolved["preserves_phone_timing"])
        self.assertTrue(resolved["preserves_f0_targets"])
        self.assertEqual(resolved["crossover_ms"], 40.0)
        self.assertEqual(resolved["runtime"], "native-crossover")

    def test_zero_default_uses_stock_unless_an_occurrence_requests_crossover(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {}})

        disabled = backend.resolve_join_settings(
            "fixture", fault_mode={"_join_settings": {
                "crossover_ms": 0.0,
            }})
        selected = backend.resolve_join_settings(
            "fixture", fault_mode={"_join_settings": {
                "crossover_ms": 0.0,
                "crossover_overrides": {
                    "4": {"left_ms": 12.0, "right_ms": 18.0},
                },
            }})

        self.assertEqual(disabled["runtime"], "stock-festival")
        self.assertEqual(selected["runtime"], "native-crossover")

    def test_legacy_join_fault_overrides_manual_window_exactly(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {}})
        resolved = backend.resolve_join_settings(
            "fixture", fault_mode={
                "legacy_joins": True,
                "_join_settings": {
                    "mode": "asymmetric", "window_factor": 1.25,
                },
            })

        self.assertTrue(resolved["legacy_active"])
        self.assertTrue(resolved["window_symmetric"])
        self.assertEqual(resolved["window_factor"], 1.0)
        self.assertEqual(resolved["crossover_ms"], 0.0)
        self.assertEqual(resolved["runtime"], "stock-festival")
        self.assertEqual(resolved["source"], "legacy-fault")

    def test_voice_preamble_applies_manual_window_geometry(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {}})

        preamble = backend._voice_preamble(
            "kal_diphone", "en", window_symmetric=False,
            window_factor=1.125)

        self.assertIn(
            '(Param.set "unisyn.window_factor" 1.125)', preamble)
        self.assertTrue(preamble.endswith(
            '(Param.set "unisyn.window_symmetric" 0)'))

    def test_external_legacy_renderer_does_not_receive_new_keyword(self):
        calls = []

        def render(database, phones, speed=1.0, unit_overrides=None):
            calls.append((database, list(phones), speed, unit_overrides))
            return {
                "wav": fc.samples_to_wav_bytes(
                    np.zeros(80, dtype=np.float32), 8000),
                "phones": list(phones),
                "diphones": [],
                "skipped": [],
                "segments": [{"phone": phones[0], "start": 0.0,
                              "end": 0.01}],
                "selected_units": {},
                "splice_records": [],
            }

        fake = SimpleNamespace(
            render=render,
            _find_festvox_config=lambda _path: None,
            CROSSFADE_MS=15.0,
            EDGE_FADE_MS=8.0,
            HALF_MS=150.0,
        )
        with patch.object(fc, "import_synth_diphone", return_value=fake):
            backend = fc.DiphoneBackend({})
        backend.db = lambda _voice: object()

        result = backend.synth_phones(
            ["a"], "fixture", fault_mode={"legacy_joins": True})

        self.assertFalse(backend._render_supports_legacy_joins)
        self.assertEqual(result.sr, 8000)
        self.assertEqual(len(calls), 1)

    def test_windows_root_scan_adds_and_removes_only_discovered_voices(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            alpha = root / "alpha" / "festvox"
            alpha.mkdir(parents=True)
            (alpha / "alpha.scm").write_text(
                "(define (voice_alpha) t)\n", encoding="utf-8")
            cfg = {"festival_wsl": {
                "generated_voice_root": str(root),
                "voices": {
                    "stale": {"dir": str(root / "stale"),
                              "auto_discovered": True,
                              "scan_origin": "windows"},
                    "manual": {"dir": str(root / "manual"),
                               "voice": "voice_manual"},
                },
            }}
            backend = fc.FestivalWSLBackend(cfg)

            first = backend.refresh_voice_roots(install_kal=False)

            self.assertIn("alpha", first["added"])
            self.assertIn("stale", first["removed"])
            self.assertIn("manual", cfg["festival_wsl"]["voices"])
            self.assertTrue(cfg["festival_wsl"]["voices"]["alpha"]
                            ["auto_discovered"])

            shutil.rmtree(root / "alpha")
            second = backend.refresh_voice_roots(install_kal=False)
            self.assertIn("alpha", second["removed"])
            self.assertIn("manual", cfg["festival_wsl"]["voices"])

    def test_updated_discovered_voice_invalidates_runtime_caches(self):
        class UpdatedScanBackend(fc.FestivalWSLBackend):
            def scan_voice_dir_windows(self, path):
                return {
                    "name": "alpha", "dir": str(Path(path).parent),
                    "voice": "voice_alpha_new", "scm": "alpha.scm",
                }

        with tempfile.TemporaryDirectory() as root:
            festvox = Path(root) / "alpha" / "festvox"
            festvox.mkdir(parents=True)
            (festvox / "alpha.scm").write_text(
                "(define (voice_alpha_new) t)\n", encoding="utf-8")
            cfg = {"festival_wsl": {
                "generated_voice_root": root,
                "voices": {"alpha": {
                    "dir": str(Path(root) / "alpha"),
                    "voice": "voice_alpha_old", "scm": "alpha.scm",
                    "auto_discovered": True, "scan_origin": "windows",
                }},
            }}
            backend = UpdatedScanBackend(cfg)
            backend._alternatives["alpha"] = {"a-b": []}
            backend._voice_metadata["alpha"] = {"language": "ja"}
            backend._sustains[("alpha", "a")] = (np.zeros(8), 16000)

            report = backend.refresh_voice_roots(install_kal=False)

            self.assertIn("alpha", report["updated"])
            self.assertNotIn("alpha", backend._alternatives)
            self.assertNotIn("alpha", backend._voice_metadata)
            self.assertNotIn(("alpha", "a"), backend._sustains)

    def test_failed_wsl_root_scan_preserves_discovered_registrations(self):
        class UnavailableWSLBackend(fc.FestivalWSLBackend):
            def _run(self, args, timeout=None):
                return SimpleNamespace(
                    returncode=1, stdout="", stderr="WSL unavailable")

        with tempfile.TemporaryDirectory() as root:
            cfg = {"festival_wsl": {
                "generated_voice_root": root,
                "generated_voice_wsl_root": "/home/test/voices",
                "voices": {"remote": {
                    "dir": "/home/test/voices/remote",
                    "auto_discovered": True,
                    "scan_origin": "wsl",
                }},
            }}
            backend = UnavailableWSLBackend(cfg)

            report = backend.refresh_voice_roots(install_kal=False)

            self.assertIn("remote", cfg["festival_wsl"]["voices"])
            self.assertFalse(report["removed"])
            self.assertTrue(any("WSL voice root is unavailable" in warning
                                for warning in report["warnings"]))

    def test_windows_scan_root_wins_duplicate_name_over_wsl(self):
        class DuplicateBackend(fc.FestivalWSLBackend):
            def _run(self, args, timeout=None):
                return SimpleNamespace(
                    returncode=0, stdout="/home/test/voices/alpha\n",
                    stderr="")

            def scan_voice_dir_wsl(self, path):
                return {"name": "alpha", "dir": path,
                        "voice": "voice_alpha", "scm": "alpha.scm"}

        with tempfile.TemporaryDirectory() as root:
            scheme = Path(root) / "alpha" / "festvox"
            scheme.mkdir(parents=True)
            (scheme / "alpha.scm").write_text(
                "(define (voice_alpha) t)\n", encoding="utf-8")
            cfg = {"festival_wsl": {
                "generated_voice_root": root,
                "generated_voice_wsl_root": "/home/test/voices",
                "voices": {},
            }}
            backend = DuplicateBackend(cfg)

            report = backend.refresh_voice_roots(install_kal=False)

            registration = cfg["festival_wsl"]["voices"]["alpha"]
            self.assertEqual(registration["scan_origin"], "windows")
            self.assertTrue(registration["dir"].startswith(str(Path(root))))
            self.assertTrue(any("same name" in warning
                                for warning in report["warnings"]))

    def test_voice_portrait_is_installed_in_selected_wsl_voice(self):
        class RecordingBackend(fc.FestivalWSLBackend):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.commands = []

            def _run(self, args, timeout=None):
                self.commands.append(list(args))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

        cfg = {"festival_wsl": {"voices": {
            "voice": {"dir": "/home/test/voice"},
        }}}
        backend = RecordingBackend(cfg)
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "portrait.png"
            source.write_bytes(b"not-a-real-image")

            installed = backend.install_voice_icon("voice", str(source))

        self.assertEqual(installed, "/home/test/voice/speaker.png")
        self.assertEqual(backend.commands[0][0:3], ["rm", "-f", "--"])
        self.assertEqual(backend.commands[-1][0:2], ["cp", "--"])
        self.assertEqual(backend.commands[-1][-1], installed)

        removed = backend.remove_voice_icon("voice")
        self.assertIn("/home/test/voice/speaker.bmp", removed)
        self.assertEqual(backend.commands[-1][0:3], ["rm", "-f", "--"])

    def test_voice_icon_cleanup_keeps_only_the_new_extension(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "voice"
            folder.mkdir()
            keep = folder / "speaker.webp"
            keep.write_bytes(b"new")
            (folder / "speaker.png").write_bytes(b"old")
            (folder / "speaker.bmp").write_bytes(b"old")

            removed = fc.remove_known_voice_icons(folder, keep=keep)

            self.assertTrue(keep.exists())
            self.assertFalse((folder / "speaker.png").exists())
            self.assertFalse((folder / "speaker.bmp").exists())
            self.assertEqual(len(removed), 2)

    def test_unit_override_is_deferred_until_unisyn_has_segments(self):
        scheme = fc.festival_unit_override_scheme(
            {3: "t__u1", 99: "ignored", 4: "bad-name"}, 7)

        self.assertIn('(\"3\" \"t__u1\")', scheme)
        self.assertNotIn("utt.relation.items", scheme)
        self.assertNotIn("ignored", scheme)
        self.assertNotIn("bad-name", scheme)

    def test_recording_overrides_follow_unchanged_diphones_after_edits(self):
        old = ["pau", "a", "b", "c", "pau"]
        inserted = ["pau", "x", "a", "b", "c", "pau"]

        self.assertEqual(fc.remap_unit_overrides(
            old, inserted, {1: "a__u1", 2: "b__u2"}),
            {2: "a__u1", 3: "b__u2"})
        self.assertEqual(fc.remap_unit_overrides(
            old, ["pau", "a", "c", "pau"],
            {1: "a__u1", 2: "b__u2"}), {})
        self.assertEqual(fc.remap_unit_overrides(
            ["pau", "i", "cl", "s", "o", "pau"],
            ["pau", "i", "cl", "t", "a", "pau"],
            {1: "i__manual", 2: "s__manual"},
            old_source_phones=["pau", "i", "s", "s", "o", "pau"],
            new_source_phones=["pau", "i", "t", "t", "a", "pau"],
        ), {})

    def test_context_selector_rejects_one_sided_phrase_edge_match(self):
        choices = [
            {"left_name": "ih", "left_context": "r",
             "right_context": "dh"},
            {"left_name": "ih__u11", "left_context": "ng",
             "right_context": "pau"},
        ]

        selected = fc.contextual_unit_choice(choices, "th", "pau")

        self.assertEqual(selected["left_name"], "ih")

    def test_context_selector_rejects_legacy_inhale_alias_inside_phrase(self):
        choices = [
            {"id": "base", "left_name": "aw", "alias": "aw",
             "left_context": "*", "right_context": "*"},
            {"id": "take13", "left_name": "aw__u13", "alias": "inh aw",
             "left_context": "pau", "right_context": "n"},
        ]

        internal = fc.contextual_unit_choice(choices, "a", "n")
        phrase_edge = fc.contextual_unit_choice(choices, "pau", "n")

        self.assertEqual(internal["left_name"], "aw")
        self.assertEqual(phrase_edge["left_name"], "aw__u13")

    def test_context_selector_uses_complete_match_and_safe_fallback(self):
        choices = [
            {"left_name": "t", "left_context": "z",
             "right_context": "r"},
            {"left_name": "t__u1", "left_context": "s",
             "right_context": "l"},
        ]

        exact = fc.contextual_unit_choice(choices, "s", "l")
        mismatch = fc.contextual_unit_choice(choices, "m", "pau")

        self.assertEqual(exact["left_name"], "t__u1")
        self.assertEqual(mismatch["left_name"], "t")
        self.assertIn("base retained", mismatch["selection_reason"])

    def test_context_selector_prefers_incoming_vowel_over_base_cluster(self):
        choices = [
            {"id": "base", "left_name": "t", "index_name": "t-eh",
             "left_context": "z", "right_context": "r",
             "left_class": "fricative_voiced", "right_class": "liquid"},
            {"id": "take1", "left_name": "t__u1",
             "index_name": "t__u1-eh", "left_context": "eh",
             "right_context": "l", "left_class": "vowel",
             "right_class": "liquid"},
        ]

        selected = fc.contextual_unit_choice(
            choices, "ah", "s", right_phone="eh")

        self.assertEqual(selected["left_name"], "t__u1")
        self.assertEqual(selected["context_score"], -4)
        self.assertIn("stronger safe context", selected["selection_reason"])

    def test_context_selector_finds_base_when_rows_are_reordered(self):
        choices = [
            {"id": "take1", "left_name": "t__u1",
             "left_context": "eh", "right_context": "l"},
            {"id": "base", "left_name": "t",
             "left_context": "z", "right_context": "r"},
        ]

        selected = fc.contextual_unit_choice(
            choices, "z", "r", right_phone="eh")

        self.assertEqual(selected["left_name"], "t")
        self.assertIn("base retained", selected["selection_reason"])

    def test_readding_voice_invalidates_its_generated_metadata(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {"voices": {}}})
        backend._alternatives["lem"] = {"t-eh": [{"id": "old"}]}
        backend._sustains[("lem", "eh")] = (np.zeros(8), 16000)
        backend._alternatives["other"] = {"s-t": [{"id": "keep"}]}

        backend.add_voice({"name": "lem", "dir": "/voices/lem",
                           "voice": "voice_lem", "scm": "festvox/lem.scm"})

        self.assertNotIn("lem", backend._alternatives)
        self.assertNotIn(("lem", "eh"), backend._sustains)
        self.assertIn("other", backend._alternatives)

    def test_context_selector_matches_cross_language_vowel_features(self):
        choices = [
            {"left_name": "iy", "left_context": "r",
             "right_context": "d"},
            {"left_name": "iy__u1", "left_context": "*",
             "right_context": "u", "right_class": "vowel"},
        ]

        selected = fc.contextual_unit_choice(choices, "b", "ae")

        self.assertEqual(fc.phone_context_class("dh"), "fricative_voiced")
        self.assertEqual(selected["left_name"], "iy__u1")

    def test_context_selector_reads_japanese_recorded_context_fields(self):
        choices = [
            {"id": "base", "left_name": "t", "index_name": "t-e",
             "recorded_left_context": "z",
             "recorded_right_context": "r"},
            {"id": "jc_vowel", "left_name": "t__ja1",
             "index_name": "t__ja1-e",
             "recorded_left_context": "a",
             "recorded_right_context": "e"},
        ]

        selected = fc.contextual_unit_choice(
            choices, "a", "e", right_phone="e")

        self.assertEqual(fc.choice_recorded_context(selected, "left"), "a")
        self.assertEqual(selected["left_name"], "t__ja1")

    def test_context_edges_parse_strict_cv_aliases_directionally(self):
        self.assertEqual(fc.context_edge_info("ka", "left"), {
            "phone": "k", "class": "stop_voiceless",
            "kind": "compound_cv"})
        self.assertEqual(fc.context_edge_info("ka", "right"), {
            "phone": "a", "class": "vowel", "kind": "compound_cv"})
        self.assertEqual(fc.context_edge_info("zha", "left")["class"],
                         "fricative_voiced")
        self.assertEqual(fc.context_edge_info("ngya", "left")["class"],
                         "nasal")
        self.assertEqual(fc.context_edge_info("j", "left")["kind"],
                         "unclassified")
        self.assertEqual(fc.context_edge_info("*", "right")["kind"],
                         "wildcard_unknown")

    def test_voiced_sibilant_prefers_verified_vowel_context(self):
        choices = [
            {"id": "base", "left_name": "ay", "index_name": "ay-z",
             "left_context": "ch", "right_context": "d"},
            {"id": "take1", "left_name": "ay__u1",
             "index_name": "ay__u1-z", "left_context": "dh",
             "right_context": "ch"},
            {"id": "take2", "left_name": "ay__u2",
             "index_name": "ay__u2-z", "left_context": "b",
             "right_context": "*"},
            {"id": "take3", "left_name": "ay__u3",
             "index_name": "ay__u3-z", "left_context": "r",
             "right_context": "*"},
            {"id": "take6", "left_name": "ay__u6",
             "index_name": "ay__u6-z", "left_context": "r",
             "right_context": "oy"},
        ]

        selected = fc.contextual_unit_choice(
            choices, "s", "er", right_phone="z")

        self.assertEqual(selected["left_name"], "ay__u6")
        self.assertEqual(selected["context_quality"],
                         "verified_supportive")
        self.assertIn("verified vowel", selected["selection_reason"])
        overrides = fc.contextual_unit_overrides(
            ["pau", "s", "ay", "z", "er", "pau"], {"ay-z": choices})
        self.assertEqual(overrides[2], "ay__u6")

    def test_voiced_sibilant_unknown_fallback_and_all_risky_base(self):
        unknown = fc.contextual_unit_choice([
            {"left_name": "ay", "index_name": "ay-z",
             "left_context": "ch", "right_context": "d"},
            {"left_name": "ay__u1", "index_name": "ay__u1-z",
             "left_context": "s", "right_context": "*"},
        ], "s", "er", right_phone="z")
        risky = fc.contextual_unit_choice([
            {"left_name": "ay", "index_name": "ay-z",
             "left_context": "ch", "right_context": "d"},
            {"left_name": "ay__u1", "index_name": "ay__u1-z",
             "left_context": "s", "right_context": "ch"},
        ], "s", "er", right_phone="z")

        self.assertEqual(unknown["left_name"], "ay__u1")
        self.assertEqual(unknown["context_quality"], "unknown")
        self.assertEqual(risky["left_name"], "ay")
        self.assertIn("retained base", risky["selection_reason"])

    def test_festival_voice_pitch_reads_metadata_and_defaults_kal(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / "dic").mkdir()
            (folder / "dic" / "unit_alternatives.json").write_text(
                json.dumps({"average_pitch_hz": 164.5}), encoding="utf-8")
            backend = fc.FestivalWSLBackend({"festival_wsl": {"voices": {
                "lem": {"dir": str(folder), "scm": "festvox/lem.scm"},
            }}})

            self.assertEqual(backend.voice_pitch_hz("lem"), 164.5)
            self.assertEqual(backend.voice_pitch_hz("kal_diphone"), 110.0)

    def test_old_headroom_manifest_uses_recorded_speaker_median(self):
        metadata = {
            "average_pitch_hz": 201.736264,
            "automatic_pitch_floor_hz": 164.81,
            "automatic_pitch_headroom_semitones": 3.5,
            "default_pitch_source": "speaker_median_plus_headroom",
            "speaker_pitch_analysis": {"median_f0_hz": 164.81},
        }

        self.assertEqual(fc.metadata_voice_pitch_hz(metadata), 164.81)
        metadata["default_pitch_source"] = "builder_override"
        self.assertEqual(
            fc.metadata_voice_pitch_hz(metadata), 201.736264)

    def test_native_scheme_uses_persistent_runtime_when_enabled(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {
            "native_festival_bin": "/tmp/festvox-festival",
            "persistent_native_runtime": True,
        }})
        completed = SimpleNamespace(
            returncode=0, stdout="persistent\n", stderr="")

        with patch.object(
                backend, "_run_native_server_job",
                return_value=completed) as persistent:
            with patch.object(backend, "_run") as one_shot:
                output = backend._run_scheme(
                    "(festvox_us_generate_wave u nil)\n")

        self.assertEqual(output, "persistent\n")
        persistent.assert_called_once()
        one_shot.assert_not_called()

    def test_configured_windows_native_binary_owns_its_change_stamp(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = Path(root) / "festvox-festival"
            runtime.write_bytes(b"first")
            backend = fc.FestivalWSLBackend({"festival_wsl": {
                "native_festival_bin": str(runtime),
            }})

            first = backend._native_runtime_stamp()
            runtime.write_bytes(b"second-build")
            second = backend._native_runtime_stamp()

        self.assertNotEqual(first, second)
        self.assertEqual(first[0], "configured-file")
        self.assertIn(str(runtime), first)

    def test_wsl_unc_mapping_resolves_an_unconfigured_default_distro(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {"distro": ""}})

        with patch.object(
                backend, "_default_wsl_distro_name",
                return_value="Ubuntu"):
            mapped = backend._wsl_path_on_windows(
                "/home/test/voices/fixture")

        self.assertEqual(
            str(mapped),
            r"\\wsl.localhost\Ubuntu\home\test\voices\fixture")

    def test_default_wsl_distro_falls_back_to_one_quiet_list_probe(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {"distro": ""}})
        completed = SimpleNamespace(
            returncode=0,
            stdout="Ubuntu\r\nOther\r\n".encode("utf-16-le"),
            stderr=b"",
        )

        with patch("winreg.OpenKey", side_effect=OSError):
            with patch.object(
                    backend, "_wsl_exe", return_value="wsl.exe"):
                with patch(
                        "subprocess.run",
                        return_value=completed) as run:
                    first = backend._default_wsl_distro_name()
                    second = backend._default_wsl_distro_name()

        self.assertEqual(first, "Ubuntu")
        self.assertEqual(second, "Ubuntu")
        run.assert_called_once()

    def test_non_native_scheme_never_uses_persistent_runtime(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {
            "festival_bin": "festival",
            "persistent_native_runtime": True,
        }})
        completed = SimpleNamespace(
            returncode=0, stdout="stock\n", stderr="")

        with patch.object(
                backend, "_run", return_value=completed) as one_shot:
            with patch.object(
                    backend, "_run_native_server_job") as persistent:
                output = backend._run_scheme("(SayText \"test\")\n")

        self.assertEqual(output, "stock\n")
        one_shot.assert_called_once()
        persistent.assert_not_called()

    def test_voice_metadata_invalidation_stops_warm_native_runtime(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {"voices": {}}})

        with patch.object(backend, "_stop_native_server") as stop:
            backend.invalidate_voice_metadata("fixture")

        stop.assert_called_once_with()

    def test_wsl_voice_metadata_rebuild_invalidates_without_wsl_process(self):
        with tempfile.TemporaryDirectory() as root:
            mirror = Path(root)
            (mirror / "dic").mkdir()
            metadata = mirror / "dic" / "diphone_index.json"
            metadata.write_text('{"build": 1}', encoding="utf-8")
            backend = fc.FestivalWSLBackend({"festival_wsl": {
                "distro": "Ubuntu",
                "voices": {
                    "fixture": {
                        "dir": "/home/test/voices/fixture",
                        "scm": "festvox/fixture.scm",
                    },
                },
            }})

            with patch.object(
                    backend, "_wsl_path_on_windows",
                    return_value=mirror):
                with patch.object(backend, "_run") as wsl:
                    first = backend.refresh_voice_metadata("fixture")
                    metadata.write_text(
                        '{"build": 2, "changed": true}', encoding="utf-8")
                    with patch.object(
                            backend, "_stop_native_server") as stop:
                        second = backend.refresh_voice_metadata("fixture")

        self.assertNotEqual(first, second)
        stop.assert_called_once_with()
        wsl.assert_not_called()

    def test_festival_runtime_forces_safe_choice_for_legacy_voice(self):
        class RecordingBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {}})
                self.body = ""

            def unit_alternatives(self, _voicebank):
                return {"ih-ng": [
                    {"left_name": "ih", "left_context": "r",
                     "right_context": "dh"},
                    {"left_name": "ih__u11", "left_context": "ng",
                     "right_context": "pau"},
                ]}

            def _synth_common(self, body, voicebank, speed, text, lang,
                              **kwargs):
                self.body = body
                entries = [("pau", .1), ("th", .1), ("ih", .1),
                           ("ng", .1), ("pau", .1)]
                return fc.Synthesis(
                    np.zeros(500, np.float32), 1000,
                    fc.segments_from_durations(entries))

        backend = RecordingBackend()
        backend.synth_phones(
            ["pau", "th", "ih", "ng", "pau"], "legacy",
            seg_durs=[("pau", .1), ("th", .1), ("ih", .1),
                      ("ng", .1), ("pau", .1)], pitch=180)

        self.assertIn('(\"3\" \"ih\")', backend.body)
        self.assertNotIn("ih__u11", backend.body)

    def test_f0_edges_are_anchored_for_each_phrase(self):
        entries = [("pau", .1), ("a", .2), ("pau", .1), ("pau", .3),
                   ("b", .2), ("pau", .1)]
        result = fc.anchor_phrase_targets(
            entries, [(0.2, 170), (0.8, 180)])
        times = [round(t, 3) for t, _f in result]

        self.assertIn(0.1, times)
        self.assertIn(0.3, times)
        self.assertIn(0.7, times)
        self.assertIn(0.9, times)

    def test_pause_f0_controls_hold_sentence_edges(self):
        entries = [("pau", .1), ("a", .2), ("pau", .1), ("pau", .3),
                   ("b", .2), ("pau", .1)]
        result = fc.anchor_phrase_targets(
            entries, [(0.2, 170), (0.8, 190)])
        values = {round(time, 2): round(value, 1)
                  for time, value in result}

        self.assertEqual(values[0.05], 170.0)
        self.assertEqual(values[0.95], 190.0)
        self.assertIn(0.35, values)
        self.assertIn(0.55, values)

    def test_intonation_overlay_preserves_middle_and_changes_boundary(self):
        generated = [(0.0, 170), (.2, 182), (.5, 176), (.8, 172), (1.0, 168)]
        result = fc.overlay_intonation_targets(
            generated, [{"start": 0, "end": 1, "kind": "?"}], 175, 16)
        values = {round(time, 2): value for time, value in result}

        self.assertAlmostEqual(values[.5], 176.0)
        self.assertGreater(values[1.0], values[.8])

    def test_zero_fall_statement_does_not_override_generated_endpoint(self):
        generated = [
            (0.0, 218.0), (.35, 204.0), (.70, 191.0), (1.0, 176.0)
        ]

        result = fc.overlay_intonation_targets(
            generated, [{"start": 0.0, "end": 1.0, "kind": "."}],
            pitch=213.0, fall=0.0)

        self.assertEqual(result, generated)

    def test_pitch_estimation_fault_is_bounded_to_one_phone(self):
        entries = [("pau", .1), ("a", .2), ("t", .08), ("i", .2),
                   ("pau", .1)]
        clean = [(0.1, 170), (0.2, 175), (0.3, 170),
                 (0.38, 165), (0.58, 160)]
        broken, index = fc.pitch_estimation_fault(
            entries, clean, rng=random.Random(4))

        self.assertIn(index, (1, 2, 3))
        self.assertTrue(all(fc.PITCH_MIN_HZ <= f <= fc.PITCH_MAX_HZ
                            for _t, f in broken))
        segment = fc.segments_from_durations(entries)[index]
        self.assertTrue(any(segment.start < t < segment.end
                            for t, _f in broken))

    def test_pitch_fault_can_be_pinned_to_one_segment(self):
        entries = [("pau", .1), ("a", .2), ("t", .08), ("i", .2),
                   ("pau", .1)]
        broken, index = fc.pitch_estimation_fault(
            entries, [(0.1, 170), (0.58, 160)],
            rng=random.Random(2), forced_index=3)
        repeated, repeated_index = fc.pitch_estimation_fault(
            entries, [(0.1, 170), (0.58, 160)],
            rng=random.Random(999), forced_index=3)

        self.assertEqual(index, 3)
        self.assertEqual(repeated_index, 3)
        self.assertEqual(broken, repeated)
        segment = fc.segments_from_durations(entries)[3]
        self.assertTrue(any(segment.start < time < segment.end
                            for time, _value in broken))

    def test_pitch_faults_can_hit_multiple_phones_with_a_bounded_count(self):
        entries = [("pau", .1)] + [("a", .08) for _ in range(24)] + [
            ("pau", .1)]
        total = sum(duration for _phone, duration in entries)

        broken, events = fc.pitch_estimation_faults(
            entries, [(0.1, 170), (total - 0.1, 160)],
            rng=random.Random(7), probability=.45, max_faults=5)

        self.assertGreater(len(events), 1)
        self.assertLessEqual(len(events), 5)
        self.assertEqual(len({event["segment"] for event in events}),
                         len(events))
        self.assertTrue(all(fc.PITCH_MIN_HZ <= value <= fc.PITCH_MAX_HZ
                            for _time, value in broken))

    def test_pinned_pitch_fault_reuses_the_exact_heard_frequency(self):
        entries = [("pau", .1), ("a", .2), ("i", .2), ("pau", .1)]
        heard, events = fc.pitch_estimation_faults(
            entries, [(0.1, 170), (0.5, 155)],
            rng=random.Random(2), probability=0.0, max_faults=1)
        pin = dict(events[0])

        repeated, repeated_events = fc.pitch_estimation_faults(
            entries, [(0.1, 230), (0.5, 215)],
            rng=random.Random(999), forced_events=[pin])

        self.assertEqual(repeated_events[0]["segment"], pin["segment"])
        self.assertEqual(repeated_events[0]["broken_hz"], pin["broken_hz"])
        segment = fc.segments_from_durations(entries)[pin["segment"]]
        inside = [value for time, value in repeated
                  if segment.start < time < segment.end]
        self.assertIn(pin["broken_hz"], inside)
        self.assertNotEqual(heard, repeated)

    def test_bit_depth_modes_have_explicit_volume_compensation(self):
        samples = np.linspace(-1.0, 1.0, 1001, dtype=np.float32)
        one = fc.apply_bit_depth(samples, 1)
        two = fc.apply_bit_depth(samples, 2)
        eight = fc.apply_bit_depth(samples, 8)

        self.assertLessEqual(float(np.max(np.abs(one))), 0.181)
        self.assertLess(float(np.max(np.abs(one))),
                        float(np.max(np.abs(two))))
        self.assertLess(float(np.max(np.abs(two))),
                        float(np.max(np.abs(eight))))
        self.assertLessEqual(len(np.unique(one)), 3)
        self.assertLessEqual(float(np.max(np.abs(one))), 0.091)

    def test_long_sustain_stretch_has_exact_non_silent_tail(self):
        sr = 8000
        time = np.arange(int(.18 * sr), dtype=np.float32) / sr
        source = np.sin(2 * np.pi * 180 * time).astype(np.float32) * .4
        result = fc.stretch_segment(
            source, sr, 20.0, sustain=source, use_sustain=True)

        self.assertEqual(len(result), len(source) * 20)
        self.assertGreater(float(np.max(np.abs(result[-sr:]))), .1)

    def test_gain_and_mixed_rate_concat(self):
        quiet = fc.apply_gain_db(np.ones(8, np.float32), -6.0)
        joined, sr = fc.concat_audio([
            (np.ones(8, np.float32), 8),
            (np.ones(4, np.float32), 4)], gap_s=.25)

        self.assertAlmostEqual(float(quiet[0]), 10 ** (-6 / 20), places=5)
        self.assertEqual(sr, 8)
        self.assertEqual(len(joined), 8 + 2 + 8)

    def test_active_speech_calibration_is_one_bounded_phrase_gain(self):
        sr = 1000
        samples = np.zeros(400, np.float32)
        samples[100:200] = 0.02
        samples[200:300] = 0.04
        synthesis = fc.Synthesis(samples, sr, [
            fc.Segment("pau", 0.0, 0.1),
            fc.Segment("a", 0.1, 0.2),
            fc.Segment("i", 0.2, 0.3),
            fc.Segment("pau", 0.3, 0.4),
        ])
        policy = {
            "schema_version": 1,
            "method": "active_speech_rms",
            "target_dbfs": -20.0,
            "minimum_gain_db": -6.0,
            "maximum_gain_db": 6.0,
            "minimum_active_seconds": 0.08,
            "peak_ceiling": 0.98,
        }

        result = fc.apply_active_speech_calibration(synthesis, policy)

        self.assertAlmostEqual(result.automatic_gain_db, 6.0, places=5)
        self.assertTrue(result.output_calibration["applied"])
        self.assertEqual(result.output_calibration["scope"], "completed_phrase")
        # One scalar preserves the deliberate 2:1 level relationship.
        self.assertAlmostEqual(
            float(np.mean(np.abs(result.samples[200:300]))) /
            float(np.mean(np.abs(result.samples[100:200]))),
            2.0,
            places=5,
        )

    def test_active_speech_calibration_ignores_pause_only_audio(self):
        synthesis = fc.Synthesis(
            np.ones(200, np.float32) * 0.2,
            1000,
            [fc.Segment("pau", 0.0, 0.2)],
        )

        fc.apply_active_speech_calibration(synthesis, {
            "method": "active_speech_rms",
            "minimum_active_seconds": 0.01,
        })

        self.assertFalse(synthesis.output_calibration["applied"])
        self.assertEqual(synthesis.automatic_gain_db, 0.0)

    def test_document_split_and_clean_dictionary_are_deterministic(self):
        self.assertEqual(fc.split_document_sentences(
            "One sentence. Two?\n\nThree!"),
            ["One sentence.", "Two?", "Three!"])
        cleaned = fc.cleaned_dictionary_text({
            "Zoo": ["z", "uw"], "apple": ["ae", "p", "ax", "l"]})

        self.assertEqual(cleaned.splitlines()[0], "apple ae p ax l")
        self.assertEqual(fc.parse_cleaned_dictionary_text(cleaned), {
            "apple": ["ae", "p", "ax", "l"],
            "zoo": ["z", "uw"],
        })
        self.assertEqual(fc.cleaned_dictionary_filename("My Dict.yaml"),
                         "My_Dict-cleaned.dict")

    def test_batch_project_round_trip_and_legacy_load(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "batch.json"
            rendered = fc.Segment("nn", 0, .2,
                                  timing_role="moraic_nasal")
            edited = fc.Segment("nn", 0, .3,
                                timing_role="moraic_nasal")
            rows = [{
                "text": "one", "engine": "festival_wsl",
                "segments": [rendered],
                "editor_segments": [edited],
                "unit_overrides": {0: "a__u1"},
                "selected_units": {0: "a__u1"},
                "target_pitchmarks": [.01, .02, .03],
                "splice_records": [{
                    "segment_index": 1, "time": .025,
                    "position_source": "festival-us-map",
                    "estimated": False,
                }],
                "fault_mode": {"bit_depth": 4, "legacy_joins": True},
                "needs_rerender": True,
            }, {
                "text": "two", "segments": [fc.Segment("b", 0, .1)],
            }]
            fc.save_batch_project(path, rows, active_sentence=1)
            loaded = fc.load_project(path)

            self.assertEqual(loaded["active_sentence"], 1)
            self.assertEqual(len(loaded["sentences"]), 2)
            self.assertEqual(loaded["sentences"][0]["editor_segments"][0].phone,
                             "nn")
            self.assertEqual(
                loaded["sentences"][0]["editor_segments"][0].timing_role,
                "moraic_nasal")
            self.assertEqual(
                loaded["sentences"][0]["segments"][0].uid, rendered.uid)
            self.assertEqual(
                loaded["sentences"][0]["editor_segments"][0].uid,
                edited.uid)
            self.assertEqual(loaded["sentences"][0]["unit_overrides"],
                             {0: "a__u1"})
            self.assertEqual(
                loaded["sentences"][0]["target_pitchmarks"],
                [.01, .02, .03])
            self.assertEqual(
                loaded["sentences"][0]["splice_records"][0]
                ["position_source"], "festival-us-map")
            self.assertTrue(
                loaded["sentences"][0]["fault_mode"]["legacy_joins"])
            self.assertTrue(loaded["sentences"][0]["needs_rerender"])

            legacy = Path(root) / "legacy.json"
            fc.save_project(
                legacy, text="old", language="English", lang_code="en",
                voicebank="test", speed=1.0,
                segments=[fc.Segment("a", 0, .2)], phones=["a"])
            old = fc.load_project(legacy)
            self.assertEqual(old["text"], "old")
            self.assertNotIn("sentences", old)
            self.assertEqual(old["segments"][0].uid,
                             fc.load_project(legacy)["segments"][0].uid)

    def test_folder_project_layout_and_legacy_migration_are_non_destructive(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            legacy = root / "old_project.json"
            legacy.write_text('{"text":"legacy","segments":[]}',
                              encoding="utf-8")
            original = legacy.read_bytes()
            segment = fc.Segment("a", 0.0, 0.2)
            project = root / "old_project"

            manifest = fc.save_project_folder(project, [{
                "text": "migrated", "segments": [segment],
                "editor_segments": [segment],
            }], settings={"phrase_pauses_ms": {
                "minor": 120, "major": 300, "sentence": 500,
            }})
            loaded_from_folder = fc.load_project(project)
            loaded_from_manifest = fc.load_project(manifest)

            self.assertEqual(manifest, project / "project.json")
            self.assertTrue((project / "cache").is_dir())
            self.assertTrue((project / "exports").is_dir())
            self.assertEqual(loaded_from_folder["version"], 4)
            self.assertEqual(loaded_from_folder["_project_root"],
                             str(project.resolve()))
            self.assertEqual(
                loaded_from_manifest["sentences"][0]["segments"][0].uid,
                segment.uid)
            self.assertEqual(
                loaded_from_folder["settings"]["phrase_pauses_ms"]
                ["sentence"], 500)
            self.assertEqual(legacy.read_bytes(), original)

    def test_project_folder_rejects_unrelated_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "foreign"
            folder.mkdir()
            manifest = folder / "project.json"
            manifest.write_text('{"name":"another application"}',
                                encoding="utf-8")
            original = manifest.read_bytes()

            with self.assertRaisesRegex(ValueError, "unrelated"):
                fc.save_project_folder(folder, [{"text": "speech"}])

            self.assertEqual(manifest.read_bytes(), original)

    def test_transfer_segment_uids_matches_repeated_occurrences(self):
        old = [fc.Segment("pau", 0.0, .1),
               fc.Segment("a", .1, .2),
               fc.Segment("a", .2, .3),
               fc.Segment("b", .3, .4),
               fc.Segment("pau", .4, .5)]
        fresh = [fc.Segment(segment.phone, segment.start, segment.end)
                 for segment in old]
        fresh_ids = [segment.uid for segment in fresh]

        fc.transfer_segment_uids(old, fresh)

        self.assertEqual([segment.uid for segment in fresh],
                         [segment.uid for segment in old])
        self.assertEqual(len(set(segment.uid for segment in fresh)),
                         len(fresh))
        self.assertNotEqual(fresh_ids, [segment.uid for segment in fresh])

    def test_festival_scheme_preserves_english_lr_spread_at_zero_fall(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {
            "installed_voices": ["kal_diphone"]}})
        scheme = backend._synth_scheme(
            "(set! u (SynthText \"test\"))\n", "kal_diphone", 1.0,
            "/tmp/out.wav", "/tmp/out.seg", lang="en", pitch=164, fall=16)
        zero_fall = backend._pitch_override(164, 0, lang="en")
        monotone = backend._pitch_override(
            164, 0, monotone=True, lang="en")

        self.assertIn("GUIUNIT", scheme)
        self.assertIn("(defSynthType FestVoxUniSyn", scheme)
        self.assertIn("(festvox_us_generate_wave utt", scheme)
        self.assertIn(
            '(Param.set "festvox.join_crossover_ms" 40.000)', scheme)
        self.assertEqual(scheme.count("("), scheme.count(")"))
        self.assertIn("native-std", scheme)
        self.assertIn("scaled-std", scheme)
        self.assertIn("(> scaled-std 26.2)", scheme)
        self.assertIn("(> scaled-std 0.0)", zero_fall)
        self.assertNotIn("target_f0_std 0.0", zero_fall)
        self.assertIn("target_f0_std 0.0", monotone)

    def test_native_scheme_keeps_asymmetric_per_join_millisecond_override(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {}})

        scheme = backend._synth_scheme(
            "(set! u (SynthText \"test\"))\n",
            "kal_diphone", 1.0, "/tmp/out.wav", "/tmp/out.seg",
            lang="en", join_settings={
                "crossover_ms": 54.0,
                "crossover_overrides": {
                    "2": {"left_ms": 14.0, "right_ms": 38.0},
                },
            })
        legacy = backend._synth_scheme(
            "(set! u (SynthText \"test\"))\n",
            "kal_diphone", 1.0, "/tmp/out.wav", "/tmp/out.seg",
            lang="en", legacy_joins=True, join_settings={
                "crossover_ms": 100.0,
            })

        self.assertIn(
            "(set! festvox_gui_join_overrides '((2 14.000 38.000)))",
            scheme)
        self.assertIn(
            '(Param.set "festvox.join_crossover_ms" 54.000)', scheme)
        self.assertIn('("r" "liquid")', scheme)
        self.assertIn('("z" "fricative_voiced")', scheme)
        self.assertIn('("m" "nasal")', scheme)
        self.assertIn('("w" "glide")', scheme)
        self.assertIn('"festvox_join_context_class"', scheme)
        self.assertLess(
            scheme.index("(festvox_gui_apply_join_contexts"),
            scheme.index("(us_get_diphones utt)"))
        self.assertNotIn("FestVoxUniSyn", legacy)
        self.assertNotIn("festvox_us_generate_wave", legacy)

    def test_sustain_sample_reads_legacy_embedded_db_layout(self):
        with tempfile.TemporaryDirectory() as root:
            voice = Path(root) / "legacy_voice"
            (voice / "db" / "dic").mkdir(parents=True)
            (voice / "db" / "wav").mkdir(parents=True)
            samples = np.linspace(-0.4, 0.4, 1000, dtype=np.float32)
            fc.write_wav(voice / "db" / "wav" / "a.wav", samples, 1000)
            metadata = {"index": {"a-a": ["a.wav", 0.2, 0.5, 0.8]}}
            (voice / "db" / "dic" / "diphone_index.json").write_text(
                json.dumps(metadata), encoding="utf-8")
            backend = fc.FestivalWSLBackend({"festival_wsl": {"voices": {
                "legacy": {"dir": str(voice)},
            }}})

            sustain = backend.sustain_sample("a", "legacy")

            self.assertIsNotNone(sustain)
            chunk, sample_rate = sustain
            self.assertEqual(sample_rate, 1000)
            self.assertEqual(len(chunk), 600)

    def test_generated_unit_pitchmark_diagnostic_reads_actual_wav_and_pm(self):
        with tempfile.TemporaryDirectory() as root:
            voice = Path(root) / "voice"
            (voice / "dic").mkdir(parents=True)
            (voice / "wav").mkdir()
            (voice / "pm").mkdir()
            samples = np.sin(
                np.arange(1000, dtype=np.float32)
                * (2.0 * np.pi * 100.0 / 1000.0)) * 0.2
            fc.write_wav(voice / "wav" / "unit.wav", samples, 1000)
            marks = [
                0.005, 0.015, 0.025, 0.035,
                0.055, 0.065, 0.075, 0.085,
            ]
            lines = [
                "EST_File Track", "DataType ascii",
                "NumFrames %d" % len(marks), "NumChannels 0",
                "NumAuxChannels 0", "EqualSpace 0", "BreaksPresent true",
                "EST_Header_End",
            ] + ["%.6f\t1 \t" % mark for mark in marks]
            (voice / "pm" / "unit.pm").write_text(
                "\n".join(lines) + "\n", encoding="ascii")
            (voice / "pm" / "unit.f0.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "f0_source": "world-harvest-stonemask",
                    "frames": [
                        [0.0, 0.0], [0.01, 101.0], [0.02, 102.0],
                        [0.03, 0.0],
                    ],
                }), encoding="utf-8")
            metadata = {
                "index": {"a-i": ["unit.wav", 0.0, 0.5, 1.0]},
            }
            (voice / "dic" / "diphone_index.json").write_text(
                json.dumps(metadata), encoding="utf-8")
            backend = fc.FestivalWSLBackend({"festival_wsl": {"voices": {
                "fixture": {"dir": str(voice)},
            }}})

            diagnostic = backend.unit_pitchmark_diagnostic(
                "fixture", "a-i", {
                    "wav_name": "unit.wav",
                    "source_slice": {
                        "start": 0.0, "phone_boundary": 0.5, "end": 1.0,
                    },
                    # Source-bank provenance must never be dereferenced here.
                    "wav": "../../source-utau/do-not-read.wav",
                })

            self.assertEqual(diagnostic["wav_name"], "unit.wav")
            self.assertEqual(diagnostic["pitchmarks"], marks)
            self.assertEqual(len(diagnostic["samples"]), len(samples))
            self.assertEqual(diagnostic["f0_track_kind"], "analyzed")
            self.assertEqual(
                diagnostic["f0_source"], "world-harvest-stonemask")
            self.assertEqual(diagnostic["f0_track"][0], (0.0, 0.0))
            self.assertAlmostEqual(
                diagnostic["epoch_f0_track"][0][1], 100.0)
            self.assertTrue(diagnostic["discontinuities"])
            self.assertEqual(
                diagnostic["source_slice"]["phone_boundary"], 0.5)

            # Voices built before analyzed F0 sidecars remain inspectable.
            (voice / "pm" / "unit.f0.json").unlink()
            legacy = backend.unit_pitchmark_diagnostic(
                "fixture", "a-i", {"wav_name": "unit.wav"})
            self.assertEqual(legacy["f0_track_kind"], "epoch-rate")
            self.assertEqual(legacy["f0_source"], "pitchmark-intervals")
            self.assertEqual(legacy["f0_track"], legacy["epoch_f0_track"])

    def test_pitchmark_helpers_parse_f0_and_find_local_period_jump(self):
        text = """EST_File Track
DataType ascii
EST_Header_End
0.005000 1
bad row
0.015000 1
0.025000 1
0.035000 1
0.055000 1
0.065000 1
0.075000 1
0.085000 1
"""

        marks = fc.parse_est_pitchmarks(text)
        track = fc.pitchmark_f0_track(marks)
        faults = fc.pitchmark_discontinuities(marks)
        sidecar = fc.parse_pitchmark_f0_sidecar(json.dumps({
            "f0_source": "utau-frq",
            "frames": [[0.0, 0.0], [0.005, 165.0], ["bad", 165.0]],
        }))

        self.assertEqual(
            marks, [0.005, 0.015, 0.025, 0.035,
                    0.055, 0.065, 0.075, 0.085])
        self.assertAlmostEqual(track[0][1], 100.0)
        self.assertEqual(len(faults), 1)
        self.assertAlmostEqual(faults[0]["period_s"], 0.02)
        self.assertEqual(sidecar["f0_source"], "utau-frq")
        self.assertEqual(sidecar["frames"], [(0.0, 0.0), (0.005, 165.0)])
        self.assertEqual(fc.parse_pitchmark_f0_sidecar("not json"), {})

    def test_unisyn_render_diagnostic_recovers_exact_frame_handoff(self):
        output = """
(GUIPM 0 0.010)
(GUIPM 1 0.020)
(GUIPM 2 0.030)
(GUIPM 3 0.040)
(GUIPM 4 0.050)
(GUIPM 5 0.060)
(GUIMAP 0 0.010 0 0.010)
(GUIMAP 1 0.020 1 0.020)
(GUIMAP 2 0.030 2 0.030)
(GUIMAP 3 0.040 3 0.040)
(GUIMAP 0 0.010 4 0.050)
(GUIMAP 0 0.010 5 0.060)
(GUIUFRAME 0 2 0 2)
(GUIUFRAME 1 2 2 4)
(GUIFRAMEFIX 1 0.018 0 1 -7 0.220 0.950 0 0 "phase-reference-corrected")
(GUIXOVER 0 2 1 4 0.020 0.050 18.0 22.0 36.0 30.0 0.91 1 "v" "vowel" "context-capped")
"""
        segments = [
            fc.Segment("a", 0.0, 0.02),
            fc.Segment("v", 0.02, 0.04),
            fc.Segment("b", 0.04, 0.06),
        ]
        diagnostic = fc.parse_unisyn_render_diagnostics(output, segments)

        self.assertEqual(
            diagnostic["target_pitchmarks"],
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        self.assertEqual(len(diagnostic["splice_records"]), 1)
        splice = diagnostic["splice_records"][0]
        self.assertEqual(splice["segment_index"], 1)
        self.assertAlmostEqual(splice["time"], 0.025)
        self.assertEqual(splice["handoff_start"], 0.02)
        self.assertEqual(splice["handoff_end"], 0.03)
        self.assertEqual(splice["source_frame_boundary"], 2)
        self.assertEqual(splice["position_source"], "festival-us-map")
        self.assertFalse(splice["estimated"])
        self.assertAlmostEqual(splice["phone_fraction"], 0.25)
        self.assertTrue(splice["crossover_active"])
        self.assertEqual(splice["crossover_context_cap_ms"], 36.0)
        self.assertEqual(splice["crossover_effective_ms"], 30.0)
        self.assertEqual(splice["crossover_epoch_intervals"], 3)
        self.assertEqual(splice["crossover_start"], 0.02)
        self.assertEqual(splice["crossover_end"], 0.05)
        self.assertEqual(splice["crossover_context"], "vowel")
        self.assertEqual(splice["crossover_reason"], "context-capped")
        self.assertEqual(len(diagnostic["crossover_records"]), 1)
        self.assertEqual(len(diagnostic["frame_trajectory_records"]), 1)
        trajectory = diagnostic["frame_trajectory_records"][0]
        self.assertEqual(trajectory["target_index"], 1)
        self.assertEqual(trajectory["previous_source_frame"], 0)
        self.assertEqual(trajectory["source_frame"], 1)
        self.assertEqual(trajectory["centre_offset_samples"], -7)
        self.assertEqual(trajectory["phone"], "a")
        self.assertEqual(trajectory["correction_kind"], "phase-reference")
        self.assertAlmostEqual(trajectory["correlation_improvement"], .73)

    def test_render_returns_festival_waveform_without_post_processing(self):
        class RenderBackend(fc.FestivalWSLBackend):
            def __init__(self, exchange):
                super().__init__({"festival_wsl": {}})
                self.exchange = Path(exchange)

            def _exchange_dir(self):
                return str(self.exchange)

            def _synth_scheme(self, body, voicebank, speed, wav_wsl, seg_wsl,
                              **kwargs):
                self.wav_name = str(wav_wsl).rsplit("/", 1)[-1]
                self.seg_name = str(seg_wsl).rsplit("/", 1)[-1]
                return body

            def _run_scheme(self, _scheme):
                sr = 16000
                times = np.arange(int(.06 * sr), dtype=np.float64) / sr
                self.expected_samples = (
                    .2 * np.sin(2 * np.pi * 200 * times)
                ).astype(np.float32)
                fc.write_wav(
                    str(self.exchange / self.wav_name),
                    self.expected_samples,
                    sr,
                )
                self.expected_samples = fc.read_wav(
                    str(self.exchange / self.wav_name)
                )[0]
                (self.exchange / self.seg_name).write_text(
                    "separator ;\n#\n0.020 100 a\n"
                    "0.040 100 a\n0.060 100 a\n",
                    encoding="utf-8",
                )
                return """
(GUIPM 0 0.010)
(GUIPM 1 0.020)
(GUIPM 2 0.030)
(GUIPM 3 0.040)
(GUIPM 4 0.050)
(GUIPM 5 0.060)
(GUIMAP 0 0.010 0 0.010)
(GUIMAP 1 0.020 1 0.020)
(GUIMAP 2 0.030 2 0.030)
(GUIMAP 3 0.040 3 0.040)
(GUIUFRAME 0 2 0 2)
(GUIUFRAME 1 2 2 4)
(GUIUNIT 0 "a" "take7")
(GUIUNIT 1 "a" "base")
"""

        with tempfile.TemporaryDirectory() as temp:
            backend = RenderBackend(temp)
            result = backend._synth_common(
                "(set! u nil)\n", "voice", 1.0, "aaa", "en",
                fault_mode={"_join_settings": {
                    "mode": "asymmetric", "window_factor": 1.11}})

        self.assertEqual(result.selected_units, {0: "take7", 1: "base"})
        self.assertEqual(result.join_repairs, [])
        self.assertFalse(result.join_settings["window_symmetric"])
        self.assertAlmostEqual(result.join_settings["window_factor"], 1.11)
        self.assertTrue(result.join_settings["preserves_unit_selection"])
        self.assertEqual(len(result.samples), int(.06 * 16000))
        np.testing.assert_array_equal(result.samples, backend.expected_samples)

    def test_pitch_curve_keeps_all_dense_targets_within_safe_range(self):
        class RecordingBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {}})
                self.body = ""

            def _synth_common(self, body, voicebank, speed, text, lang,
                              **kwargs):
                self.body = body
                return fc.Synthesis(
                    np.zeros(120, np.float32), 100,
                    fc.segments_from_durations([
                        ("pau", .1), ("a", 1.0), ("pau", .1)]))

        backend = RecordingBackend()
        targets = [(time, 180.0 + index * 40.0)
                   for index, time in enumerate((.2, .3, .4, .5, .6, .7))]
        backend.synth_phones(
            ["pau", "a", "pau"], "voice", seg_durs=[
                ("pau", .1), ("a", 1.0), ("pau", .1)],
            pitch_targets=targets, pitch_mode="curve")

        self.assertIn("380.00", backend.body)
        self.assertGreaterEqual(backend.body.count("("), 10)

    def test_unedited_explicit_pass_publishes_rendered_pitch_baseline(self):
        class RecordingBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {}})
                self.kwargs = {}

            def _synth_common(self, body, voicebank, speed, text, lang,
                              **kwargs):
                self.kwargs = kwargs
                segments = fc.segments_from_durations(entries)
                targets = list(kwargs.get("ground_truth_targets") or [])
                return fc.Synthesis(
                    np.zeros(1000, np.float32), 1000, segments,
                    targets=targets, generated_targets=list(targets))

        entries = [
            ("pau", .04), ("pau", .08), ("a", .20), ("b", .20),
            ("pau", .08), ("pau", .04),
        ]
        old_segments = fc.segments_from_durations(entries)
        carried = [(0.16, 232.0), (0.30, 218.0), (0.46, 205.0)]
        stale_ground = list(carried)
        expected = fc.remap_targets(
            carried, old_segments, [duration for _phone, duration in entries])
        expected = fc.pitch_domain.recenter_targets_log(
            expected, 160.0, fc.PITCH_MIN_HZ, fc.PITCH_MAX_HZ)
        expected = fc.anchor_phrase_targets(entries, expected, 160.0)
        backend = RecordingBackend()

        result = backend.synth_phones(
            [phone for phone, _duration in entries],
            "voice",
            seg_durs=entries,
            old_segments=old_segments,
            prev_targets=carried,
            pitch=160.0,
            fall=18.0,
            ground_truth_targets=stale_ground,
        )

        published = backend.kwargs["ground_truth_targets"]
        self.assertEqual(published, expected)
        self.assertNotEqual(published, stale_ground)
        self.assertEqual(result.generated_targets, expected)

    def test_rerender_preserves_already_rendered_pitch_register(self):
        class RecordingBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {}})
                self.published = []

            def _synth_common(self, body, voicebank, speed, text, lang,
                              **kwargs):
                targets = list(kwargs.get("ground_truth_targets") or [])
                self.published.append(targets)
                return fc.Synthesis(
                    np.zeros(700, np.float32), 1000,
                    fc.segments_from_durations(entries),
                    targets=targets, generated_targets=list(targets))

        entries = [
            ("pau", .025), ("pau", .025), ("a", .20),
            ("pau", .05), ("pau", .10), ("pau", .10), ("pau", .05),
            ("i", .20), ("pau", .025), ("pau", .025),
        ]
        segments = fc.segments_from_durations(entries)
        baseline = fc.anchor_phrase_targets(
            entries,
            [(0.05, 188.0), (0.15, 196.0), (0.25, 181.0),
             (0.55, 212.0), (0.65, 198.0), (0.75, 186.0)],
            165.0,
        )
        backend = RecordingBackend()

        first = backend.synth_phones(
            [phone for phone, _duration in entries],
            "voice", seg_durs=entries, old_segments=segments,
            prev_targets=baseline, pitch=165.0,
            preserve_pitch_register=True,
        )
        second = backend.synth_phones(
            [phone for phone, _duration in entries],
            "voice", seg_durs=entries, old_segments=segments,
            prev_targets=first.generated_targets, pitch=165.0,
            preserve_pitch_register=True,
        )

        self.assertEqual(first.generated_targets, baseline)
        self.assertEqual(second.generated_targets, baseline)
        self.assertEqual(backend.published, [baseline, baseline])

    def test_asaxi_explicit_pass_does_not_recenter_inferred_mora_offsets(self):
        class RecordingBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {}})
                self.kwargs = {}

            def _synth_common(self, body, voicebank, speed, text, lang,
                              **kwargs):
                self.kwargs = kwargs
                targets = list(kwargs.get("ground_truth_targets") or [])
                return fc.Synthesis(
                    np.zeros(600, np.float32), 1000,
                    fc.segments_from_durations(entries),
                    targets=targets,
                    generated_targets=list(targets),
                    lang=lang,
                )

        entries = [
            ("pau", .08), ("sh", .08), ("er", .12),
            ("s", .08), ("o", .12), ("pau", .12),
        ]
        segments = fc.segments_from_durations(entries)
        inferred = [(0.12, 240.0), (0.22, 240.0),
                    (0.36, 120.0), (0.44, 120.0)]
        recentered = fc.anchor_phrase_targets(
            entries,
            fc.pitch_domain.recenter_targets_log(
                inferred, 165.0, fc.PITCH_MIN_HZ, fc.PITCH_MAX_HZ),
            165.0,
        )
        backend = RecordingBackend()

        result = backend.synth_phones(
            [phone for phone, _duration in entries],
            "voice",
            seg_durs=entries,
            old_segments=segments,
            prev_targets=inferred,
            pitch=165.0,
            fall=18.0,
            lang="asaxi",
        )

        published = backend.kwargs["ground_truth_targets"]
        self.assertIn((0.12, 240.0), published)
        self.assertIn((0.36, 120.0), published)
        self.assertAlmostEqual(
            max(value for _time, value in published) /
            min(value for _time, value in published),
            2.0,
            places=5,
        )
        self.assertNotEqual(result.generated_targets, recentered)
        self.assertEqual(result.generated_targets, published)

    def test_normal_text_uses_explicit_psola_pass_with_edge_reference(self):
        class RecordingBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {}})
                self.calls = []

            def _synth_common(self, body, voicebank, speed, text, lang,
                              **kwargs):
                self.calls.append((body, kwargs))
                segments = [fc.Segment("pau", 0.0, 0.1),
                            fc.Segment("a", 0.1, 0.3),
                            fc.Segment("pau", 0.3, 0.4)]
                targets = [(0.2, 180.0)]
                return fc.Synthesis(
                    np.zeros(6400, np.float32), 16000, segments,
                    text=text, lang=lang, voicebank=voicebank,
                    targets=targets, generated_targets=list(
                        kwargs.get("ground_truth_targets") or targets))

        backend = RecordingBackend()
        result = backend.synth(
            "hello.", "en", "test_voice", pitch=180.0, fall=18.0)

        self.assertEqual(len(backend.calls), 2)
        self.assertIn("SynthText", backend.calls[0][0])
        self.assertIn("Utterance Segments", backend.calls[1][0])
        ground = backend.calls[1][1]["ground_truth_targets"]
        self.assertEqual([round(t, 2) for t, _f in ground],
                         [0.0, 0.01, 0.02, 0.06, 0.1, 0.2,
                          0.3, 0.34, 0.38, 0.39, 0.4])
        self.assertEqual(result.generated_targets, ground)

    def test_japanese_text_pass_retimes_quote_pause_before_explicit_render(self):
        class RecordingBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {}})
                self.calls = []

            def _synth_common(self, body, voicebank, speed, text, lang,
                              **kwargs):
                self.calls.append((body, kwargs))
                entries = [
                    ("pau", .10), ("a", .20), ("r", .08),
                    ("pau", .80), ("w", .08), ("a", .20), ("pau", .10),
                ]
                return fc.Synthesis(
                    np.zeros(1560, np.float32), 1000,
                    fc.segments_from_durations(entries),
                    text=text, lang=lang, voicebank=voicebank,
                    targets=[(.20, 180.0), (1.30, 170.0)],
                )

        backend = RecordingBackend()
        backend.synth(
            "\u300c\u30a2\u30e9\u30fc\u30c8\u300d\u306f\u3002",
            "ja", "test_voice", pitch=180.0,
        )

        self.assertEqual(len(backend.calls), 2)
        explicit = backend.calls[1][1]["explicit_durations"]
        internal = []
        index = 0
        while index < len(explicit):
            if explicit[index][0] != "pau":
                index += 1
                continue
            end = index
            while end < len(explicit) and explicit[end][0] == "pau":
                end += 1
            if index > 0 and end < len(explicit):
                internal = explicit[index:end]
            index = end
        self.assertEqual(len(internal), 4)
        self.assertAlmostEqual(
            sum(duration for _phone, duration in internal), .06)

    def test_local_japanese_runtime_metadata_invalidates_on_file_change(self):
        with tempfile.TemporaryDirectory() as root:
            voice = Path(root) / "voice"
            (voice / "dic").mkdir(parents=True)
            japanese = {
                "language": "ja",
                "voice_entry_point": "voice_fixture_ja",
                "candidate_units": {},
            }
            index = voice / "dic" / "diphone_index.json"
            index.write_text(json.dumps(japanese), encoding="utf-8")
            backend = fc.FestivalWSLBackend({"festival_wsl": {"voices": {
                "fixture": {"dir": str(voice)},
            }}})

            self.assertEqual(
                backend.japanese_runtime_metadata("fixture"), japanese)
            index.write_text(json.dumps({"language": "ja"}),
                             encoding="utf-8")
            self.assertEqual(
                backend.japanese_runtime_metadata("fixture"), {})
            backend.invalidate_voice_metadata("fixture")
            self.assertEqual(
                backend.japanese_runtime_metadata("fixture"), {})

    def test_integrated_runtime_selects_japanese_entry_point(self):
        with tempfile.TemporaryDirectory() as root:
            voice = Path(root) / "voice"
            (voice / "dic").mkdir(parents=True)
            metadata = {
                "source_bundle_id": "srb_a",
                "configuration_id": "vcfg_a",
                "primary_language": "en",
                "supported_languages": ["en", "asaxi", "ja"],
                "alias_system": "arpasing-integrated-v1",
                "voice_entry_points": {
                    "en": "voice_fixture_en",
                    "asaxi": "voice_fixture_asaxi",
                    "ja": "voice_fixture_ja",
                },
                "language": "en",
                "voice_entry_point": "voice_fixture_en",
            }
            (voice / "dic" / "diphone_index.json").write_text(
                json.dumps(metadata), encoding="utf-8")
            backend = fc.FestivalWSLBackend({"festival_wsl": {"voices": {
                "fixture": {"dir": str(voice)},
            }}})

            runtime = backend.japanese_runtime_metadata("fixture")

            self.assertEqual(runtime["language"], "ja")
            self.assertEqual(runtime["voice_entry_point"],
                             "voice_fixture_ja")

    def test_local_festival_metadata_detects_in_place_restored_time_rewrites(self):
        with tempfile.TemporaryDirectory() as root:
            voice = Path(root) / "voice"
            (voice / "dic").mkdir(parents=True)
            index = voice / "dic" / "diphone_index.json"
            alternatives = voice / "dic" / "unit_alternatives.json"
            index.write_text('{"revision":"old"}', encoding="utf-8")
            alternatives.write_text(
                '{"diphones":{"a-b":[{"id":"old"}]}}',
                encoding="utf-8")
            index_stat = index.stat()
            alternatives_stat = alternatives.stat()
            backend = fc.FestivalWSLBackend({"festival_wsl": {"voices": {
                "fixture": {"dir": str(voice)},
            }}})

            self.assertEqual(
                backend.voice_metadata("fixture")["revision"], "old")
            self.assertEqual(
                backend.unit_alternatives("fixture")["a-b"][0]["id"],
                "old")

            index.write_text('{"revision":"new"}', encoding="utf-8")
            os.utime(index, ns=(index_stat.st_atime_ns,
                               index_stat.st_mtime_ns))
            self.assertEqual(index.stat().st_size, index_stat.st_size)
            self.assertEqual(index.stat().st_mtime_ns,
                             index_stat.st_mtime_ns)
            self.assertEqual(
                backend.voice_metadata("fixture")["revision"], "new")

            alternatives.write_text(
                '{"diphones":{"a-b":[{"id":"new"}]}}',
                encoding="utf-8")
            os.utime(alternatives,
                     ns=(alternatives_stat.st_atime_ns,
                         alternatives_stat.st_mtime_ns))
            self.assertEqual(alternatives.stat().st_size,
                             alternatives_stat.st_size)
            self.assertEqual(alternatives.stat().st_mtime_ns,
                             alternatives_stat.st_mtime_ns)
            self.assertEqual(
                backend.unit_alternatives("fixture")["a-b"][0]["id"],
                "new")

    def test_runtime_database_change_invalidates_warm_voice(self):
        with tempfile.TemporaryDirectory() as root:
            voice = Path(root) / "voice"
            (voice / "dic").mkdir(parents=True)
            (voice / "group").mkdir()
            (voice / "festvox").mkdir()
            (voice / "dic" / "diphone_index.json").write_text(
                "{}", encoding="utf-8")
            (voice / "dic" / "unit_alternatives.json").write_text(
                "{}", encoding="utf-8")
            (voice / "festvox" / "fixture.scm").write_text(
                ";; fixture", encoding="utf-8")
            index = voice / "dic" / "fixture_diphone.est"
            group = voice / "group" / "fixture_diphone.group"
            index.write_bytes(b"old-index")
            group.write_bytes(b"old-group")
            backend = fc.FestivalWSLBackend({"festival_wsl": {"voices": {
                "fixture": {
                    "dir": str(voice),
                    "scm": "festvox/fixture.scm",
                },
            }}})
            backend.refresh_voice_metadata("fixture")

            with patch.object(backend, "_stop_native_server") as stop:
                group.write_bytes(b"new-group-payload")
                backend.refresh_voice_metadata("fixture")
                stop.assert_called_once()

            with patch.object(backend, "_stop_native_server") as stop:
                index.write_bytes(b"new-index-payload")
                backend.refresh_voice_metadata("fixture")
                stop.assert_called_once()

    def test_concurrent_invalidation_cannot_publish_stale_metadata(self):
        class RacingBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {"voices": {
                    "fixture": {"dir": "/voices/fixture"},
                }}})
                self.started = threading.Event()
                self.release = threading.Event()
                self.reads = 0

            def _run(self, args, timeout=None):
                self.reads += 1
                if self.reads == 1:
                    self.started.set()
                    self.release.wait(3.0)
                    payload = {"revision": "old"}
                else:
                    payload = {"revision": "new"}
                return SimpleNamespace(
                    returncode=0, stdout=json.dumps(payload), stderr="")

        backend = RacingBackend()
        results = []
        worker = threading.Thread(
            target=lambda: results.append(backend.voice_metadata("fixture")))
        worker.start()
        self.assertTrue(backend.started.wait(2.0))
        backend.invalidate_voice_metadata("fixture")
        backend.release.set()
        worker.join(5.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(results[0]["revision"], "new")
        self.assertEqual(
            backend.voice_metadata("fixture")["revision"], "new")

    def test_published_festival_metadata_is_recursively_read_only(self):
        class StaticBackend(fc.FestivalWSLBackend):
            def _run(self, args, timeout=None):
                if str(args[-1]).endswith("unit_alternatives.json"):
                    payload = {"diphones": {
                        "a-b": [{"id": "base", "tags": ["safe"]}],
                    }}
                else:
                    payload = {"nested": {"items": [1, 2]}}
                return SimpleNamespace(
                    returncode=0, stdout=json.dumps(payload), stderr="")

        backend = StaticBackend({"festival_wsl": {"voices": {
            "fixture": {"dir": "/voices/fixture"},
        }}})
        metadata = backend.voice_metadata("fixture")
        alternatives = backend.unit_alternatives("fixture")

        with self.assertRaises(TypeError):
            metadata["nested"]["poison"] = True
        with self.assertRaises(TypeError):
            alternatives["a-b"][0]["id"] = "poison"
        self.assertEqual(
            backend.unit_alternatives("fixture")["a-b"][0]["id"],
            "base")
        self.assertEqual(
            backend.cache_info()["voice"]["bytes"],
            fc.estimate_size_bytes(metadata) +
            fc.estimate_size_bytes(alternatives),
        )

    def test_runtime_voice_metadata_caches_without_duplicate_choice_graph(self):
        with tempfile.TemporaryDirectory() as root:
            voice = Path(root) / "voice"
            (voice / "dic").mkdir(parents=True)
            metadata = {
                "revision": "current",
                "phones": ["aa", "q", "pau"],
                "index": {"q-aa": ["q.wav", 0.0, 0.1, 0.2]},
                "alternatives": {
                    "q-aa": [
                        {"id": "take-%d" % index, "blob": "x" * 4096}
                        for index in range(800)
                    ],
                },
                "alias_metadata": {"audit_blob": "y" * 100_000},
            }
            (voice / "dic" / "diphone_index.json").write_text(
                json.dumps(metadata), encoding="utf-8")
            backend = fc.FestivalWSLBackend({
                "festival_voice_cache_mib": 4,
                "festival_wsl": {"voices": {
                    "fixture": {"dir": str(voice)},
                }},
            })

            first = backend.voice_metadata("fixture")
            second = backend.voice_metadata("fixture")

            self.assertIs(first, second)
            self.assertEqual(first["revision"], "current")
            self.assertEqual(first["phones"], ("aa", "q", "pau"))
            self.assertIn("q-aa", first["index"])
            self.assertNotIn("alternatives", first)
            self.assertNotIn("alias_metadata", first)
            self.assertEqual(
                backend.cache_info()["voice"]["metadata_entries"], 1)

    def test_voice_cache_reserves_three_quarters_for_alternatives(self):
        backend = fc.FestivalWSLBackend({
            "festival_voice_cache_mib": 64,
            "festival_wsl": {"voices": {}},
        })

        self.assertEqual(
            backend._voice_metadata.info()["max_bytes"], 16 * 1024 * 1024)
        self.assertEqual(
            backend._alternatives.info()["max_bytes"], 48 * 1024 * 1024)

    def test_festival_invalidation_state_does_not_grow_per_voice_name(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {"voices": {}}})
        for index in range(1000):
            backend.invalidate_voice_metadata("missing-%d" % index)

        self.assertFalse(hasattr(backend, "_cache_generations"))
        self.assertEqual(backend._voice_fingerprints, {})
        self.assertEqual(backend._cache_epoch, 1000)

    def test_japanese_automatic_selection_is_left_to_generated_voice(self):
        backend = fc.FestivalWSLBackend({"festival_wsl": {}})
        inventory = {"a-k": [{
            "left_name": "a__generic_choice",
            "index_name": "a__generic_choice-k",
            "left_context": "*",
            "right_context": "*",
        }]}
        backend.unit_alternatives = lambda _voicebank: inventory
        backend.japanese_runtime_metadata = lambda _voicebank: {
            "language": "ja",
            "voice_entry_point": "voice_fixture_ja",
        }

        self.assertEqual(
            backend.automatic_unit_overrides(
                ["pau", "a", "k", "a", "pau"], "fixture"),
            {},
        )

        backend.japanese_runtime_metadata = lambda _voicebank: {}
        self.assertEqual(
            backend.automatic_unit_overrides(
                ["pau", "a", "k", "a", "pau"], "fixture"),
            {1: "a__generic_choice"},
        )

    def test_project_json_preserves_japanese_overlay_without_aliasing(self):
        overlay = {
            "schema_version": 1,
            "accent_overrides": {"0": {"accent_state": "unaccented"}},
        }
        with tempfile.TemporaryDirectory() as root:
            project = Path(root) / "project"
            fc.save_project_folder(project, [{
                "text": "かな", "japanese_state": overlay,
            }])
            overlay["accent_overrides"]["0"]["accent_state"] = "accented"
            loaded = fc.load_project(project)

        self.assertEqual(
            loaded["sentences"][0]["japanese_state"]
            ["accent_overrides"]["0"]["accent_state"],
            "unaccented")

    def test_generated_voice_cl_uses_consonant_sources_but_stays_visible(self):
        class RecordingBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {}})
                self.body = ""
                self.kwargs = {}

            def voice_metadata(self, _voicebank):
                return {
                    "context_model": "oto_directional_v1",
                    "index": {
                        "pau-pau": [], "pau-i": [], "i-s": [],
                        "s-s": [], "s-o": [], "o-pau": [],
                    },
                }

            def automatic_unit_overrides(self, _phones, _voicebank):
                return {}

            def _synth_common(self, body, voicebank, speed, text, lang,
                              **kwargs):
                self.body = body
                self.kwargs = kwargs
                entries = kwargs["explicit_durations"]
                return fc.Synthesis(
                    np.zeros(1000, np.float32), 1000,
                    fc.segments_from_durations(entries),
                    text=text, lang=lang, voicebank=voicebank,
                    phones=list(kwargs.get("phones_used") or []),
                )

        backend = RecordingBackend()
        result = backend.synth_phones(
            ["pau", "i", "cl", "s", "o", "pau"],
            "generated",
            seg_durs=[
                ("pau", .10), ("i", .10), ("cl", .08),
                ("s", .10), ("o", .10), ("pau", .10),
            ],
            pitch=180.0,
        )

        self.assertNotIn("(cl ", backend.body)
        self.assertIn("(s 0.08000", backend.body)
        closure_index = next(
            index for index, segment in enumerate(result.segments)
            if segment.phone == "cl"
        )
        self.assertEqual(result.render_phones[closure_index], "s")
        self.assertEqual(
            result.special_phone_realizations[0]["status"], "resolved"
        )
        self.assertEqual(result.phones, ["i", "cl", "s", "o"])

    def test_manual_cl_uses_structural_source_path_in_every_language(self):
        class RecordingBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {}})
                self.body = ""

            def voice_metadata(self, _voicebank):
                return {
                    "context_model": "oto_directional_v1",
                    "index": {
                        "pau-i": [], "i-s": [], "s-s": [],
                        "s-o": [], "o-pau": [],
                    },
                }

            def automatic_unit_overrides(self, _phones, _voicebank):
                return {}

            def _synth_common(self, body, voicebank, speed, text, lang,
                              **kwargs):
                self.body = body
                render_phones = ["pau", "i", "s", "s", "o", "pau"]
                entries = [(phone, 0.1) for phone in render_phones]
                return fc.Synthesis(
                    np.zeros(600, np.float32), 1000,
                    fc.segments_from_durations(entries),
                    text=text, lang=lang, voicebank=voicebank,
                    phones=list(kwargs.get("phones_used") or []),
                )

        for language in ("en", "asaxi", "ja"):
            with self.subTest(language=language):
                backend = RecordingBackend()
                result = backend.synth_phones(
                    ["i", "cl", "s", "o"],
                    "generated",
                    lang=language,
                    pitch=180.0,
                )

                self.assertIn(
                    "(Utterance Phones (pau i s s o pau))",
                    backend.body,
                )
                self.assertNotIn(" cl ", backend.body)
                self.assertEqual(result.phones, ["i", "cl", "s", "o"])
                self.assertEqual(
                    [segment.phone for segment in result.segments],
                    ["pau", "i", "cl", "s", "o", "pau"],
                )
                self.assertEqual(
                    result.render_phones,
                    ["pau", "i", "s", "s", "o", "pau"],
                )
                self.assertEqual(
                    result.special_phone_realizations[0]["mode"],
                    "anticipatory_consonant",
                )

    def test_manual_literal_and_structural_cl_can_coexist(self):
        class RecordingBackend(fc.FestivalWSLBackend):
            def __init__(self):
                super().__init__({"festival_wsl": {}})
                self.body = ""

            def voice_metadata(self, _voicebank):
                return {
                    "context_model": "oto_directional_v1",
                    "special_phone_realizations": {
                        "schema_version": 1,
                        "phones": {
                            "cl": {"mode": "anticipatory_consonant"},
                        },
                        "literal_phone_mappings": {
                            "cl_literal": {
                                "mode": "literal_alias",
                                "source_phone": "cl",
                            },
                        },
                    },
                    "index": {
                        "pau-aa": [], "aa-t": [], "t-t": [],
                        "t-aa": [],
                        "aa-cl": [], "cl-aa": [], "aa-pau": [],
                    },
                }

            def automatic_unit_overrides(self, _phones, _voicebank):
                return {}

            def _synth_common(self, body, voicebank, speed, text, lang,
                              **kwargs):
                self.body = body
                render_phones = [
                    "pau", "aa", "t", "t", "aa", "cl", "aa",
                    "pau",
                ]
                entries = [(phone, 0.1) for phone in render_phones]
                return fc.Synthesis(
                    np.zeros(800, np.float32), 1000,
                    fc.segments_from_durations(entries),
                    text=text, lang=lang, voicebank=voicebank,
                    phones=list(kwargs.get("phones_used") or []),
                )

        backend = RecordingBackend()
        result = backend.synth_phones(
            ["aa", "cl", "t", "aa", "cl_literal", "aa"],
            "generated",
            lang="ja",
        )

        self.assertIn(
            "(Utterance Phones (pau aa t t aa cl aa pau))",
            backend.body,
        )
        self.assertEqual(
            result.phones,
            ["aa", "cl", "t", "aa", "cl_literal", "aa"],
        )
        self.assertEqual(
            result.render_phones,
            [
                "pau", "aa", "t", "t", "aa", "cl", "aa",
                "pau",
            ],
        )
        self.assertEqual(
            [segment.phone for segment in result.segments],
            [
                "pau", "aa", "cl", "t", "aa",
                "cl_literal", "aa", "pau",
            ],
        )

    def test_old_generated_voice_with_no_hold_is_rejected_not_literal(self):
        with self.assertRaisesRegex(
                fc.BackendError, "refused to substitute a literal cl"):
                fc.resolve_voice_special_phones(
                ["i", "cl", "s", "o"],
                {"context_model": "oto_directional_v1"},
                voicebank="old",
                available_diphones={"i-cl", "cl-s", "i-s", "s-o"},
            )

    def test_kal_explicit_cl_keeps_native_structural_phone(self):
        result = fc.resolve_voice_special_phones(
            ["ih", "cl", "t", "eh"],
            {},
            voicebank="kal_diphone",
            available_diphones=None,
        )

        self.assertEqual(result.render_phones, ("ih", "cl", "t", "eh"))
        self.assertEqual(result.realizations[0].status, "literal")

    def test_structural_cl_relabel_mismatch_fails_instead_of_exposing_source(self):
        synthesis = fc.Synthesis(
            np.zeros(200, np.float32), 1000,
            [fc.Segment("i", 0.0, .1), fc.Segment("s", .1, .2)],
        )
        resolution = fc.resolve_voice_special_phones(
            ["i", "cl", "s"],
            {"context_model": "oto_directional_v1"},
            voicebank="generated",
            available_diphones={"i-s", "s-s"},
        )

        with self.assertRaisesRegex(
                fc.BackendError, "cannot be aligned"):
            fc.apply_special_phone_display(
                synthesis,
                resolution.display_phones,
                resolution.render_phones,
                resolution.realizations,
            )

    def test_generated_structural_cl_rejects_unavailable_inventory(self):
        with self.assertRaisesRegex(
                fc.BackendError, "inventory is unavailable"):
            fc.resolve_voice_special_phones(
                ["i", "cl", "s"],
                {"context_model": "oto_directional_v1"},
                voicebank="generated",
                available_diphones=None,
            )


if __name__ == "__main__":
    unittest.main()
