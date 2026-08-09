# -*- coding: utf-8 -*-
"""Deterministic Asaxi utterance and F0 planning tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import asaxi_frontend as af
import asaxi_prosody as ap


def dictionary() -> af.AsaxiSynthesisDictionary:
    def entry(
        word: str,
        pattern: str,
        accent_class: str = "lexical",
        lexical_type: str = "noun",
        variants=(),
    ) -> af.AsaxiLexiconEntry:
        analysis = af.analyze_word(word)
        return af.AsaxiLexiconEntry(
            word=word,
            phones=analysis.phones,
            moras=analysis.moras,
            pitch_accent=pattern,
            pitch_accent_class=accent_class,
            source_note=f"fixtures/{word}.md",
            variants=tuple(variants),
        )

    return af.AsaxiSynthesisDictionary(
        schema_version=1,
        ruleset="asaxi-pitch-v1",
        entries={
            "shěsonů": entry("shěsonů", "H.L.L", lexical_type="verb"),
            "to": entry("to", "L", "atonal", "particle"),
            "kè": entry("kè", "L", "atonal", "particle"),
            "kvå": entry("kvå", "H", "dominant", "particle"),
            "xoxo": entry("xoxo", "H.L", lexical_type="verb"),
            "bi": entry(
                "bi",
                "H",
                variants=(
                    {
                        "lexical_type": "noun",
                        "phones": af.g2p_asaxi("bi"),
                        "pitch_accent": "H",
                        "pitch_accent_class": "lexical",
                    },
                    {
                        "lexical_type": "particle",
                        "phones": af.g2p_asaxi("bi"),
                        "pitch_accent": "L",
                        "pitch_accent_class": "atonal",
                    },
                ),
            ),
        },
        phrases={
            "to shěsonů": {
                "pitch_accent": "L | H.L.L",
                "pitch_accent_class": "phrase",
            },
            "kè to shěsonů": {
                "pitch_accent": "L | L | L.H.H",
                "pitch_accent_class": "phrase",
                "source_note": "fixtures/three-word-idiom.md",
            },
        },
        source_summary={},
    )


class AsaxiProsodyTests(unittest.TestCase):
    @staticmethod
    def _pitch(plan: ap.AsaxiProsodyPlan) -> str:
        return ".".join(mora.pitch for mora in plan.moras)

    def test_capitalized_term_uses_english_phones_and_borrowed_syllable(self):
        plan = ap.analyze_utterance(
            "to JOHN shěsonů.",
            dictionary(),
            capitalized_pronunciations={
                "john": ("jh", "ao", "n"),
            },
        )

        term = plan.words[1]
        self.assertEqual(term.surface, "JOHN")
        self.assertEqual(term.lexical_type, "borrowed_term")
        self.assertEqual(term.phones, ("jh", "ao", "n"))
        term_moras = plan.moras[term.mora_start:term.mora_end]
        self.assertEqual(len(term_moras), 1)
        self.assertEqual(term_moras[0].text, "JOHN")
        self.assertEqual(term_moras[0].phones, ("jh", "ao", "n"))
        self.assertEqual(term_moras[0].kind, "borrowed_syllable")
        self.assertIn(
            "capitalized_english_g2p",
            {diagnostic.code for diagnostic in plan.diagnostics},
        )

    def test_lowercase_term_retains_native_asaxi_g2p(self):
        plan = ap.analyze_utterance(
            "to john shěsonů.",
            dictionary(),
        )

        self.assertEqual(plan.words[1].surface, "john")
        self.assertEqual(plan.words[1].phones, ("y", "o", "h", "n"))
        self.assertNotIn(
            "capitalized_english_g2p",
            {diagnostic.code for diagnostic in plan.diagnostics},
        )

    def test_capitalized_term_requires_external_or_user_pronunciation(self):
        with self.assertRaisesRegex(
                ValueError, "Capitalized term 'UNKNOWN'"):
            ap.analyze_utterance("to UNKNOWN.", dictionary())

    def test_statement_boundary_and_phrase_override(self) -> None:
        plan = ap.analyze_utterance("To shěsonů.", dictionary())
        self.assertEqual([word.pitch_accent for word in plan.words],
                         ["L", "H.L.L"])
        self.assertEqual(plan.moras[0].lexical_pitch, "L")
        self.assertEqual(plan.moras[0].pitch, "H")
        self.assertFalse(plan.interrogative)

    def test_dotted_nasal_inflection_reaches_runtime_phone_plan(self) -> None:
        plan = ap.analyze_utterance("găxănă ono kem.ma")
        self.assertEqual(
            plan.phones[-5:],
            ("k", "e", "m", "m", "a"),
        )
        self.assertEqual(
            [
                (mora.text, mora.phones)
                for mora in plan.moras[-2:]
            ],
            [
                ("kem", ("k", "e", "m")),
                ("ma", ("m", "a")),
            ],
        )
        self.assertEqual(
            [
                (mora.phone_start, mora.phone_end)
                for mora in plan.moras[-2:]
            ],
            [(12, 15), (15, 17)],
        )

    def test_particleless_question_deaccents_but_preserves_wh_peak(self) -> None:
        plan = ap.analyze_utterance("To kvå xoxo?", dictionary())
        self.assertEqual(
            [mora.pitch for mora in plan.moras],
            ["L", "H", "L", "L"],
        )

    def test_polish_closing_quote_preserves_boundary_mark(self) -> None:
        question = ap.analyze_utterance("„To kvå xoxo?”", dictionary())
        directive = ap.analyze_utterance("„To shěsonů!”", dictionary())

        self.assertEqual(question.boundary_mark, "?")
        self.assertTrue(question.interrogative)
        self.assertEqual(directive.boundary_mark, "!")
        self.assertTrue(directive.directive)

    def test_typed_homograph_hint_selects_particle(self) -> None:
        plan = ap.analyze_utterance(
            "bi", dictionary(), lexical_type_hints={0: "particle"}
        )
        self.assertEqual(plan.words[0].pitch_accent, "L")
        self.assertEqual(plan.words[0].lexical_type, "particle")
        self.assertFalse(any(
            item.code == "ambiguous_homograph_default"
            for item in plan.diagnostics
        ))

    def test_idiom_phrase_keeps_word_and_mora_boundaries(self) -> None:
        plan = ap.analyze_utterance("to shěsonů", dictionary())
        self.assertEqual(len(plan.words), 2)
        self.assertEqual(
            [(word.mora_start, word.mora_end) for word in plan.words],
            [(0, 1), (1, 4)],
        )
        self.assertEqual(
            [word.phrase_expression for word in plan.words],
            ["to shěsonů", "to shěsonů"],
        )

    def test_idiom_is_recognized_inside_a_longer_utterance(self) -> None:
        plan = ap.analyze_utterance(
            "xoxo to shěsonů xoxo.",
            dictionary(),
        )

        self.assertEqual(
            [word.phrase_expression for word in plan.words],
            ["", "to shěsonů", "to shěsonů", ""],
        )
        self.assertEqual(
            sum(
                item.code == "phrase_dictionary_override"
                for item in plan.diagnostics
            ),
            1,
        )

    def test_longest_three_word_idiom_wins_over_embedded_shorter_idiom(
        self,
    ) -> None:
        plan = ap.analyze_utterance(
            "xoxo kè to shěsonů xoxo.",
            dictionary(),
        )

        self.assertEqual(
            [word.phrase_expression for word in plan.words],
            [
                "",
                "kè to shěsonů",
                "kè to shěsonů",
                "kè to shěsonů",
                "",
            ],
        )
        self.assertEqual(
            [word.pitch_accent for word in plan.words[1:4]],
            ["L", "L", "L.H.H"],
        )

    def test_repeated_idiom_occurrences_are_matched_independently(self) -> None:
        plan = ap.analyze_utterance(
            "to shěsonů to shěsonů.",
            dictionary(),
        )

        self.assertEqual(
            [word.phrase_expression for word in plan.words],
            ["to shěsonů"] * 4,
        )
        self.assertEqual(
            sum(
                item.code == "phrase_dictionary_override"
                for item in plan.diagnostics
            ),
            1,
        )

    def test_unsupported_letter_cannot_silently_split_an_idiom(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported Asaxi grapheme"):
            ap.analyze_utterance("to shěsonū", dictionary())

    def test_user_phone_override_is_final_and_remains_mora_aligned(self) -> None:
        plan = ap.analyze_utterance(
            "shěsonů",
            dictionary(),
            phone_overrides={"shěsonů": ("sh", "e", "s", "o", "n", "u")},
        )
        self.assertEqual(
            plan.phones,
            ("sh", "e", "s", "o", "n", "u"),
        )
        self.assertEqual(plan.moras[-1].phone_end, len(plan.phones))
        self.assertTrue(any(
            item.code == "user_g2p_override"
            for item in plan.diagnostics
        ))

    def test_question_target_rises_and_statement_target_falls(self) -> None:
        statement = ap.analyze_utterance("shěsonů.", dictionary())
        question = ap.analyze_utterance("shěsonů?", dictionary())
        segments = []
        cursor = 0.1
        for phone in statement.phones:
            segments.append((phone, cursor, cursor + 0.06))
            cursor += 0.06
        statement_targets, _ = ap.targets_for_segments(
            statement, segments, base_pitch_hz=160.0
        )
        question_targets, _ = ap.targets_for_segments(
            question, segments, base_pitch_hz=160.0
        )
        self.assertLess(statement_targets[-1][1], 160.0)
        self.assertGreater(question_targets[-1][1], 160.0)

    def test_mora_pitch_offset_changes_only_its_linguistic_target(self) -> None:
        plan = ap.analyze_utterance("shěsonů.", dictionary())
        segments = []
        cursor = 0.1
        for phone in plan.phones:
            segments.append((phone, cursor, cursor + 0.06))
            cursor += 0.06
        aligned, _ = ap.rendered_morae(plan, segments)
        target_mora = aligned[1]
        target_time = round(
            target_mora.start + (target_mora.end - target_mora.start) * 0.58,
            6,
        )
        baseline, _ = ap.targets_for_segments(
            plan, segments, base_pitch_hz=160.0)
        raised, _ = ap.targets_for_segments(
            plan,
            segments,
            base_pitch_hz=160.0,
            mora_pitch_offsets_cents={1: 1200},
        )

        self.assertAlmostEqual(
            dict(raised)[target_time],
            dict(baseline)[target_time] * 2.0,
            places=2,
        )
        other_time = round(
            aligned[2].start +
            (aligned[2].end - aligned[2].start) * 0.58,
            6,
        )
        self.assertEqual(dict(raised)[other_time], dict(baseline)[other_time])

    def test_mora_tone_override_changes_the_selected_h_l_target(self) -> None:
        plan = ap.analyze_utterance("shěso.", dictionary())
        segments = []
        cursor = 0.1
        for phone in plan.phones:
            segments.append((phone, cursor, cursor + 0.06))
            cursor += 0.06
        aligned, _ = ap.rendered_morae(plan, segments)
        target = aligned[1]
        target_time = round(
            target.start + (target.end - target.start) * 0.58, 6)
        replacement = "L" if target.pitch == "H" else "H"
        original_semitones = 1.8 if target.pitch == "H" else -1.2
        replacement_semitones = 1.8 if replacement == "H" else -1.2
        baseline, _ = ap.targets_for_segments(
            plan, segments, base_pitch_hz=160.0)
        edited, _ = ap.targets_for_segments(
            plan,
            segments,
            base_pitch_hz=160.0,
            mora_tone_overrides={target.index: replacement},
        )

        self.assertAlmostEqual(
            dict(edited)[target_time] / dict(baseline)[target_time],
            2.0 ** ((replacement_semitones - original_semitones) / 12.0),
            places=4,
        )

    def test_multi_phrase_alignment_uses_sentence_level_mora_indexes(
        self,
    ) -> None:
        first = ap.analyze_utterance("shěsonů.", dictionary())
        second = ap.analyze_utterance("xoxo?", dictionary())
        segments = []
        cursor = 0.0
        for phone in first.phones:
            segments.append((phone, cursor, cursor + 0.05))
            cursor += 0.05
        segments.extend([
            ("pau", cursor, cursor + 0.08),
            ("pau", cursor + 0.08, cursor + 0.16),
        ])
        cursor += 0.16
        for phone in second.phones:
            segments.append((phone, cursor, cursor + 0.05))
            cursor += 0.05

        aligned, _ = ap.rendered_morae((first, second), segments)
        targets, _ = ap.targets_for_plans(
            (first, second), segments, base_pitch_hz=160.0)

        self.assertEqual(
            [mora.index for mora in aligned],
            list(range(len(aligned))))
        self.assertEqual(
            [mora.phrase_index for mora in aligned],
            [0] * len(first.moras) + [1] * len(second.moras))
        self.assertGreater(targets[-1][1], 160.0)

    def test_substantial_festival_phone_mismatch_is_rejected(self) -> None:
        plan = ap.analyze_utterance("shěsonů", dictionary())
        with self.assertRaisesRegex(ValueError, "do not align"):
            ap.targets_for_segments(plan, [("z", 0.0, 0.1)])

    def test_attested_morphology_is_composed_from_dictionary_parts(
        self,
    ) -> None:
        real = ap.load_dictionary()
        expected = {
            "zènáshěsonů.": "H.L.H.L.L",
            "pazènáchỏnů.": "H.L.H.H.H",
            "ŕoŕo daohè!": "H.L.H.L.L",
            "haśùnáhè!": "L.H.H.L",
            "wo zèxăcè wő.": "L.L.H.H.H",
        }
        for text, pitch in expected.items():
            with self.subTest(text=text):
                plan = ap.analyze_utterance(text, real)
                self.assertEqual(self._pitch(plan), pitch)
                self.assertTrue(any(
                    item.code == "morphological_pitch_inference"
                    for item in plan.diagnostics
                ))
                self.assertFalse(any(
                    item.code == "no_matching_lexical_units"
                    for item in plan.diagnostics
                ))

    def test_documented_plural_allomorphs_inherit_root_pitch(self) -> None:
        real = ap.load_dictionary()
        cases = {
            "sháma": ("H.H", "shá", "-ma", "plural-a-ma"),
            "shěsa": ("H.L", "shěso", "-a", "plural-o-replacement"),
            "dăa": ("H.H", "dă", "-a", "plural-simple-suffix"),
            "kama": (
                "H.L",
                "kamm",
                "-a",
                "plural-syllabic-resolution",
            ),
            "gaviwa": ("H.L.L", "gavi", "-wa", "plural-vowel-bridge"),
            "pỏpa": (
                "H.L",
                "pỏpỏ",
                "-a",
                "plural-reduplicated-diphthong-reduction",
            ),
        }
        for surface, expected in cases.items():
            with self.subTest(surface=surface):
                pitch, root, suffix, rule = expected
                plan = ap.analyze_utterance(surface + ".", real)
                word = plan.words[0]
                self.assertEqual(word.pitch_accent, pitch)
                self.assertEqual(word.pitch_accent_class, "morphological")
                self.assertEqual(word.lexical_type, "noun")
                self.assertEqual(
                    [morpheme.lemma for morpheme in word.morphemes],
                    [root, suffix],
                )
                self.assertEqual(word.morphemes[-1].role, "plural")
                self.assertEqual(word.morphemes[-1].rule, rule)
                self.assertFalse(any(
                    item.code == "no_matching_lexical_units"
                    for item in plan.diagnostics
                ))

    def test_plural_parser_is_typed_and_does_not_inflect_verbs(self) -> None:
        real = ap.load_dictionary()
        exact = ap.analyze_utterance("ma.", real)
        hypothetical = ap.analyze_utterance("mama.", real)

        self.assertFalse(exact.words[0].morphemes)
        self.assertFalse(exact.diagnostics)
        self.assertFalse(hypothetical.words[0].morphemes)
        self.assertTrue(any(
            item.code == "no_matching_lexical_units"
            for item in hypothetical.diagnostics
        ))

    def test_morpheme_analysis_is_structurally_serialized(self) -> None:
        plan = ap.analyze_utterance("sháma.", ap.load_dictionary())
        serialized = plan.to_dict()["words"][0]["morphemes"]

        self.assertEqual(
            [(item["lemma"], item["role"]) for item in serialized],
            [("shá", "root"), ("-ma", "plural")],
        )
        self.assertEqual(
            serialized[-1]["pitch_accent_class"],
            "atonal",
        )
        self.assertEqual(serialized[-1]["rule"], "plural-a-ma")
        self.assertFalse(Path(serialized[-1]["source_note"]).is_absolute())

    def test_compound_analysis_preserves_nested_head_inflection(self) -> None:
        plan = ap.analyze_utterance(
            "gapỏbifùbiwa.",
            ap.load_dictionary(),
        )
        word = plan.words[0]

        self.assertEqual(word.pitch_accent, "L.H.L.L.L.L")
        self.assertEqual(word.pitch_accent_class, "morphological")
        self.assertEqual(
            [(item.lemma, item.role) for item in word.morphemes],
            [
                ("ga-", "compound-prefix"),
                ("pỏbi", "compound-modifier"),
                ("fùbiwa", "compound-head"),
            ],
        )
        self.assertEqual(
            [
                (item.lemma, item.role)
                for item in word.morphemes[-1].children
            ],
            [("fùbi", "root"), ("-wa", "plural")],
        )
        serialized = word.to_dict()["morphemes"]
        self.assertEqual(
            serialized[-1]["children"][-1]["rule"],
            "plural-vowel-bridge",
        )
        self.assertIn(
            "fùbiwa (compound head: fùbi (root) + -wa (plural))",
            ap.format_morpheme_analysis(word.morphemes),
        )

    def test_compound_accent_unifies_on_first_lexical_member(self) -> None:
        plan = ap.analyze_utterance("gaviŕoŕo.", ap.load_dictionary())

        self.assertEqual(plan.words[0].pitch_accent, "H.L.L.L")
        self.assertEqual(
            [item.lemma for item in plan.words[0].morphemes],
            ["gavi", "ŕoŕo"],
        )

    def test_attested_morphophonology_can_merge_adjacent_vowels(self) -> None:
        real = ap.load_dictionary()
        attested = replace(
            real,
            morphological_analyses={
                "gaksamipỏpỏ": {
                    "default_analysis": 0,
                    "analyses": [{
                        "notation": "ga-aksami-pỏpỏ",
                        "segments": ["ga", "aksami", "pỏpỏ"],
                        "source_notes": ["fixtures/interlinear.md#L1"],
                    }],
                },
            },
        )
        plan = ap.analyze_utterance("gaksamipỏpỏ.", attested)

        self.assertEqual(plan.words[0].pitch_accent, "H.L.L.L.L")
        self.assertEqual(
            [item.lemma for item in plan.words[0].morphemes],
            ["ga-", "aksami", "pỏpỏ"],
        )
        self.assertFalse(any(
            item.code == "no_matching_lexical_units"
            for item in plan.diagnostics
        ))

    def test_attested_analysis_still_requires_known_lexical_units(self) -> None:
        real = ap.load_dictionary()
        attested = replace(
            real,
            morphological_analyses={
                "gafnè": {
                    "default_analysis": 0,
                    "analyses": [{
                        "notation": "ga-notalexeme",
                        "segments": ["ga", "notalexeme"],
                        "source_notes": ["fixtures/interlinear.md#L1"],
                    }],
                },
            },
        )
        plan = ap.analyze_utterance("gafnè.", attested)

        self.assertTrue(any(
            item.code == "no_matching_lexical_units"
            for item in plan.diagnostics
        ))

    def test_written_bound_form_resolves_through_morpheme_index(self) -> None:
        base = dictionary()
        indexed = replace(
            base,
            morphemes={
                "va-": {
                    "surface": "va",
                    "canonical_form": "va-",
                    "role": "relational-locative-prefix",
                    "attachment": "prefix",
                    "lexical_type": "particle",
                    "phones": list(af.g2p_asaxi("va")),
                    "pitch_accent": "L",
                    "pitch_accent_class": "atonal",
                    "source_notes": ["fixtures/va.md"],
                },
            },
        )
        plan = ap.analyze_utterance("va.", indexed)

        self.assertEqual(plan.words[0].pitch_accent, "L")
        self.assertEqual(plan.words[0].lexical_type, "particle")
        self.assertEqual(plan.words[0].morphemes[0].lemma, "va-")
        self.assertTrue(any(
            item.code == "standalone_morpheme_resolution"
            for item in plan.diagnostics
        ))
        self.assertFalse(any(
            item.code == "no_matching_lexical_units"
            for item in plan.diagnostics
        ))

    def test_nonmoraic_bridge_can_precede_its_attested_root(self) -> None:
        base = dictionary()
        analysis = af.analyze_word("ů")
        variant = {
            "lexical_type": "verb",
            "phones": analysis.phones,
            "pitch_accent": "H",
            "pitch_accent_class": "lexical",
            "source_note": "fixtures/active-be.md",
        }
        active_be = af.AsaxiLexiconEntry(
            word="ů",
            phones=analysis.phones,
            moras=analysis.moras,
            pitch_accent="H",
            pitch_accent_class="lexical",
            source_note="fixtures/active-be.md",
            variants=(variant,),
        )
        indexed = replace(
            base,
            entries={**base.entries, "ů": active_be},
            morphemes={
                "-b-": {
                    "surface": "b",
                    "canonical_form": "-b-",
                    "role": "verbal-bridge",
                    "attachment": "infix",
                    "lexical_type": "particle",
                    "phones": ["b"],
                    "pitch_accent": "none",
                    "pitch_accent_class": "atonal",
                    "source_notes": ["fixtures/bridge.md"],
                    "registry_source": "fixtures/grammar.toml",
                },
            },
            morphological_analyses={
                "bů": {
                    "default_analysis": 0,
                    "analyses": [{
                        "notation": "b-ů",
                        "segments": ["b", "ů"],
                        "source_notes": ["fixtures/interlinear.md#L1"],
                    }],
                },
            },
        )
        plan = ap.analyze_utterance("bů.", indexed)

        self.assertEqual(
            [item.lemma for item in plan.words[0].morphemes],
            ["-b-", "-ů"],
        )
        self.assertFalse(any(
            item.code == "no_matching_lexical_units"
            for item in plan.diagnostics
        ))

    def test_boundary_tones_are_structural_and_serializable(self) -> None:
        real = ap.load_dictionary()
        cases = {
            "shěsonů.": "L%",
            "no xogă?": "LH%",
            "ŕoŕo daohè!": "LH%",
            "wo zèxăcè wő.": "H%",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                plan = ap.analyze_utterance(text, real)
                self.assertEqual(plan.boundary_tone, expected)
                self.assertEqual(plan.to_dict()["boundary_tone"], expected)

    def test_directive_preserves_lexical_onset(self) -> None:
        plan = ap.analyze_utterance("ŕoŕo daohè!", ap.load_dictionary())
        self.assertEqual(self._pitch(plan), "H.L.H.L.L")
        self.assertEqual(plan.moras[0].lexical_pitch, "H")
        self.assertEqual(plan.moras[0].pitch, "H")

    def test_vocative_deaccents_name_without_flattening_clause(self) -> None:
        plan = ap.analyze_utterance(
            "ăjo lem, måmå natăka!",
            ap.load_dictionary(),
        )
        self.assertEqual(self._pitch(plan), "H.L.L.H.L.L.H.L")
        self.assertEqual(
            [
                mora.pitch
                for mora in plan.moras[
                    plan.words[1].mora_start:plan.words[1].mora_end
                ]
            ],
            ["L"],
        )
        self.assertTrue(any(
            item.code == "vocative_name_deaccenting"
            for item in plan.diagnostics
        ))

    def test_unanalyzable_unknown_word_keeps_explicit_warning(self) -> None:
        plan = ap.analyze_utterance("lem.", ap.load_dictionary())
        self.assertTrue(any(
            item.code == "no_matching_lexical_units"
            for item in plan.diagnostics
        ))


if __name__ == "__main__":
    unittest.main()
