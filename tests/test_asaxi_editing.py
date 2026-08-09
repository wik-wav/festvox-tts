# -*- coding: utf-8 -*-

import unittest

import asaxi_editing as ae


class AsaxiEditingTests(unittest.TestCase):
    def metadata(self):
        return {
            "moras": [
                {"mora_index": 0, "text": "shě"},
                {"mora_index": 1, "text": "so"},
            ]
        }

    def test_tone_and_zero_voicing_overrides_are_preserved(self):
        state = ae.new_edit_state("shěso")
        state = ae.with_mora_edit(state, "tone", [0], "L")
        state = ae.with_mora_edit(state, "voicing", [0], 0.0)

        self.assertEqual(state["mora_tone_overrides"], {"0": "L"})
        self.assertEqual(state["mora_voicing_overrides"], {"0": 0.0})

    def test_legacy_breathiness_migrates_into_single_voicing_control(self):
        state = ae.normalize_edit_state({
            "schema_version": 1,
            "source_text": "ox",
            "mora_breathiness_overrides": {"0": 0.5},
        })

        self.assertNotIn("mora_breathiness_overrides", state)
        self.assertAlmostEqual(
            state["mora_voicing_overrides"]["0"],
            1.0 - 0.5 * ae.BREATHINESS_HARMONIC_DEPTH,
        )

    def test_plan_reconciliation_keeps_same_text_overlays(self):
        state = ae.new_edit_state("shěso")
        state = ae.with_mora_edit(state, "tone", [0], "L")
        state = ae.with_mora_edit(state, "pitch", [0], 130)
        state = ae.with_mora_edit(state, "voicing", [1], 0.4)

        reconciled = ae.reconcile_plan(
            state, "shěso", self.metadata())

        self.assertEqual(reconciled["mora_tone_overrides"], {"0": "L"})
        self.assertEqual(
            reconciled["mora_pitch_offsets_cents"], {"0": 130})
        self.assertEqual(
            reconciled["mora_voicing_overrides"], {"1": 0.4})

    def test_changed_text_clears_stale_overlays(self):
        state = ae.new_edit_state("shěso")
        state = ae.with_mora_edit(state, "tone", [0], "L")
        state = ae.with_mora_edit(state, "pitch", [0], 130)
        state = ae.with_mora_edit(state, "voicing", [1], 0.7)

        reconciled = ae.reconcile_plan(
            state, "xoxo", self.metadata())

        self.assertEqual(reconciled["mora_tone_overrides"], {})
        self.assertEqual(reconciled["mora_pitch_offsets_cents"], {})
        self.assertEqual(reconciled["mora_voicing_overrides"], {})

    def test_missing_mora_drops_only_its_overlay(self):
        state = ae.new_edit_state("shěso")
        state = ae.with_mora_edit(state, "tone", [0, 1], "H")
        state = ae.with_mora_edit(state, "pitch", [0, 1], 80)

        reconciled = ae.reconcile_plan(
            state, "shěso",
            {"moras": [{"mora_index": 1, "text": "so"}]})

        self.assertEqual(reconciled["mora_tone_overrides"], {"1": "H"})
        self.assertEqual(
            reconciled["mora_pitch_offsets_cents"], {"1": 80})

    def test_exact_boundary_refinement_remaps_saved_mora_edits(self):
        old_metadata = {
            "moras": [
                {
                    "mora_index": 0,
                    "phrase_index": 0,
                    "word_index": 0,
                    "word": "nihèka",
                    "text": "nihè",
                },
                {
                    "mora_index": 1,
                    "phrase_index": 0,
                    "word_index": 0,
                    "word": "nihèka",
                    "text": "ka",
                },
            ],
        }
        new_metadata = {
            "moras": [
                {
                    "mora_index": 0,
                    "phrase_index": 0,
                    "word_index": 0,
                    "word": "nihèka",
                    "text": "ni",
                },
                {
                    "mora_index": 1,
                    "phrase_index": 0,
                    "word_index": 0,
                    "word": "nihèka",
                    "text": "hè",
                },
                {
                    "mora_index": 2,
                    "phrase_index": 0,
                    "word_index": 0,
                    "word": "nihèka",
                    "text": "ka",
                },
            ],
        }
        state = ae.new_edit_state("nihèka")
        state["last_plan"] = old_metadata
        state = ae.with_mora_edit(state, "tone", [0], "H")
        state = ae.with_mora_edit(state, "pitch", [1], 90)
        state = ae.with_mora_edit(state, "voicing", [1], 0.4)

        reconciled = ae.reconcile_plan(
            state, "nihèka", new_metadata)

        self.assertEqual(
            reconciled["mora_tone_overrides"],
            {"0": "H", "1": "H"},
        )
        self.assertEqual(
            reconciled["mora_pitch_offsets_cents"], {"2": 90})
        self.assertEqual(
            reconciled["mora_voicing_overrides"], {"2": 0.4})


if __name__ == "__main__":
    unittest.main()
