# -*- coding: utf-8 -*-
"""Tests for inventory-aware Asaxi realization fallbacks."""

from __future__ import annotations

import unittest

import asaxi_frontend as af
import asaxi_phone_fallback as fallback
import asaxi_prosody as ap


def pairs(*values: str) -> dict[str, object]:
    return {"index": {value: {} for value in values}}


class AsaxiPhoneFallbackTests(unittest.TestCase):
    TEXT = "b\u1ecfhj\u00e1"

    def test_canonical_g2p_remains_compact(self) -> None:
        self.assertEqual(
            af.g2p_asaxi(self.TEXT),
            ("b", "o", "w", "hy", "ao"),
        )

    def test_supported_compound_transition_is_unchanged(self) -> None:
        plan = ap.analyze_utterance(self.TEXT)
        result = fallback.adapt_plan_for_inventory(
            plan,
            pairs("b-o", "o-w", "w-hy", "hy-ao"),
        )

        self.assertEqual(result.plan.phones, plan.phones)
        self.assertFalse(result.records)
        self.assertFalse(result.diagnostics)

    def test_missing_compound_transition_uses_verified_glide_path(self) -> None:
        plan = ap.analyze_utterance(self.TEXT)
        result = fallback.adapt_plan_for_inventory(
            plan,
            pairs(
                "b-o", "o-w", "w-hy",
                "w-h", "h-y", "y-y", "y-ao",
            ),
        )

        self.assertEqual(
            result.plan.phones,
            ("b", "o", "w", "h", "y", "y", "ao"),
        )
        self.assertEqual(
            result.plan.moras[1].phones,
            ("h", "y", "y", "ao"),
        )
        self.assertEqual(
            (result.plan.moras[1].phone_start,
             result.plan.moras[1].phone_end),
            (3, 7),
        )
        self.assertEqual(result.records[0].canonical_phone, "hy")
        self.assertEqual(
            result.records[0].missing_canonical_diphones,
            ("hy-ao",),
        )
        self.assertEqual(
            result.diagnostics[0].code,
            "asaxi_phone_fallback_applied",
        )

    def test_incomplete_fallback_retains_canonical_phones(self) -> None:
        plan = ap.analyze_utterance(self.TEXT)
        result = fallback.adapt_plan_for_inventory(
            plan,
            pairs("b-o", "o-w", "w-hy", "w-h", "h-y", "y-ao"),
        )

        self.assertEqual(result.plan.phones, plan.phones)
        self.assertFalse(result.records)
        self.assertEqual(
            result.diagnostics[0].code,
            "asaxi_phone_fallback_unavailable",
        )
        self.assertIn("y-y", result.diagnostics[0].message)

    def test_explicit_pronunciation_word_is_not_adapted(self) -> None:
        plan = ap.analyze_utterance(self.TEXT)
        result = fallback.adapt_plan_for_inventory(
            plan,
            pairs("w-h", "h-y", "y-y", "y-ao"),
            protected_word_indices={0},
        )

        self.assertEqual(result.plan.phones, plan.phones)
        self.assertFalse(result.records)
        self.assertFalse(result.diagnostics)

    def test_all_palatal_phone_classes_use_same_general_rule(self) -> None:
        for compound in sorted(af.PALATAL_PHONES):
            with self.subTest(compound=compound):
                base = compound[:-1]
                plan = ap.analyze_utterance("ma")
                mora = plan.moras[0]
                synthetic_mora = ap.AsaxiProsodyMora(
                    index=0,
                    word_index=0,
                    word="fixture",
                    text="fixture",
                    phones=(compound, "ao"),
                    phone_start=0,
                    phone_end=2,
                    lexical_pitch=mora.lexical_pitch,
                    pitch=mora.pitch,
                    accentable=True,
                    kind="ordinary",
                )
                synthetic_word = ap.AsaxiProsodyWord(
                    index=0,
                    surface="fixture",
                    lexical_type="fixture",
                    phones=(compound, "ao"),
                    pitch_accent="H",
                    pitch_accent_class="fixture",
                    mora_start=0,
                    mora_end=1,
                    dictionary_source="fixture",
                )
                synthetic_plan = ap.AsaxiProsodyPlan(
                    source_text="fixture",
                    normalized_text="fixture",
                    words=(synthetic_word,),
                    moras=(synthetic_mora,),
                    phones=(compound, "ao"),
                    boundary_mark="",
                    boundary_tone="L",
                    interrogative=False,
                    directive=False,
                    dictionary_ruleset="fixture",
                )
                result = fallback.adapt_plan_for_inventory(
                    synthetic_plan,
                    pairs(f"{base}-y", "y-y", "y-ao"),
                )
                self.assertEqual(
                    result.plan.phones,
                    (base, "y", "y", "ao"),
                )

    def test_diphone_metadata_parser_ignores_non_pair_fields(self) -> None:
        inventory = fallback.available_diphones({
            "name": "voice",
            "index": {
                "hy-ao": {},
                "invalid": {},
                "a__wl-hy": {},
            },
        })

        self.assertIn(("hy", "ao"), inventory)
        self.assertIn(("a__wl", "hy"), inventory)
        self.assertNotIn(("invalid", ""), inventory)


if __name__ == "__main__":
    unittest.main()
