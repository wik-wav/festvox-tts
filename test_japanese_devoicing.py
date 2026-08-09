from types import SimpleNamespace
import json
import unittest

import numpy as np

from festvox_gui.festvox_core import Segment, Synthesis
from japanese_devoicing import (
    apply_voicing_override,
    apply_vowel_realizations,
    hold_first_post_phrase_pause,
    initialize_voicing_metadata,
    periodicity_score,
    predict_mora_voicing,
    requested_realizations,
    restore_pause_samples,
)


SR = 16000


def _plan(*, requested=True, phone="u", duration=0.12):
    effects = {"devoiced_high_vowel": -0.22} if requested else {}
    allocation = SimpleNamespace(
        segment_index=0,
        phone=phone,
        context_effects=effects,
        final_duration=duration,
    )
    mora = SimpleNamespace(mora_index=0, phone_allocation=(allocation,))
    return SimpleNamespace(
        mora_timings=(mora,),
        segments=(SimpleNamespace(phone=phone),),
        speed=1.0,
    )


def _context_plan(*, speed=1.0, phrase_final=False,
                  accent_distance=1, previous_devoiced=False,
                  special_mora=None):
    context = {
        "previous_phone": "k",
        "following_phone": None if phrase_final else "s",
        "openjtalk_devoiced": True,
        "accent_distance": accent_distance,
        "accent_phrase_final": phrase_final,
        "phrase_final": phrase_final,
        "previous_mora_devoiced": previous_devoiced,
        "special_mora": special_mora,
    }
    effects = ({"devoiced_high_vowel": -0.22}
               if not previous_devoiced and special_mora != "long_vowel"
               else {"consecutive_devoicing_avoidance": 0.16}
               if previous_devoiced else {})
    allocation = SimpleNamespace(
        segment_index=0, phone="u", context_effects=effects,
        context=context, final_duration=0.10,
    )
    mora = SimpleNamespace(mora_index=0, phone_allocation=(allocation,))
    return SimpleNamespace(
        mora_timings=(mora,), segments=(SimpleNamespace(phone="u"),),
        speed=speed,
    )


def _synthesis(samples, *, phone="u"):
    values = np.asarray(samples, np.float32)
    duration = len(values) / float(SR)
    return Synthesis(
        samples=values.copy(),
        sr=SR,
        segments=[Segment(phone, 0.0, duration)],
        text="kutsu",
        lang="ja",
    )


class JapaneseDevoicingTests(unittest.TestCase):
    def test_mora_prediction_is_contextual_continuous_and_overridable(self):
        neutral = predict_mora_voicing(_context_plan())[0]
        protected = predict_mora_voicing(_context_plan(
            speed=0.7, phrase_final=True, accent_distance=0))[0]
        fast = predict_mora_voicing(_context_plan(speed=1.8))[0]
        consecutive = predict_mora_voicing(_context_plan(
            previous_devoiced=True))[0]
        overridden = predict_mora_voicing(
            _context_plan(), {0: 0.43})[0]

        self.assertLess(fast.automatic_voicing,
                        neutral.automatic_voicing)
        self.assertGreater(protected.automatic_voicing,
                           neutral.automatic_voicing)
        self.assertEqual(consecutive.automatic_voicing, 1.0)
        self.assertAlmostEqual(overridden.automatic_voicing,
                               neutral.automatic_voicing)
        self.assertAlmostEqual(overridden.final_voicing, 0.43)
        self.assertIn("explicit mora voicing override is final",
                      overridden.reasons)

    def test_multiple_mora_overrides_keep_automatic_diagnostics(self):
        allocations = []
        moras = []
        for index, phone in enumerate(("i", "u")):
            allocation = SimpleNamespace(
                segment_index=index, phone=phone,
                context_effects={"devoiced_high_vowel": -0.22},
                context={"previous_phone": "k", "following_phone": "s"},
                final_duration=0.12,
            )
            allocations.append(allocation)
            moras.append(SimpleNamespace(
                mora_index=index, phone_allocation=(allocation,)))
        plan = SimpleNamespace(
            mora_timings=tuple(moras),
            segments=tuple(SimpleNamespace(phone=item.phone)
                           for item in allocations),
            speed=1.0,
        )
        time = np.arange(round(SR * 0.24), dtype=np.float64) / SR
        rng = np.random.default_rng(121)
        source = (
            0.30 * np.sin(2.0 * np.pi * 180.0 * time)
            + 0.025 * np.convolve(
                rng.normal(size=time.size), [1.0, -0.9], mode="same")
        ).astype(np.float32)
        synthesis = Synthesis(
            source.copy(), SR,
            [Segment("i", 0.0, 0.12), Segment("u", 0.12, 0.24)],
            lang="ja",
        )

        rendered = apply_vowel_realizations(
            synthesis, plan,
            mora_voicing_overrides={0: 0.28, 1: 0.52},
        )

        self.assertEqual(
            [row["manual_mora_override"]
             for row in rendered.vowel_realizations],
            [0.28, 0.52],
        )
        diagnostic = next(
            row for row in rendered.voicing_diagnostics
            if row.get("kind") == "japanese_mora_voicing_predictions")
        self.assertEqual(len(diagnostic["decisions"]), 2)
        self.assertTrue(all(row["overridden"]
                            for row in diagnostic["decisions"]))
        self.assertEqual(rendered.voicing_override, [])

    def test_first_post_phrase_pause_holds_preceding_control_value_only(self):
        points = [(index * .01, index / 30.0) for index in range(31)]
        segments = [
            Segment("pau", 0.0, .02),
            Segment("a", .02, .08),
            Segment("pau", .08, .12),
            Segment("pau", .12, .18),
            Segment("a", .18, .24),
            Segment("pau", .24, .30),
        ]
        held = dict(hold_first_post_phrase_pause(points, segments))
        preceding = dict(points)[.07]
        self.assertAlmostEqual(held[.08], preceding, places=6)
        self.assertAlmostEqual(held[.09], preceding, places=6)
        self.assertAlmostEqual(held[.11], preceding, places=6)
        self.assertAlmostEqual(held[.12], dict(points)[.12], places=6)
        self.assertAlmostEqual(held[.15], dict(points)[.15], places=6)
        self.assertAlmostEqual(held[.24], dict(points)[.23], places=6)
        self.assertAlmostEqual(held[.01], dict(points)[.01], places=6)

    def test_pause_mask_restores_original_silence_exactly(self):
        source = np.zeros(round(SR * .30), np.float32)
        time = np.arange(round(SR * .10), dtype=np.float64) / SR
        voiced = np.asarray(.2 * np.sin(2 * np.pi * 180.0 * time),
                            np.float32)
        source[:len(voiced)] = voiced
        source[-len(voiced):] = voiced
        rendered = source + .05
        segments = [Segment("a", 0.0, .10),
                    Segment("pau", .10, .20),
                    Segment("a", .20, .30)]
        restored = restore_pause_samples(rendered, source, SR, segments)
        first = round(.10 * SR)
        last = round(.20 * SR)
        np.testing.assert_array_equal(restored[first:last], source[first:last])

    def test_manual_curve_can_edit_voiced_audio_inside_pause_region(self):
        time = np.arange(round(SR * .30), dtype=np.float64) / SR
        source = np.asarray(.25 * np.sin(2 * np.pi * 180.0 * time),
                            np.float32)
        synthesis = Synthesis(
            source.copy(), SR,
            [Segment("a", 0.0, .10), Segment("pau", .10, .15),
             Segment("pau", .15, .20), Segment("a", .20, .30)],
        )
        initialize_voicing_metadata(synthesis)
        first_pause = [value for time, value in
                       synthesis.generated_voicing_targets
                       if .10 <= time < .15]
        self.assertTrue(first_pause)
        self.assertLess(max(first_pause) - min(first_pause), 1e-8)
        rendered = apply_voicing_override(
            synthesis,
            [(0.0, 1.0), (.095, 1.0), (.105, 0.0),
             (.195, 0.0), (.205, 1.0), (.30, 1.0)],
        )
        first = round(.115 * SR)
        last = round(.185 * SR)
        self.assertGreater(
            float(np.max(np.abs(rendered.samples[first:last]
                                - source[first:last]))),
            1.0e-4,
        )

    def test_requested_realization_is_structural_not_an_f0_zero(self):
        rows = requested_realizations(_plan())

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].phone, "u")
        self.assertEqual(rows[0].strategy, "pending")
        self.assertGreater(rows[0].target_duration, 0.0)

    def test_source_filter_mix_is_deterministic_and_reduces_periodicity(self):
        time = np.arange(round(SR * 0.12), dtype=np.float64) / SR
        rng = np.random.default_rng(21)
        source = (
            0.30 * np.sin(2.0 * np.pi * 180.0 * time)
            + 0.025 * np.convolve(
                rng.normal(size=time.size), [1.0, -0.9], mode="same"
            )
        ).astype(np.float32)
        before = periodicity_score(source, SR)

        first = apply_vowel_realizations(_synthesis(source), _plan())
        second = apply_vowel_realizations(_synthesis(source), _plan())
        decision = first.vowel_realizations[0]

        self.assertEqual(
            decision["strategy"], "source_filter_residual_devoiced"
        )
        self.assertTrue(decision["applied"])
        self.assertGreater(before, decision["periodicity_after"])
        self.assertGreaterEqual(
            before - decision["periodicity_after"], 0.06
        )
        self.assertLess(decision["spectral_envelope_distance"], 0.62)
        self.assertLessEqual(
            abs(decision["level_step_db"]
                - decision["expected_level_step_db"]),
            3.0,
        )
        json.dumps(first.vowel_realizations)
        np.testing.assert_array_equal(first.samples, second.samples)
        self.assertFalse(np.array_equal(first.samples, source))

    def test_naturally_aperiodic_source_is_preferred_and_unchanged(self):
        rng = np.random.default_rng(42)
        source = (rng.normal(0.0, 0.12, round(SR * 0.12))).astype(np.float32)

        rendered = apply_vowel_realizations(_synthesis(source), _plan())
        decision = rendered.vowel_realizations[0]

        self.assertEqual(decision["strategy"], "naturally_devoiced_source")
        self.assertTrue(decision["source_was_naturally_aperiodic"])
        np.testing.assert_array_equal(rendered.samples, source)

    def test_shortened_voiced_mode_is_an_honest_non_destructive_fallback(self):
        time = np.arange(round(SR * 0.12), dtype=np.float64) / SR
        source = (0.25 * np.sin(2.0 * np.pi * 160.0 * time)).astype(np.float32)

        rendered = apply_vowel_realizations(
            _synthesis(source), _plan(), renderer="shortened_voiced"
        )
        decision = rendered.vowel_realizations[0]

        self.assertEqual(decision["strategy"], "shortened_voiced_fallback")
        self.assertFalse(decision["applied"])
        np.testing.assert_array_equal(rendered.samples, source)

    def test_legacy_and_unrequested_vowels_are_stable(self):
        time = np.arange(round(SR * 0.08), dtype=np.float64) / SR
        source = (0.2 * np.sin(2.0 * np.pi * 200.0 * time)).astype(np.float32)

        legacy = apply_vowel_realizations(
            _synthesis(source), _plan(), mode="legacy"
        )
        ordinary = apply_vowel_realizations(
            _synthesis(source), _plan(requested=False)
        )

        self.assertEqual(legacy.vowel_realizations, [])
        self.assertEqual(ordinary.vowel_realizations, [])
        np.testing.assert_array_equal(legacy.samples, source)
        np.testing.assert_array_equal(ordinary.samples, source)

    def test_silence_and_short_intervals_do_not_emit_nan(self):
        source = np.zeros(16, np.float32)
        rendered = apply_vowel_realizations(_synthesis(source), _plan())
        decision = rendered.vowel_realizations[0]

        self.assertEqual(decision["strategy"], "naturally_devoiced_source")
        self.assertIsNone(decision["periodicity_before"])
        self.assertFalse(np.isnan(rendered.samples).any())

    def test_manual_curve_does_not_replace_automatic_dashed_reference(self):
        time = np.arange(round(SR * 0.12), dtype=np.float64) / SR
        rng = np.random.default_rng(91)
        source = (
            0.30 * np.sin(2.0 * np.pi * 180.0 * time)
            + 0.025 * np.convolve(
                rng.normal(size=time.size), [1.0, -0.9], mode="same"
            )
        ).astype(np.float32)
        manual = [(0.0, 0.78), (0.06, 0.72), (0.12, 0.80)]

        automatic = apply_vowel_realizations(
            _synthesis(source), _plan()
        )
        edited = apply_vowel_realizations(
            _synthesis(source), _plan(), voicing_override=manual
        )
        regenerated = apply_vowel_realizations(
            _synthesis(source), _plan(), voicing_override=manual
        )

        self.assertEqual(
            edited.generated_voicing_targets,
            automatic.generated_voicing_targets,
        )
        self.assertEqual(
            regenerated.generated_voicing_targets,
            edited.generated_voicing_targets,
        )
        self.assertEqual(edited.voicing_override, manual)
        self.assertNotEqual(
            edited.generated_voicing_targets, edited.voicing_override
        )

    def test_mora_override_does_not_replace_automatic_dashed_reference(self):
        time = np.arange(round(SR * 0.12), dtype=np.float64) / SR
        rng = np.random.default_rng(92)
        source = (
            0.30 * np.sin(2.0 * np.pi * 180.0 * time)
            + 0.025 * np.convolve(
                rng.normal(size=time.size), [1.0, -0.9], mode="same")
        ).astype(np.float32)

        automatic = apply_vowel_realizations(_synthesis(source), _plan())
        voiced = apply_vowel_realizations(
            _synthesis(source), _plan(), mora_voicing_overrides={0: 1.0}
        )

        self.assertEqual(voiced.generated_voicing_targets,
                         automatic.generated_voicing_targets)
        self.assertEqual(voiced.voicing_override, [])
        np.testing.assert_array_equal(voiced.samples, source)
        self.assertEqual(
            voiced.vowel_realizations[0]["strategy"],
            "manual_voiced_source",
        )


if __name__ == "__main__":
    unittest.main()
