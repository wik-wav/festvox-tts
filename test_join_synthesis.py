import array
import math
import random
import unittest

from join_synthesis import (
    JoinConstraintError,
    JoinSynthesisConfig,
    adaptive_join_pcm16,
    legacy_linear_join_pcm16,
)


SAMPLE_RATE = 16000


def _sine(frequency=200.0, seconds=0.20, amplitude=12000.0,
          phase=0.0, sample_offset=0):
    count = int(round(SAMPLE_RATE * seconds))
    return array.array("h", [
        int(round(amplitude * math.sin(
            2.0 * math.pi * frequency * (index + sample_offset) /
            SAMPLE_RATE + phase)))
        for index in range(count)
    ])


def _harmonic_vowel(seconds=0.20, sample_offset=0):
    count = int(round(SAMPLE_RATE * seconds))
    return array.array("h", [
        int(round(
            9500.0 * math.sin(
                2.0 * math.pi * 200.0 * (index + sample_offset) /
                SAMPLE_RATE) +
            6500.0 * math.sin(
                2.0 * math.pi * 600.0 * (index + sample_offset) /
                SAMPLE_RATE)
        ))
        for index in range(count)
    ])


class AdaptiveJoinTests(unittest.TestCase):
    def test_silent_non_pause_halves_cannot_pass_as_clean_join(self):
        left = array.array("h", [0] * 1600)
        right = array.array("h", [0] * 1600)
        decision = adaptive_join_pcm16(
            left, right, 16000, left_phone="a", right_phone="i"
        )

        self.assertFalse(decision.validation_passed)
        self.assertFalse(decision.left_source_content_present)
        self.assertFalse(decision.right_source_content_present)
        self.assertIn(
            "MISSING_LEFT_SOURCE_CONTENT", decision.validation_failures
        )
        self.assertIn(
            "MISSING_RIGHT_SOURCE_CONTENT", decision.validation_failures
        )

    def test_one_sided_missing_vowel_content_cannot_pass(self):
        left = array.array("h", [0] * 1600)
        right = _sine(180.0, 0.1, amplitude=9000)
        decision = adaptive_join_pcm16(
            left, right, 16000, expected_f0_hz=180.0,
            left_phone="a", right_phone="i"
        )

        self.assertFalse(decision.validation_passed)
        self.assertFalse(decision.left_source_content_present)
        self.assertTrue(decision.right_source_content_present)
        self.assertIn(
            "MISSING_LEFT_SOURCE_CONTENT", decision.validation_failures
        )

    def test_cross_phone_bridge_ranks_but_does_not_require_similarity(self):
        left = _sine(180.0, amplitude=6000)
        right = _harmonic_vowel()
        decision = adaptive_join_pcm16(
            left, right, SAMPLE_RATE, expected_f0_hz=180.0,
            left_phone="a", right_phone="k",
            enforce_acoustic_similarity=False,
        )

        self.assertFalse(decision.acoustic_validation_gate_active)
        self.assertTrue(decision.acoustic_validation_passed)
        self.assertNotIn("LEVEL_MISMATCH", decision.validation_failures)
        self.assertNotIn(
            "SPECTRAL_ENVELOPE_MISMATCH", decision.validation_failures
        )

    def test_continuous_periodic_signal_keeps_nominal_cut(self):
        left = _sine()
        right = _sine(sample_offset=len(left))

        decision = adaptive_join_pcm16(
            left, right, SAMPLE_RATE, expected_f0_hz=200.0)

        self.assertTrue(decision.voiced)
        self.assertEqual(decision.left_trim_samples, 0)
        self.assertEqual(decision.right_skip_samples, 0)
        self.assertAlmostEqual(decision.level_step_db, 0.0, places=3)
        self.assertGreater(decision.zero_lag_correlation, 0.99)
        self.assertTrue(decision.validation_passed)

    def test_crossover_duration_is_stable_while_period_count_tracks_f0(self):
        rows = []
        for frequency in (100.0, 400.0):
            left = _sine(frequency=frequency)
            right = _sine(
                frequency=frequency, sample_offset=len(left))
            rows.append(adaptive_join_pcm16(
                left, right, SAMPLE_RATE,
                expected_f0_hz=frequency))

        for decision in rows:
            self.assertAlmostEqual(
                decision.requested_crossover_ms, 40.0)
            self.assertAlmostEqual(
                decision.effective_crossover_ms, 40.0, delta=3.0)
        self.assertGreater(
            rows[1].crossover_period_count,
            rows[0].crossover_period_count)

    def test_clean_sustained_vowels_cover_280_and_650_ms(self):
        for seconds in (0.280, 0.650):
            with self.subTest(seconds=seconds):
                half = seconds * 0.5
                left = _sine(seconds=half)
                right = _sine(seconds=half, sample_offset=len(left))

                decision = adaptive_join_pcm16(
                    left, right, SAMPLE_RATE, expected_f0_hz=200.0)

                self.assertTrue(decision.voiced)
                self.assertTrue(decision.validation_passed)
                self.assertGreater(decision.source_energy_retention, 0.80)
                self.assertGreater(decision.crossfade_energy_retention, 0.80)
                self.assertLess(decision.content_attenuation_db, 1.0)

    def test_silent_source_boundary_cannot_pass_join_validation(self):
        left = _sine(seconds=0.325)
        right = _sine(seconds=0.325, sample_offset=len(left))
        config = JoinSynthesisConfig(search_ms=0.0)
        period = int(round(SAMPLE_RATE / 200.0))
        overlap = (
            int(round(SAMPLE_RATE * config.crossover_ms / 1000.0))
            // period * period)
        for index in range(len(left) - overlap, len(left)):
            left[index] = 0
        for index in range(overlap):
            right[index] = 0

        decision = adaptive_join_pcm16(
            left, right, SAMPLE_RATE, expected_f0_hz=200.0,
            config=config)

        self.assertLess(decision.source_energy_retention,
                        config.minimum_source_energy_retention)
        self.assertLess(decision.crossfade_energy_retention,
                        config.minimum_crossfade_energy_retention)
        self.assertGreater(decision.content_attenuation_db, 12.0)
        self.assertFalse(decision.validation_passed)

    def test_short_cancellation_notch_cannot_pass_aggregate_energy_gate(self):
        left = _sine()
        right = _sine(
            sample_offset=len(left), phase=math.pi)
        config = JoinSynthesisConfig(search_ms=0.0)

        decision = adaptive_join_pcm16(
            left,
            right,
            SAMPLE_RATE,
            expected_f0_hz=200.0,
            config=config,
        )

        self.assertGreaterEqual(
            decision.phase_mix_energy_retention,
            config.minimum_crossfade_energy_retention,
        )
        self.assertLess(
            decision.local_crossfade_energy_retention,
            config.minimum_local_crossfade_energy_retention,
        )
        self.assertFalse(decision.content_preservation_passed)
        self.assertTrue(decision.legacy_fallback_used)
        self.assertIn("CONTENT_RETENTION", decision.validation_failures)

    def test_phase_shift_is_aligned_without_gain_change(self):
        left = _sine()
        right = _sine(sample_offset=len(left), phase=math.pi / 2.0)

        decision = adaptive_join_pcm16(
            left, right, SAMPLE_RATE, expected_f0_hz=200.0)

        self.assertTrue(decision.voiced)
        self.assertGreater(
            decision.left_trim_samples + decision.right_skip_samples, 0)
        self.assertAlmostEqual(decision.level_step_db, 0.0, delta=0.05)
        self.assertGreater(decision.zero_lag_correlation, 0.90)
        self.assertGreaterEqual(
            decision.best_lag_correlation,
            decision.zero_lag_correlation)
        self.assertTrue(decision.validation_passed)

    def test_level_gate_rejects_six_db_step_independently(self):
        left = _sine(amplitude=6000.0)
        right = _sine(amplitude=12000.0, sample_offset=len(left))

        decision = adaptive_join_pcm16(
            left, right, SAMPLE_RATE, expected_f0_hz=200.0)

        self.assertAlmostEqual(decision.level_step_db, 6.02, delta=0.15)
        self.assertEqual(decision.gain_ratio, 0.80)
        self.assertFalse(decision.level_validation_passed)
        self.assertEqual(decision.validation_failures,
                         ("LEVEL_MISMATCH",))
        self.assertFalse(decision.validation_passed)
        self.assertTrue(decision.legacy_fallback_used)

    def test_level_gate_threshold_is_configurable(self):
        left = _sine(amplitude=6000.0)
        right = _sine(amplitude=12000.0, sample_offset=len(left))
        config = JoinSynthesisConfig(validation_level_step_limit_db=6.1)

        decision = adaptive_join_pcm16(
            left, right, SAMPLE_RATE, expected_f0_hz=200.0,
            config=config)

        self.assertTrue(decision.level_validation_passed)
        self.assertTrue(decision.validation_passed)

    def test_f0_step_is_reported_separately(self):
        left = _sine(frequency=180.0)
        right = _sine(frequency=220.0, sample_offset=len(left))

        decision = adaptive_join_pcm16(
            left, right, SAMPLE_RATE, expected_f0_hz=200.0)

        self.assertTrue(decision.voiced)
        self.assertGreater(abs(decision.f0_step_semitones), 2.0)
        self.assertIsNotNone(decision.spectral_distance)
        self.assertFalse(decision.f0_validation_passed)
        self.assertIn("F0_MISMATCH", decision.validation_failures)
        self.assertTrue(decision.level_validation_passed)
        self.assertFalse(decision.validation_passed)

    def test_period_correlation_gate_is_independent(self):
        left = _sine()
        right = _harmonic_vowel(sample_offset=len(left))
        config = JoinSynthesisConfig(
            validation_level_step_limit_db=60.0,
            validation_f0_step_limit_semitones=12.0,
            validation_best_period_correlation_minimum=0.90,
            validation_period_shape_mismatch_limit=2.0,
            validation_spectral_envelope_distance_limit=10.0,
        )

        decision = adaptive_join_pcm16(
            left, right, SAMPLE_RATE, expected_f0_hz=200.0,
            config=config)

        self.assertLess(decision.best_lag_correlation, 0.90)
        self.assertFalse(decision.period_correlation_validation_passed)
        self.assertEqual(decision.validation_failures,
                         ("PERIOD_CORRELATION",))

    def test_period_shape_gate_is_independent(self):
        left = _sine()
        right = _harmonic_vowel(sample_offset=len(left))
        config = JoinSynthesisConfig(
            validation_level_step_limit_db=60.0,
            validation_f0_step_limit_semitones=12.0,
            validation_best_period_correlation_minimum=-1.0,
            validation_period_shape_mismatch_limit=0.20,
            validation_spectral_envelope_distance_limit=10.0,
        )

        decision = adaptive_join_pcm16(
            left, right, SAMPLE_RATE, expected_f0_hz=200.0,
            config=config)

        self.assertGreater(decision.period_shape_mismatch, 0.20)
        self.assertFalse(decision.period_shape_validation_passed)
        self.assertEqual(decision.validation_failures,
                         ("PERIOD_SHAPE_MISMATCH",))

    def test_spectral_envelope_gate_is_independent(self):
        left = _sine()
        right = _harmonic_vowel(sample_offset=len(left))
        config = JoinSynthesisConfig(
            validation_level_step_limit_db=60.0,
            validation_f0_step_limit_semitones=12.0,
            validation_best_period_correlation_minimum=-1.0,
            validation_period_shape_mismatch_limit=2.0,
            validation_spectral_envelope_distance_limit=0.50,
        )

        decision = adaptive_join_pcm16(
            left, right, SAMPLE_RATE, expected_f0_hz=200.0,
            config=config)

        self.assertGreater(decision.spectral_distance, 0.50)
        self.assertFalse(decision.spectral_validation_passed)
        self.assertEqual(decision.validation_failures,
                         ("SPECTRAL_ENVELOPE_MISMATCH",))

    def test_periodic_fricative_hint_forces_unvoiced_analysis(self):
        left = _sine()
        right = _sine(sample_offset=len(left))

        decision = adaptive_join_pcm16(
            left, right, SAMPLE_RATE, expected_f0_hz=200.0,
            left_phone="s", right_phone="a")

        self.assertFalse(decision.voiced)
        self.assertIsNone(decision.left_f0_hz)
        self.assertIsNone(decision.best_lag_correlation)
        self.assertEqual(decision.voicing_hint_reason,
                         "phone-context-forced-aperiodic")
        self.assertTrue(decision.validation_passed)

    def test_closure_hint_allows_a_real_silent_collar(self):
        left = _sine(seconds=0.325)
        right = _sine(seconds=0.325, sample_offset=len(left))
        silent_collar = int(round(SAMPLE_RATE * 0.040))
        left[-silent_collar:] = array.array("h", [0] * silent_collar)
        right[:silent_collar] = array.array("h", [0] * silent_collar)

        decision = adaptive_join_pcm16(
            left, right, SAMPLE_RATE, expected_f0_hz=200.0,
            left_phone="cl", right_phone="cl")

        self.assertTrue(decision.left_silence_allowed)
        self.assertTrue(decision.right_silence_allowed)
        self.assertFalse(decision.acoustic_validation_gate_active)
        self.assertTrue(decision.content_preservation_passed)
        self.assertTrue(decision.validation_passed)

    def test_true_silence_to_voicing_is_not_a_level_mismatch(self):
        left = array.array("h", [0] * int(round(SAMPLE_RATE * 0.20)))
        right = _sine()

        decision = adaptive_join_pcm16(
            left, right, SAMPLE_RATE, expected_f0_hz=200.0,
            allow_silent_left=True, left_phone="pau", right_phone="a")

        self.assertFalse(decision.acoustic_validation_gate_active)
        self.assertTrue(decision.level_validation_passed)
        self.assertNotIn("LEVEL_MISMATCH", decision.validation_failures)
        self.assertTrue(decision.validation_passed)

    def test_validation_failure_uses_exact_legacy_overlap_on_same_units(self):
        original_left = _sine(amplitude=6000.0)
        right = _sine(amplitude=12000.0,
                      sample_offset=len(original_left), phase=math.pi / 2.0)
        rendered = array.array("h", original_left)

        decision = adaptive_join_pcm16(
            rendered, right, SAMPLE_RATE, expected_f0_hz=200.0)
        expected = array.array("h", original_left)
        legacy_linear_join_pcm16(expected, right, decision.overlap_samples)

        self.assertTrue(decision.legacy_fallback_used)
        self.assertEqual(decision.left_trim_samples, 0)
        self.assertEqual(decision.right_skip_samples, 0)
        self.assertEqual(rendered.tobytes(), expected.tobytes())

    def test_unvoiced_join_is_deterministic_and_skips_period_metrics(self):
        generator = random.Random(417)
        left_values = [generator.randrange(-5000, 5001)
                       for _ in range(3200)]
        right_values = [generator.randrange(-5000, 5001)
                        for _ in range(3200)]
        first = array.array("h", left_values)
        second = array.array("h", left_values)

        first_decision = adaptive_join_pcm16(
            first, array.array("h", right_values), SAMPLE_RATE)
        second_decision = adaptive_join_pcm16(
            second, array.array("h", right_values), SAMPLE_RATE)

        self.assertFalse(first_decision.voiced)
        self.assertIsNone(first_decision.left_f0_hz)
        self.assertIsNone(first_decision.zero_lag_correlation)
        self.assertEqual(first_decision, second_decision)
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_right_cut_search_preserves_the_indexed_phone_tail(self):
        left = _sine()
        right = _sine(
            sample_offset=len(left), phase=math.pi / 2.0)
        indexed_length = 300
        required_tail = 120

        decision = adaptive_join_pcm16(
            left,
            right,
            SAMPLE_RATE,
            expected_f0_hz=200.0,
            right_indexed_length_samples=indexed_length,
            minimum_right_indexed_tail_samples=required_tail,
        )

        self.assertLessEqual(
            decision.right_skip_samples + decision.overlap_samples // 2 +
            required_tail,
            indexed_length,
        )
        self.assertTrue(decision.right_skip_limit_applied)
        self.assertEqual(
            decision.right_indexed_length_samples, indexed_length)
        self.assertGreaterEqual(
            decision.right_indexed_tail_samples, required_tail)

    def test_impossible_indexed_tail_raises_before_mutating_audio(self):
        left = _sine()
        original = left.tobytes()
        right = _sine(
            sample_offset=len(left), phase=math.pi / 2.0)

        with self.assertRaises(JoinConstraintError) as caught:
            adaptive_join_pcm16(
                left,
                right,
                SAMPLE_RATE,
                expected_f0_hz=200.0,
                right_indexed_length_samples=80,
                minimum_right_indexed_tail_samples=40,
            )

        self.assertEqual(caught.exception.code,
                         "right_indexed_region_too_short")
        self.assertLess(
            caught.exception.details["maximum_right_skip_samples"], 0)
        self.assertEqual(left.tobytes(), original)

    def test_exact_indexed_tail_budget_keeps_zero_skip_available(self):
        left = _sine()
        right = _sine(sample_offset=len(left))

        decision = adaptive_join_pcm16(
            left,
            right,
            SAMPLE_RATE,
            expected_f0_hz=200.0,
            config=JoinSynthesisConfig(crossover_ms=15.0),
            right_indexed_length_samples=160,
            minimum_right_indexed_tail_samples=40,
        )

        self.assertEqual(decision.overlap_samples, 240)
        self.assertEqual(decision.right_skip_samples, 0)
        self.assertEqual(decision.right_skip_limit_samples, 0)

    def test_legacy_join_is_exact_pre_fix_linear_formula(self):
        left = array.array("h", [1000, 2000, 3000, 4000])
        right = array.array("h", [-4000, -3000, -2000, -1000])

        legacy_linear_join_pcm16(left, right, 4)

        self.assertEqual(
            list(left), [1000, 750, 500, 250])


if __name__ == "__main__":
    unittest.main()
