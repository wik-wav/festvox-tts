import math
import unittest

import numpy as np

from formant_analysis import estimate_f0
from vocal_tract import (
    VocalTractRange,
    VocalTractTransformConfig,
    control_position_to_ratio,
    load_vocal_tract_range,
    phone_warp_strength,
    ratio_curves_close,
    ratio_from_formant_semitones,
    ratio_to_control_position,
    ratio_to_formant_multiplier,
    ratio_to_formant_semitones,
    sample_ratio_targets,
    transform_vocal_tract,
)


def _vowel(sample_rate=16000, seconds=0.55, f0=180.0):
    count = int(sample_rate * seconds)
    source = np.zeros(count, np.float64)
    source[::max(1, int(round(sample_rate / f0)))] = 1.0
    values = source
    for formant, bandwidth in ((550.0, 80.0), (1950.0, 110.0),
                               (3050.0, 150.0), (3900.0, 190.0)):
        radius = math.exp(-math.pi * bandwidth / sample_rate)
        coefficient = 2.0 * radius * math.cos(
            2.0 * math.pi * formant / sample_rate
        )
        output = np.zeros_like(values)
        for index in range(values.size):
            output[index] = values[index]
            if index:
                output[index] += coefficient * output[index - 1]
            if index > 1:
                output[index] -= radius * radius * output[index - 2]
        values = output
    values /= max(1e-9, float(np.max(np.abs(values))))
    return values.astype(np.float32), sample_rate


class VocalTractTests(unittest.TestCase):
    def test_profile_loads_reference_derived_ranges(self):
        profile = load_vocal_tract_range()
        self.assertEqual(profile.identity_ratio, 1.0)
        self.assertAlmostEqual(profile.realistic_min_ratio, 0.93363995)
        self.assertAlmostEqual(profile.realistic_max_ratio, 1.05740237)
        self.assertAlmostEqual(profile.expanded_min_ratio, 0.65)
        self.assertAlmostEqual(profile.expanded_max_ratio, 1.5)
        self.assertLess(profile.expanded_min_ratio,
                        profile.realistic_min_ratio - 0.20)

    def test_ratio_conversions_and_direction(self):
        self.assertAlmostEqual(ratio_to_formant_multiplier(0.8), 1.25)
        self.assertGreater(ratio_to_formant_semitones(0.9), 0.0)
        self.assertLess(ratio_to_formant_semitones(1.1), 0.0)
        for value in (0.88, 1.0, 1.17):
            self.assertAlmostEqual(
                ratio_from_formant_semitones(
                    ratio_to_formant_semitones(value)
                ), value,
            )

    def test_control_mapping_has_exact_identity_center(self):
        profile = load_vocal_tract_range()
        self.assertEqual(control_position_to_ratio(0.5, profile), 1.0)
        self.assertAlmostEqual(ratio_to_control_position(1.0, profile), 0.5)
        self.assertAlmostEqual(
            control_position_to_ratio(0.0, profile),
            profile.realistic_max_ratio,
        )
        self.assertAlmostEqual(
            control_position_to_ratio(1.0, profile, True),
            profile.expanded_min_ratio,
        )
        for position in (0.0, 0.13, 0.5, 0.77, 1.0):
            ratio = control_position_to_ratio(position, profile, True)
            self.assertAlmostEqual(
                ratio_to_control_position(ratio, profile, True), position,
                places=7,
            )

    def test_clamping_switches_between_realistic_and_expanded(self):
        profile = load_vocal_tract_range()
        self.assertEqual(
            profile.clamp(profile.expanded_min_ratio, False),
            profile.realistic_min_ratio,
        )
        self.assertEqual(
            profile.clamp(profile.expanded_min_ratio, True),
            profile.expanded_min_ratio,
        )

    def test_identity_is_exact_and_deterministic(self):
        samples, rate = _vowel()
        first = transform_vocal_tract(samples, rate, 1.0)
        second = transform_vocal_tract(samples, rate, 1.0)
        self.assertTrue(first.identity_bypass)
        self.assertTrue(np.array_equal(first.samples, samples))
        self.assertTrue(np.array_equal(first.samples, second.samples))
        self.assertEqual(len(first.samples), len(samples))

    def test_ratio_curve_uses_log_interpolation_and_audible_equivalence(self):
        curve = [(0.0, 0.8), (1.0, 1.25)]
        self.assertAlmostEqual(sample_ratio_targets(curve, 0.5), 1.0)
        self.assertTrue(ratio_curves_close(
            [(0.0, 1.0), (1.0, 1.0)], [(0.5, 1.0)]))
        self.assertFalse(ratio_curves_close(curve, [(0.0, 1.0)]))

    def test_time_varying_curve_is_sampled_per_frame(self):
        samples, rate = _vowel(seconds=0.7)
        result = transform_vocal_tract(
            samples, rate, 1.0, chipmunk_range=True,
            ratio_targets=[(0.0, 0.70), (0.35, 1.0), (0.70, 1.40)],
            segments=[{"phone": "e", "start": 0.0, "end": 0.7}],
            config=VocalTractTransformConfig(
                runtime_true_envelope_iterations=6),
        )
        ratios = [row.applied_ratio for row in result.frame_diagnostics
                  if row.applied]
        self.assertGreater(max(ratios) - min(ratios), 0.5)
        self.assertEqual(result.applied_targets[0], (0.0, 0.7))
        self.assertEqual(result.applied_targets[-1], (0.7, 1.4))
        self.assertEqual(len(result.samples), len(samples))
        self.assertTrue(np.all(np.isfinite(result.samples)))

    def test_non_identity_preserves_duration_f0_and_finiteness(self):
        samples, rate = _vowel()
        segments = [{"phone": "e", "start": 0.0,
                     "end": len(samples) / rate}]
        result = transform_vocal_tract(
            samples, rate, 0.94, segments=segments,
            config=VocalTractTransformConfig(
                runtime_true_envelope_iterations=8
            ),
        )
        before = estimate_f0(samples[1600:6400], rate)[0]
        after = estimate_f0(result.samples[1600:6400], rate)[0]
        self.assertEqual(len(result.samples), len(samples))
        self.assertTrue(np.all(np.isfinite(result.samples)))
        self.assertGreater(result.modified_frame_count, 0)
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertLess(abs(12.0 * math.log2(after / before)), 0.08)

    def test_high_f0_expanded_transform_is_bounded_and_repeatable(self):
        samples, rate = _vowel(f0=420.0)
        first = transform_vocal_tract(samples, rate, 0.5,
                                      chipmunk_range=True)
        second = transform_vocal_tract(samples, rate, 0.5,
                                       chipmunk_range=True)
        profile = load_vocal_tract_range()
        self.assertEqual(first.applied_ratio, profile.expanded_min_ratio)
        self.assertTrue(np.array_equal(first.samples, second.samples))
        self.assertTrue(np.all(np.isfinite(first.samples)))

    def test_phone_protection_is_conservative(self):
        self.assertEqual(phone_warp_strength("pau", 1.0), 0.0)
        self.assertGreater(phone_warp_strength("e", 0.8), 0.9)
        self.assertLess(phone_warp_strength("s", 0.0), 0.2)
        self.assertLess(phone_warp_strength("k", 0.0), 0.2)


if __name__ == "__main__":
    unittest.main()
