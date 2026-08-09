import unittest

import numpy as np

from japanese_devoicing import periodicity_score, spectral_envelope_distance
from source_filter_voicing import curve_for_regions, transform_voicing


def _mixed_vowel(seconds=0.7, sample_rate=16000):
    time = np.arange(int(seconds * sample_rate), dtype=np.float64) / sample_rate
    harmonic = sum(
        (0.30 / number) * np.sin(
            2.0 * np.pi * 150.0 * number * time + 0.11 * number
        )
        for number in range(1, 31)
    )
    noise = np.random.default_rng(73).normal(size=time.size)
    noise = np.convolve(noise, [1.0, -0.92], mode="same")
    # A slow vowel-like amplitude trajectory also verifies that the renderer
    # does not flatten the source envelope while changing excitation.
    trajectory = 0.72 + 0.28 * np.sin(np.pi * np.linspace(0.0, 1.0, time.size))
    return np.asarray((0.15 * harmonic + 0.025 * noise) * trajectory,
                      np.float32), sample_rate


class SourceFilterVoicingTests(unittest.TestCase):
    def test_analysis_and_source_curve_are_reconstructive(self):
        samples, sample_rate = _mixed_vowel()
        analysis = transform_voicing(samples, sample_rate)
        rendered = transform_voicing(
            samples, sample_rate, analysis.source_curve
        )

        self.assertEqual(rendered.modified_frame_count, 0)
        self.assertLess(rendered.reconstruction_nrmse, 1e-7)
        np.testing.assert_allclose(rendered.samples, samples, atol=2e-7)

    def test_lower_curve_reduces_periodicity_without_replacing_envelope(self):
        samples, sample_rate = _mixed_vowel()
        rendered = transform_voicing(
            samples, sample_rate, [(0.0, 0.12), (0.7, 0.12)]
        )

        before = periodicity_score(samples, sample_rate)
        after = periodicity_score(rendered.samples, sample_rate)
        self.assertGreater(rendered.modified_frame_count, 20)
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertLess(after, before - 0.10)
        self.assertLess(
            spectral_envelope_distance(samples, rendered.samples), 0.30
        )
        level_step = 20.0 * np.log10(
            (np.sqrt(np.mean(rendered.samples ** 2)) + 1e-9)
            / (np.sqrt(np.mean(samples ** 2)) + 1e-9)
        )
        # The supplied real /e/ pair loses about 15.8 dB at full devoicing.
        # A 0.12 target is intentionally quieter, not level-normalized.
        self.assertGreater(float(level_step), -14.0)
        self.assertLess(float(level_step), -8.0)

    def test_zero_curve_is_stochastic_not_a_growling_residual(self):
        samples, sample_rate = _mixed_vowel()
        rendered = transform_voicing(
            samples, sample_rate, [(0.0, 0.0), (0.7, 0.0)]
        )

        before = periodicity_score(samples, sample_rate)
        after = periodicity_score(rendered.samples, sample_rate)
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertGreater(before, 0.50)
        self.assertLess(after, 0.08)
        self.assertTrue(any(
            frame.stochastic_excitation and frame.applied
            for frame in rendered.frame_diagnostics
        ))
        self.assertTrue(all(np.isfinite(rendered.samples)))
        self.assertLess(
            spectral_envelope_distance(samples, rendered.samples), 0.30
        )
        level_step = 20.0 * np.log10(
            (np.sqrt(np.mean(rendered.samples ** 2)) + 1e-9)
            / (np.sqrt(np.mean(samples ** 2)) + 1e-9)
        )
        self.assertGreater(float(level_step), -16.5)
        self.assertLess(float(level_step), -13.0)

    def test_voicing_level_is_monotonic_between_source_and_noise(self):
        samples, sample_rate = _mixed_vowel()
        source = transform_voicing(samples, sample_rate)
        middle = transform_voicing(
            samples, sample_rate, [(0.0, 0.5), (0.7, 0.5)]
        )
        noise = transform_voicing(
            samples, sample_rate, [(0.0, 0.0), (0.7, 0.0)]
        )

        rms = lambda values: float(np.sqrt(np.mean(
            np.asarray(values, np.float64) ** 2)))
        self.assertLess(rms(noise.samples), rms(middle.samples))
        self.assertLess(rms(middle.samples), rms(source.samples))

    def test_render_is_deterministic_and_contains_no_random_synthesis(self):
        samples, sample_rate = _mixed_vowel()
        curve = [(0.0, 0.20), (0.25, 0.08), (0.70, 0.30)]
        first = transform_voicing(samples, sample_rate, curve)
        second = transform_voicing(samples, sample_rate, curve)

        np.testing.assert_array_equal(first.samples, second.samples)
        self.assertEqual(first.diagnostic_dict(), second.diagnostic_dict())

    def test_silence_is_finite_and_unvoiced(self):
        result = transform_voicing(
            np.zeros(800, np.float32), 16000, [(0.0, 0.0), (0.05, 0.0)]
        )

        self.assertTrue(np.all(np.isfinite(result.samples)))
        self.assertEqual(float(np.max(np.abs(result.samples))), 0.0)
        self.assertTrue(all(value == 0.0 for _time, value in
                            result.source_curve))

    def test_automatic_region_keeps_source_valued_edges(self):
        samples, sample_rate = _mixed_vowel()
        analysis = transform_voicing(samples, sample_rate)
        curve = curve_for_regions(
            analysis.source_curve, [{"start": 0.20, "end": 0.42}],
            target_voicing=0.10,
        )
        source = dict(analysis.source_curve)
        target = dict(curve)
        nearest_start = min(source, key=lambda value: abs(value - 0.20))
        middle = min(source, key=lambda value: abs(value - 0.31))

        self.assertIn(0.20, target)
        self.assertIn(0.42, target)
        self.assertAlmostEqual(
            target[0.20], float(np.interp(
                0.20, list(source), list(source.values()))), places=5
        )
        self.assertLess(target[middle], source[middle])
        self.assertAlmostEqual(
            target[0.42], float(np.interp(
                0.42, list(source), list(source.values()))), places=5
        )

    def test_regions_can_carry_independent_continuous_targets(self):
        source = [(index * .01, .9) for index in range(31)]
        curve = dict(curve_for_regions(source, [
            {"start": .02, "end": .12, "target_voicing": .2},
            {"start": .18, "end": .28, "target_voicing": .6},
        ]))

        self.assertLess(curve[.07], curve[.23])
        self.assertAlmostEqual(curve[.07], .2, places=3)
        self.assertAlmostEqual(curve[.23], .6, places=3)


if __name__ == "__main__":
    unittest.main()
