import json
import math
import unittest

import numpy as np

from diphone_loudness import (
    _biquad_coefficients,
    analyze_rendered_joins,
    loudness_curve,
)


def _sine(rate: int, duration: float, amplitude: float) -> np.ndarray:
    times = np.arange(int(round(rate * duration)), dtype=np.float64) / rate
    return amplitude * np.sin(2.0 * math.pi * 1000.0 * times + 0.23)


def _fixture(amplitude_after: float):
    rate = 48000
    samples = _sine(rate, 0.8, 0.05)
    samples[int(0.4 * rate):] *= amplitude_after / 0.05
    segments = [
        {"phone": "a", "start": 0.0, "end": 0.2},
        {"phone": "b", "start": 0.2, "end": 0.6},
        {"phone": "c", "start": 0.6, "end": 0.8},
    ]
    alternatives = {
        "a-b": [{
            "left_name": "a_u7", "alias": "a b incoming",
            "wav": "incoming.wav",
            "join_conditioning": {"effective_end_collar_ms": 15.0},
        }],
        "b-c": [{
            "left_name": "b_u2", "alias": "b c outgoing",
            "wav": "outgoing.wav",
            "oto_timing_ms": {"overlap": 18.0},
            "join_conditioning": {"effective_start_collar_ms": 12.0},
        }],
    }
    return rate, samples, segments, alternatives


class DiphoneLoudnessTests(unittest.TestCase):
    def test_k_weighting_uses_bs1770_reference_coefficients_at_48k(self):
        shelf_b, shelf_a = _biquad_coefficients("shelf", 48000)
        high_b, high_a = _biquad_coefficients("highpass", 48000)
        np.testing.assert_allclose(
            shelf_b,
            [1.53512485958697, -2.69169618940638, 1.19839281085285],
            rtol=0.0, atol=2e-8,
        )
        np.testing.assert_allclose(
            shelf_a,
            [1.0, -1.69065929318241, 0.73248077421585],
            rtol=0.0, atol=2e-8,
        )
        np.testing.assert_allclose(
            high_b / high_b[0], [1.0, -2.0, 1.0],
            rtol=0.0, atol=1e-12,
        )
        np.testing.assert_allclose(
            high_a,
            [1.0, -1.99004745483398, 0.99007225036621],
            rtol=0.0, atol=2e-8,
        )

    def test_loudness_curve_tracks_known_amplitude_ratio(self):
        quiet = loudness_curve(_sine(48000, 0.5, 0.1), 48000)
        loud = loudness_curve(_sine(48000, 0.5, 0.2), 48000)
        quiet_levels = np.asarray(quiet["levels_lkfs"])[20:-20]
        loud_levels = np.asarray(loud["levels_lkfs"])[20:-20]
        self.assertAlmostEqual(
            float(np.median(loud_levels) - np.median(quiet_levels)),
            20.0 * math.log10(2.0), places=3,
        )

    def test_center_step_is_flagged_and_join_provenance_is_retained(self):
        rate, samples, segments, alternatives = _fixture(0.30)
        result = analyze_rendered_joins(
            samples, rate, segments,
            selected_units={0: "a_u7", 1: "b_u2"},
            alternatives=alternatives,
        )

        self.assertEqual(result["summary"]["join_count"], 1)
        self.assertEqual(result["summary"]["flagged_join_count"], 1)
        join = result["joins"][0]
        self.assertEqual(join["phone"], "b")
        self.assertTrue(join["flagged"])
        self.assertGreater(join["absolute_step_lu"], 14.0)
        self.assertEqual(join["incoming_wav"], "incoming.wav")
        self.assertEqual(join["outgoing_wav"], "outgoing.wav")
        self.assertEqual(join["incoming_collar_ms"], 15.0)
        self.assertEqual(join["outgoing_collar_ms"], 12.0)
        self.assertEqual(join["declared_oto_overlap_ms"], 18.0)
        self.assertEqual(join["overlap_start"], 0.385)
        self.assertEqual(join["overlap_end"], 0.412)
        self.assertEqual(result["units"][0]["start"], 0.1)
        self.assertEqual(result["units"][0]["end"], 0.4)
        self.assertEqual(result["units"][1]["start"], 0.4)
        self.assertEqual(result["units"][1]["end"], 0.7)
        self.assertEqual(result["momentary_curve"]["window_ms"], 400.0)
        json.dumps(result, sort_keys=True)

    def test_equal_level_handoff_is_not_flagged(self):
        rate, samples, segments, alternatives = _fixture(0.05)
        result = analyze_rendered_joins(
            samples, rate, segments,
            selected_units={0: "a_u7", 1: "b_u2"},
            alternatives=alternatives,
        )
        self.assertEqual(result["summary"]["flagged_join_count"], 0)
        self.assertLess(result["joins"][0]["absolute_step_lu"], 0.1)


if __name__ == "__main__":
    unittest.main()
