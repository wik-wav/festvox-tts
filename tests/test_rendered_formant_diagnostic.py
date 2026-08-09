import json
import math
import unittest

import numpy as np

from rendered_formant_diagnostic import analyze_rendered_formants


def _resonant_vowel(sample_rate=16000, seconds=0.8, f0=170.0):
    count = int(round(sample_rate * seconds))
    source = np.zeros(count, np.float64)
    source[::max(1, int(round(sample_rate / f0)))] = 1.0
    values = source
    for formant, bandwidth in ((520.0, 85.0), (1850.0, 120.0),
                               (2850.0, 170.0), (3700.0, 210.0)):
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
    values /= max(1.0e-9, float(np.max(np.abs(values))))
    return np.asarray(values * 0.35, np.float32), sample_rate


class RenderedFormantDiagnosticTests(unittest.TestCase):
    def test_final_waveform_is_read_only_and_exact_join_evidence_is_ranked(self):
        samples, sample_rate = _resonant_vowel()
        before = samples.copy()
        segments = [
            {"phone": "e", "start": 0.0, "end": 0.4},
            {"phone": "e", "start": 0.4, "end": 0.8},
        ]
        join_diagnostic = {"joins": [{
            "time": 0.4,
            "segment_index": 1,
            "left_phone": "e",
            "right_phone": "e",
            "formants_available": True,
            "formant_frequency_jump_normalized": 0.24,
            "formant_frequency_jump_novelty": 5.0,
            "formant_tracks": [{"formant": 2}],
        }]}
        report = analyze_rendered_formants(
            samples, sample_rate, segments,
            join_diagnostic=join_diagnostic,
        )
        np.testing.assert_array_equal(samples, before)
        self.assertEqual(report["kind"], "rendered_formant_diagnostic")
        self.assertGreater(report["accepted_frame_count"], 0)
        self.assertEqual(report["potential_jump_count"], 1)
        jump = report["jumps"][0]
        self.assertEqual(jump["kind"], "EXACT_SPLICE_FORMANT_JUMP")
        self.assertTrue(jump["exact_splice_evidence"])
        self.assertEqual(jump["rank"], 1)
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("rendered-result.wav", serialized)

    def test_pause_and_short_phone_are_explicitly_unanalyzed(self):
        samples, sample_rate = _resonant_vowel(seconds=0.25)
        report = analyze_rendered_formants(
            samples, sample_rate,
            [
                {"phone": "pau", "start": 0.0, "end": 0.1},
                {"phone": "k", "start": 0.1, "end": 0.12},
                {"phone": "e", "start": 0.12, "end": 0.25},
            ],
        )
        self.assertEqual(
            report["phones"][0]["rejection_reasons"],
            ["pause_or_silence"],
        )
        self.assertEqual(
            report["phones"][1]["rejection_reasons"],
            ["phone_too_short_for_contained_window"],
        )
        self.assertTrue(report["phones"][2]["analyzed"])


if __name__ == "__main__":
    unittest.main()
