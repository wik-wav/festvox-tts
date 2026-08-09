import unittest

import numpy as np

from japanese_phrase_edges import (
    compare_phrase_edges,
    detect_acoustic_phrase_edge,
)


def _tone(start, end, *, duration=1.0, sample_rate=16000):
    times = np.arange(int(duration * sample_rate)) / sample_rate
    values = np.zeros(times.size, dtype=np.float64)
    active = (times >= start) & (times < end)
    values[active] = 0.22 * np.sin(2.0 * np.pi * 200.0 * times[active])
    return values, sample_rate


class PhraseEdgeAcousticTests(unittest.TestCase):
    def test_initial_edge_reports_early_activity_as_positive_extension(self):
        samples, sample_rate = _tone(0.480, 0.750)
        result = detect_acoustic_phrase_edge(
            samples, sample_rate, 0.500, edge="initial")
        self.assertTrue(result.available)
        self.assertAlmostEqual(result.extension_ms, 20.0, delta=8.0)

    def test_final_edge_reports_late_activity_as_positive_extension(self):
        samples, sample_rate = _tone(0.250, 0.532)
        result = detect_acoustic_phrase_edge(
            samples, sample_rate, 0.500, edge="final")
        self.assertTrue(result.available)
        self.assertAlmostEqual(result.extension_ms, 32.0, delta=8.0)

    def test_delayed_onset_is_negative_extension(self):
        samples, sample_rate = _tone(0.520, 0.750)
        result = detect_acoustic_phrase_edge(
            samples, sample_rate, 0.500, edge="initial")
        self.assertTrue(result.available)
        self.assertAlmostEqual(result.extension_ms, -20.0, delta=8.0)

    def test_silence_is_unavailable_instead_of_zero(self):
        result = detect_acoustic_phrase_edge(
            np.zeros(16000), 16000, 0.5, edge="initial")
        self.assertFalse(result.available)
        self.assertIsNone(result.extension_ms)
        self.assertEqual(result.reason, "insufficient_speech_pause_contrast")

    def test_source_relative_comparison_removes_shared_detector_bias(self):
        source_samples, sample_rate = _tone(0.490, 0.750)
        synth_samples, _ = _tone(0.470, 0.750)
        source = detect_acoustic_phrase_edge(
            source_samples, sample_rate, 0.500, edge="initial")
        synthesis = detect_acoustic_phrase_edge(
            synth_samples, sample_rate, 0.500, edge="initial")
        result = compare_phrase_edges(source, synthesis)
        self.assertAlmostEqual(
            result["synthesis_excess_extension_ms"], 20.0, delta=6.0)


if __name__ == "__main__":
    unittest.main()
