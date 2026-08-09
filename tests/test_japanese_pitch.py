import unittest
import statistics

import pitch_domain as pd
from japanese_frontend import analyze_japanese
from japanese_pitch import load_japanese_pitch_model, mora_pitch_contour


class JapanesePitchModelTests(unittest.TestCase):
    def test_profile_is_versioned_and_kokoro_calibrated(self):
        model = load_japanese_pitch_model()
        self.assertEqual(model.model_version, 5)
        self.assertEqual(
            model.model_id,
            "japanese_speaker_relative_log_f0_kokoro_b453f6caf042_v5")
        self.assertEqual(
            model.analysis_evidence["accepted_kokoro_phrase_count"], 106)
        self.assertGreater(
            model.analysis_evidence[
                "kokoro_phrase_range_p10_p90_semitones_median"], 7.0)

    def test_components_and_headroom_are_semitones(self):
        model = load_japanese_pitch_model()
        self.assertGreater(model.component("accent_nucleus"), 0.0)
        self.assertLess(model.component("post_accent_drop"), 0.0)
        baseline = 165.0
        low = pd.semitone_offset(
            baseline, -model.headroom_below_semitones)
        high = pd.semitone_offset(
            baseline, model.headroom_above_semitones)
        self.assertLess(low, 135.0)
        self.assertGreater(high, 180.0)
        with self.assertRaises(TypeError):
            model.components["accent_nucleus"] = 999.0
        self.assertNotEqual(
            load_japanese_pitch_model().components["accent_nucleus"], 999.0)

    def test_repeated_phrases_change_shape_without_frequency_drift(self):
        utterance = analyze_japanese(
            "kore wa tesuto desu. kore wa tesuto desu.", mode="kana")
        times = {mora.index: mora.index * 0.12 for mora in utterance.moras}
        contour = mora_pitch_contour(
            utterance, mora_times_seconds=times)
        phrases = {}
        for target in contour:
            phrases.setdefault(target.phrase_index, []).append(target)
        self.assertEqual(sorted(phrases), [0, 1])
        first = phrases[0]
        second = phrases[1]
        self.assertEqual(len(first), len(second))
        differences = [
            left.semitones_from_baseline - right.semitones_from_baseline
            for left, right in zip(first, second)
        ]
        # A constant offset would be frequency drift, not contour variation.
        # The shape changes while the phrase-average register stays fixed.
        self.assertAlmostEqual(statistics.mean(differences), 0.0, places=7)
        self.assertGreater(max(differences) - min(differences), 0.8)
        self.assertGreater(statistics.pstdev(differences), 0.2)
        self.assertTrue(all(
            "later_phrase_shape" in target.kind for target in second))

    def test_phrase_and_contextual_shape_are_separate_components(self):
        model = load_japanese_pitch_model()
        self.assertAlmostEqual(
            model.component("phrase_declination_total"), -1.46)
        self.assertAlmostEqual(
            model.later_phrase_accent_contrast_scale, -0.13)
        utterance = analyze_japanese(
            "kore wa tesuto desu. kore wa tesuto desu.", mode="kana")
        times = {mora.index: mora.index * 0.12
                 for mora in utterance.moras}
        contour = mora_pitch_contour(
            utterance, model=model, mora_times_seconds=times)
        self.assertTrue(all(
            "phrase_declination" in target.components_semitones
            and "later_phrase_declination_shape" in target.components_semitones
            and "later_phrase_accent_shape" in target.components_semitones
            and "later_phrase_boundary_shape" in target.components_semitones
            and "utterance_declination" not in target.components_semitones
            for target in contour))


if __name__ == "__main__":
    unittest.main()
