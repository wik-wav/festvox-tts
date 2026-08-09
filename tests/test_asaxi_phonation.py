# -*- coding: utf-8 -*-

import unittest

import numpy as np

import asaxi_phonation as ap


class Segment:
    def __init__(self, phone, start, end):
        self.phone = phone
        self.start = start
        self.end = end


class Synthesis:
    def __init__(self, segments):
        self.sr = 16000
        duration = segments[-1].end
        times = np.arange(int(round(duration * self.sr))) / self.sr
        self.samples = (
            0.18 * np.sin(2.0 * np.pi * 160.0 * times)
        ).astype(np.float32)
        self.segments = segments
        self.asaxi_prosody = {}
        self.vowel_realizations = []


class AsaxiPhonationTests(unittest.TestCase):
    def test_voiceless_context_predicts_vowel_devoicing(self):
        segments = [
            Segment("sh", 0.00, 0.05),
            Segment("er", 0.05, 0.13),
            Segment("s", 0.13, 0.18),
            Segment("o", 0.18, 0.28),
        ]
        metadata = {"moras": [{
            "mora_index": 0,
            "phrase_index": 0,
            "word": "shěso",
            "text": "shě",
            "phones": ["sh", "er"],
            "segment_indices": [0, 1],
        }, {
            "mora_index": 1,
            "phrase_index": 0,
            "word": "shěso",
            "text": "so",
            "phones": ["s", "o"],
            "segment_indices": [2, 3],
        }]}

        predictions = ap.predict_mora_phonation(metadata, segments)

        self.assertAlmostEqual(
            predictions[0].automatic_voicing,
            ap.AUTOMATIC_DEVOICED_VOICING)
        self.assertEqual(predictions[1].automatic_voicing, 1.0)
        self.assertIn(
            "vowel between voiceless sh and s",
            predictions[0].reasons)

    def test_aspiration_and_breathy_interjection_are_distinct(self):
        aspiration_segments = [
            Segment("t", 0.00, 0.04),
            Segment("hh", 0.04, 0.07),
            Segment("a", 0.07, 0.15),
            Segment("y", 0.15, 0.18),
        ]
        aspiration = ap.predict_mora_phonation(
            {"moras": [{
                "mora_index": 0,
                "phrase_index": 0,
                "word": "txă",
                "text": "txă",
                "phones": ["t", "hh", "a", "y"],
                "segment_indices": [0, 1, 2, 3],
            }]},
            aspiration_segments,
        )[0]
        sigh = ap.predict_mora_phonation(
            {"moras": [{
                "mora_index": 0,
                "phrase_index": 0,
                "word": "ox",
                "text": "ox",
                "phones": ["o", "hh"],
                "segment_indices": [0, 1],
            }]},
            [Segment("o", 0.0, 0.10), Segment("hh", 0.10, 0.14)],
        )[0]

        self.assertEqual(aspiration.automatic_breathiness, 0.55)
        self.assertEqual(sigh.automatic_breathiness, 0.72)
        self.assertGreater(aspiration.final_voicing, 0.5)
        self.assertGreater(sigh.final_voicing, 0.5)

    def test_manual_overrides_are_final_at_mora_level(self):
        segments = [
            Segment("sh", 0.00, 0.05),
            Segment("er", 0.05, 0.13),
            Segment("s", 0.13, 0.18),
        ]
        metadata = {"moras": [{
            "mora_index": 0,
            "phrase_index": 0,
            "word": "shěso",
            "text": "shě",
            "phones": ["sh", "er"],
            "segment_indices": [0, 1],
        }]}

        prediction = ap.predict_mora_phonation(
            metadata,
            segments,
            voicing_overrides={0: 0.8},
            breathiness_overrides={0: 0.25},
        )[0]

        self.assertTrue(prediction.voicing_overridden)
        self.assertTrue(prediction.breathiness_overridden)
        self.assertAlmostEqual(prediction.final_voicing, 0.8)

    def test_block_override_updates_preview_curve_without_audio_analysis(self):
        segments = [
            Segment("sh", 0.00, 0.05),
            Segment("er", 0.05, 0.13),
            Segment("s", 0.13, 0.18),
        ]
        metadata = {"moras": [{
            "mora_index": 0,
            "phrase_index": 0,
            "word": "shěso",
            "text": "shě",
            "phones": ["sh", "er"],
            "segment_indices": [0, 1],
        }]}
        source = [(index / 1000.0, 0.95) for index in range(181)]

        automatic, _ = ap.mora_voicing_curve(
            metadata, segments, source)
        manual, predictions = ap.mora_voicing_curve(
            metadata, segments, source, voicing_overrides={0: 0.6})

        self.assertLess(min(value for _time, value in automatic), 0.3)
        self.assertGreater(min(value for _time, value in manual), 0.55)
        self.assertAlmostEqual(predictions[0].final_voicing, 0.6)

    def test_render_keeps_structured_predictions_and_finite_audio(self):
        segments = [
            Segment("sh", 0.00, 0.05),
            Segment("er", 0.05, 0.13),
            Segment("s", 0.13, 0.18),
        ]
        metadata = {"moras": [{
            "mora_index": 0,
            "phrase_index": 0,
            "word": "shěso",
            "text": "shě",
            "phones": ["sh", "er"],
            "segment_indices": [0, 1],
        }]}
        synthesis = Synthesis(segments)

        ap.apply_phonation(synthesis, metadata)

        self.assertTrue(np.all(np.isfinite(synthesis.samples)))
        self.assertEqual(
            synthesis.asaxi_prosody[
                "mora_phonation_predictions"][0]["mora_index"],
            0)
        self.assertTrue(synthesis.generated_voicing_targets)
        self.assertEqual(synthesis.voicing_override, [])
        self.assertEqual(synthesis.voicing_mode, "")


if __name__ == "__main__":
    unittest.main()
