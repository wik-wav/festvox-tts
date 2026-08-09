from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from japanese_alignment_verification import (
    PhraseAlignment,
    analyze_phrase_edge_acoustics,
    canonical_phone_sequences,
    render_alignment_plot,
    select_alignment_candidates,
    split_phrase_alignments,
    summarize_phrase_alignment,
    summarize_edge_acoustics,
    waveform_envelope,
)
from japanese_prosody_benchmark import AlignedPlanPhone
from kokoro_reference import SilverPhoneAlignment, SilverUtteranceAlignment


def _row(phone, index, reference_start, reference_end,
         predicted_start, predicted_end, phrase_index=0):
    reference_duration = reference_end - reference_start
    predicted_duration = predicted_end - predicted_start
    return AlignedPlanPhone(
        utterance_id="fixture",
        partition="test",
        reference_index=index,
        plan_index=index,
        phone=phone,
        phone_class="vowel",
        mora_index=index,
        phrase_index=phrase_index,
        reference_start_seconds=reference_start,
        reference_end_seconds=reference_end,
        predicted_start_seconds=predicted_start,
        predicted_end_seconds=predicted_end,
        reference_duration_seconds=reference_duration,
        predicted_duration_seconds=predicted_duration,
        signed_duration_error_seconds=(predicted_duration -
                                       reference_duration),
        absolute_duration_error_seconds=abs(predicted_duration -
                                            reference_duration),
        duration_ratio=predicted_duration / reference_duration,
        boundary_drift_seconds=predicted_end - reference_end,
        reference_confidence=0.9,
        reference_rejected=False,
        phenomena=(),
    )


def _phone(index, raw, phone, start, end, phrase_index):
    return SilverPhoneAlignment(
        index=index,
        raw_phone=raw,
        phone=phone,
        start_seconds=start,
        end_seconds=end,
        confidence=0.9,
        boundary_confidence_left=0.9,
        boundary_confidence_right=0.9,
        mora_index=index,
        phrase_index=phrase_index,
    )


def _two_phrase_alignment():
    phones = (
        _phone(0, "k", "k", 0.0, 0.1, 0),
        _phone(1, "a", "a", 0.1, 0.3, 0),
        _phone(2, ".", "pau", 0.3, 0.8, 0),
        _phone(3, "n", "n", 0.8, 0.9, 1),
        _phone(4, "o", "o", 0.9, 1.1, 1),
    )
    return SilverUtteranceAlignment(
        utterance_id="fixture",
        sample_rate=8000,
        sample_count=8800,
        phones=phones,
        confidence=0.9,
        accepted=True,
    )


class JapaneseAlignmentVerificationTests(unittest.TestCase):
    def test_selection_is_best_median_worst_per_partition(self):
        rows = []
        for partition in ("test", "validation"):
            rows.extend({
                "utterance_id": f"{partition}-{index}",
                "partition": partition,
                "median_absolute_boundary_drift_ms": float(index),
                "match_fraction": 1.0,
            } for index in range(5))

        selected = select_alignment_candidates(rows, per_partition=3)

        self.assertEqual(
            [(row["partition"], row["utterance_id"],
              row["selection_tier"]) for row in selected],
            [
                ("test", "test-0", "best"),
                ("test", "test-2", "median"),
                ("test", "test-4", "worst"),
                ("validation", "validation-0", "best"),
                ("validation", "validation-2", "median"),
                ("validation", "validation-4", "worst"),
            ],
        )

    def test_selection_excludes_examples_over_phrase_limit(self):
        rows = [
            {
                "utterance_id": f"test-{index}",
                "partition": "test",
                "median_absolute_boundary_drift_ms": float(index),
                "match_fraction": 1.0,
                "phrase_count": 3 if index == 2 else 1 + (index % 2),
            }
            for index in range(7)
        ]

        selected = select_alignment_candidates(
            rows, per_partition=5, max_phrases=2)

        self.assertEqual(len(selected), 5)
        self.assertNotIn("test-2", {
            row["utterance_id"] for row in selected})
        self.assertTrue(all(int(row["phrase_count"]) <= 2
                            for row in selected))

    def test_waveform_envelope_preserves_single_sample_transient(self):
        samples = np.zeros(8000, dtype=np.float64)
        samples[4011] = 1.0
        _times, low, high = waveform_envelope(
            samples, 8000, origin_seconds=0.0,
            start_seconds=0.0, end_seconds=1.0, columns=200)
        self.assertEqual(float(np.max(high)), 1.0)
        self.assertEqual(float(np.min(low)), 0.0)

    def test_plot_contains_both_real_waveforms(self):
        sample_rate = 8000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        source = 0.7 * np.sin(2.0 * np.pi * 170.0 * time)
        synthesis = 0.55 * np.sin(2.0 * np.pi * 170.0 * time + 0.2)
        rows = (
            _row("k", 0, 0.0, 0.1, 0.0, 0.12),
            _row("a", 1, 0.1, 0.3, 0.12, 0.34),
            _row("N", 2, 0.3, 0.42, 0.34, 0.46),
        )
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "alignment.svg"
            render_alignment_plot(
                output,
                source_samples=source,
                source_sample_rate=sample_rate,
                synthesis_samples=synthesis,
                synthesis_sample_rate=sample_rate,
                source_origin_seconds=0.1,
                synthesis_origin_seconds=0.1,
                aligned_phones=rows,
                title="Synthetic alignment",
                subtitle="source and synthesis",
                annotation_lines=(
                    "Text: 関係ないです。",
                    "Linguistic phones: exact canonical match",
                ),
                fit_to_rows=True,
            )
            data = output.read_text(encoding="utf-8")
            self.assertTrue(data.startswith("<?xml"))
            self.assertIn("Target waveform", data)
            self.assertIn("Target phones", data)
            self.assertIn("Synthesized phones", data)
            self.assertIn("Synthesized waveform", data)
            self.assertIn("Phone duration delta", data)
            self.assertIn("+20 ms", data)
            self.assertIn("each red connector joins", data)
            self.assertIn("関係ないです。", data)
            self.assertIn("exact canonical match", data)
            self.assertIn("<path", data)
            self.assertGreater(len(data), 20_000)

    def test_canonical_sequence_comparison_is_strict(self):
        alignment = _two_phrase_alignment()
        exact = tuple(SimpleNamespace(phone=phone)
                      for phone in ("k", "a", "n", "o"))
        mismatch = tuple(SimpleNamespace(phone=phone)
                         for phone in ("k", "a", "m", "o"))

        source, synthesis = canonical_phone_sequences(alignment, exact)
        _source_again, wrong = canonical_phone_sequences(alignment, mismatch)

        self.assertEqual(source, ("k", "a", "n", "o"))
        self.assertEqual(source, synthesis)
        self.assertNotEqual(source, wrong)

    def test_phrase_local_alignment_separates_pause_from_active_timing(self):
        alignment = _two_phrase_alignment()
        rows = (
            _row("k", 0, 0.0, 0.1, 0.0, 0.11, phrase_index=0),
            _row("a", 1, 0.1, 0.3, 0.11, 0.32, phrase_index=0),
            _row("n", 2, 0.8, 0.9, 0.60, 0.70, phrase_index=1),
            _row("o", 3, 0.9, 1.1, 0.70, 0.91, phrase_index=1),
        )

        phrases = split_phrase_alignments(rows, alignment)
        summary = summarize_phrase_alignment(phrases)

        self.assertEqual(len(phrases), 2)
        self.assertAlmostEqual(
            phrases[1].localized_rows()[0].reference_start_seconds, 0.0)
        self.assertAlmostEqual(
            phrases[1].localized_rows()[0].predicted_start_seconds, 0.0)
        self.assertAlmostEqual(
            summary["source_interphrase_pause_seconds"], 0.5)
        self.assertAlmostEqual(
            summary["synthesis_interphrase_pause_seconds"], 0.28)
        self.assertAlmostEqual(summary["interphrase_pause_error_ms"], -220.0)
        self.assertAlmostEqual(summary["source_active_phrase_seconds"], 0.6)
        self.assertAlmostEqual(
            summary["synthesis_active_phrase_seconds"], 0.63)

    def test_phrase_edge_audit_compares_rendered_bleed_to_source(self):
        sample_rate = 16000
        times = np.arange(sample_rate, dtype=np.float64) / sample_rate
        source = np.zeros(times.size)
        synthesis = np.zeros(times.size)
        source_active = (times >= 0.49) & (times < 0.71)
        synth_active = (times >= 0.47) & (times < 0.73)
        source[source_active] = 0.2 * np.sin(
            2.0 * np.pi * 180.0 * times[source_active])
        synthesis[synth_active] = 0.2 * np.sin(
            2.0 * np.pi * 180.0 * times[synth_active])
        row = _row("a", 0, 0.0, 0.2, 0.0, 0.2)
        phrase = PhraseAlignment(0, 0, (row,))

        result = analyze_phrase_edge_acoustics(
            phrases=(phrase,),
            source_samples=source,
            source_sample_rate=sample_rate,
            synthesis_samples=synthesis,
            synthesis_sample_rate=sample_rate,
            source_origin_seconds=0.5,
            synthesis_origin_seconds=0.5,
        )
        summary = summarize_edge_acoustics(result)

        self.assertEqual(result[0]["initial_edge_type"], "vowel_initial")
        self.assertAlmostEqual(
            result[0]["initial"]["synthesis_excess_extension_ms"],
            20.0,
            delta=6.0,
        )
        self.assertAlmostEqual(
            result[0]["final"]["synthesis_excess_extension_ms"],
            20.0,
            delta=6.0,
        )
        self.assertEqual(summary["vowel_initial"]["count"], 1)
        self.assertAlmostEqual(
            summary["vowel_initial"][
                "median_effective_first_mora_excess_ms"],
            20.0,
            delta=7.0,
        )


if __name__ == "__main__":
    unittest.main()
