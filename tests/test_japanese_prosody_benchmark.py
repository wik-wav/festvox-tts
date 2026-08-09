import math
import unittest

from japanese_prosody_benchmark import (
    AlignedPlanPhone,
    filter_pitch_observation_outliers,
    final_plan_phone_timings,
    normalize_kokoro_frontend_text,
    summarize_duration_rows,
)
from japanese_frontend import analyze_japanese
from japanese_synthesis import create_synthesis_plan


def _timing(
    utterance_id,
    index,
    reference_duration,
    predicted_duration,
):
    reference_start = index * reference_duration
    predicted_start = index * predicted_duration
    return AlignedPlanPhone(
        utterance_id=utterance_id,
        partition="test",
        reference_index=index,
        plan_index=index,
        phone="a",
        phone_class="vowel",
        mora_index=index,
        phrase_index=0,
        reference_start_seconds=reference_start,
        reference_end_seconds=reference_start + reference_duration,
        predicted_start_seconds=predicted_start,
        predicted_end_seconds=predicted_start + predicted_duration,
        reference_duration_seconds=reference_duration,
        predicted_duration_seconds=predicted_duration,
        signed_duration_error_seconds=(
            predicted_duration - reference_duration
        ),
        absolute_duration_error_seconds=abs(
            predicted_duration - reference_duration
        ),
        duration_ratio=predicted_duration / reference_duration,
        boundary_drift_seconds=(
            predicted_start + predicted_duration
            - reference_start - reference_duration
        ),
        reference_confidence=0.9,
        reference_rejected=False,
        phenomena=(),
    )


class JapaneseProsodyBenchmarkTests(unittest.TestCase):
    def test_kokoro_token_spacing_is_not_spoken_as_pauses(self):
        self.assertEqual(
            normalize_kokoro_frontend_text("一 時間 目 。"),
            "一時間目。",
        )
        self.assertEqual(
            normalize_kokoro_frontend_text("これ\tは\nテスト？"),
            "これはテスト？",
        )

    def test_rate_normalized_error_removes_utterance_scale(self):
        rows = (
            _timing("a", 0, 0.050, 0.060),
            _timing("a", 1, 0.100, 0.120),
            _timing("b", 0, 0.060, 0.048),
            _timing("b", 1, 0.090, 0.072),
        )
        summary = summarize_duration_rows(rows)["all_accepted"]
        self.assertGreater(summary["mean_absolute_error_ms"], 0.0)
        self.assertAlmostEqual(
            summary["rate_normalized_log_rmse"], 0.0, places=12)

    def test_pitch_octave_outlier_is_preserved_as_rejection(self):
        rows = [
            {"mora_index": 0, "observed_f0_hz": 120.0},
            {"mora_index": 1, "observed_f0_hz": 123.0},
            {"mora_index": 2, "observed_f0_hz": 125.0},
            {"mora_index": 3, "observed_f0_hz": 246.0},
        ]
        accepted, rejected = filter_pitch_observation_outliers(
            rows, utterance_id="sample", partition="test")
        self.assertEqual([row["mora_index"] for row in accepted], [0, 1, 2])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["mora_index"], 3)
        self.assertEqual(
            rejected[0]["reason"],
            "probable_octave_or_pulse_rate_error",
        )
        self.assertGreater(
            abs(rejected[0]["offset_from_utterance_median_semitones"]),
            rejected[0]["robust_outlier_limit_semitones"],
        )

    def test_pitch_filter_is_deterministic(self):
        rows = [
            {"mora_index": index, "observed_f0_hz": value}
            for index, value in enumerate((130.0, 132.0, 128.0, 260.0))
        ]
        first = filter_pitch_observation_outliers(
            rows, utterance_id="sample", partition="validation")
        second = filter_pitch_observation_outliers(
            rows, utterance_id="sample", partition="validation")
        self.assertEqual(first, second)
        self.assertTrue(all(math.isfinite(float(row["observed_f0_hz"]))
                            for row in first[0]))

    def test_final_plan_rows_preserve_timing_decision_trace(self):
        utterance = analyze_japanese("kya", mode="kana")
        plan = create_synthesis_plan(utterance)
        rows = final_plan_phone_timings(utterance, plan)

        onset = next(row for row in rows if row.phone == "ky")
        self.assertEqual(onset.phone_class, "stop")
        self.assertEqual(len(onset.timing_decisions), 1)
        trace = onset.timing_decisions[0]
        self.assertEqual(trace["duration_model"], "contextual")
        self.assertEqual(trace["duration_model_id"], plan.duration_model_id)
        self.assertEqual(trace["rendered_phone"], "ky")
        self.assertIn("source_safe_min", trace)


if __name__ == "__main__":
    unittest.main()
