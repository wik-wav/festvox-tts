"""Deterministic acceptance tests for Asaxi acoustic pitch realization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

import asaxi_pitch as ap


@dataclass(frozen=True)
class _Plan:
    boundary_tone: str = "L%"
    interrogative: bool = False
    directive: bool = False


@dataclass(frozen=True)
class _Mora:
    index: int
    phrase_index: int
    local_mora_index: int
    word_index: int
    word: str
    text: str
    pitch: str
    lexical_pitch: str
    accentable: bool
    start: float
    end: float


def _phrase(
    phrase_index: int,
    tones: str,
    *,
    start: float = 0.0,
    duration: float = 0.12,
    first_index: int = 0,
) -> tuple[_Mora, ...]:
    rows = []
    for local_index, tone in enumerate(tones):
        mora_start = start + local_index * duration
        rows.append(_Mora(
            index=first_index + local_index,
            phrase_index=phrase_index,
            local_mora_index=local_index,
            word_index=local_index,
            word=f"word-{local_index}",
            text=f"mora-{local_index}",
            pitch=tone,
            lexical_pitch=tone,
            accentable=True,
            start=mora_start,
            end=mora_start + duration,
        ))
    return tuple(rows)


class AsaxiPitchTests(unittest.TestCase):
    def test_default_model_is_versioned_and_immutable(self) -> None:
        model = ap.load_asaxi_pitch_model()
        self.assertEqual(model.model_id, "asaxi-hierarchical-log-f0-v1")
        self.assertEqual(model.model_version, 1)
        self.assertGreater(model.tone_goal("H"), model.tone_goal("L"))
        with self.assertRaises(TypeError):
            model.tone_goals_semitones["H"] = 99.0

    def test_profile_rejects_cumulative_frequency_drift(self) -> None:
        data = json.loads(ap.DEFAULT_MODEL_PATH.read_text(encoding="utf-8"))
        data["phrase_model"][
            "utterance_declination_semitones_per_second"
        ] = -0.2
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "drifting.json"
            source.write_text(
                json.dumps(data, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "frequency drift"):
                ap.load_asaxi_pitch_model(source)

    def test_trace_is_deterministic_and_json_serializable(self) -> None:
        plans = (_Plan("L%"),)
        moras = _phrase(0, "HLHL")
        first = ap.realize_pitch(plans, moras, base_pitch_hz=165.0)
        second = ap.realize_pitch(plans, moras, base_pitch_hz=165.0)

        self.assertEqual(first.targets, second.targets)
        self.assertEqual(
            json.dumps(first.trace_dict(), sort_keys=True),
            json.dumps(second.trace_dict(), sort_keys=True),
        )
        self.assertEqual(
            first.trace["cumulative_frequency_drift"], "disabled")
        self.assertTrue(first.trace["trajectory"])

    def test_later_phrase_shape_changes_without_register_drift(self) -> None:
        first = _phrase(0, "HLH", duration=0.11)
        second = _phrase(
            1,
            "HLH",
            start=0.58,
            duration=0.11,
            first_index=len(first),
        )
        result = ap.realize_pitch(
            (_Plan("H-"), _Plan("L%")),
            first + second,
            base_pitch_hz=160.0,
        )
        rows = result.trace["mora_goals"]
        first_goals = [
            row["automatic_goal_semitones"]
            for row in rows if row["phrase_index"] == 0
        ]
        second_goals = [
            row["automatic_goal_semitones"]
            for row in rows if row["phrase_index"] == 1
        ]

        self.assertNotEqual(first_goals, second_goals)
        self.assertAlmostEqual(
            sum(first_goals) / len(first_goals),
            sum(second_goals) / len(second_goals),
            places=6,
        )
        states = result.trace["phrase_states"]
        self.assertEqual(states[1]["previous_boundary_tone"], "H-")
        self.assertGreater(states[1]["reset_strength"], 0.0)
        self.assertLess(states[1]["reset_strength"], 1.0)

    def test_question_boundary_is_a_timed_low_high_region(self) -> None:
        result = ap.realize_pitch(
            (_Plan("LH%", interrogative=True),),
            _phrase(0, "LLL", duration=0.11),
            base_pitch_hz=160.0,
        )
        events = result.trace["boundary_events"]
        kinds = [row["kind"] for row in events]
        self.assertEqual(
            kinds,
            [
                "boundary_region_start",
                "boundary_low",
                "boundary_high",
                "boundary_high_hold",
            ],
        )
        times = [row["time_seconds"] for row in events]
        self.assertEqual(times, sorted(times))
        self.assertGreater(result.targets[-1][1], 160.0)

    def test_pause_gap_has_latent_state_but_no_render_targets(self) -> None:
        first = _phrase(0, "HL", duration=0.1)
        first_end = first[-1].end
        second_start = 0.65
        second = _phrase(
            1,
            "HL",
            start=second_start,
            duration=0.1,
            first_index=len(first),
        )
        result = ap.realize_pitch(
            (_Plan("H-"), _Plan("L%")), first + second)

        self.assertFalse(any(
            first_end < time < second_start
            for time, _frequency in result.targets
        ))
        self.assertNotEqual(
            result.trace["phrase_states"][1]["carry_in_semitones"],
            result.trace["phrase_states"][1][
                "state_after_reset_semitones"],
        )

    def test_short_moras_produce_more_target_undershoot(self) -> None:
        tones = "HLHLHL"
        slow = ap.realize_pitch(
            (_Plan("H-"),),
            _phrase(0, tones, duration=0.14),
        )
        fast = ap.realize_pitch(
            (_Plan("H-"),),
            _phrase(0, tones, duration=0.05),
        )

        def mean_error(result) -> float:
            rows = [
                row for row in result.trace["trajectory"]
                if row["time_seconds"] < result.targets[-1][0] * 0.65
            ]
            return sum(
                abs(
                    row["desired_semitones"]
                    - row["automatic_realized_semitones"]
                )
                for row in rows
            ) / len(rows)

        self.assertGreater(mean_error(fast), mean_error(slow))

    def test_manual_cents_overlay_is_local_and_exact_at_mora_target(
        self,
    ) -> None:
        moras = _phrase(0, "HLH", duration=0.12)
        baseline = ap.realize_pitch((_Plan("L%"),), moras)
        edited = ap.realize_pitch(
            (_Plan("L%"),),
            moras,
            mora_pitch_offsets_cents={1: 1200},
        )
        center = round(
            moras[1].start + (moras[1].end - moras[1].start) * 0.58,
            6,
        )
        untouched = round(
            moras[0].start + (moras[0].end - moras[0].start) * 0.58,
            6,
        )

        self.assertAlmostEqual(
            dict(edited.targets)[center],
            dict(baseline.targets)[center] * 2.0,
            places=2,
        )
        self.assertEqual(
            dict(edited.targets)[untouched],
            dict(baseline.targets)[untouched],
        )


if __name__ == "__main__":
    unittest.main()
