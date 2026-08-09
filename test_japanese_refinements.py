from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import japanese_frontend
import japanese_refinements as jr
from japanese_synthesis import (
    JapaneseF0Target,
    JapanesePlannedSegment,
    JapaneseSynthesisPlan,
)


def _plan(unit_overrides=None) -> JapaneseSynthesisPlan:
    segments = (
        JapanesePlannedSegment(0, "pau", 0.05, pause_role="leading"),
        JapanesePlannedSegment(1, "a", 0.10, mora_index=0),
        JapanesePlannedSegment(2, "k", 0.06, mora_index=1),
        JapanesePlannedSegment(3, "a", 0.10, mora_index=1),
        JapanesePlannedSegment(4, "pau", 0.05, pause_role="trailing"),
    )
    targets = (
        JapaneseF0Target(0.10, math.log2(440.0), math.log2(440.0),
                         0, 0, 0, "test"),
        JapaneseF0Target(0.25, math.log2(440.0), math.log2(440.0),
                         0, 0, 1, "test"),
    )
    return JapaneseSynthesisPlan(
        source_text="aka",
        normalized_reading="aka",
        frontend_name="fixture",
        segments=segments,
        f0_targets=targets,
        unit_overrides=dict(unit_overrides or {}),
        manual_candidate_overrides={1: "manual"} if unit_overrides else {},
        diagnostics=(),
        base_pitch_hz=440.0,
        speed=1.0,
    )


def _runtime():
    def choices(pair):
        return [
            {
                "candidate_id": pair + "-low",
                "left_name": pair + "_low",
                "source_pitch_tags": ["C3"],
                "subbank_ids": ["red-low"],
                "selection_cost": 0.0,
            },
            {
                "candidate_id": pair + "-high",
                "left_name": pair + "_high",
                "source_pitch_tags": ["A4"],
                "subbank_ids": ["blue-high"],
                "selection_cost": 0.0,
            },
        ]
    return {
        "language": "ja",
        "alternatives": {
            "pau-a": choices("pau-a"),
            "a-k": choices("a-k"),
            "k-a": choices("k-a"),
            "a-pau": choices("a-pau"),
        },
        "subbanks": [
            {"subbank_id": "red-low", "color": "Soft",
             "tone_ranges": ["C3-E3"]},
            {"subbank_id": "blue-high", "color": "Power",
             "tone_ranges": ["G4-C5"]},
        ],
    }


class JapaneseRefinementTests(unittest.TestCase):
    def test_note_and_tone_range_parsing(self):
        self.assertEqual(jr.note_to_midi("C4"), 60.0)
        self.assertEqual(jr.note_to_midi("F#3"), 54.0)
        self.assertEqual(jr.note_to_midi("Gb3"), 54.0)
        self.assertEqual(jr.tone_range_center("C4-E4"), 62.0)
        self.assertIsNone(jr.note_to_midi("strong"))

    def test_dynamic_pitch_routing_is_deterministic(self):
        plan = _plan()
        policy = jr.JapaneseRoutingPolicy(dynamic_pitch=True)

        first = jr.route_dynamic_candidates(plan, _runtime(), policy)
        second = jr.route_dynamic_candidates(plan, _runtime(), policy)

        self.assertEqual(first.to_json_bytes(), second.to_json_bytes())
        self.assertEqual(first.unit_overrides[1], "a-k_high")
        self.assertIn("dynamic_source_routing_applied",
                      [item.code for item in first.diagnostics])

    def test_voice_color_is_exact_and_falls_back_without_data_loss(self):
        power = jr.route_dynamic_candidates(
            _plan(), _runtime(),
            jr.JapaneseRoutingPolicy(voice_color="Power"),
        )
        missing = jr.route_dynamic_candidates(
            _plan(), _runtime(),
            jr.JapaneseRoutingPolicy(voice_color="Whisper"),
        )

        self.assertEqual(power.unit_overrides[2], "k-a_high")
        self.assertNotIn(2, missing.unit_overrides)
        self.assertIn("voice_color_fallback",
                      [item.code for item in missing.diagnostics])
        self.assertEqual(jr.available_voice_colors(_runtime()),
                         ("Power", "Soft"))

    def test_manual_unit_override_is_always_final(self):
        plan = _plan({1: "manual_left_name"})

        routed = jr.route_dynamic_candidates(
            plan, _runtime(),
            jr.JapaneseRoutingPolicy(
                dynamic_pitch=True, voice_color="Power"
            ),
        )

        self.assertEqual(routed.unit_overrides[1], "manual_left_name")
        self.assertEqual(routed.manual_candidate_overrides, {1: "manual"})

    def test_external_trajectory_is_optional_and_preserves_manual_units(self):
        plan = _plan({1: "manual_left_name"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            path.write_text(json.dumps({
                "language": "ja",
                "provider_version": "fixture-1",
                "phones": plan.phones,
                "durations": [0.06, 0.11, 0.07, 0.11, 0.06],
                "f0_targets": [[0.11, 430.0], [0.27, 450.0]],
            }), encoding="utf-8")
            result = jr.ExternalHTSTrajectoryProvider(path).provide(
                japanese_frontend.analyze_japanese("aka", mode="kana")
            )

        self.assertIsNotNone(result.trajectory)
        applied = jr.apply_baseline_trajectory(plan, result.trajectory)
        self.assertEqual(applied.unit_overrides, plan.unit_overrides)
        self.assertEqual(applied.manual_candidate_overrides,
                         plan.manual_candidate_overrides)
        self.assertEqual(applied.segments[0].duration, 0.06)
        self.assertEqual(applied.f0_targets[0].hz, 430.0)
        self.assertFalse(result.trajectory.provenance["waveform_used"])

    def test_missing_optional_providers_degrade_gracefully(self):
        utterance = japanese_frontend.analyze_japanese("aka", mode="kana")
        with tempfile.TemporaryDirectory() as directory:
            missing = jr.ExternalHTSTrajectoryProvider(
                Path(directory) / "missing.json"
            ).provide(utterance)
        with mock.patch.object(
                japanese_frontend, "analyze_japanese",
                side_effect=RuntimeError("not installed")):
            openjtalk = jr.OpenJTalkLabelBaselineProvider().provide(utterance)

        self.assertIsNone(missing.trajectory)
        self.assertEqual(missing.diagnostics[0].severity, "info")
        self.assertIsNone(openjtalk.trajectory)
        self.assertEqual(openjtalk.diagnostics[0].code,
                         "openjtalk_baseline_unavailable")

    def test_openjtalk_label_provider_never_calls_waveform_synthesis(self):
        utterance = japanese_frontend.analyze_japanese("aka", mode="kana")
        fake_openjtalk = types.SimpleNamespace(
            tts=mock.Mock(side_effect=AssertionError("waveform requested")),
            synthesize=mock.Mock(
                side_effect=AssertionError("waveform requested")
            ),
        )
        with mock.patch.object(
                japanese_frontend, "analyze_japanese",
                return_value=utterance), mock.patch.object(
                jr.importlib.metadata, "version", return_value="fixture"), \
                mock.patch.dict(sys.modules, {
                    "pyopenjtalk": fake_openjtalk,
                }):
            result = jr.OpenJTalkLabelBaselineProvider().provide(utterance)

        self.assertIsNotNone(result.trajectory)
        self.assertFalse(result.trajectory.provenance["waveform_used"])
        fake_openjtalk.tts.assert_not_called()
        fake_openjtalk.synthesize.assert_not_called()

    def test_mismatched_optional_baseline_is_rejected(self):
        plan = _plan({1: "manual"})
        trajectory = jr.JapaneseBaselineTrajectory(
            provider_name="fixture",
            phones=("pau", "i", "pau"),
            durations=(0.05, 0.10, 0.05),
            f0_targets=((0.10, 200.0),),
        )

        result = jr.apply_baseline_trajectory(plan, trajectory)

        self.assertEqual(result.segments, plan.segments)
        self.assertEqual(result.unit_overrides, plan.unit_overrides)
        self.assertEqual(result.diagnostics[-1].code,
                         "optional_baseline_phone_mismatch")


if __name__ == "__main__":
    unittest.main()
