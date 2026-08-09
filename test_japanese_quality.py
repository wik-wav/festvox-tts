from __future__ import annotations

import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import wave

import japanese_quality as jq


def _write_wave(path: Path, values, rate: int = 16000) -> None:
    import struct

    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [max(-32768, min(32767, int(round(value * 32767.0))))
               for value in values]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<%dh" % len(samples), *samples))


class JapaneseQualityTests(unittest.TestCase):
    def _fixture(self, root: Path):
        rate = 16000
        values = [0.35 * math.sin(2.0 * math.pi * 200.0 * i / rate)
                  for i in range(rate)]
        wav = root / "voice" / "wav" / "tone.wav"
        _write_wave(wav, values, rate)
        runtime = {
            "language": "ja",
            "index": {
                "a": ["tone.wav", 0.10, 0.20, 0.40],
                "b": ["tone.wav", 0.40, 0.60, 0.80],
            },
            "alternatives": {
                "a-k": [{
                    "id": "left", "candidate_id": "left",
                    "left_name": "a", "index_name": "a",
                }],
                "k-a": [{
                    "id": "right", "candidate_id": "right",
                    "left_name": "b", "index_name": "b",
                }],
            },
        }
        plan = SimpleNamespace(
            phones=["a", "k", "a"], unit_overrides={}
        )
        return root / "voice", runtime, plan

    def test_plan_join_report_is_relative_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            voice, runtime, plan = self._fixture(Path(directory))

            first = jq.analyze_plan_joins(plan, runtime, voice)
            second = jq.analyze_plan_joins(plan, runtime, voice)

            self.assertEqual(first.to_json_bytes(), second.to_json_bytes())
            self.assertEqual(len(first.metrics), 1)
            metric = first.metrics[0]
            self.assertEqual(metric.left_wav, "wav/tone.wav")
            self.assertNotIn(str(voice), first.to_json_bytes().decode("utf-8"))
            self.assertEqual(metric.rating, "good")

    def test_content_addressed_cache_is_reused_without_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voice, runtime, plan = self._fixture(root)
            cache = root / "cache"

            first = jq.analyze_plan_joins(
                plan, runtime, voice, cache_directory=cache)
            second = jq.analyze_plan_joins(
                plan, runtime, voice, cache_directory=cache)

            self.assertEqual(first.cache_hits, 0)
            self.assertEqual(second.cache_hits, 1)
            self.assertEqual(first.metrics, second.metrics)
            cached = list(cache.rglob("*.json"))
            self.assertEqual(len(cached), 1)
            self.assertNotIn("timestamp", cached[0].read_text(encoding="utf-8"))

    def test_each_generated_wav_is_hashed_once_per_report(self):
        with tempfile.TemporaryDirectory() as directory:
            voice, runtime, plan = self._fixture(Path(directory))
            original = jq._sha256
            with mock.patch.object(jq, "_sha256", wraps=original) as digest:
                jq.analyze_plan_joins(plan, runtime, voice)

        self.assertEqual(digest.call_count, 1)

    def test_cache_and_report_targets_inside_utau_bank_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "source-bank"
            bank.mkdir()
            (bank / "oto.ini").write_text("", encoding="utf-8")
            cache = bank / "analysis-cache"
            report = bank / "quality.json"

            with self.assertRaisesRegex(ValueError, "source UTAU bank"):
                jq.JapaneseQualityCache(cache)
            with self.assertRaisesRegex(ValueError, "source UTAU bank"):
                jq._require_non_source_output(
                    report, "the Japanese quality report"
                )

            self.assertFalse(cache.exists())
            self.assertFalse(report.exists())

    def test_abrupt_clipped_join_is_flagged_for_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "voice"
            left = root / "wav" / "left.wav"
            right = root / "wav" / "right.wav"
            _write_wave(left, [1.0] * 3200)
            _write_wave(right, [-1.0] * 3200)

            metric, hit = jq.measure_join(
                join_index=0,
                shared_phone="a",
                left_diphone="k-a",
                right_diphone="a-t",
                left_candidate_id="left",
                right_candidate_id="right",
                left_path=left,
                right_path=right,
                left_position=0.10,
                right_position=0.10,
                generated_voice_root=root,
            )

            self.assertFalse(hit)
            self.assertEqual(metric.rating, "poor")
            self.assertGreater(metric.clipping_ratio, 0.9)

    def test_missing_choice_is_visible_instead_of_silently_dropped(self):
        plan = SimpleNamespace(
            phones=["a", "k", "a"], unit_overrides={}
        )
        with tempfile.TemporaryDirectory() as directory:
            report = jq.analyze_plan_joins(
                plan, {"language": "ja", "alternatives": {}, "index": {}},
                directory,
            )

        self.assertEqual(report.metrics, ())
        self.assertEqual(report.requested_join_count, 1)
        self.assertEqual(report.diagnostics[0].code,
                         "join_choice_unavailable")


if __name__ == "__main__":
    unittest.main()
