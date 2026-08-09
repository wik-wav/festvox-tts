from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import japanese_duration_ab as ab


class JapaneseDurationABTests(unittest.TestCase):
    def test_committed_validation_manifest_is_runtime_source_of_truth(self):
        config = ab.load_validation_config()

        self.assertEqual(config["schema_version"], 1)
        self.assertEqual(
            [item[0] for item in ab.AB_FIXTURES],
            [item["id"] for item in config["ab_fixtures"]],
        )
        self.assertEqual(
            [item[0] for item in ab.AB_SYSTEMS],
            [item["id"] for item in config["ab_systems"]],
        )

    def test_fixture_matrix_covers_required_devoicing_and_timing_controls(self):
        identifiers = {item[0] for item in ab.AB_FIXTURES}

        self.assertTrue({
            "suki_desu", "tsukue_fuku", "kutsu_tsukutte", "kita_iku",
            "i_stop", "i_voiced_control", "u_fricative",
            "u_voiced_control", "sushi_tsukutta", "hitotsu_zutsu",
            "long_vowel", "geminate", "nasal", "phrase_final",
            "ambiguous",
        }.issubset(identifiers))

    def test_ab_command_uses_same_backend_with_distinct_duration_modes(self):
        import sys
        gui = Path(ab.__file__).resolve().parent / "festvox_gui"
        if str(gui) not in sys.path:
            sys.path.insert(0, str(gui))
        import festvox_core as fc

        class Backend:
            def synth_phones(self, phones, _voicebank, **kwargs):
                entries = list(kwargs["seg_durs"])
                segments = fc.segments_from_durations(entries)
                sr = 16000
                chunks = []
                for phone, duration in entries:
                    count = max(1, int(round(duration * sr)))
                    time = np.arange(count) / sr
                    if phone in {"i", "u"}:
                        chunk = 0.2 * np.sin(2 * np.pi * 180 * time)
                    else:
                        chunk = np.zeros(count)
                    chunks.append(chunk.astype(np.float32))
                return fc.Synthesis(
                    np.concatenate(chunks), sr, segments,
                    text=kwargs.get("text", ""), lang="ja", phones=list(phones),
                )

        runtime = {
            "language": "ja",
            "voice_name": "fixture",
            "voice_entry_point": "voice_fixture",
            "candidate_units": {},
            "average_pitch_hz": 180.0,
        }
        with tempfile.TemporaryDirectory() as root:
            voice = Path(root) / "voice"
            output = Path(root) / "output"
            voice.mkdir()
            with mock.patch.object(ab, "load_japanese_runtime_metadata",
                                   return_value=runtime), \
                    mock.patch.object(
                        ab, "AB_FIXTURES",
                        (("u_stop", "u stop", "kutsu"),)):
                manifest = ab.render_duration_ab(
                    voice, output, frontend_mode="kana", backend=Backend()
                )

            self.assertTrue((output / "u_stop__legacy.wav").is_file())
            self.assertTrue((output / "u_stop__duration_only.wav").is_file())
            self.assertTrue((output / "u_stop__contextual.wav").is_file())
            systems = manifest["examples"][0]["systems"]
            self.assertEqual(systems["legacy"]["duration_model"], "legacy")
            self.assertEqual(
                systems["duration_only"]["duration_model"], "contextual")
            self.assertTrue(
                all(item["strategy"] == "shortened_voiced_fallback"
                    for item in systems["duration_only"]
                    ["vowel_realizations"])
            )
            self.assertEqual(
                systems["contextual"]["duration_model"], "contextual")
            self.assertEqual(manifest["render_failure_count"], 0)
            self.assertEqual(
                manifest["psola_no_f0_experiment"]["status"], "completed"
            )
            minimum = manifest["psola_no_f0_experiment"][
                "minimum_f0_attempt"
            ]
            self.assertIsNotNone(
                minimum["median_periodicity_25_450_hz"]
            )
            self.assertIsNotNone(
                minimum["longest_vowel_periodicity_25_450_hz"]
            )
            self.assertFalse(manifest["acoustic_naturalness_verified"])
            self.assertEqual(manifest["summary"]["rendered_pair_count"], 1)
            self.assertIn("strategy_counts", manifest["summary"])
            self.assertTrue((output / "report.md").is_file())
            self.assertIn(
                "Acoustic naturalness remains pending human listening",
                (output / "report.md").read_text(encoding="utf-8"),
            )

    def test_output_guard_rejects_voice_and_utau_bank_locations(self):
        with tempfile.TemporaryDirectory() as root:
            voice = Path(root) / "generated"
            voice.mkdir()
            with self.assertRaisesRegex(ValueError, "generated voice"):
                ab._safe_output(voice, voice / "ab")


if __name__ == "__main__":
    unittest.main()
