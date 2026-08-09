from pathlib import Path
import tempfile
import unittest

import numpy as np

import vocal_tract_listening as listening


class VocalTractListeningTests(unittest.TestCase):
    def test_paired_frame_f0_measurement_uses_same_frames(self):
        sample_rate = 16000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        samples = np.asarray(
            0.2 * np.sin(2.0 * np.pi * 180.0 * time), np.float32)
        drift = listening._paired_frame_f0_drift(
            samples, samples.copy(), sample_rate)
        self.assertIsNotNone(drift)
        self.assertAlmostEqual(drift, 0.0, places=10)

    def test_fixed_inventory_covers_required_japanese_material(self):
        identifiers = {row[0] for row in listening.VALIDATION_FIXTURES}
        self.assertTrue({
            "isolated_vowels", "connected_blue_house", "long_tokyo",
            "accent_rain", "accent_candy", "creak_end",
        }.issubset(identifiers))
        self.assertEqual(set(listening.BLIND_CONDITIONS), {
            "pre_transform_synthesis", "identity_transform",
            "realistic_longer_tract", "realistic_shorter_tract",
            "expanded_shorter_tract", "expanded_longer_tract",
            "pitch_only", "tract_only", "combined_pitch_and_tract",
        })

    def test_blind_codes_are_stable_and_do_not_reveal_conditions(self):
        first = listening._blind_codes("fixture")
        second = listening._blind_codes("fixture")
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(listening.BLIND_CONDITIONS))
        self.assertEqual(set(first.values()), set("ABCDEFGHI"))

    def test_runner_uses_production_transform_and_writes_blind_key(self):
        import sys
        gui = Path(listening.__file__).resolve().parent / "festvox_gui"
        if str(gui) not in sys.path:
            sys.path.insert(0, str(gui))
        import festvox_core as fc

        class Backend:
            def synth_phones(self, phones, _voicebank, **kwargs):
                entries = list(kwargs["seg_durs"])
                segments = fc.segments_from_durations(entries)
                sample_rate = 16000
                chunks = []
                for phone, duration in entries:
                    count = max(1, int(round(float(duration) * sample_rate)))
                    time = np.arange(count, dtype=np.float64) / sample_rate
                    if phone in {"a", "i", "u", "e", "o"}:
                        chunk = (
                            0.18 * np.sin(2.0 * np.pi * 180.0 * time)
                            + 0.05 * np.sin(2.0 * np.pi * 540.0 * time)
                        )
                    else:
                        chunk = np.zeros(count, np.float64)
                    chunks.append(np.asarray(chunk, np.float32))
                return fc.Synthesis(
                    np.concatenate(chunks), sample_rate, segments,
                    text=kwargs.get("text", ""), lang="ja",
                    phones=list(phones),
                )

        runtime = {
            "language": "ja",
            "voice_name": "fixture",
            "voice_entry_point": "voice_fixture",
            "candidate_units": {},
            "average_pitch_hz": 180.0,
        }
        fixture = (("vowels", "five vowels", "あいうえお。"),)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voice = root / "voice"
            output = root / "output"
            voice.mkdir()
            report = listening.render_vocal_tract_listening_suite(
                voice,
                output,
                frontend_mode="kana",
                backend=Backend(),
                runtime_metadata=runtime,
                fixtures=fixture,
                blind_fixture_ids=("vowels",),
                analyze_joins=False,
            )
            public = (output / "blind_manifest.json").read_text(
                encoding="utf-8")
            key = (output / "blind_key.json").read_text(encoding="utf-8")
            blind_wavs = list((output / "blind").glob("*.wav"))
        self.assertEqual(report["render_failure_count"], 0)
        self.assertTrue(report["identity_exact_for_all_rendered_fixtures"])
        self.assertEqual(report["maximum_duration_drift_samples"], 0)
        self.assertLessEqual(
            report[
                "maximum_realistic_tract_paired_frame_f0_drift_semitones"],
            0.08,
        )
        self.assertLessEqual(
            report[
                "maximum_expanded_tract_paired_frame_f0_drift_semitones"],
            0.08,
        )
        self.assertTrue(report["structural_validation_passed"])
        self.assertFalse(report["acoustic_naturalness_verified"])
        self.assertEqual(len(blind_wavs), 9)
        self.assertNotIn("identity_transform", public)
        self.assertIn("identity_transform", key)


if __name__ == "__main__":
    unittest.main()
