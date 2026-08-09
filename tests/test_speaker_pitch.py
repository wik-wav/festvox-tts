import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from speaker_pitch import (
    analyze_speaker_pitch,
    automatic_pitch_metadata,
    pitchmark_bounds,
    recommended_default_pitch_hz,
)


def _write_frq(path, average, values, *, declared_count=None):
    payload = bytearray(b"FREQ0003")
    payload.extend(struct.pack("<i", 256))
    payload.extend(struct.pack("<d", float(average)))
    payload.extend(b"\0" * 16)
    payload.extend(struct.pack(
        "<i", len(values) if declared_count is None else declared_count
    ))
    for value in values:
        payload.extend(struct.pack("<dd", float(value), 1.0))
    path.write_bytes(bytes(payload))


def _write_tone(path, frequency=180.0, seconds=0.8, rate=16000):
    frames = bytearray()
    for index in range(int(seconds * rate)):
        sample = int(
            9000.0 * math.sin(2.0 * math.pi * frequency * index / rate)
        )
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))


class SpeakerPitchTests(unittest.TestCase):
    def test_default_pitch_is_the_recorded_median(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("a", "b", "c"):
                _write_frq(
                    root / (name + ".frq"), 180.0,
                    [170.0, 180.0, 190.0],
                )
            result = analyze_speaker_pitch(root)

            recommended = recommended_default_pitch_hz(result)
            self.assertEqual(recommended, result.median_f0_hz)
            policy = automatic_pitch_metadata(
                result, default_is_automatic=True
            )
            self.assertEqual(policy["automatic_pitch_floor_hz"], 180.0)
            self.assertEqual(
                policy["automatic_pitch_headroom_semitones"], 0.0)
            self.assertEqual(
                policy["default_pitch_source"],
                "speaker_median",
            )

    def test_explicit_headroom_remains_an_opt_in_transposition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_frq(root / "a.frq", 164.81, [164.0, 164.81, 165.0])
            result = analyze_speaker_pitch(root)

            shifted = recommended_default_pitch_hz(
                result, headroom_semitones=3.5)

            self.assertAlmostEqual(
                12.0 * math.log2(shifted / result.median_f0_hz),
                3.5,
                places=5,
            )

    def test_explicit_builder_pitch_retains_floor_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("a", "b", "c"):
                _write_frq(
                    root / (name + ".frq"), 160.0,
                    [155.0, 160.0, 165.0],
                )
            result = analyze_speaker_pitch(root)

            policy = automatic_pitch_metadata(
                result, default_is_automatic=False
            )

            self.assertEqual(policy["automatic_pitch_floor_hz"], 160.0)
            self.assertEqual(policy["default_pitch_source"], "builder_override")

    def test_frq_statistics_use_header_medians_and_voiced_frame_range(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sub = root / "pitch"
            sub.mkdir()
            _write_frq(sub / "c.frq", 200.0, [190.0, 200.0, 210.0])
            _write_frq(sub / "a.frq", 160.0, [150.0, 160.0, 170.0])
            _write_frq(sub / "b.frq", 180.0, [170.0, 180.0, 190.0])

            result = analyze_speaker_pitch(root)

            self.assertEqual(result.source, "frq")
            self.assertEqual(result.median_f0_hz, 180.0)
            self.assertEqual(result.voiced_sample_count, 9)
            self.assertEqual(result.low_percentile_f0_hz, 158.0)
            self.assertEqual(result.high_percentile_f0_hz, 202.0)
            self.assertEqual(result.files_used, (
                "pitch/a.frq", "pitch/b.frq", "pitch/c.frq",
            ))
            self.assertEqual(pitchmark_bounds(result), (111.6, 306.0))

    def test_bad_frq_is_visible_and_waveform_fallback_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_frq(root / "bad.frq", 180.0, [180.0], declared_count=4)
            _write_tone(root / "voice.wav", 200.0)

            first = analyze_speaker_pitch(root)
            second = analyze_speaker_pitch(root)

            self.assertEqual(first.source, "waveform_estimation")
            self.assertAlmostEqual(first.median_f0_hz, 200.0, delta=6.0)
            self.assertEqual(first.files_used, ("voice.wav",))
            self.assertEqual(first.to_dict(), second.to_dict())
            self.assertIn(
                "frq_truncated",
                {item.code for item in first.diagnostics},
            )
            serialized = json.dumps(first.to_dict(), sort_keys=True)
            self.assertNotIn(str(root), serialized)

    def test_invalid_sources_use_documented_fixed_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "silent.wav").write_bytes(b"not a wav")

            result = analyze_speaker_pitch(root)

            self.assertEqual(result.source, "fallback")
            self.assertEqual(result.median_f0_hz, 185.0)
            self.assertEqual(result.voiced_sample_count, 0)
            self.assertEqual(result.files_used, ())
            self.assertIn(
                "speaker_pitch_fallback",
                {item.code for item in result.diagnostics},
            )

    def test_recording_scope_never_borrows_another_pitch_subbank(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            low = root / "E3"
            high = root / "F4"
            low.mkdir()
            high.mkdir()
            selected = []
            for index, frequency in enumerate((160.0, 170.0, 180.0)):
                wav = low / f"low_{index}.wav"
                _write_tone(wav, frequency)
                _write_frq(
                    low / f"low_{index}_wav.frq",
                    frequency,
                    [frequency - 2.0, frequency, frequency + 2.0],
                )
                selected.append(wav)
            for index, frequency in enumerate((300.0, 320.0, 340.0)):
                wav = high / f"high_{index}.wav"
                _write_tone(wav, frequency)
                _write_frq(
                    high / f"high_{index}_wav.frq",
                    frequency,
                    [frequency - 2.0, frequency, frequency + 2.0],
                )

            result = analyze_speaker_pitch(
                root, recording_files=selected
            )

            self.assertEqual(result.source, "frq")
            self.assertEqual(result.median_f0_hz, 170.0)
            self.assertTrue(result.files_used)
            self.assertTrue(all(path.startswith("E3/")
                                for path in result.files_used))


if __name__ == "__main__":
    unittest.main()
