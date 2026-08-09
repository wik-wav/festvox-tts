import hashlib
import math
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from vocal_tract_validation import (
    _annotate_vowel_identity,
    default_validation_ratios,
    envelope_display_range,
    formant_view_spectrogram_db,
    identity_anchored_formants,
    validate_source_vowel_suite,
    validate_vocal_tract_recording,
)


def _synthetic_vowel(sample_rate=16000, seconds=0.72, f0=170.0):
    count = int(round(sample_rate * seconds))
    source = np.zeros(count, np.float64)
    source[::max(1, int(round(sample_rate / f0)))] = 1.0
    values = source
    for formant, bandwidth in ((520.0, 85.0), (1850.0, 120.0),
                               (2850.0, 170.0), (3700.0, 210.0)):
        radius = math.exp(-math.pi * bandwidth / sample_rate)
        coefficient = 2.0 * radius * math.cos(
            2.0 * math.pi * formant / sample_rate
        )
        output = np.zeros_like(values)
        for index in range(values.size):
            output[index] = values[index]
            if index:
                output[index] += coefficient * output[index - 1]
            if index > 1:
                output[index] -= radius * radius * output[index - 2]
        values = output
    values /= max(1.0e-9, float(np.max(np.abs(values))))
    values *= 0.45
    return np.asarray(values, np.float32), sample_rate


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.asarray(np.rint(samples * 32767.0), np.int16)
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(pcm.tobytes())


class VocalTractValidationTests(unittest.TestCase):
    def test_profile_derived_sweep_contains_all_required_ranges(self):
        ratios = default_validation_ratios()
        self.assertEqual(ratios, tuple(sorted(ratios)))
        self.assertIn(1.0, ratios)
        self.assertLess(ratios[0], 0.94)
        self.assertGreater(ratios[-1], 1.06)
        self.assertGreaterEqual(len(ratios), 7)

    def test_identity_anchored_formants_preserve_shifted_correspondence(self):
        frequencies = np.linspace(0.0, 6000.0, 1201)
        identity = (500.0, 1500.0, 2800.0, 3600.0)
        multiplier = 1.35
        envelope = np.full(frequencies.shape, -42.0)
        for formant in identity:
            shifted = formant * multiplier
            envelope += 18.0 * np.exp(
                -0.5 * ((frequencies - shifted) / 85.0) ** 2
            )
        # Add a strong irrelevant low-frequency peak which an unanchored
        # tracker could incorrectly relabel as F1.
        envelope += 24.0 * np.exp(
            -0.5 * ((frequencies - 260.0) / 55.0) ** 2
        )
        measured = identity_anchored_formants(
            frequencies, envelope, identity, multiplier
        )
        for actual, formant in zip(measured, identity):
            self.assertIsNotNone(actual)
            self.assertLess(abs(actual - formant * multiplier), 35.0)

    def test_envelope_display_range_contains_every_plotted_value(self):
        measured = np.asarray([-47.0, -18.0, 16.0])
        target = np.asarray([-42.0, -12.0, 11.0])
        lower, upper, tick = envelope_display_range([measured, target])
        self.assertLess(lower, float(np.min(measured)))
        self.assertGreater(upper, float(np.max(measured)))
        self.assertGreaterEqual(tick, 5.0)

    def test_formant_view_reduces_harmonic_comb_contrast(self):
        frequencies = np.linspace(0.0, 6000.0, 257)
        decibels = np.full((frequencies.size, 8), -72.0)
        for harmonic in range(170, 6000, 170):
            index = int(np.argmin(np.abs(frequencies - harmonic)))
            formant_gain = 18.0 * math.exp(
                -0.5 * ((harmonic - 1900.0) / 330.0) ** 2
            )
            decibels[index, :] = -22.0 + formant_gain
        smoothed = formant_view_spectrogram_db(decibels, frequencies)
        self.assertEqual(smoothed.shape, decibels.shape)
        self.assertTrue(np.all(np.isfinite(smoothed)))
        self.assertLess(float(np.std(smoothed[:, 3])),
                        float(np.std(decibels[:, 3])))
        peak = int(np.argmax(smoothed[:, 3]))
        self.assertLess(abs(frequencies[peak] - 1900.0), 500.0)

    def test_final_waveform_follows_requested_envelope(self):
        samples, sample_rate = _synthetic_vowel()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            _write_wav(source, samples, sample_rate)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            report, plot = validate_vocal_tract_recording(
                source,
                root / "result",
                ratios=(0.80, 1.0, 1.25),
                render_plot=False,
            )
            after = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertIsNone(plot)
        self.assertTrue(report["identity_exact"])
        self.assertIn("source_filter_warp_v2", report["method"])
        points = {row["ratio"]: row for row in report["points"]}
        for ratio in (0.80, 1.0, 1.25):
            self.assertEqual(points[ratio]["duration_drift_samples"], 0)
            self.assertEqual(points[ratio]["clipped_sample_count"], 0)
            self.assertLess(abs(points[ratio]["f0_drift_semitones"]), 0.08)
        # The synthetic all-pole fixture has much deeper spectral nulls than a
        # recorded voice, so the production 30 dB safety limiter is active.
        self.assertLess(points[0.80]["envelope_target_rmse_db"], 6.5)
        self.assertLess(points[1.25]["envelope_target_rmse_db"], 6.5)
        self.assertGreater(points[0.80]["envelope_change_rms_db"], 3.0)
        self.assertGreater(points[1.25]["envelope_change_rms_db"], 3.0)
        for ratio in (0.80, 1.0, 1.25):
            self.assertIn("median_absolute_formant_error_hz", points[ratio])
            self.assertIn("median_absolute_formant_ratio_error",
                          points[ratio])

    def test_vowel_identity_uses_ratio_compensated_f1_f2(self):
        centroids = {
            "a": (800.0, 1200.0),
            "e": (500.0, 2000.0),
            "i": (300.0, 2500.0),
            "o": (600.0, 1000.0),
            "u": (350.0, 900.0),
        }
        rows = {}
        for vowel, (f1, f2) in centroids.items():
            rows[vowel] = {"points": [
                {
                    "ratio": ratio,
                    "measured_formants_hz": [f1 / ratio, f2 / ratio],
                }
                for ratio in (0.8, 1.0, 1.2)
            ]}
        audit = _annotate_vowel_identity(rows)
        self.assertEqual(audit["status"], "measured")
        self.assertEqual(audit["check_count"], 15)
        self.assertEqual(audit["preserved_count"], 15)
        for vowel, row in rows.items():
            self.assertTrue(all(
                point["nearest_identity_vowel"] == vowel and
                point["vowel_identity_preserved"]
                for point in row["points"]
            ))

    def test_visual_audit_writes_png_json_and_comparison_wavs(self):
        samples, sample_rate = _synthetic_vowel(seconds=0.42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            output = root / "result"
            _write_wav(source, samples, sample_rate)
            _report, plot = validate_vocal_tract_recording(
                source,
                output,
                ratios=(0.80, 1.0, 1.25),
                title="Synthetic final-waveform validation",
                render_plot=True,
            )
            self.assertIsNotNone(plot)
            self.assertTrue(plot.is_file())
            self.assertEqual(plot.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(plot.stat().st_size, 20_000)
            self.assertTrue((output / "vocal_tract_validation.json").is_file())
            for ratio in (0.80, 1.0, 1.25):
                self.assertTrue((output / f"ratio_{ratio:.3f}.wav").is_file())

    def test_five_vowel_suite_is_read_only_and_machine_readable(self):
        samples, sample_rate = _synthetic_vowel(seconds=0.72)
        fixtures = {
            vowel: (f"{vowel}.wav", 0.04, 0.68)
            for vowel in "aeiou"
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            before = {}
            for vowel in "aeiou":
                path = source / f"{vowel}.wav"
                _write_wav(path, samples, sample_rate)
                before[vowel] = hashlib.sha256(path.read_bytes()).hexdigest()
            report = validate_source_vowel_suite(
                source,
                output,
                ratios=(0.94, 1.0, 1.06),
                render_plots=False,
                fixtures=fixtures,
            )
            after = {
                vowel: hashlib.sha256(
                    (source / f"{vowel}.wav").read_bytes()).hexdigest()
                for vowel in "aeiou"
            }
            saved = (output / "stage_b_source_vowels.json").read_text(
                encoding="utf-8")
        self.assertEqual(before, after)
        self.assertTrue(report["passed"])
        self.assertTrue(report["source_hashes_unchanged"])
        self.assertEqual(report["summary"]["vowel_count"], 5)
        self.assertEqual(report["summary"]["maximum_duration_drift_samples"],
                         0)
        self.assertIn("median_absolute_formant_error_hz", report["summary"])
        self.assertIn("median_absolute_formant_ratio_error",
                      report["summary"])
        self.assertIn("metrics_by_range", report)
        self.assertIn("metrics_by_f0_range", report)
        self.assertIn("metrics_by_phonation", report)
        self.assertIn('"kind": "prompt20_stage_b_source_vowel_validation"',
                      saved)


if __name__ == "__main__":
    unittest.main()
