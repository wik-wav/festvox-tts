import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from join_spectrogram import (
    _is_expected_burst_phone,
    _join_marker_time,
    _phone_sequence_text,
    render_join_spectrogram,
    spectrogram_db,
)


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class JoinSpectrogramTests(unittest.TestCase):
    def test_stop_and_palatalized_stop_phone_contexts_are_recognized(self):
        self.assertTrue(_is_expected_burst_phone("k"))
        self.assertTrue(_is_expected_burst_phone("ky"))
        self.assertTrue(_is_expected_burst_phone("CH"))
        self.assertFalse(_is_expected_burst_phone("s"))
        self.assertFalse(_is_expected_burst_phone("a"))

    def test_broadband_issue_marker_uses_measured_event_time(self):
        row = {
            "time": 0.5,
            "issues": ["BROADBAND_IMPULSE"],
            "broadband_impulse_time_seconds": 0.50325,
        }
        self.assertAlmostEqual(_join_marker_time(row), 0.50325)
        row["issues"] = ["PHASE_MISMATCH"]
        self.assertAlmostEqual(_join_marker_time(row), 0.5)

    def test_complete_rendered_phone_sequence_is_available_for_footer(self):
        diagnostic = {"segments": [
            {"phone": "pau", "start": 0.0, "end": 0.1},
            {"phone": "a", "start": 0.1, "end": 0.3},
            {"phone": "k", "start": 0.3, "end": 0.4},
            {"phone": "a", "start": 0.4, "end": 0.6},
        ]}
        self.assertEqual(_phone_sequence_text(diagnostic), "pau a k a")

    def test_phase_step_fixture_renders_deterministic_png_shape(self):
        sample_rate = 16000
        half = np.arange(sample_rate // 2, dtype=np.float64) / sample_rate
        samples = np.concatenate((
            0.2 * np.sin(2.0 * np.pi * 200.0 * half),
            0.2 * np.cos(2.0 * np.pi * 200.0 * half),
        ))
        diagnostic = {
            "summary": {"flagged_join_count": 1},
            "segments": [
                {"phone": "a", "start": 0.0, "end": 0.45},
                {"phone": "k", "start": 0.45, "end": 0.55},
                {"phone": "a", "start": 0.55, "end": 1.0},
            ],
            "joins": [{
                "time": 0.5,
                "flagged": True,
                "dominant_issue": "PHASE_MISMATCH",
                "issues": ["BROADBAND_IMPULSE"],
                "broadband_context_may_be_expected": True,
                "broadband_impulse_time_seconds": 0.501,
            }],
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "phase_step.png"
            render_join_spectrogram(
                samples, sample_rate, output,
                diagnostic=diagnostic, title="Synthetic phase step",
                width=900, height=600,
            )
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            from PyQt5 import QtGui

            image = QtGui.QImage(str(output))
            # The upper-right legend interior must contain rendered glyphs,
            # not only its border. This catches broken system font lookup.
            dark_pixels = 0
            for y in range(20, 54):
                for x in range(310, 870):
                    color = image.pixelColor(x, y)
                    if max(color.red(), color.green(), color.blue()) < 180:
                        dark_pixels += 1
            self.assertGreater(dark_pixels, 50)
            # Compact exports must still contain the time-aligned phone strip
            # below the spectrogram.  The middle phone is k and therefore has
            # the blue expected-burst tint rather than the neutral fill.
            k_pixel = image.pixelColor(484, 450)
            self.assertGreater(k_pixel.blue(), k_pixel.red())
            phone_footer_ink = 0
            for y in range(474, 509):
                for x in range(96, 870):
                    color = image.pixelColor(x, y)
                    if max(color.red(), color.green(), color.blue()) < 150:
                        phone_footer_ink += 1
            self.assertGreater(
                phone_footer_ink, 12,
                "phone sequence text was not rendered under the spectrogram")
            nearly_black = 0
            for y in range(image.height()):
                for x in range(image.width()):
                    color = image.pixelColor(x, y)
                    if max(color.red(), color.green(), color.blue()) < 20:
                        nearly_black += 1
            self.assertLess(
                nearly_black,
                image.width() * image.height() * 0.03,
                "font rendering painted opaque blocks into the diagnostic",
            )

    def test_spectrogram_is_finite_and_exposes_frequency_bins(self):
        sample_rate = 16000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        times, frequencies, values = spectrogram_db(
            np.sin(2.0 * np.pi * 220.0 * time), sample_rate,
            fft_size=512, hop_size=64,
        )

        self.assertEqual(values.shape, (len(frequencies), len(times)))
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(float(frequencies[-1]), 8000.0)


if __name__ == "__main__":
    unittest.main()
