import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from source_filter_voicing import transform_voicing
from test_source_filter_voicing import _mixed_vowel
import voicing_validation as validation


class VoicingValidationTests(unittest.TestCase):
    def test_metrics_distinguish_harmonics_from_filtered_noise(self):
        source, sample_rate = _mixed_vowel()
        duration = len(source) / float(sample_rate)
        partial = transform_voicing(
            source, sample_rate, [(0.0, 0.50), (duration, 0.50)]
        ).samples
        zero = transform_voicing(
            source, sample_rate, [(0.0, 0.0), (duration, 0.0)]
        ).samples
        metrics = validation.analyze_voicing_variants(
            source, partial, zero, sample_rate,
            (int(0.10 * sample_rate), int(0.45 * sample_rate)),
        )

        comparison = metrics["zero_vs_source"]
        self.assertTrue(comparison["periodicity_removed"])
        self.assertTrue(comparison["harmonic_ridges_reduced"])
        self.assertTrue(comparison["tract_envelope_retained"])
        self.assertGreater(
            comparison["harmonic_ridge_contrast_drop_db"], 1.0
        )
        self.assertGreater(comparison["tract_envelope_correlation"], 0.90)
        self.assertGreater(metrics["source_f0_hz"], 130.0)
        self.assertLess(metrics["source_f0_hz"], 180.0)

    @unittest.skipUnless(
        importlib.util.find_spec("PyQt5") is not None,
        "PyQt5 is optional outside the GUI runtime",
    )
    def test_cli_artifacts_are_complete_and_path_private(self):
        source, sample_rate = _mixed_vowel()
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            source_path = root_path / "source.wav"
            output_path = root_path / "validation"
            validation._write_pcm_wav(source_path, source, sample_rate)

            payload = validation.generate_voicing_validation(
                source_path, output_path, prefix="fixture"
            )

            for filename in payload["outputs"].values():
                artifact = output_path / filename
                self.assertTrue(artifact.is_file(), artifact)
                self.assertGreater(artifact.stat().st_size, 100)
            serialized = json.dumps(payload, sort_keys=True)
            self.assertNotIn(str(root_path), serialized)
            self.assertEqual(
                payload["zero_transform"]["method"],
                "continuous_stochastic_source_filter_v2",
            )
            self.assertTrue(all(payload["zero_vs_source"][key] for key in (
                "periodicity_removed",
                "harmonic_ridges_reduced",
                "tract_envelope_retained",
            )))


if __name__ == "__main__":
    unittest.main()
