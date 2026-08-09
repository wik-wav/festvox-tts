import math
import unittest

import pitch_domain as pd


class PitchDomainTests(unittest.TestCase):
    def test_round_trip(self):
        for hz in (50.0, 129.7, 185.0, 500.0):
            self.assertAlmostEqual(pd.log_f0_to_hz(pd.hz_to_log_f0(hz)), hz)
            self.assertAlmostEqual(
                pd.semitone_number_to_hz(pd.hz_to_semitone_number(hz)), hz)

    def test_offsets_are_speaker_relative(self):
        self.assertAlmostEqual(pd.semitone_offset(100.0, 12.0), 200.0)
        self.assertAlmostEqual(pd.semitone_offset(200.0, -12.0), 100.0)
        self.assertAlmostEqual(pd.semitone_difference(100.0, 200.0), 12.0)

    def test_interpolation_is_logarithmic(self):
        value = pd.interpolate_hz_log([(0.0, 100.0), (1.0, 200.0)], 0.5)
        self.assertAlmostEqual(value, math.sqrt(20000.0))

    def test_recenter_preserves_semitone_shape(self):
        source = [(0.0, 100.0), (1.0, 200.0)]
        shifted = pd.recenter_targets_log(source, 200.0)
        self.assertAlmostEqual(
            pd.semitone_difference(shifted[0][1], shifted[1][1]), 12.0)
        self.assertAlmostEqual(
            pd.geometric_mean_hz(value for _time, value in shifted), 200.0)

    def test_legacy_fall_mapping_is_bounded(self):
        self.assertEqual(pd.fall_percent_to_span_semitones(0.0), 0.0)
        self.assertAlmostEqual(
            pd.fall_percent_to_span_semitones(40.0),
            12.0 * math.log2(1.2),
        )
        self.assertEqual(
            pd.fall_percent_to_span_semitones(400.0),
            pd.fall_percent_to_span_semitones(40.0),
        )


if __name__ == "__main__":
    unittest.main()
