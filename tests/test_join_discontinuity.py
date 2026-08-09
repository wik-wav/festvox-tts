import json
import math
import unittest
from unittest.mock import patch

import numpy as np

import join_discontinuity as jd
from join_discontinuity import (
    JoinAnalysisConfig,
    JoinDiscontinuityAnalyzer,
)


class JoinDiscontinuityTests(unittest.TestCase):
    sample_rate = 16000
    duration = 0.8
    splice_time = 0.4

    def setUp(self):
        self.times = (np.arange(int(self.sample_rate * self.duration),
                                dtype=np.float64) / self.sample_rate)
        self.segments = [
            {"phone": "a", "start": 0.0, "end": 0.2},
            {"phone": "v", "start": 0.2, "end": 0.6},
            {"phone": "b", "start": 0.6, "end": 0.8},
        ]
        self.splice = [{
            "segment_index": 1,
            "time": self.splice_time,
            "position_source": "synthetic-exact",
            "estimated": False,
        }]
        self.marks_200 = np.arange(
            0.0, self.duration + 1.0e-9, 1.0 / 200.0)

    def periodic(self, frequency=200.0, phase=0.3, amplitude=0.2):
        return amplitude * np.sin(
            2.0 * math.pi * frequency * self.times + phase)

    def analyze(self, samples, marks=None, config=None):
        result = JoinDiscontinuityAnalyzer(
            samples,
            self.sample_rate,
            self.segments,
            splice_records=self.splice,
            target_pitchmarks=(self.marks_200 if marks is None else marks),
            config=config,
        ).analyze()
        self.assertEqual(len(result["joins"]), 1)
        return result, result["joins"][0]

    def analyze_sustained_join(self, duration, *, notch_ms=0.0,
                               phone="a"):
        """Analyze a duration-specific vowel join with an exact handoff."""
        sample_count = int(round(self.sample_rate * duration))
        times = np.arange(sample_count, dtype=np.float64) / self.sample_rate
        splice_time = duration * 0.5
        samples = 0.2 * np.sin(
            2.0 * math.pi * 200.0 * times + 0.3)
        if notch_ms > 0.0:
            half_notch = notch_ms / 2000.0
            samples[np.abs(times - splice_time) < half_notch] = 0.0
        segments = [
            {"phone": "a", "start": 0.0, "end": duration * 0.25},
            {"phone": phone, "start": duration * 0.25,
             "end": duration * 0.75},
            {"phone": "a", "start": duration * 0.75, "end": duration},
        ]
        half_handoff = max(0.004, notch_ms / 2000.0)
        splices = [{
            "segment_index": 1,
            "time": splice_time,
            "handoff_start": splice_time - half_handoff,
            "handoff_end": splice_time + half_handoff,
            "position_source": "synthetic-sustained-exact",
            "estimated": False,
        }]
        marks = np.arange(0.0, duration + 1.0e-9, 1.0 / 200.0)
        result = JoinDiscontinuityAnalyzer(
            samples,
            self.sample_rate,
            segments,
            splice_records=splices,
            target_pitchmarks=marks,
        ).analyze()
        self.assertEqual(len(result["joins"]), 1)
        return result, result["joins"][0]

    def harmonic_signal(self, mixture):
        mixture = np.asarray(mixture, dtype=np.float64)
        base = np.sin(2.0 * math.pi * 200.0 * self.times + 0.3)
        harmonic = np.sin(2.0 * math.pi * 400.0 * self.times + 0.1)
        return 0.2 * (base + mixture * harmonic) / np.sqrt(
            1.0 + mixture * mixture)

    def analyze_tilted_speech_transition(self, *, click_amplitude=0.0):
        """Make a strong but steeply tilted sh-o-w-like local transition."""
        samples = np.zeros_like(self.times)
        for harmonic in range(1, 31):
            samples += np.sin(
                2.0 * math.pi * 200.0 * harmonic * self.times +
                0.2 * harmonic) / harmonic ** 2.5
        samples *= 0.08 / math.sqrt(float(np.mean(samples ** 2)))

        half_transition = 0.006
        transition = np.abs(self.times - self.splice_time) < half_transition
        envelope = np.ones_like(self.times)
        envelope[transition] += 1.5 * (
            1.0 + np.cos(
                math.pi * (self.times[transition] - self.splice_time) /
                half_transition))
        samples *= envelope

        event_sample = int(round(
            (self.splice_time + 0.0015) * self.sample_rate))
        samples[event_sample] += float(click_amplitude)
        segments = [
            {"phone": "sh", "start": 0.0, "end": 0.35},
            {"phone": "o", "start": 0.35, "end": 0.60},
            {"phone": "w", "start": 0.60, "end": self.duration},
        ]
        splices = [{
            "segment_index": 1,
            "time": self.splice_time,
            "handoff_start": self.splice_time - half_transition,
            "handoff_end": self.splice_time + half_transition,
            "position_source": "synthetic-tilted-speech",
            "estimated": False,
        }]
        result = JoinDiscontinuityAnalyzer(
            samples, self.sample_rate, segments,
            splice_records=splices, target_pitchmarks=self.marks_200,
        ).analyze()
        return result["joins"][0], event_sample

    def resonant_vowel(self, formants, *, gains=None, bandwidths=None,
                       parallel=False):
        """Generate a deterministic periodic vowel without audio fixtures."""
        formants = tuple(float(value) for value in formants)
        bandwidths = tuple(float(value) for value in (
            bandwidths or (80.0,) * len(formants)))
        gains = tuple(float(value) for value in (
            gains or (1.0,) * len(formants)))
        warmup = self.sample_rate // 4
        total = len(self.times) + warmup
        excitation = np.zeros(total, dtype=np.float64)
        period = int(round(self.sample_rate / 200.0))
        excitation[::period] = 1.0

        def resonate(source, frequency, bandwidth):
            radius = math.exp(-math.pi * bandwidth / self.sample_rate)
            coefficient = 2.0 * radius * math.cos(
                2.0 * math.pi * frequency / self.sample_rate)
            radius_squared = radius * radius
            output = np.zeros_like(source)
            for index in range(2, len(source)):
                output[index] = (source[index] +
                                 coefficient * output[index - 1] -
                                 radius_squared * output[index - 2])
            return output

        if parallel:
            filtered = np.zeros_like(excitation)
            for frequency, bandwidth, gain in zip(
                    formants, bandwidths, gains):
                component = resonate(excitation, frequency, bandwidth)
                component_rms = math.sqrt(float(np.mean(component ** 2)))
                filtered += gain * component / max(1.0e-12, component_rms)
        else:
            filtered = excitation
            for frequency, bandwidth in zip(formants, bandwidths):
                filtered = resonate(filtered, frequency, bandwidth)
        filtered = filtered[warmup:]
        filtered -= float(np.mean(filtered))
        filtered *= 0.16 / max(1.0e-12,
                               math.sqrt(float(np.mean(filtered ** 2))))
        return filtered

    def test_continuous_periodic_same_phase_is_clean(self):
        _result, join = self.analyze(self.periodic())
        self.assertEqual(join["voicing"], "voiced")
        self.assertEqual(join["dominant_issue"], "OK")
        self.assertLess(abs(join["level_step_db"]), 0.01)
        self.assertGreater(join["zero_lag_period_correlation"], 0.99)
        self.assertGreater(join["best_lag_period_correlation"], 0.99)
        self.assertLess(join["period_shape_mismatch"], 0.01)

    def test_clean_sustained_vowels_cover_280_and_650_ms(self):
        for duration in (0.280, 0.650):
            with self.subTest(duration=duration):
                _result, join = self.analyze_sustained_join(duration)
                self.assertEqual(join["dominant_issue"], "OK")
                self.assertLess(join["severity_score"], 1.0)
                self.assertGreater(join["content_retention_ratio"], 0.80)
                self.assertLess(join["content_attenuation_db"], 1.0)
                self.assertNotIn("CONTENT_DROPOUT", join["issues"])

    def test_zero_notch_over_audible_vowel_cannot_improve_benchmark(self):
        for duration in (0.280, 0.650):
            with self.subTest(duration=duration):
                _clean_result, clean = self.analyze_sustained_join(duration)
                _notched_result, notched = self.analyze_sustained_join(
                    duration, notch_ms=8.0)

                self.assertLess(notched["content_retention_ratio"], 0.10)
                self.assertGreater(notched["content_attenuation_db"], 12.0)
                self.assertFalse(notched["content_dropout_expected"])
                self.assertIn("CONTENT_DROPOUT", notched["issues"])
                self.assertGreater(
                    notched["severity_components"]["CONTENT_DROPOUT"]
                    ["weighted_score"], 1.0)
                self.assertGreater(notched["severity_score"],
                                   clean["severity_score"])

    def test_low_f0_period_trough_is_not_mistaken_for_dropout(self):
        duration = 0.650
        frequency = 80.0
        sample_count = int(round(self.sample_rate * duration))
        times = np.arange(sample_count, dtype=np.float64) / self.sample_rate
        splice_time = duration * 0.5
        segments = [
            {"phone": "a", "start": 0.0, "end": duration * 0.25},
            {"phone": "a", "start": duration * 0.25,
             "end": duration * 0.75},
            {"phone": "a", "start": duration * 0.75, "end": duration},
        ]
        splices = [{
            "segment_index": 1,
            "time": splice_time,
            "handoff_start": splice_time - 0.020,
            "handoff_end": splice_time + 0.020,
            "position_source": "synthetic-low-f0-exact",
            "estimated": False,
        }]
        marks = np.arange(0.0, duration + 1.0e-9, 1.0 / frequency)
        clean = 0.2 * np.sin(
            2.0 * math.pi * frequency * times + 0.3)
        clean_join = JoinDiscontinuityAnalyzer(
            clean, self.sample_rate, segments,
            splice_records=splices, target_pitchmarks=marks,
        ).analyze()["joins"][0]
        self.assertTrue(clean_join["content_pitch_synchronous"])
        self.assertGreaterEqual(clean_join["content_frame_samples"], 190)
        self.assertGreater(clean_join["content_retention_ratio"], 0.80)
        self.assertNotIn("CONTENT_DROPOUT", clean_join["issues"])

        notched = clean.copy()
        notched[np.abs(times - splice_time) < 0.0125] = 0.0
        notched_join = JoinDiscontinuityAnalyzer(
            notched, self.sample_rate, segments,
            splice_records=splices, target_pitchmarks=marks,
        ).analyze()["joins"][0]
        self.assertLess(notched_join["content_retention_ratio"], 0.25)
        self.assertIn("CONTENT_DROPOUT", notched_join["issues"])

    def test_one_sided_pause_to_vowel_fade_is_not_a_dropout(self):
        samples = self.periodic()
        samples[self.times < self.splice_time] = 0.0
        _result, join = self.analyze(samples)
        self.assertLess(join["content_left_reference_rms"],
                        join["content_right_reference_rms"] * 0.1)
        self.assertFalse(join["content_dropout_eligible"])
        self.assertNotIn("CONTENT_DROPOUT", join["issues"])

    def test_expected_silence_context_keeps_raw_dropout_without_flag(self):
        for phone in ("pau", "cl", "k", "brth"):
            with self.subTest(phone=phone):
                _result, join = self.analyze_sustained_join(
                    0.650, notch_ms=8.0, phone=phone)

                # The raw measurement must remain visible for auditing even
                # when the label makes a low-energy closure legitimate.
                self.assertLess(join["content_retention_ratio"], 0.10)
                self.assertGreater(join["content_attenuation_db"], 12.0)
                self.assertTrue(join["content_dropout_expected"])
                self.assertIn("CONTENT_DROPOUT",
                              join["severity_components"])
                self.assertEqual(
                    join["severity_components"]["CONTENT_DROPOUT"]
                    ["weighted_score"], 0.0)
                self.assertNotIn("CONTENT_DROPOUT", join["issues"])

    def test_quarter_cycle_phase_shift_is_not_hidden_by_best_alignment(self):
        samples = self.periodic()
        right = self.times >= self.splice_time
        samples[right] = 0.2 * np.sin(
            2.0 * math.pi * 200.0 * self.times[right] + 0.3 + math.pi / 2.0)
        _result, join = self.analyze(samples)
        self.assertLess(abs(join["level_step_db"]), 0.05)
        self.assertLess(abs(join["zero_lag_period_correlation"]), 0.1)
        self.assertGreater(join["best_lag_period_correlation"], 0.98)
        self.assertGreater(join["phase_mismatch"], 0.9)
        self.assertLess(join["spectral_step"], 0.01)
        self.assertIn("PHASE_MISMATCH", join["issues"])

    def test_equal_rms_harmonic_change_hits_shape_and_spectrum(self):
        mixture = np.where(self.times < self.splice_time, 0.0, 0.8)
        _result, join = self.analyze(self.harmonic_signal(mixture))
        self.assertLess(abs(join["level_step_db"]), 0.05)
        self.assertLess(abs(join["f0_step_semitones"]), 0.05)
        self.assertGreater(join["period_shape_mismatch"], 0.18)
        self.assertGreater(join["spectral_step"], 0.04)
        self.assertTrue({"PERIOD_SHAPE_MISMATCH", "SPECTRAL_STEP"}
                        & set(join["issues"]))

    def test_six_db_gain_step_is_reported_as_level(self):
        samples = self.periodic()
        samples[self.times >= self.splice_time] *= 2.0
        _result, join = self.analyze(samples)
        self.assertAlmostEqual(join["level_step_db"],
                               20.0 * math.log10(2.0), places=3)
        self.assertIn("LEVEL_STEP", join["issues"])

    def test_extreme_local_novelty_just_below_raw_level_reference_is_ranked(self):
        samples = self.periodic()
        ratio = 10.0 ** (2.9 / 20.0)
        samples[self.times >= self.splice_time] *= ratio
        _result, join = self.analyze(samples)
        component = join["severity_components"]["LEVEL_STEP"]
        self.assertLess(abs(join["level_step_db"]), 3.0)
        self.assertGreater(join["level_step_novelty"], 10.0)
        self.assertGreater(component["locally_unusual_score"], 1.0)
        self.assertGreater(component["weighted_score"], 1.0)
        self.assertIn("LEVEL_STEP", join["issues"])

    def test_large_level_ratio_near_silence_is_raw_but_not_severe(self):
        samples = self.periodic(amplitude=1.0e-5)
        samples[self.times >= self.splice_time] *= 20.0
        _result, join = self.analyze(samples)
        component = join["severity_components"]["LEVEL_STEP"]
        self.assertGreater(abs(join["level_step_db"]), 20.0)
        self.assertLess(join["level_energy_gate"], 0.05)
        self.assertLess(component["weighted_score"], 0.1)
        self.assertNotIn("LEVEL_STEP", join["issues"])

    def test_sample_offset_hits_immediate_discontinuity(self):
        samples = self.periodic()
        samples[self.times >= self.splice_time] += 0.15
        _result, join = self.analyze(samples)
        self.assertGreater(join["sample_value_jump"], 0.1)
        self.assertGreater(join["sample_jump_novelty"], 5.0)
        self.assertGreater(join["slope_jump_novelty"], 5.0)
        self.assertIn("SAMPLE_DISCONTINUITY", join["issues"])

    def test_tiny_jump_in_quiet_closure_cannot_explode_relative_score(self):
        samples = self.periodic()
        sample = int(round(self.splice_time * self.sample_rate))
        collar = int(round(0.008 * self.sample_rate))
        samples[sample - collar:sample + collar] = 0.0
        samples[sample:sample + collar] = 0.002
        segments = [dict(row) for row in self.segments]
        segments[1]["phone"] = "b"
        result = JoinDiscontinuityAnalyzer(
            samples, self.sample_rate, segments,
            splice_records=self.splice, target_pitchmarks=self.marks_200,
        ).analyze()
        join = result["joins"][0]

        self.assertGreater(join["sample_value_jump"], 0.001)
        self.assertLess(join["sample_value_jump"], 0.003)
        component = join["severity_components"]["SAMPLE_DISCONTINUITY"]
        self.assertLess(component["absolute_click_gate"], 0.01)
        self.assertNotIn("SAMPLE_DISCONTINUITY", join["issues"])

    def test_displaced_flat_spectrum_impulse_is_found_inside_handoff(self):
        samples = self.periodic()
        event_sample = int(round(
            (self.splice_time + 0.0015) * self.sample_rate))
        samples[event_sample] += 1.0
        splice = [{
            "segment_index": 1,
            "time": self.splice_time,
            "handoff_start": self.splice_time - 0.003,
            "handoff_end": self.splice_time + 0.003,
            "position_source": "synthetic-exact",
            "estimated": False,
        }]
        result = JoinDiscontinuityAnalyzer(
            samples, self.sample_rate, self.segments,
            splice_records=splice, target_pitchmarks=self.marks_200,
        ).analyze()
        join = result["joins"][0]

        # The nominal splice itself is continuous; the new collar scan must
        # locate the displaced full-band impulse instead.
        self.assertLess(join["sample_value_jump"], 0.05)
        self.assertIn("BROADBAND_IMPULSE", join["issues"])
        self.assertEqual(join["dominant_issue"], "BROADBAND_IMPULSE")
        self.assertFalse(join["broadband_context_may_be_expected"])
        self.assertEqual(join["broadband_context_interpretation"],
                         "UNEXPECTED_BROADBAND_EVENT")
        self.assertGreater(join["broadband_impulse_score"], 0.34)
        self.assertGreater(join["broadband_impulse_novelty"], 4.0)
        self.assertGreater(join["broadband_band_spectral_flatness"], 0.42)
        self.assertLess(abs(join["broadband_spectral_tilt_db_per_octave"]),
                        12.0)
        self.assertLessEqual(
            abs(join["broadband_impulse_sample"] - event_sample),
            int(round(0.001 * self.sample_rate)))

    def test_stop_release_context_is_annotated_without_hiding_raw_event(self):
        samples = self.periodic()
        event_sample = int(round(
            (self.splice_time + 0.001) * self.sample_rate))
        samples[event_sample] += 1.0
        segments = [dict(row) for row in self.segments]
        segments[1]["phone"] = "k"
        result = JoinDiscontinuityAnalyzer(
            samples, self.sample_rate, segments,
            splice_records=self.splice, target_pitchmarks=self.marks_200,
        ).analyze()
        join = result["joins"][0]

        self.assertIn("BROADBAND_IMPULSE", join["issues"])
        self.assertTrue(join["broadband_context_may_be_expected"])
        self.assertEqual(join["broadband_context_interpretation"],
                         "EXPECTED_BURST_CONTEXT_REVIEW")

    def test_sustained_high_frequency_frication_is_not_a_crackle(self):
        noise = np.random.default_rng(73).normal(0.0, 1.0, len(self.times) + 1)
        samples = np.diff(noise)
        samples *= 0.08 / float(np.std(samples))
        _result, join = self.analyze(samples, [])

        self.assertIn(join["voicing"], {"unvoiced", "mixed"})
        self.assertNotIn("BROADBAND_IMPULSE", join["issues"])
        self.assertLess(join["severity_components"]["BROADBAND_IMPULSE"]
                        ["weighted_score"], 2.5)

    def test_steeply_tilted_speech_novelty_is_not_broadband(self):
        join, _event_sample = self.analyze_tilted_speech_transition()

        # This ordinary envelope transition has extremely novel floor energy,
        # but its steep tilt and non-flat bands are unlike a full-band click.
        self.assertGreater(join["broadband_impulse_novelty"], 12.0)
        self.assertGreater(
            abs(join["broadband_spectral_tilt_db_per_octave"]), 12.0)
        self.assertLess(join["broadband_absolute_shape_score"], 0.05)
        self.assertLess(join["broadband_relative_shape_score"], 0.05)
        self.assertNotIn("BROADBAND_IMPULSE", join["issues"])
        self.assertEqual(
            join["severity_components"]["BROADBAND_IMPULSE"]
            ["weighted_score"], 0.0)

    def test_flat_spectrum_click_survives_spectral_shape_gate(self):
        join, event_sample = self.analyze_tilted_speech_transition(
            click_amplitude=0.5)

        self.assertGreater(join["broadband_absolute_shape_score"], 0.8)
        self.assertGreater(join["broadband_relative_shape_score"], 0.8)
        self.assertLess(
            abs(join["broadband_spectral_tilt_db_per_octave"]), 3.0)
        self.assertIn("BROADBAND_IMPULSE", join["issues"])
        self.assertLessEqual(
            abs(join["broadband_impulse_sample"] - event_sample),
            int(round(0.001 * self.sample_rate)))

    def test_pitch_period_change_hits_f0_metric(self):
        samples = self.periodic()
        right = self.times >= self.splice_time
        phase_at_splice = 2.0 * math.pi * 200.0 * self.splice_time + 0.3
        samples[right] = 0.2 * np.sin(
            2.0 * math.pi * 240.0 *
            (self.times[right] - self.splice_time) + phase_at_splice)
        marks = np.concatenate((
            np.arange(0.0, self.splice_time + 1.0e-9, 1.0 / 200.0),
            self.splice_time + np.arange(
                1, int((self.duration - self.splice_time) * 240.0) + 1)
            / 240.0,
        ))
        _result, join = self.analyze(samples, marks)
        self.assertGreater(abs(join["f0_step_semitones"]), 2.0)
        self.assertLess(join["spectral_step"], 0.01)
        self.assertIn("F0_STEP", join["issues"])

    def test_smooth_spectral_trajectory_has_low_break(self):
        mixture = np.linspace(0.0, 0.8, len(self.times))
        _result, join = self.analyze(self.harmonic_signal(mixture))
        self.assertEqual(join["dominant_issue"], "OK")
        self.assertLess(join["spectral_step"], 0.01)
        self.assertLess(join["spectral_slope_break"], 0.01)
        self.assertLess(join["spectral_slope_break_novelty"], 1.0)

    def test_continuous_value_with_spectral_slope_break_is_flagged(self):
        mixture = 0.4 + np.where(
            self.times < self.splice_time,
            5.0 * (self.times - self.splice_time),
            -5.0 * (self.times - self.splice_time))
        mixture = np.clip(mixture, 0.0, 0.8)
        _result, join = self.analyze(self.harmonic_signal(mixture))
        self.assertLess(join["spectral_step"], 0.01)
        self.assertGreater(join["spectral_slope_break"], 0.002)
        self.assertGreater(join["spectral_slope_break_novelty"], 4.0)
        self.assertEqual(join["dominant_issue"],
                         "SPECTRAL_TRAJECTORY_BREAK")

    def test_abrupt_spectral_trajectory_is_flagged(self):
        mixture = np.where(self.times < self.splice_time, 0.0, 0.8)
        _result, join = self.analyze(self.harmonic_signal(mixture))
        self.assertGreater(join["spectral_step"], 0.04)
        self.assertGreater(join["spectral_step_novelty"], 4.0)
        self.assertIn("SPECTRAL_STEP", join["issues"])

    def test_stationary_unvoiced_noise_skips_period_metrics(self):
        samples = np.random.default_rng(41).normal(
            0.0, 0.08, len(self.times))
        _result, join = self.analyze(samples, [])
        self.assertEqual(join["voicing"], "unvoiced")
        self.assertIsNone(join["left_f0_hz"])
        self.assertIsNone(join["zero_lag_period_correlation"])
        self.assertIsNone(join["period_shape_mismatch"])
        self.assertEqual(join["dominant_issue"], "OK")

    def test_phone_prior_does_not_force_noise_like_t_to_voiced(self):
        samples = np.random.default_rng(17).normal(
            0.0, 0.06, len(self.times))
        segments = [dict(row) for row in self.segments]
        segments[1]["phone"] = "t"
        result = JoinDiscontinuityAnalyzer(
            samples, self.sample_rate, segments,
            splice_records=self.splice,
            target_pitchmarks=self.marks_200).analyze()
        join = result["joins"][0]
        self.assertIn(join["voicing"], {"unvoiced", "mixed"})
        self.assertFalse(join["left_voiced_eligible"] and
                         join["right_voiced_eligible"])
        self.assertIsNone(join["left_f0_hz"])
        self.assertIsNone(join["phase_mismatch"])

    def test_acoustically_voiced_z_can_overcome_soft_phone_prior(self):
        segments = [dict(row) for row in self.segments]
        segments[1]["phone"] = "z"
        result = JoinDiscontinuityAnalyzer(
            self.periodic(), self.sample_rate, segments,
            splice_records=self.splice,
            target_pitchmarks=self.marks_200).analyze()
        join = result["joins"][0]
        self.assertEqual(join["voicing"], "voiced")
        self.assertGreater(join["left_voicing_confidence"], 0.6)
        self.assertGreater(join["right_voicing_confidence"], 0.6)
        self.assertIsNotNone(join["left_f0_hz"])

    def test_unvoiced_spectral_change_is_flagged(self):
        samples = np.random.default_rng(41).normal(
            0.0, 0.08, len(self.times))
        split = int(round(self.splice_time * self.sample_rate))
        kernel = np.ones(31, dtype=np.float64) / 31.0
        filtered = np.convolve(samples[split:], kernel, mode="same")
        filtered *= 0.08 / max(1.0e-9, float(np.std(filtered)))
        samples[split:] = filtered
        _result, join = self.analyze(samples, [])
        self.assertEqual(join["voicing"], "unvoiced")
        self.assertGreater(join["spectral_step_novelty"], 4.0)
        self.assertIn("UNVOICED_SPECTRAL_BREAK", join["issues"])

    def test_stable_vowel_formants_are_measurable_and_continuous(self):
        samples = self.resonant_vowel((500.0, 1500.0, 2500.0, 3500.0))
        _result, join = self.analyze(samples)

        self.assertTrue(join["formants_available"])
        self.assertGreaterEqual(
            sum(track.get("available", False)
                for track in join["formant_tracks"]), 2)
        self.assertLess(join["formant_frequency_jump_normalized"], 0.03)
        self.assertNotIn("FORMANT_FREQUENCY_BREAK", join["issues"])
        self.assertNotIn("FORMANT_BALANCE_BREAK", join["issues"])

    def test_abrupt_formant_frequency_change_is_detected(self):
        left = self.resonant_vowel((500.0, 1500.0, 2500.0, 3500.0))
        right = self.resonant_vowel((800.0, 1800.0, 2500.0, 3500.0))
        samples = left.copy()
        samples[self.times >= self.splice_time] = right[
            self.times >= self.splice_time]
        _result, join = self.analyze(samples)

        self.assertTrue(join["formants_available"])
        self.assertGreater(join["formant_frequency_jump_normalized"], 0.12)
        self.assertIn("FORMANT_FREQUENCY_BREAK", join["issues"])

    def test_impossible_extrapolated_formant_cannot_create_severity(self):
        def observations(times, frequency_rows):
            rows = []
            for time, frequencies in zip(times, frequency_rows):
                rows.append({
                    "time": time,
                    "formants": [
                        {
                            "track_index": index,
                            "frequency_hz": frequency,
                            "bandwidth_hz": 90.0,
                            "prominence_db": 10.0,
                            "normalized_energy": 1.0 / (index + 1),
                            "confidence": 0.9,
                        }
                        for index, frequency in frequencies
                    ],
                })
            return rows

        left = observations(
            (0.37, 0.38, 0.39),
            (
                ((0, 500.0), (1, 1500.0), (3, 4000.0)),
                ((0, 500.0), (1, 1500.0), (3, 2000.0)),
                ((0, 500.0), (1, 1500.0), (3, 500.0)),
            ),
        )
        right = observations(
            (0.41, 0.42, 0.43),
            (
                ((0, 500.0), (1, 1500.0), (3, 4200.0)),
                ((0, 500.0), (1, 1500.0), (3, 4250.0)),
                ((0, 500.0), (1, 1500.0), (3, 4300.0)),
            ),
        )
        frames = [{"samples": np.zeros(256)} for _ in range(3)]
        with patch.object(
            jd, "_formant_observations", side_effect=(left, right)
        ):
            metrics = jd._formant_metrics(
                frames,
                frames,
                self.sample_rate,
                self.splice_time,
                1.0 / 200.0,
                JoinAnalysisConfig(),
                eligible=True,
            )

        self.assertTrue(metrics["available"])
        self.assertFalse(metrics["per_formant"][3]["available"])
        self.assertEqual(
            metrics["per_formant"][3]["reason"],
            "trajectory_extrapolation_out_of_range",
        )
        self.assertLess(metrics["frequency_jump_normalized"], 0.01)

    def test_formant_balance_change_does_not_pose_as_frequency_change(self):
        left = self.resonant_vowel(
            (500.0, 1500.0, 2500.0), gains=(1.0, 0.25, 0.1),
            bandwidths=(70.0, 90.0, 120.0), parallel=True)
        right = self.resonant_vowel(
            (500.0, 1500.0, 2500.0), gains=(0.12, 1.0, 0.3),
            bandwidths=(70.0, 90.0, 120.0), parallel=True)
        samples = left.copy()
        samples[self.times >= self.splice_time] = right[
            self.times >= self.splice_time]
        _result, join = self.analyze(samples)

        self.assertTrue(join["formants_available"])
        self.assertGreater(join["formant_balance_jump"], 0.35)
        self.assertIn("FORMANT_BALANCE_BREAK", join["issues"])
        self.assertNotIn("FORMANT_FREQUENCY_BREAK", join["issues"])

    def test_formants_are_explicitly_unavailable_for_unvoiced_join(self):
        samples = np.random.default_rng(91).normal(
            0.0, 0.06, len(self.times))
        _result, join = self.analyze(samples, [])

        self.assertFalse(join["formants_available"])
        self.assertEqual(join["formants_unavailable_reason"],
                         "formants_require_voiced_context")
        self.assertIsNone(join["formant_frequency_jump_normalized"])

    def test_near_silence_cannot_create_formant_severity(self):
        samples = self.resonant_vowel((500.0, 1500.0, 2500.0)) * 1.0e-5
        _result, join = self.analyze(samples)

        self.assertFalse(join["formants_available"])
        self.assertNotIn("FORMANT_FREQUENCY_BREAK", join["issues"])
        self.assertNotIn("FORMANT_BALANCE_BREAK", join["issues"])

    def test_output_is_strict_json_and_preserves_exact_splice(self):
        result, join = self.analyze(self.periodic())
        serialised = json.dumps(result, sort_keys=True, allow_nan=False)
        repeated, _repeated_join = self.analyze(self.periodic())
        self.assertEqual(
            serialised,
            json.dumps(repeated, sort_keys=True, allow_nan=False))
        self.assertEqual(join["splice_sample"],
                         int(self.splice_time * self.sample_rate))
        self.assertEqual(join["position_source"], "synthetic-exact")
        self.assertFalse(join["position_estimated"])
        self.assertEqual(result["summary"]["exact_splice_count"], 1)
        self.assertEqual(result["schema_version"], 6)
        self.assertEqual(join["phone_context_string"], "a v b")
        self.assertIn("formants_available", join)
        self.assertIn("severity_weights", result)
        self.assertIn("above_analysis_floor", join)
        self.assertNotIn("audible", join)
        self.assertFalse(result["ranking_is_calibrated"])
        self.assertIn("severity_components", join)

    def test_near_boundary_reports_insufficient_context_without_nan(self):
        splice = [{
            "segment_index": 1, "time": 0.0002,
            "position_source": "synthetic-edge", "estimated": False,
        }]
        result = JoinDiscontinuityAnalyzer(
            self.periodic(), self.sample_rate, self.segments,
            splice_records=splice, target_pitchmarks=[]).analyze()
        self.assertEqual(result["joins"][0]["dominant_issue"],
                         "INSUFFICIENT_CONTEXT")
        json.dumps(result, allow_nan=False)

    def test_thresholds_and_weights_are_configurable(self):
        config = JoinAnalysisConfig(
            level_step_db=20.0,
            severity_weights=(("LEVEL_STEP", 0.1),
                              ("SAMPLE_DISCONTINUITY", 1.0)),
        )
        samples = self.periodic()
        samples[self.times >= self.splice_time] *= 2.0
        _result, join = self.analyze(samples, config=config)
        self.assertNotEqual(join["dominant_issue"], "LEVEL_STEP")


if __name__ == "__main__":
    unittest.main()
