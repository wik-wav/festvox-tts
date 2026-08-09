import math
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from formant_analysis import (
    AnalysisSegment,
    FormantAnalysisConfig,
    FormantSegment,
    analyze_frame,
    analyze_segment,
    derive_reference_voice_space,
    estimate_f0,
    read_audio,
    smooth_formant_track,
    supplied_reference_segments,
    true_envelope,
)


def _resonant_vowel(formants=(700.0, 1200.0, 2500.0, 3500.0),
                    f0=160.0, seconds=0.55, sample_rate=16000):
    count = int(seconds * sample_rate)
    excitation = np.zeros(count, dtype=np.float64)
    excitation[::max(1, int(round(sample_rate / f0)))] = 1.0
    values = excitation
    for index, formant in enumerate(formants):
        bandwidth = 70.0 + index * 35.0
        radius = math.exp(-math.pi * bandwidth / sample_rate)
        coefficient = 2.0 * radius * math.cos(
            2.0 * math.pi * formant / sample_rate
        )
        output = np.zeros_like(values)
        for sample in range(values.size):
            output[sample] = values[sample]
            if sample:
                output[sample] += coefficient * output[sample - 1]
            if sample > 1:
                output[sample] -= radius * radius * output[sample - 2]
        scale = np.max(np.abs(output))
        values = output / max(scale, 1e-9)
    ramp = min(count // 6, int(0.04 * sample_rate))
    if ramp:
        values[:ramp] *= np.linspace(0.0, 1.0, ramp)
        values[-ramp:] *= np.linspace(1.0, 0.0, ramp)
    return np.asarray(values * 0.25, dtype=np.float64), sample_rate


def _segment(identifier="a", *, path=Path("fixture.wav"), vowel="a"):
    return AnalysisSegment(
        segment_id=identifier,
        speaker_id="speaker",
        audio_path=path,
        start_seconds=0.04,
        end_seconds=0.50,
        vowel=vowel,
        phone=vowel,
        source_corpus="synthetic",
    )


class FormantAnalysisTests(unittest.TestCase):
    def test_temporal_tracker_rejects_harmonic_or_shoulder_jump(self):
        values = [505.0, 510.0, 515.0, 1180.0, 520.0, 525.0, 530.0]
        tracked, confidence, outliers = smooth_formant_track(
            values, [0.8] * len(values)
        )
        self.assertTrue(outliers[3])
        self.assertLess(tracked[3], 600.0)
        self.assertLess(confidence[3], confidence[2])
        self.assertLess(max(abs(tracked[i + 1] - tracked[i])
                            for i in range(len(tracked) - 1)), 35.0)

    def test_temporal_tracker_preserves_gradual_vowel_motion(self):
        values = [500.0, 520.0, 545.0, 570.0, 600.0, 625.0, 650.0]
        tracked, _confidence, outliers = smooth_formant_track(values)
        self.assertFalse(any(outliers))
        self.assertGreater(tracked[-1] - tracked[0], 100.0)
        self.assertTrue(all(tracked[index] < tracked[index + 1]
                            for index in range(len(tracked) - 1)))

    def test_f0_estimator_finds_periodic_source(self):
        samples, sample_rate = _resonant_vowel(f0=173.0)
        center = samples[int(0.12 * sample_rate):int(0.24 * sample_rate)]

        f0, confidence, ambiguity = estimate_f0(
            center, sample_rate, minimum_hz=70.0, maximum_hz=350.0
        )

        self.assertIsNotNone(f0)
        self.assertAlmostEqual(f0, 173.0, delta=6.0)
        self.assertGreater(confidence, 0.55)
        self.assertLess(ambiguity, 0.80)

    def test_true_envelope_is_f0_adaptive_and_finite(self):
        low, sample_rate = _resonant_vowel(f0=100.0)
        high, _ = _resonant_vowel(f0=330.0)

        low_frequencies, low_envelope, low_order, _ = true_envelope(
            low[1200:2400], sample_rate, 100.0
        )
        high_frequencies, high_envelope, high_order, _ = true_envelope(
            high[1200:2400], sample_rate, 330.0
        )

        self.assertEqual(low_frequencies.shape, low_envelope.shape)
        self.assertEqual(high_frequencies.shape, high_envelope.shape)
        self.assertTrue(np.all(np.isfinite(low_envelope)))
        self.assertTrue(np.all(np.isfinite(high_envelope)))
        self.assertLess(high_order, low_order)

    def test_frame_retains_independent_estimates_and_reasons(self):
        samples, sample_rate = _resonant_vowel()
        frame = samples[int(0.16 * sample_rate):int(0.23 * sample_rate)]
        result = analyze_frame(
            frame, sample_rate, segment=_segment(), frame_time_seconds=0.2
        )

        self.assertEqual(len(result.formants_hz), 4)
        self.assertEqual(len(result.lpc_formants_hz), 4)
        self.assertEqual(len(result.estimator_disagreement_hz), 4)
        self.assertEqual(len(result.envelope_frequencies_hz),
                         len(result.envelope_db))
        self.assertTrue(np.isfinite(result.frame_power_db))
        self.assertIsInstance(result.rejection_reasons, list)
        expected = (700.0, 1200.0)
        for measured, target in zip(result.formants_hz[:2], expected):
            self.assertIsNotNone(measured)
            self.assertAlmostEqual(measured, target, delta=500.0)
        self.assertTrue(any(
            measured is not None and measured > 2300.0
            for measured in result.formants_hz[2:]
        ))

    def test_unvoiced_frame_is_preserved_but_rejected(self):
        noise = np.random.default_rng(17).normal(0.0, 0.03, 1200)
        segment = _segment("devoiced")
        result = analyze_frame(
            noise, 16000, segment=segment, frame_time_seconds=0.1
        )

        self.assertFalse(result.accepted)
        self.assertTrue(any(
            reason in result.rejection_reasons
            for reason in ("f0_unavailable", "devoiced_or_unreliably_voiced")
        ))

    def test_segment_uses_stable_body_and_is_deterministic(self):
        samples, sample_rate = _resonant_vowel()
        from formant_analysis import AudioData
        audio = AudioData(samples, sample_rate, 1, Path("fixture.wav"))
        segment = _segment()

        first = analyze_segment(segment, audio=audio)
        second = analyze_segment(segment, audio=audio)

        self.assertEqual(first.to_row(), second.to_row())
        self.assertEqual([row.to_row() for row in first.frames],
                         [row.to_row() for row in second.frames])
        self.assertGreater(first.stable_start_seconds, segment.start_seconds)
        self.assertLess(first.stable_end_seconds, segment.end_seconds)
        self.assertGreater(len(first.frames), 4)

    def test_pcm_wav_reader_preserves_rate_and_mono_data(self):
        samples, sample_rate = _resonant_vowel(seconds=0.1)
        pcm = np.asarray(np.clip(samples, -1.0, 1.0) * 32767, dtype="<i2")
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "fixture.wav"
            with wave.open(str(path), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(sample_rate)
                target.writeframes(pcm.tobytes())
            decoded = read_audio(path)

        self.assertEqual(decoded.sample_rate, sample_rate)
        self.assertEqual(decoded.channels, 1)
        self.assertEqual(decoded.samples.shape, samples.shape)
        self.assertLess(float(np.max(np.abs(decoded.samples - samples))), 1e-4)

    def test_supplied_formant_shift_references_are_labelled_e(self):
        samples, sample_rate = _resonant_vowel(seconds=0.1)
        pcm = np.asarray(np.clip(samples, -1.0, 1.0) * 32767, dtype="<i2")
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "neutral.wav"
            with wave.open(str(path), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(sample_rate)
                target.writeframes(pcm.tobytes())
            segments = supplied_reference_segments(root)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].vowel, "e")
        self.assertEqual(segments[0].phone, "e")
        self.assertEqual(segments[0].metadata["reference_vowel"], "e")

    def test_range_derivation_is_robust_and_byte_stable(self):
        def analyzed(identifier, speaker, vowel, dispersion):
            base = _segment(identifier, vowel=vowel)
            base = AnalysisSegment(**{
                **base.__dict__, "speaker_id": speaker,
                "source_corpus": "source" if speaker == "source" else "ref",
            })
            return FormantSegment(
                segment=base,
                frames=[],
                accepted_frame_count=8,
                rejected_frame_count=1,
                stable_start_seconds=0.1,
                stable_end_seconds=0.4,
                median_f0_hz=160.0,
                median_formants_hz=[500.0, 1500.0, 2500.0, 3500.0],
                median_bandwidths_hz=[70.0, 90.0, 130.0, 180.0],
                median_formant_confidences=[0.8, 0.8, 0.7, 0.6],
                median_formant_dispersion_hz=dispersion,
                apparent_vocal_tract_length_cm=34300.0 / (2.0 * dispersion),
                median_spectral_tilt_db_per_octave=-8.0,
                accepted=True,
                rejection_reasons=[],
            )

        rows = []
        for vowel_index, vowel in enumerate("aiueo"):
            rows.extend([
                analyzed(f"source-{vowel}", "source", vowel,
                         1000.0 + vowel_index * 15.0),
                analyzed(f"low-{vowel}", "reference-low", vowel,
                         900.0 + vowel_index * 15.0),
                analyzed(f"high-{vowel}", "reference-high", vowel,
                         1120.0 + vowel_index * 15.0),
            ])
        hashes = {"fixture.wav": "abc123"}

        first = derive_reference_voice_space(
            rows, source_speaker_id="source", reference_hashes=hashes,
        )
        second = derive_reference_voice_space(
            list(reversed(rows)), source_speaker_id="source",
            reference_hashes=hashes,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["identity_vocal_tract_ratio"], 1.0)
        self.assertLess(first["realistic_min_ratio"], 1.0)
        self.assertGreater(first["realistic_max_ratio"], 1.0)
        self.assertLessEqual(first["expanded_min_ratio"],
                             first["realistic_min_ratio"])
        self.assertGreaterEqual(first["expanded_max_ratio"],
                                first["realistic_max_ratio"])
        self.assertEqual(set(first["per_vowel_limits"]), set("aiueo"))

    def test_supplied_reference_ratios_are_kept_on_e_vowel(self) -> None:
        def analyzed(segment_id: str, speaker_id: str, vowel: str,
                     dispersion: float, corpus: str) -> FormantSegment:
            return FormantSegment(
                segment=AnalysisSegment(
                    segment_id=segment_id,
                    speaker_id=speaker_id,
                    vowel=vowel,
                    audio_path="fixture.wav",
                    start_seconds=0.0,
                    end_seconds=0.1,
                    source_corpus=corpus,
                ),
                frames=[],
                accepted_frame_count=1,
                rejected_frame_count=0,
                stable_start_seconds=0.02,
                stable_end_seconds=0.08,
                median_f0_hz=170.0,
                median_formants_hz=[500.0, 1500.0, 2500.0, 3500.0],
                median_bandwidths_hz=[70.0, 90.0, 130.0, 180.0],
                median_formant_confidences=[0.8, 0.8, 0.7, 0.6],
                median_formant_dispersion_hz=dispersion,
                apparent_vocal_tract_length_cm=34300.0 / (2.0 * dispersion),
                median_spectral_tilt_db_per_octave=-8.0,
                accepted=True,
                rejection_reasons=[],
            )

        rows = [
            analyzed(f"source-{vowel}", "source", vowel, 1000.0,
                     "source_voicebank")
            for vowel in "aiueo"
        ]
        rows.extend([
            analyzed("neutral-e", "provided-neutral", "e", 1000.0,
                     "prompt20_supplied_references"),
            analyzed("high-e", "provided-high", "e", 900.0,
                     "prompt20_supplied_references"),
        ])

        result = derive_reference_voice_space(rows, source_speaker_id="source")

        self.assertEqual(
            result["per_vowel_limits"]["e"]["supplied_reference_ratio_count"],
            2,
        )
        for vowel in "aiuo":
            self.assertEqual(
                result["per_vowel_limits"][vowel][
                    "supplied_reference_ratio_count"
                ],
                0,
            )


if __name__ == "__main__":
    unittest.main()
