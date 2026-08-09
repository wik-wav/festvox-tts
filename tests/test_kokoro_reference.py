import gzip
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np

from kokoro_reference import (
    KOKORO_ALIGN_METHOD,
    KokoroAlignCheckpoint,
    KokoroRecord,
    SilverPhoneAlignment,
    SilverUtteranceAlignment,
    _ctc_viterbi_path,
    _kokoro_mfcc,
    align_kokoro_record,
    canonical_phone,
    inventory_kokoro_prefix,
    parse_metadata_text,
    partition_for_id,
    refine_phrase_pauses,
    safe_extract_kokoro_archive,
    select_stratified_records,
)


def _nested_archive(path, members):
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        for name, payload, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = payload.decode("utf-8")
                archive.addfile(info)
            else:
                raise AssertionError(kind)
    first = gzip.compress(tar_buffer.getvalue(), mtime=0)
    path.write_bytes(gzip.compress(first, mtime=0))


class KokoroReferenceTests(unittest.TestCase):
    def test_metadata_and_partitions_are_deterministic(self):
        text = (
            "book-0001|\u8cea\u554f\u3067\u3059\u304b|k o r e _ w a _ sh i ts u m o N _ d e s u ?\n"
            "book-0002|\u304d\u3063\u3068|k i cl t o\n"
            "book-0003|\u3059\u3053\u3057|s u k o sh i\n"
        )
        first = parse_metadata_text(text)
        second = parse_metadata_text(text)

        self.assertEqual(first, second)
        self.assertEqual(first[0].partition, partition_for_id("book-0001"))
        self.assertIn("interrogative", first[0].strata)
        self.assertIn("geminate", first[1].strata)
        self.assertIn("devoicing_context", first[2].strata)

    def test_stratified_selection_is_stable_and_partitioned(self):
        records = tuple(
            KokoroRecord(
                f"book-{index:04d}", "text", "k a N _ s u .",
                ("k", "a", "N", "_", "s", "u", "."),
                partition_for_id(f"book-{index:04d}"),
                ("moraic_nasal", "phrase_boundary", "vowel_a", "vowel_u"),
            )
            for index in range(240)
        )
        first = select_stratified_records(
            records, train_count=8, validation_count=3, test_count=3
        )
        second = select_stratified_records(
            tuple(reversed(records)),
            train_count=8, validation_count=3, test_count=3,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 14)
        self.assertEqual(sum(row.partition == "train" for row in first), 8)
        self.assertEqual(sum(row.partition == "validation" for row in first), 3)
        self.assertEqual(sum(row.partition == "test" for row in first), 3)

    def test_safe_extraction_rejects_escape_and_links(self):
        metadata = b"safe-0001|text|k a\n"
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            archive_path = root / "kokoro.tar.gz"
            _nested_archive(archive_path, [
                ("./metadata.csv", metadata, "file"),
                ("../escape.txt", b"bad", "file"),
                ("./linked", b"elsewhere", "symlink"),
                ("./wavs/safe-0001.flac", b"fixture", "file"),
            ])
            output = root / "out"

            report = safe_extract_kokoro_archive(
                archive_path, output, record_ids=("safe-0001",),
                archive_sha256="fixture-digest",
            )

            self.assertEqual(report["selected_record_count"], 1)
            self.assertEqual(report["archive"]["sha256"], "fixture-digest")
            self.assertTrue(report["archive"]["sha256_verified"])
            self.assertEqual((output / "wavs" / "safe-0001.flac").read_bytes(),
                             b"fixture")
            self.assertFalse((root / "escape.txt").exists())
            self.assertEqual(len(report["rejected_members"]), 2)
            saved = json.loads(
                (output / "extraction_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved, report)

    def test_bounded_inventory_keeps_archive_order_and_metadata(self):
        metadata = (
            b"safe-0001|one|k a\n"
            b"safe-0002|two|s u k o sh i\n"
        )
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            archive_path = root / "kokoro.tar.gz"
            _nested_archive(archive_path, [
                ("./metadata.csv", metadata, "file"),
                ("./wavs/safe-0002.flac", b"second", "file"),
                ("./wavs/safe-0001.flac", b"first", "file"),
            ])

            report = inventory_kokoro_prefix(
                archive_path, maximum_audio_members=2
            )

        self.assertEqual(
            [row["utterance_id"] for row in report["candidate_records"]],
            ["safe-0002", "safe-0001"],
        )
        self.assertEqual(report["candidate_record_count"], 2)
        self.assertEqual(
            report["metadata_sha256"],
            __import__("hashlib").sha256(metadata).hexdigest(),
        )

    def test_ctc_path_handles_repeated_labels(self):
        # Blank, a, b.  Repeated a labels require a separating blank.
        logits = np.full((8, 3), -12.0, dtype=np.float64)
        for frame, label in enumerate((0, 1, 1, 0, 1, 0, 2, 0)):
            logits[frame, label] = 0.0
        states, frames = _ctc_viterbi_path(logits, (1, 1, 2))

        self.assertEqual(states.shape, (8,))
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(frame.size for frame in frames))
        self.assertLess(frames[0][-1], frames[1][0])
        self.assertLess(frames[1][-1], frames[2][0])

    def test_numpy_mfcc_has_official_shape_and_is_deterministic(self):
        sample_rate = 22050
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        samples = 0.2 * np.sin(2.0 * np.pi * 173.0 * time)

        first = _kokoro_mfcc(samples, sample_rate)
        second = _kokoro_mfcc(samples, sample_rate)

        self.assertEqual(first.shape[1], 40)
        self.assertGreater(first.shape[0], 80)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(np.isfinite(first)))

    def test_acoustic_alignment_preserves_all_tokens_and_flags_devoicing(self):
        sample_rate = 16000
        duration = 0.9
        time = np.arange(int(duration * sample_rate)) / sample_rate
        envelope = np.sin(np.pi * np.clip(time / duration, 0.0, 1.0)) ** 2
        samples = 0.18 * envelope * np.sin(2 * np.pi * 170 * time)
        record = KokoroRecord(
            "fixture", "\u3059\u3053\u3057", "s u k o sh i .",
            ("s", "u", "k", "o", "sh", "i", "."), "test",
        )

        result = align_kokoro_record(record, samples, sample_rate)

        self.assertEqual(len(result.phones), len(record.phones))
        self.assertTrue(all(
            left.end_seconds <= right.end_seconds
            for left, right in zip(result.phones, result.phones[1:])
        ))
        self.assertTrue(result.phones[1].probable_devoicing)
        self.assertTrue(result.phones[5].probable_devoicing)
        self.assertIn("silver references", " ".join(result.diagnostics))

    def test_long_vowels_and_unknown_labels_remain_representable(self):
        self.assertEqual(canonical_phone("a:"), ("a", True))
        self.assertEqual(canonical_phone("N"), ("N", False))
        self.assertEqual(canonical_phone("_"), ("sp", False))
        self.assertEqual(canonical_phone("alien"), ("alien", False))

    def test_word_separator_does_not_become_a_spoken_phone_or_phrase(self):
        sample_rate = 16000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        samples = 0.16 * np.sin(2 * np.pi * 170 * time)
        record = KokoroRecord(
            "separator", "かな は", "k a _ h a .",
            ("k", "a", "_", "h", "a", "."), "test",
        )

        result = align_kokoro_record(record, samples, sample_rate)

        separator = result.phones[2]
        self.assertEqual(separator.phone, "sp")
        self.assertEqual(separator.phrase_index,
                         result.phones[1].phrase_index)
        self.assertLess(separator.duration_seconds, 0.04)
        self.assertEqual(result.phones[-1].phone, "pau")

    def test_punctuation_pause_refinement_absorbs_the_full_silence(self):
        sample_rate = 16000
        tone_time = np.arange(int(0.25 * sample_rate)) / sample_rate
        left = 0.18 * np.sin(2.0 * np.pi * 170.0 * tone_time)
        silence = np.zeros(int(0.40 * sample_rate), dtype=np.float64)
        right = 0.18 * np.sin(2.0 * np.pi * 190.0 * tone_time)
        samples = np.concatenate((left, silence, right))

        def phone(index, raw, canonical, start, end, phrase):
            return SilverPhoneAlignment(
                index=index,
                raw_phone=raw,
                phone=canonical,
                start_seconds=start,
                end_seconds=end,
                confidence=0.70,
                boundary_confidence_left=0.60,
                boundary_confidence_right=0.60,
                mora_index=index,
                phrase_index=phrase,
            )

        # Deliberately leave much of the true 0.25-0.65 s silence attached
        # to the vowels around the punctuation token.
        alignment = SilverUtteranceAlignment(
            utterance_id="pause-refinement",
            sample_rate=sample_rate,
            sample_count=len(samples),
            phones=(
                phone(0, "a", "a", 0.00, 0.42, 0),
                phone(1, ".", "pau", 0.42, 0.50, 0),
                phone(2, "a", "a", 0.50, 0.90, 1),
            ),
            confidence=0.70,
            accepted=True,
        )

        refined = refine_phrase_pauses(alignment, samples, sample_rate)
        pause = refined.phones[1]

        self.assertLess(abs(pause.start_seconds - 0.25), 0.035)
        self.assertLess(abs(pause.end_seconds - 0.65), 0.035)
        self.assertEqual(refined.phones[0].end_seconds,
                         pause.start_seconds)
        self.assertEqual(refined.phones[2].start_seconds,
                         pause.end_seconds)
        self.assertTrue(any(
            "punctuation_pause_energy_v2" in diagnostic
            for diagnostic in refined.diagnostics
        ))

    @unittest.skipUnless(
        os.environ.get("KOKORO_ALIGN_CHECKPOINT"),
        "set KOKORO_ALIGN_CHECKPOINT for the optional real-model test",
    )
    def test_optional_checkpoint_loads_in_weights_only_mode(self):
        checkpoint = KokoroAlignCheckpoint(
            os.environ["KOKORO_ALIGN_CHECKPOINT"]
        )
        model = checkpoint._load_model()
        self.assertEqual(model.dense.out_features, 39)
        self.assertTrue(checkpoint.sha256)
        self.assertEqual(KOKORO_ALIGN_METHOD,
                         "kokoro_align_ctc_20221201_acoustic_refinement_v1")


if __name__ == "__main__":
    unittest.main()
