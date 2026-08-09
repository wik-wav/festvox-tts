import json
from pathlib import Path
import tempfile
import unittest

from japanese_duration import load_duration_priors
from japanese_duration_corpus import (
    CorpusUtterance,
    FIXED_VALIDATION_IDS,
    TimedPhone,
    align_phone_sequences,
    duration_metrics,
    evaluate_duration_model,
    fit_duration_priors,
    load_fixed_validation_ids,
    load_jsut,
    normalize_label_phone,
    parse_htk_lab,
    parse_htk_label_line,
    parse_textgrid,
    select_heldout_ids,
    write_priors,
)


def _phone(start_ms, duration_ms, phone, *, devoiced=False):
    start = int(round(start_ms * 10_000))
    end = int(round((start_ms + duration_ms) * 10_000))
    raw = phone.upper() if devoiced else phone
    return TimedPhone(start, end, phone, raw, devoiced)


def _utterance(identifier, vowel="a", *, long=False, devoiced=False,
               geminate=False, nasal=False):
    phones = [_phone(0, 50, "sil")]
    cursor = 50
    if geminate:
        phones.append(_phone(cursor, 75, "cl")); cursor += 75
    phones.append(_phone(cursor, 55, "k")); cursor += 55
    phones.append(_phone(cursor, 45 if devoiced else 92, vowel,
                         devoiced=devoiced)); cursor += 45 if devoiced else 92
    if long:
        phones.append(_phone(cursor, 64, vowel)); cursor += 64
    if nasal:
        phones.append(_phone(cursor, 88, "N")); cursor += 88
    phones.append(_phone(cursor, 60, "sil"))
    return CorpusUtterance(identifier, tuple(phones))


class JapaneseDurationCorpusTests(unittest.TestCase):
    def test_fixed_validation_ids_come_from_committed_manifest(self):
        self.assertEqual(load_fixed_validation_ids(), FIXED_VALIDATION_IDS)
        self.assertEqual(len(FIXED_VALIDATION_IDS), 6)
        self.assertIn("BASIC5000_0004", FIXED_VALIDATION_IDS)

    def test_htk_timestamp_and_full_context_phone_parsing(self):
        row = parse_htk_label_line(
            "800000 1100000 xx^i-cl+sh=u/A:0+1+2", line_number=4
        )

        self.assertEqual(row.phone, "cl")
        self.assertAlmostEqual(row.duration_ms, 30.0)
        self.assertAlmostEqual(row.start_seconds, 0.08)

    def test_uppercase_vowels_are_preserved_as_devoicing_annotations(self):
        self.assertEqual(normalize_label_phone("I"), ("i", True))
        self.assertEqual(normalize_label_phone("U"), ("u", True))
        self.assertEqual(normalize_label_phone("N"), ("N", False))

    def test_label_file_is_ordered_and_wav_is_optional(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "BASIC5000_0004.lab"
            path.write_text(
                "0 800000 i\n800000 1100000 cl\n"
                "1100000 2600000 sh\n2600000 3300000 u\n",
                encoding="utf-8",
            )
            utterance = parse_htk_lab(path)

        self.assertEqual(utterance.utterance_id, "BASIC5000_0004")
        self.assertEqual([row.phone for row in utterance.phones],
                         ["i", "cl", "sh", "u"])
        self.assertEqual([round(row.duration_ms) for row in utterance.phones],
                         [80, 30, 150, 70])

    def test_textgrid_parser_retains_csj_uppercase_devoicing(self):
        text = '''File type = "ooTextFile"
Object class = "TextGrid"
item [1]:
    class = "IntervalTier"
    name = "phone"
    intervals [1]:
        xmin = 0.0
        xmax = 0.05
        text = "k"
    intervals [2]:
        xmin = 0.05
        xmax = 0.09
        text = "U"
'''
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "core.TextGrid"
            path.write_text(text, encoding="utf-8")
            utterance = parse_textgrid(path)

        self.assertEqual([row.phone for row in utterance.phones], ["k", "u"])
        self.assertTrue(utterance.phones[-1].devoiced)
        self.assertEqual(utterance.corpus, "csj-core")

    def test_jsut_discovery_has_actionable_missing_path(self):
        with self.assertRaisesRegex(FileNotFoundError, "JSUT root"):
            load_jsut(Path("definitely-missing-jsut-corpus"))

    def test_alignment_normalizes_equivalent_phones_and_reports_edits(self):
        exact = align_phone_sequences(["I", "q", "N"], ["i", "cl", "N"])
        mismatch = align_phone_sequences(["k", "a"], ["k", "u", "a"])

        self.assertTrue(exact.exact)
        self.assertEqual(mismatch.cost, 1)
        self.assertIn("required 1", mismatch.diagnostics[0])

    def test_fixed_validation_and_heldout_ids_never_enter_fit(self):
        utterances = [
            _utterance(identifier, "u", devoiced=index % 2 == 0)
            for index, identifier in enumerate(FIXED_VALIDATION_IDS)
        ]
        utterances.extend(
            _utterance(f"TRAIN_{index:03d}", "u" if index % 2 else "a",
                       long=index % 3 == 0, geminate=index % 4 == 0,
                       nasal=index % 5 == 0, devoiced=index % 2 == 1)
            for index in range(20)
        )
        expected_heldout = set(select_heldout_ids(utterances, 0.2))

        result = fit_duration_priors(utterances, heldout_fraction=0.2)
        provenance = result.priors.fit_provenance
        training_ids = set(provenance["source_file_ids"])

        self.assertTrue(set(FIXED_VALIDATION_IDS).isdisjoint(training_ids))
        self.assertTrue(expected_heldout.isdisjoint(training_ids))
        self.assertEqual(set(provenance["heldout_exclusions"]), expected_heldout)

    def test_fit_and_serialized_priors_are_byte_deterministic(self):
        utterances = [
            _utterance(f"TRAIN_{index:03d}", "i" if index % 2 else "a",
                       long=index % 3 == 0, geminate=index % 4 == 0,
                       nasal=index % 5 == 0, devoiced=index % 2 == 1)
            for index in range(24)
        ]
        first = fit_duration_priors(utterances, heldout_fraction=0.2)
        second = fit_duration_priors(utterances, heldout_fraction=0.2)
        seed = load_duration_priors()

        self.assertEqual(first.priors.to_dict(), second.priors.to_dict())
        self.assertEqual(first.report, second.report)
        self.assertEqual(first.priors.mora_allocation_seconds,
                         seed.mora_allocation_seconds)
        self.assertEqual(first.priors.mora_anchor_seconds,
                         seed.mora_anchor_seconds)
        with tempfile.TemporaryDirectory() as root:
            a = write_priors(Path(root) / "a.json", first.priors)
            b = write_priors(Path(root) / "b.json", second.priors)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            json.loads(a.read_text(encoding="utf-8"))

    def test_evaluation_keeps_timing_and_periodicity_separate(self):
        training = [_utterance(f"TRAIN_{index}", "u", devoiced=index % 2 == 0)
                    for index in range(12)]
        evaluation = [_utterance("EVAL_1", "u", devoiced=True),
                      _utterance("EVAL_2", "a", long=True)]
        fitted = fit_duration_priors(training, heldout_fraction=0.1)

        report = evaluate_duration_model(training, evaluation, fitted.priors)

        self.assertIn("legacy", report["phone_metrics"])
        self.assertIn("contextual", report["phone_metrics"])
        self.assertIn("long_vowel", report["metrics_by_phenomenon"])
        self.assertIn("contextual", report["mora_total_metrics"])
        self.assertIn("contextual", report["accent_phrase_total_metrics"])
        self.assertIn("contextual", report["rate_normalized_residual_metrics"])
        self.assertIn("geminate_closure", report["contrast_statistics"])
        self.assertFalse(report["voicing_periodicity"]["available"])
        self.assertIn("separately", report["voicing_periodicity"]["note"])

    def test_word_separator_is_not_a_phrase_or_duration_observation(self):
        phones = (
            _phone(0, 30, "k"),
            _phone(30, 70, "a"),
            _phone(100, 8, "sp"),
            _phone(108, 30, "t"),
            _phone(138, 70, "o"),
            _phone(208, 80, "pau"),
        )
        utterance = CorpusUtterance("WITH_SP", phones)
        training = [
            _utterance(f"TRAIN_{index}", "a") for index in range(12)
        ]
        fitted = fit_duration_priors(training, heldout_fraction=0.1)

        report = evaluate_duration_model(
            training, [utterance], fitted.priors
        )

        self.assertEqual(report["phone_metrics"]["legacy"]["count"], 4)
        self.assertEqual(
            report["accent_phrase_total_metrics"]["legacy"]["count"], 1
        )

    def test_duration_metrics_cover_required_error_families(self):
        metrics = duration_metrics([0.05, 0.1, 0.2], [0.06, 0.09, 0.18])

        self.assertEqual(metrics["count"], 3)
        self.assertGreater(metrics["mae_ms"], 0)
        self.assertGreater(metrics["rmse_ms"], 0)
        self.assertGreater(metrics["log_rmse"], 0)
        self.assertIsNotNone(metrics["pearson"])
        self.assertIsNotNone(metrics["spearman"])


if __name__ == "__main__":
    unittest.main()
