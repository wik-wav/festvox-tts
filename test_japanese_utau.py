import tempfile
import unittest
from pathlib import Path

import japanese_utau as ju


class JapaneseUtauAnalysisTests(unittest.TestCase):
    @staticmethod
    def write_oto(path, aliases, encoding="utf-8", bom=False):
        lines = [
            f"sample_{index}.wav={alias},0,100,-200,50,10"
            for index, alias in enumerate(aliases)
        ]
        data = ("\n".join(lines) + "\n").encode(encoding)
        if bom:
            data = b"\xef\xbb\xbf" + data
        path.write_bytes(data)

    def test_cp932_decoding_is_strict_and_reported(self):
        with tempfile.TemporaryDirectory() as root:
            oto = Path(root) / "oto.ini"
            self.write_oto(oto, ["あF3", "- かF3"], encoding="cp932")

            document = ju.parse_oto_file(oto)

            self.assertEqual(document.source.encoding, "cp932")
            self.assertFalse(document.source.ambiguous)
            self.assertEqual(document.entries[0].normalization.analysis_alias, "あ")
            self.assertEqual(document.entries[0].evidence.family, "cv")

    def test_utf8_bom_is_detected_without_becoming_part_of_wav_name(self):
        with tempfile.TemporaryDirectory() as root:
            oto = Path(root) / "oto.ini"
            self.write_oto(oto, ["a あE3"], bom=True)

            document = ju.parse_oto_file(oto)

            self.assertEqual(document.source.encoding, "utf-8-sig")
            self.assertEqual(document.entries[0].wav_raw, "sample_0.wav")
            self.assertEqual(document.entries[0].evidence.family, "vcv")

    def test_ascii_metadata_is_not_falsely_ambiguous(self):
        with tempfile.TemporaryDirectory() as root:
            oto = Path(root) / "oto.ini"
            self.write_oto(oto, ["kaF3"])

            decoded = ju.decode_text_file(oto)

            self.assertEqual(decoded.encoding, "utf-8")
            self.assertEqual(decoded.confidence, 1.0)
            self.assertFalse(decoded.ambiguous)

    def test_invalid_utf8_and_cp932_never_fall_back_to_latin1(self):
        with tempfile.TemporaryDirectory() as root:
            oto = Path(root) / "oto.ini"
            oto.write_bytes(b"broken=\x81")

            with self.assertRaises(ju.TextDecodeError):
                ju.decode_text_file(oto)

    def test_malformed_timing_keeps_alias_and_precise_diagnostic(self):
        with tempfile.TemporaryDirectory() as root:
            oto = Path(root) / "oto.ini"
            oto.write_bytes(
                "sample.wav=びょF3,,100,-200,50,10\n".encode("cp932")
            )

            document = ju.parse_oto_file(oto)

            self.assertEqual(len(document.entries), 1)
            self.assertIsNone(document.entries[0].offset)
            self.assertEqual(document.entries[0].alias_raw, "びょF3")
            diagnostic = next(
                item for item in document.diagnostics
                if item.code == "oto_invalid_number"
            )
            self.assertEqual(diagnostic.line, 1)
            self.assertEqual(diagnostic.byte_offset, 0)

    def test_alias_identity_is_preserved_while_match_forms_normalize(self):
        source = "  か\u3099F3  "

        normalized = ju.normalize_alias(source)

        self.assertEqual(normalized.source_alias, source)
        self.assertIn("が", normalized.canonical_alias)
        self.assertEqual(normalized.analysis_alias, "が")

    def test_declared_suffix_works_before_or_after_pitch_tag(self):
        before = ju.normalize_alias("ayPE3", alias_suffixes=["P"])
        after = ju.normalize_alias("ayE3P", alias_suffixes=["P"])

        self.assertEqual(before.analysis_alias, "ay")
        self.assertEqual(after.analysis_alias, "ay")
        self.assertEqual(before.removed_suffixes, ("P",))
        self.assertEqual(after.removed_suffixes, ("P",))

    def test_cv_bank_can_contain_small_vcv_fallback_inventory(self):
        with tempfile.TemporaryDirectory() as root:
            oto = Path(root) / "oto.ini"
            self.write_oto(oto, [
                "あF3", "いF3", "うF3", "えF3", "おF3",
                "- かF3", "* さF3", "a あF3",
            ], encoding="cp932")

            analysis = ju.analyze_bank(Path(root))

            self.assertEqual(analysis.bank_type, "cv")
            self.assertGreater(analysis.family_counts["cv"],
                               analysis.family_counts["vcv"])

    def test_asterisk_vowel_is_blend_material_not_phrase_start(self):
        evidence = ju.classify_alias(ju.normalize_alias("* \u3044"))
        phrase_start = ju.classify_alias(ju.normalize_alias("- \u3044"))

        self.assertEqual(evidence.family, "cv")
        self.assertEqual(evidence.subtype, "vowel_blend")
        self.assertEqual(phrase_start.subtype, "phrase_initial_cv")

    def test_vcv_bank_is_identified_by_vowel_to_kana_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            oto = Path(root) / "oto.ini"
            self.write_oto(oto, [
                "a か_F3", "i き_F3", "u く_F3", "e け_F3",
                "o こ_F3", "n か_F3", "- か_F3", "か_F3",
                "a k_F3",
            ], encoding="cp932")

            analysis = ju.analyze_bank(Path(root))

            self.assertEqual(analysis.bank_type, "vcv")
            self.assertGreaterEqual(analysis.family_counts["vcv"], 6)

    def test_cvvc_bank_is_identified_by_vc_transition_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            oto = Path(root) / "oto.ini"
            self.write_oto(oto, [
                "a k1E3", "i k1E3", "u k1E3", "e k1E3",
                "o k1E3", "n k1E3", "u k-1E3", "- か1E3",
                "か1E3", "a か1E3",
            ], encoding="cp932")

            analysis = ju.analyze_bank(Path(root))

            self.assertEqual(analysis.bank_type, "cvvc")
            self.assertGreaterEqual(analysis.family_counts["cvvc"], 7)

    def test_bank_type_override_is_explicit_in_report(self):
        with tempfile.TemporaryDirectory() as root:
            oto = Path(root) / "oto.ini"
            self.write_oto(oto, ["あF3"], encoding="cp932")

            analysis = ju.analyze_bank(Path(root), bank_type="mixed")

            self.assertEqual(analysis.bank_type, "mixed")
            self.assertEqual(analysis.bank_type_override, "mixed")
            self.assertEqual(analysis.confidence, 1.0)

    def test_analysis_does_not_write_to_source_bank(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            oto = bank / "oto.ini"
            self.write_oto(oto, ["あF3"], encoding="cp932")
            before = {path.name: path.read_bytes() for path in bank.iterdir()}

            analysis = ju.analyze_bank(bank)

            after = {path.name: path.read_bytes() for path in bank.iterdir()}
            self.assertEqual(before, after)
            self.assertEqual(len(analysis.entries), 1)

    def test_report_writer_refuses_source_voicebank(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            oto = bank / "oto.ini"
            self.write_oto(oto, ["あF3"], encoding="cp932")
            analysis = ju.analyze_bank(bank)
            report = bank / "analysis.json"

            with self.assertRaisesRegex(ValueError, "source voicebank"):
                ju.write_report(analysis, report)
            self.assertFalse(report.exists())


if __name__ == "__main__":
    unittest.main()
