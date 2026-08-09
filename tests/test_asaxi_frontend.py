# -*- coding: utf-8 -*-
"""Deterministic tests for the canonical Asaxi word frontend."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import asaxi_frontend as af


class AsaxiFrontendTests(unittest.TestCase):
    def test_full_cap_terms_are_detected_without_lowercasing_source(self) -> None:
        text = "to JOHN anő apo chỏnů. JOHN"

        self.assertEqual(
            af.words_in_text(text, preserve_case=True),
            ("to", "JOHN", "anő", "apo", "chỏnů", "JOHN"),
        )
        self.assertEqual(
            af.capitalized_terms_in_text(text),
            ("JOHN",),
        )
        self.assertTrue(af.is_capitalized_term("JOHN"))
        self.assertFalse(af.is_capitalized_term("John"))
        self.assertFalse(af.is_capitalized_term("john"))
        self.assertEqual(
            af.CAPITALIZED_ENGLISH_PRONUNCIATION_OVERRIDES["JOHN"],
            ("jh", "ao", "n"),
        )

    def test_g2p_matches_existing_integrated_voice_symbols(self) -> None:
        self.assertEqual(
            af.g2p_asaxi("shěsonů"),
            ("sh", "er", "s", "o", "n", "u", "w"),
        )
        self.assertEqual(af.g2p_asaxi("tte"), ("cl", "t", "e"))

    def test_diphthong_is_one_mora_but_two_bank_phones(self) -> None:
        moras = af.split_morae("tăka")
        self.assertEqual([mora.text for mora in moras], ["tă", "ka"])
        self.assertEqual(moras[0].phones, ("t", "a", "y"))

    def test_atomic_palatalized_grapheme_finishes_its_own_mora(self) -> None:
        self.assertEqual(
            [mora.text for mora in af.split_morae("nihè")],
            ["ni", "hè"],
        )
        self.assertEqual(
            [mora.phones for mora in af.split_morae("nihè")],
            [("ny", "i"), ("h", "ax")],
        )
        self.assertEqual(
            [mora.text for mora in af.split_morae("sino")],
            ["si", "no"],
        )

    def test_syllabic_nasal_and_geminate_each_occupy_a_mora(self) -> None:
        nasal = af.split_morae("mmbă")
        self.assertEqual([mora.kind for mora in nasal],
                         ["syllabic_nasal", "ordinary"])
        self.assertFalse(nasal[0].accentable)
        geminate = af.split_morae("tte")
        self.assertEqual([mora.kind for mora in geminate],
                         ["geminate", "ordinary"])
        self.assertEqual(geminate[0].phones, ("cl",))

    def test_dotted_nasal_geminate_spans_adjacent_syllables(self) -> None:
        self.assertEqual(
            af.g2p_asaxi("kem.ma"),
            ("k", "e", "m", "m", "a"),
        )
        moras = af.split_morae("kem.ma")
        self.assertEqual(
            [(mora.text, mora.phones) for mora in moras],
            [
                ("kem", ("k", "e", "m")),
                ("ma", ("m", "a")),
            ],
        )
        self.assertEqual(
            [(mora.start, mora.end) for mora in moras],
            [(0, 3), (4, 6)],
        )

        self.assertEqual(
            af.g2p_asaxi("ken.ná"),
            ("k", "e", "n", "n", "ao"),
        )
        self.assertEqual(
            [
                (mora.text, mora.phones)
                for mora in af.split_morae("ken.ná")
            ],
            [
                ("ken", ("k", "e", "n")),
                ("ná", ("n", "ao")),
            ],
        )

    def test_dotted_and_undotted_nasals_remain_distinct(self) -> None:
        self.assertEqual(af.g2p_asaxi("mm"), ("mm",))
        self.assertEqual(af.g2p_asaxi("nn"), ("nn",))
        self.assertEqual(af.g2p_asaxi("m.m"), ("m", "m"))
        self.assertEqual(af.g2p_asaxi("n.n"), ("n", "n"))

    def test_dotted_nasal_rule_applies_inside_full_text(self) -> None:
        text = "găxănă ono kem.ma"
        self.assertEqual(
            af.words_in_text(text),
            ("găxănă", "ono", "kem.ma"),
        )
        analysis = af.analyze_word(af.words_in_text(text)[-1])
        self.assertEqual(
            analysis.phones,
            ("k", "e", "m", "m", "a"),
        )
        self.assertEqual(
            [mora.phones for mora in analysis.moras],
            [("k", "e", "m"), ("m", "a")],
        )

    def test_default_accent_skips_syllabic_nasal(self) -> None:
        self.assertEqual(af.default_pitch_pattern("mmbănă"), "L.H.L")
        self.assertEqual(
            af.default_pitch_pattern("to", atonal=True), "L")

    def test_pitch_pattern_parser_supports_phrases(self) -> None:
        self.assertEqual(
            af.parse_pitch_pattern("H.L | L.H"),
            (("H", "L"), ("L", "H")),
        )
        with self.assertRaises(ValueError):
            af.parse_pitch_pattern("H.X")

    def test_text_tokenizer_preserves_idiom_words_and_rejects_typos(self) -> None:
        self.assertEqual(
            af.words_in_text(
                "xi fůjå ma.",
                reject_unsupported_letters=True,
            ),
            ("xi", "fůjå", "ma"),
        )
        with self.assertRaisesRegex(ValueError, "ū"):
            af.words_in_text(
                "fūjå ma.",
                reject_unsupported_letters=True,
            )

    def test_dictionary_loader_validates_mora_and_g2p_consistency(self) -> None:
        analysis = af.analyze_word("ai")
        entry = {
            "phones": list(analysis.phones),
            "moras": [
                {
                    "text": mora.text,
                    "phones": list(mora.phones),
                    "start": mora.start,
                    "end": mora.end,
                    "accentable": mora.accentable,
                    "kind": mora.kind,
                }
                for mora in analysis.moras
            ],
            "pitch_accent": "H.L",
            "pitch_accent_class": "lexical",
            "source_note": "Lexicon/ai (noun).md",
        }
        payload = {
            "schema_version": 1,
            "language": "asaxi",
            "ruleset": "asaxi-pitch-v1",
            "entries": {"ai": entry},
            "phrases": {},
            "source_summary": {"entries": 1},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dictionary.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            loaded = af.load_synthesis_dictionary(path)
        self.assertEqual(loaded.lookup("AI").pitch_values, ("H", "L"))
        self.assertEqual(loaded.pronunciations(), {"ai": ["a", "i"]})

    def test_dictionary_loader_preserves_explicit_g2p_override(self) -> None:
        analysis = af.analyze_word("gavi")
        payload = {
            "schema_version": 1,
            "language": "asaxi",
            "ruleset": "asaxi-pitch-v1",
            "entries": {
                "gavi": {
                    "phones": ["g", "aa", "v", "i"],
                    "g2p_source": "override",
                    "moras": [
                        {
                            "text": mora.text,
                            "phones": list(mora.phones),
                            "start": mora.start,
                            "end": mora.end,
                            "accentable": mora.accentable,
                            "kind": mora.kind,
                        }
                        for mora in analysis.moras
                    ],
                    "pitch_accent": "H.L",
                    "pitch_accent_class": "lexical",
                }
            },
            "phrases": {},
            "source_summary": {"entries": 1},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dictionary.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            loaded = af.load_synthesis_dictionary(path)
        entry = loaded.lookup("gavi")
        self.assertEqual(entry.phones, ("g", "aa", "v", "i"))
        self.assertTrue(entry.g2p_override)

    def test_generated_dictionary_preserves_atomic_nucleus_boundary(
        self,
    ) -> None:
        dictionary_path = (
            Path(af.__file__).resolve().parent
            / "dictionaries"
            / "asaxi_lexicon.json"
        )
        entry = af.load_synthesis_dictionary(dictionary_path).lookup("nihè")
        self.assertIsNotNone(entry)
        self.assertEqual(
            [mora.text for mora in entry.moras],
            ["ni", "hè"],
        )
        self.assertEqual(entry.pitch_values, ("L", "L"))
        self.assertEqual(entry.phones, ("ny", "i", "h", "ax"))

    def test_generated_dictionary_preserves_dotted_nasal_structure(
        self,
    ) -> None:
        dictionary_path = (
            Path(af.__file__).resolve().parent
            / "dictionaries"
            / "asaxi_lexicon.json"
        )
        dictionary = af.load_synthesis_dictionary(dictionary_path)
        entry = dictionary.lookup("kem.mo")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.phones, ("k", "e", "m", "m", "o"))
        self.assertEqual(
            [(mora.text, mora.phones) for mora in entry.moras],
            [
                ("kem", ("k", "e", "m")),
                ("mo", ("m", "o")),
            ],
        )


if __name__ == "__main__":
    unittest.main()
