import json
import unittest

import english_syllables as es


class EnglishSyllableTests(unittest.TestCase):
    def test_rabbit_uses_intervocalic_consonant_as_next_onset(self):
        result = es.syllabify_english(["r", "ae1", "b", "ih0", "t"])
        self.assertEqual(
            [row.phones for row in result.syllables],
            [("r", "ae1"), ("b", "ih0", "t")],
        )
        self.assertEqual([row.stress for row in result.syllables], [1, 0])
        self.assertEqual(result.boundaries, (2,))

    def test_maximal_onset_keeps_legal_str_cluster(self):
        result = es.syllabify_english(
            ["eh1", "k", "s", "t", "r", "ah0"])
        self.assertEqual(
            [row.phones for row in result.syllables],
            [("eh1", "k"), ("s", "t", "r", "ah0")],
        )
        self.assertEqual(result.syllables[1].onset, ("s", "t", "r"))

    def test_illegal_tl_cluster_splits_as_coda_plus_onset(self):
        result = es.syllabify_english(["ae1", "t", "l", "ah0", "s"])
        self.assertEqual(
            [row.phones for row in result.syllables],
            [("ae1", "t"), ("l", "ah0", "s")],
        )

    def test_pause_phones_are_boundaries_not_syllables(self):
        result = es.syllabify_english(
            ["pau", "hh", "eh1", "l", "ow0", "pau", "w", "er1", "l", "d"])
        self.assertEqual(result.pause_indices, (0, 5))
        self.assertEqual(
            [row.phones for row in result.syllables],
            [("hh", "eh1"), ("l", "ow0"), ("w", "er1", "l", "d")],
        )
        self.assertEqual(result.syllables[0].boundary_before, "pause")
        self.assertEqual(result.syllables[2].boundary_before, "pause")

    def test_syllabic_consonant_is_a_nucleus(self):
        result = es.syllabify_english(["b", "ah1", "t", "en"])
        self.assertEqual(
            [row.phones for row in result.syllables],
            [("b", "ah1"), ("t", "en")],
        )
        self.assertEqual(result.syllables[1].nucleus, ("en",))

    def test_inline_multilingual_vowel_splits_exact_phone_example(self):
        result = es.syllabify_english(
            ["dh", "ih", "s", "ih", "z", "a", "t", "eh", "s", "t"])
        self.assertEqual(
            [row.phones for row in result.syllables],
            [
                ("dh", "ih"),
                ("s", "ih"),
                ("z", "a"),
                ("t", "eh", "s", "t"),
            ],
        )
        self.assertEqual(result.boundaries, (2, 4, 6))
        self.assertNotIn(
            "UNKNOWN_PHONE",
            {row.code for row in result.diagnostics},
        )

    def test_all_integrated_non_english_nuclei_split_syllables(self):
        for nucleus in sorted(es.INTEGRATED_VOWEL_PHONES):
            with self.subTest(nucleus=nucleus):
                result = es.syllabify_english(
                    ["s", "ih", "z", nucleus, "t"])
                self.assertEqual(
                    [row.phones for row in result.syllables],
                    [("s", "ih"), ("z", nucleus, "t")],
                )

    def test_voice_profile_can_declare_an_additional_vowel_nucleus(self):
        result = es.syllabify_english(
            ["s", "ih", "z", "oe", "t"],
            extra_nucleus_phones={"oe"},
        )
        self.assertEqual(
            [row.phones for row in result.syllables],
            [("s", "ih"), ("z", "oe", "t")],
        )
        self.assertNotIn(
            "UNKNOWN_PHONE",
            {row.code for row in result.diagnostics},
        )
        self.assertEqual(result.declared_nucleus_phones, ("oe",))

    def test_profile_vowel_types_exclude_non_syllabic_breath(self):
        self.assertEqual(
            es.profile_nucleus_phones({
                "oe": "vowel",
                "inh": "vowel",
                "x": "fricative",
            }),
            ("oe",),
        )

    def test_inhale_timing_vowel_is_not_a_syllable_nucleus(self):
        result = es.syllabify_english(["inh"])
        self.assertEqual(len(result.syllables), 1)
        self.assertEqual(result.syllables[0].nucleus, ())
        self.assertIn(
            "NO_NUCLEUS",
            {row.code for row in result.diagnostics},
        )

    def test_lexical_boundaries_prevent_cross_word_onset_stealing(self):
        phrase_only = es.syllabify_english(["ae1", "n", "ey1", "m"])
        word_aware = es.syllabify_english(
            ["ae1", "n", "ey1", "m"], word_boundaries=[2])
        self.assertEqual(
            [row.phones for row in phrase_only.syllables],
            [("ae1",), ("n", "ey1", "m")],
        )
        self.assertEqual(
            [row.phones for row in word_aware.syllables],
            [("ae1", "n"), ("ey1", "m")],
        )
        self.assertEqual(word_aware.syllables[1].boundary_before, "word")

    def test_unknown_and_nucleus_free_phones_are_preserved(self):
        result = es.syllabify_english(["xyz", "cl"])
        self.assertEqual(result.syllables[0].phones, ("xyz", "cl"))
        self.assertEqual(result.syllables[0].diagnostics, ("NO_NUCLEUS",))
        self.assertEqual(
            {row.code for row in result.diagnostics},
            {"UNKNOWN_PHONE", "NO_NUCLEUS"},
        )

    def test_alternative_suffix_does_not_replace_stress_digit(self):
        result = es.syllabify_english(["r", "AE1__u12", "b"])
        self.assertEqual(result.normalized_phones, ("r", "ae", "b"))
        self.assertEqual(result.syllables[0].stress, 1)

    def test_json_shape_round_trips_deterministically(self):
        original = es.syllabify_english(
            ["s", "t", "r", "ih1", "ng", "pau", "t", "eh2", "s", "t"])
        encoded = json.dumps(
            original.to_dict(), sort_keys=True, ensure_ascii=True)
        restored = es.EnglishSyllabification.from_dict(
            json.loads(encoded))
        self.assertEqual(restored, original)
        self.assertEqual(
            json.dumps(restored.to_dict(), sort_keys=True),
            json.dumps(original.to_dict(), sort_keys=True),
        )


if __name__ == "__main__":
    unittest.main()
