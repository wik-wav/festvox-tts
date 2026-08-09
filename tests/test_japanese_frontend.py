import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from japanese_frontend import (
    analyze_japanese,
    clear_japanese_frontend_cache,
    japanese_frontend_cache_info,
    resolve_japanese_frontend,
)
from japanese_kana_frontend import KanaJapaneseFrontend, segment_kana_reading
from japanese_models import PROVISIONAL_SCHEMA
from japanese_openjtalk import (
    OpenJTalkJapaneseFrontend,
    OpenJTalkUnavailableError,
    canonicalize_openjtalk_phone,
    is_pyopenjtalk_available,
    parse_full_context_label,
    parse_openjtalk_labels,
    pyopenjtalk_status,
)


def _label(quinphone, a_context, f_context, *, i_context, k_context, extra=""):
    return (
        f"{quinphone}/A:{a_context}/B:xx-xx_xx/C:xx_xx+xx/"
        f"D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:{f_context}/"
        f"G:xx_xx%xx_xx_xx/H:xx_xx/I:{i_context}/J:xx_xx/"
        f"K:{k_context}{extra}"
    )


_EMPTY_F = "xx_xx#xx_xx@xx_xx|xx_xx"
_SIMPLE_I = "1-2@1+1&1-1|1+2"
_SIMPLE_K = "1+1-2"

# Static Open JTalk-shaped fixture for かな.  No optional package is required.
SIMPLE_LABELS = (
    _label("xx^xx-sil+k=a", "xx+xx+xx", _EMPTY_F,
           i_context=_SIMPLE_I, k_context=_SIMPLE_K),
    _label("xx^sil-k+a=n", "0+1+2", "2_0#0_xx@1_1|1_2",
           i_context=_SIMPLE_I, k_context=_SIMPLE_K),
    _label("sil^k-a+n=a", "0+1+2", "2_0#0_xx@1_1|1_2",
           i_context=_SIMPLE_I, k_context=_SIMPLE_K),
    _label("k^a-n+a=sil", "0+2+1", "2_0#0_xx@1_1|1_2",
           i_context=_SIMPLE_I, k_context=_SIMPLE_K),
    _label("a^n-a+sil=xx", "0+2+1", "2_0#0_xx@1_1|1_2",
           i_context=_SIMPLE_I, k_context=_SIMPLE_K),
    _label("n^a-sil+xx=xx", "xx+xx+xx", _EMPTY_F,
           i_context=_SIMPLE_I, k_context=_SIMPLE_K),
)


_MULTI_I = "2-4@1+1&1-2|1+4"
_MULTI_K = "1+2-4"
MULTI_ACCENT_LABELS = (
    _label("xx^xx-sil+k=a", "xx+xx+xx", _EMPTY_F,
           i_context=_MULTI_I, k_context=_MULTI_K),
    _label("xx^sil-k+a=n", "0+1+2", "2_1#0_xx@1_2|1_2",
           i_context=_MULTI_I, k_context=_MULTI_K),
    _label("sil^k-a+n=a", "0+1+2", "2_1#0_xx@1_2|1_2",
           i_context=_MULTI_I, k_context=_MULTI_K),
    _label("k^a-n+a=k", "1+2+1", "2_1#0_xx@1_2|1_2",
           i_context=_MULTI_I, k_context=_MULTI_K),
    _label("a^n-a+k=a", "1+2+1", "2_1#0_xx@1_2|1_2",
           i_context=_MULTI_I, k_context=_MULTI_K),
    _label("n^a-k+a=n", "0+1+2", "2_0#1_xx@2_1|3_4",
           i_context=_MULTI_I, k_context=_MULTI_K),
    _label("a^k-a+n=a", "0+1+2", "2_0#1_xx@2_1|3_4",
           i_context=_MULTI_I, k_context=_MULTI_K),
    _label("k^a-n+a=sil", "0+2+1", "2_0#1_xx@2_1|3_4",
           i_context=_MULTI_I, k_context=_MULTI_K),
    _label("a^n-a+sil=xx", "0+2+1", "2_0#1_xx@2_1|3_4",
           i_context=_MULTI_I, k_context=_MULTI_K),
    _label("n^a-sil+xx=xx", "xx+xx+xx", _EMPTY_F,
           i_context=_MULTI_I, k_context=_MULTI_K),
)


_PAUSE_I = "1-1@1+2&1-1|1+1"
_PAUSE_K = "2+2-2"
PAUSE_LABELS = (
    _label("xx^xx-sil+k=a", "xx+xx+xx", _EMPTY_F,
           i_context=_PAUSE_I, k_context=_PAUSE_K),
    _label("xx^sil-k+a=pau", "0+1+1", "1_0#0_xx@1_1|1_1",
           i_context=_PAUSE_I, k_context=_PAUSE_K),
    _label("sil^k-a+pau=n", "0+1+1", "1_0#0_xx@1_1|1_1",
           i_context=_PAUSE_I, k_context=_PAUSE_K),
    _label("k^a-pau+n=a", "xx+xx+xx", _EMPTY_F,
           i_context=_PAUSE_I, k_context=_PAUSE_K),
    _label("a^pau-n+a=sil", "0+1+1", "1_0#1_xx@1_1|1_1",
           i_context=_PAUSE_I, k_context=_PAUSE_K),
    _label("pau^n-a+sil=xx", "0+1+1", "1_0#1_xx@1_1|1_1",
           i_context=_PAUSE_I, k_context=_PAUSE_K),
    _label("n^a-sil+xx=xx", "xx+xx+xx", _EMPTY_F,
           i_context=_PAUSE_I, k_context=_PAUSE_K),
)

INLINE_QUOTE_PAUSE_LABELS = (
    _label("xx^xx-sil+k=a", "xx+xx+xx", _EMPTY_F,
           i_context=_PAUSE_I, k_context=_PAUSE_K),
    _label("xx^sil-k+a=pau", "0+1+1", "1_0#0_xx@1_2|1_1",
           i_context=_PAUSE_I, k_context=_PAUSE_K),
    _label("sil^k-a+pau=w", "0+1+1", "1_0#0_xx@1_2|1_1",
           i_context=_PAUSE_I, k_context=_PAUSE_K),
    _label("k^a-pau+w=a", "xx+xx+xx", _EMPTY_F,
           i_context=_PAUSE_I, k_context=_PAUSE_K),
    _label("a^pau-w+a=sil", "0+1+1", "1_0#1_xx@2_1|2_2",
           i_context=_PAUSE_I, k_context=_PAUSE_K),
    _label("pau^w-a+sil=xx", "0+1+1", "1_0#1_xx@2_1|2_2",
           i_context=_PAUSE_I, k_context=_PAUSE_K),
    _label("w^a-sil+xx=xx", "xx+xx+xx", _EMPTY_F,
           i_context=_PAUSE_I, k_context=_PAUSE_K),
)


class CanonicalModelAndKanaTests(unittest.TestCase):
    def setUp(self):
        self.frontend = KanaJapaneseFrontend()

    def test_canonical_model_serialization_is_provisional_and_json_safe(self):
        utterance = self.frontend.analyze("かな？")
        data = utterance.to_dict()
        self.assertEqual(data["schema"], PROVISIONAL_SCHEMA)
        self.assertEqual(
            data["phrases"][0]["accent_phrases"][0]["accent_state"],
            "unavailable",
        )
        self.assertEqual(json.loads(utterance.to_json())["source_text"], "かな？")
        self.assertIsInstance(data["phones"], list)

    def test_basic_hiragana_input(self):
        utterance = self.frontend.analyze("かな")
        self.assertEqual(utterance.normalized_reading, "かな")
        self.assertEqual([phone.symbol for phone in utterance.phones],
                         ["sil", "k", "a", "n", "a", "sil"])
        self.assertEqual(len(utterance.moras), 2)

    def test_katakana_input_normalizes_to_hiragana_reading(self):
        utterance = self.frontend.analyze("カナ")
        self.assertEqual(utterance.source_text, "カナ")
        self.assertEqual(utterance.normalized_reading, "かな")
        self.assertEqual([mora.consonant for mora in utterance.moras], ["k", "n"])

    def test_supported_romaji_input(self):
        utterance = self.frontend.analyze("konnichiwa")
        self.assertEqual(utterance.normalized_reading, "こんにちわ")
        self.assertNotIn("unsupported_romaji",
                         {item.code for item in utterance.diagnostics})

    def test_long_vowels_are_separate_moras(self):
        utterance = self.frontend.analyze("スーパー")
        long_moras = [
            mora for mora in utterance.moras
            if mora.special_mora == "long_vowel"
        ]
        self.assertEqual(len(long_moras), 2)
        self.assertEqual([phone.symbol for phone in long_moras[0].phones], ["u"])
        self.assertIsNone(long_moras[0].consonant)

    def test_moraic_nasal(self):
        utterance = self.frontend.analyze("ほん")
        nasal = utterance.moras[-1]
        self.assertEqual(nasal.special_mora, "moraic_nasal")
        self.assertEqual([phone.symbol for phone in nasal.phones], ["N"])

    def test_geminate(self):
        utterance = self.frontend.analyze("がっこう")
        geminate = utterance.moras[1]
        self.assertEqual(geminate.special_mora, "geminate")
        self.assertEqual(geminate.phones[0].symbol, "cl")

    def test_palatalized_mora(self):
        utterance = self.frontend.analyze("きゃ")
        self.assertEqual(len(utterance.moras), 1)
        self.assertEqual(utterance.moras[0].consonant, "ky")
        self.assertEqual([phone.symbol for phone in utterance.moras[0].phones],
                         ["ky", "a"])

    def test_punctuation_creates_pause_and_phrase_boundaries(self):
        utterance = self.frontend.analyze("かな、な。")
        self.assertEqual(len(utterance.phrases), 2)
        self.assertEqual(utterance.phrases[0].boundary_strength, 1)
        self.assertEqual(utterance.phrases[0].punctuation_after, "、")
        self.assertEqual([phone.symbol for phone in utterance.phones].count("pau"), 2)

    def test_question_punctuation_sets_interrogative(self):
        utterance = self.frontend.analyze("かな？")
        self.assertTrue(utterance.phrases[0].interrogative)
        self.assertTrue(utterance.accent_phrases[0].interrogative)

    def test_nonspoken_quotes_do_not_create_reading_moras(self):
        plain, plain_diagnostics = segment_kana_reading(
            "\u3053\u304f\u3057\u3087\u3073"
        )
        quoted, quoted_diagnostics = segment_kana_reading(
            "\u300c\u3053\u304f\u3057\u3087\u3073\u300d"
        )
        self.assertEqual(
            [item.phones for item in quoted],
            [item.phones for item in plain],
        )
        self.assertFalse(plain_diagnostics)
        self.assertFalse(quoted_diagnostics)

    def test_unsupported_kanji_is_diagnostic_and_representable(self):
        utterance = self.frontend.analyze("猫")
        self.assertIn("unsupported_kanji",
                      {item.code for item in utterance.diagnostics})
        self.assertEqual(utterance.moras[0].special_mora, "unknown")
        self.assertTrue(utterance.moras[0].phones[0].unknown)
        self.assertEqual(utterance.source_text, "猫")

    def test_fallback_reports_neutral_nonlexical_accent(self):
        utterance = self.frontend.analyze("かな")
        self.assertEqual(utterance.accent_phrases[0].accent_state, "unavailable")
        self.assertIsNone(utterance.accent_phrases[0].accent_nucleus)
        self.assertIn("lexical_accent_unavailable",
                      {item.code for item in utterance.diagnostics})


class DispatcherTests(unittest.TestCase):
    def test_repeated_analysis_uses_bounded_utterance_cache(self):
        clear_japanese_frontend_cache()
        expected = KanaJapaneseFrontend().analyze("かな")
        with mock.patch("japanese_frontend.KanaJapaneseFrontend") as factory:
            factory.return_value.analyze.return_value = expected
            first = analyze_japanese("かな", "kana")
            second = analyze_japanese("かな", "kana")

        self.assertIsNot(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())
        first.provenance["poison"] = True
        third = analyze_japanese("かな", "kana")
        self.assertNotIn("poison", third.provenance)
        factory.return_value.analyze.assert_called_once_with("かな")
        info = japanese_frontend_cache_info()
        self.assertEqual(info["entries"], 1)
        self.assertGreaterEqual(info["hits"], 1)
        clear_japanese_frontend_cache()

    def test_openjtalk_mode_requires_optional_dependency(self):
        with mock.patch(
            "japanese_frontend.is_pyopenjtalk_available", return_value=False
        ):
            with self.assertRaises(OpenJTalkUnavailableError) as raised:
                resolve_japanese_frontend("openjtalk")
        self.assertEqual(raised.exception.diagnostic.code,
                         "pyopenjtalk_unavailable")

    def test_auto_mode_falls_back_inspectably(self):
        with mock.patch(
            "japanese_frontend.is_pyopenjtalk_available", return_value=False
        ):
            utterance = analyze_japanese("かな", mode="auto")
        self.assertEqual(utterance.frontend_name, "kana")
        self.assertEqual(utterance.provenance["dispatcher_selected"], "kana")
        self.assertEqual(utterance.diagnostics[0].code,
                         "openjtalk_unavailable_auto_fallback")

    def test_explicit_kana_mode_does_not_probe_openjtalk(self):
        with mock.patch(
            "japanese_frontend.is_pyopenjtalk_available"
        ) as availability:
            utterance = analyze_japanese("かな", mode="kana")
        availability.assert_not_called()
        self.assertEqual(utterance.frontend_name, "kana")


class FullContextParserTests(unittest.TestCase):
    def test_named_full_context_fields(self):
        parsed = parse_full_context_label(SIMPLE_LABELS[1])
        self.assertEqual(parsed.quinphone.previous, "sil")
        self.assertEqual(parsed.quinphone.current, "k")
        self.assertEqual(parsed.quinphone.following, "a")
        self.assertEqual(parsed.mora.position_forward, 1)
        self.assertEqual(parsed.accent_phrase.mora_count, 2)
        self.assertEqual(parsed.accent_phrase.accent_nucleus, 0)
        self.assertFalse(parsed.accent_phrase.interrogative)
        self.assertEqual(parsed.breath_group.accent_phrase_count, 1)
        self.assertEqual(parsed.utterance.mora_count, 2)

    def test_representative_labels_form_unaccented_utterance(self):
        utterance = parse_openjtalk_labels(
            SIMPLE_LABELS, source_text="かな", normalized_reading="カナ"
        )
        self.assertEqual(utterance.normalized_reading, "かな")
        self.assertEqual(len(utterance.phrases), 1)
        self.assertEqual(len(utterance.moras), 2)
        self.assertEqual(utterance.accent_phrases[0].accent_state, "unaccented")
        self.assertIsNone(utterance.accent_phrases[0].accent_nucleus)

    def test_multiple_accent_phrases_and_zero_based_nucleus(self):
        utterance = parse_openjtalk_labels(
            MULTI_ACCENT_LABELS,
            source_text="かなかな？",
            normalized_reading="カナカナ",
        )
        self.assertEqual(len(utterance.phrases), 1)
        self.assertEqual(len(utterance.accent_phrases), 2)
        first, second = utterance.accent_phrases
        self.assertEqual(first.accent_state, "accented")
        self.assertEqual(first.accent_nucleus, 0)
        self.assertEqual(second.accent_state, "unaccented")

    def test_interrogative_label_is_extracted(self):
        utterance = parse_openjtalk_labels(
            MULTI_ACCENT_LABELS,
            source_text="かなかな",
            normalized_reading="かなかな",
        )
        self.assertTrue(utterance.accent_phrases[1].interrogative)
        self.assertTrue(utterance.phrases[0].interrogative)

    def test_pause_and_utterance_boundaries_form_phrases(self):
        utterance = parse_openjtalk_labels(
            PAUSE_LABELS,
            source_text="か、な？",
            normalized_reading="カナ",
        )
        self.assertEqual(len(utterance.phrases), 2)
        self.assertEqual(utterance.phrases[0].boundary_strength, 1)
        self.assertTrue(utterance.phrases[1].interrogative)
        self.assertEqual([phone.symbol for phone in utterance.phones].count("pau"), 1)
        self.assertEqual([phone.symbol for phone in utterance.phones].count("sil"), 2)

    def test_inline_quote_pause_stays_within_one_phrase(self):
        utterance = parse_openjtalk_labels(
            INLINE_QUOTE_PAUSE_LABELS,
            source_text="「か」は。",
            normalized_reading="「カ」ワ",
        )
        self.assertEqual(len(utterance.phrases), 1)
        self.assertEqual(len(utterance.accent_phrases), 2)
        self.assertEqual(
            [phone.symbol for phone in utterance.phones].count("pau"), 1
        )
        self.assertIn(
            "openjtalk_inline_bracket_pause",
            {item.code for item in utterance.diagnostics},
        )
        self.assertNotIn(
            "openjtalk_source_phrase_mismatch",
            {item.code for item in utterance.diagnostics},
        )
        self.assertEqual(
            utterance.provenance["inline_bracket_pause_label_indices"],
            [3],
        )

    def test_explicit_comma_at_quote_boundary_remains_a_phrase_pause(self):
        utterance = parse_openjtalk_labels(
            PAUSE_LABELS,
            source_text="「か」、な？",
            normalized_reading="「カ」、ナ",
        )
        self.assertEqual(len(utterance.phrases), 2)
        self.assertEqual(utterance.phrases[0].boundary_strength, 1)
        self.assertNotIn(
            "openjtalk_inline_bracket_pause",
            {item.code for item in utterance.diagnostics},
        )

    def test_decimal_point_is_not_a_sentence_boundary(self):
        utterance = parse_openjtalk_labels(
            PAUSE_LABELS,
            source_text="40.2、な？",
            normalized_reading="カナ",
        )
        self.assertEqual(len(utterance.phrases), 2)
        self.assertEqual(utterance.phrases[0].surface, "40.2")
        self.assertEqual(utterance.phrases[0].punctuation_after, "、")
        self.assertNotIn(
            "openjtalk_source_phrase_mismatch",
            {item.code for item in utterance.diagnostics},
        )

    def test_fullwidth_decimal_point_is_not_a_sentence_boundary(self):
        utterance = parse_openjtalk_labels(
            PAUSE_LABELS,
            source_text="40．2、な？",
            normalized_reading="カナ",
        )
        self.assertEqual(len(utterance.phrases), 2)
        self.assertEqual(utterance.phrases[0].surface, "40．2")

    def test_middle_dot_is_a_minor_list_boundary(self):
        utterance = parse_openjtalk_labels(
            PAUSE_LABELS,
            source_text="か・な。",
            normalized_reading="カ・ナ",
        )
        self.assertEqual(len(utterance.phrases), 2)
        self.assertEqual(utterance.phrases[0].boundary_strength, 1)
        self.assertEqual(utterance.phrases[0].punctuation_after, "・")
        self.assertNotIn(
            "openjtalk_source_phrase_mismatch",
            {item.code for item in utterance.diagnostics},
        )

    def test_triangle_bullet_is_a_list_item_boundary(self):
        utterance = parse_openjtalk_labels(
            PAUSE_LABELS,
            source_text="か▽な。",
            normalized_reading="カ▽ナ",
        )
        self.assertEqual(len(utterance.phrases), 2)
        self.assertEqual(utterance.phrases[0].boundary_strength, 2)
        self.assertEqual(utterance.phrases[0].punctuation_after, "▽")
        self.assertNotIn(
            "openjtalk_source_phrase_mismatch",
            {item.code for item in utterance.diagnostics},
        )

    def test_unknown_group_and_phone_are_preserved(self):
        raw = _label(
            "xx^xx-xyz+xx=xx", "0+1+1", "1_0#0_xx@1_1|1_1",
            i_context="1-1@1+1&1-1|1+1", k_context="1+1-1",
            extra="/L:future_field",
        )
        parsed = parse_full_context_label(raw)
        self.assertEqual(parsed.raw_groups["L"], "future_field")
        utterance = parse_openjtalk_labels(
            (raw,), source_text="あ", normalized_reading="あ"
        )
        self.assertTrue(utterance.phones[0].unknown)
        self.assertEqual(utterance.phones[0].symbol, "xyz")
        self.assertIn("unknown_openjtalk_phone",
                      {item.code for item in utterance.diagnostics})

    def test_malformed_or_missing_fields_produce_diagnostics(self):
        raw = "xx^xx-k+a=xx/A:bad/F:broken/K:1+1-1"
        utterance = parse_openjtalk_labels(
            (raw,), source_text="か", normalized_reading="か"
        )
        self.assertIn("openjtalk_label_parse_issue",
                      {item.code for item in utterance.diagnostics})
        self.assertEqual(utterance.provenance["raw_labels"], [raw])

    def test_devoiced_openjtalk_vowel_is_canonical_but_marked(self):
        labels = (
            _label("xx^xx-s+U=xx", "0+1+1", "1_0#0_xx@1_1|1_1",
                   i_context="1-1@1+1&1-1|1+1", k_context="1+1-1"),
            _label("xx^s-U+xx=xx", "0+1+1", "1_0#0_xx@1_1|1_1",
                   i_context="1-1@1+1&1-1|1+1", k_context="1+1-1"),
        )
        utterance = parse_openjtalk_labels(
            labels, source_text="す", normalized_reading="す"
        )
        self.assertEqual([phone.symbol for phone in utterance.moras[0].phones],
                         ["s", "u"])
        self.assertTrue(utterance.moras[0].devoiced)
        self.assertEqual(utterance.moras[0].phones[-1].raw_symbol, "U")

    def test_openjtalk_long_vowel_uses_mora_structure(self):
        labels = (
            _label("xx^xx-s+u=u", "0+1+2", "2_0#0_xx@1_1|1_2",
                   i_context=_SIMPLE_I, k_context=_SIMPLE_K),
            _label("xx^s-u+u=xx", "0+1+2", "2_0#0_xx@1_1|1_2",
                   i_context=_SIMPLE_I, k_context=_SIMPLE_K),
            _label("s^u-u+xx=xx", "0+2+1", "2_0#0_xx@1_1|1_2",
                   i_context=_SIMPLE_I, k_context=_SIMPLE_K),
        )
        utterance = parse_openjtalk_labels(
            labels, source_text="スー", normalized_reading="スー"
        )
        self.assertEqual(len(utterance.moras), 2)
        self.assertEqual(utterance.moras[1].special_mora, "long_vowel")
        self.assertEqual([phone.symbol for phone in utterance.moras[1].phones],
                         ["u"])

    def test_raw_label_provenance_is_available_at_both_levels(self):
        utterance = parse_openjtalk_labels(
            SIMPLE_LABELS, source_text="かな", normalized_reading="かな"
        )
        self.assertEqual(utterance.provenance["raw_labels"], list(SIMPLE_LABELS))
        self.assertEqual(utterance.phones[1].raw_label, SIMPLE_LABELS[1])
        self.assertEqual(utterance.moras[0].provenance["raw_label_indices"], [1, 2])

    def test_morphology_nodes_are_preserved_and_mapped_by_mora(self):
        node = {
            "string": "です",
            "orig": "です",
            "read": "デス",
            "pron": "デス",
            "pos": "助動詞",
            "ctype": "特殊・デス",
            "cform": "基本形",
            "acc": 1,
            "mora_size": 2,
            "chain_flag": 1,
        }
        utterance = parse_openjtalk_labels(
            SIMPLE_LABELS,
            source_text="です",
            normalized_reading="カナ",
            morphology_nodes=(node,),
        )
        first = utterance.moras[0].provenance["morphology"]
        second = utterance.moras[1].provenance["morphology"]
        self.assertEqual(first["grammatical_role"], "polite_copula")
        self.assertTrue(first["function_word"])
        self.assertEqual(first["mora_position_in_node_zero_based"], 0)
        self.assertEqual(second["mora_position_in_node_zero_based"], 1)
        self.assertEqual(second["mora_count_in_node"], 2)
        self.assertEqual(second["ctype"], "特殊・デス")
        self.assertTrue(utterance.provenance["morphology_available"])

    def test_morphology_mismatch_is_diagnostic_not_destructive(self):
        utterance = parse_openjtalk_labels(
            SIMPLE_LABELS,
            source_text="かな",
            normalized_reading="かな",
            morphology_nodes=({
                "string": "かなかな",
                "pos": "名詞",
                "mora_size": 4,
            },),
        )
        self.assertEqual(len(utterance.moras), 2)
        self.assertEqual(
            [phone.symbol for phone in utterance.phones],
            ["sil", "k", "a", "n", "a", "sil"],
        )
        self.assertIn(
            "openjtalk_morphology_mora_mismatch",
            {item.code for item in utterance.diagnostics},
        )

    def test_explicit_canonical_phone_mapping(self):
        expected = {
            "a": ("a", False), "I": ("i", True), "U": ("u", True),
            "N": ("N", None), "cl": ("cl", None),
            "pau": ("pau", None), "sil": ("sil", None),
            "ky": ("ky", None), "ch": ("ch", None), "ts": ("ts", None),
        }
        for raw, (symbol, devoiced) in expected.items():
            with self.subTest(raw=raw):
                identity = canonicalize_openjtalk_phone(raw)
                self.assertEqual(identity.symbol, symbol)
                self.assertEqual(identity.devoiced, devoiced)
                self.assertFalse(identity.unknown)

    def test_fake_adapter_records_version_without_synthesizing(self):
        class FakeOpenJTalk:
            __version__ = "fixture-1"

            @staticmethod
            def g2p(text, kana=False):
                return "カナ" if kana else "k a n a"

            @staticmethod
            def extract_fullcontext(text):
                return list(SIMPLE_LABELS)

            @staticmethod
            def run_frontend(text):
                return [{
                    "string": "かな",
                    "pos": "名詞",
                    "mora_size": 2,
                }]

        utterance = OpenJTalkJapaneseFrontend(FakeOpenJTalk).analyze("かな")
        self.assertEqual(utterance.frontend_name, "openjtalk")
        self.assertEqual(utterance.frontend_version, "fixture-1")
        self.assertFalse(utterance.provenance["duration_authority"])
        self.assertFalse(utterance.provenance["f0_authority"])
        self.assertEqual(
            utterance.moras[0].provenance["morphology"]["grammatical_role"],
            "content_word",
        )


class IsolationAndOptionalIntegrationTests(unittest.TestCase):
    def test_installed_module_without_dictionary_is_not_operational(self):
        class IncompleteOpenJTalk:
            OPEN_JTALK_DICT_DIR = "missing-open-jtalk-dictionary"

            @staticmethod
            def g2p(_text, kana=False):
                return "kana" if kana else "k a n a"

            @staticmethod
            def extract_fullcontext(_text):
                return []

        with mock.patch(
            "japanese_openjtalk.importlib.util.find_spec",
            return_value=object(),
        ), mock.patch(
            "japanese_openjtalk.importlib.import_module",
            return_value=IncompleteOpenJTalk,
        ):
            available, reason = pyopenjtalk_status()
        self.assertFalse(available)
        self.assertIn("dictionary", reason)

    def test_windows_bytes_dictionary_path_is_accepted(self):
        class BytesPathOpenJTalk:
            g2p = staticmethod(lambda _text: "k a")
            extract_fullcontext = staticmethod(lambda _text: [])
            OPEN_JTALK_DICT_DIR = os.fsencode(str(Path(__file__).parent))

        with mock.patch(
            "japanese_openjtalk.importlib.util.find_spec",
            return_value=object(),
        ), mock.patch(
            "japanese_openjtalk.importlib.import_module",
            return_value=BytesPathOpenJTalk,
        ):
            available, reason = pyopenjtalk_status()
        self.assertTrue(available)
        self.assertEqual(reason, "")

    def test_importing_frontend_does_not_import_english_pipeline(self):
        module_dir = Path(sys.modules["japanese_frontend"].__file__).parent
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(module_dir)!r}); "
            "import japanese_frontend; "
            "print(any(name in sys.modules for name in "
            "('utau2festvox', 'festvox_core', 'synth_diphone')))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "False")

    @unittest.skipUnless(
        is_pyopenjtalk_available(), "optional pyopenjtalk is not installed"
    )
    def test_local_pyopenjtalk_integration(self):
        utterance = OpenJTalkJapaneseFrontend().analyze("かな？")
        self.assertEqual(utterance.frontend_name, "openjtalk")
        self.assertGreater(len(utterance.moras), 0)
        self.assertGreater(len(utterance.provenance["raw_labels"]), 0)

    @unittest.skipUnless(
        is_pyopenjtalk_available(), "optional pyopenjtalk is not installed"
    )
    def test_news_sentence_quotes_keep_reading_aligned(self):
        text = (
            "24\u65e5\u3082\u6771\u6d77\u304b\u3089\u4e5d\u5dde\u3092"
            "\u4e2d\u5fc3\u306b\u6c17\u6e29\u304c\u4e0a\u304c\u308a"
            "\u5c90\u961c\u770c\u3068\u4e09\u91cd\u770c\u3067\u306f"
            "40\u5ea6\u4ee5\u4e0a\u3092\u89b3\u6e2c\u3057\u3066"
            "\u300c\u9177\u6691\u65e5\u300d\u3068\u306a\u308a"
            "\u307e\u3057\u305f\u3002"
        )
        utterance = OpenJTalkJapaneseFrontend().analyze(text)
        self.assertNotIn(
            "openjtalk_reading_mora_mismatch",
            {item.code for item in utterance.diagnostics},
        )
        liquid_readings = [
            mora.reading
            for mora in utterance.moras
            if any(phone.symbol == "r" for phone in mora.phones)
        ]
        self.assertEqual(
            liquid_readings,
            ["\u3089", "\u308a", "\u308a"],
        )

    @unittest.skipUnless(
        is_pyopenjtalk_available(), "optional pyopenjtalk is not installed"
    )
    def test_quoted_topic_particle_does_not_create_a_phrase_gap(self):
        text = (
            "\u300c\u71b1\u4e2d\u75c7\u8b66\u6212\u30a2\u30e9\u30fc"
            "\u30c8\u300d\u306f\u95a2\u6771\u304b\u3089\u6c96\u7e04"
            "\u306b\u304b\u3051\u3066\u306e32\u90fd\u5e9c\u770c\u306b"
            "\u767a\u8868\u3055\u308c\u3066\u3044\u307e\u3059\u3002"
        )
        utterance = OpenJTalkJapaneseFrontend().analyze(text)
        self.assertEqual(len(utterance.phrases), 1)
        self.assertIn(
            "openjtalk_inline_bracket_pause",
            {item.code for item in utterance.diagnostics},
        )
        self.assertNotIn(
            "openjtalk_source_phrase_mismatch",
            {item.code for item in utterance.diagnostics},
        )


if __name__ == "__main__":
    unittest.main()
