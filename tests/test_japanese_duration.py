import json
import math
from dataclasses import replace
import unittest
from unittest import mock

from japanese_duration import (
    build_duration_contexts,
    clear_duration_model_cache,
    duration_model_cache_info,
    load_duration_priors,
    predict_mora_durations,
)
from japanese_frontend import analyze_japanese
from japanese_openjtalk import parse_openjtalk_labels
import japanese_synthesis as js


def _label(quinphone, a_context, f_context):
    return (
        f"{quinphone}/A:{a_context}/B:xx-xx_xx/C:xx_xx+xx/"
        "D:xx+xx_xx/E:xx_xx!xx_xx-xx/"
        f"F:{f_context}/G:xx_xx%xx_xx_xx/H:xx_xx/"
        "I:1-2@1+1&1-1|1+2/J:xx_xx/K:1+1-2"
    )


OPENJTALK_LABELS = (
    _label("xx^xx-sil+k=a", "xx+xx+xx",
           "xx_xx#xx_xx@xx_xx|xx_xx"),
    _label("xx^sil-k+a=n", "0+1+2", "2_1#0_xx@1_1|1_2"),
    _label("sil^k-a+n=a", "0+1+2", "2_1#0_xx@1_1|1_2"),
    _label("k^a-n+a=sil", "1+2+1", "2_1#0_xx@1_1|1_2"),
    _label("a^n-a+sil=xx", "1+2+1", "2_1#0_xx@1_1|1_2"),
    _label("n^a-sil+xx=xx", "xx+xx+xx",
           "xx_xx#xx_xx@xx_xx|xx_xx"),
)


class JapaneseDurationModelTests(unittest.TestCase):
    def test_versioned_priors_are_deterministic_and_source_scaled(self):
        clear_duration_model_cache()
        first = load_duration_priors()
        second = load_duration_priors()

        self.assertIs(first, second)
        self.assertEqual(duration_model_cache_info()["entries"], 1)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.schema_version, 1)
        self.assertEqual(
            first.fit_provenance["absolute_scale"],
            "project_source_speaker_engineering_anchor_with_kokoro_relative_class_ratios",
        )
        self.assertEqual(first.model_id,
                         "japanese_contextual_source_anchor_kokoro_b453f6caf042_v7")
        self.assertEqual(first.acoustic_edge_compensation_ms,
                         {"phrase_initial_vowel": 50.0,
                          "phrase_final_vowel": 0.0})
        self.assertEqual(first.class_target_ratio_bounds["moraic_nasal"],
                         (0.65, 1.35))
        self.assertEqual(first.mora_allocation_seconds["cv_vowel"], 0.061)
        self.assertEqual(first.mora_allocation_seconds["stop"], 0.029)
        with self.assertRaises(TypeError):
            first.coefficients["poison"] = 1.0
        self.assertNotIn("poison", load_duration_priors().coefficients)
        json.dumps(first.to_dict(), sort_keys=True)

    def test_context_preserves_named_openjtalk_a_f_i_k_fields(self):
        utterance = parse_openjtalk_labels(
            OPENJTALK_LABELS,
            source_text="kana",
            normalized_reading="kana",
        )
        mora = utterance.moras[0]

        contexts = build_duration_contexts(
            utterance, mora, [phone.symbol for phone in mora.phones]
        )
        vowel = contexts[-1]

        self.assertEqual(vowel.openjtalk_mora_forward, 1)
        self.assertEqual(vowel.openjtalk_mora_backward, 2)
        self.assertEqual(vowel.openjtalk_breath_group_forward, 1)
        self.assertEqual(vowel.raw_a, "0+1+2")
        self.assertEqual(vowel.raw_f, "2_1#0_xx@1_1|1_2")
        self.assertEqual(vowel.raw_i, "1-2@1+1&1-1|1+2")
        self.assertEqual(vowel.raw_k, "1+1-2")

    def test_context_preserves_openjtalk_morphology(self):
        utterance = parse_openjtalk_labels(
            OPENJTALK_LABELS,
            source_text="です",
            normalized_reading="かな",
            morphology_nodes=({
                "string": "です",
                "orig": "です",
                "pos": "助動詞",
                "ctype": "特殊・デス",
                "cform": "基本形",
                "mora_size": 2,
            },),
        )
        first = build_duration_contexts(
            utterance,
            utterance.moras[0],
            [phone.symbol for phone in utterance.moras[0].phones],
        )
        second = build_duration_contexts(
            utterance,
            utterance.moras[1],
            [phone.symbol for phone in utterance.moras[1].phones],
        )
        self.assertTrue(all(item.lexical_surface == "です" for item in first))
        self.assertTrue(all(item.grammatical_role == "polite_copula"
                            for item in first + second))
        self.assertEqual(first[0].mora_position_in_node, 0)
        self.assertEqual(second[0].mora_position_in_node, 1)
        self.assertEqual(second[0].mora_count_in_node, 2)
        self.assertTrue(second[0].function_word)

    def test_replicated_grammar_effects_are_conservative(self):
        utterance = analyze_japanese("ka", mode="kana")
        base = build_duration_contexts(
            utterance, utterance.moras[0], ("k", "a"))
        auxiliary = tuple(replace(item, grammatical_role="auxiliary")
                          for item in base)
        negative = tuple(replace(item, grammatical_role="negative_auxiliary")
                         for item in base)
        baseline = predict_mora_durations(
            base, (0.05, 0.09), (0.05, 0.08), speed=1.0)
        auxiliary_rows = predict_mora_durations(
            auxiliary, (0.05, 0.09), (0.05, 0.08), speed=1.0)
        negative_rows = predict_mora_durations(
            negative, (0.05, 0.09), (0.05, 0.08), speed=1.0)
        self.assertLess(sum(row.predicted_duration for row in auxiliary_rows),
                        sum(row.predicted_duration for row in baseline))
        self.assertLess(sum(row.predicted_duration for row in negative_rows),
                        sum(row.predicted_duration for row in baseline))
        self.assertIn("auxiliary", auxiliary_rows[0].effects)
        self.assertIn("negative_auxiliary", negative_rows[0].effects)

    def test_acoustic_edge_compensation_is_bounded_and_explicit(self):
        utterance = analyze_japanese("a", mode="kana")
        context = build_duration_contexts(
            utterance, utterance.moras[0], ("a",))
        prediction = predict_mora_durations(
            context, (0.10,), (0.11,), speed=1.0)[0]
        self.assertIn("phrase_initial_vowel_acoustic_compensation_seconds",
                      prediction.effects)
        self.assertNotIn("phrase_final_vowel_acoustic_compensation_seconds",
                         prediction.effects)
        self.assertGreaterEqual(prediction.predicted_duration, 0.030)
        self.assertGreaterEqual(prediction.predicted_duration, 0.45 * 0.11)

    def test_source_geometry_is_relative_to_bank_profile(self):
        utterance = analyze_japanese("kaka", mode="kana")
        mora = utterance.moras[0]
        contexts = build_duration_contexts(utterance, mora, ("k", "a"))

        predictions = predict_mora_durations(
            contexts,
            (0.07, 0.14),
            (0.05, 0.095),
            speed=1.0,
            source_profile_references=(0.05, 0.10),
        )

        self.assertEqual(
            [item.baseline_source for item in predictions],
            ["source_unit_geometry_profiled",
             "source_unit_geometry_profiled"],
        )
        self.assertEqual(
            [item.source_baseline_duration for item in predictions],
            [0.056, 0.1007],
        )
        self.assertEqual(
            [item.source_geometry_ratio for item in predictions],
            [1.4, 1.4],
        )
        self.assertEqual(
            [item.source_geometry_ratio_bounded for item in predictions],
            [1.12, 1.06],
        )
        self.assertGreater(predictions[1].predicted_duration, 0.08)
        self.assertFalse(any(
            "acoustic_compensation" in name
            for name in predictions[1].effects
        ))

    def test_speed_compresses_vowels_more_than_stop_releases(self):
        utterance = analyze_japanese("ka", mode="kana")
        contexts = build_duration_contexts(
            utterance, utterance.moras[0], ("k", "a")
        )
        normal = predict_mora_durations(
            contexts, (0.06, 0.10), (0.05, 0.095), speed=1.0
        )
        fast = predict_mora_durations(
            contexts, (0.06, 0.10), (0.05, 0.095), speed=2.0
        )

        stop_ratio = fast[0].predicted_duration / normal[0].predicted_duration
        vowel_ratio = fast[1].predicted_duration / normal[1].predicted_duration
        self.assertGreater(stop_ratio, vowel_ratio)

    def test_devoicing_is_shortened_without_becoming_zero_duration(self):
        utterance = analyze_japanese("kutsu", mode="kana")
        first = utterance.moras[0]
        contexts = build_duration_contexts(
            utterance, first,
            tuple(phone.symbol for phone in first.phones),
        )
        predictions = predict_mora_durations(
            contexts,
            (0.05, 0.10),
            (0.05, 0.095),
            speed=1.0,
        )
        vowel = predictions[-1]

        self.assertIn("devoiced_high_vowel", vowel.effects)
        self.assertGreater(vowel.predicted_duration, 0.01)
        self.assertLess(vowel.predicted_duration, 0.08)

    def test_consecutive_devoicing_environment_is_suppressed(self):
        utterance = analyze_japanese("kutsu", mode="kana")
        second = utterance.moras[1]
        contexts = build_duration_contexts(
            utterance, second,
            tuple(phone.symbol for phone in second.phones),
        )
        predictions = predict_mora_durations(
            contexts,
            (0.075, 0.09),
            (0.075, 0.08),
            speed=1.0,
        )

        self.assertTrue(contexts[-1].previous_mora_devoiced)
        self.assertNotIn("devoiced_high_vowel", predictions[-1].effects)
        self.assertIn(
            "consecutive_devoicing_avoidance", predictions[-1].effects
        )

    def test_normal_plan_uses_contextual_model_and_legacy_is_available(self):
        def choice(start, middle, end):
            return {"source_slice": {
                "start": start,
                "phone_boundary": middle,
                "end": end,
            }}

        runtime = {
            "language": "ja",
            "alternatives": {
                "pau-k": [choice(0.0, 0.02, 0.05)],
                "k-a": [choice(0.0, 0.04, 0.10)],
                "a-pau": [choice(0.0, 0.08, 0.11)],
            },
        }
        utterance = analyze_japanese("ka", mode="kana")

        with mock.patch.object(
                js, "load_duration_priors",
                wraps=js.load_duration_priors) as load_priors:
            contextual = js.create_synthesis_plan(
                utterance, runtime_metadata=runtime
            )
        self.assertEqual(load_priors.call_count, 1)
        legacy = js.create_synthesis_plan(
            utterance, runtime_metadata=runtime, duration_model="legacy"
        )

        self.assertEqual(contextual.duration_model, "contextual")
        self.assertEqual(legacy.duration_model, "legacy")
        self.assertTrue(all(
            item.baseline_source == "source_unit_geometry_profiled"
            for item in contextual.mora_timings[0].phone_allocation
        ))
        self.assertNotEqual(
            contextual.mora_timings[0].final_duration,
            legacy.mora_timings[0].final_duration,
        )

    def test_contextual_duration_snapshot_covers_special_morae(self):
        utterance = analyze_japanese("kitte koon", mode="kana")
        plan = js.create_synthesis_plan(utterance)

        self.assertEqual(
            [round(item.final_duration, 6) for item in plan.mora_timings],
            [0.090673, 0.078098, 0.117335, 0.117335, 0.11, 0.099031],
        )
        self.assertTrue(all(
            item.duration_model_id == plan.duration_model_id
            for item in plan.mora_timings
        ))

    def test_mapped_moraic_nasal_keeps_semantic_class_and_bounded_length(self):
        utterance = analyze_japanese("an", mode="kana")
        mora = next(item for item in utterance.moras
                    if item.special_mora == "moraic_nasal")
        contexts = build_duration_contexts(utterance, mora, ("nn",))

        prediction = predict_mora_durations(
            contexts,
            (0.55,),
            (0.115,),
            speed=1.0,
            source_profile_references=(0.10,),
        )[0]

        self.assertEqual(prediction.context.phone, "nn")
        self.assertEqual(prediction.context.phone_class, "moraic_nasal")
        self.assertLessEqual(prediction.predicted_duration, 0.115 * 1.25)
        self.assertIn("class_duration_bound", prediction.effects)

    def test_moraic_nasal_is_shorter_before_voiceless_obstruent(self):
        utterance = analyze_japanese("an", mode="kana")
        mora = next(item for item in utterance.moras
                    if item.special_mora == "moraic_nasal")
        context = replace(
            build_duration_contexts(utterance, mora, ("nn",))[0],
            accent_phrase_final=False,
            phrase_final=False,
            utterance_final=False,
            boundary_strength=0,
        )
        voiced = replace(context, following_phone="b")
        voiceless = replace(context, following_phone="k")

        voiced_prediction, voiceless_prediction = predict_mora_durations(
            (voiced, voiceless),
            (0.115, 0.115),
            (0.115, 0.115),
            speed=1.0,
            source_profile_references=(0.115, 0.115),
        )

        self.assertLess(voiceless_prediction.predicted_duration,
                        voiced_prediction.predicted_duration)
        self.assertIn("moraic_nasal_before_voiceless",
                      voiceless_prediction.effects)

    def test_plans_mark_canonical_and_integrated_nasal_as_timing_nuclei(self):
        utterance = analyze_japanese("an", mode="kana")
        japanese_only = js.create_synthesis_plan(utterance)
        integrated = js.create_synthesis_plan(utterance, runtime_metadata={
            "language": "en",
            "supported_languages": ["en", "ja"],
            "phones": ["a", "nn", "pau"],
            "japanese_phoneme_map": {
                "grapheme_to_phones": {"あ": ["a"], "ん": ["nn"]},
                "canonical_fallbacks": {"N": "nn"},
                "moraic_nasal_routes": {"default": "nn"},
                "timing_multipliers": {},
            },
        })

        pure_nasal = next(segment for segment in japanese_only.segments
                          if segment.mora_index == 1)
        mapped_nasal = next(segment for segment in integrated.segments
                            if segment.mora_index == 1)
        self.assertEqual((pure_nasal.phone, pure_nasal.timing_role),
                         ("N", "moraic_nasal"))
        self.assertEqual((mapped_nasal.phone, mapped_nasal.timing_role),
                         ("nn", "moraic_nasal"))
        self.assertLessEqual(
            integrated.mora_timings[1].final_duration, 0.115 * 1.25
        )


if __name__ == "__main__":
    unittest.main()
