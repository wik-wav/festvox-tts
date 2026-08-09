import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path

import japanese_editing as je
import japanese_frontend as frontend
import japanese_profiles as jp


def _digest_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_silence(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\0\0" * 4410)


class JapaneseEditingTests(unittest.TestCase):
    def setUp(self):
        self.utterance = frontend.analyze_japanese(
            "きゃく。ねこ？", mode="kana"
        )

    def test_state_and_utterance_round_trip_are_serializable(self):
        state = je.new_edit_state(self.utterance, frontend_mode="kana")
        state["mora_pitch_offsets_cents"] = {"0": 75}
        state["mora_voicing_overrides"] = {"0": 0.375}
        serialized = json.loads(json.dumps(state, ensure_ascii=False))
        normalized = je.normalize_edit_state(serialized)
        restored = je.utterance_from_dict(normalized["utterance"])

        self.assertEqual(restored.to_dict(), self.utterance.to_dict())
        self.assertEqual(normalized["mora_pitch_offsets_cents"], {"0": 75})
        self.assertEqual(normalized["mora_voicing_overrides"],
                         {"0": 0.375})
        self.assertEqual(normalized["continuous_pitch_authority"],
                         "pitch_override")

    def test_legacy_overlay_names_migrate_and_values_are_bounded(self):
        state = je.normalize_edit_state({
            "accent_edits": {2: {"accent_state": "unaccented"}},
            "mora_pitch_offsets": {0: 9999, 1: -9999, "bad": "x"},
            "mora_voicing_overrides": {0: -2, 1: 4, 2: "bad"},
        })

        self.assertEqual(state["accent_overrides"]["2"]["accent_state"],
                         "unaccented")
        self.assertEqual(state["mora_pitch_offsets_cents"],
                         {"0": 600, "1": -600})
        self.assertEqual(state["mora_voicing_overrides"],
                         {"0": 0.0, "1": 1.0})

    def test_new_text_clears_occurrence_edits_but_keeps_bank_settings(self):
        state = je.new_edit_state(self.utterance, frontend_mode="kana")
        state["accent_overrides"] = {"0": {"accent_state": "unaccented"}}
        state["mora_pitch_offsets_cents"] = {"0": 120}
        state["mora_voicing_overrides"] = {"0": 0.25}
        state["manual_candidate_overrides"] = {"0": "jc_manual"}
        state["bank_analysis"] = {"source_count": 12}
        state["profile_path"] = "profile.json"
        state["needs_voice_rebuild"] = True
        state["dynamic_multipitch"] = True
        state["voice_color"] = "Power"

        same = je.reconcile_analyzed_utterance(state, self.utterance)
        replacement = frontend.analyze_japanese("ねこ", mode="kana")
        changed = je.reconcile_analyzed_utterance(state, replacement)

        self.assertEqual(same["mora_pitch_offsets_cents"], {"0": 120})
        self.assertEqual(changed["accent_overrides"], {})
        self.assertEqual(changed["mora_pitch_offsets_cents"], {})
        self.assertEqual(changed["mora_voicing_overrides"], {})
        self.assertEqual(changed["manual_candidate_overrides"], {})
        self.assertEqual(changed["bank_analysis"], {"source_count": 12})
        self.assertEqual(changed["profile_path"], "profile.json")
        self.assertTrue(changed["needs_voice_rebuild"])
        self.assertTrue(changed["dynamic_multipitch"])
        self.assertEqual(changed["voice_color"], "Power")

    def test_dynamic_routing_uses_final_mora_pitch_and_is_deterministic(self):
        state = je.new_edit_state(self.utterance, frontend_mode="kana")
        state["dynamic_multipitch"] = True
        state["mora_pitch_offsets_cents"] = {"0": 100}
        baseline = je.create_edited_plan(self.utterance, state)
        edge = next(
            index for index in range(len(baseline.segments) - 1)
            if baseline.segments[index].phone != "pau"
            and baseline.segments[index + 1].phone != "pau"
        )
        pair = "%s-%s" % (
            baseline.segments[edge].phone,
            baseline.segments[edge + 1].phone,
        )
        runtime = {
            "language": "ja",
            "candidate_units": {},
            "subbanks": [],
            "alternatives": {pair: [
                {"candidate_id": "low", "left_name": "low",
                 "source_pitch_tags": ["C2"]},
                {"candidate_id": "near", "left_name": "near",
                 "source_pitch_tags": ["G3"]},
            ]},
        }

        first = je.create_edited_plan(
            self.utterance, state, runtime_metadata=runtime,
            allow_experimental_routing=True)
        second = je.create_edited_plan(
            self.utterance, state, runtime_metadata=runtime,
            allow_experimental_routing=True)

        self.assertEqual(first.unit_overrides[edge], "near")
        self.assertEqual(first.to_json_bytes(), second.to_json_bytes())

    def test_stable_plan_ignores_stored_multipitch_and_color_routing(self):
        state = je.new_edit_state(self.utterance, frontend_mode="kana")
        state["dynamic_multipitch"] = True
        state["voice_color"] = "Power"

        plan = je.create_edited_plan(
            self.utterance, state,
            runtime_metadata={"language": "ja", "alternatives": {}})

        self.assertEqual(plan.unit_overrides, {})
        self.assertIn(
            "experimental_routing_disabled",
            [item.code for item in plan.diagnostics],
        )

    def test_optional_baseline_stays_below_manual_accent(self):
        state = je.new_edit_state(self.utterance, frontend_mode="kana")
        accent = self.utterance.accent_phrases[0]
        state["accent_overrides"] = {
            str(accent.index): {"accent_state": "unaccented"}
        }
        structural = je.create_edited_plan(self.utterance, state)
        with tempfile.TemporaryDirectory() as root:
            trajectory = Path(root) / "trajectory.json"
            trajectory.write_text(json.dumps({
                "language": "ja",
                "phones": structural.phones,
                "durations": [round(duration * 1.1, 6)
                              for _phone, duration in
                              structural.segment_durations],
                "f0_targets": [[time, 300.0]
                               for time, _hz in structural.pitch_targets],
            }), encoding="utf-8")
            state["baseline_provider"] = "external_hts"
            state["external_hts_trajectory"] = str(trajectory)
            result = je.create_edited_plan(self.utterance, state)

        self.assertAlmostEqual(
            result.segments[0].duration,
            structural.segments[0].duration * 1.1,
            places=5,
        )
        self.assertTrue(result.f0_targets)
        self.assertNotEqual(result.f0_targets[0].hz, 300.0)
        self.assertNotIn("external_hts_baseline", result.f0_targets[0].kind)
        for mora in result.mora_timings:
            self.assertAlmostEqual(
                mora.final_duration,
                sum(result.segments[item.segment_index].duration
                    for item in mora.phone_allocation),
                places=6,
            )

    def test_accent_question_and_boundary_overrides_are_structural(self):
        accent = self.utterance.accent_phrases[0]
        phrase = self.utterance.phrases[0]
        state = je.new_edit_state(self.utterance)
        state["accent_overrides"] = {
            str(accent.index): {
                "accent_state": "accented",
                "accent_nucleus": len(accent.moras) - 1,
                "boundary_strength": 2,
                "interrogative": True,
            }
        }
        state["phrase_overrides"] = {
            str(phrase.index): {
                "interrogative": True,
                "boundary_strength": 1,
            }
        }

        edited = je.apply_linguistic_edits(self.utterance, state)
        edited_accent = edited.accent_phrases[0]
        self.assertEqual(edited_accent.accent_state, "accented")
        self.assertEqual(edited_accent.accent_nucleus,
                         len(edited_accent.moras) - 1)
        self.assertTrue(edited_accent.interrogative)
        self.assertEqual(edited.phrases[0].boundary_strength, 1)
        self.assertTrue(edited.phrases[0].interrogative)

    def test_accent_phrase_split_rebuilds_mora_and_phone_membership(self):
        utterance = frontend.analyze_japanese(
            "\u304b\u306a\u304b\u306a", mode="kana")
        phrase = utterance.phrases[0]
        state = je.new_edit_state(utterance)
        split_mora = phrase.moras[2]
        state["accent_phrase_boundaries"] = {
            str(phrase.index): [split_mora.index]
        }

        edited = je.apply_linguistic_edits(utterance, state)

        self.assertEqual(len(edited.phrases[0].accent_phrases), 2)
        self.assertEqual(
            [mora.index for mora in edited.phrases[0].accent_phrases[0].moras],
            [phrase.moras[0].index, phrase.moras[1].index],
        )
        self.assertEqual(
            [mora.index for mora in edited.phrases[0].accent_phrases[1].moras],
            [phrase.moras[2].index, phrase.moras[3].index],
        )
        top_level = {phone.index: phone for phone in edited.phones}
        for accent in edited.accent_phrases:
            for mora in accent.moras:
                self.assertEqual(mora.accent_phrase_index, accent.index)
                for phone in mora.phones:
                    self.assertEqual(phone.accent_phrase_index, accent.index)
                    self.assertEqual(
                        top_level[phone.index].accent_phrase_index,
                        accent.index,
                    )

    def test_empty_boundary_override_merges_analyzed_accent_phrases(self):
        utterance = frontend.analyze_japanese(
            "\u304b\u306a\u304b\u306a", mode="kana")
        phrase = utterance.phrases[0]
        split_state = je.new_edit_state(utterance)
        split_state["accent_phrase_boundaries"] = {
            str(phrase.index): [phrase.moras[2].index]
        }
        split = je.apply_linguistic_edits(utterance, split_state)
        stored = je.new_edit_state(split)
        stored["accent_phrase_boundaries"] = {str(phrase.index): []}

        normalized = je.normalize_edit_state(stored)
        merged = je.apply_linguistic_edits(split, normalized)

        self.assertIn(str(phrase.index),
                      normalized["accent_phrase_boundaries"])
        self.assertEqual(len(merged.phrases[0].accent_phrases), 1)
        self.assertEqual(len(merged.phrases[0].moras), 4)

    def test_mora_pitch_offsets_modify_baseline_but_not_continuous_points(self):
        mora = self.utterance.moras[0]
        state = je.new_edit_state(self.utterance)
        baseline = je.create_edited_plan(
            self.utterance, state, base_pitch_hz=180.0
        )
        state["mora_pitch_offsets_cents"] = {str(mora.index): 120}
        shifted = je.create_edited_plan(
            self.utterance, state, base_pitch_hz=180.0
        )
        baseline_target = next(item for item in baseline.f0_targets
                               if item.mora_index == mora.index)
        shifted_target = next(item for item in shifted.f0_targets
                              if item.mora_index == mora.index)
        self.assertGreater(shifted_target.hz, baseline_target.hz)
        self.assertAlmostEqual(
            shifted_target.hz / baseline_target.hz,
            2.0 ** (120.0 / 1200.0), places=3,
        )
        continuous = [(0.1, 202.0), (0.2, 199.0)]
        self.assertEqual(
            je.final_pitch_targets(shifted.pitch_targets, continuous),
            continuous,
        )

    def test_manual_candidate_overrides_survive_unrelated_pitch_edits(self):
        state = je.new_edit_state(self.utterance)
        state["manual_candidate_overrides"] = {"0": "jc_abc"}
        before = je.normalize_edit_state(state)
        state["mora_pitch_offsets_cents"] = {"1": 40}
        after = je.normalize_edit_state(state)

        self.assertEqual(before["manual_candidate_overrides"],
                         after["manual_candidate_overrides"])
        self.assertEqual(je.invalidation_for_edit("mora_pitch"), "rerender")
        self.assertEqual(je.invalidation_for_edit("mora_voicing"),
                         "rerender")
        self.assertEqual(je.invalidation_for_edit("alias_override"), "rebuild")

    def test_bank_preview_is_traceable_read_only_and_profile_guarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            bank = Path(temporary) / "bank"
            bank.mkdir()
            _write_silence(bank / "a.wav")
            _write_silence(bank / "mystery.wav")
            (bank / "oto.ini").write_text(
                "a.wav=a,0,80,-100,40,20\n"
                "mystery.wav=??? token,0,80,-100,40,20\n",
                encoding="utf-8",
            )
            before = _digest_tree(bank)

            analysis = je.analyze_bank(bank)
            snapshot = analysis.to_state_dict()

            self.assertTrue(snapshot["all_entries_traceable"])
            self.assertEqual(snapshot["source_entry_count"], 2)
            self.assertGreaterEqual(snapshot["unresolved_count"], 1)
            self.assertEqual(_digest_tree(bank), before)
            with self.assertRaises(ValueError):
                je.write_analysis_profile(
                    analysis, bank / "japanese-profile.json"
                )
            target = Path(temporary) / "profile.json"
            je.write_analysis_profile(analysis, target)
            self.assertEqual(
                jp.load_profile(target).to_dict(),
                analysis.profile.to_dict(),
            )
            self.assertEqual(_digest_tree(bank), before)

    def test_exact_alias_override_is_retained_in_profile(self):
        base = jp.JapaneseBankProfile(
            bank_configuration="auto",
            inferred_configuration="mixed",
        )
        override = jp.JapaneseAliasOverride(
            role="mora_cv", family="cv", mora="ka"
        )
        edited = je.profile_with_override(base, "alias:か", override)

        self.assertNotIn("alias:か", base.alias_overrides)
        self.assertEqual(edited.alias_overrides["alias:か"], override)

    def test_edited_plan_is_byte_deterministic(self):
        state = je.new_edit_state(self.utterance, frontend_mode="kana")
        state["mora_pitch_offsets_cents"] = {"0": 35, "2": -20}

        first = je.create_edited_plan(
            self.utterance, state, base_pitch_hz=180.0)
        second = je.create_edited_plan(
            self.utterance, state, base_pitch_hz=180.0)

        self.assertEqual(first.to_json_bytes(), second.to_json_bytes())


if __name__ == "__main__":
    unittest.main()
