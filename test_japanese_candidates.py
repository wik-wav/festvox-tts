import json
import sys
import tempfile
import unittest
from pathlib import Path

import japanese_candidates as jc
import japanese_profiles as jp


KA = "\u304b"
KI = "\u304d"
SA = "\u3055"


class JapaneseCandidateCompilerTests(unittest.TestCase):
    @staticmethod
    def write_oto(path, aliases, *, encoding="utf-8", create_wavs=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for index, alias in enumerate(aliases):
            wav_name = f"sample_{index}.wav"
            lines.append(f"{wav_name}={alias},0,100,-200,50,10")
            if create_wavs:
                (path.parent / wav_name).write_bytes(b"RIFF")
        path.write_bytes(("\n".join(lines) + "\n").encode(encoding))

    @staticmethod
    def write_character_yaml(path, *, suffix="F3"):
        path.write_text(
            "text_file_encoding: utf-8\n"
            "subbanks:\n"
            "- color: \"\"\n"
            "  prefix: \"\"\n"
            f"  suffix: {suffix}\n"
            "  tone_ranges:\n"
            "  - F3-G3\n"
            "- color: Soft\n"
            "  prefix: S_\n"
            "  suffix: F3S\n"
            "  tone_ranges: [F3-G3]\n",
            encoding="utf-8",
        )

    def test_profile_inference_finds_parent_character_yaml(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            scope = bank / "F3"
            bank.mkdir()
            self.write_character_yaml(bank / "character.yaml")
            self.write_oto(scope / "oto.ini", [f"{KA}F3"])

            profile = jp.infer_bank_profile(scope)

            self.assertEqual(profile.source_scope, "F3")
            self.assertEqual(profile.default_encoding, "utf-8")
            self.assertIn("F3", profile.alias_suffixes)
            self.assertIn("F3S", profile.alias_suffixes)
            self.assertEqual(profile.inferred_configuration, "cv")
            self.assertEqual(profile.effective_configuration, "cv")
            self.assertEqual(profile.metadata_files["character.yaml"]["path"],
                             "character.yaml")

    def test_unquoted_sharp_pitch_suffix_is_not_parsed_as_yaml_comment(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            (bank / "character.yaml").write_text(
                "text_file_encoding: utf-8\n"
                "subbanks:\n"
                "- color: Headvoice # retained comment\n"
                "  prefix: \"\"\n"
                "  suffix: C#4H\n"
                "  tone_ranges: [C#4-D#4]\n",
                encoding="utf-8",
            )
            self.write_oto(bank / "oto.ini", [f"{KA}1C#4H"])

            profile = jp.infer_bank_profile(bank)
            graph = jc.compile_candidate_graph(bank, profile=profile)

            self.assertIn("C#4H", profile.alias_suffixes)
            self.assertEqual(graph.candidates[0].role, "mora_cv")
            self.assertEqual(
                graph.candidates[0].source.alternative_numbers, (1,)
            )

    def test_explicit_profile_configuration_overrides_inference_only(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", [KA])

            profile = jp.infer_bank_profile(
                bank, bank_configuration="mixed"
            )

            self.assertEqual(profile.inferred_configuration, "cv")
            self.assertEqual(profile.effective_configuration, "mixed")

    def test_profile_serialization_is_relative_and_deterministic(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "private-bank"
            bank.mkdir()
            self.write_character_yaml(bank / "character.yaml")
            self.write_oto(bank / "oto.ini", [f"{KA}F3"])
            profile = jp.infer_bank_profile(bank)

            first = jp.profile_json_bytes(profile)
            second = jp.profile_json_bytes(profile)
            loaded = jp.JapaneseBankProfile.from_dict(
                json.loads(first.decode("utf-8"))
            )

            self.assertEqual(first, second)
            self.assertEqual(loaded.to_dict(), profile.to_dict())
            self.assertNotIn(str(bank), first.decode("utf-8"))

    def test_profile_serializes_bank_defined_moraic_nasal_allophones(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", ["\u3093"])
            rules = {
                "coronal": jp.JapaneseMoraicNasalAllophone(
                    mora_aliases=("\u3093",),
                    context_aliases=("n",),
                    following_phones=("t", "d", "n"),
                ),
                "uvular": jp.JapaneseMoraicNasalAllophone(
                    mora_aliases=("\u3093n", "\u30931"),
                    context_aliases=("nn",),
                    following_phones=("s",),
                    default=True,
                ),
            }
            profile = jp.infer_bank_profile(
                bank,
                bank_configuration="cvvc",
                moraic_nasal_allophones=rules,
            )

            loaded = jp.JapaneseBankProfile.from_dict(profile.to_dict())

            self.assertEqual(loaded.to_dict(), profile.to_dict())
            self.assertTrue(
                loaded.moraic_nasal_allophones["uvular"].default
            )

    def test_profile_rejects_ambiguous_moraic_nasal_routes(self):
        with self.assertRaisesRegex(ValueError, "assigned to both"):
            jp.JapaneseBankProfile(moraic_nasal_allophones={
                "first": jp.JapaneseMoraicNasalAllophone(
                    context_aliases=("n",), following_phones=("t",)
                ),
                "second": jp.JapaneseMoraicNasalAllophone(
                    context_aliases=("m",), following_phones=("t",)
                ),
            })

    def test_cv_vcv_vc_and_release_roles_are_structurally_distinct(self):
        aliases = [KA, f"- {KA}", f"a {KA}", "a k", "u k-"]
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", aliases)

            graph = jc.compile_candidate_graph(bank, requested_moras=[KA])
            roles = [item.role for item in graph.candidates]
            keys = [item.target.key for item in graph.candidates]

            self.assertEqual(roles, [
                "mora_cv", "phrase_start_cv", "vcv_mora",
                "vc_transition", "release",
            ])
            self.assertEqual(len(set(keys)), len(keys))
            self.assertTrue(graph.coverage.all_entries_traceable)

    def test_asterisk_vowel_has_its_own_candidate_role(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", ["* \u3044", "- \u3044"])

            graph = jc.compile_candidate_graph(bank)

            self.assertEqual(
                [item.role for item in graph.candidates],
                ["vowel_blend", "phrase_start_cv"],
            )
            self.assertEqual(graph.candidates[0].target.phones, ("i",))
            self.assertNotEqual(
                graph.candidates[0].target.key,
                graph.candidates[1].target.key,
            )

    def test_profile_preserves_and_labels_nasal_allophone_aliases(self):
        aliases = [
            "\u3093", "\u3093m", "\u3093n", "\u3093ng",
            "\u30931", "\u30932", "\u30933",
            "a \u3093m", "m b", "ng k", "n t", "nn s",
        ]
        rules = {
            "coronal": {
                "mora_aliases": ["\u3093"],
                "context_aliases": ["n"],
                "following_phones": ["t", "d", "n"],
            },
            "labial": {
                "mora_aliases": ["\u3093m", "\u30932"],
                "context_aliases": ["m"],
                "following_phones": ["p", "b", "m"],
            },
            "velar": {
                "mora_aliases": ["\u3093ng", "\u30933"],
                "context_aliases": ["ng"],
                "following_phones": ["k", "g"],
            },
            "uvular": {
                "mora_aliases": ["\u3093n", "\u30931"],
                "context_aliases": ["nn"],
                "following_phones": ["s"],
                "default": True,
            },
        }
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", aliases)
            profile = jp.infer_bank_profile(
                bank,
                bank_configuration="cvvc",
                moraic_nasal_allophones=rules,
            )

            graph = jc.compile_candidate_graph(bank, profile=profile)
            by_alias = {
                item.source.alias_raw: item for item in graph.candidates
            }

            self.assertEqual(
                by_alias["\u30931"].target.moraic_nasal_allophone,
                "uvular",
            )
            self.assertEqual(
                by_alias["\u30931"].source.alternative_numbers, (),
            )
            self.assertEqual(
                by_alias["a \u3093m"].target.moraic_nasal_allophone,
                "labial",
            )
            self.assertEqual(
                by_alias["ng k"].target.moraic_nasal_allophone,
                "velar",
            )
            self.assertFalse(any(
                diagnostic.code == "moraic_nasal_allophone_unconfigured"
                for item in graph.candidates
                for diagnostic in item.diagnostics
            ))

    def test_cvvc_profile_distinguishes_n_consonant_from_nasal_transition(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", ["e n", "e \u3093"])
            profile = jp.infer_bank_profile(
                bank,
                bank_configuration="cvvc",
                moraic_nasal_allophones={
                    "coronal": {
                        "mora_aliases": ["\u3093"],
                        "context_aliases": ["n"],
                        "following_phones": ["t", "d", "n"],
                    }
                },
            )

            graph = jc.compile_candidate_graph(bank, profile=profile)
            romaji, kana = graph.candidates

            self.assertEqual(romaji.role, "vc_transition")
            self.assertEqual(romaji.target.key, "vc:e>n")
            self.assertIsNone(romaji.target.moraic_nasal_allophone)
            self.assertEqual(kana.role, "vc_transition")
            self.assertEqual(kana.family, "cvvc")
            self.assertEqual(kana.target.key, "vc:e>N")
            self.assertEqual(kana.target.phones, ("N",))
            self.assertEqual(
                kana.target.moraic_nasal_allophone, "coronal"
            )

    def test_unconfigured_nasal_allophone_is_preserved_with_diagnostic(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", ["m b"])

            graph = jc.compile_candidate_graph(bank)
            candidate = graph.candidates[0]

            self.assertEqual(candidate.role, "vc_transition")
            self.assertEqual(candidate.source.alias_raw, "m b")
            self.assertTrue(any(
                item.code == "moraic_nasal_allophone_unconfigured"
                for item in candidate.diagnostics
            ))

    def test_mixed_profile_keeps_secondary_family_candidates(self):
        aliases = [KA, f"a {KA}", "a k"]
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", aliases)
            profile = jp.infer_bank_profile(
                bank, bank_configuration="vcv"
            )

            graph = jc.compile_candidate_graph(bank, profile=profile)

            self.assertEqual(
                {item.family for item in graph.candidates},
                {"cv", "vcv", "cvvc"},
            )
            self.assertEqual(len(graph.candidates), len(aliases))

    def test_explicit_cvvc_keeps_vcv_traceable_but_not_selectable(self):
        aliases = [KA, f"a {KA}", "a k", "a i", "i n"]
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", aliases)
            profile = jp.infer_bank_profile(
                bank, bank_configuration="cvvc"
            )

            graph = jc.compile_candidate_graph(bank, profile=profile)
            by_alias = {
                item.source.alias_raw: item for item in graph.candidates
            }

            self.assertTrue(by_alias[KA].selectable)
            self.assertTrue(by_alias["a k"].selectable)
            self.assertFalse(by_alias[f"a {KA}"].selectable)
            self.assertEqual(
                by_alias[f"a {KA}"].family, "vcv"
            )
            self.assertTrue(any(
                item.code == "candidate_family_excluded_by_configuration"
                for item in by_alias[f"a {KA}"].diagnostics
            ))
            for alias in ("a i", "i n"):
                self.assertEqual(by_alias[alias].role, "vc_transition")
                self.assertEqual(by_alias[alias].family, "cvvc")
                self.assertTrue(by_alias[alias].selectable)
                self.assertTrue(any(
                    item.code == "alias_reinterpreted_for_explicit_cvvc"
                    for item in by_alias[alias].diagnostics
                ))
            self.assertEqual(graph.coverage.family_counts["vcv"], 1)
            self.assertTrue(graph.coverage.all_entries_traceable)
            self.assertEqual(
                graph.to_dict()["runtime_family_policy"],
                {
                    "mode": "strict",
                    "requested_configuration": "cvvc",
                    "effective_configuration": "cvvc",
                    "excluded_families": ["vcv"],
                    "cvvc_components": ["cv", "cvvc"],
                    "excluded_entries_preserved_for_analysis": True,
                },
            )

    def test_numbered_alternative_after_suffix_is_recovered(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_character_yaml(bank / "character.yaml")
            self.write_oto(bank / "oto.ini", ["a r1_F3"])

            graph = jc.compile_candidate_graph(bank)
            candidate = graph.candidates[0]

            self.assertEqual(candidate.role, "vc_transition")
            self.assertEqual(candidate.target.right_context, "r")
            self.assertEqual(candidate.source.alternative_numbers, (1,))
            self.assertEqual(candidate.source.alias_raw, "a r1_F3")

    def test_rest_breath_rb_never_competes_with_tapped_r_vc(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_character_yaml(bank / "character.yaml", suffix="PE3")
            self.write_oto(bank / "oto.ini", [
                "a RBPE3", "a RB1PE3", "a RPE3", "a R1PE3", "a rPE3",
            ])

            graph = jc.compile_candidate_graph(bank)
            rb, numbered_rb, rest, numbered_rest, tapped = graph.candidates

            self.assertEqual(rb.role, "breath")
            self.assertEqual(rb.family, "extra")
            self.assertFalse(rb.selectable)
            self.assertEqual(rb.source.alias_raw, "a RBPE3")
            self.assertEqual(numbered_rb.role, "breath")
            self.assertFalse(numbered_rb.selectable)
            self.assertEqual(numbered_rb.source.alternative_numbers, (1,))
            self.assertEqual(rest.role, "breath")
            self.assertFalse(rest.selectable)
            self.assertEqual(rest.source.alias_raw, "a RPE3")
            self.assertEqual(numbered_rest.role, "breath")
            self.assertFalse(numbered_rest.selectable)
            self.assertEqual(
                numbered_rest.source.alternative_numbers, (1,))
            self.assertEqual(tapped.role, "vc_transition")
            self.assertEqual(tapped.target.right_context, "r")
            self.assertTrue(tapped.selectable)

    def test_numbered_aliases_form_one_alternative_group(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_character_yaml(bank / "character.yaml")
            self.write_oto(bank / "oto.ini", [f"{KA}F3", f"{KA}1F3"])

            graph = jc.compile_candidate_graph(bank, requested_moras=[KA])
            group = next(
                item for item in graph.groups if item.role == "mora_cv"
            )

            self.assertEqual(len(group.candidate_ids), 2)
            self.assertEqual(graph.coverage.alternative_group_count, 1)

    def test_extended_small_kana_and_nasal_context_are_canonical(self):
        small_rye = "\u308a\u3047"
        aliases = [small_rye, "m h", "i ng"]
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", aliases)

            graph = jc.compile_candidate_graph(bank)

            self.assertEqual(graph.candidates[0].target.phones, ("ry", "e"))
            self.assertEqual(graph.candidates[1].role, "vc_transition")
            self.assertEqual(graph.candidates[1].target.left_context, "N")
            self.assertEqual(graph.candidates[2].role, "vcv_mora")
            self.assertEqual(graph.candidates[2].target.phones, ("N",))

    def test_vcv_wildcard_start_and_explicit_release_are_recovered(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", [f"\u30fb {KA}", "a -"])

            graph = jc.compile_candidate_graph(bank)

            self.assertEqual(graph.candidates[0].role, "phrase_start_cv")
            self.assertEqual(graph.candidates[0].family, "vcv")
            self.assertEqual(graph.candidates[1].role, "release")

    def test_named_breath_extra_is_not_left_unresolved(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", ["breath_A3-1"])

            graph = jc.compile_candidate_graph(bank)

            self.assertEqual(graph.candidates[0].role, "breath")
            self.assertEqual(graph.candidates[0].family, "extra")

    def test_unresolved_alias_is_visible_and_verbatim(self):
        alias = "unknown alias payload"
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", [alias])

            graph = jc.compile_candidate_graph(bank)
            candidate = graph.candidates[0]

            self.assertEqual(candidate.role, "unresolved")
            self.assertEqual(candidate.source.alias_raw, alias)
            self.assertFalse(candidate.selectable)
            self.assertEqual(graph.coverage.unresolved_count, 1)
            self.assertIn(
                candidate.candidate_id,
                graph.coverage.unresolved_candidate_ids,
            )

    def test_exact_alias_override_resolves_unknown_without_replacing_alias(self):
        override = jp.JapaneseAliasOverride(
            role="mora_cv", mora=KA, note="confirmed by bank author"
        )
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", ["odd-token"])
            profile = jp.infer_bank_profile(
                bank, alias_overrides={"odd-token": override}
            )

            graph = jc.compile_candidate_graph(bank, profile=profile)
            candidate = graph.candidates[0]

            self.assertEqual(candidate.role, "mora_cv")
            self.assertTrue(candidate.profile_override)
            self.assertEqual(candidate.override_key, "odd-token")
            self.assertEqual(candidate.source.alias_raw, "odd-token")
            self.assertEqual(graph.coverage.unresolved_count, 0)

    def test_disabled_family_is_retained_but_not_selectable(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", [KA, f"a {KA}"])
            profile = jp.infer_bank_profile(
                bank, enabled_families=("cv", "cvvc", "extra")
            )

            graph = jc.compile_candidate_graph(bank, profile=profile)
            vcv = next(item for item in graph.candidates
                       if item.role == "vcv_mora")

            self.assertFalse(vcv.selectable)
            self.assertEqual(graph.coverage.family_counts["vcv"], 1)

    def test_malformed_timing_is_retained_and_not_selectable(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            oto = bank / "oto.ini"
            oto.write_text(
                f"sample.wav={KA},,100,-200,50,10\n",
                encoding="utf-8",
            )
            (bank / "sample.wav").write_bytes(b"RIFF")

            graph = jc.compile_candidate_graph(bank, requested_moras=[KA])
            candidate = graph.candidates[0]

            self.assertFalse(candidate.timing.valid)
            self.assertFalse(candidate.selectable)
            self.assertEqual(graph.coverage.invalid_timing_count, 1)
            self.assertTrue(any(
                item.code == "oto_invalid_number"
                for item in graph.diagnostics
            ))

    def test_candidate_ids_and_metadata_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(
                bank / "oto.ini", [KA, f"- {KA}", f"a {KA}", "a k"]
            )

            first = jc.compile_candidate_graph(bank, requested_moras=[KA])
            second = jc.compile_candidate_graph(bank, requested_moras=[KA])

            self.assertEqual(
                [item.candidate_id for item in first.candidates],
                [item.candidate_id for item in second.candidates],
            )
            self.assertEqual(
                jc.candidate_metadata_bytes(first),
                jc.candidate_metadata_bytes(second),
            )

    def test_metadata_contains_no_absolute_source_path(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "private-bank"
            bank.mkdir()
            self.write_oto(bank / "oto.ini", [KA])

            graph = jc.compile_candidate_graph(bank, requested_moras=[KA])
            text = jc.candidate_metadata_bytes(graph).decode("utf-8")

            self.assertNotIn(str(bank), text)
            self.assertEqual(graph.candidates[0].source.oto_path, "oto.ini")
            self.assertEqual(
                graph.candidates[0].source.wav_path, "sample_0.wav"
            )

    def test_outside_wav_is_never_opened_and_entry_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            parent = Path(root)
            bank = parent / "bank"
            bank.mkdir()
            (parent / "outside.wav").write_bytes(b"RIFF")
            (bank / "oto.ini").write_text(
                f"../outside.wav={KA},0,100,-200,50,10\n",
                encoding="utf-8",
            )

            graph = jc.compile_candidate_graph(bank, requested_moras=[KA])
            candidate = graph.candidates[0]

            self.assertFalse(candidate.source.wav_within_bank)
            self.assertFalse(candidate.selectable)
            self.assertEqual(len(graph.candidates), 1)
            self.assertTrue(any(
                item.code == "source_wav_outside_bank"
                for item in candidate.diagnostics
            ))

    def test_writers_refuse_source_bank(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", [KA])
            profile = jp.infer_bank_profile(bank)
            graph = jc.compile_candidate_graph(bank, profile=profile)

            with self.assertRaisesRegex(ValueError, "source voicebank"):
                jp.write_profile(profile, bank / "profile.json")
            with self.assertRaisesRegex(ValueError, "source voicebank"):
                jc.write_candidate_metadata(graph, bank / "candidates.json")
            self.assertFalse((bank / "profile.json").exists())
            self.assertFalse((bank / "candidates.json").exists())

    def test_compilation_performs_no_source_writes(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root)
            self.write_oto(bank / "oto.ini", [KA, f"a {KA}", "a k"])
            before = {
                path.relative_to(bank).as_posix(): path.read_bytes()
                for path in bank.rglob("*") if path.is_file()
            }

            graph = jc.compile_candidate_graph(bank)

            after = {
                path.relative_to(bank).as_posix(): path.read_bytes()
                for path in bank.rglob("*") if path.is_file()
            }
            self.assertEqual(before, after)
            self.assertTrue(graph.coverage.all_entries_traceable)

    def test_phase2_modules_do_not_import_english_converter(self):
        self.assertNotIn("utau2festvox", sys.modules)
        self.assertNotIn("build_festival_voice", sys.modules)


if __name__ == "__main__":
    unittest.main()
