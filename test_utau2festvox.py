import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path

from arpasing_profile import load_arpasing_profile


HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


u2f = load_module("utau2festvox_test", "utau2festvox.py")
builder = load_module("build_festival_voice_test", "build_festival_voice.py")
spec = importlib.util.spec_from_file_location(
    "synth_diphone_test", HERE / "synth_diphone.py")
synth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synth)


def scheme_parentheses_are_balanced(source):
    depth = 0
    in_string = False
    escaped = False
    for line in str(source).splitlines():
        for char in line:
            if escaped:
                escaped = False
                continue
            if in_string and char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string and char == ";":
                break
            if not in_string and char == "(":
                depth += 1
            elif not in_string and char == ")":
                depth -= 1
                if depth < 0:
                    return False
    return depth == 0 and not in_string


class UtauVariantTests(unittest.TestCase):
    @staticmethod
    def write_silence(path, seconds=1.0, rate=16000):
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(b"\0\0" * int(seconds * rate))

    def test_profile_maps_japanese_alias_without_reinterpreting_arpasing(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            out = Path(root) / "db"
            bank.mkdir()
            self.write_silence(bank / "sample.wav")
            (bank / "oto.ini").write_text(
                "sample.wav=か,0,300,-700,120,20\n"
                "sample.wav=m,0,300,-700,120,20\n",
                encoding="utf-8",
            )
            u2f.convert(
                bank, out, "test", copy_wavs=False,
                phoneme_profile=load_arpasing_profile(),
            )
            metadata = json.loads(
                (out / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8")
            )
            self.assertIn("k-a", metadata["index"])
            self.assertIn("m-m", metadata["index"])
            self.assertNotIn("mm-mm", metadata["index"])
            self.assertEqual(
                metadata["profile_conversion"]["mapped_alias_count"], 1
            )

    def test_unsupported_breath_token_does_not_become_vowel_sustain(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            out = Path(root) / "db"
            bank.mkdir()
            self.write_silence(bank / "sample.wav")
            (bank / "oto.ini").write_text(
                "sample.wav=inh aw,0,300,-700,120,20\n"
                "sample.wav=aw,0,300,-700,120,20\n",
                encoding="utf-8",
            )
            u2f.convert(
                bank, out, "test", copy_wavs=False,
                phoneme_profile=load_arpasing_profile(),
            )
            metadata = json.loads(
                (out / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(len(metadata["alternatives"]["aw-aw"]), 1)
            self.assertEqual(
                metadata["alternatives"]["aw-aw"][0]["alias"], "aw"
            )

    def test_asaxi_diphthongs_are_vowel_plus_glide_in_both_frontends(self):
        expected = {
            "å": ["a", "w"], "ă": ["a", "y"],
            "ë": ["e", "y"], "ỏ": ["o", "w"],
            "ő": ["o", "y"], "ů": ["u", "w"],
        }
        festival_rules = dict(builder.ASAXI_RULES)
        for grapheme, phones in expected.items():
            with self.subTest(grapheme=grapheme):
                self.assertEqual(festival_rules[grapheme], phones)
                self.assertEqual(synth.g2p_asaxi(grapheme), phones)

    def test_runtime_asaxi_frontend_preserves_dotted_nasal_geminates(self):
        self.assertEqual(
            synth.g2p_asaxi("găxănă ono kem.ma"),
            [
                "g", "a", "y", "hh", "a", "y", "n", "a", "y",
                "o", "n", "o", "k", "e", "m", "m", "a",
            ],
        )

    def test_runtime_asaxi_routes_full_cap_terms_through_english_g2p(self):
        expected = (
            list(synth.asaxi_frontend.g2p_asaxi("to"))
            + ["jh", "ao", "n"]
            + list(synth.asaxi_frontend.g2p_asaxi("anő"))
        )

        self.assertEqual(
            synth.g2p_asaxi("to JOHN anő"),
            expected,
        )
        self.assertEqual(
            synth.g2p_asaxi("john"),
            list(synth.asaxi_frontend.g2p_asaxi("john")),
        )

    def test_numbered_takes_keep_their_recording_contexts(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            out = Path(root) / "db"
            bank.mkdir()
            self.write_silence(bank / "z_t_eh_r.wav")
            self.write_silence(bank / "s_t_eh_l.wav")
            self.write_silence(bank / "b_a_l_i.wav")
            self.write_silence(bank / "b_a_l_t.wav")
            (bank / "oto.ini").write_text(
                "z_t_eh_r.wav=z tF#3,0,0,-180,50,0\n"
                "z_t_eh_r.wav=t ehF#3,100,0,-180,50,0\n"
                "z_t_eh_r.wav=eh rF#3,200,0,-180,50,0\n"
                "s_t_eh_l.wav=s tF#3,0,0,-180,50,0\n"
                "s_t_eh_l.wav=t eh1F#3,100,0,-180,50,0\n"
                "s_t_eh_l.wav=eh lF#3,200,0,-180,50,0\n"
                "b_a_l_i.wav=b aF#3,0,0,-180,50,0\n"
                "b_a_l_i.wav=a lF#3,100,0,-180,50,0\n"
                "b_a_l_i.wav=l iF#3,200,0,-180,50,0\n"
                "b_a_l_t.wav=b a1F#3,0,0,-180,50,0\n"
                "b_a_l_t.wav=a l1F#3,100,0,-180,50,0\n"
                "b_a_l_t.wav=l tF#3,200,0,-180,50,0\n",
                encoding="utf-8")

            u2f.convert(bank, out, "test", copy_wavs=True)
            meta = json.loads((out / "dic" / "diphone_index.json")
                              .read_text(encoding="utf-8"))
            choices = meta["alternatives"]["t-eh"]

            self.assertEqual(len(choices), 2)
            self.assertTrue(all({"start", "mid", "end"} <= set(choice)
                                for choice in choices))
            self.assertEqual({(c["left_context"], c["right_context"])
                              for c in choices}, {("z", "r"), ("s", "l")})
            self.assertTrue(all(c["left_context_kind"] == "atomic"
                                for c in choices))
            self.assertTrue(all(c["right_context_kind"] == "atomic"
                                for c in choices))
            self.assertEqual(meta["context_model"], "oto_directional_v1")
            self.assertTrue(all(c["tail_clamped"] for c in choices))
            self.assertTrue(all(abs(c["end"] - .200) < 1e-9
                                for c in choices))
            self.assertTrue(all(abs(c["raw_end"] - .28) < 1e-9
                                for c in choices))
            self.assertTrue(all(
                c["right_center_method"] == "next_oto_overlap_end"
                for c in choices
            ))
            self.assertEqual(
                meta["diphone_geometry_model"], "oto_overlap_centers_v3"
            )
            self.assertIn("t-eh", meta["index"])
            self.assertIn("t__u1-eh", meta["index"])
            db = synth.DiphoneDB(out)
            self.assertEqual(db.choose("t-eh", "uw", "l"), "t-eh")
            self.assertEqual(db.choose("t-eh", "s", "l"), "t__u1-eh")
            self.assertEqual({c["l_class"] for c in
                              meta["alternatives"]["a-l"]},
                             {"light", "dark"})
            self.assertEqual(db.choose("a-l", "b", "t"), "a__u1-l")

            voice_out = Path(root) / "festival_voice"
            runtime_index = builder.write_runtime_metadata(
                meta, meta["index"], meta["alternatives"], voice_out)
            installed = json.loads(runtime_index.read_text(encoding="utf-8"))
            self.assertIn("t-eh", installed["index"])
            self.assertEqual(len(installed["alternatives"]["t-eh"]), 2)
            self.assertEqual(installed["context_model"],
                             "oto_directional_v1")
            self.assertEqual(
                installed["alternatives"]["t-eh"][0]["left_context_kind"],
                "atomic")

            phones = builder.phone_inventory(meta["index"])
            self.assertIn("t", phones)
            self.assertNotIn("t__u1", phones)
            scheme = builder.gen_scheme(
                "test", out, meta["index"], phones, 185.0,
                meta["alternatives"]).read_text(encoding="utf-8")
            self.assertIn("test_select_unit_variants", scheme)
            self.assertIn("us_diphone_left", scheme)
            self.assertIn("festvox_gui_unit_variant_overrides", scheme)
            self.assertNotIn("(let*", scheme)
            self.assertFalse(any(
                "(=" in line.split(";", 1)[0]
                for line in scheme.splitlines()))
            self.assertTrue(scheme_parentheses_are_balanced(scheme))
            self.assertIn("(t -8)", scheme)
            self.assertIn("test_unsafe_phrase_edge_shortcut", scheme)
            self.assertIn("test_variant_score (car choices)", scheme)
            self.assertIn("test_phone_context_classes", scheme)
            self.assertIn('("eh" "vowel" "vowel")', scheme)
            self.assertIn("expected_class", scheme)
            self.assertIn("test_phone_left_edge_class", scheme)
            self.assertIn("test_phone_right_edge_class", scheme)
            self.assertIn("test_sibilant_context_tier", scheme)
            self.assertIn("test_automatic_variant", scheme)
            self.assertIn("group/test_diphone.group", scheme)
            self.assertIn("test_grouped_db_params", scheme)
            self.assertIn("(probe_file test_group_file)", scheme)
            self.assertIn("festvox_gui_force_separate_database", scheme)

            rendered = synth.render(db, ["s", "t", "eh", "l"])
            self.assertIn("t__u1", rendered["selected_units"].values())

    def test_missing_internal_transition_uses_adjacent_oto_edge_context(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            out = Path(root) / "db"
            bank.mkdir()
            self.write_silence(bank / "recording.wav", seconds=2.0)
            (bank / "oto.ini").write_text(
                "recording.wav=ae s2E3,900,220,-260,120,35\n"
                "recording.wav=t k2E3,1120,100,-120,70,25\n"
                "recording.wav=k ae1E3,1270,180,-260,70,20\n",
                encoding="utf-8",
            )

            u2f.convert(bank, out, "test", copy_wavs=False)
            metadata = json.loads(
                (out / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8")
            )
            choice = metadata["alternatives"]["t-k"][0]

            self.assertEqual(choice["left_context"], "s")
            self.assertEqual(choice["right_context"], "ae")
            self.assertEqual(choice["left_context_source"],
                             "adjacent_oto_edge")
            self.assertEqual(choice["right_context_source"],
                             "adjacent_transition")

    def test_oto_overlap_anchors_form_stable_diphone_centers(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            out = Path(root) / "db"
            full_out = Path(root) / "full_db"
            bank.mkdir()
            self.write_silence(bank / "sample.wav")
            (bank / "oto.ini").write_text(
                "sample.wav=a t,100,260,-500,100,30\n"
                "sample.wav=t eh,250,300,-500,100,40\n",
                encoding="utf-8",
            )

            u2f.convert(bank, out, "test", copy_wavs=True)
            metadata = json.loads(
                (out / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8")
            )
            first = metadata["alternatives"]["a-t"][0]
            second = metadata["alternatives"]["t-eh"][0]

            self.assertAlmostEqual(first["raw_start"], 0.100)
            self.assertAlmostEqual(first["start"], 0.140)
            self.assertAlmostEqual(first["mid"], 0.200)
            self.assertAlmostEqual(first["end"], 0.260)
            self.assertAlmostEqual(second["start"], 0.290)
            self.assertAlmostEqual(second["mid"], 0.350)
            # The adaptive primary unit exposes at most 60 ms on either
            # side of the boundary, while retaining the complete OTO span.
            self.assertAlmostEqual(second["end"], 0.410)
            self.assertEqual(
                second["source_window"]["full"],
                {"start": 0.29, "phone_boundary": 0.35, "end": 0.55},
            )
            self.assertAlmostEqual(
                first["source_window"]["full"]["end"],
                second["source_window"]["full"]["start"],
            )
            self.assertTrue(second["window_right_name"].endswith("__wr"))
            self.assertIn(
                second["window_right_name"] + "-eh", metadata["index"])
            self.assertEqual(
                first["left_center_method"], "oto_overlap_end"
            )
            self.assertEqual(
                first["right_center_method"], "next_oto_overlap_end"
            )
            self.assertEqual(second["right_center_method"], "oto_fixed_end")
            self.assertEqual(first["oto_timing_ms"]["overlap"], 30.0)
            self.assertEqual(second["oto_timing_ms"]["overlap"], 40.0)

            # Full mode is the explicit, reversible legacy geometry.
            u2f.convert(
                bank, full_out, "test", copy_wavs=False,
                source_window_mode="full",
            )
            full_metadata = json.loads(
                (full_out / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8")
            )
            full_second = full_metadata["alternatives"]["t-eh"][0]
            self.assertAlmostEqual(full_second["end"], 0.550)
            self.assertEqual(
                full_second["source_slice"]
                if "source_slice" in full_second else {
                    "start": full_second["start"],
                    "phone_boundary": full_second["mid"],
                    "end": full_second["end"],
                },
                full_second["source_window"]["full"],
            )
            self.assertEqual(
                full_second["window_right_name"], full_second["left_name"])

    def test_zero_overlap_defaults_to_raw_offset_and_guard_is_opt_in(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            guarded = Path(root) / "guarded"
            legacy = Path(root) / "legacy"
            bank.mkdir()
            self.write_silence(bank / "sample.wav")
            (bank / "oto.ini").write_text(
                "sample.wav=a t,100,260,-500,100,0\n"
                "sample.wav=t eh,250,300,-500,100,0\n",
                encoding="utf-8",
            )

            u2f.convert(
                bank, guarded, "test", copy_wavs=False,
                zero_overlap_guard_ms=12,
            )
            u2f.convert(
                bank, legacy, "test", copy_wavs=False,
                zero_overlap_guard_ms=0,
            )
            guarded_meta = json.loads(
                (guarded / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8")
            )
            legacy_meta = json.loads(
                (legacy / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8")
            )
            first = guarded_meta["alternatives"]["a-t"][0]
            second = guarded_meta["alternatives"]["t-eh"][0]
            legacy_second = legacy_meta["alternatives"]["t-eh"][0]

            self.assertAlmostEqual(first["start"], 0.140)
            self.assertAlmostEqual(
                first["source_window"]["full"]["start"], 0.112
            )
            self.assertAlmostEqual(first["end"], 0.260)
            self.assertAlmostEqual(second["start"], 0.290)
            self.assertAlmostEqual(
                first["source_window"]["full"]["end"],
                second["source_window"]["full"]["start"],
            )
            self.assertAlmostEqual(
                second["source_window"]["full"]["start"], 0.262
            )
            self.assertEqual(
                second["left_center_method"],
                "inferred_zero_overlap_guard",
            )
            self.assertEqual(second["effective_overlap_ms"], 12.0)
            self.assertEqual(second["oto_timing_ms"]["overlap"], 0.0)
            self.assertAlmostEqual(legacy_second["start"], 0.290)
            self.assertAlmostEqual(
                legacy_second["source_window"]["full"]["start"],
                0.250,
            )
            self.assertEqual(
                legacy_second["left_center_method"],
                "oto_offset_fallback",
            )


    def test_voiced_sibilant_selection_uses_only_ordered_oto_aliases(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            out = Path(root) / "db"
            bank.mkdir()
            # Deliberately misleading filenames: only the OTO sequence may
            # classify these contexts.
            for filename in ("filename_says_stop.wav", "opaque.wav",
                             "filename_says_vowel.wav"):
                self.write_silence(bank / filename)
            (bank / "oto.ini").write_text(
                "filename_says_vowel.wav=ch ayF#3,0,0,-180,50,0\n"
                "filename_says_vowel.wav=ay zF#3,100,0,-180,50,0\n"
                "filename_says_vowel.wav=z dF#3,200,0,-180,50,0\n"
                "opaque.wav=b ayF#3,0,0,-180,50,0\n"
                "opaque.wav=ay z1F#3,100,0,-180,50,0\n"
                "filename_says_stop.wav=ka ayF#3,0,0,-180,50,0\n"
                "filename_says_stop.wav=ay z2F#3,100,0,-180,50,0\n"
                "filename_says_stop.wav=z oyF#3,200,0,-180,50,0\n",
                encoding="utf-8")

            u2f.convert(bank, out, "test", copy_wavs=True)
            meta = json.loads((out / "dic" / "diphone_index.json")
                              .read_text(encoding="utf-8"))
            choices = meta["alternatives"]["ay-z"]
            by_right = {choice["right_context"]: choice
                        for choice in choices}

            self.assertEqual(set(by_right), {"d", "*", "oy"})
            self.assertEqual(by_right["*"]["right_context_kind"],
                             "wildcard_unknown")
            self.assertEqual(by_right["*"]["right_class"], "wildcard")
            self.assertEqual(by_right["oy"]["left_context"], "ka")
            self.assertEqual(by_right["oy"]["left_context_edge"], "a")
            self.assertEqual(by_right["oy"]["left_class"], "vowel")
            self.assertEqual(by_right["oy"]["left_context_kind"],
                             "compound_cv")

            db = synth.DiphoneDB(out)
            self.assertEqual(db.choose("ay-z", "s", "er"),
                             by_right["oy"]["index_name"])
            self.assertEqual(
                db.choose("ay-z", "s", "er",
                          override=by_right["d"]["left_name"]),
                by_right["d"]["index_name"])

    def test_character_yaml_imports_nested_subbanks_without_mixing_colors(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            default_out = Path(root) / "default_db"
            all_out = Path(root) / "all_db"
            voice_out = Path(root) / "headvoice"
            neutral = bank / "P3_E3"
            headvoice = bank / "PH3_E4"
            neutral.mkdir(parents=True)
            headvoice.mkdir(parents=True)
            (bank / "oto.ini").write_text("", encoding="utf-8")
            (bank / "character.yaml").write_text(
                "text_file_encoding: shift_jis\n"
                "subbanks:\n"
                "- color: \"\"\n"
                "  prefix: \"\"\n"
                "  suffix: PE3\n"
                "  tone_ranges:\n"
                "  - E3-F3\n"
                "- color: Headvoice\n"
                "  prefix: \"\"\n"
                "  suffix: PE4H\n"
                "  tone_ranges: [E4-F4]\n",
                encoding="utf-8")
            for folder, suffix in ((neutral, "PE3"),
                                   (headvoice, "PE4H")):
                self.write_silence(folder / "shared.wav")
                oto_text = (
                    f"shared.wav=a ay{suffix},0,0,-180,50,0\n"
                    f"shared.wav=ay z{suffix},200,0,-180,50,0\n")
                if folder == headvoice:
                    oto_text += (
                        "shared.wav=z rrE4,400,0,-180,50,0\n"
                        "shared.wav=rr sPE4,600,0,-180,50,0\n")
                (folder / "oto.ini").write_text(
                    oto_text, encoding="shift_jis")

            u2f.convert(bank, default_out, "test", copy_wavs=True)
            default_meta = json.loads(
                (default_out / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8"))
            default_choice = default_meta["alternatives"]["a-ay"][0]

            self.assertEqual(default_meta["alias_metadata"]
                             ["selected_voice_color"], "")
            self.assertEqual(default_meta["alias_metadata"]
                             ["default_subbank"]["suffix"], "PE3")
            self.assertEqual(default_choice["alias"], "a ay")
            self.assertEqual(default_choice["raw_alias"], "a ayPE3")
            self.assertEqual(default_choice["source_subbank"], "P3_E3")
            self.assertEqual(default_choice["source_color"], "")
            self.assertEqual(default_choice["source_suffix"], "PE3")
            self.assertEqual(default_choice["oto_file"], "P3_E3/oto.ini")
            self.assertEqual(len(default_meta["alternatives"]["a-ay"]), 1)
            self.assertTrue((default_out / "wav" /
                             "P3_E3_shared.wav").is_file())
            report = (default_out / "conversion_report.txt").read_text(
                encoding="utf-8")
            self.assertIn("4 other-color entries skipped", report)
            self.assertIn("dynamic F0 routing is not implemented", report)

            u2f.convert(bank, all_out, "test", copy_wavs=True,
                        voice_color="all")
            all_meta = json.loads(
                (all_out / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8"))
            all_choices = all_meta["alternatives"]["a-ay"]
            self.assertEqual(len(all_choices), 2)
            self.assertEqual({choice["source_color"] for choice in all_choices},
                             {"", "Headvoice"})
            used_wavs = {row[0] for row in all_meta["index"].values()}
            self.assertIn("P3_E3_shared.wav", used_wavs)
            self.assertIn("PH3_E4_shared.wav", used_wavs)

            db = builder.run_utau_conversion(
                bank, voice_out, "test",
                character_yaml=bank / "character.yaml",
                voice_color="Headvoice",
                oto_files=(headvoice,))
            head_meta = json.loads(
                (db / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8"))
            self.assertEqual(head_meta["alias_metadata"]
                             ["selected_voice_color"], "Headvoice")
            self.assertEqual(head_meta["alternatives"]["a-ay"][0]
                             ["source_suffix"], "PE4H")
            inferred = head_meta["alternatives"]["z-rr"][0]
            self.assertEqual(inferred["source_color"], "Headvoice")
            self.assertTrue(inferred["source_affix_inferred_from_oto"])
            self.assertNotIn("rr-s", head_meta["alternatives"])
            head_report = (db / "conversion_report.txt").read_text(
                encoding="utf-8")
            self.assertIn("1 unresolved affix entries skipped", head_report)

    def test_prefix_map_and_manual_suffix_handle_affix_orders(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            mapped_out = Path(root) / "mapped"
            manual_out = Path(root) / "manual"
            bank.mkdir()
            self.write_silence(bank / "shared.wav")
            (bank / "prefix.map").write_text(
                "E3\tStrong_\tPE3\n", encoding="utf-8")
            (bank / "oto.ini").write_text(
                "shared.wav=Strong_a ayPE3,0,0,-180,50,0\n",
                encoding="utf-8")

            u2f.convert(bank, mapped_out, "test", copy_wavs=False)
            mapped = json.loads(
                (mapped_out / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8"))
            choice = mapped["alternatives"]["a-ay"][0]
            self.assertEqual(choice["source_prefix"], "Strong_")
            self.assertEqual(choice["source_suffix"], "PE3")
            self.assertEqual(choice["source_affix_source"], "prefix.map")

            (bank / "prefix.map").unlink()
            (bank / "oto.ini").write_text(
                "shared.wav=a ayPE3,0,0,-180,50,0\n"
                "shared.wav=a ay1E3P,200,0,-180,50,0\n",
                encoding="utf-8")
            u2f.convert(bank, manual_out, "test", copy_wavs=False,
                        alias_suffixes=["P"])
            manual = json.loads(
                (manual_out / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8"))
            choices = manual["alternatives"]["a-ay"]
            self.assertEqual(len(choices), 2)
            self.assertEqual({choice["alias"] for choice in choices},
                             {"a ay", "a ay1"})
            self.assertTrue(all(choice["source_suffix"] == "P"
                                for choice in choices))
            self.assertEqual(manual["alias_metadata"]["manual_suffixes"],
                             ["P"])

    def test_empty_conversion_explains_metadata_and_manual_affixes(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            out = Path(root) / "db"
            bank.mkdir()
            self.write_silence(bank / "bad.wav")
            (bank / "oto.ini").write_text(
                "bad.wav=a b cPE3,0,0,-180,50,0\n",
                encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                u2f.convert(bank, out, "test", copy_wavs=False)
            message = str(raised.exception)
            self.assertIn("--character-yaml PATH", message)
            self.assertIn("--prefix-map PATH", message)
            self.assertIn("--alias-suffix P", message)
            self.assertIn("ayPE3 and ayE3P", message)

    def test_coda_triphone_alias_is_ignored_but_explicit_pause_is_kept(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            out = Path(root) / "db"
            bank.mkdir()
            self.write_silence(bank / "coda.wav")
            (bank / "oto.ini").write_text(
                "coda.wav=eh l-,0,0,-180,50,0\n"
                "coda.wav=l -,200,0,-180,50,0\n",
                encoding="utf-8")

            u2f.convert(bank, out, "test", copy_wavs=False)
            meta = json.loads((out / "dic" / "diphone_index.json")
                              .read_text(encoding="utf-8"))
            report = (out / "conversion_report.txt").read_text(
                encoding="utf-8")

            self.assertNotIn("eh-l_", meta["index"])
            self.assertIn("l-pau", meta["index"])
            self.assertIn("coda triphones   : 1 ignored", report)

    def test_generated_context_metadata_is_language_neutral(self):
        self.assertEqual(u2f.phone_context_class("u"), "vowel")
        self.assertEqual(u2f.phone_context_class("ae"), "vowel")
        self.assertEqual(u2f.phone_context_class("dh"), "fricative_voiced")
        self.assertEqual(u2f.phone_context_class("*"), "wildcard")
        self.assertEqual(u2f.context_edge_info("zha", "left")["class"],
                         "fricative_voiced")
        self.assertEqual(u2f.context_edge_info("zha", "right")["class"],
                         "vowel")
        self.assertEqual(u2f.context_edge_info("j", "left")["kind"],
                         "unclassified")
        self.assertEqual(builder.phone_context_class("*"), "wildcard")
        self.assertEqual(builder.context_edge_info("ka", "left")["class"],
                         "stop_voiceless")
        self.assertEqual(builder.context_edge_info("ka", "right")["class"],
                         "vowel")

    def test_builder_adds_structural_stop_hold(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            out = Path(root) / "db"
            bank.mkdir()
            self.write_silence(bank / "sample.wav", seconds=1.5)
            (bank / "oto.ini").write_text(
                "sample.wav=a t,0,300,-500,120,20\n"
                "sample.wav=t a,350,300,-500,120,20\n",
                encoding="utf-8",
            )

            u2f.convert(bank, out, "fixture", copy_wavs=True)
            metadata = json.loads(
                (out / "dic" / "diphone_index.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertIn("t-t", metadata["index"])
        self.assertEqual(
            metadata["alternatives"]["t-t"][0]["role"],
            "structural_consonant_hold",
        )
        self.assertEqual(
            metadata["special_phone_realizations"]["phones"]["cl"]["mode"],
            "anticipatory_consonant",
        )

    def test_builder_adds_structural_voiced_stop_hold(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            out = Path(root) / "db"
            bank.mkdir()
            self.write_silence(bank / "sample.wav", seconds=1.5)
            (bank / "oto.ini").write_text(
                "sample.wav=a b,0,300,-500,120,20\n"
                "sample.wav=b a,350,300,-500,120,20\n",
                encoding="utf-8",
            )

            u2f.convert(bank, out, "fixture", copy_wavs=True)
            metadata = json.loads(
                (out / "dic" / "diphone_index.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertIn("b-b", metadata["index"])
        self.assertEqual(
            metadata["alternatives"]["b-b"][0]["role"],
            "structural_consonant_hold",
        )

    def test_explicit_cl_oto_does_not_change_default_structural_policy(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            out = Path(root) / "db"
            bank.mkdir()
            self.write_silence(bank / "sample.wav", seconds=1.5)
            (bank / "oto.ini").write_text(
                "sample.wav=a cl,0,250,-500,110,20\n"
                "sample.wav=cl a,300,250,-500,110,20\n"
                "sample.wav=a t,600,250,-500,110,20\n"
                "sample.wav=t a,900,250,-500,110,20\n",
                encoding="utf-8",
            )

            u2f.convert(bank, out, "fixture", copy_wavs=True)
            metadata = json.loads(
                (out / "dic" / "diphone_index.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertIn("a-cl", metadata["index"])
        self.assertIn("cl-a", metadata["index"])
        self.assertEqual(
            metadata["special_phone_realizations"]["phones"]["cl"]["mode"],
            "anticipatory_consonant",
        )

    def test_custom_consonant_also_receives_a_structural_hold(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            out = Path(root) / "db"
            bank.mkdir()
            self.write_silence(bank / "sample.wav", seconds=1.0)
            (bank / "oto.ini").write_text(
                "sample.wav=a xq,0,250,-500,110,20\n"
                "sample.wav=xq a,300,250,-500,110,20\n",
                encoding="utf-8",
            )

            u2f.convert(bank, out, "fixture", copy_wavs=True)
            metadata = json.loads(
                (out / "dic" / "diphone_index.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(u2f.phone_context_class("xq"), "other")
        self.assertIn("xq-xq", metadata["index"])
        self.assertEqual(
            metadata["alternatives"]["xq-xq"][0]["role"],
            "structural_consonant_hold",
        )


class StandaloneRendererCacheTests(unittest.TestCase):
    @staticmethod
    def _write_wave(path, samples=80, rate=8000):
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(b"\0\0" * samples)

    def _database(self, root):
        root = Path(root)
        (root / "dic").mkdir()
        (root / "wav").mkdir()
        index = {}
        for number, name in enumerate(("one.wav", "two.wav", "three.wav")):
            self._write_wave(root / "wav" / name, samples=80 + number)
            index["p%d-p%d" % (number, number)] = [name, 0, .005, .01]
        (root / "dic" / "diphone_index.json").write_text(
            json.dumps({"index": index}), encoding="utf-8")
        return root

    def test_tests_and_gui_share_the_bundled_renderer(self):
        self.assertEqual(Path(synth.__file__).resolve(),
                         (HERE / "synth_diphone.py").resolve())

    def test_decoded_wave_cache_is_lru_bounded_and_clearable(self):
        with tempfile.TemporaryDirectory() as root:
            database = synth.DiphoneDB(
                self._database(root), cache_max_files=2,
                cache_max_bytes=1024 * 1024)
            database._load("one.wav")
            database._load("two.wav")
            database._load("one.wav")
            database._load("three.wav")

            self.assertEqual(list(database._cache),
                             ["one.wav", "three.wav"])
            self.assertEqual(database.cache_info()["files"], 2)
            self.assertLessEqual(database.cache_info()["bytes"],
                                 database.cache_info()["max_bytes"])
            database.clear_cache()
            self.assertEqual(database.cache_info()["files"], 0)
            self.assertEqual(database.cache_info()["bytes"], 0)

    def test_byte_limit_refuses_oversized_sources(self):
        with tempfile.TemporaryDirectory() as root:
            database = synth.DiphoneDB(
                self._database(root), cache_max_files=3,
                cache_max_bytes=32)
            database._load("one.wav")
            database._load("two.wav")

            self.assertEqual(database.cache_info()["files"], 0)
            self.assertEqual(list(database._cache), [])
            self.assertLessEqual(database.cache_info()["bytes"], 32)


if __name__ == "__main__":
    unittest.main()
