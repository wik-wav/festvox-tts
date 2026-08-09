import math
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave

import japanese_assembly as ja
from japanese_assembly_listening import ASSEMBLY_LISTENING_FIXTURES
import japanese_candidates as jc
import japanese_festival as jf
from japanese_frontend import analyze_japanese
from japanese_profiles import infer_bank_profile
import japanese_synthesis as js


def _tone_wav(path, frequency, duration=0.8, sample_rate=16000):
    frames = bytearray()
    for index in range(int(duration * sample_rate)):
        sample = int(
            8000.0 * math.sin(2.0 * math.pi * frequency * index / sample_rate)
        )
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))


def _write_bank(root, bank_type):
    moras = (
        "\u3042", "\u3044", "\u3046", "\u3048", "\u304a",
        "\u304b", "\u304d", "\u3055", "\u305f", "\u3081", "\u3068",
        "\u304d\u3083", "\u304d\u3087", "\u3093",
    )
    aliases = list(moras) + [f"- {mora}" for mora in moras]
    aliases.extend(("a -", "i -", "u -", "e -", "o -", "m -"))
    if bank_type == "vcv":
        aliases.extend((
            "a \u304b", "a \u304d", "a \u3055", "a \u305f",
            "a \u3081", "e \u3068", "o \u3046", "a \u3093",
        ))
    elif bank_type == "cvvc":
        # The VCV row is deliberate secondary evidence: explicit CVVC must
        # still select its VC + CV assembly when both are available.
        aliases.extend((
            "a \u304b", "o \u3046", "a \u3093",
            "a k", "a s", "a t", "a m", "e t", "o a",
            "a k-",
        ))
    lines = []
    for index, alias in enumerate(aliases):
        wav_name = f"unit_{index}.wav"
        _tone_wav(root / wav_name, 160.0 + index * 5.0)
        lines.append(f"{wav_name}={alias},40,260,-620,120,20")
    (root / "oto.ini").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _compile(root, bank_type):
    bank = root / "bank"
    bank.mkdir()
    _write_bank(bank, bank_type)
    profile = infer_bank_profile(bank, bank_configuration=bank_type)
    graph = jc.compile_candidate_graph(bank, profile=profile)
    output = root / "voice"
    jf.compile_festival_voice(
        graph, output, voice_name=f"assembly_{bank_type}", pitchmark=False
    )
    return graph, output, jf.load_japanese_runtime_metadata(output)


def _tree_bytes(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class JapaneseAssemblyTests(unittest.TestCase):
    def test_automatic_selector_excludes_medial_phrase_start_and_release(self):
        phrase_start = {
            "candidate_id": "start", "role": "phrase_start_cv",
            "edge_offset": 0, "selection_cost": -10,
            "recorded_left_context": "pau",
            "recorded_right_context": "*",
        }
        plain_cv = {
            "candidate_id": "cv", "role": "mora_cv",
            "edge_offset": 0, "selection_cost": 0,
            "recorded_left_context": "*",
            "recorded_right_context": "*",
        }
        release = {
            "candidate_id": "release", "role": "release",
            "edge_offset": -1, "selection_cost": 0,
            "recorded_left_context": "a",
            "recorded_right_context": "*",
        }
        vc = {
            "candidate_id": "vc", "role": "vc_transition",
            "edge_offset": -1, "selection_cost": 0,
            "recorded_left_context": "a",
            "recorded_right_context": "*",
        }

        self.assertEqual(
            ja.select_automatic_choice(
                [phrase_start, plain_cv], "a", "i", "k", "a"
            )["candidate_id"],
            "cv",
        )
        self.assertEqual(
            ja.select_automatic_choice(
                [release, vc], "pau", "a", "a", "k"
            )["candidate_id"],
            "vc",
        )

    def test_exact_vowel_vcv_outranks_asterisk_blend_fallback(self):
        exact = {
            "candidate_id": "exact", "role": "vcv_mora",
            "edge_offset": -1, "selection_cost": 20,
            "recorded_left_context": "a",
            "recorded_right_context": "*",
        }
        blend = {
            "candidate_id": "blend", "role": "vowel_blend",
            "edge_offset": 0, "selection_cost": 0,
            "recorded_left_context": "*",
            "recorded_right_context": "*",
        }

        selected = ja.select_automatic_choice(
            [blend, exact], "pau", "k", "a", "i"
        )

        self.assertEqual(selected["candidate_id"], "exact")

    def test_profile_defined_nasal_allophone_routes_both_edges(self):
        routing = {
            "following_phones": {
                "b": "labial", "k": "velar", "t": "coronal",
                "n": "coronal", "s": "uvular",
            },
            "default": "uvular",
        }
        choices = [
            {
                "candidate_id": name, "role": "vc_transition",
                "edge_offset": -1, "selection_cost": 0,
                "recorded_left_context": "N",
                "recorded_right_context": "*",
                "moraic_nasal_allophone": name,
            }
            for name in ("coronal", "labial", "velar", "uvular")
        ]
        for following, expected in (
            ("b", "labial"), ("k", "velar"), ("t", "coronal"),
            ("n", "coronal"), ("s", "uvular"), ("pau", "uvular"),
        ):
            with self.subTest(edge="N-C", following=following):
                selected = ja.select_automatic_choice(
                    choices, "a", "a", "N", following, routing
                )
                self.assertEqual(selected["candidate_id"], expected)
            with self.subTest(edge="V-N", following=following):
                selected = ja.select_automatic_choice(
                    choices, "pau", following, "a", "N", routing
                )
                self.assertEqual(selected["candidate_id"], expected)

    def test_compiled_bank_uses_one_declared_nasal_allophone_per_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "bank"
            bank.mkdir()
            aliases = (
                "\u3042", "- \u3042", "\u3070", "\u304b", "\u305f", "\u3055",
                "\u3093", "\u3093m", "\u3093ng", "\u3093n",
                "a \u3093", "a \u3093m", "a \u3093ng", "a \u3093n",
                "n t", "m b", "ng k", "nn s",
                "n -", "m -", "ng -", "nn -",
            )
            lines = []
            for index, alias in enumerate(aliases):
                wav_name = f"nasal_{index}.wav"
                _tone_wav(bank / wav_name, 145.0 + index * 3.0)
                lines.append(
                    f"{wav_name}={alias},40,260,-620,120,20"
                )
            (bank / "oto.ini").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            profile = infer_bank_profile(
                bank,
                bank_configuration="cvvc",
                moraic_nasal_allophones={
                    "coronal": {
                        "mora_aliases": ["\u3093"],
                        "context_aliases": ["n"],
                        "following_phones": ["t", "d", "n", "ny"],
                    },
                    "labial": {
                        "mora_aliases": ["\u3093m"],
                        "context_aliases": ["m"],
                        "following_phones": ["p", "b", "m"],
                    },
                    "velar": {
                        "mora_aliases": ["\u3093ng"],
                        "context_aliases": ["ng"],
                        "following_phones": ["k", "g"],
                    },
                    "uvular": {
                        "mora_aliases": ["\u3093n"],
                        "context_aliases": ["nn"],
                        "following_phones": ["s"],
                        "default": True,
                    },
                },
            )
            graph = jc.compile_candidate_graph(bank, profile=profile)
            output = root / "voice"
            jf.compile_festival_voice(
                graph, output, voice_name="nasal_routes", pitchmark=False
            )
            runtime = jf.load_japanese_runtime_metadata(output)
            cases = (
                ("\u3042\u3093\u3070", "labial", "a \u3093m", "m b"),
                ("\u3042\u3093\u304b", "velar", "a \u3093ng", "ng k"),
                ("\u3042\u3093\u305f", "coronal", "a \u3093", "n t"),
                ("\u3042\u3093\u3055", "uvular", "a \u3093n", "nn s"),
                ("\u3042\u3093", "uvular", "a \u3093n", "nn -"),
            )
            for text, expected, incoming_alias, outgoing_alias in cases:
                with self.subTest(text=text):
                    plan = js.create_synthesis_plan(
                        analyze_japanese(text, mode="kana"),
                        runtime_metadata=runtime,
                    )
                    assembly = ja.create_source_contribution_plan(
                        plan, runtime
                    )
                    incoming = next(
                        item for item in assembly.contributions
                        if item.right_phone == "N"
                    )
                    outgoing = next(
                        item for item in assembly.contributions
                        if item.left_phone == "N"
                    )
                    self.assertEqual(
                        incoming.moraic_nasal_allophone, expected
                    )
                    self.assertEqual(
                        outgoing.moraic_nasal_allophone, expected
                    )
                    self.assertIsNone(incoming.fallback_reason)
                    self.assertEqual(incoming.source_alias, incoming_alias)
                    self.assertEqual(outgoing.source_alias, outgoing_alias)
    def test_contextual_pairs_share_one_source_phone_center(self):
        with tempfile.TemporaryDirectory() as temp:
            graph, _, _ = _compile(Path(temp), "cvvc")
            candidates = {
                item.source.alias_raw: item for item in graph.candidates
            }
            for alias in ("- \u304b", "a \u304b", "a k-"):
                proposals = jf.candidate_edge_proposals(
                    candidates[alias], 0.8
                )
                self.assertEqual(len(proposals), 2)
                self.assertAlmostEqual(
                    proposals[0].end_ms, proposals[1].start_ms
                )
                self.assertAlmostEqual(
                    proposals[0].shared_anchor_ms,
                    proposals[1].shared_anchor_ms,
                )
                self.assertLessEqual(
                    proposals[0].end_ms, proposals[1].start_ms
                )

            cv = jf.candidate_edge_proposals(candidates["\u304b"], 0.8)[0]
            self.assertEqual(cv.method, "oto_centered_cv")
            self.assertGreater(cv.start_ms, 40.0)
            self.assertLess(cv.start_ms, cv.midpoint_ms)

    def test_cvvc_plan_uses_real_vc_and_pair_coherent_initial_cv(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, runtime = _compile(Path(temp), "cvvc")
            plan = js.create_synthesis_plan(
                analyze_japanese("\u304b\u304b", mode="kana"),
                runtime_metadata=runtime,
            )
            assembly = ja.create_source_contribution_plan(plan, runtime)
            by_edge = {item.diphone: [] for item in assembly.contributions}
            for item in assembly.contributions:
                by_edge.setdefault(item.diphone, []).append(item)

            initial_left = next(
                item for item in assembly.contributions
                if item.diphone == "pau-k"
            )
            initial_right = next(
                item for item in assembly.contributions
                if item.edge_index == initial_left.edge_index + 1
            )
            self.assertEqual(initial_left.source_alias, "- \u304b")
            self.assertEqual(
                initial_left.candidate_id, initial_right.candidate_id
            )
            self.assertAlmostEqual(
                initial_left.source_end, initial_right.source_start
            )

            vc = next(
                item for item in assembly.contributions
                if item.diphone == "a-k"
            )
            following_cv = assembly.contributions[vc.edge_index + 1]
            self.assertEqual(vc.role, "vc_transition")
            self.assertEqual(following_cv.role, "mora_cv")
            self.assertEqual(vc.fallback_reason, None)
            self.assertEqual(following_cv.fallback_reason, None)
            self.assertTrue(vc.oto_timing_ms)
            self.assertTrue(assembly.all_spoken_edges_sourced)
            self.assertFalse(any(
                item.severity == "error" for item in assembly.diagnostics
            ))
            self.assertEqual(
                assembly.to_json_bytes(),
                ja.create_source_contribution_plan(plan, runtime).to_json_bytes(),
            )

    def test_vcv_plan_uses_one_vcv_alias_for_both_transition_halves(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, runtime = _compile(Path(temp), "vcv")
            plan = js.create_synthesis_plan(
                analyze_japanese("\u3042\u304b", mode="kana"),
                runtime_metadata=runtime,
            )
            assembly = ja.create_source_contribution_plan(plan, runtime)
            left = next(
                item for item in assembly.contributions
                if item.diphone == "a-k"
            )
            right = assembly.contributions[left.edge_index + 1]
            self.assertEqual(left.role, "vcv_mora")
            self.assertEqual(right.role, "vcv_mora")
            self.assertEqual(left.candidate_id, right.candidate_id)
            self.assertAlmostEqual(left.source_end, right.source_start)
            self.assertFalse(any(
                item.code == "paired_candidate_mismatch"
                for item in assembly.diagnostics
            ))

    def test_missing_cv_transition_uses_visible_audible_bridge_not_silence(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, runtime = _compile(Path(temp), "cv")
            plan = js.create_synthesis_plan(
                analyze_japanese("\u3042\u304b", mode="kana"),
                runtime_metadata=runtime,
            )
            assembly = ja.create_source_contribution_plan(plan, runtime)
            fallback = next(
                item for item in assembly.contributions
                if item.diphone == "a-k"
            )
            self.assertEqual(fallback.source_kind, "generated_fallback")
            self.assertEqual(fallback.role, "generated_cv_bridge")
            self.assertIsNotNone(fallback.fallback_reason)
            self.assertEqual(len(fallback.source_components), 2)
            self.assertTrue(assembly.all_spoken_edges_sourced)
            self.assertEqual(assembly.hidden_silence_count, 0)
            self.assertTrue(any(
                item.code == "generated_transition_fallback"
                for item in assembly.diagnostics
            ))

    def test_asterisk_vowel_drives_missing_transition_blend(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "bank"
            bank.mkdir()
            aliases = ("\u3042", "\u3044", "- \u3042", "* \u3044")
            lines = []
            for index, alias in enumerate(aliases):
                wav_name = f"blend_{index}.wav"
                _tone_wav(bank / wav_name, 170.0 + index * 10.0)
                timing = (
                    "16,52,69,26,50" if alias == "* \u3044"
                    else "40,260,-620,120,20"
                )
                lines.append(f"{wav_name}={alias},{timing}")
            (bank / "oto.ini").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            profile = infer_bank_profile(
                bank, bank_configuration="cv"
            )
            graph = jc.compile_candidate_graph(bank, profile=profile)
            output = root / "voice"
            jf.compile_festival_voice(
                graph, output, voice_name="vowel_blend", pitchmark=False
            )
            runtime = jf.load_japanese_runtime_metadata(output)
            plan = js.create_synthesis_plan(
                analyze_japanese("\u3042\u3044", mode="kana"),
                runtime_metadata=runtime,
            )
            assembly = ja.create_source_contribution_plan(plan, runtime)
            transition = next(
                item for item in assembly.contributions
                if item.diphone == "a-i"
            )

            self.assertEqual(transition.role, "generated_cv_bridge")
            incoming = next(
                item for item in transition.source_components
                if item["purpose"] == "right_vowel_blend"
            )
            self.assertEqual(incoming["alias"], "* \u3044")
            self.assertEqual(incoming["preferred_crossfade_ms"], 50.0)
            self.assertNotIn("phrase_start_cv", {
                item["role"] for item in transition.source_components
            })

    def test_required_fixture_matrix_has_no_unsourced_spoken_edges(self):
        required_texts = (
            "\u3042", "\u304b", "\u3042\u304b", "\u304b\u304b",
            "\u3042\u304d", "\u304b\u3055", "\u3055\u304b",
            "\u3042\u3093", "\u3042\u3063\u305f",
            "\u304d\u3083", "\u304d\u3087\u3046",
        )
        with tempfile.TemporaryDirectory() as temp:
            for bank_type in ("cv", "vcv", "cvvc"):
                root = Path(temp) / bank_type
                root.mkdir()
                _, _, runtime = _compile(root, bank_type)
                for text in required_texts:
                    with self.subTest(bank_type=bank_type, text=text):
                        plan = js.create_synthesis_plan(
                            analyze_japanese(text, mode="kana"),
                            runtime_metadata=runtime,
                        )
                        assembly = ja.create_source_contribution_plan(
                            plan, runtime
                        )
                        self.assertTrue(assembly.all_spoken_edges_sourced)
                        self.assertEqual(assembly.hidden_silence_count, 0)
                        self.assertFalse(any(
                            item.severity == "error"
                            for item in assembly.diagnostics
                        ))
                        if text == "\u3042\u3063\u305f":
                            by_display = {
                                item.diphone: item
                                for item in assembly.contributions
                            }
                            self.assertEqual(
                                by_display["a-cl"].source_diphone,
                                "a-t",
                            )
                            self.assertEqual(
                                by_display["cl-t"].source_diphone,
                                "t-t",
                            )
                            self.assertEqual(
                                by_display["cl-t"].role,
                                "structural_consonant_hold",
                            )

    def test_listening_matrix_keeps_human_audit_categories(self):
        self.assertGreaterEqual(len(ASSEMBLY_LISTENING_FIXTURES), 18)
        self.assertTrue(
            {
                "vowels", "ordinary CV", "VCV transition",
                "CVVC transition", "moraic nasal", "geminate",
                "palatalized", "long vowels", "phrase boundaries",
                "statement", "question", "accent carrier",
                "nasal allophone: labial", "nasal allophone: velar",
                "nasal allophone: coronal",
                "nasal allophone: configured uvular",
                "exact vowel transition", "Open JTalk vowel sequence",
            } <= {
                category for _, category, _, _
                in ASSEMBLY_LISTENING_FIXTURES
            }
        )

    def test_old_overlapping_consonant_geometry_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, runtime = _compile(Path(temp), "cvvc")
            broken = copy.deepcopy(runtime)
            left = next(
                choice for choice in broken["alternatives"]["pau-k"]
                if choice["alias"] == "- \u304b"
            )
            right = next(
                choice for choice in broken["alternatives"]["k-a"]
                if choice["candidate_id"] == left["candidate_id"]
            )
            right["source_slice"]["start"] = round(
                float(left["source_slice"]["end"]) - 0.05, 6
            )
            plan = js.create_synthesis_plan(
                analyze_japanese("\u304b", mode="kana"),
                runtime_metadata=broken,
            )
            assembly = ja.create_source_contribution_plan(plan, broken)
            codes = {item.code for item in assembly.diagnostics}
            self.assertIn("shared_anchor_mismatch", codes)
            self.assertIn("duplicate_consonant_overlap", codes)

    def test_generated_bridges_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            first_root = Path(temp) / "first"
            second_root = Path(temp) / "second"
            first_root.mkdir()
            second_root.mkdir()
            _, first_output, first_runtime = _compile(first_root, "cv")
            _, second_output, second_runtime = _compile(second_root, "cv")
            self.assertEqual(
                json.dumps(first_runtime, ensure_ascii=False, sort_keys=True),
                json.dumps(second_runtime, ensure_ascii=False, sort_keys=True),
            )
            first_bridges = {
                path.name: path.read_bytes()
                for path in (first_output / "wav").glob("_jfb_*.wav")
            }
            second_bridges = {
                path.name: path.read_bytes()
                for path in (second_output / "wav").glob("_jfb_*.wav")
            }
            self.assertTrue(first_bridges)
            self.assertEqual(first_bridges, second_bridges)
            self.assertEqual(
                _tree_bytes(first_output), _tree_bytes(second_output)
            )


if __name__ == "__main__":
    unittest.main()
