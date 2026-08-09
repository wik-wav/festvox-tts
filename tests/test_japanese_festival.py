import hashlib
import importlib.util
import json
import math
import os
from dataclasses import replace
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import wave

import japanese_candidates as jc
import japanese_festival as jf
import japanese_listening_set as jls
from japanese_profiles import JapaneseSubbank, infer_bank_profile
from japanese_frontend import analyze_japanese
import japanese_synthesis as js


FESTVOX_DIR = Path(jf.__file__).resolve().parent
GUI_DIR = FESTVOX_DIR / "festvox_gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))


def _tone_wav(path, frequency, duration=0.75, sample_rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    attack = int(sample_rate * 0.02)
    release = int(sample_rate * 0.05)
    count = int(sample_rate * duration)
    for index in range(count):
        envelope = min(
            1.0,
            index / max(1, attack),
            (count - index - 1) / max(1, release),
        )
        sample = int(
            9000.0 * max(0.0, envelope)
            * math.sin(2.0 * math.pi * frequency * index / sample_rate)
        )
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))


def _write_frq(path, average, values, hop_samples=256):
    payload = bytearray(b"FREQ0003")
    payload.extend(struct.pack("<i", int(hop_samples)))
    payload.extend(struct.pack("<d", float(average)))
    payload.extend(b"\0" * 16)
    payload.extend(struct.pack("<i", len(values)))
    for value in values:
        payload.extend(struct.pack("<dd", float(value), 1.0))
    path.write_bytes(bytes(payload))


def _write_synthetic_bank(root):
    aliases = [
        "あ",
        "- あ",
        "か",
        "か1",
        "- か",
        "a か",
        "a k",
        "a -",
        "う k-",
        "ん",
        "っ",
        "きゃ",
    ]
    lines = []
    for index, alias in enumerate(aliases):
        wav_name = f"unit_{index}.wav"
        _tone_wav(root / wav_name, 165.0 + index * 4.0)
        # 40 ms source lead, 20 ms overlap estimate, 120 ms vowel alignment,
        # 260 ms fixed region, and a 620 ms region end.
        lines.append(f"{wav_name}={alias},40,260,-620,120,20")
    (root / "oto.ini").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return aliases


def _write_source_continuity_bank(root):
    """Write the smallest CV bank that exposes both bridge tie-breaks."""
    units = (
        ("blend.wav", "* \u3044", 180.0, 26.0, 50.0),
        ("ordinary.wav", "\u3044", 205.0, 26.0, 50.0),
        ("a.wav", "\u3042", 170.0, 26.0, 8.0),
        ("na.wav", "\u306a", 190.0, 60.0, 8.0),
    )
    rows = []
    for wav_name, alias, frequency, preutterance, overlap in units:
        _tone_wav(root / wav_name, frequency, duration=0.194)
        rows.append(
            f"{wav_name}={alias},16,52,69,{preutterance:g},{overlap:g}"
        )
    (root / "oto.ini").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def _explicit_graph(bank, bank_type="cvvc"):
    profile = infer_bank_profile(
        bank, bank_configuration=bank_type
    )
    return jc.compile_candidate_graph(bank, profile=profile)


def _tree_hash(root, *, excluded=()):
    excluded = set(excluded)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_est_index(path):
    rows = {}
    in_data = False
    for line in path.read_text(encoding="ascii").splitlines():
        if line == "EST_Header_End":
            in_data = True
            continue
        if not in_data or not line.strip():
            continue
        key, wav_stem, start, midpoint, end = line.split()
        rows[key] = (
            wav_stem, float(start), float(midpoint), float(end)
        )
    return rows


def _wsl_festival_available():
    if os.name != "nt":
        return shutil.which("festival") is not None
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        return False
    try:
        result = subprocess.run(
            [wsl, "-d", "Ubuntu", "--", "which", "festival"],
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _wsl_tools_available():
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="pkg_resources is deprecated as an API.*")
            import pyworld  # noqa: F401
    except ImportError:
        return False
    return _wsl_festival_available()


class JapaneseFestivalCompilerTests(unittest.TestCase):
    def test_phone_definitions_distinguish_continuants_from_closures(self):
        self.assertEqual(
            jf._phone_definition("N"), "   (N - 0 - - - n a +)")
        self.assertEqual(
            jf._phone_definition("m"), "   (m - 0 - - - n a +)")
        self.assertEqual(
            jf._phone_definition("r"), "   (r - 0 - - - l a +)")
        self.assertEqual(
            jf._phone_definition("w"), "   (w - 0 - - - r a +)")
        self.assertEqual(
            jf._phone_definition("z"), "   (z - 0 - - - f a +)")
        self.assertEqual(
            jf._phone_definition("k"), "   (k - 0 - - - s a -)")
        self.assertEqual(
            jf._phone_definition("cl"), "   (cl - 0 - - - s a -)")

    def test_cv_structural_closure_uses_consonant_holds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "bank"
            bank.mkdir()
            rows = []
            for wav_name, alias, frequency in (
                ("a.wav", "\u3042", 170.0),
                ("ka.wav", "\u304b", 175.0),
                ("ba.wav", "\u3070", 180.0),
            ):
                _tone_wav(bank / wav_name, frequency, duration=0.30)
                rows.append(
                    f"{wav_name}={alias},16,100,-220,60,12"
                )
            (bank / "oto.ini").write_text(
                "\n".join(rows) + "\n", encoding="utf-8"
            )
            graph = _explicit_graph(bank, bank_type="cv")
            output = root / "voice"
            build = jf.compile_festival_voice(
                graph,
                output,
                voice_name="cv_closure_test",
                average_pitch_hz=180.0,
                pitchmark=False,
                wsl_distro=None,
            )

            self.assertIn("k-k", build.alternatives)
            self.assertIn("b-b", build.alternatives)
            voiceless = build.alternatives["k-k"][0]
            voiced = build.alternatives["b-b"][0]
            self.assertEqual(
                voiceless["role"],
                "structural_consonant_hold",
            )
            self.assertEqual(
                voiced["role"],
                "structural_consonant_hold",
            )

    def test_vowel_blend_seeds_both_generated_bridge_halves(self):
        with tempfile.TemporaryDirectory() as temp:
            bank = Path(temp) / "bank"
            bank.mkdir()
            _write_source_continuity_bank(bank)
            graph = _explicit_graph(bank, bank_type="cv")
            raw_units = []
            for candidate in graph.candidates:
                for proposal in jf.candidate_edge_proposals(
                    candidate, 0.194
                ):
                    raw_units.append(
                        (candidate, proposal, candidate.source.wav_raw)
                    )

            pools = jf._collect_bridge_half_pools(raw_units)
            incoming = pools.right_best["i"]
            ordinary = pools.left_best["i"]
            outgoing = pools.left_continuity["i"][0]

            self.assertEqual(incoming.candidate.role, "vowel_blend")
            self.assertEqual(ordinary.candidate.role, "mora_cv")
            self.assertEqual(outgoing.candidate.role, "vowel_blend")
            self.assertEqual(
                incoming.candidate.candidate_id,
                outgoing.candidate.candidate_id,
            )
            self.assertEqual(incoming.purpose, "right_vowel_blend")
            self.assertEqual(outgoing.purpose, "left_stable_phone")
            self.assertAlmostEqual(outgoing.start_ms, 45.0)
            self.assertLess(outgoing.start_ms, incoming.end_ms)
            self.assertAlmostEqual(outgoing.end_ms, 125.0)
            self.assertAlmostEqual(
                outgoing.end_ms - outgoing.start_ms, 80.0
            )

            output = Path(temp) / "voice"
            build = jf.compile_festival_voice(
                graph,
                output,
                voice_name="continuity_test",
                average_pitch_hz=180.0,
                pitchmark=False,
                wsl_distro=None,
            )
            incoming_choices = [
                choice for choice in build.alternatives["a-i"]
                if choice["role"] == "generated_cv_bridge"
            ]
            self.assertIn(
                "i-n", build.alternatives,
                msg=repr(sorted(build.alternatives)),
            )
            outgoing_choices = [
                choice for choice in build.alternatives["i-n"]
                if choice["role"] == "generated_cv_bridge"
                and choice["recorded_right_context"] == "*"
            ]
            incoming_source = incoming_choices[0][
                "right_source_candidate_id"
            ]

            self.assertGreaterEqual(len(outgoing_choices), 2)
            self.assertNotEqual(
                outgoing_choices[0]["left_source_candidate_id"],
                incoming_source,
            )
            self.assertTrue(any(
                choice["left_source_candidate_id"] == incoming_source
                for choice in outgoing_choices[1:]
            ))
            companion = next(
                choice for choice in outgoing_choices[1:]
                if choice["left_source_candidate_id"] == incoming_source
            )
            self.assertEqual(companion["left_source_role"], "vowel_blend")
            self.assertAlmostEqual(
                companion["vowel_blend_activation_seconds"], 0.165
            )
            scheme = (
                output / "festvox" / "continuity_test_ja.scm"
            ).read_text(encoding="utf-8")
            self.assertIn("selected_right_source_candidate_id", scheme)
            self.assertIn("choice_continuity", scheme)
            self.assertIn("choice_blend_activation", scheme)
            self.assertIn("phrase_boundary_phone outer_left", scheme)
            selector = scheme.split(
                "(define (continuity_test_ja_select_one", 1
            )[1].split(
                "(define (continuity_test_ja_select_list", 1
            )[0]
            self.assertNotIn("festvox_gui_legacy_joins", selector)
            self.assertIn("seg\n                                          t", selector)
            self.assertNotIn("(= score best_score)", scheme)
            self.assertEqual(scheme.count("("), scheme.count(")"))

    def test_source_continuity_selector_is_sequenced_and_tie_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "bank"
            bank.mkdir()
            _write_source_continuity_bank(bank)
            graph = _explicit_graph(bank, bank_type="cv")
            output = root / "voice"
            build = jf.compile_festival_voice(
                graph,
                output,
                voice_name="selector_structure",
                average_pitch_hz=180.0,
                pitchmark=False,
                wsl_distro=None,
            )

            incoming = next(
                choice for choice in build.alternatives["a-i"]
                if choice["role"] == "generated_cv_bridge"
            )
            all_choices = [
                choice for choice in build.alternatives["i-n"]
                if choice["role"] == "generated_cv_bridge"
            ]
            choices = [
                choice for choice in all_choices
                if choice["recorded_right_context"] == "*"
            ]
            self.assertEqual(len(choices), 2)
            base, companion = choices
            self.assertEqual(base["left_source_role"], "mora_cv")
            self.assertEqual(companion["left_source_role"], "vowel_blend")
            self.assertEqual(
                companion["left_source_candidate_id"],
                incoming["right_source_candidate_id"],
            )
            for field in (
                "role",
                "recorded_left_context",
                "recorded_right_context",
                "selection_cost",
                "moraic_nasal_allophone",
                "continuity_group_id",
            ):
                self.assertEqual(base[field], companion[field], field)
            self.assertTrue(base["continuity_group_id"])
            self.assertAlmostEqual(
                companion["vowel_blend_activation_seconds"], 0.165
            )

            scheme = (
                output / "festvox" / "selector_structure_ja.scm"
            ).read_text(encoding="utf-8")
            select_list = scheme.split(
                "(define (selector_structure_ja_select_list", 1
            )[1].split(
                "(define (selector_structure_ja_select_units", 1
            )[0]
            self.assertNotIn("(cons ", select_list)
            self.assertRegex(
                select_list,
                r"\(begin\s+"
                r"\(selector_structure_ja_select_one\s+"
                r"\(car segments\) index\)\s+"
                r"\(selector_structure_ja_select_list\s+"
                r"\(cdr segments\) \(\+ index 1\)\)",
            )

            best_choice = scheme.split(
                "(define (selector_structure_ja_best_choice", 1
            )[1].split(
                "(define (selector_structure_ja_select_one", 1
            )[0]
            exact_score = (
                "(and (not (or (> score best_score)\n"
                "                                    (< score best_score)))"
            )
            self.assertIn(exact_score, best_choice)
            same_group = (
                "(string-equal\n"
                "                            "
                "(selector_structure_ja_choice_continuity_group\n"
                "                             (car choices))\n"
                "                            "
                "(selector_structure_ja_choice_continuity_group best))"
            )
            self.assertIn(same_group, best_choice)
            continuity_rank = "(> continuity best_continuity)"
            self.assertIn(continuity_rank, best_choice)
            self.assertLess(
                best_choice.index(exact_score),
                best_choice.index(same_group),
            )
            self.assertLess(
                best_choice.index(same_group),
                best_choice.index(continuity_rank),
            )
            self.assertNotIn("(= score best_score)", best_choice)

            continuity = scheme.split(
                "(define (selector_structure_ja_choice_continuity", 1
            )[1].split(
                "(define (selector_structure_ja_segment_duration", 1
            )[0]
            source_match = continuity.index(
                "(string-equal "
                "(selector_structure_ja_choice_left_source choice)"
            )
            blend_fallback = continuity.index(
                '"vowel_blend"'
            )
            self.assertLess(source_match, blend_fallback)
            self.assertIn(
                "selector_structure_ja_phrase_boundary_phone outer_left",
                continuity,
            )
            for boundary in ('"pau"', '"sil"', '"sp"', '"*"'):
                self.assertIn(
                    f"(string-equal phone {boundary})",
                    scheme,
                )
            self.assertIn(
                "(selector_structure_ja_choice_blend_activation choice)",
                continuity,
            )
            select_one = scheme.split(
                "(define (selector_structure_ja_select_one", 1
            )[1].split(
                "(define (selector_structure_ja_select_list", 1
            )[0]
            self.assertNotIn("festvox_gui_legacy_joins", select_one)
            self.assertIn("seg\n                                          t", select_one)
            manual_guard = "(if (and wanted"
            manual_choice = "(selector_structure_ja_variant_by_left"
            automatic_choice = "(selector_structure_ja_best_choice"
            self.assertIn(manual_guard, select_one)
            self.assertIn(manual_choice, select_one)
            self.assertIn(automatic_choice, select_one)
            self.assertLess(
                select_one.index(manual_guard),
                select_one.index(manual_choice),
            )
            self.assertLess(
                select_one.index(manual_choice),
                select_one.index(automatic_choice),
            )

    def test_generated_bridge_failures_are_deterministic_and_path_private(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "bank"
            bank.mkdir()
            _write_source_continuity_bank(bank)
            oto_path = bank / "oto.ini"
            oto_rows = oto_path.read_text(encoding="utf-8").splitlines()
            oto_path.write_text(
                "\n".join(
                    row.rsplit(",", 1)[0] + ",50"
                    if row.startswith("na.wav=") else row
                    for row in oto_rows
                ) + "\n",
                encoding="utf-8",
            )
            graph = _explicit_graph(bank, bank_type="cv")

            serialized = []
            for name in ("first", "second"):
                build = jf.compile_festival_voice(
                    graph,
                    root / name,
                    voice_name="bridge_failure_fixture",
                    average_pitch_hz=180.0,
                    pitchmark=False,
                    wsl_distro=None,
                )
                diagnostic = next(
                    row for row in build.diagnostics
                    if row.code == "generated_transition_source_unavailable"
                )
                failures = diagnostic.details["failures"]
                self.assertGreater(len(failures), 0)
                for failure in failures:
                    self.assertRegex(failure["failure_id"], r"^jbf_[0-9a-f]{20}$")
                    self.assertIn(failure["stage"], {
                        "join_search", "source_geometry", "index_geometry",
                        "join_validation", "source_read", "bridge_render",
                        "source_selection",
                    })
                    self.assertNotIn(str(root), json.dumps(failure))
                    self.assertNotIn("\\", failure["code"])
                    self.assertNotIn("/", failure["code"])
                serialized.append(json.dumps(
                    failures,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ))

            self.assertEqual(serialized[0], serialized[1])

    def test_failed_continuity_companion_is_reported_but_not_selectable(self):
        def failed_measurement(left, right, rate, *_args, **_kwargs):
            overlap = min(64, len(left), len(right))
            start = len(left) - overlap
            return tuple(left) + tuple(right), {
                "validation_failures": ("CONTENT_RETENTION",),
                "validation_passed": False,
                "content_preservation_passed": False,
                "legacy_fallback_used": True,
                "splice_sample": start + overlap // 2,
                "handoff_start_sample": start,
                "handoff_end_sample": start + overlap,
                "left_trim_samples": 0,
                "right_skip_samples": 0,
            }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "bank"
            bank.mkdir()
            _write_source_continuity_bank(bank)
            graph = _explicit_graph(bank, bank_type="cv")
            with mock.patch.object(
                jf, "_measured_bridge", side_effect=failed_measurement
            ):
                build = jf.compile_festival_voice(
                    graph,
                    root / "voice",
                    voice_name="failed_companion_fixture",
                    average_pitch_hz=180.0,
                    pitchmark=False,
                    wsl_distro=None,
                )

            outgoing = [
                choice for choice in build.alternatives["i-n"]
                if choice["role"] == "generated_cv_bridge"
                and choice["recorded_right_context"] == "*"
            ]
            self.assertEqual(len(outgoing), 1)
            self.assertEqual(outgoing[0]["left_source_role"], "mora_cv")
            diagnostic = next(
                row for row in build.diagnostics
                if row.code == "generated_transition_source_unavailable"
            )
            companion_failures = [
                row for row in diagnostic.details["failures"]
                if row["code"] == "continuity_companion_validation_failed"
            ]
            self.assertGreater(len(companion_failures), 0)
            self.assertTrue(all(
                row["stage"] == "join_validation"
                for row in companion_failures
            ))

    @unittest.skipUnless(
        _wsl_festival_available(),
        "Festival is not installed in the local WSL distro",
    )
    def test_wsl_source_continuity_selector_runtime_precedence(self):
        from festvox_core import win_to_wsl_path

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "bank"
            bank.mkdir()
            _write_source_continuity_bank(bank)
            graph = _explicit_graph(bank, bank_type="cv")
            output = root / "voice"
            build = jf.compile_festival_voice(
                graph,
                output,
                voice_name="selector_runtime",
                average_pitch_hz=180.0,
                pitchmark=False,
                wsl_distro=None,
            )
            all_choices = [
                choice for choice in build.alternatives["i-n"]
                if choice["role"] == "generated_cv_bridge"
            ]
            choices = [
                choice for choice in all_choices
                if choice["recorded_right_context"] == "*"
            ]
            self.assertEqual(len(choices), 2)
            base, companion = choices
            context_base = next(
                choice for choice in all_choices
                if choice["left_source_role"] == "mora_cv"
                and choice["recorded_right_context"] == "a"
            )

            scheme_path = win_to_wsl_path(str(
                output / "festvox" / "selector_runtime_ja.scm"
            ))
            runtime_root = win_to_wsl_path(str(output))
            script = root / "selector_runtime_test.scm"
            script.write_text(
                f'''(set! load-path (cons "{runtime_root}" load-path))
(load "{scheme_path}")
(define (selector_runtime_append_segments utt rows)
  (if (null rows)
      utt
      (let ((seg (utt.relation.append utt 'Segment)))
        (item.set_name seg (car (car rows)))
        (item.set_feat seg "end" (car (cdr (car rows))))
        (selector_runtime_append_segments utt (cdr rows)))))
(define (selector_runtime_utterance rows)
  (let ((utt (Utterance Text "")))
    (utt.relation.create utt 'Segment)
    (selector_runtime_append_segments utt rows)))
(define (selector_runtime_pick seg outer-left outer-right prior enabled)
  (let ((row (assoc_string "i-n" selector_runtime_ja_unit_variants)))
    (selector_runtime_ja_best_choice
     (cadr row) outer-left outer-right "" prior seg enabled
     nil -100000 -1)))

(set! short-u (selector_runtime_utterance
               '(("a" 0.100) ("i" 0.200) ("n" 0.280))))
(set! short-i (car (cdr (utt.relation.items short-u 'Segment))))
(set! long-u (selector_runtime_utterance
              '(("a" 0.100) ("i" 0.320) ("n" 0.400))))
(set! long-i (car (cdr (utt.relation.items long-u 'Segment))))
(set! initial-u (selector_runtime_utterance
                 '(("pau" 0.100) ("i" 0.320) ("n" 0.400))))
(set! initial-i (car (cdr (utt.relation.items initial-u 'Segment))))

(set! tie-row (assoc_string "i-n" selector_runtime_ja_unit_variants))
(set! tie-choices (cadr tie-row))
(set! tie-base (car tie-choices))
(set! tie-companion (car (cdr tie-choices)))
(format t "SELECTOR score_base %f\\n"
        (selector_runtime_ja_choice_score tie-base "a" "*" ""))
(format t "SELECTOR score_companion %f\\n"
        (selector_runtime_ja_choice_score tie-companion "a" "*" ""))
(format t "SELECTOR exact_match %s\\n"
        (car (selector_runtime_pick
              short-i "a" "*"
              "{companion['left_source_candidate_id']}" t)))
(format t "SELECTOR legacy %s\\n"
        (car (selector_runtime_pick
              long-i "a" "*"
              "{companion['left_source_candidate_id']}" nil)))
(format t "SELECTOR short_blend %s\\n"
        (car (selector_runtime_pick short-i "a" "*" "" t)))
(format t "SELECTOR long_blend %s\\n"
        (car (selector_runtime_pick long-i "a" "*" "" t)))
(format t "SELECTOR phrase_initial %s\\n"
        (car (selector_runtime_pick initial-i "pau" "*" "" t)))
(format t "SELECTOR phrase_initial_sil %s\\n"
        (car (selector_runtime_pick long-i "sil" "*" "" t)))
(format t "SELECTOR phrase_initial_sp %s\\n"
        (car (selector_runtime_pick long-i "sp" "*" "" t)))
(format t "SELECTOR utterance_initial %s\\n"
        (car (selector_runtime_pick long-i "*" "*" "" t)))
(format t "SELECTOR higher_score %s\\n"
        (car (selector_runtime_ja_best_choice
              (list (car (cdr tie-choices))
                    (car (cdr (cdr tie-choices))))
              "a" "a" "" "{companion['left_source_candidate_id']}"
              long-i
              t nil -100000 -1)))

(set! normal-u (selector_runtime_utterance
                '(("a" 0.100) ("i" 0.200) ("n" 0.280)
                  ("pau" 0.380))))
(set! festvox_gui_legacy_joins nil)
(set! festvox_gui_unit_variant_overrides nil)
(selector_runtime_ja_select_list
 (utt.relation.items normal-u 'Segment) 0)
(format t "SELECTOR normal %s\\n"
        (item.feat (car (cdr (utt.relation.items normal-u 'Segment)))
                   "us_diphone_left"))

(set! legacy-u (selector_runtime_utterance
                '(("a" 0.100) ("i" 0.200) ("n" 0.280)
                  ("pau" 0.380))))
(set! festvox_gui_legacy_joins t)
(set! festvox_gui_unit_variant_overrides nil)
(selector_runtime_ja_select_list
 (utt.relation.items legacy-u 'Segment) 0)
(format t "SELECTOR legacy_runtime %s\\n"
        (item.feat (car (cdr (utt.relation.items legacy-u 'Segment)))
                   "us_diphone_left"))

(set! manual-u (selector_runtime_utterance
                '(("a" 0.100) ("i" 0.200) ("n" 0.280)
                  ("pau" 0.380))))
(set! festvox_gui_legacy_joins nil)
(set! festvox_gui_unit_variant_overrides
      '(("1" "{base['left_name']}")))
(selector_runtime_ja_select_list
 (utt.relation.items manual-u 'Segment) 0)
(format t "SELECTOR manual %s\\n"
        (item.feat (car (cdr (utt.relation.items manual-u 'Segment)))
                   "us_diphone_left"))
''',
                encoding="utf-8",
            )
            wsl = shutil.which("wsl.exe") or shutil.which("wsl")
            result = subprocess.run(
                [
                    wsl,
                    "-d",
                    "Ubuntu",
                    "--",
                    "festival",
                    "-b",
                    win_to_wsl_path(str(script)),
                ],
                capture_output=True,
                text=True,
                timeout=45,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=result.stderr or result.stdout,
            )
            observed = {}
            for line in result.stdout.splitlines():
                if not line.startswith("SELECTOR "):
                    continue
                _marker, key, value = line.split(maxsplit=2)
                observed[key] = value

            self.assertAlmostEqual(
                float(observed["score_base"]),
                float(observed["score_companion"]),
            )
            self.assertEqual(observed["exact_match"], companion["left_name"])
            self.assertEqual(observed["normal"], companion["left_name"])
            self.assertEqual(observed["legacy"], base["left_name"])
            self.assertEqual(
                observed["legacy_runtime"], observed["normal"]
            )
            self.assertEqual(observed["manual"], base["left_name"])
            self.assertEqual(observed["short_blend"], base["left_name"])
            self.assertEqual(observed["long_blend"], companion["left_name"])
            self.assertEqual(observed["phrase_initial"], base["left_name"])
            self.assertEqual(
                observed["phrase_initial_sil"], base["left_name"])
            self.assertEqual(
                observed["phrase_initial_sp"], base["left_name"])
            self.assertEqual(
                observed["utterance_initial"], base["left_name"])
            self.assertEqual(
                observed["higher_score"],
                context_base["left_name"],
            )

    def test_bridge_slice_reader_is_bounded_and_cached(self):
        with tempfile.TemporaryDirectory() as temp:
            wav_path = Path(temp) / "long.wav"
            _tone_wav(wav_path, 180.0, duration=2.0)
            jf._read_pcm_mono_slice.cache_clear()
            first, rate = jf._read_pcm_mono_slice(
                str(wav_path.resolve()), 700.0, 780.0
            )
            before = jf._read_pcm_mono_slice.cache_info()
            second, second_rate = jf._read_pcm_mono_slice(
                str(wav_path.resolve()), 700.0, 780.0
            )
            after = jf._read_pcm_mono_slice.cache_info()

            self.assertEqual(rate, 16000)
            self.assertEqual(second_rate, rate)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 1280)
            self.assertEqual(after.hits, before.hits + 1)
            self.assertLess(len(first), int(rate * 2.0))
            jf._read_pcm_mono_slice.cache_clear()

    def test_periodic_fricative_bridge_uses_phone_hint_not_periodicity(self):
        sample_rate = 16000
        samples = tuple(
            0.2 * math.sin(2.0 * math.pi * 200.0 * index / sample_rate)
            for index in range(int(sample_rate * 0.08))
        )

        _output, conditioning = jf._measured_bridge(
            samples, samples, sample_rate, 8.0, 200.0,
            left_phone="s", right_phone="s")

        self.assertFalse(conditioning["voiced"])
        self.assertEqual(
            conditioning["voicing_hint_reason"],
            "phone-context-forced-aperiodic")

    def compile_fixture(
        self, root, *, pitchmark=False, bank_type="cvvc", **compile_options
    ):
        bank = root / "bank"
        bank.mkdir()
        _write_synthetic_bank(bank)
        graph = _explicit_graph(bank, bank_type=bank_type)
        output = root / "voice"
        build = jf.compile_festival_voice(
            graph,
            output,
            voice_name="phase3_test",
            average_pitch_hz=180.0,
            pitchmark=pitchmark,
            wsl_distro="Ubuntu",
            **compile_options,
        )
        return bank, graph, output, build

    def test_frq_guided_pitchmarks_are_deterministic_and_repair_octaves(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "bank"
            output = root / "voice"
            (output / "wav").mkdir(parents=True)
            bank.mkdir()
            source = bank / "tone.wav"
            _tone_wav(source, 165.0, duration=0.12)
            # One deliberate half-frequency frame models the bridge failure
            # that formerly created an isolated 104 Hz source epoch.
            _write_frq(
                bank / "tone_wav.frq",
                165.0,
                [165.0, 165.0, 82.5, 165.0, 165.0, 165.0, 165.0, 165.0],
            )
            shutil.copyfile(source, output / "wav" / "tone.wav")
            unit = jf.JapaneseCompiledUnit(
                candidate_id="fixture",
                edge_index=0,
                edge_offset=0,
                diphone="a-a",
                left_phone="a",
                right_phone="a",
                left_name="a",
                index_name="a-a",
                wav_name="tone.wav",
                start=0.0,
                midpoint=0.06,
                end=0.12,
                role="mora_cv",
                family="cv",
                selection_cost=0.0,
                geometry_method="fixture",
                source_path="tone.wav",
                source_alias="a",
                source_oto_path="oto.ini",
                source_oto_line=1,
                shared_anchor=None,
                oto_offset_ms=0.0,
                oto_consonant_ms=0.0,
                oto_cutoff_ms=0.0,
                oto_preutterance_ms=0.0,
                oto_overlap_ms=0.0,
            )

            jf.make_pitchmarks(
                output,
                ["tone.wav"],
                f0_min=100.0,
                f0_max=280.0,
                default_f0=165.0,
                source_root=bank,
                units=(unit,),
                distro=None,
            )
            pitchmark = output / "pm" / "tone.pm"
            first = pitchmark.read_bytes()
            legacy_pitchmark = output / "pm" / "tone.legacy.pm"
            first_legacy = legacy_pitchmark.read_bytes()
            f0_sidecar = output / "pm" / "tone.f0.json"
            first_f0 = f0_sidecar.read_bytes()
            jf.make_pitchmarks(
                output,
                ["tone.wav"],
                f0_min=100.0,
                f0_max=280.0,
                default_f0=165.0,
                source_root=bank,
                units=(unit,),
                distro=None,
            )
            self.assertEqual(first, pitchmark.read_bytes())
            self.assertEqual(first_legacy, legacy_pitchmark.read_bytes())
            self.assertEqual(first_f0, f0_sidecar.read_bytes())
            marks = [
                float(line.split()[0])
                for line in pitchmark.read_text(encoding="ascii").splitlines()
                if line and line[0].isdigit()
            ]
            periods = [right - left for left, right in zip(marks, marks[1:])]
            ratios = [max(a / b, b / a)
                      for a, b in zip(periods, periods[1:])]
            self.assertGreater(len(marks), 10)
            self.assertLess(max(ratios), 1.30)
            manifest = json.loads(
                (output / "pm" / "pitchmark_sources.json").read_text(
                    encoding="utf-8"))
            self.assertEqual(
                manifest["units"]["tone.wav"]["f0_source"], "utau-frq")
            self.assertEqual(
                manifest["units"]["tone.wav"]["f0_file"], "tone.f0.json")
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(
                manifest["units"]["tone.wav"]["legacy_pitchmark_file"],
                "tone.legacy.pm")
            self.assertEqual(
                manifest["units"]["tone.wav"]["phase_reference"],
                "negative-zero-crossing")
            self.assertEqual(first, first_legacy)
            self.assertGreater(
                manifest["units"]["tone.wav"]["aligned_epoch_count"], 0)
            sidecar = json.loads(f0_sidecar.read_text(encoding="utf-8"))
            self.assertEqual(sidecar["f0_source"], "utau-frq")
            self.assertTrue(any(row[1] > 0.0 for row in sidecar["frames"]))

    def test_excitation_epochs_keep_phase_across_shifted_recordings(self):
        sample_rate = 16000
        period = 80
        frequency = sample_rate / period
        guide = jf._F0Guide(
            (0.0, 0.5), (frequency, frequency), "fixture")

        def source(shift):
            return [
                math.exp(-(((index - shift) % period)) / 12.0)
                for index in range(sample_rate // 2)
            ]

        phases = []
        for shift in (0, 17):
            marks, diagnostics = jf._generate_pitchmarks_with_diagnostics(
                source(shift), sample_rate, guide,
                default_f0=frequency, f0_min=80.0, f0_max=500.0)
            phase = int(round(statistics.median(
                [(mark * sample_rate) % period for mark in marks]))) % period
            distance = min((phase - shift) % period,
                           (shift - phase) % period)
            self.assertLessEqual(distance, 4)
            self.assertGreater(diagnostics["aligned_epoch_count"], 20)
            phases.append((phase - shift) % period)

        phase_delta = min((phases[0] - phases[1]) % period,
                          (phases[1] - phases[0]) % period)
        self.assertLessEqual(phase_delta, 3)

    @unittest.skipUnless(
        importlib.util.find_spec("pyworld") is not None,
        "optional pyworld dependency is unavailable",
    )
    def test_world_harvest_and_dio_are_explicit_deterministic_fallbacks(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "tone.wav"
            _tone_wav(source, 165.0, duration=0.35)
            samples, sample_rate = jf._read_pcm_mono(source)
            for estimator in jf.F0_FALLBACK_ESTIMATORS:
                first = jf._world_f0_guide(
                    samples, sample_rate, estimator=estimator,
                    f0_min=100.0, f0_max=280.0)
                second = jf._world_f0_guide(
                    samples, sample_rate, estimator=estimator,
                    f0_min=100.0, f0_max=280.0)
                self.assertEqual(first, second)
                voiced = [value for value in first.values if value > 0.0]
                self.assertTrue(voiced)
                self.assertAlmostEqual(
                    statistics.median(voiced), 165.0, delta=4.0)
                self.assertEqual(
                    first.provenance,
                    "world-%s-stonemask" % estimator)

            with self.assertRaisesRegex(ValueError, "harvest, dio"):
                jf._world_f0_guide(
                    samples, sample_rate, estimator="crepe",
                    f0_min=100.0, f0_max=280.0)

            silence = [0.0] * sample_rate
            unvoiced = jf._world_f0_guide(
                silence, sample_rate, estimator="harvest",
                f0_min=100.0, f0_max=280.0)
            self.assertTrue(all(value == 0.0 for value in unvoiced.values))
            self.assertEqual(
                unvoiced.provenance,
                "world-harvest-stonemask-unvoiced",
            )

    def test_compiled_vowel_edge_is_labeled_vv_not_vcv(self):
        unit = jf.JapaneseCompiledUnit(
            candidate_id="vv",
            edge_index=0,
            edge_offset=0,
            diphone="a-i",
            left_phone="a",
            right_phone="i",
            left_name="a",
            index_name="a-i",
            wav_name="vv.wav",
            start=0.0,
            midpoint=0.05,
            end=0.10,
            role="vcv_mora",
            family="vcv",
            selection_cost=0.0,
            geometry_method="fixture",
            source_path="vv.wav",
            source_alias="a い",
            source_oto_path="oto.ini",
            source_oto_line=1,
            shared_anchor=None,
            oto_offset_ms=0.0,
            oto_consonant_ms=0.0,
            oto_cutoff_ms=0.0,
            oto_preutterance_ms=0.0,
            oto_overlap_ms=0.0,
        )

        payload = jf._choice_payload(unit)

        self.assertEqual(payload["transition_kind"], "vv")
        self.assertEqual(payload["family"], "vcv")
        self.assertEqual(payload["wav_name"], "vv.wav")

    def test_cv_vcv_cvvc_and_release_edges_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            bank = Path(temp) / "bank"
            bank.mkdir()
            _write_synthetic_bank(bank)
            graph = _explicit_graph(bank)
            by_alias = {
                candidate.source.alias_raw: candidate
                for candidate in graph.candidates
            }

            cv = jf.candidate_edge_proposals(by_alias["か"], 0.75)
            vcv = jf.candidate_edge_proposals(by_alias["a か"], 0.75)
            vc = jf.candidate_edge_proposals(by_alias["a k"], 0.75)
            release = jf.candidate_edge_proposals(by_alias["う k-"], 0.75)

            self.assertEqual([(x.left, x.right) for x in cv], [("k", "a")])
            self.assertEqual(
                [(x.left, x.right) for x in vcv],
                [("a", "k"), ("k", "a")],
            )
            self.assertEqual(
                [(x.left, x.right) for x in vc],
                [("a", "k")],
            )
            self.assertEqual(
                [(x.left, x.right) for x in release],
                [("u", "k"), ("k", "pau")],
            )

    def test_zero_overlap_guard_changes_bridge_clip_not_recorded_cv_edge(self):
        with tempfile.TemporaryDirectory() as temp:
            bank = Path(temp) / "bank"
            bank.mkdir()
            _write_synthetic_bank(bank)
            oto = bank / "oto.ini"
            oto.write_text(
                oto.read_text(encoding="utf-8").replace(",120,20", ",120,0"),
                encoding="utf-8",
            )
            graph = _explicit_graph(bank)
            candidate = next(
                item for item in graph.candidates
                if item.role == "mora_cv" and len(item.target.phones) >= 2
            )

            guarded_edge = jf.candidate_edge_proposals(candidate, 0.75)[0]
            legacy_edge = jf.candidate_edge_proposals(
                candidate, 0.75, zero_overlap_guard_ms=0
            )[0]
            raw_units = []
            for item in graph.candidates:
                for proposal in jf.candidate_edge_proposals(item, 0.75):
                    raw_units.append((
                        item,
                        proposal,
                        str(item.source.wav_path or "source.wav"),
                    ))
            _left, guarded_pool = jf._bridge_half_pools(
                raw_units, zero_overlap_guard_ms=12
            )
            _left, legacy_pool = jf._bridge_half_pools(
                raw_units, zero_overlap_guard_ms=0
            )
            guarded_half = guarded_pool[candidate.target.phones[-2]]
            legacy_half = legacy_pool[candidate.target.phones[-2]]

            self.assertEqual(candidate.timing.overlap, 0.0)
            self.assertEqual(guarded_edge.start_ms, legacy_edge.start_ms)
            self.assertEqual(guarded_edge.midpoint_ms, legacy_edge.midpoint_ms)
            self.assertEqual(guarded_edge.effective_overlap_ms, 0.0)
            self.assertEqual(
                guarded_edge.overlap_method, "oto_offset_fallback"
            )
            self.assertGreater(guarded_half.start_ms, legacy_half.start_ms)
            self.assertEqual(guarded_half.effective_overlap_ms, 12.0)
            self.assertEqual(
                guarded_half.overlap_method, "inferred_zero_overlap_guard"
            )

    def test_phrase_initial_vcv_vowel_uses_preutterance_as_onset(self):
        with tempfile.TemporaryDirectory() as temp:
            bank = Path(temp) / "bank"
            bank.mkdir()
            _write_synthetic_bank(bank)
            graph = _explicit_graph(bank, bank_type="vcv")
            candidate = next(
                item for item in graph.candidates
                if item.role == "phrase_start_cv"
                and len(item.target.phones) == 1
            )

            proposals = jf.candidate_edge_proposals(candidate, 0.75)
            self.assertEqual(len(proposals), 1)
            proposal = proposals[0]
            expected = (
                float(candidate.timing.offset)
                + float(candidate.timing.preutterance)
            )

            self.assertEqual((proposal.left, proposal.right),
                             ("pau", candidate.target.phones[0]))
            self.assertAlmostEqual(proposal.midpoint_ms, expected)
            self.assertGreater(proposal.end_ms, proposal.midpoint_ms)
            self.assertIsNone(proposal.shared_anchor_ms)
            self.assertEqual(
                proposal.method, "oto_preutterance_phrase_start_vowel")

    def test_adaptive_source_window_preserves_context_choice_and_full_variant(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, output, build = self.compile_fixture(
                Path(temp), source_window_mode="adaptive",
                source_window_ms=80.0,
            )
            choice = next(
                row for rows in build.alternatives.values() for row in rows
                if row["role"] == "phrase_start_cv"
                and str(row["diphone"]).startswith("pau-")
            )
            source_slice = choice["source_slice"]
            full_slice = choice["source_window"]["full"]

            self.assertAlmostEqual(
                source_slice["phone_boundary"],
                (choice["oto_timing_ms"]["offset"]
                 + choice["oto_timing_ms"]["preutterance"]) / 1000.0,
            )
            self.assertLessEqual(
                source_slice["end"] - source_slice["phone_boundary"], .080001)
            self.assertGreater(full_slice["end"], source_slice["end"])
            self.assertEqual(choice["candidate_id"], choice["id"])
            self.assertIn(
                choice["window_right_name"] + "-" + choice["diphone"].split("-", 1)[1],
                build.index,
            )
            scheme = (output / "festvox" / "phase3_test_ja.scm").read_text(
                encoding="utf-8")
            self.assertIn("source_window_name", scheme)
            self.assertIn("variant_by_left", scheme)
            self.assertIn("group/phase3_test_diphone.group", scheme)
            self.assertIn("phase3_test_ja_grouped_db_params", scheme)
            self.assertIn(
                "festvox_gui_force_separate_database", scheme)
            self.assertIn("festvox_gui_legacy_joins", scheme)
            self.assertIn(".legacy.pm", scheme)
            self.assertIn("phase3_test_ja_active_db_name", scheme)
            self.assertIn("phase3_test_ja_configure_join_windows", scheme)
            self.assertIn(
                '(Param.set "unisyn.window_name" "hanning")', scheme)
            self.assertIn(
                '(Param.set "unisyn.window_factor" 1.0)', scheme)
            self.assertNotIn(
                '(Param.set "unisyn.window_symmetric" 0)', scheme)
            self.assertIn(
                '(Param.set "unisyn.window_symmetric" 1)', scheme)
            self.assertFalse(
                build.source_window_policy[
                    "normal_unisyn_window_symmetric"])
            self.assertTrue(
                build.source_window_policy[
                    "legacy_unisyn_window_symmetric"])

    def test_full_source_window_mode_restores_legacy_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, build = self.compile_fixture(
                Path(temp), source_window_mode="full",
                source_window_ms=80.0,
            )
            choice = next(
                row for rows in build.alternatives.values() for row in rows
                if row["role"] == "phrase_start_cv"
                and str(row["diphone"]).startswith("pau-")
            )

            self.assertEqual(choice["source_slice"],
                             choice["source_window"]["full"])
            self.assertEqual(choice["window_left_name"], choice["left_name"])
            self.assertEqual(choice["window_right_name"], choice["left_name"])

    def test_compiler_writes_japanese_only_voice_and_versioned_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, output, build = self.compile_fixture(Path(temp))
            scheme = (output / "festvox" / "phase3_test_ja.scm").read_text(
                encoding="utf-8"
            )
            runtime = jf.load_japanese_runtime_metadata(output)

            self.assertEqual(build.voice_entry_point, "voice_phase3_test_ja")
            self.assertIn("(define (voice_phase3_test_ja)", scheme)
            self.assertIn("(Parameter.set 'Language 'japanese)", scheme)
            self.assertIn("(Parameter.set 'Synth_Method 'UniSyn)", scheme)
            self.assertNotIn("voice_kal_diphone", scheme)
            self.assertNotIn("ARPAbet", scheme)
            self.assertNotIn("ARPAsing", scheme)
            self.assertEqual(runtime["language"], "ja")
            self.assertEqual(runtime["schema_version"], 1)
            self.assertTrue(runtime["source_bundle_id"].startswith("srb_"))
            self.assertTrue(runtime["configuration_id"].startswith("vcfg_"))
            self.assertEqual(runtime["primary_language"], "ja")
            self.assertEqual(runtime["supported_languages"], ["ja"])
            self.assertEqual(
                runtime["voice_entry_points"]["ja"],
                "voice_phase3_test_ja",
            )
            self.assertEqual(
                runtime["voice_configuration"]["bank_type"], "cvvc"
            )
            self.assertIn("pau-pau", runtime["index"])
            self.assertIn("a-k", runtime["alternatives"])
            self.assertIn("k-a", runtime["alternatives"])
            self.assertNotIn("a-cl", runtime["alternatives"])
            self.assertNotIn("cl-k", runtime["index"])
            self.assertIn("k-k", runtime["alternatives"])
            self.assertTrue(any(
                choice["role"] == "structural_consonant_hold"
                for choice in runtime["alternatives"]["k-k"]
            ))
            self.assertEqual(
                runtime["special_phone_realizations"]["phones"]["cl"]["mode"],
                "anticipatory_consonant",
            )
            self.assertIn("(PhoneSet.silences '(pau sil))", scheme)
            self.assertNotIn("(PhoneSet.silences '(pau sil cl))", scheme)
            self.assertGreater(
                runtime["structural_consonant_hold_count"], 0
            )
            generated_count = sum(
                1
                for choices in runtime["alternatives"].values()
                for choice in choices
                if choice["role"] == "generated_cv_bridge"
            )
            validation = runtime["generated_bridge_validation"]
            self.assertEqual(validation["candidate_count"], generated_count)
            self.assertEqual(
                validation["passed_count"] + validation["failed_count"],
                generated_count,
            )
            self.assertEqual(
                [
                    (row["diphone"], row["candidate_id"])
                    for row in validation["candidates"]
                ],
                sorted(
                    (row["diphone"], row["candidate_id"])
                    for row in validation["candidates"]
                ),
            )
            alternatives = json.loads(
                (output / "dic" / "unit_alternatives.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                alternatives["generated_bridge_validation"], validation
            )

    def test_legacy_bridge_est_geometry_matches_paired_wav_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, output, build = self.compile_fixture(Path(temp))
            normal_rows = _read_est_index(
                output / "dic" / "phase3_test_ja_diphone.est"
            )
            legacy_rows = _read_est_index(
                output / "dic" / "phase3_test_ja_diphone_legacy.est"
            )
            runtime = jf.load_japanese_runtime_metadata(output)
            paired_wavs = runtime["join_databases"]["legacy"][
                "generated_bridge_wavs"
            ]
            generated_units = {
                unit.wav_name: unit
                for unit in build.units
                if unit.wav_name in paired_wavs
            }

            self.assertTrue(paired_wavs)
            self.assertEqual(set(normal_rows), set(legacy_rows))
            different_geometry = 0
            for normal_wav, legacy_wav in paired_wavs.items():
                unit = generated_units[normal_wav]
                conditioning = unit.source_components[0][
                    "join_conditioning"
                ]
                geometry = conditioning["legacy_geometry"]
                legacy_path = output / "wav" / legacy_wav
                with wave.open(str(legacy_path), "rb") as handle:
                    sample_rate = handle.getframerate()
                    frame_count = handle.getnframes()
                self.assertEqual(geometry["sample_rate"], sample_rate)
                self.assertEqual(geometry["end_sample"], frame_count)
                expected_midpoint = (
                    float(geometry["midpoint_sample"]) / sample_rate
                )
                expected_end = frame_count / float(sample_rate)
                matching_keys = [
                    key for key, row in normal_rows.items()
                    if row[0] == Path(normal_wav).stem
                ]
                self.assertTrue(matching_keys)
                for key in matching_keys:
                    normal_row = normal_rows[key]
                    legacy_row = legacy_rows[key]
                    self.assertEqual(
                        legacy_row[0], Path(legacy_wav).stem
                    )
                    self.assertLessEqual(
                        abs(legacy_row[2] - expected_midpoint) * sample_rate,
                        1.0,
                    )
                    self.assertLessEqual(
                        abs(legacy_row[3] - expected_end) * sample_rate,
                        1.0,
                    )
                    self.assertLessEqual(
                        legacy_row[3] * sample_rate,
                        frame_count + 1e-7,
                    )
                    if normal_row[1:] != legacy_row[1:]:
                        different_geometry += 1

            self.assertGreater(different_geometry, 0)
            for _key, (wav_stem, start, midpoint, end) in legacy_rows.items():
                wav_path = output / "wav" / f"{wav_stem}.wav"
                with wave.open(str(wav_path), "rb") as handle:
                    sample_rate = handle.getframerate()
                    frame_count = handle.getnframes()
                self.assertLessEqual(0.0, start)
                self.assertLessEqual(start, midpoint)
                self.assertLessEqual(midpoint, end)
                self.assertLessEqual(
                    end * sample_rate, frame_count + 1e-7
                )

    def test_legacy_bridge_pitchmarks_reuse_component_frq_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "bank"
            bank.mkdir()
            _write_synthetic_bank(bank)
            for wav_path in sorted(bank.glob("*.wav")):
                _write_frq(
                    wav_path.with_suffix(".frq"),
                    180.0,
                    [180.0] * 64,
                )
            graph = _explicit_graph(bank)
            output = root / "voice"
            build = jf.compile_festival_voice(
                graph,
                output,
                voice_name="phase3_test",
                average_pitch_hz=180.0,
                pitchmark=True,
                wsl_distro=None,
            )
            runtime = jf.load_japanese_runtime_metadata(output)
            paired_wavs = runtime["join_databases"]["legacy"][
                "generated_bridge_wavs"
            ]
            manifest_path = output / "pm" / "pitchmark_sources.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertTrue(paired_wavs)
            guards = runtime["generated_bridge_pitchmark_guards"]
            self.assertGreater(guards["applied_count"], 0)
            self.assertEqual(guards["unavailable_count"], 0)
            normal_rows = _read_est_index(
                output / "dic" / "phase3_test_ja_diphone.est")
            legacy_rows = _read_est_index(
                output / "dic" / "phase3_test_ja_diphone_legacy.est")
            guarded_units = [
                unit for unit in build.units
                if unit.role == "generated_cv_bridge"
                and any(
                    component.get("purpose") == "left_stable_phone"
                    and component.get("analysis_guard", {}).get(
                        "pitchmark_index"
                    ) == 1
                    for component in unit.source_components
                )
            ]
            self.assertEqual(len(guarded_units), guards["applied_count"])
            for unit in guarded_units:
                marks = jf._read_est_pitchmarks(
                    output / "pm" / (Path(unit.wav_name).stem + ".pm")
                )
                self.assertGreaterEqual(len(marks), 2)
                self.assertAlmostEqual(unit.start, marks[1], places=6)
                self.assertAlmostEqual(
                    normal_rows[unit.index_name][1], marks[1], places=6
                )
                self.assertEqual(legacy_rows[unit.index_name][1], 0.0)
            for legacy_wav in paired_wavs.values():
                self.assertEqual(
                    manifest["units"][legacy_wav]["f0_source"],
                    "utau-frq-components",
                )
                sidecar = json.loads(
                    (output / "pm" / (
                        Path(legacy_wav).stem + ".f0.json"
                    )).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    sidecar["f0_source"], "utau-frq-components"
                )

            sample_legacy_wav = next(iter(paired_wavs.values()))
            sample_stem = Path(sample_legacy_wav).stem
            before = {
                "manifest": manifest_path.read_bytes(),
                "pitchmarks": (
                    output / "pm" / f"{sample_stem}.legacy.pm"
                ).read_bytes(),
                "f0": (
                    output / "pm" / f"{sample_stem}.f0.json"
                ).read_bytes(),
            }
            jf.make_pitchmarks(
                output,
                [path.name for path in (output / "wav").glob("*.wav")],
                f0_min=80.0,
                f0_max=500.0,
                default_f0=180.0,
                source_root=bank,
                units=build.units,
                distro=None,
            )
            self.assertEqual(before["manifest"], manifest_path.read_bytes())
            self.assertEqual(
                before["pitchmarks"],
                (output / "pm" / f"{sample_stem}.legacy.pm").read_bytes(),
            )
            self.assertEqual(
                before["f0"],
                (output / "pm" / f"{sample_stem}.f0.json").read_bytes(),
            )

    def test_numbered_take_is_an_arbitrary_alternative(self):
        with tempfile.TemporaryDirectory() as temp:
            _, graph, _, build = self.compile_fixture(Path(temp))
            ka_ids = {
                candidate.candidate_id
                for candidate in graph.candidates
                if candidate.source.alias_raw in {"か", "か1"}
            }
            choices = build.alternatives["k-a"]
            present = {
                str(choice["candidate_id"])
                for choice in choices
            }

            self.assertTrue(ka_ids <= present)
            self.assertGreaterEqual(len(choices), 2)
            self.assertEqual(choices[0]["left_name"], "k")
            self.assertTrue(all(
                str(choice["left_name"]).isalnum()
                or "__j" in str(choice["left_name"])
                for choice in choices
            ))

    def test_explicit_cvvc_runtime_excludes_vcv_source_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            _, graph, output, build = self.compile_fixture(Path(temp))
            vcv_candidates = tuple(
                item for item in graph.candidates if item.family == "vcv"
            )

            self.assertTrue(vcv_candidates)
            self.assertTrue(all(
                not item.selectable for item in vcv_candidates
            ))
            self.assertTrue(all(
                item.candidate_id not in build.candidate_units
                for item in vcv_candidates
            ))
            choices = [
                choice
                for rows in build.alternatives.values()
                for choice in rows
            ]
            self.assertFalse(any(
                choice.get("family") == "vcv" for choice in choices
            ))
            self.assertFalse(any(
                choice.get("role") == "vcv_mora" for choice in choices
            ))
            runtime = jf.load_japanese_runtime_metadata(output)
            self.assertEqual(
                runtime["runtime_family_policy"]["excluded_families"],
                ["vcv"],
            )
            self.assertEqual(
                runtime["configuration_excluded_candidate_count"],
                len(vcv_candidates),
            )
            self.assertTrue(any(
                item.code == "strict_runtime_family_policy_applied"
                and item.details["excluded_candidate_count"]
                == len(vcv_candidates)
                for item in build.diagnostics
            ))

    def test_explicit_cvvc_keeps_recorded_vv_and_single_phone_vc(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "bank"
            bank.mkdir()
            aliases = ["あ", "い", "な", "a i", "i n", "a か"]
            lines = []
            for index, alias in enumerate(aliases):
                wav_name = f"mixed_{index}.wav"
                _tone_wav(bank / wav_name, 170.0 + index * 3.0)
                lines.append(
                    f"{wav_name}={alias},40,260,-620,120,20"
                )
            (bank / "oto.ini").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

            graph = _explicit_graph(bank, bank_type="cvvc")
            by_alias = {
                item.source.alias_raw: item for item in graph.candidates
            }
            build = jf.compile_festival_voice(
                graph, root / "voice", pitchmark=False
            )

            self.assertFalse(by_alias["a か"].selectable)
            for alias, diphone in (("a i", "a-i"), ("i n", "i-n")):
                self.assertTrue(by_alias[alias].selectable)
                self.assertEqual(by_alias[alias].family, "cvvc")
                self.assertTrue(any(
                    choice["alias"] == alias
                    and choice["family"] == "cvvc"
                    for choice in build.alternatives[diphone]
                ))
            self.assertFalse(any(
                choice.get("alias") == "a か"
                for choices in build.alternatives.values()
                for choice in choices
            ))

    def test_duplicate_audio_keeps_every_stable_candidate_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            bank = Path(temp) / "bank"
            bank.mkdir()
            _tone_wav(bank / "shared.wav", 180.0)
            (bank / "oto.ini").write_text(
                "shared.wav=か,40,260,-620,120,20\n"
                "shared.wav=か1,40,260,-620,120,20\n",
                encoding="utf-8",
            )
            graph = _explicit_graph(bank)
            build = jf.compile_festival_voice(
                graph, Path(temp) / "voice", pitchmark=False
            )

            candidate_ids = {
                candidate.candidate_id for candidate in graph.candidates
            }
            self.assertEqual(
                candidate_ids, set(build.candidate_units)
            )
            self.assertEqual(len(build.alternatives["k-a"]), 2)

    def test_generated_consonant_bridge_matches_following_cv_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "bank"
            bank.mkdir()
            lines = []
            for index, (alias, overlap) in enumerate((
                ("a", 0), ("i", 0), ("wa", 15),
                ("\u3046\u3049", 0),
            )):
                wav_name = f"context_{index}.wav"
                _tone_wav(bank / wav_name, 170.0 + index * 3.0)
                lines.append(
                    f"{wav_name}={alias},0,180,-500,58,{overlap}"
                )
            (bank / "oto.ini").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")

            graph = _explicit_graph(bank, bank_type="cv")
            build = jf.compile_festival_voice(
                graph, root / "voice", voice_name="context_bridge",
                average_pitch_hz=180.0, pitchmark=False)

            contexts = {
                str(choice["recorded_right_context"]): choice
                for choice in build.alternatives["pau-w"]
                if choice["role"] == "generated_cv_bridge"
            }
            self.assertTrue(
                {"*", "a", "o"}.issubset(contexts),
                sorted(contexts),
            )
            wa_choice = next(
                choice for choice in build.alternatives["w-a"]
                if choice["alias"] == "wa")
            contextual = contexts["a"]
            right_component = next(
                component for component in contextual["source_components"]
                if component["purpose"] == "right_consonant_onset")
            self.assertEqual(
                right_component["candidate_id"],
                wa_choice["candidate_id"])
            self.assertAlmostEqual(
                right_component["source_slice"]["end"],
                wa_choice["source_slice"]["phone_boundary"],
                places=6)
            self.assertLessEqual(
                right_component["source_slice"]["start"],
                wa_choice["source_slice"]["start"],
            )
            self.assertGreater(
                right_component["source_slice"]["end"],
                wa_choice["source_slice"]["start"],
            )
            self.assertAlmostEqual(
                right_component["indexed_source_end"],
                wa_choice["source_slice"]["start"],
                places=6,
            )
            self.assertGreater(
                right_component["analysis_guard"]["wav_end"],
                right_component["analysis_guard"]["indexed_end"],
            )
            self.assertAlmostEqual(
                contextual["source_slice"]["end"],
                right_component["analysis_guard"]["indexed_end"],
                places=6,
            )

            # Context-specific bridge generation cannot replace or renumber
            # the following CV candidates themselves.
            self.assertEqual(len(build.alternatives["w-a"]), 1)
            self.assertEqual(len(build.alternatives["w-o"]), 1)

    def test_compilation_is_metadata_deterministic_and_path_private(self):
        with tempfile.TemporaryDirectory() as temp:
            first_root = Path(temp) / "first"
            second_root = Path(temp) / "second"
            first_root.mkdir()
            second_root.mkdir()
            _, _, first_output, first = self.compile_fixture(first_root)
            _, _, second_output, second = self.compile_fixture(second_root)

            self.assertEqual(first.metadata_bytes(), second.metadata_bytes())
            for relative in (
                "dic/diphone_index.json",
                "dic/unit_alternatives.json",
                "dic/phase3_test_ja_diphone.est",
            ):
                self.assertEqual(
                    (first_output / relative).read_bytes(),
                    (second_output / relative).read_bytes(),
                )
            serialized = first.metadata_bytes().decode("utf-8")
            self.assertNotIn(str(first_root), serialized)
            self.assertNotIn(str(second_root), serialized)

    def test_runtime_metadata_advertises_declared_subbanks_and_colors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "bank"
            bank.mkdir()
            _write_synthetic_bank(bank)
            graph = _explicit_graph(bank)
            profile = replace(graph.profile, subbanks=(
                JapaneseSubbank(
                    subbank_id="power-e4", color="Power",
                    suffix="E4P", tone_ranges=("E4-G4",), order=1,
                ),
                JapaneseSubbank(
                    subbank_id="soft-c3", color="Soft",
                    suffix="C3S", tone_ranges=("C3-E3",), order=0,
                ),
            ), voice_color="Power")
            graph = replace(graph, profile=profile)
            output = root / "voice"

            jf.compile_festival_voice(graph, output, pitchmark=False)
            runtime = jf.load_japanese_runtime_metadata(output)

            self.assertEqual(
                [row["subbank_id"] for row in runtime["subbanks"]],
                ["soft-c3", "power-e4"])
            self.assertEqual(runtime["available_voice_colors"],
                             ["Power", "Soft"])
            self.assertEqual(runtime["selected_voice_color"], "Power")

    def test_source_bank_is_byte_unchanged_and_output_guarded(self):
        with tempfile.TemporaryDirectory() as temp:
            bank = Path(temp) / "bank"
            bank.mkdir()
            _write_synthetic_bank(bank)
            before = _tree_hash(bank)
            graph = _explicit_graph(bank)

            with self.assertRaisesRegex(ValueError, "source UTAU"):
                jf.compile_festival_voice(
                    graph, bank / "generated", pitchmark=False
                )
            jf.compile_festival_voice(
                graph, Path(temp) / "voice", pitchmark=False
            )

            self.assertEqual(before, _tree_hash(bank))
            self.assertFalse((bank / "generated").exists())

    def test_every_compiled_geometry_is_coherent(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, build = self.compile_fixture(Path(temp))
            self.assertGreater(build.compiled_candidate_count, 0)
            for unit in build.units:
                self.assertGreaterEqual(unit.start, 0.0)
                self.assertLess(unit.start, unit.midpoint)
                self.assertLess(unit.midpoint, unit.end)
                self.assertIn(unit.index_name, build.index)

    def test_nonlinguistic_breath_is_preserved_as_an_explicit_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            bank = Path(temp) / "bank"
            bank.mkdir()
            _tone_wav(bank / "breath.wav", 170.0)
            (bank / "oto.ini").write_text(
                "breath.wav=breath_A3-1,40,260,-620,120,20\n",
                encoding="utf-8",
            )
            graph = _explicit_graph(bank)
            build = jf.compile_festival_voice(
                graph, Path(temp) / "voice", pitchmark=False
            )

            self.assertEqual(build.compiled_candidate_count, 0)
            self.assertTrue(any(
                item.code == "nonlinguistic_candidate_not_phone"
                and item.severity == "info"
                for item in build.diagnostics
            ))

    def test_synthesis_plan_has_three_internal_pauses_and_structural_f0(self):
        utterance = analyze_japanese("あ。か？", mode="kana")
        plan = js.create_synthesis_plan(
            utterance, base_pitch_hz=180.0
        )
        phones = plan.phones
        runs = []
        index = 0
        while index < len(phones):
            if phones[index] != "pau":
                index += 1
                continue
            end = index
            while end < len(phones) and phones[end] == "pau":
                end += 1
            runs.append((index, end - index))
            index = end

        self.assertEqual([length for _, length in runs], [2, 3, 2])
        internal_start = runs[1][0]
        self.assertEqual(
            [segment.pause_role for segment in
             plan.segments[internal_start:internal_start + 3]],
            ["phrase_guard_out", "phrase_gap", "phrase_guard_in"],
        )
        self.assertTrue(plan.f0_targets)
        self.assertFalse(any(
            "interrogative" in target.kind for target in plan.f0_targets
        ))
        self.assertTrue(any(
            item.code == "question_intonation_uses_blocks"
            for item in plan.diagnostics
        ))
        self.assertTrue(all(
            plan.f0_targets[index].time < plan.f0_targets[index + 1].time
            for index in range(len(plan.f0_targets) - 1)
        ))
        self.assertEqual(plan.schema_version, 3)
        self.assertEqual(
            plan.pitch_model_id,
            "japanese_speaker_relative_log_f0_kokoro_b453f6caf042_v5")
        self.assertTrue(all(math.isfinite(target.log_f0)
                            for target in plan.f0_targets))
        self.assertTrue(all(target.components_semitones
                            for target in plan.f0_targets))
        self.assertTrue(all(abs(
            target.semitones_from_baseline
            - sum(target.components_semitones.values())) < 0.02
            for target in plan.f0_targets))
        self.assertEqual(len(plan.mora_timings), len(utterance.moras))

    def test_manual_vcv_candidate_overrides_only_target_mora_edges(self):
        with tempfile.TemporaryDirectory() as temp:
            _, graph, output, _ = self.compile_fixture(
                Path(temp), bank_type="vcv"
            )
            runtime = jf.load_japanese_runtime_metadata(output)
            candidate = next(
                item for item in graph.candidates
                if item.source.alias_raw == "a か"
            )
            utterance = analyze_japanese("あかあか", mode="kana")
            plan = js.create_synthesis_plan(
                utterance,
                runtime_metadata=runtime,
                manual_candidate_overrides={1: candidate.candidate_id},
            )

            self.assertEqual(len(plan.unit_overrides), 2)
            overridden = set(plan.unit_overrides)
            first_target_start = next(
                segment.index for segment in plan.segments
                if segment.mora_index == 1
            )
            self.assertEqual(overridden, {
                first_target_start - 1, first_target_start,
            })
            later_start = next(
                segment.index for segment in plan.segments
                if segment.mora_index == 3
            )
            self.assertNotIn(later_start - 1, overridden)
            self.assertNotIn(later_start, overridden)

    def test_unknown_candidate_is_diagnostic_not_silent(self):
        utterance = analyze_japanese("かな", mode="kana")
        plan = js.create_synthesis_plan(
            utterance,
            runtime_metadata={"language": "ja", "candidate_units": {}},
            manual_candidate_overrides={0: "jc_missing"},
        )
        self.assertFalse(plan.unit_overrides)
        self.assertTrue(any(
            item.code == "manual_candidate_unknown"
            for item in plan.diagnostics
        ))

    def test_plan_serialization_is_deterministic(self):
        utterance = analyze_japanese("きゃく。", mode="kana")
        first = js.create_synthesis_plan(utterance)
        second = js.create_synthesis_plan(utterance)
        self.assertEqual(first.to_json_bytes(), second.to_json_bytes())
        self.assertEqual(first.to_dict()["language"], "ja")

    def test_plan_uses_shared_speaker_pitch_when_no_override_is_given(self):
        utterance = analyze_japanese("縺九↑", mode="kana")
        runtime = {
            "language": "ja",
            "average_pitch_hz": 220.0,
            "speaker_pitch_analysis": {
                "median_f0_hz": 220.0,
                "low_percentile_f0_hz": 200.0,
                "high_percentile_f0_hz": 245.0,
                "voiced_sample_count": 12,
                "source": "frq",
                "files_used": ["a.frq", "b.frq", "c.frq"],
                "diagnostics": [],
            },
        }

        automatic = js.create_synthesis_plan(
            utterance, runtime_metadata=runtime
        )
        explicit = js.create_synthesis_plan(
            utterance, runtime_metadata=runtime, base_pitch_hz=180.0
        )

        self.assertEqual(automatic.base_pitch_hz, 220.0)
        self.assertEqual(explicit.base_pitch_hz, 180.0)
        self.assertLessEqual(automatic.speaker_low_hz, 200.0)
        self.assertGreaterEqual(automatic.speaker_high_hz, 245.0)
        self.assertTrue(all(
            automatic.speaker_low_hz <= target.hz <=
            automatic.speaker_high_hz
            for target in automatic.f0_targets
        ))

    def test_mora_first_duration_model_and_diagnostics_are_coherent(self):
        utterance = analyze_japanese("きゃんってコー。", mode="kana")
        plan = js.create_synthesis_plan(utterance)
        by_mora = {item.mora_index: item for item in plan.mora_timings}

        self.assertEqual(set(by_mora), {mora.index for mora in utterance.moras})
        for mora in utterance.moras:
            timing = by_mora[mora.index]
            self.assertAlmostEqual(
                timing.final_duration,
                sum(item.final_duration for item in timing.phone_allocation),
                places=6,
            )
            self.assertLessEqual(
                timing.source_safe_min, timing.final_duration)
            self.assertLessEqual(
                timing.final_duration, timing.source_safe_max)
        palatalized = next(
            by_mora[mora.index] for mora in utterance.moras
            if mora.consonant == "ky"
        )
        self.assertEqual(
            [item.phone for item in palatalized.phone_allocation], ["ky", "a"])
        self.assertLess(palatalized.final_duration, 0.19)
        specials = {
            mora.special_mora: by_mora[mora.index]
            for mora in utterance.moras if mora.special_mora
        }
        self.assertIn("moraic_nasal", specials)
        self.assertIn("geminate", specials)
        self.assertIn("long_vowel", specials)
        self.assertLess(
            specials["geminate"].final_duration,
            specials["long_vowel"].final_duration,
        )

    def test_oto_edges_bound_extreme_requested_phone_stretch(self):
        def choice():
            return {"source_slice": {
                "start": 0.0, "phone_boundary": 0.02, "end": 0.04}}

        runtime = {
            "language": "ja",
            "candidate_units": {},
            "alternatives": {
                "pau-k": [choice()],
                "k-a": [choice()],
                "a-pau": [choice()],
            },
        }
        utterance = analyze_japanese("か。", mode="kana")
        plan = js.create_synthesis_plan(
            utterance, runtime_metadata=runtime, speed=4.0,
            duration_model="legacy")
        timing = plan.mora_timings[0]
        consonant = next(
            item for item in timing.phone_allocation if item.phone == "k")

        self.assertEqual(consonant.constraint_source, "profiled_oto_edges")
        # A phone is assembled from the adjacent diphone halves. The
        # profiled reference therefore uses the 20 ms half, not the complete
        # 40 ms source slice.
        self.assertAlmostEqual(consonant.source_reference_duration, 0.02)
        self.assertAlmostEqual(
            consonant.source_profile_reference_duration, 0.02
        )
        self.assertGreaterEqual(
            consonant.final_duration, consonant.source_safe_min)
        self.assertTrue(any(
            item.code == "duration_oto_safe_clamp"
            for item in plan.diagnostics
        ))

    def test_structural_contour_has_declination_and_downstep(self):
        utterance = analyze_japanese("かかかか。", mode="kana")
        phrase = utterance.phrases[0]
        source = phrase.accent_phrases[0]
        first_moras = tuple(
            replace(mora, accent_phrase_index=0)
            for mora in phrase.moras[:2]
        )
        second_moras = tuple(
            replace(mora, accent_phrase_index=1)
            for mora in phrase.moras[2:]
        )
        first = replace(
            source, index=0, moras=first_moras,
            accent_state="accented", accent_nucleus=1)
        second = replace(
            source, index=1, moras=second_moras,
            accent_state="unaccented", accent_nucleus=None)
        edited = replace(
            utterance,
            phrases=(replace(
                phrase, accent_phrases=(first, second)),),
        )
        plan = js.create_synthesis_plan(edited, base_pitch_hz=200.0)

        self.assertTrue(any(
            target.components_semitones.get("phrase_declination", 0.0) < 0.0
            for target in plan.f0_targets[1:]
        ))
        self.assertTrue(all(
            "sentence_declination" not in target.kind
            and "utterance_declination" not in target.components_semitones
            for target in plan.f0_targets
        ))
        self.assertTrue(any(
            "downstep" in target.kind
            for target in plan.f0_targets
        ))
        first_high = max(
            target.hz for target in plan.f0_targets
            if target.accent_phrase_index == 0)
        second_high = max(
            target.hz for target in plan.f0_targets
            if target.accent_phrase_index == 1)
        self.assertLess(second_high, first_high)

    def test_listening_inventory_covers_phase3_quality_cases(self):
        categories = set(jls.required_listening_categories())
        self.assertTrue({
            "vowels",
            "ordinary CV morae",
            "VV transitions",
            "CVVC transitions",
            "moraic nasal",
            "geminate consonant",
            "palatalized mora",
            "long vowels",
            "phrase boundaries",
            "statement",
            "question",
            "accented and unaccented analysis",
            "boundary stress joins",
            "devoiced vowels",
            "multiple accent-phrase downstep",
            "long-phrase declination",
        } <= categories)

    def test_phase3_modules_do_not_import_english_converter(self):
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "import japanese_festival\n"
                    "bad = [name for name in "
                    "('utau2festvox', 'build_festival_voice') "
                    "if name in sys.modules]\n"
                    "print(','.join(bad))\n"
                    "raise SystemExit(bool(bad))\n"
                ),
            ],
            cwd=FESTVOX_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            probe.returncode,
            0,
            msg=f"Japanese compiler imported English modules: {probe.stdout}",
        )

    def test_integrated_arpasing_manifest_adapts_shared_index_for_japanese(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "integrated"
            (root / "dic").mkdir(parents=True)
            (root / "dic" / "diphone_index.json").write_text(
                json.dumps({
                    "kind": "festival_unisyn_runtime_index",
                    "language": "en",
                    "name": "integrated_fixture",
                    "average_pitch_hz": 172.0,
                    "alternatives": {"a-k": []},
                }), encoding="utf-8")
            (root / "dic" / "voice_manifest.json").write_text(
                json.dumps({
                    "kind": "generated_festival_voice_manifest",
                    "supported_languages": ["en", "asaxi", "ja"],
                    "voice_entry_points": {
                        "en": "voice_integrated_fixture_en",
                        "ja": "voice_integrated_fixture_ja",
                    },
                }), encoding="utf-8")

            runtime = jf.load_japanese_runtime_metadata(root)

            self.assertEqual(runtime["language"], "ja")
            self.assertEqual(runtime["voice_entry_point"],
                             "voice_integrated_fixture_ja")
            self.assertEqual(runtime["voice_scm"],
                             "festvox/integrated_fixture.scm")
            self.assertEqual(runtime["shared_runtime_index_language"], "en")
            self.assertEqual(runtime["metadata_adapter"],
                             "integrated-arpasing-japanese-v1")
            self.assertEqual(runtime["average_pitch_hz"], 172.0)

    @unittest.skipUnless(
        _wsl_tools_available(),
        "Festival and EST tools are not installed in the local WSL distro",
    )
    def test_wsl_festival_preserves_timing_f0_and_manual_override(self):
        from festvox_core import FestivalWSLBackend

        with tempfile.TemporaryDirectory() as temp:
            bank = Path(temp) / "bank"
            bank.mkdir()
            _write_synthetic_bank(bank)
            graph = _explicit_graph(bank)
            output = Path(temp) / "voice"
            build = jf.compile_festival_voice(
                graph,
                output,
                voice_name="phase3_render",
                average_pitch_hz=180.0,
                pitchmark=True,
                wsl_distro="Ubuntu",
            )
            runtime = jf.load_japanese_runtime_metadata(output)
            numbered_cv = next(
                item for item in graph.candidates
                if item.source.alias_raw == "か1"
            )
            utterance = analyze_japanese("あか", mode="kana")
            automatic_plan = js.create_synthesis_plan(
                utterance,
                runtime_metadata=runtime,
                base_pitch_hz=180.0,
            )
            manual_plan = js.create_synthesis_plan(
                utterance,
                runtime_metadata=runtime,
                manual_candidate_overrides={1: numbered_cv.candidate_id},
                base_pitch_hz=180.0,
            )
            config = {
                "festival_wsl": {
                    "distro": "Ubuntu",
                    "timeout_s": 180,
                    "voices": {
                        "phase3_render": {
                            "dir": str(output),
                            "voice": build.voice_entry_point,
                            "voice_en": None,
                            "scm": "festvox/phase3_render_ja.scm",
                        }
                    },
                }
            }
            backend = FestivalWSLBackend(config)
            automatic_arguments = automatic_plan.backend_arguments()
            automatic_phones = automatic_arguments.pop("phones")
            automatic = backend.synth_phones(
                automatic_phones,
                "phase3_render",
                **automatic_arguments,
            )
            manual_arguments = manual_plan.backend_arguments()
            manual_phones = manual_arguments.pop("phones")
            result = backend.synth_phones(
                manual_phones,
                "phase3_render",
                **manual_arguments,
            )
            legacy_arguments = manual_plan.backend_arguments()
            legacy_phones = legacy_arguments.pop("phones")
            legacy_arguments["fault_mode"] = {"legacy_joins": True}
            legacy = backend.synth_phones(
                legacy_phones,
                "phase3_render",
                **legacy_arguments,
            )

            self.assertGreater(len(result.samples), 100)
            self.assertGreater(len(legacy.samples), 100)
            self.assertNotEqual(
                result.samples.tobytes(), legacy.samples.tobytes(),
                "Legacy joins must select the retained pre-fix pitchmarks",
            )
            self.assertGreater(result.sr, 0)
            self.assertEqual(
                [segment.phone for segment in result.segments],
                manual_plan.phones,
            )
            for actual, planned in zip(result.segments, manual_plan.segments):
                self.assertAlmostEqual(actual.dur, planned.duration, delta=0.003)
            self.assertTrue(result.targets)
            self.assertTrue(all(
                50.0 <= f0 <= 500.0 for _, f0 in result.targets
            ))
            for index, left_name in manual_plan.unit_overrides.items():
                self.assertEqual(result.selected_units.get(index), left_name)
                self.assertNotEqual(
                    automatic.selected_units.get(index), left_name
                )
            unchanged_edges = (
                set(automatic.selected_units)
                & set(result.selected_units)
                - set(manual_plan.unit_overrides)
            )
            self.assertTrue(unchanged_edges)
            self.assertTrue(all(
                automatic.selected_units[index] == result.selected_units[index]
                for index in unchanged_edges
            ))
            self.assertEqual(result.selected_units, legacy.selected_units)
            self.assertEqual(
                [segment.phone for segment in legacy.segments],
                manual_plan.phones,
            )
            peak = max(abs(float(value)) for value in result.samples)
            self.assertLessEqual(peak, 1.0)


if __name__ == "__main__":
    unittest.main()
