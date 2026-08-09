import json
import math
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from festvox_gui import festvox_core as fc
from voice_paths import (
    VoicePathError,
    migrate_voice_registration,
    validate_build_layout,
    windows_to_wsl_path,
    wsl_to_windows_path,
)


def _tone_wav(path, frequency=180.0, duration=0.8, sample_rate=16000):
    frames = bytearray()
    for index in range(int(duration * sample_rate)):
        sample = int(
            7000.0 * math.sin(2.0 * math.pi * frequency * index / sample_rate)
        )
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))


def _frq(path, average, values):
    payload = bytearray(b"FREQ0003")
    payload.extend(struct.pack("<i", 256))
    payload.extend(struct.pack("<d", float(average)))
    payload.extend(b"\0" * 16)
    payload.extend(struct.pack("<i", len(values)))
    for value in values:
        payload.extend(struct.pack("<dd", float(value), 1.0))
    path.write_bytes(bytes(payload))


def _shared_bank(root):
    bank = root / "samples"
    (bank / "english").mkdir(parents=True)
    (bank / "japanese").mkdir()
    _tone_wav(bank / "sample.wav")
    _frq(bank / "sample_a.frq", 170.0, [165.0, 170.0, 175.0])
    _frq(bank / "sample_b.frq", 180.0, [175.0, 180.0, 185.0])
    _frq(bank / "sample_c.frq", 190.0, [185.0, 190.0, 195.0])
    # This marker lets the Japanese resolver retain the selected OTO scope
    # while treating the shared parent as the read-only sample root.
    (bank / "prefix.map").write_text("", encoding="utf-8")
    english = bank / "english" / "oto.ini"
    english.write_text(
        "../sample.wav=aa,0,300,-700,120,20\n"
        "../sample.wav=- aa,0,300,-700,120,20\n",
        encoding="utf-8",
    )
    japanese = bank / "japanese" / "oto.ini"
    japanese.write_text(
        "../sample.wav=\u3042,0,300,-700,120,20\n"
        "../sample.wav=- \u3042,0,300,-700,120,20\n",
        encoding="utf-8",
    )
    return bank, english, japanese


def _build_args(
    language, bank_type, bank, oto, output, name, explicit_pitch=True
):
    result = [
        "--language", language,
        "--bank-type", bank_type,
        "--samples", str(bank),
        "--oto", str(oto),
        "--output", str(output),
        "--name", name,
        "--skip-pm",
    ]
    if explicit_pitch:
        result.extend([
            "--f0", "180", "--f0-min", "80", "--f0-max", "500",
        ])
    return result


def _tree_bytes(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run_builder(args):
    import build_festival_voice as builder

    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = builder.main(args)
    return result, stdout.getvalue(), stderr.getvalue()


class UnifiedVoiceBuilderTests(unittest.TestCase):
    def test_generated_phone_inventory_declares_structural_cl_without_oto(self):
        import build_festival_voice as builder

        phones = builder.phone_inventory({
            "pau-a": ["unit.wav", 0.0, 0.1, 0.2],
            "a-t": ["unit.wav", 0.1, 0.2, 0.3],
            "t-t": ["unit.wav", 0.2, 0.25, 0.3],
            "t-a": ["unit.wav", 0.2, 0.3, 0.4],
        })

        self.assertIn("cl", phones)
        self.assertIn("pau", phones)

    def test_generated_phoneset_preserves_articulatory_phone_classes(self):
        import build_festival_voice as builder

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scheme_path = builder.gen_scheme(
                "phone_classes",
                root,
                {"pau-pau": ["_silence.wav", 0.0, 0.15, 0.30]},
                ["pau", "aa", "r", "z", "m", "w", "t", "s"],
                165.0,
            )
            scheme = scheme_path.read_text(encoding="utf-8")

        self.assertIn("   (aa  +  l 1 1 - 0 0 +)", scheme)
        self.assertIn("   (r  -  0 - - - l a +)", scheme)
        self.assertIn("   (z  -  0 - - - f a +)", scheme)
        self.assertIn("   (m  -  0 - - - n a +)", scheme)
        self.assertIn("   (w  -  0 - - - r a +)", scheme)
        self.assertIn("   (t  -  0 - - - s a -)", scheme)
        self.assertIn("   (s  -  0 - - - f a -)", scheme)

    def test_structural_cl_is_not_declared_as_festival_silence(self):
        import build_festival_voice as builder

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scheme_path = builder.gen_scheme(
                "structural_cl",
                root,
                {"pau-pau": ["_silence.wav", 0.0, 0.15, 0.30]},
                ["pau", "cl", "t", "a"],
                165.0,
            )
            scheme = scheme_path.read_text(encoding="utf-8")

        self.assertNotIn("cl", builder.SILENCE_PHONES)
        self.assertIn("(PhoneSet.silences '(pau))", scheme)
        self.assertNotIn("(PhoneSet.silences '(pau cl", scheme)
        self.assertIn("   (cl  -  0 - - - s a -)", scheme)

    def test_literal_cl_mapping_requires_creator_opt_in_and_authored_edges(self):
        import build_festival_voice as builder

        default = builder._special_phone_policy([])
        literal = builder._special_phone_policy(
            [], ["cl_literal=cl"]
        )
        shorthand = builder._special_phone_policy(["cl=literal"])
        self.assertEqual(
            default["phones"]["cl"]["mode"],
            "anticipatory_consonant",
        )
        self.assertEqual(
            literal["phones"]["cl"]["mode"],
            "anticipatory_consonant",
        )
        self.assertEqual(
            literal["literal_phone_mappings"]["cl_literal"]
            ["source_phone"],
            "cl",
        )
        self.assertEqual(
            shorthand["phones"]["cl"]["mode"],
            "anticipatory_consonant",
        )
        self.assertEqual(
            shorthand["literal_phone_mappings"]["cl_literal"]
            ["source_phone"],
            "cl",
        )
        with self.assertRaisesRegex(
                VoicePathError, "incoming X-cl"):
            builder._validate_literal_special_phone_sources(
                literal,
                {"cl-a": ["authored.wav", 0.0, 0.1, 0.2]},
            )
        builder._validate_literal_special_phone_sources(
            literal,
            {
                "a-cl": ["authored.wav", 0.0, 0.1, 0.2],
                "cl-a": ["authored.wav", 0.0, 0.1, 0.2],
            },
        )
        with self.assertRaisesRegex(VoicePathError, "collides"):
            builder._validate_literal_special_phone_sources(
                builder._special_phone_policy(
                    [], ["a=cl"]
                ),
                {
                    "a-cl": ["authored.wav", 0.0, 0.1, 0.2],
                    "cl-a": ["authored.wav", 0.0, 0.1, 0.2],
                },
            )

    def test_generated_bank_exposes_literal_and_structural_cl_together(self):
        import special_phones

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "samples"
            bank.mkdir()
            _tone_wav(bank / "sample.wav", duration=1.2)
            oto = bank / "oto.ini"
            oto.write_text(
                "sample.wav=aa cl,0,220,-300,100,20\n"
                "sample.wav=cl aa,240,220,-300,100,20\n"
                "sample.wav=aa t,480,220,-300,100,20\n"
                "sample.wav=t aa,720,220,-300,100,20\n",
                encoding="utf-8",
            )
            output = root / "generated" / "literal_and_structural"
            args = _build_args(
                "en", "arpasing", bank, oto, output,
                "literal_and_structural",
            )
            args.extend([
                "--literal-phone-map", "cl_literal=cl",
            ])

            code, _stdout, _stderr = _run_builder(args)
            self.assertEqual(code, 0)
            payloads = [
                json.loads(
                    (output / "dic" / filename).read_text(encoding="utf-8")
                )
                for filename in (
                    "diphone_index.json",
                    "unit_alternatives.json",
                    "voice_manifest.json",
                )
            ]
            runtime = payloads[0]

        for payload in payloads:
            self.assertEqual(
                payload["special_phone_realizations"]
                ["phones"]["cl"]["mode"],
                "anticipatory_consonant",
            )
            self.assertEqual(
                payload["special_phone_realizations"]
                ["literal_phone_mappings"]["cl_literal"]["source_phone"],
                "cl",
            )
        self.assertIn("cl", runtime["phones"])
        self.assertIn("cl_literal", runtime["phones"])
        self.assertIn("aa-t", runtime["index"])
        self.assertIn("t-t", runtime["index"])
        structural = special_phones.resolve_special_phone_sequence(
            ["aa", "cl", "t", "aa"],
            metadata=runtime,
            available_diphones=runtime["index"],
        )
        literal = special_phones.resolve_special_phone_sequence(
            ["aa", "cl_literal", "aa"],
            metadata=runtime,
            available_diphones=runtime["index"],
        )
        self.assertEqual(
            structural.render_phones, ("aa", "t", "t", "aa")
        )
        self.assertEqual(literal.render_phones, ("aa", "cl", "aa"))

    def test_legacy_copy_refuses_in_place_input_database(self):
        import build_festival_voice as builder

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "separate"):
                builder.copy_wavs(
                    {"a-b": ["unit.wav", 0.0, 0.1, 0.2]}, root, root
                )

    def test_arpasing_build_uses_frq_world_pitchmarks_without_est_pda(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, english_oto, _ = _shared_bank(root)
            _frq(
                bank / "sample_wav.frq", 180.0,
                [180.0] * 64,
            )
            before = _tree_bytes(bank)
            outputs = []
            for folder in ("first", "second"):
                output = root / folder / "shared_pm"
                args = _build_args(
                    "en", "arpasing", bank, english_oto, output,
                    "shared_pm",
                )
                args.remove("--skip-pm")
                args.extend([
                    "--f0-estimator", "dio",
                    "--runtime-audio-storage", "separate",
                ])
                self.assertEqual(_run_builder(args)[0], 0)
                outputs.append(output)

            self.assertEqual(before, _tree_bytes(bank))
            self.assertEqual(_tree_bytes(outputs[0]), _tree_bytes(outputs[1]))
            manifest = json.loads(
                (outputs[0] / "pm" / "pitchmark_sources.json")
                .read_text(encoding="utf-8")
            )
            voiced = [
                row for name, row in manifest["units"].items()
                if name != "_silence.wav"
            ]
            self.assertTrue(voiced)
            self.assertTrue(all(
                row["f0_source"] == "utau-frq" for row in voiced
            ))
            self.assertTrue(all(
                (outputs[0] / "pm" / row["f0_file"]).is_file()
                for row in voiced
            ))
            metadata = json.loads(
                (outputs[0] / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["f0_fallback_estimator"], "dio")

    def test_arpasing_build_can_explicitly_enable_all_three_languages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, english_oto, _ = _shared_bank(root)
            output = root / "generated" / "integrated"
            second_output = root / "generated_again" / "integrated"
            args = _build_args(
                "en", "arpasing", bank, english_oto, output, "integrated"
            )
            args.extend([
                "--enable-language", "asaxi",
                "--enable-language", "ja",
            ])
            self.assertEqual(_run_builder(args)[0], 0)
            second_args = _build_args(
                "en", "arpasing", bank, english_oto, second_output,
                "integrated",
            )
            second_args.extend([
                "--enable-language", "asaxi",
                "--enable-language", "ja",
            ])
            self.assertEqual(_run_builder(second_args)[0], 0)
            self.assertEqual(_tree_bytes(output), _tree_bytes(second_output))
            metadata = json.loads(
                (output / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["supported_languages"], ["en", "asaxi", "ja"]
            )
            self.assertEqual(metadata["voice_entry_points"], {
                "asaxi": "voice_integrated",
                "en": "voice_integrated_en",
                "ja": "voice_integrated_ja",
            })
            self.assertEqual(
                metadata["japanese_phoneme_map"]["source_sha256"],
                "6356B50F3C25417797130F94F47E9D52C3B4A96B7DC5FFDB511A18358C517A99",
            )
            scheme = (output / "festvox" / "integrated.scm").read_text(
                encoding="utf-8"
            )
            self.assertIn("(define (voice_integrated)", scheme)
            self.assertIn("(define (voice_integrated_en)", scheme)
            self.assertIn("(define (voice_integrated_ja)", scheme)
            english_entry = scheme.split(
                "(define (voice_integrated_en)", 1
            )[1].split("(proclaim_voice", 1)[0]
            self.assertIn("(voice_kal_diphone)", english_entry)
            self.assertIn(
                "(PhoneSet.select 'integrated)", english_entry
            )
            self.assertLess(
                english_entry.index("(voice_kal_diphone)"),
                english_entry.index("(PhoneSet.select 'integrated)"),
            )
            self.assertIn("(Parameter.set 'Synth_Method 'UniSyn)", scheme)
            self.assertIn("festvox_gui_legacy_joins", scheme)
            self.assertIn(".legacy.pm", scheme)
            self.assertIn("integrated_active_db_name", scheme)
            self.assertIn("integrated_configure_join_windows", scheme)
            self.assertIn(
                '(Param.set "unisyn.window_name" "hanning")', scheme)
            self.assertIn(
                '(Param.set "unisyn.window_factor" 1.0)', scheme)
            self.assertNotIn(
                '(Param.set "unisyn.window_symmetric" 0)', scheme)
            self.assertIn(
                '(Param.set "unisyn.window_symmetric" 1)', scheme)
            self.assertTrue(
                metadata["source_window_policy"][
                    "normal_unisyn_window_symmetric"])
            self.assertTrue(
                metadata["source_window_policy"][
                    "legacy_unisyn_window_symmetric"])
            self.assertEqual(
                metadata["source_window_policy"]["half_window_ms"],
                60.0,
            )
            self.assertNotIn(str(bank), json.dumps(metadata))

    def test_english_generated_tree_is_destination_independent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, english_oto, _ = _shared_bank(root)
            first = root / "first" / "same_voice"
            second = root / "second" / "same_voice"
            self.assertEqual(_run_builder(_build_args(
                "en", "arpasing", bank, english_oto, first, "same_voice"
            ))[0], 0)
            self.assertEqual(_run_builder(_build_args(
                "en", "arpasing", bank, english_oto, second, "same_voice"
            ))[0], 0)
            self.assertEqual(_tree_bytes(first), _tree_bytes(second))

    def test_shared_samples_build_isolated_english_and_japanese_configs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, english_oto, japanese_oto = _shared_bank(root)
            english_out = root / "generated" / "shared_en"
            japanese_out = root / "generated" / "shared_ja"

            self.assertEqual(_run_builder(_build_args(
                "en", "arpasing", bank, english_oto, english_out,
                "shared_en", explicit_pitch=False,
            ))[0], 0)
            japanese_args = _build_args(
                "ja", "cv", bank, japanese_oto, japanese_out,
                "shared_ja", explicit_pitch=False,
            )
            japanese_args.extend(["--f0-estimator", "dio"])
            self.assertEqual(_run_builder(japanese_args)[0], 0)

            english = json.loads(
                (english_out / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8")
            )
            japanese = json.loads(
                (japanese_out / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8")
            )
            for output, runtime in (
                    (english_out, english), (japanese_out, japanese)):
                alternatives = json.loads(
                    (output / "dic" / "unit_alternatives.json")
                    .read_text(encoding="utf-8")
                )
                manifest = json.loads(
                    (output / "dic" / "voice_manifest.json")
                    .read_text(encoding="utf-8")
                )
                for payload in (runtime, alternatives, manifest):
                    self.assertEqual(
                        payload["special_phone_realizations"]
                        ["phones"]["cl"]["mode"],
                        "anticipatory_consonant",
                    )
            self.assertEqual(
                english["source_bundle_id"], japanese["source_bundle_id"]
            )
            self.assertNotEqual(
                english["configuration_id"], japanese["configuration_id"]
            )
            self.assertEqual(english["supported_languages"], ["en"])
            self.assertEqual(japanese["supported_languages"], ["ja"])
            self.assertIn("en", english["voice_entry_points"])
            self.assertNotIn("ja", english["voice_entry_points"])
            self.assertIn("ja", japanese["voice_entry_points"])
            self.assertNotIn("en", japanese["voice_entry_points"])
            self.assertNotEqual(
                english["alias_namespace"], japanese["alias_namespace"]
            )
            self.assertEqual(
                english["speaker_pitch_analysis"],
                japanese["speaker_pitch_analysis"],
            )
            self.assertEqual(
                english["speaker_pitch_analysis"]["source"],
                "waveform_estimation",
            )
            self.assertEqual(
                english["speaker_pitch_analysis"]["files_used"],
                ["sample.wav"],
            )
            self.assertEqual(
                english["source_recording_bundle"]["speaker_pitch_analysis"],
                japanese["source_recording_bundle"]["speaker_pitch_analysis"],
            )
            self.assertAlmostEqual(
                english["average_pitch_hz"],
                english["speaker_pitch_analysis"]["median_f0_hz"],
                places=5,
            )
            self.assertEqual(
                english["default_pitch_source"], "speaker_median")
            self.assertEqual(
                english["automatic_pitch_headroom_semitones"], 0.0)
            self.assertEqual(
                english["automatic_pitch_floor_hz"],
                english["speaker_pitch_analysis"]["median_f0_hz"],
            )
            self.assertEqual(
                japanese["average_pitch_hz"], english["average_pitch_hz"]
            )
            self.assertEqual(
                japanese["automatic_pitch_floor_hz"],
                japanese["speaker_pitch_analysis"]["median_f0_hz"],
            )
            self.assertEqual(japanese["f0_fallback_estimator"], "dio")
            self.assertEqual(english["f0_min_hz"], japanese["f0_min_hz"])
            self.assertEqual(english["f0_max_hz"], japanese["f0_max_hz"])
            self.assertEqual(
                english["source_recording_bundle"]["oto_files"][0]["path"],
                "english/oto.ini",
            )
            self.assertEqual(
                japanese["voice_configuration"]["bank_type"], "cv"
            )
            english_scheme = (
                english_out / "festvox" / "shared_en.scm"
            ).read_text(encoding="utf-8")
            self.assertIn("(car load-path)", english_scheme)
            self.assertEqual(english_scheme.count("(define (voice_"), 1)
            self.assertNotIn("voice_shared_en_en", english_scheme)
            manifest_text = (
                english_out / "dic" / "voice_manifest.json"
            ).read_text(encoding="utf-8")
            self.assertNotIn(str(bank), manifest_text)

    def test_multiple_explicit_oto_scopes_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, _english_oto, japanese_oto = _shared_bank(root)
            second = bank / "japanese_2" / "oto.ini"
            second.parent.mkdir()
            second.write_text(
                "../sample.wav=\u3044,0,300,-700,120,20\n",
                encoding="utf-8",
            )
            ignored = bank / "ignored" / "oto.ini"
            ignored.parent.mkdir()
            ignored.write_text(
                "../sample.wav=\u3046,0,300,-700,120,20\n",
                encoding="utf-8",
            )
            output = root / "generated" / "scoped_ja"
            args = _build_args(
                "ja", "cv", bank, japanese_oto, output, "scoped_ja"
            )
            args.extend(["--oto", str(second)])

            code, _stdout, stderr = _run_builder(args)
            self.assertEqual(code, 2)
            self.assertIn("More than one --oto scope", stderr)

    def test_legacy_builder_refuses_automatic_multi_folder_discovery(self):
        import build_festival_voice as builder

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, _english_oto, _japanese_oto = _shared_bank(root)
            output = root / "generated" / "legacy"
            stderr = StringIO()

            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    builder.main([
                        "--utau", str(bank),
                        "--out", str(output),
                        "--skip-pm",
                    ])

            self.assertEqual(caught.exception.code, 2)
            self.assertIn("Multiple OTO folders", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_conversion_api_refuses_multiple_explicit_oto_scopes(self):
        import build_festival_voice as builder

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, english_oto, japanese_oto = _shared_bank(root)
            output = root / "generated" / "conversion"

            with self.assertRaisesRegex(
                VoicePathError, "More than one --oto scope"
            ):
                builder.run_utau_conversion(
                    bank,
                    output,
                    "conversion",
                    oto_files=(english_oto, japanese_oto),
                )

            self.assertFalse((output / "db").exists())

    def test_one_pitch_folder_may_contain_split_oto_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, _english_oto, japanese_oto = _shared_bank(root)
            pitch = japanese_oto.parent
            second = pitch / "split" / "oto.ini"
            second.parent.mkdir()
            second.write_text(
                "../../sample.wav=\u3044E3,0,300,-700,120,20\n",
                encoding="utf-8",
            )
            output = root / "generated" / "scoped_ja"
            args = _build_args(
                "ja", "cv", bank, pitch, output, "scoped_ja"
            )

            self.assertEqual(_run_builder(args)[0], 0)
            metadata = json.loads(
                (output / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8")
            )
            selected = {
                row["path"] for row in
                metadata["source_recording_bundle"]["oto_files"]
            }
            self.assertEqual(selected, {
                "japanese/oto.ini", "japanese/split/oto.ini",
            })

    def test_arpasing_pitch_folder_may_contain_split_oto_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, english_oto, _japanese_oto = _shared_bank(root)
            pitch = english_oto.parent
            second = pitch / "split" / "oto.ini"
            second.parent.mkdir()
            second.write_text(
                "../../sample.wav=ae,0,300,-700,120,20\n",
                encoding="utf-8",
            )
            output = root / "generated" / "scoped_en"
            args = _build_args(
                "en", "arpasing", bank, pitch, output, "scoped_en"
            )

            self.assertEqual(_run_builder(args)[0], 0)
            metadata = json.loads(
                (output / "dic" / "diphone_index.json")
                .read_text(encoding="utf-8")
            )
            selected = {
                row["path"] for row in
                metadata["source_recording_bundle"]["oto_files"]
            }
            self.assertEqual(selected, {
                "english/oto.ini", "english/split/oto.ini",
            })

    def test_one_folder_with_multiple_detected_pitches_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, _english_oto, japanese_oto = _shared_bank(root)
            pitch = japanese_oto.parent
            japanese_oto.write_text(
                "../sample.wav=\u3042E3,0,300,-700,120,20\n",
                encoding="utf-8",
            )
            second = pitch / "split" / "oto.ini"
            second.parent.mkdir()
            second.write_text(
                "../../sample.wav=\u3044F3,0,300,-700,120,20\n",
                encoding="utf-8",
            )
            output = root / "generated" / "mixed_pitch"
            args = _build_args(
                "ja", "cv", bank, pitch, output, "mixed_pitch"
            )

            code, _stdout, stderr = _run_builder(args)
            self.assertEqual(code, 2)
            self.assertIn("multiple pitch tags", stderr)

    def test_one_oto_file_with_multiple_detected_pitches_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, _english_oto, japanese_oto = _shared_bank(root)
            japanese_oto.write_text(
                "../sample.wav=\u3042E3,0,300,-700,120,20\n"
                "../sample.wav=\u3044F3,0,300,-700,120,20\n",
                encoding="utf-8",
            )
            output = root / "generated" / "mixed_pitch_file"
            args = _build_args(
                "ja", "cv", bank, japanese_oto, output, "mixed_pitch_file"
            )

            code, _stdout, stderr = _run_builder(args)
            self.assertEqual(code, 2)
            self.assertIn("multiple pitch tags", stderr)

    def test_numbered_alias_takes_do_not_look_like_extra_pitches(self):
        import build_festival_voice as builder

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "P3_E3"
            bank.mkdir()
            oto = bank / "oto.ini"
            oto.write_text(
                "sample.wav=a11E3,0,300,-700,120,20\n"
                "sample.wav=b1PE3,0,300,-700,120,20\n"
                "sample.wav=aliasE1PE3,0,300,-700,120,20\n"
                "sample_A1_E3.wav=kaE3P,0,300,-700,120,20\n"
                "sample.wav=a3,0,300,-700,120,20\n",
                encoding="utf-8",
            )

            selected = builder._selected_oto_files(bank, (oto,))
            builder._validate_single_pitch_oto_scope(
                bank, (oto,), selected
            )

    def test_source_wav_names_cannot_hide_multiple_pitches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, _english_oto, japanese_oto = _shared_bank(root)
            japanese_oto.write_text(
                "../sample_E3.wav=あ,0,300,-700,120,20\n"
                "../sample_F3.wav=い,0,300,-700,120,20\n",
                encoding="utf-8",
            )
            output = root / "generated" / "mixed_source_names"
            args = _build_args(
                "ja", "cv", bank, japanese_oto, output,
                "mixed_source_names",
            )

            code, _stdout, stderr = _run_builder(args)
            self.assertEqual(code, 2)
            self.assertIn("multiple pitch tags", stderr)

    def test_source_pitch_cannot_be_hidden_by_alias_pitch(self):
        import build_festival_voice as builder

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "P3_E3"
            bank.mkdir()
            oto = bank / "oto.ini"
            oto.write_text(
                "sample_F3.wav=kaE3,0,300,-700,120,20\n",
                encoding="utf-8",
            )

            selected = builder._selected_oto_files(bank, (oto,))
            with self.assertRaisesRegex(
                VoicePathError, "multiple pitch tags"
            ):
                builder._validate_single_pitch_oto_scope(
                    bank, (oto,), selected
                )

    def test_output_protection_requires_explicit_non_destructive_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, english_oto, _ = _shared_bank(root)
            output = root / "generated" / "voice"
            output.mkdir(parents=True)
            marker = output / "keep.txt"
            marker.write_text("user data", encoding="utf-8")
            args = _build_args(
                "en", "arpasing", bank, english_oto, output, "protected"
            )
            self.assertEqual(_run_builder(args)[0], 2)
            self.assertEqual(marker.read_text(encoding="utf-8"), "user data")
            self.assertEqual(_run_builder(args + ["--overwrite"])[0], 0)
            self.assertEqual(marker.read_text(encoding="utf-8"), "user data")

    def test_windows_wsl_path_and_registration_migration(self):
        self.assertEqual(
            windows_to_wsl_path(r"D:\Generated Voices\lem_ja"),
            "/mnt/d/Generated Voices/lem_ja",
        )
        self.assertEqual(
            wsl_to_windows_path("/mnt/d/Generated Voices/lem_ja"),
            r"D:\Generated Voices\lem_ja",
        )
        self.assertEqual(wsl_to_windows_path("/mnt/d"), "D:\\")
        migrated = migrate_voice_registration({
            "dir": "/mnt/d/Generated Voices/lem_ja",
            "voice": "voice_lem_ja",
            "scm": "festvox/lem_ja.scm",
        })
        self.assertEqual(migrated["path_status"], "current")
        self.assertEqual(
            migrated["runtime_path"], "/mnt/d/Generated Voices/lem_ja"
        )
        legacy = migrate_voice_registration({
            "dir": "/home/user/voices/old", "voice": "voice_old"
        })
        self.assertEqual(legacy["path_status"], "legacy_wsl_only")
        self.assertEqual(legacy["windows_path"], "")

    def test_gui_registration_stores_canonical_and_derived_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, english_oto, _ = _shared_bank(root)
            output = root / "generated" / "voice"
            self.assertEqual(_run_builder(_build_args(
                "en", "arpasing", bank, english_oto, output, "registered"
            ))[0], 0)
            config = json.loads(json.dumps(fc.DEFAULT_CONFIG))
            config["festival_wsl"]["generated_voice_root"] = str(
                root / "generated"
            )
            backend = fc.FestivalWSLBackend(config)
            info = backend.scan_voice_dir(str(output))
            name = backend.add_voice(info)
            stored = backend.fcfg()["voices"][name]
            self.assertEqual(stored["windows_path"], str(output.resolve()))
            self.assertEqual(
                stored["runtime_path"], windows_to_wsl_path(output.resolve())
            )
            self.assertEqual(stored["language"], "en")
            self.assertTrue(stored["source_bundle_id"].startswith("srb_"))
            self.assertTrue(stored["configuration_id"].startswith("vcfg_"))
            self.assertEqual(stored["alias_system"],
                             "utau-english-arpasing-v1")

    def test_source_output_nesting_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            with self.assertRaises(VoicePathError):
                validate_build_layout(source, source / "generated")


if __name__ == "__main__":
    unittest.main()
