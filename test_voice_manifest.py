import json
from pathlib import Path
import sys
import tempfile
import unittest


FESTVOX_DIR = Path(__file__).resolve().parent
GUI_DIR = FESTVOX_DIR / "festvox_gui"
for folder in (FESTVOX_DIR, GUI_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from festvox_gui import festvox_core as fc
import japanese_candidates as jc
import japanese_festival as jf
from japanese_profiles import infer_bank_profile
from voice_manifest import (
    DEFAULT_GENERATED_VOICE_OUTPUT_CALIBRATION,
    SourceRecordingBundle,
    VoiceConfiguration,
    generated_voice_output_calibration,
    read_voice_compatibility,
)


class VoiceManifestTests(unittest.TestCase):
    @staticmethod
    def _bank(root):
        bank = root / "private-bank"
        bank.mkdir()
        (bank / "sample.wav").write_bytes(b"RIFF")
        (bank / "oto.ini").write_text(
            "sample.wav=か,0,100,-200,50,10\n",
            encoding="utf-8",
        )
        return bank

    def test_source_bundle_and_configurations_are_stable_and_separate(self):
        manifest = {
            "source_scope": "P3_E3",
            "fingerprint_sha256": "a" * 64,
            "oto_files": [{"path": "P3_E3/oto.ini", "sha256": "b" * 64}],
            "metadata_files": {},
        }
        bundle = SourceRecordingBundle.from_source_manifest(manifest)
        cv = VoiceConfiguration.japanese(
            source_bundle_id=bundle.source_bundle_id,
            bank_type="cv",
            configuration_policy={"source_scope": "P3_E3"},
        )
        vcv = VoiceConfiguration.japanese(
            source_bundle_id=bundle.source_bundle_id,
            bank_type="vcv",
            configuration_policy={"source_scope": "P3_E3"},
        )

        self.assertEqual(bundle.source_bundle_id, "srb_" + "a" * 24)
        self.assertNotEqual(cv.configuration_id, vcv.configuration_id)
        self.assertNotEqual(cv.alias_namespace, vcv.alias_namespace)
        self.assertIn(cv.configuration_id, cv.canonical_phone_namespace)

    def test_candidate_identity_is_configuration_scoped_and_private(self):
        with tempfile.TemporaryDirectory() as temp:
            bank = self._bank(Path(temp))
            cv = infer_bank_profile(bank, bank_configuration="cv")
            vcv = infer_bank_profile(bank, bank_configuration="vcv")
            cv_graph = jc.compile_candidate_graph(bank, profile=cv)
            vcv_graph = jc.compile_candidate_graph(bank, profile=vcv)

            cv_candidate = cv_graph.candidates[0]
            vcv_candidate = vcv_graph.candidates[0]
            self.assertNotEqual(
                cv_candidate.candidate_id, vcv_candidate.candidate_id
            )
            self.assertEqual(
                cv_candidate.configuration_id,
                cv_graph.voice_configuration.configuration_id,
            )
            self.assertIn(
                cv_candidate.canonical_phone_namespace,
                cv_candidate.to_dict()["scoped_target_key"],
            )
            serialized = jc.candidate_metadata_bytes(cv_graph).decode("utf-8")
            self.assertNotIn(str(bank), serialized)

    def test_recording_byte_change_creates_a_new_source_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            bank = self._bank(Path(temp))
            profile = infer_bank_profile(bank, bank_configuration="cv")
            first = jc.compile_candidate_graph(bank, profile=profile)
            (bank / "sample.wav").write_bytes(b"RIFF-changed")
            second = jc.compile_candidate_graph(bank, profile=profile)

            self.assertNotEqual(
                first.source_bundle.source_bundle_id,
                second.source_bundle.source_bundle_id,
            )
            self.assertNotEqual(
                first.candidates[0].candidate_id,
                second.candidates[0].candidate_id,
            )

    def test_festival_build_refuses_analysis_only_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = self._bank(root)
            graph = jc.compile_candidate_graph(bank)
            with self.assertRaisesRegex(ValueError, "explicit bank type"):
                jf.compile_festival_voice(
                    graph, root / "voice", pitchmark=False
                )
            self.assertFalse((root / "voice").exists())

    def test_current_and_legacy_compatibility_are_inspectable(self):
        current = read_voice_compatibility({
            "source_bundle_id": "srb_a",
            "configuration_id": "vcfg_a",
            "primary_language": "ja",
            "supported_languages": ["ja"],
            "alias_system": "utau-japanese-vcv-v1",
            "voice_entry_points": {"ja": "voice_example_ja"},
            "phones": ["pau", "a", "k"],
        })
        legacy = read_voice_compatibility({
            "language": "ja",
            "voice_entry_point": "voice_old_ja",
        })
        unknown = read_voice_compatibility({})

        self.assertTrue(current.is_current)
        self.assertTrue(current.supports("ja"))
        self.assertFalse(current.supports("en"))
        self.assertTrue(legacy.is_legacy)
        self.assertIn("rebuild", legacy.reason.casefold())
        self.assertEqual(unknown.metadata_status, "unknown")

    def test_generated_voice_calibration_is_explicit_or_safe_legacy_default(self):
        legacy_generated = {
            "kind": "festival_unisyn_runtime_index",
            "voice_manifest_schema_version": 1,
            "source_bundle_id": "srb_a",
            "configuration_id": "vcfg_a",
            "builder_version": "unified-festival-builder-v1",
        }
        policy = generated_voice_output_calibration(legacy_generated)
        self.assertEqual(policy["method"], "active_speech_rms")
        self.assertEqual(policy["maximum_gain_db"], 12.0)
        self.assertEqual(
            policy["policy_source"], "legacy_generated_voice_default")
        self.assertEqual(
            DEFAULT_GENERATED_VOICE_OUTPUT_CALIBRATION[
                "maximum_gain_db"], 12.0)

        explicit = generated_voice_output_calibration({
            **legacy_generated,
            "output_calibration": {
                "method": "active_speech_rms",
                "target_dbfs": -23.0,
                "maximum_gain_db": 4.0,
            },
        })
        self.assertEqual(explicit["target_dbfs"], -23.0)
        self.assertEqual(explicit["maximum_gain_db"], 4.0)
        self.assertEqual(explicit["policy_source"], "voice_metadata")

        self.assertEqual(generated_voice_output_calibration({}), {})
        self.assertEqual(generated_voice_output_calibration({
            "kind": "festival_unisyn_runtime_index",
        }), {})
        self.assertEqual(generated_voice_output_calibration({
            **legacy_generated,
            "output_calibration": {},
        }), {})

    def test_backend_uses_manifest_entry_point_and_labels_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "voice"
            (root / "dic").mkdir(parents=True)
            metadata = {
                "source_bundle_id": "srb_a",
                "configuration_id": "vcfg_a",
                "primary_language": "ja",
                "supported_languages": ["ja"],
                "alias_system": "utau-japanese-vcv-v1",
                "voice_entry_points": {"ja": "voice_manifest_ja"},
                "phones": ["pau", "a", "k"],
            }
            (root / "dic" / "diphone_index.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            config = {"festival_wsl": {"voices": {
                "fixture": {
                    "dir": str(root),
                    "voice": "voice_wrong_default",
                    "scm": "festvox/fixture.scm",
                }
            }}}
            backend = fc.FestivalWSLBackend(config)

            compatibility = backend.voice_compatibility("fixture")
            preamble = backend._voice_preamble("fixture", "ja")
            self.assertTrue(compatibility.is_current)
            self.assertIn("(voice_manifest_ja)", preamble)
            with self.assertRaisesRegex(fc.BackendError, "does not support"):
                backend._voice_preamble("fixture", "en")


if __name__ == "__main__":
    unittest.main()
