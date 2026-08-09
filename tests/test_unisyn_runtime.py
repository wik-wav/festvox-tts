import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from unisyn_runtime import (
    apply_runtime_audio_metadata,
    build_grouped_runtime,
    grouped_audio_relative_path,
)


class UniSynRuntimeTests(unittest.TestCase):
    def _voice(self, root: Path):
        (root / "festvox").mkdir(parents=True)
        scheme = root / "festvox" / "demo.scm"
        scheme.write_text("(define (voice_demo) t)\n", encoding="utf-8")
        return scheme

    @staticmethod
    def _successful_runner(expected_bytes=b"deterministic-group-data" * 8):
        def run(args, **_kwargs):
            script = Path(args[-1])
            text = script.read_text(encoding="utf-8")
            root = script.parent
            staging = root / "group" / ".demo_diphone.group.building"
            staging.write_bytes(expected_bytes)
            assert "festvox_gui_force_separate_database t" in text
            assert "sig_sample_format short" in text
            return SimpleNamespace(
                returncode=0,
                stdout="FESTVOX-GROUP-RUNTIME-OK\n",
                stderr="",
            )
        return run

    def test_group_build_is_atomic_deterministic_and_forces_separate_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scheme = self._voice(root)
            first = build_grouped_runtime(
                root,
                voice_name="demo",
                scheme_path=scheme,
                voice_entry_point="voice_demo",
                festival_bin="festival",
                run_external=self._successful_runner(),
            )
            target = root / grouped_audio_relative_path("demo")
            original = target.read_bytes()
            second = build_grouped_runtime(
                root,
                voice_name="demo",
                scheme_path=scheme,
                voice_entry_point="voice_demo",
                festival_bin="festival",
                run_external=self._successful_runner(),
            )

            self.assertEqual(first, second)
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(first["effective"], "grouped")
            self.assertEqual(first["signal_sample_format"], "short")
            self.assertTrue(first["contextual_unit_selection_preserved"])
            self.assertFalse((root / ".build_grouped_runtime.scm").exists())

    def test_failed_group_rebuild_preserves_previous_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scheme = self._voice(root)
            target = root / grouped_audio_relative_path("demo")
            target.parent.mkdir(parents=True)
            target.write_bytes(b"working-cache")

            def fail(args, **_kwargs):
                script = Path(args[-1])
                staging = (
                    script.parent / "group" /
                    ".demo_diphone.group.building"
                )
                staging.write_bytes(b"incomplete" * 20)
                return SimpleNamespace(
                    returncode=2, stdout="", stderr="fixture failure")

            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                build_grouped_runtime(
                    root,
                    voice_name="demo",
                    scheme_path=scheme,
                    voice_entry_point="voice_demo",
                    festival_bin="festival",
                    run_external=fail,
                )
            self.assertEqual(target.read_bytes(), b"working-cache")
            self.assertFalse(
                (target.parent / ".demo_diphone.group.building").exists())

    def test_runtime_metadata_is_applied_to_existing_views(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dic = root / "dic"
            dic.mkdir()
            for name in (
                "diphone_index.json",
                "unit_alternatives.json",
                "voice_manifest.json",
                "japanese_build_report.json",
            ):
                payload = {"kind": name}
                if name == "japanese_build_report.json":
                    payload["output_relative_files"] = ["wav/unit.wav"]
                (dic / name).write_text(
                    json.dumps(payload), encoding="utf-8")
            storage = {
                "effective": "grouped",
                "group_file": "group/demo_diphone.group",
            }

            apply_runtime_audio_metadata(root, storage)

            for path in dic.glob("*.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["runtime_audio_storage"], storage)
            report = json.loads(
                (dic / "japanese_build_report.json").read_text(
                    encoding="utf-8"))
            self.assertIn(
                "group/demo_diphone.group",
                report["output_relative_files"],
            )

    def test_rejects_unsafe_voice_symbols(self):
        with self.assertRaises(ValueError):
            grouped_audio_relative_path("../source-bank")


if __name__ == "__main__":
    unittest.main()
