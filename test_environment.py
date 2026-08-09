import unittest
from unittest import mock

import check_environment as env


class EnvironmentCheckTests(unittest.TestCase):
    def test_required_runtime_files_are_local_to_festvox(self):
        report = env.inspect_environment(check_wsl=False)

        self.assertEqual(report["missing_required_files"], [])
        self.assertTrue((env.ROOT / "synth_diphone.py").is_file())
        self.assertNotIn("vocab_forge", " ".join(env.REQUIRED_FILES))

    def test_ready_state_distinguishes_required_from_optional_modules(self):
        with mock.patch.object(
                env.importlib.util, "find_spec", return_value=object()):
            report = env.inspect_environment(check_wsl=False)

        self.assertTrue(report["ready"])
        self.assertEqual(report["missing_required_modules"], [])
        self.assertIsNone(report["festival_wsl"]["available"])

    def test_wsl_utf16_diagnostic_decoding(self):
        message = "Windows Subsystem for Linux has no distributions."

        decoded = env.decode_process_output(message.encode("utf-16-le"))

        self.assertEqual(decoded, message)
        self.assertNotIn("\x00", decoded)


if __name__ == "__main__":
    unittest.main()
