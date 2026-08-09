from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import japanese_release as release


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(release.__file__).resolve().parent
LICENSE_INVENTORY = (
    PROJECT_ROOT / "docs" / "JAPANESE_DEPENDENCIES_AND_LICENSES.md"
)


def _dependencies(required_installed=True):
    return (
        release.DependencyCheck(
            distribution="required",
            required=True,
            purpose="fixture",
            origin="https://example.invalid/required",
            installed=required_installed,
            version="1" if required_installed else None,
            declared_license="MIT" if required_installed else None,
        ),
        release.DependencyCheck(
            distribution="optional",
            required=False,
            purpose="fixture",
            origin="https://example.invalid/optional",
            installed=False,
        ),
    )


class JapaneseReleaseTests(unittest.TestCase):
    def test_license_inventory_contains_every_required_component(self):
        complete, missing = release.check_license_inventory(LICENSE_INVENTORY)

        self.assertTrue(complete)
        self.assertEqual(missing, ())

    def test_repository_bundles_no_dictionary_or_hts_voice(self):
        self.assertEqual(
            release.scan_prohibited_bundled_assets(PROJECT_ROOT),
            (),
        )

    def test_asset_scanner_reports_relative_paths_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dictionary = root / "assets" / "open_jtalk_dic_utf_8-1.11"
            dictionary.mkdir(parents=True)
            voice = root / "assets" / "mei.htsvoice"
            voice.write_bytes(b"fixture")

            result = release.scan_prohibited_bundled_assets(root)

        self.assertEqual(result, (
            "assets/mei.htsvoice",
            "assets/open_jtalk_dic_utf_8-1.11/",
        ))
        self.assertNotIn(str(root), repr(result))

    def test_release_report_is_deterministic_and_not_false_green(self):
        with tempfile.TemporaryDirectory() as directory:
            first = release.run_release_checks(
                directory, LICENSE_INVENTORY,
                dependencies=_dependencies())
            second = release.run_release_checks(
                directory, LICENSE_INVENTORY,
                dependencies=_dependencies())

        self.assertEqual(first.to_json_bytes(), second.to_json_bytes())
        self.assertTrue(first.implementation_checks_passed)
        self.assertFalse(first.redistribution_ready)
        self.assertIn("project_license_not_declared",
                      [item.code for item in first.diagnostics])
        self.assertIn("optional_dependency_missing",
                      [item.code for item in first.diagnostics])

    def test_missing_required_dependency_fails_implementation_check(self):
        with tempfile.TemporaryDirectory() as directory:
            report = release.run_release_checks(
                directory, LICENSE_INVENTORY,
                dependencies=_dependencies(required_installed=False))

        self.assertFalse(report.implementation_checks_passed)
        self.assertIn("required_dependency_missing",
                      [item.code for item in report.diagnostics])

    def test_optional_requirement_is_pinned_and_not_in_gui_requirements(self):
        optional = (
            PROJECT_ROOT / "requirements-japanese-optional.txt"
        ).read_text(encoding="utf-8")
        gui = (
            SOURCE_ROOT / "festvox_gui" / "requirements.txt"
        ).read_text(encoding="utf-8")

        self.assertIn("pyopenjtalk==0.4.1", optional)
        self.assertNotIn("pyopenjtalk", gui)


if __name__ == "__main__":
    unittest.main()
