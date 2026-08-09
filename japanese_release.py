"""Deterministic packaging and dependency checks for Japanese support."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence


RELEASE_CHECK_SCHEMA_VERSION = 1
LICENSE_INVENTORY_TOKENS = (
    "pyopenjtalk",
    "Open JTalk",
    "open_jtalk_dic_utf_8",
    "HTS Voice Mei",
    "Festival",
    "Speech Tools",
    "PyQt5",
    "NumPy",
    "pyqtgraph",
    "sounddevice",
    "UTAU voicebanks",
    "not bundled",
)


DEPENDENCIES = (
    ("numpy", True, "GUI/DSP arrays", "https://numpy.org/"),
    ("PyQt5", True, "Windows GUI", "https://www.riverbankcomputing.com/software/pyqt/"),
    ("pyqtgraph", True, "waveform and parameter views", "https://pyqtgraph.readthedocs.io/"),
    ("sounddevice", False, "optional low-latency playback", "https://python-sounddevice.readthedocs.io/"),
    ("pyopenjtalk", False, "optional Japanese linguistic labels", "https://github.com/r9y9/pyopenjtalk"),
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n").encode("utf-8")


@dataclass(frozen=True)
class DependencyCheck:
    distribution: str
    required: bool
    purpose: str
    origin: str
    installed: bool
    version: Optional[str] = None
    declared_license: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "distribution": self.distribution,
            "required": self.required,
            "purpose": self.purpose,
            "origin": self.origin,
            "installed": self.installed,
        }
        if self.version is not None:
            result["version"] = self.version
        if self.declared_license is not None:
            result["declared_license"] = self.declared_license
        return result


@dataclass(frozen=True)
class PackagingDiagnostic:
    code: str
    message: str
    severity: str = "warning"
    relative_path: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.relative_path is not None:
            result["relative_path"] = self.relative_path
        return result


@dataclass(frozen=True)
class JapaneseReleaseCheckReport:
    dependencies: tuple[DependencyCheck, ...]
    diagnostics: tuple[PackagingDiagnostic, ...]
    license_inventory_complete: bool
    prohibited_bundled_assets: tuple[str, ...]
    implementation_checks_passed: bool
    redistribution_ready: bool = False
    schema_version: int = RELEASE_CHECK_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "japanese_release_check",
            "dependencies": [item.to_dict() for item in self.dependencies],
            "license_inventory_complete": self.license_inventory_complete,
            "prohibited_bundled_assets": list(self.prohibited_bundled_assets),
            "implementation_checks_passed": self.implementation_checks_passed,
            "redistribution_ready": self.redistribution_ready,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    def to_json_bytes(self) -> bytes:
        return _json_bytes(self.to_dict())


def inspect_python_dependencies() -> tuple[DependencyCheck, ...]:
    result = []
    for name, required, purpose, origin in DEPENDENCIES:
        try:
            distribution = importlib.metadata.distribution(name)
            metadata = distribution.metadata
            license_value = metadata.get("License-Expression") or \
                metadata.get("License") or None
            if license_value and len(license_value) > 240:
                license_value = "See installed distribution license files"
            result.append(DependencyCheck(
                distribution=name,
                required=required,
                purpose=purpose,
                origin=origin,
                installed=True,
                version=distribution.version,
                declared_license=license_value,
            ))
        except importlib.metadata.PackageNotFoundError:
            result.append(DependencyCheck(
                distribution=name,
                required=required,
                purpose=purpose,
                origin=origin,
                installed=False,
            ))
    return tuple(result)


def scan_prohibited_bundled_assets(project_root: Path | str) -> tuple[str, ...]:
    """Find model/dictionary assets that this repository must not bundle."""
    root = Path(project_root).resolve()
    matches = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        lowered = path.name.casefold()
        if path.is_dir() and lowered.startswith("open_jtalk_dic_"):
            matches.add(relative + "/")
        elif path.is_file() and path.suffix.casefold() == ".htsvoice":
            matches.add(relative)
    return tuple(sorted(matches))


def check_license_inventory(path: Path | str) -> tuple[bool, tuple[str, ...]]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return False, LICENSE_INVENTORY_TOKENS
    missing = tuple(token for token in LICENSE_INVENTORY_TOKENS
                    if token not in text)
    return not missing, missing


def run_release_checks(
    project_root: Path | str,
    license_inventory: Path | str,
    *,
    dependencies: Sequence[DependencyCheck] | None = None,
) -> JapaneseReleaseCheckReport:
    dependency_rows = tuple(
        dependencies if dependencies is not None
        else inspect_python_dependencies()
    )
    bundled = scan_prohibited_bundled_assets(project_root)
    inventory_ok, missing_tokens = check_license_inventory(license_inventory)
    diagnostics = []
    for item in dependency_rows:
        if item.required and not item.installed:
            diagnostics.append(PackagingDiagnostic(
                code="required_dependency_missing",
                message=f"Required distribution {item.distribution} is missing.",
                severity="error",
            ))
        elif not item.required and not item.installed:
            diagnostics.append(PackagingDiagnostic(
                code="optional_dependency_missing",
                message=(
                    f"Optional distribution {item.distribution} is absent; "
                    "its feature must degrade gracefully."
                ),
                severity="info",
            ))
    for relative in bundled:
        diagnostics.append(PackagingDiagnostic(
            code="prohibited_asset_bundled",
            message=(
                "Open JTalk dictionary/model assets must not be bundled until "
                "their exact release notices are packaged and reviewed."
            ),
            severity="error",
            relative_path=relative,
        ))
    if missing_tokens:
        diagnostics.append(PackagingDiagnostic(
            code="license_inventory_incomplete",
            message="License inventory is missing: " + ", ".join(missing_tokens),
            severity="error",
        ))
    diagnostics.extend((
        PackagingDiagnostic(
            code="project_license_not_declared",
            message=(
                "This local vault has no declared project license. Choose one "
                "before distributing the application."
            ),
        ),
        PackagingDiagnostic(
            code="utau_bank_licenses_user_supplied",
            message=(
                "Every UTAU bank has independent terms; no generated voice is "
                "redistributable until its source-bank license is recorded."
            ),
        ),
        PackagingDiagnostic(
            code="pyqt_distribution_choice_required",
            message=(
                "PyQt5 is GPLv3/commercial. Confirm the application's release "
                "license or obtain the appropriate commercial terms."
            ),
        ),
    ))
    passed = not any(item.severity == "error" for item in diagnostics)
    return JapaneseReleaseCheckReport(
        dependencies=dependency_rows,
        diagnostics=tuple(diagnostics),
        license_inventory_complete=inventory_ok,
        prohibited_bundled_assets=bundled,
        implementation_checks_passed=passed,
        redistribution_ready=False,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Japanese support dependencies and release hygiene."
    )
    parser.add_argument("project_root", type=Path)
    parser.add_argument("license_inventory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_release_checks(args.project_root, args.license_inventory)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(report.to_json_bytes())
    print(report.to_json_bytes().decode("utf-8"), end="")
    return 0 if report.implementation_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
