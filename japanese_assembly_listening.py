"""Render the ignored Stage 2 CV/VCV/CVVC assembly comparison corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

from japanese_assembly import create_source_contribution_plan
from japanese_festival import load_japanese_runtime_metadata
from japanese_frontend import analyze_japanese
from japanese_quality import analyze_plan_joins
from japanese_synthesis import create_synthesis_plan


ASSEMBLY_LISTENING_SCHEMA_VERSION = 1
ASSEMBLY_LISTENING_FIXTURES = (
    ("vowels", "vowels", "\u3042\u3044\u3046\u3048\u304a", "kana"),
    ("ordinary_cv", "ordinary CV", "\u304b\u304d\u304f\u3051\u3053", "kana"),
    ("vcv_transition", "VCV transition", "\u3042\u304b", "kana"),
    ("cvvc_transition", "CVVC transition", "\u3055\u304b", "kana"),
    ("moraic_nasal", "moraic nasal", "\u3042\u3093", "kana"),
    ("nasal_labial", "nasal allophone: labial", "\u3042\u3093\u3070", "kana"),
    ("nasal_velar", "nasal allophone: velar", "\u3042\u3093\u304b", "kana"),
    ("nasal_coronal", "nasal allophone: coronal", "\u3042\u3093\u305f", "kana"),
    ("nasal_uvular", "nasal allophone: configured uvular", "\u30d5\u30a7\u30f3\u30b9", "kana"),
    ("geminate", "geminate", "\u3042\u3063\u305f", "kana"),
    ("palatalized", "palatalized", "\u304d\u3083", "kana"),
    ("long_vowels", "long vowels", "\u304d\u3087\u3046", "kana"),
    ("vowel_ei", "exact vowel transition", "\u3048\u3044", "kana"),
    ("kanji_vowel_sequence", "Open JTalk vowel sequence", "\u95a2\u4fc2\u306a\u3044\u3067\u3059", "openjtalk"),
    ("phrase_boundaries", "phrase boundaries", "\u3042\u304b\u3002\u304b\u3055\u3002", "kana"),
    ("statement", "statement", "\u3055\u304b\u3002", "kana"),
    ("question", "question", "\u3055\u304b\uff1f", "kana"),
    ("accent_carrier", "accent carrier", "\u3042\u3081\u3068\u3042\u3081", "kana"),
)
_UTAU_SOURCE_MARKERS = ("oto.ini", "character.yaml", "prefix.map")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n").encode("utf-8")


def _require_generated_output(path: Path | str) -> Path:
    resolved = Path(path).expanduser().resolve()
    for directory in (resolved, *resolved.parents):
        if any((directory / marker).is_file() for marker in _UTAU_SOURCE_MARKERS):
            raise ValueError(
                f"Refusing to write a listening corpus inside a source bank: {resolved}"
            )
    return resolved


def render_assembly_listening_set(
    voice_dir: Path | str,
    output_dir: Path | str,
    *,
    wsl_distro: str = "Ubuntu",
) -> Mapping[str, object]:
    voice_root = Path(voice_dir).expanduser().resolve()
    output_root = _require_generated_output(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = load_japanese_runtime_metadata(voice_root)
    voice_name = str(runtime.get("voice_name") or "japanese_assembly")
    entry_point = str(runtime.get("voice_entry_point") or "")
    if not entry_point:
        raise ValueError("generated Japanese voice has no entry point")

    gui_dir = Path(__file__).resolve().parent / "festvox_gui"
    if str(gui_dir) not in sys.path:
        sys.path.insert(0, str(gui_dir))
    from festvox_core import FestivalWSLBackend, write_wav

    backend_key = "stage2_assembly_voice"
    backend = FestivalWSLBackend({
        "festival_wsl": {
            "distro": wsl_distro,
            "timeout_s": 240,
            "voices": {
                backend_key: {
                    "dir": str(voice_root),
                    "voice": entry_point,
                    "voice_en": None,
                    "scm": f"festvox/{voice_name}_ja.scm",
                }
            },
        }
    })
    rows = []
    failures = 0
    for identifier, category, text, frontend_mode in ASSEMBLY_LISTENING_FIXTURES:
        row: dict[str, object] = {
            "id": identifier,
            "category": category,
            "source_text": text,
            "frontend_mode": frontend_mode,
            "output_wav": f"{identifier}.wav",
            "contribution_plan": f"{identifier}.contributions.json",
        }
        try:
            utterance = analyze_japanese(text, mode=frontend_mode)
            plan = create_synthesis_plan(
                utterance,
                runtime_metadata=runtime,
                base_pitch_hz=float(runtime.get("average_pitch_hz") or 180.0),
            )
            arguments = plan.backend_arguments()
            phones = arguments.pop("phones")
            result = backend.synth_phones(phones, backend_key, **arguments)
            write_wav(
                str(output_root / f"{identifier}.wav"),
                result.samples,
                result.sr,
            )
            contributions = create_source_contribution_plan(
                plan, runtime, selected_units=result.selected_units
            )
            (output_root / f"{identifier}.contributions.json").write_bytes(
                contributions.to_json_bytes()
            )
            quality = analyze_plan_joins(
                plan,
                runtime,
                voice_root,
                selected_units=result.selected_units,
                cache_directory=output_root / "quality-cache",
            )
            structural_errors = [
                item.code for item in contributions.diagnostics
                if item.severity == "error"
            ]
            peak = max(
                (abs(float(sample)) for sample in result.samples), default=0.0
            )
            row.update({
                "status": "rendered" if not structural_errors else "invalid",
                "phones": list(plan.phones),
                "duration_seconds": round(result.duration, 6),
                "peak_absolute_sample": round(peak, 8),
                "selected_unit_count": len(result.selected_units),
                "skipped_diphones": sorted(set(result.skipped)),
                "fallback_count": contributions.fallback_count,
                "hidden_silence_count": contributions.hidden_silence_count,
                "structural_errors": structural_errors,
                "join_quality": {
                    "analyzed": len(quality.metrics),
                    "poor": sum(item.rating == "poor" for item in quality.metrics),
                    "review": sum(
                        item.rating == "review" for item in quality.metrics
                    ),
                    "worst_risk_score": max(
                        (item.risk_score for item in quality.metrics),
                        default=None,
                    ),
                    "diagnostics": [
                        item.code for item in quality.diagnostics
                    ],
                },
            })
            if (
                structural_errors
                or len(result.skipped) > 0
                or len(result.samples) == 0
            ):
                failures += 1
        except Exception as error:
            failures += 1
            row.update({
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            })
        rows.append(row)

    manifest = {
        "schema_version": ASSEMBLY_LISTENING_SCHEMA_VERSION,
        "kind": "japanese_stage2_assembly_listening_set",
        "voice_name": voice_name,
        "voice_entry_point": entry_point,
        "bank_type": (
            (runtime.get("voice_configuration") or {}).get("bank_type")
            if isinstance(runtime.get("voice_configuration"), dict)
            else runtime.get("alias_system")
        ),
        "acoustic_naturalness_verified": False,
        "purpose": (
            "Human CV/VCV/CVVC comparison with exact source-contribution "
            "plans; automated success validates structure, not naturalness."
        ),
        "example_count": len(rows),
        "failure_count": failures,
        "examples": rows,
    }
    (output_root / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the Stage 2 Japanese assembly listening corpus."
    )
    parser.add_argument("voice_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--wsl-distro", default="Ubuntu")
    args = parser.parse_args(argv)
    manifest = render_assembly_listening_set(
        args.voice_dir, args.output_dir, wsl_distro=args.wsl_distro
    )
    rendered = manifest["example_count"] - manifest["failure_count"]
    print(f"Rendered {rendered}/{manifest['example_count']} assembly examples.")
    print("Acoustic naturalness remains pending human listening.")
    return 1 if manifest["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
