"""Generate the ignored Japanese structural and refinement listening corpus.

The corpus exists for human review.  Successful rendering proves pipeline
structure only; its manifest always marks acoustic naturalness as unverified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

from japanese_festival import load_japanese_runtime_metadata
from japanese_frontend import analyze_japanese
import japanese_editing as japanese_editing
import japanese_quality as japanese_quality


LISTENING_SET_SCHEMA_VERSION = 3
LISTENING_EXAMPLES = (
    ("vowels", "vowels", "あいうえお。"),
    ("ordinary_cv", "ordinary CV morae", "かきくけこ。"),
    ("vcv_transitions", "VV transitions", "あかあきあく。"),
    ("cvvc_transitions", "CVVC transitions", "さか。"),
    ("moraic_nasal", "moraic nasal", "ほん。"),
    ("geminate", "geminate consonant", "きって。"),
    ("palatalized", "palatalized mora", "きゃく。"),
    ("long_vowels", "long vowels", "コーヒー。"),
    ("devoiced_vowel", "devoiced vowels", "すきです。"),
    ("phrase_boundaries", "phrase boundaries", "これはテストです。つぎです。"),
    ("statement", "statement", "これはテストです。"),
    ("question", "question", "これはテストですか？"),
    ("accent_contrast", "accented and unaccented analysis", "雨と飴。"),
    ("downstep", "multiple accent-phrase downstep", "赤い花を見ました。"),
    ("declination", "long-phrase declination", "静かな朝に長い手紙を読みました。"),
    ("join_stress", "boundary stress joins", "かきくけこ、さしすせそ。"),
)


def required_listening_categories() -> tuple[str, ...]:
    return tuple(item[1] for item in LISTENING_EXAMPLES)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def render_listening_set(
    voice_dir: Path | str,
    output_dir: Path | str,
    *,
    frontend_mode: str = "auto",
    base_pitch_hz: Optional[float] = None,
    wsl_distro: str = "Ubuntu",
) -> dict[str, object]:
    runtime = load_japanese_runtime_metadata(voice_dir)
    voice_name = str(runtime.get("voice_name") or "japanese_phase3")
    entry_point = str(runtime.get("voice_entry_point") or "")
    if not entry_point:
        raise ValueError("generated Japanese voice has no entry point")
    pitch = float(base_pitch_hz or runtime.get("average_pitch_hz") or 180.0)
    voice_root = Path(voice_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    gui_dir = Path(__file__).resolve().parent / "festvox_gui"
    if str(gui_dir) not in sys.path:
        sys.path.insert(0, str(gui_dir))
    from festvox_core import (
        FestivalWSLBackend,
        overlay_intonation_targets,
        phrase_blocks,
        segments_from_durations,
        write_wav,
    )

    backend_key = "phase3_listening_voice"
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

    rows: list[dict[str, object]] = []
    failures = 0
    for example in LISTENING_EXAMPLES:
        identifier, category, text = example[:3]
        options = dict(example[3]) if len(example) > 3 else {}
        example_pitch = max(50.0, min(
            500.0, pitch * float(options.get("pitch_scale", 1.0))
        ))
        row: dict[str, object] = {
            "id": identifier,
            "category": category,
            "source_text": text,
            "output_wav": f"{identifier}.wav",
            "base_pitch_hz": example_pitch,
        }
        try:
            utterance = analyze_japanese(text, mode=frontend_mode)
            edit_state = japanese_editing.new_edit_state(
                utterance, frontend_mode=frontend_mode)
            plan = japanese_editing.create_edited_plan(
                utterance, edit_state,
                runtime_metadata=runtime,
                base_pitch_hz=example_pitch,
            )
            arguments = plan.backend_arguments()
            phones = arguments.pop("phones")
            baseline = list(arguments["pitch_targets"])
            blocks = phrase_blocks(
                segments_from_durations(arguments["seg_durs"]), text)
            arguments["pitch_targets"] = overlay_intonation_targets(
                baseline, blocks, example_pitch, 18.0)
            arguments["ground_truth_targets"] = baseline
            arguments["intonation_blocks"] = blocks
            arguments["pitch_mode"] = "intonation"
            result = backend.synth_phones(
                phones,
                backend_key,
                **arguments,
            )
            write_wav(
                str(output_root / f"{identifier}.wav"),
                result.samples,
                result.sr,
            )
            peak = max((abs(float(value)) for value in result.samples), default=0.0)
            quality = japanese_quality.analyze_plan_joins(
                plan,
                runtime,
                voice_root,
                selected_units=result.selected_units,
                cache_directory=output_root / "quality-cache",
            )
            row.update({
                "status": "rendered",
                "normalized_reading": utterance.normalized_reading,
                "frontend_name": utterance.frontend_name,
                "frontend_version": utterance.frontend_version,
                "phones": plan.phones,
                "segment_count": len(result.segments),
                "f0_target_count": len(result.targets),
                "contour_model": plan.contour_model,
                "speaker_range_hz": [
                    plan.speaker_low_hz, plan.speaker_high_hz],
                "mora_timing_count": len(plan.mora_timings),
                "mora_timings": [item.to_dict()
                                 for item in plan.mora_timings],
                "intonation_blocks": blocks,
                "duration_seconds": round(result.duration, 6),
                "peak_absolute_sample": round(peak, 8),
                "skipped_diphones": sorted(set(result.skipped)),
                "warning": result.warning,
                "frontend_diagnostics": [
                    diagnostic.code for diagnostic in utterance.diagnostics
                ],
                "plan_diagnostics": [
                    diagnostic.code for diagnostic in plan.diagnostics
                ],
                "join_quality": {
                    "analyzed": len(quality.metrics),
                    "poor": sum(item.rating == "poor"
                                for item in quality.metrics),
                    "review": sum(item.rating == "review"
                                  for item in quality.metrics),
                    "worst_risk_score": max(
                        (item.risk_score for item in quality.metrics),
                        default=None,
                    ),
                    "diagnostics": [item.code for item in quality.diagnostics],
                },
            })
        except Exception as error:
            failures += 1
            row.update({
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            })
        rows.append(row)

    manifest = {
        "schema_version": LISTENING_SET_SCHEMA_VERSION,
        "kind": "japanese_remediation_stage6_listening_set",
        "language": "ja",
        "voice_name": voice_name,
        "voice_entry_point": entry_point,
        "frontend_mode": frontend_mode,
        "base_pitch_hz": pitch,
        "acoustic_naturalness_verified": False,
        "purpose": (
            "Human listening review; automated success validates structure "
            "and rendering only."
        ),
        "example_count": len(rows),
        "failure_count": failures,
        "examples": rows,
    }
    (output_root / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the Japanese remediation listening corpus."
    )
    parser.add_argument("voice_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--frontend", choices=("auto", "openjtalk", "kana"), default="auto"
    )
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--wsl-distro", default="Ubuntu")
    args = parser.parse_args(argv)
    manifest = render_listening_set(
        args.voice_dir,
        args.output_dir,
        frontend_mode=args.frontend,
        base_pitch_hz=args.pitch,
        wsl_distro=args.wsl_distro,
    )
    print(
        f"Rendered {manifest['example_count'] - manifest['failure_count']}/"
        f"{manifest['example_count']} listening examples."
    )
    print("Acoustic naturalness remains unverified pending human listening.")
    return 1 if manifest["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
