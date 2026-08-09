"""Render ignored legacy/current Japanese pitch A/B listening fixtures.

Both sides use the same final contextual phone durations and unit-selection
path.  Only the structural F0 targets differ, so the output answers whether
the Prompt 20 pitch model actually reaches Festival/UniSyn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from japanese_duration_ab import _backend, _safe_output
from japanese_festival import load_japanese_runtime_metadata
from japanese_frontend import analyze_japanese
from japanese_prosody_benchmark import (
    LEGACY_PITCH_MODEL_ID,
    legacy_pitch_semitones,
)
from japanese_synthesis import JapaneseSynthesisPlan, create_synthesis_plan
import pitch_domain


PROSODY_AB_FIXTURES = (
    (
        "repeated_phrase",
        "Repeated lexical material later in one utterance",
        "これはテストです。これはテストです。",
    ),
    (
        "statement_question",
        "Statement followed by an interrogative boundary",
        "これはテストです。これはテストですか？",
    ),
    (
        "accent_and_downstep",
        "Multiple lexical accent phrases and downstep",
        "雨と飴を比べます。",
    ),
)


def _gui_core():
    gui_dir = Path(__file__).resolve().parent / "festvox_gui"
    if str(gui_dir) not in sys.path:
        sys.path.insert(0, str(gui_dir))
    import festvox_core
    return festvox_core


def build_pitch_systems(
    utterance,
    plan: JapaneseSynthesisPlan,
    *,
    base_pitch_hz: float,
    fall_percent: float,
) -> dict[str, dict[str, object]]:
    """Return legacy/current targets on the exact same segment timeline."""
    core = _gui_core()
    entries = list(plan.segment_durations)
    kinds = [
        str(phrase.punctuation_after or ".")[-1:]
        for phrase in utterance.phrases
    ]
    blocks = core.phrase_blocks(
        core.segments_from_durations(entries), kinds=kinds)
    legacy_by_mora = legacy_pitch_semitones(utterance)
    legacy_raw = [
        (
            target.time,
            pitch_domain.semitone_offset(
                base_pitch_hz,
                legacy_by_mora.get(target.mora_index, 0.0),
            ),
        )
        for target in plan.f0_targets
    ]
    current_raw = list(plan.pitch_targets)
    return {
        "legacy_pitch": {
            "pitch_model_id": LEGACY_PITCH_MODEL_ID,
            "raw_targets": legacy_raw,
            "render_targets": core.overlay_intonation_targets(
                legacy_raw, blocks, base_pitch_hz, fall_percent),
            "intonation_blocks": blocks,
        },
        "contextual_pitch": {
            "pitch_model_id": plan.pitch_model_id,
            "raw_targets": current_raw,
            "render_targets": core.overlay_intonation_targets(
                current_raw, blocks, base_pitch_hz, fall_percent),
            "intonation_blocks": blocks,
        },
    }


def render_prosody_ab(
    voice_dir: Path | str,
    output_dir: Path | str,
    *,
    frontend_mode: str = "openjtalk",
    wsl_distro: str = "",
    base_pitch_hz: float | None = None,
    fall_percent: float = 18.0,
    backend=None,
) -> dict[str, object]:
    voice_root = Path(voice_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    _safe_output(voice_root, output_root)
    runtime = load_japanese_runtime_metadata(voice_root)
    renderer = backend or _backend(voice_root, runtime, wsl_distro)
    core = _gui_core()
    output_root.mkdir(parents=True, exist_ok=True)
    pitch = float(base_pitch_hz or runtime.get("average_pitch_hz") or 180.0)
    rows = []
    failure_count = 0
    for identifier, category, text in PROSODY_AB_FIXTURES:
        utterance = analyze_japanese(text, mode=frontend_mode)
        plan = create_synthesis_plan(
            utterance,
            runtime_metadata=runtime,
            duration_model="contextual",
            base_pitch_hz=pitch,
        )
        systems = build_pitch_systems(
            utterance,
            plan,
            base_pitch_hz=pitch,
            fall_percent=fall_percent,
        )
        row = {
            "id": identifier,
            "category": category,
            "source_text": text,
            "frontend_name": utterance.frontend_name,
            "duration_model_id": plan.duration_model_id,
            "phone_durations": [list(item) for item in
                                plan.segment_durations],
            "systems": {},
        }
        for system_name, details in systems.items():
            filename = f"{identifier}__{system_name}.wav"
            try:
                rendered = renderer.synth_phones(
                    plan.phones,
                    "japanese_duration_ab",
                    speed=1.0,
                    text=text,
                    lang="ja",
                    seg_durs=plan.segment_durations,
                    pitch=pitch,
                    fall=fall_percent,
                    pitch_targets=details["render_targets"],
                    ground_truth_targets=details["raw_targets"],
                    intonation_blocks=details["intonation_blocks"],
                    pitch_mode="intonation",
                    unit_overrides=plan.unit_overrides,
                )
                core.write_wav(
                    str(output_root / filename),
                    rendered.samples,
                    rendered.sr,
                )
                row["systems"][system_name] = {
                    "status": "rendered",
                    "output_wav": filename,
                    "pitch_model_id": details["pitch_model_id"],
                    "duration_seconds": round(rendered.duration, 6),
                    "raw_targets": [list(item) for item in
                                    details["raw_targets"]],
                    "render_targets": [list(item) for item in
                                       details["render_targets"]],
                    "selected_units": dict(sorted(
                        rendered.selected_units.items())),
                    "warning": rendered.warning,
                }
            except Exception as error:
                failure_count += 1
                row["systems"][system_name] = {
                    "status": "failed",
                    "pitch_model_id": details["pitch_model_id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        rows.append(row)
    manifest = {
        "schema_version": 1,
        "kind": "japanese_pitch_model_ab",
        "duration_model_id": (
            rows[0]["duration_model_id"] if rows else None),
        "base_pitch_hz": pitch,
        "fall_percent": float(fall_percent),
        "fixture_count": len(rows),
        "render_failure_count": failure_count,
        "acoustic_naturalness_verified": False,
        "listening_status": "pending human review",
        "fixtures": rows,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render Japanese legacy/current pitch A/B clips")
    parser.add_argument("voice_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--frontend", choices=("openjtalk", "auto", "kana"),
                        default="openjtalk")
    parser.add_argument("--wsl-distro", default="")
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--fall", type=float, default=18.0)
    args = parser.parse_args(argv)
    manifest = render_prosody_ab(
        args.voice_dir,
        args.output_dir,
        frontend_mode=args.frontend,
        wsl_distro=args.wsl_distro,
        base_pitch_hz=args.pitch,
        fall_percent=args.fall,
    )
    total = len(PROSODY_AB_FIXTURES) * 2
    print(f"Rendered {total - manifest['render_failure_count']}/{total} "
          "Japanese pitch A/B clips.")
    print("Acoustic naturalness remains pending human listening.")
    return 1 if manifest["render_failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
