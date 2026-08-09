"""Render Prompt 20 final-waveform and blind vocal-tract listening sets.

The Festival/UniSyn render is produced once per pitch condition.  Apparent
tract length is then applied by the same source/filter transform used by the
GUI, so these files exercise production ordering rather than an internal
envelope target.  Outputs are ignored build artifacts; the source voice and
UTAU recordings are never opened for writing.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

import diphone_loudness
from formant_analysis import estimate_f0
import japanese_devoicing
from japanese_duration_ab import _backend, _safe_output
from japanese_festival import load_japanese_runtime_metadata
from japanese_frontend import analyze_japanese
from japanese_prosody_ab import build_pitch_systems
from japanese_synthesis import create_synthesis_plan
import vocal_tract


VALIDATION_FIXTURES = (
    ("isolated_vowels", "isolated five-vowel sequence", "あ、い、う、え、お。"),
    ("connected_blue_house", "connected five-vowel coverage",
     "青い家を上に置いてください。"),
    ("connected_red_hat", "connected five-vowel coverage",
     "母は赤い帽子を家に置いた。"),
    ("connected_large_voice", "connected five-vowel coverage",
     "大きな声でゆっくり話してください。"),
    ("long_tokyo", "long vowels", "東京で大きな声を出した。"),
    ("long_high_school", "long vowels", "高校の先生と相談した。"),
    ("long_park", "long vowels", "今日は公園へ行こう。"),
    ("accent_rain", "pitch-accent contrast", "雨が降っています。"),
    ("accent_candy", "pitch-accent contrast", "飴を食べています。"),
    ("accent_chopsticks", "pitch-accent contrast", "箸を置いてください。"),
    ("accent_bridge", "pitch-accent contrast", "橋を渡ってください。"),
    ("accent_edge", "pitch-accent contrast", "端を見てください。"),
    ("creak_end", "creak interaction", "これで終わりです。"),
    ("creak_return", "creak interaction", "今日はもう帰ろう。"),
    ("creak_really", "creak interaction", "本当にそうなの。"),
)

BLIND_FIXTURE_IDS = ("connected_blue_house", "accent_rain", "creak_end")
BLIND_CONDITIONS = (
    "pre_transform_synthesis",
    "identity_transform",
    "realistic_longer_tract",
    "realistic_shorter_tract",
    "expanded_shorter_tract",
    "expanded_longer_tract",
    "pitch_only",
    "tract_only",
    "combined_pitch_and_tract",
)


def _gui_core():
    gui_dir = Path(__file__).resolve().parent / "festvox_gui"
    if str(gui_dir) not in sys.path:
        sys.path.insert(0, str(gui_dir))
    import festvox_core
    return festvox_core


def _median_f0(samples: Sequence[float], sample_rate: int) -> float | None:
    values = np.asarray(samples, np.float64).reshape(-1)
    frame = max(256, int(round(sample_rate * 0.050)))
    hop = max(64, int(round(sample_rate * 0.020)))
    estimates = []
    if values.size < frame:
        values = np.pad(values, (0, frame - values.size))
    for start in range(0, values.size - frame + 1, hop):
        f0_hz, confidence, ambiguity = estimate_f0(
            values[start:start + frame], sample_rate)
        if (f0_hz is not None and confidence >= 0.42 and
                ambiguity <= 0.82):
            estimates.append(float(f0_hz))
    return float(np.median(estimates)) if estimates else None


def _paired_frame_f0_drift(
    source_samples: Sequence[float],
    transformed_samples: Sequence[float],
    sample_rate: int,
) -> float | None:
    """Measure F0 drift on identical frames accepted on both sides.

    Separate whole-utterance medians can compare different accepted-frame
    populations after a spectral-envelope change. Pairing frames measures
    pulse-rate preservation without concealing that estimator-selection bias.
    """
    source = np.asarray(source_samples, np.float64).reshape(-1)
    transformed = np.asarray(transformed_samples, np.float64).reshape(-1)
    count = min(source.size, transformed.size)
    frame = max(256, int(round(sample_rate * 0.050)))
    hop = max(64, int(round(sample_rate * 0.020)))
    differences = []
    for start in range(0, count - frame + 1, hop):
        first = estimate_f0(source[start:start + frame], sample_rate)
        second = estimate_f0(transformed[start:start + frame], sample_rate)
        if (first[0] is None or second[0] is None or
                first[1] < 0.42 or second[1] < 0.42 or
                first[2] > 0.82 or second[2] > 0.82):
            continue
        differences.append(12.0 * math.log2(
            float(second[0]) / float(first[0])))
    return float(np.median(differences)) if differences else None


def _render_sentence(renderer, runtime, text: str, frontend_mode: str,
                     pitch_hz: float, fall_percent: float):
    utterance = analyze_japanese(text, mode=frontend_mode)
    plan = create_synthesis_plan(
        utterance,
        runtime_metadata=runtime,
        duration_model="contextual",
        base_pitch_hz=float(pitch_hz),
    )
    details = build_pitch_systems(
        utterance,
        plan,
        base_pitch_hz=float(pitch_hz),
        fall_percent=float(fall_percent),
    )["contextual_pitch"]
    rendered = renderer.synth_phones(
        plan.phones,
        # _backend registers the selected generated voice under this stable
        # local key; the runtime entry point itself remains voice-specific.
        "japanese_duration_ab",
        speed=1.0,
        text=text,
        lang="ja",
        seg_durs=plan.segment_durations,
        pitch=float(pitch_hz),
        fall=float(fall_percent),
        pitch_targets=details["render_targets"],
        ground_truth_targets=details["raw_targets"],
        intonation_blocks=details["intonation_blocks"],
        pitch_mode="intonation",
        unit_overrides=plan.unit_overrides,
    )
    japanese_devoicing.apply_vowel_realizations(
        rendered,
        plan,
        mode="contextual",
        renderer="auto",
    )
    return utterance, plan, rendered


def _transform(rendered, ratio: float, *, expanded: bool):
    return vocal_tract.transform_vocal_tract(
        rendered.samples,
        rendered.sr,
        float(ratio),
        chipmunk_range=bool(expanded),
        segments=rendered.segments,
    )


def _write_variant(path: Path, samples: Sequence[float], sample_rate: int):
    core = _gui_core()
    path.parent.mkdir(parents=True, exist_ok=True)
    core.write_wav(str(path), np.asarray(samples, np.float32), sample_rate)


def _variant_metrics(samples: Sequence[float], sample_rate: int,
                     source_samples: Sequence[float], source_f0: float | None,
                     *, output_file: str) -> dict[str, object]:
    values = np.asarray(samples, np.float32).reshape(-1)
    source = np.asarray(source_samples, np.float32).reshape(-1)
    f0_hz = _median_f0(values, sample_rate)
    f0_drift = (
        12.0 * math.log2(f0_hz / source_f0)
        if f0_hz and source_f0 else None
    )
    return {
        "output_wav": output_file.replace("\\", "/"),
        "duration_samples": int(values.size),
        "duration_drift_samples": int(values.size - source.size),
        "peak": float(np.max(np.abs(values), initial=0.0)),
        "clipped_sample_count": int(np.count_nonzero(np.abs(values) > 1.0)),
        "median_f0_hz": f0_hz,
        "f0_drift_from_pre_transform_semitones": f0_drift,
        "paired_frame_f0_drift_semitones": _paired_frame_f0_drift(
            source, values, sample_rate),
    }


def _blind_codes(identifier: str) -> dict[str, str]:
    order = sorted(
        BLIND_CONDITIONS,
        key=lambda condition: hashlib.sha256(
            f"prompt20:{identifier}:{condition}".encode("utf-8")
        ).hexdigest(),
    )
    return {condition: chr(ord("A") + index)
            for index, condition in enumerate(order)}


def _join_diagnostic(rendered, samples: Sequence[float]):
    report = diphone_loudness.analyze_rendered_joins(
        samples,
        rendered.sr,
        rendered.segments,
        splice_records=rendered.splice_records,
        target_pitchmarks=rendered.target_pitchmarks,
        selected_units=rendered.selected_units,
        include_curves=False,
        compute_k_weighted_level=False,
    )
    joins = list(report.get("joins") or ())
    dominant = Counter(str(row.get("dominant_issue") or "UNKNOWN")
                       for row in joins)
    return report, {
        "join_count": len(joins),
        "dominant_issue_counts": dict(sorted(dominant.items())),
        "highest_severity": max(
            (float(row.get("severity_score") or 0.0) for row in joins),
            default=0.0,
        ),
    }


def render_vocal_tract_listening_suite(
    voice_dir: Path | str,
    output_dir: Path | str,
    *,
    frontend_mode: str = "openjtalk",
    wsl_distro: str = "",
    base_pitch_hz: float | None = None,
    pitch_shift_semitones: float = 3.0,
    fall_percent: float = 18.0,
    backend=None,
    runtime_metadata: Mapping[str, object] | None = None,
    fixtures: Sequence[tuple[str, str, str]] | None = None,
    blind_fixture_ids: Sequence[str] | None = None,
    analyze_joins: bool = True,
) -> dict[str, object]:
    voice_root = Path(voice_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    _safe_output(voice_root, output_root)
    runtime = (dict(runtime_metadata) if runtime_metadata is not None
               else load_japanese_runtime_metadata(voice_root))
    renderer = backend or _backend(voice_root, runtime, wsl_distro)
    profile = vocal_tract.load_vocal_tract_range()
    source_pitch = float(base_pitch_hz or runtime.get("average_pitch_hz")
                         or 180.0)
    shifted_pitch = source_pitch * (2.0 **
                                    (float(pitch_shift_semitones) / 12.0))
    fixture_rows = tuple(fixtures or VALIDATION_FIXTURES)
    blind_ids = set(blind_fixture_ids or BLIND_FIXTURE_IDS)
    output_root.mkdir(parents=True, exist_ok=True)
    validation_root = output_root / "validation"
    blind_root = output_root / "blind"
    join_root = output_root / "join_diagnostics"
    validation_rows = []
    blind_public = []
    blind_key = []
    failures = 0
    all_identity = True
    maximum_duration_drift = 0
    tract_f0_drifts = []
    paired_tract_f0_drifts = []
    expanded_f0_drifts = []
    paired_expanded_f0_drifts = []

    for identifier, category, text in fixture_rows:
        row: dict[str, object] = {
            "id": identifier,
            "category": category,
            "source_text": text,
            "systems": {},
        }
        try:
            utterance, plan, rendered = _render_sentence(
                renderer, runtime, text, frontend_mode,
                source_pitch, fall_percent,
            )
            source_samples = np.asarray(rendered.samples, np.float32)
            source_f0 = _median_f0(source_samples, rendered.sr)
            identity = _transform(rendered, 1.0, expanded=False)
            longer = _transform(
                rendered, profile.realistic_max_ratio, expanded=False)
            shorter = _transform(
                rendered, profile.realistic_min_ratio, expanded=False)
            all_identity = all_identity and np.array_equal(
                source_samples, identity.samples)
            systems = {
                "pre_transform_synthesis": source_samples,
                "identity_transform": identity.samples,
                "realistic_longer_tract": longer.samples,
                "realistic_shorter_tract": shorter.samples,
            }
            for condition, samples in systems.items():
                relative = Path("validation") / (
                    f"{identifier}__{condition}.wav")
                _write_variant(output_root / relative, samples, rendered.sr)
                metrics = _variant_metrics(
                    samples, rendered.sr, source_samples, source_f0,
                    output_file=str(relative),
                )
                if condition != "pre_transform_synthesis":
                    maximum_duration_drift = max(
                        maximum_duration_drift,
                        abs(int(metrics["duration_drift_samples"])),
                    )
                if (condition.startswith("realistic_") and
                        metrics["f0_drift_from_pre_transform_semitones"]
                        is not None):
                    tract_f0_drifts.append(abs(float(
                        metrics["f0_drift_from_pre_transform_semitones"])))
                if (condition.startswith("realistic_") and
                        metrics["paired_frame_f0_drift_semitones"]
                        is not None):
                    paired_tract_f0_drifts.append(abs(float(
                        metrics["paired_frame_f0_drift_semitones"])))
                if (condition.startswith("expanded_") and
                        metrics["f0_drift_from_pre_transform_semitones"]
                        is not None):
                    expanded_f0_drifts.append(abs(float(
                        metrics["f0_drift_from_pre_transform_semitones"])))
                if (condition.startswith("expanded_") and
                        metrics["paired_frame_f0_drift_semitones"]
                        is not None):
                    paired_expanded_f0_drifts.append(abs(float(
                        metrics["paired_frame_f0_drift_semitones"])))
                row["systems"][condition] = metrics
            row.update({
                "frontend_name": utterance.frontend_name,
                "duration_model_id": plan.duration_model_id,
                "pitch_model_id": plan.pitch_model_id,
                "source_f0_hz": source_f0,
                "identity_exact": bool(np.array_equal(
                    source_samples, identity.samples)),
            })

            if identifier in blind_ids:
                _high_utterance, high_plan, high_rendered = _render_sentence(
                    renderer, runtime, text, frontend_mode,
                    shifted_pitch, fall_percent,
                )
                high_samples = np.asarray(high_rendered.samples, np.float32)
                expanded_shorter = _transform(
                    rendered, profile.expanded_min_ratio, expanded=True)
                expanded_longer = _transform(
                    rendered, profile.expanded_max_ratio, expanded=True)
                for condition, transformed in (
                        ("expanded_shorter_tract", expanded_shorter),
                        ("expanded_longer_tract", expanded_longer)):
                    relative = Path("validation") / (
                        f"{identifier}__{condition}.wav")
                    _write_variant(
                        output_root / relative, transformed.samples,
                        rendered.sr,
                    )
                    metrics = _variant_metrics(
                        transformed.samples, rendered.sr, source_samples,
                        source_f0, output_file=str(relative),
                    )
                    maximum_duration_drift = max(
                        maximum_duration_drift,
                        abs(int(metrics["duration_drift_samples"])),
                    )
                    if metrics[
                            "f0_drift_from_pre_transform_semitones"] is not None:
                        expanded_f0_drifts.append(abs(float(metrics[
                            "f0_drift_from_pre_transform_semitones"])))
                    if metrics[
                            "paired_frame_f0_drift_semitones"] is not None:
                        paired_expanded_f0_drifts.append(abs(float(metrics[
                            "paired_frame_f0_drift_semitones"])))
                    row["systems"][condition] = metrics
                combined = _transform(
                    high_rendered, profile.realistic_min_ratio,
                    expanded=False,
                )
                blind_systems = {
                    "pre_transform_synthesis": source_samples,
                    "identity_transform": identity.samples,
                    "realistic_longer_tract": longer.samples,
                    "realistic_shorter_tract": shorter.samples,
                    "expanded_shorter_tract": expanded_shorter.samples,
                    "expanded_longer_tract": expanded_longer.samples,
                    "pitch_only": high_samples,
                    "tract_only": shorter.samples,
                    "combined_pitch_and_tract": combined.samples,
                }
                codes = _blind_codes(identifier)
                public_conditions = []
                key_conditions = []
                for condition in BLIND_CONDITIONS:
                    code = codes[condition]
                    relative = Path("blind") / f"{identifier}__{code}.wav"
                    _write_variant(
                        output_root / relative, blind_systems[condition],
                        rendered.sr,
                    )
                    public_conditions.append({
                        "code": code,
                        "output_wav": str(relative).replace("\\", "/"),
                    })
                    key_conditions.append({
                        "code": code,
                        "condition": condition,
                        "tract_ratio": {
                            "identity_transform": 1.0,
                            "realistic_longer_tract":
                                profile.realistic_max_ratio,
                            "realistic_shorter_tract":
                                profile.realistic_min_ratio,
                            "expanded_shorter_tract":
                                profile.expanded_min_ratio,
                            "expanded_longer_tract":
                                profile.expanded_max_ratio,
                            "tract_only": profile.realistic_min_ratio,
                            "combined_pitch_and_tract":
                                profile.realistic_min_ratio,
                        }.get(condition, 1.0),
                        "pitch_shift_semitones": (
                            float(pitch_shift_semitones)
                            if condition in {
                                "pitch_only", "combined_pitch_and_tract"
                            } else 0.0
                        ),
                    })
                blind_public.append({
                    "id": identifier,
                    "category": category,
                    "source_text": text,
                    "conditions": sorted(
                        public_conditions, key=lambda item: item["code"]),
                })
                blind_key.append({
                    "id": identifier,
                    "conditions": sorted(
                        key_conditions, key=lambda item: item["code"]),
                })

                if analyze_joins:
                    row["join_diagnostics"] = {}
                    for condition, samples in (
                            ("pre_transform_synthesis", source_samples),
                            ("realistic_longer_tract", longer.samples),
                            ("realistic_shorter_tract", shorter.samples),
                            ("expanded_shorter_tract",
                             expanded_shorter.samples),
                            ("expanded_longer_tract",
                             expanded_longer.samples)):
                        try:
                            report, summary = _join_diagnostic(
                                rendered, samples)
                            relative = Path("join_diagnostics") / (
                                f"{identifier}__{condition}.json")
                            (output_root / relative).parent.mkdir(
                                parents=True, exist_ok=True)
                            (output_root / relative).write_text(
                                json.dumps(report, ensure_ascii=False,
                                           indent=2, sort_keys=True) + "\n",
                                encoding="utf-8",
                            )
                            row["join_diagnostics"][condition] = {
                                "report": str(relative).replace("\\", "/"),
                                "summary": summary,
                            }
                        except (OSError, TypeError, ValueError) as error:
                            row["join_diagnostics"][condition] = {
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                row["shifted_pitch_model_id"] = high_plan.pitch_model_id
        except Exception as error:
            failures += 1
            row["status"] = "failed"
            row["error_type"] = type(error).__name__
            row["error"] = str(error)
        else:
            row["status"] = "rendered"
        validation_rows.append(row)

    public_manifest = {
        "schema_version": 1,
        "kind": "prompt20_vocal_tract_blind_listening",
        "naturalness_verified": False,
        "listening_status": "pending human review",
        "questions": [
            "vowel identity", "intelligibility", "apparent vocal size",
            "naturalness", "speaker identity", "gender presentation",
            "metallic artifacts", "muffling or excessive brightness",
            "consonant damage", "unexpected pitch change",
            "unexpected creak or breathiness change",
        ],
        "fixtures": blind_public,
    }
    key_manifest = {
        "schema_version": 1,
        "kind": "prompt20_vocal_tract_blind_key",
        "fixtures": blind_key,
    }
    (output_root / "blind_manifest.json").write_text(
        json.dumps(public_manifest, ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "blind_key.json").write_text(
        json.dumps(key_manifest, ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "kind": "prompt20_vocal_tract_listening_and_integration",
        "voice_name": str(runtime.get("voice_name") or voice_root.name),
        "base_pitch_hz": source_pitch,
        "pitch_shift_semitones": float(pitch_shift_semitones),
        "reference_profile_model_version": profile.model_version,
        "ratios": {
            "identity": profile.identity_ratio,
            "realistic_min": profile.realistic_min_ratio,
            "realistic_max": profile.realistic_max_ratio,
            "expanded_min": profile.expanded_min_ratio,
            "expanded_max": profile.expanded_max_ratio,
        },
        "fixture_count": len(validation_rows),
        "render_failure_count": failures,
        "identity_exact_for_all_rendered_fixtures": all_identity,
        "maximum_duration_drift_samples": maximum_duration_drift,
        "maximum_realistic_tract_median_f0_estimator_shift_semitones": max(
            tract_f0_drifts, default=0.0),
        "maximum_realistic_tract_paired_frame_f0_drift_semitones": max(
            paired_tract_f0_drifts, default=0.0),
        "maximum_expanded_tract_median_f0_estimator_shift_semitones": max(
            expanded_f0_drifts, default=0.0),
        "maximum_expanded_tract_paired_frame_f0_drift_semitones": max(
            paired_expanded_f0_drifts, default=0.0),
        "structural_validation_passed": bool(
            failures == 0 and all_identity and maximum_duration_drift == 0),
        "acoustic_naturalness_verified": False,
        "listening_status": "pending human review",
        "uniform_warp_retained": True,
        "vowel_conditioned_corrections": False,
        "validation_fixtures": validation_rows,
        "blind_manifest": "blind_manifest.json",
        "blind_key": "blind_key.json",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render Prompt 20 Japanese vocal-tract listening fixtures")
    parser.add_argument("voice_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--frontend", choices=("openjtalk", "auto", "kana"),
                        default="openjtalk")
    parser.add_argument("--wsl-distro", default="")
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--pitch-shift", type=float, default=3.0)
    parser.add_argument("--fall", type=float, default=18.0)
    parser.add_argument("--no-join-diagnostics", action="store_true")
    args = parser.parse_args(argv)
    report = render_vocal_tract_listening_suite(
        args.voice_dir,
        args.output_dir,
        frontend_mode=args.frontend,
        wsl_distro=args.wsl_distro,
        base_pitch_hz=args.pitch,
        pitch_shift_semitones=args.pitch_shift,
        fall_percent=args.fall,
        analyze_joins=not args.no_join_diagnostics,
    )
    print(
        f"Rendered {report['fixture_count'] - report['render_failure_count']}/"
        f"{report['fixture_count']} vocal-tract validation fixtures."
    )
    print("Acoustic naturalness remains pending human listening.")
    return 1 if report["render_failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
