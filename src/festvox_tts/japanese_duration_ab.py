"""Render ignored legacy/contextual Japanese timing and devoicing A/B clips."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
from typing import Sequence

from japanese_devoicing import apply_vowel_realizations, periodicity_score
import japanese_editing
from japanese_festival import load_japanese_runtime_metadata
from japanese_frontend import analyze_japanese


DEFAULT_VALIDATION_PATH = (
    Path(__file__).resolve().parent
    / "profiles" / "japanese_duration_validation_v1.json"
)


def load_validation_config(path: Path | str | None = None) -> dict:
    source = Path(path) if path is not None else DEFAULT_VALIDATION_PATH
    data = json.loads(source.read_text(encoding="utf-8"))
    if int(data.get("schema_version") or 0) != 1:
        raise ValueError("unsupported Japanese duration validation schema")
    fixtures = list(data.get("ab_fixtures") or ())
    systems = list(data.get("ab_systems") or ())
    if not fixtures or not systems:
        raise ValueError("Japanese duration validation matrix is empty")
    fixture_ids = [str(item.get("id") or "") for item in fixtures]
    system_ids = [str(item.get("id") or "") for item in systems]
    if any(not item for item in fixture_ids + system_ids):
        raise ValueError("Japanese duration validation IDs must be nonempty")
    if len(set(fixture_ids)) != len(fixture_ids):
        raise ValueError("duplicate Japanese duration fixture ID")
    if len(set(system_ids)) != len(system_ids):
        raise ValueError("duplicate Japanese duration system ID")
    return data


_VALIDATION_CONFIG = load_validation_config()
AB_FIXTURES = tuple(
    (str(row["id"]), str(row["category"]), str(row["text"]))
    for row in _VALIDATION_CONFIG["ab_fixtures"]
)
AB_SYSTEMS = tuple(
    (str(row["id"]), str(row["duration_model"]),
     str(row["devoicing_mode"]), str(row["devoicing_renderer"]))
    for row in _VALIDATION_CONFIG["ab_systems"]
)


def fixture_categories() -> tuple[str, ...]:
    return tuple(category for _identifier, category, _text in AB_FIXTURES)


def _safe_output(voice_root: Path, output_root: Path) -> None:
    if output_root == voice_root or voice_root in output_root.parents:
        raise ValueError("A/B output must not be inside the generated voice")
    lowered = {part.casefold() for part in output_root.parts}
    if "utau" in lowered and "voice" in lowered:
        raise ValueError("Refusing to write A/B output inside an UTAU bank")


def _backend(voice_root: Path, runtime: dict, distro: str):
    gui_dir = Path(__file__).resolve().parent / "festvox_gui"
    if str(gui_dir) not in sys.path:
        sys.path.insert(0, str(gui_dir))
    from festvox_core import FestivalWSLBackend
    voice_name = str(runtime.get("voice_name") or "japanese_duration_ab")
    entry_point = str(runtime.get("voice_entry_point") or "")
    if not entry_point:
        raise ValueError("generated Japanese voice has no Festival entry point")
    voice_scm = str(runtime.get("voice_scm") or
                    f"festvox/{voice_name}_ja.scm")
    return FestivalWSLBackend({
        "festival_wsl": {
            "distro": distro,
            "timeout_s": 240,
            "voices": {
                "japanese_duration_ab": {
                    "dir": str(voice_root),
                    "voice": entry_point,
                    "voice_en": None,
                    "scm": voice_scm,
                }
            },
        }
    })


def _rounded_mean(values):
    rows = [float(value) for value in values if value is not None]
    return round(float(statistics.fmean(rows)), 6) if rows else None


def _rounded_median(values):
    rows = [float(value) for value in values if value is not None]
    return round(float(statistics.median(rows)), 6) if rows else None


def summarize_ab_rows(rows) -> dict[str, object]:
    """Aggregate objective timing and voicing measures without listening claims."""
    duration_deltas = []
    decisions = []
    rendered_pairs = 0
    for row in rows:
        systems = dict(row.get("systems") or {})
        legacy = dict(systems.get("legacy") or {})
        contextual = dict(systems.get("contextual") or {})
        if (legacy.get("status") == "rendered" and
                contextual.get("status") == "rendered"):
            rendered_pairs += 1
            duration_deltas.append(
                float(contextual["duration_seconds"])
                - float(legacy["duration_seconds"])
            )
        decisions.extend(contextual.get("vowel_realizations") or ())
    strategies = Counter(str(item.get("strategy") or "unknown")
                         for item in decisions)
    phones = Counter(str(item.get("phone") or "unknown")
                     for item in decisions)
    applied = [item for item in decisions if bool(item.get("applied"))]
    periodicity_rows = [item for item in decisions
                        if item.get("periodicity_before") is not None
                        and item.get("periodicity_after") is not None]
    periodicity_drops = [
        float(item["periodicity_before"])
        - float(item["periodicity_after"])
        for item in periodicity_rows
    ]
    return {
        "rendered_pair_count": rendered_pairs,
        "duration_delta_seconds_mean": _rounded_mean(duration_deltas),
        "duration_delta_seconds_median": _rounded_median(duration_deltas),
        "duration_delta_absolute_seconds_mean": _rounded_mean(
            abs(value) for value in duration_deltas
        ),
        "vowel_decision_count": len(decisions),
        "vowel_decision_applied_count": len(applied),
        "strategy_counts": dict(sorted(strategies.items())),
        "phone_counts": dict(sorted(phones.items())),
        "periodicity_pair_count": len(periodicity_rows),
        "periodicity_before_median": _rounded_median(
            item["periodicity_before"] for item in periodicity_rows
        ),
        "periodicity_after_median": _rounded_median(
            item["periodicity_after"] for item in periodicity_rows
        ),
        "periodicity_drop_mean": _rounded_mean(periodicity_drops),
        "spectral_envelope_distance_median": _rounded_median(
            item.get("spectral_envelope_distance") for item in applied
        ),
        "absolute_level_step_db_max": (
            round(max(abs(float(item["level_step_db"])) for item in applied
                      if item.get("level_step_db") is not None), 6)
            if any(item.get("level_step_db") is not None for item in applied)
            else None
        ),
        "coverage": {
            "i": phones.get("i", 0),
            "u": phones.get("u", 0),
            "stop_fixtures": sum(
                1 for row in rows if row.get("id") in {
                    "i_stop", "i_voiced_control", "ambiguous"
                }
            ),
            "fricative_fixtures": sum(
                1 for row in rows
                if row.get("id") in {
                    "suki_desu", "tsukue_fuku", "u_fricative",
                    "sushi_tsukutta",
                }
            ),
            "ambiguous_fixtures": sum(
                1 for row in rows if row.get("id") == "ambiguous"
            ),
        },
    }


def markdown_report(manifest: dict[str, object]) -> str:
    summary = dict(manifest.get("summary") or {})
    lines = [
        "# Japanese Duration and Voicing A/B Report",
        "",
        "This report is generated from the same rendered utterances for the "
        "legacy and contextual systems. Acoustic naturalness remains pending "
        "human listening.",
        "",
        "## Summary",
        "",
        f"- Rendered pairs: {summary.get('rendered_pair_count', 0)}",
        f"- Mean contextual-minus-legacy duration: "
        f"{summary.get('duration_delta_seconds_mean')} s",
        f"- Vowel decisions applied: "
        f"{summary.get('vowel_decision_applied_count', 0)} / "
        f"{summary.get('vowel_decision_count', 0)}",
        f"- Median periodicity: {summary.get('periodicity_before_median')} "
        f"before, {summary.get('periodicity_after_median')} after",
        f"- Mean periodicity reduction: "
        f"{summary.get('periodicity_drop_mean')}",
        f"- Median spectral-envelope distance: "
        f"{summary.get('spectral_envelope_distance_median')}",
        f"- Maximum absolute local level change: "
        f"{summary.get('absolute_level_step_db_max')} dB",
        f"- Strategies: {json.dumps(summary.get('strategy_counts', {}), sort_keys=True)}",
        "",
        "## Fixtures",
        "",
        "| ID | Category | Legacy (s) | Duration only (s) | Contextual (s) | Decisions |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in manifest.get("examples") or ():
        systems = dict(row.get("systems") or {})
        legacy = dict(systems.get("legacy") or {})
        duration_only = dict(systems.get("duration_only") or {})
        contextual = dict(systems.get("contextual") or {})
        decisions = ", ".join(
            f"{item.get('phone')}:{item.get('strategy')}"
            for item in contextual.get("vowel_realizations") or ()
        ) or "none"
        lines.append(
            f"| {row.get('id')} | {row.get('category')} | "
            f"{legacy.get('duration_seconds', 'failed')} | "
            f"{duration_only.get('duration_seconds', 'failed')} | "
            f"{contextual.get('duration_seconds', 'failed')} | {decisions} |"
        )
    lines.extend([
        "",
        "## Source-filter Spectrogram Check",
        "",
        json.dumps(manifest.get("voicing_spectrogram_validation") or {},
                   ensure_ascii=False, sort_keys=True),
        "",
        "## Direct PSOLA Check",
        "",
        json.dumps(manifest.get("psola_no_f0_experiment") or {},
                   ensure_ascii=False, sort_keys=True),
        "",
        "## Interpretation Limits",
        "",
        "Periodicity, duration, level, and envelope measurements are reported "
        "separately. They do not establish naturalness, moraic rhythm, word "
        "identity, or transition quality. The clips require human comparison.",
        "",
    ])
    return "\n".join(lines)


def _render_no_f0_experiment(renderer, runtime, output_root, frontend_mode,
                             pitch):
    """Test whether removing explicit targets removes source periodicity."""
    selected = next(
        (item for item in AB_FIXTURES if item[0] == "suki_desu"),
        AB_FIXTURES[0],
    )
    identifier, category, text = selected
    try:
        utterance = analyze_japanese(text, mode=frontend_mode)
        edit_state = japanese_editing.new_edit_state(
            utterance, frontend_mode=frontend_mode
        )
        plan = japanese_editing.create_edited_plan(
            utterance,
            edit_state,
            runtime_metadata=runtime,
            base_pitch_hz=pitch,
            duration_model="contextual",
        )
        gui_dir = Path(__file__).resolve().parent / "festvox_gui"
        if str(gui_dir) not in sys.path:
            sys.path.insert(0, str(gui_dir))
        from festvox_core import write_wav

        def extended_control(arguments):
            arguments = dict(arguments)
            entries = list(arguments.get("seg_durs") or ())
            for index, (phone, duration) in enumerate(entries):
                if str(phone) in {"i", "u"}:
                    entries[index] = (phone, max(0.30, float(duration)))
                    break
            arguments["seg_durs"] = entries
            return arguments

        def render(arguments, filename):
            arguments = dict(arguments)
            phones = arguments.pop("phones")
            result = renderer.synth_phones(
                phones, "japanese_duration_ab", **arguments
            )
            apply_vowel_realizations(
                result, plan, mode="contextual", renderer="shortened_voiced"
            )
            write_wav(str(output_root / filename), result.samples, result.sr)
            decisions = list(result.vowel_realizations)
            wide_periodicity = []
            longest_vowel = None
            for item in decisions:
                index = int(item.get("segment_index", -1))
                if not 0 <= index < len(result.segments):
                    continue
                segment = result.segments[index]
                first = max(0, int(round(float(segment.start) * result.sr)))
                last = min(len(result.samples), int(round(
                    float(segment.end) * result.sr
                )))
                value = periodicity_score(
                    result.samples[first:last], result.sr,
                    minimum_f0=25.0, maximum_f0=450.0,
                )
                if value is not None:
                    wide_periodicity.append(value)
                    duration = float(segment.end) - float(segment.start)
                    if longest_vowel is None or duration > longest_vowel[0]:
                        longest_vowel = (duration, value)
            return {
                "status": "rendered",
                "output_wav": filename,
                "vowel_realizations": decisions,
                "median_periodicity": _rounded_median(
                    item.get("periodicity_before") for item in decisions
                ),
                "median_periodicity_25_450_hz": _rounded_median(
                    wide_periodicity
                ),
                "longest_vowel_periodicity_25_450_hz": (
                    round(float(longest_vowel[1]), 6)
                    if longest_vowel is not None else None
                ),
            }

        absent_arguments = extended_control(plan.backend_arguments())
        absent_arguments["pitch_targets"] = []
        absent_arguments["pitch"] = None
        absent_arguments["prev_targets"] = []
        try:
            absent = render(
                absent_arguments, f"{identifier}__no_explicit_f0.wav"
            )
        except Exception as error:
            absent = {
                "status": "unsupported",
                "error_type": type(error).__name__,
                "error": str(error),
            }

        minimum_arguments = extended_control(plan.backend_arguments())
        minimum_arguments["pitch"] = 40.0
        minimum_targets = []
        elapsed = 0.0
        for phone, raw_duration in minimum_arguments["seg_durs"]:
            duration = max(0.0, float(raw_duration))
            if phone != "pau":
                minimum_targets.append((elapsed + duration * 0.5, 40.0))
            elapsed += duration
        minimum_arguments["pitch_targets"] = minimum_targets
        minimum = render(
            minimum_arguments, f"{identifier}__minimum_f0.wav"
        )
        return {
            "status": "completed",
            "fixture_id": identifier,
            "category": category,
            "control_vowel_minimum_seconds": 0.30,
            "absent_target_attempt": absent,
            "minimum_f0_attempt": minimum,
            "interpretation": (
                "An absent target contour and a 40 Hz floor are tested, not "
                "assumed, as possible UniSyn equivalents of an unvoiced "
                "target. The wide-band periodicity metric includes 40 Hz so "
                "a slow pulse train cannot be misclassified as devoicing."
            ),
        }
    except Exception as error:
        return {
            "status": "failed",
            "fixture_id": identifier,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def render_duration_ab(
    voice_dir: Path | str,
    output_dir: Path | str,
    *,
    frontend_mode: str = "auto",
    wsl_distro: str = "Ubuntu",
    base_pitch_hz: float | None = None,
    backend=None,
) -> dict[str, object]:
    voice_root = Path(voice_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    _safe_output(voice_root, output_root)
    runtime = load_japanese_runtime_metadata(voice_root)
    output_root.mkdir(parents=True, exist_ok=True)
    renderer = backend or _backend(voice_root, runtime, wsl_distro)
    gui_dir = Path(__file__).resolve().parent / "festvox_gui"
    if str(gui_dir) not in sys.path:
        sys.path.insert(0, str(gui_dir))
    from festvox_core import write_wav

    pitch = float(base_pitch_hz or runtime.get("average_pitch_hz") or 180.0)
    rows = []
    failures = 0
    for identifier, category, text in AB_FIXTURES:
        utterance = analyze_japanese(text, mode=frontend_mode)
        edit_state = japanese_editing.new_edit_state(
            utterance, frontend_mode=frontend_mode
        )
        row = {
            "id": identifier,
            "category": category,
            "source_text": text,
            "frontend_name": utterance.frontend_name,
            "systems": {},
        }
        for system, duration_model, devoicing_mode, devoicing_renderer in AB_SYSTEMS:
            try:
                plan = japanese_editing.create_edited_plan(
                    utterance,
                    edit_state,
                    runtime_metadata=runtime,
                    base_pitch_hz=pitch,
                    duration_model=duration_model,
                )
                arguments = plan.backend_arguments()
                phones = arguments.pop("phones")
                result = renderer.synth_phones(
                    phones, "japanese_duration_ab", **arguments
                )
                apply_vowel_realizations(
                    result, plan, mode=devoicing_mode,
                    renderer=devoicing_renderer,
                )
                filename = f"{identifier}__{system}.wav"
                write_wav(str(output_root / filename), result.samples, result.sr)
                row["systems"][system] = {
                    "status": "rendered",
                    "output_wav": filename,
                    "duration_seconds": round(result.duration, 6),
                    "duration_model": plan.duration_model,
                    "duration_model_id": plan.duration_model_id,
                    "phone_durations": [
                        [segment.phone, round(segment.duration, 6)]
                        for segment in plan.segments
                    ],
                    "vowel_realizations": list(result.vowel_realizations),
                    "voicing_diagnostics": list(
                        result.voicing_diagnostics
                    ),
                    "skipped_diphones": sorted(set(result.skipped)),
                    "warning": result.warning,
                }
            except Exception as error:
                failures += 1
                row["systems"][system] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        rows.append(row)
    manifest = {
        "schema_version": 1,
        "kind": "japanese_duration_and_devoicing_ab",
        "voice_name": runtime.get("voice_name"),
        "frontend_mode": frontend_mode,
        "base_pitch_hz": pitch,
        "systems": [item[0] for item in AB_SYSTEMS],
        "example_count": len(rows),
        "render_failure_count": failures,
        "acoustic_naturalness_verified": False,
        "listening_status": "pending human review",
        "examples": rows,
    }
    manifest["psola_no_f0_experiment"] = _render_no_f0_experiment(
        renderer, runtime, output_root, frontend_mode, pitch
    )
    duration_control = next(
        (row for row in rows if row.get("id") == "suki_desu"), None
    )
    if duration_control:
        systems = dict(duration_control.get("systems") or {})
        duration_only = dict(systems.get("duration_only") or {})
        contextual = dict(systems.get("contextual") or {})
        experiment = manifest["psola_no_f0_experiment"]
        experiment["ordinary_target_periodicity"] = _rounded_median(
            item.get("periodicity_before") for item in
            (duration_only.get("vowel_realizations") or ())
        )
        experiment["source_filter_periodicity"] = _rounded_median(
            item.get("periodicity_after") for item in
            (contextual.get("vowel_realizations") or ())
        )
        try:
            from voicing_validation import generate_voicing_validation
            source_name = duration_only.get("output_wav")
            if not source_name:
                raise ValueError("duration-only validation WAV is unavailable")
            validation = generate_voicing_validation(
                output_root / str(source_name), output_root,
                prefix="suki_desu_voicing",
            )
            manifest["voicing_spectrogram_validation"] = {
                "status": "generated",
                "source_fixture": duration_control.get("id"),
                "outputs": validation.get("outputs"),
                "source_f0_hz": validation.get("source_f0_hz"),
                "zero_vs_source": validation.get("zero_vs_source"),
                "naturalness_verified": False,
            }
        except Exception as error:
            manifest["voicing_spectrogram_validation"] = {
                "status": "failed",
                "source_fixture": duration_control.get("id"),
                "error_type": type(error).__name__,
                "error": str(error),
                "naturalness_verified": False,
            }
    else:
        manifest["voicing_spectrogram_validation"] = {
            "status": "unavailable",
            "error": "the fixed suki_desu fixture was not rendered",
            "naturalness_verified": False,
        }
    manifest["summary"] = summarize_ab_rows(rows)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "report.md").write_text(
        markdown_report(manifest), encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render Japanese legacy/contextual duration A/B clips"
    )
    parser.add_argument("voice_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--frontend", choices=("auto", "openjtalk", "kana"),
                        default="auto")
    parser.add_argument("--wsl-distro", default="Ubuntu")
    parser.add_argument("--pitch", type=float)
    args = parser.parse_args(argv)
    manifest = render_duration_ab(
        args.voice_dir,
        args.output_dir,
        frontend_mode=args.frontend,
        wsl_distro=args.wsl_distro,
        base_pitch_hz=args.pitch,
    )
    expected = len(AB_FIXTURES) * len(AB_SYSTEMS)
    rendered = expected - manifest["render_failure_count"]
    print(f"Rendered {rendered}/{expected} A/B clips.")
    print("Acoustic naturalness remains pending human listening.")
    return 1 if manifest["render_failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
