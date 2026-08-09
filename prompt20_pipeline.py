"""Reproducible Prompt 20 reference, alignment, and Stage A analysis CLI."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

from formant_analysis import (
    AnalysisSegment,
    FormantAnalysisConfig,
    analyze_segment,
    derive_reference_voice_space,
    read_audio,
    segments_from_kokoro_alignment,
    source_voice_segments,
    speaker_formant_summary,
    supplied_reference_segments,
    write_csv,
    write_json,
)
from formant_plots import write_alignment_plot, write_stage_a_plot_suite
from japanese_duration_corpus import (
    CorpusUtterance,
    TimedPhone,
    evaluate_duration_model,
    fit_duration_priors,
    write_priors,
)
from kokoro_reference import (
    KokoroAlignCheckpoint,
    KokoroRecord,
    align_kokoro_record,
    align_kokoro_with_checkpoint,
    inventory_kokoro_prefix,
    load_alignment,
    load_selection,
    safe_extract_kokoro_archive,
    select_stratified_records,
    sha256_file,
    write_alignment,
)


PIPELINE_VERSION = "prompt20-stage-a-v1"


def _print(message: str) -> None:
    print(message, flush=True)


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _record_from_dict(row: Mapping[str, object]) -> KokoroRecord:
    return KokoroRecord(
        utterance_id=str(row["utterance_id"]),
        transcript=str(row.get("transcript") or ""),
        reading=str(row.get("reading") or ""),
        phones=tuple(str(item) for item in (row.get("phones") or ())),
        partition=str(row.get("partition") or "train"),
        strata=tuple(str(item) for item in (row.get("strata") or ())),
    )


def _hash_with_progress(path: Path) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    read = 0
    next_report = 512 * 1024 * 1024
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            read += len(chunk)
            if read >= next_report:
                _print(f"archive hash: {read / (1024 ** 3):.1f} / "
                       f"{total / (1024 ** 3):.1f} GiB")
                next_report += 512 * 1024 * 1024
    return digest.hexdigest()


def extract_kokoro_sample(
    archive: Path,
    output: Path,
    *,
    candidate_count: int,
    train_count: int,
    validation_count: int,
    test_count: int,
    hash_archive: bool,
) -> dict[str, object]:
    _print(f"inventory: first {candidate_count} Kokoro FLAC members")
    inventory = inventory_kokoro_prefix(
        archive, maximum_audio_members=candidate_count
    )
    _write_json(output / "bounded_inventory.json", inventory)
    candidates = tuple(
        _record_from_dict(row) for row in inventory["candidate_records"]
    )
    selected = select_stratified_records(
        candidates,
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
    )
    expected = train_count + validation_count + test_count
    if len(selected) != expected:
        raise ValueError(
            f"bounded archive prefix supplied {len(selected)} of {expected} "
            "requested partition records; increase --candidate-count"
        )
    digest = _hash_with_progress(archive) if hash_archive else None
    _print(f"extract: {len(selected)} selected FLAC files")
    report = safe_extract_kokoro_archive(
        archive,
        output,
        record_ids=tuple(record.utterance_id for record in selected),
        archive_sha256=digest,
    )
    _print("extract: complete")
    return report


def align_kokoro_sample(
    corpus_root: Path,
    output: Path,
    *,
    checkpoint_path: Path | None,
    minimum_confidence: float,
) -> dict[str, object]:
    records = load_selection(corpus_root / "partitions.json")
    checkpoint = (KokoroAlignCheckpoint(checkpoint_path)
                  if checkpoint_path is not None else None)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    visual_candidates: list[tuple[KokoroRecord, object, object]] = []
    for index, record in enumerate(records, 1):
        audio_path = corpus_root / "wavs" / f"{record.utterance_id}.flac"
        audio = read_audio(audio_path, expected_sample_rate=22050)
        if checkpoint is not None:
            alignment = align_kokoro_with_checkpoint(
                checkpoint, record, audio.samples, audio.sample_rate,
                minimum_confidence=minimum_confidence,
            )
        else:
            alignment = align_kokoro_record(
                record, audio.samples, audio.sample_rate,
                minimum_confidence=minimum_confidence,
            )
        write_alignment(output / f"{record.utterance_id}.json", alignment)
        results.append({
            "utterance_id": record.utterance_id,
            "partition": record.partition,
            "strata": list(record.strata),
            "method": alignment.method,
            "accepted": alignment.accepted,
            "confidence": alignment.confidence,
            "phone_count": len(alignment.phones),
            "rejected_phone_count": sum(
                bool(phone.rejection_reasons) for phone in alignment.phones
            ),
            "diagnostics": list(alignment.diagnostics),
        })
        if len(visual_candidates) < 6 and (
                not visual_candidates or
                record.partition not in {row[0].partition
                                         for row in visual_candidates} or
                "geminate" in record.strata or
                "long_vowel" in record.strata):
            visual_candidates.append((record, alignment, audio))
        _print(
            f"align {index:02d}/{len(records):02d}: {record.utterance_id} "
            f"confidence={alignment.confidence:.3f} "
            f"accepted={alignment.accepted}"
        )
    visual_root = output.parent / "alignment_plots"
    for record, alignment, audio in visual_candidates[:6]:
        write_alignment_plot(
            visual_root / f"{record.utterance_id}.svg",
            audio.samples,
            audio.sample_rate,
            alignment.phones,
            title=f"Silver alignment: {record.utterance_id}",
        )
    accepted = [row for row in results if row["accepted"]]
    report = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "kind": "kokoro_alignment_summary",
        "record_count": len(results),
        "accepted_count": len(accepted),
        "rejected_count": len(results) - len(accepted),
        "median_confidence": (
            round(float(np.median([row["confidence"] for row in results])), 6)
            if results else None
        ),
        "checkpoint": ({
            "file_name": checkpoint.path.name,
            "sha256": checkpoint.sha256,
        } if checkpoint is not None else None),
        "visual_validation": {
            "status": "plots_generated_human_judgment_pending",
            "utterance_ids": [row[0].utterance_id
                              for row in visual_candidates[:6]],
            "plot_directory": "alignment_plots",
            "silver_boundary_warning": True,
        },
        "records": results,
    }
    _write_json(output / "alignment_summary.json", report)
    return report


def _verify_source_bundle(source_root: Path, manifest_path: Path,
                          *, label: str) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = manifest.get("source_recording_bundle") or {}
    rows = list(bundle.get("oto_files") or ()) + \
        list(bundle.get("recording_files") or ())
    mismatches = []
    actual_rows = []
    for index, row in enumerate(rows, 1):
        relative = str(row["path"])
        path = (source_root / relative).resolve()
        try:
            path.relative_to(source_root.resolve())
        except ValueError:
            mismatches.append({"path": relative, "reason": "path_escape"})
            continue
        if not path.is_file():
            mismatches.append({"path": relative, "reason": "missing"})
            continue
        actual = sha256_file(path)
        expected = str(row.get("sha256") or "")
        if actual != expected:
            mismatches.append({
                "path": relative,
                "reason": "sha256_mismatch",
                "expected": expected,
                "actual": actual,
            })
        actual_rows.append(f"{relative}\0{actual}")
        if index % 100 == 0:
            _print(f"source verification {label}: {index}/{len(rows)}")
    digest = hashlib.sha256(
        "\n".join(sorted(actual_rows)).encode("utf-8")
    ).hexdigest()
    result = {
        "label": label,
        "expected_inventory_sha256": bundle.get("inventory_sha256"),
        "verified_file_count": len(actual_rows),
        "actual_content_inventory_sha256": digest,
        "mismatches": mismatches,
        "passed": not mismatches and len(actual_rows) == len(rows),
    }
    if not result["passed"]:
        raise RuntimeError(
            f"source-bank verification failed during {label}: "
            + json.dumps(mismatches[:5], ensure_ascii=False)
        )
    return result


def _duration_utterance(record: KokoroRecord, alignment,
                        audio_path: Path) -> CorpusUtterance:
    phones = tuple(TimedPhone(
        start_100ns=int(round(phone.start_seconds * 10_000_000)),
        end_100ns=int(round(phone.end_seconds * 10_000_000)),
        phone=("sp" if phone.phone == "_" else phone.phone),
        raw_label=phone.raw_phone,
        devoiced=phone.probable_devoicing,
    ) for phone in alignment.phones)
    return CorpusUtterance(
        utterance_id=record.utterance_id,
        phones=phones,
        wav_path=str(audio_path),
        corpus="Kokoro-Speech-Dataset-v1.3-xlarge-silver",
        diagnostics=(
            f"alignment_method={alignment.method}",
            f"alignment_confidence={alignment.confidence}",
        ),
    )


def _kokoro_measurements(records, alignments, analyzed_segments) -> dict[str, object]:
    phone_groups: dict[str, list[float]] = defaultdict(list)
    mora_durations: list[float] = []
    phrase_durations: list[float] = []
    pause_durations: list[float] = []
    short_pause_durations: list[float] = []
    speaking_rates: list[float] = []
    utterances = []
    for record in records:
        alignment = alignments[record.utterance_id]
        if not alignment.accepted:
            continue
        mora_groups: dict[tuple[int, int], list[object]] = defaultdict(list)
        phrase_groups: dict[int, list[object]] = defaultdict(list)
        for phone in alignment.phones:
            phone_groups[phone.phone].append(phone.duration_seconds)
            if phone.phone in {"sp", "_"}:
                short_pause_durations.append(phone.duration_seconds)
                continue
            if phone.phone == "pau":
                pause_durations.append(phone.duration_seconds)
                continue
            mora_groups[(phone.phrase_index, phone.mora_index)].append(phone)
            phrase_groups[phone.phrase_index].append(phone)
        utterance_mora = [
            max(item.end_seconds for item in group) -
            min(item.start_seconds for item in group)
            for group in mora_groups.values()
        ]
        utterance_phrase = [
            max(item.end_seconds for item in group) -
            min(item.start_seconds for item in group)
            for group in phrase_groups.values()
        ]
        mora_durations.extend(utterance_mora)
        phrase_durations.extend(utterance_phrase)
        active = [phone for phone in alignment.phones if phone.phone != "pau"]
        active_duration = (
            max(phone.end_seconds for phone in active) -
            min(phone.start_seconds for phone in active)
            if active else 0.0
        )
        rate = len(utterance_mora) / active_duration if active_duration else 0.0
        if rate:
            speaking_rates.append(rate)
        utterances.append({
            "utterance_id": record.utterance_id,
            "partition": record.partition,
            "mora_count": len(utterance_mora),
            "phrase_count": len(utterance_phrase),
            "active_duration_seconds": round(active_duration, 6),
            "morae_per_second": round(rate, 6),
            "alignment_confidence": alignment.confidence,
        })
    phrase_f0: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for row in analyzed_segments:
        if row.segment.source_corpus != "Kokoro-Speech-Dataset-v1.3-xlarge":
            continue
        utterance_id = str(row.segment.segment_id).split(":")[1]
        phrase_index = int(row.segment.metadata.get("phrase_index", 0))
        for frame in row.frames:
            if frame.f0_hz is not None and frame.f0_confidence >= 0.42:
                phrase_f0[(utterance_id, phrase_index)].append(
                    (frame.frame_time_seconds, frame.f0_hz)
                )
    phrase_pitch = []
    for (utterance_id, phrase_index), values in sorted(phrase_f0.items()):
        values.sort()
        frequencies = np.asarray([value for _time, value in values])
        baseline = float(np.median(frequencies))
        semitones = 12.0 * np.log2(frequencies / max(baseline, 1e-9))
        times = np.asarray([time for time, _value in values])
        slope = 0.0
        if len(values) >= 2 and float(np.ptp(times)) > 1e-6:
            slope = float(np.polyfit(times - times[0], semitones, 1)[0])
        phrase_pitch.append({
            "utterance_id": utterance_id,
            "phrase_index": phrase_index,
            "frame_count": len(values),
            "median_f0_hz": round(baseline, 6),
            "f0_range_semitones_p10_p90": round(
                float(np.percentile(semitones, 90) -
                      np.percentile(semitones, 10)), 6
            ),
            "declination_semitones_per_second": round(slope, 6),
        })

    def summary(values):
        finite = np.asarray(values, dtype=np.float64)
        return {
            "count": int(finite.size),
            "median": round(float(np.median(finite)), 8) if finite.size else None,
            "p10": round(float(np.percentile(finite, 10)), 8) if finite.size else None,
            "p90": round(float(np.percentile(finite, 90)), 8) if finite.size else None,
        }

    return {
        "schema_version": 1,
        "kind": "kokoro_phone_mora_phrase_f0_rate_measurements",
        "silver_alignment_warning": True,
        "phone_duration_seconds": {
            phone: summary(values) for phone, values in sorted(phone_groups.items())
        },
        "mora_duration_seconds": summary(mora_durations),
        "phrase_duration_seconds": summary(phrase_durations),
        "pause_duration_seconds": summary(pause_durations),
        "short_pause_duration_seconds": summary(short_pause_durations),
        "speaking_rate_morae_per_second": summary(speaking_rates),
        "phrase_pitch": phrase_pitch,
        "utterances": utterances,
    }


def _write_analysis_report(output: Path, *, alignment_summary, summary,
                           voice_space, gate, duration_report,
                           source_before, source_after) -> Path:
    confidence = alignment_summary.get("median_confidence")
    lines = [
        "# Prompt 20 Stage A Formant Analysis",
        "",
        f"Pipeline version: `{PIPELINE_VERSION}`",
        "",
        "## Gate",
        "",
        f"Stage A passed: **{gate['passed']}**",
        "",
    ]
    for check in gate["checks"]:
        lines.append(f"- {'PASS' if check['passed'] else 'FAIL'}: {check['name']} - {check['detail']}")
    lines.extend([
        "",
        "## Inputs and provenance",
        "",
        "- The project UTAU source recordings are the absolute speaker baseline.",
        "- The user-supplied formant-shift WAVs are the primary transformation references.",
        "- Kokoro Speech v1.3 is public-domain secondary duration/prosody evidence.",
        "- Kokoro-Align is MIT-licensed; its epoch-200 CTC boundaries are silver references.",
        "- Reports retain path-neutral file names and cryptographic hashes.",
        "",
        "## Alignment",
        "",
        f"Accepted {alignment_summary['accepted_count']} of {alignment_summary['record_count']} sampled utterances; median confidence `{confidence}`.",
        "Spaces and punctuation absent from the 39-label CTC encoder are interpolated and explicitly marked in each diagnostic.",
        "",
        "## Estimators",
        "",
        "The primary estimate is an iterative F0-adaptive cepstral true envelope using a max-and-resmooth loop. The independent cross-check is dynamically ordered Burg LPC. Frames with missing/ambiguous F0, low voicing, probable devoicing, excessive creak, implausible bandwidth, near-Nyquist resonances, estimator disagreement, or trajectory jumps remain in the CSV with rejection reasons.",
        "The supplied formant-shift references are `/e/`; their measured ratios directly validate `/e/` and provisionally bound the global control until Stage B revalidates all five source vowels.",
        "",
        "Roebel and Rodet (DAFx 2005) motivate the F0-adaptive true-envelope order and the requirement not to trace individual harmonics. Hatano et al. (Interspeech 2012) motivate per-vowel validation rather than treating global resonance scaling as exact anatomy.",
        "",
        "## Resulting range",
        "",
        f"- Identity ratio: `{voice_space['identity_vocal_tract_ratio']}`",
        f"- Realistic ratio: `{voice_space['realistic_min_ratio']}` to `{voice_space['realistic_max_ratio']}`",
        f"- Expanded engineering ratio: `{voice_space['expanded_min_ratio']}` to `{voice_space['expanded_max_ratio']}`",
        "",
        "The range is an acoustic engineering range. It is not a male/female threshold and does not claim anatomical tract length.",
        "",
        "## Duration model",
        "",
        f"Fitted `{duration_report.get('model_id')}` from Kokoro training-partition silver labels. Validation and test records were excluded from fitting.",
        "",
        "## Source safety",
        "",
        f"Before: `{source_before['actual_content_inventory_sha256']}`",
        f"After: `{source_after['actual_content_inventory_sha256']}`",
        f"Unchanged: `{source_before['actual_content_inventory_sha256'] == source_after['actual_content_inventory_sha256']}`",
        "",
        "## Stage B boundary",
        "",
        "No production spectral warp exists in Stage A. Stage B may begin only when `stage_a_gate.json` reports `passed: true`; it must then reanalyze final rendered waveforms for measured formant movement, F0 drift, duration drift, vowel identity, creak preservation, joins, clipping, and processing cost.",
        "",
    ])
    path = output / "formant_analysis_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def analyze_stage_a(
    corpus_root: Path,
    alignment_root: Path,
    source_root: Path,
    diphone_index: Path,
    voice_manifest: Path,
    reference_root: Path,
    output: Path,
    *,
    maximum_source_per_vowel: int,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    source_before = _verify_source_bundle(
        source_root, voice_manifest, label="before_stage_a"
    )
    records = load_selection(corpus_root / "partitions.json")
    by_id = {record.utterance_id: record for record in records}
    alignments = {
        record.utterance_id: load_alignment(
            alignment_root / f"{record.utterance_id}.json"
        ) for record in records
    }
    source_segments = source_voice_segments(
        source_root, diphone_index,
        maximum_per_vowel=maximum_source_per_vowel,
    )
    provided_segments = supplied_reference_segments(reference_root)
    kokoro_segments = []
    for record in records:
        alignment = alignments[record.utterance_id]
        if not alignment.accepted:
            continue
        kokoro_segments.extend(segments_from_kokoro_alignment(
            record,
            alignment,
            corpus_root / "wavs" / f"{record.utterance_id}.flac",
        ))
    segments: tuple[AnalysisSegment, ...] = tuple(
        list(source_segments) + list(provided_segments) + kokoro_segments
    )
    _print(
        f"formants: {len(source_segments)} source, "
        f"{len(provided_segments)} supplied-reference, "
        f"{len(kokoro_segments)} Kokoro vowel segments"
    )
    cache = {}
    analyzed = []
    config = FormantAnalysisConfig()
    for index, segment in enumerate(segments, 1):
        path = segment.audio_path.resolve()
        if path not in cache:
            cache[path] = read_audio(
                path,
                expected_sample_rate=(22050 if path.suffix.casefold() == ".flac"
                                      else None),
            )
        analyzed.append(analyze_segment(
            segment, audio=cache[path], config=config
        ))
        if index == 1 or index % 25 == 0 or index == len(segments):
            _print(f"formants {index}/{len(segments)}: {segment.segment_id}")
    frames = [frame for row in analyzed for frame in row.frames]
    write_csv(output / "formant_frames.csv",
              [frame.to_row() for frame in frames])
    write_csv(output / "formant_segments.csv",
              [row.to_row() for row in analyzed])
    manifest = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "kind": "prompt20_reference_manifest",
        "license_notes": {
            "project_source_voicebank": "user-supplied; read-only; not redistributed",
            "prompt20_supplied_references": "user-supplied; redistribution unspecified",
            "Kokoro-Speech-Dataset-v1.3-xlarge": "public domain per upstream repository",
            "Kokoro-Align": "MIT",
        },
        "segments": [segment.to_manifest_dict() for segment in segments],
    }
    _write_json(output / "reference_manifest.json", manifest)
    summary = speaker_formant_summary(analyzed)
    _write_json(output / "speaker_formant_summary.json", summary)
    reference_hashes = {
        path.name: sha256_file(path)
        for path in sorted(reference_root.glob("*")) if path.is_file()
    }
    extraction = json.loads(
        (corpus_root / "extraction_report.json").read_text(encoding="utf-8")
    )
    if extraction.get("archive", {}).get("sha256"):
        reference_hashes[extraction["archive"]["file_name"]] = \
            extraction["archive"]["sha256"]
    voice_space = derive_reference_voice_space(
        analyzed,
        source_speaker_id="project_source_speaker",
        reference_hashes=reference_hashes,
    )
    _write_json(output / "reference_voice_space.json", voice_space)
    plot_paths = write_stage_a_plot_suite(output / "plots", analyzed)
    duration_rows = []
    for record in records:
        alignment = alignments[record.utterance_id]
        if alignment.accepted:
            duration_rows.append((
                record,
                _duration_utterance(
                    record, alignment,
                    corpus_root / "wavs" / f"{record.utterance_id}.flac",
                ),
            ))
    training = [row for record, row in duration_rows
                if record.partition == "train"]
    validation = [row for record, row in duration_rows
                  if record.partition == "validation"]
    testing = [row for record, row in duration_rows
               if record.partition == "test"]
    fit = fit_duration_priors(training, heldout_fraction=0.0)
    write_priors(output / "kokoro_duration_priors.json", fit.priors)
    duration_report = {
        **fit.report,
        "partition_policy": "fit=train only; validation and test held out",
        "validation": evaluate_duration_model(training, validation, fit.priors),
        "test": evaluate_duration_model(training, testing, fit.priors),
    }
    _write_json(output / "kokoro_duration_benchmark.json", duration_report)
    measurements = _kokoro_measurements(
        records, alignments, analyzed
    )
    _write_json(output / "kokoro_measurements.json", measurements)
    source_after = _verify_source_bundle(
        source_root, voice_manifest, label="after_stage_a"
    )
    source_rows = [row for row in analyzed
                   if row.segment.speaker_id == "project_source_speaker"]
    kokoro_rows = [row for row in analyzed
                   if row.segment.source_corpus ==
                   "Kokoro-Speech-Dataset-v1.3-xlarge"]
    supplied_rows = [row for row in analyzed
                     if row.segment.source_corpus ==
                     "prompt20_supplied_references"]
    source_vowels = {row.segment.vowel for row in source_rows if row.accepted}
    kokoro_vowels = {row.segment.vowel for row in kokoro_rows if row.accepted}
    checks = [
        {"name": "source_voice_processed", "passed": bool(source_rows),
         "detail": f"{len(source_rows)} source segments"},
        {"name": "supplied_references_processed", "passed": bool(supplied_rows),
         "detail": f"{len(supplied_rows)} supplied-reference segments"},
        {"name": "inspectable_measurements", "passed": bool(frames),
         "detail": f"{len(frames)} retained frames"},
        {"name": "all_source_vowels_accepted",
         "passed": set("aiueo") <= source_vowels,
         "detail": "accepted=" + ",".join(sorted(source_vowels))},
        {"name": "all_kokoro_vowels_accepted",
         "passed": set("aiueo") <= kokoro_vowels,
         "detail": "accepted=" + ",".join(sorted(kokoro_vowels))},
        {"name": "independent_estimator_comparison",
         "passed": any(any(value is not None for value in
                            frame.estimator_disagreement_hz)
                       for frame in frames),
         "detail": "true-envelope and Burg outputs retained per frame"},
        {"name": "reference_range_derived",
         "passed": (voice_space["realistic_min_ratio"] <= 1.0 <=
                    voice_space["realistic_max_ratio"] and
                    voice_space["expanded_min_ratio"] <=
                    voice_space["realistic_min_ratio"] and
                    voice_space["expanded_max_ratio"] >=
                    voice_space["realistic_max_ratio"]),
         "detail": (f"realistic {voice_space['realistic_min_ratio']}.."
                    f"{voice_space['realistic_max_ratio']}; expanded "
                    f"{voice_space['expanded_min_ratio']}.."
                    f"{voice_space['expanded_max_ratio']}")},
        {"name": "source_bank_unchanged",
         "passed": (source_before["actual_content_inventory_sha256"] ==
                    source_after["actual_content_inventory_sha256"]),
         "detail": source_after["actual_content_inventory_sha256"]},
        {"name": "diagnostic_plots_generated",
         "passed": len(plot_paths) >= 10 and all(path.is_file()
                                                for path in plot_paths),
         "detail": f"{len(plot_paths)} SVG plots"},
        {"name": "heldout_partitions_present",
         "passed": bool(validation) and bool(testing),
         "detail": f"train={len(training)}, validation={len(validation)}, test={len(testing)}"},
    ]
    gate = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "kind": "prompt20_stage_a_gate",
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "production_transform_present": False,
        "stage_b_permitted": all(check["passed"] for check in checks),
    }
    _write_json(output / "stage_a_gate.json", gate)
    _write_json(output / "source_safety.json", {
        "before": source_before,
        "after": source_after,
        "unchanged": source_before["actual_content_inventory_sha256"] ==
                     source_after["actual_content_inventory_sha256"],
    })
    alignment_summary = json.loads(
        (alignment_root / "alignment_summary.json").read_text(encoding="utf-8")
    )
    _write_analysis_report(
        output,
        alignment_summary=alignment_summary,
        summary=summary,
        voice_space=voice_space,
        gate=gate,
        duration_report=duration_report,
        source_before=source_before,
        source_after=source_after,
    )
    if not gate["passed"]:
        failed = [check["name"] for check in checks if not check["passed"]]
        raise RuntimeError("Stage A gate failed: " + ", ".join(failed))
    return gate


def _extract_command(args) -> int:
    extract_kokoro_sample(
        Path(args.archive), Path(args.output),
        candidate_count=args.candidate_count,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
        hash_archive=args.hash_archive,
    )
    return 0


def _align_command(args) -> int:
    align_kokoro_sample(
        Path(args.corpus), Path(args.output),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        minimum_confidence=args.minimum_confidence,
    )
    return 0


def _analyze_command(args) -> int:
    analyze_stage_a(
        Path(args.corpus), Path(args.alignments), Path(args.source_root),
        Path(args.diphone_index), Path(args.voice_manifest),
        Path(args.references), Path(args.output),
        maximum_source_per_vowel=args.maximum_source_per_vowel,
    )
    return 0


def _run_command(args) -> int:
    root = Path(args.output)
    corpus = root / "kokoro_sample"
    alignments = root / "kokoro_alignments"
    extract_kokoro_sample(
        Path(args.archive), corpus,
        candidate_count=args.candidate_count,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
        hash_archive=args.hash_archive,
    )
    align_kokoro_sample(
        corpus, alignments,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        minimum_confidence=args.minimum_confidence,
    )
    gate = analyze_stage_a(
        corpus, alignments, Path(args.source_root), Path(args.diphone_index),
        Path(args.voice_manifest), Path(args.references), root / "analysis",
        maximum_source_per_vowel=args.maximum_source_per_vowel,
    )
    _print(f"Stage A gate passed: {gate['passed']}")
    return 0


def _common_extract(parser):
    parser.add_argument("--archive", required=True)
    parser.add_argument("--candidate-count", type=int, default=800)
    parser.add_argument("--train-count", type=int, default=36)
    parser.add_argument("--validation-count", type=int, default=12)
    parser.add_argument("--test-count", type=int, default=12)
    parser.add_argument("--hash-archive", action=argparse.BooleanOptionalAction,
                        default=True)


def _common_alignment(parser):
    parser.add_argument("--checkpoint")
    parser.add_argument("--minimum-confidence", type=float, default=0.42)


def _common_analysis(parser):
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--diphone-index", required=True)
    parser.add_argument("--voice-manifest", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--maximum-source-per-vowel", type=int, default=24)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prompt 20 safe Kokoro and Stage A formant analysis"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract-kokoro")
    _common_extract(extract)
    extract.add_argument("--output", required=True)
    extract.set_defaults(handler=_extract_command)
    align = subparsers.add_parser("align-kokoro")
    align.add_argument("--corpus", required=True)
    align.add_argument("--output", required=True)
    _common_alignment(align)
    align.set_defaults(handler=_align_command)
    analyze = subparsers.add_parser("analyze-formants")
    analyze.add_argument("--corpus", required=True)
    analyze.add_argument("--alignments", required=True)
    analyze.add_argument("--output", required=True)
    _common_analysis(analyze)
    analyze.set_defaults(handler=_analyze_command)
    run = subparsers.add_parser("run-stage-a")
    _common_extract(run)
    _common_alignment(run)
    _common_analysis(run)
    run.add_argument("--output", required=True)
    run.set_defaults(handler=_run_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
