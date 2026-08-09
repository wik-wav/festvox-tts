# -*- coding: utf-8 -*-
"""Headless bridge from Vocab Forge to the real Festival/WSL backend.

This script is launched with the FestVox virtual-environment interpreter.
It intentionally emits JSON only; the browser-facing server remains free of
NumPy, PyQt, and Festival imports.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np


FESTVOX_DIR = Path(__file__).resolve().parent
GUI_DIR = FESTVOX_DIR / "festvox_gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

import festvox_core as fc  # noqa: E402


def _config(path: str | Path | None = None) -> dict:
    source = Path(path) if path else GUI_DIR / "config.json"
    return fc.load_config(str(source))


def available_voices(
    config_path: str | Path | None = None,
    language: str = "asaxi",
) -> dict:
    cfg = _config(config_path)
    backend = fc.FestivalWSLBackend(cfg)
    try:
        voices = []
        for row in backend.voicebanks():
            name = str(row.get("name") or "")
            compatibility = backend.voice_compatibility(name)
            if not row.get("ok") or not compatibility.supports(language):
                continue
            pitch = backend.voice_pitch_hz(name)
            voices.append({
                "name": name,
                "label": name.replace("_", " "),
                "supported_languages": list(
                    compatibility.supported_languages
                ),
                "primary_language": compatibility.primary_language,
                "default_pitch_hz": (
                    round(float(pitch), 3) if pitch is not None else None
                ),
                "metadata_status": compatibility.metadata_status,
            })
        voices.sort(key=lambda item: (
            item["primary_language"] != language,
            item["name"].casefold(),
        ))
        return {"ok": True, "language": language, "voices": voices}
    finally:
        backend.shutdown()


def _project_row(syn: fc.Synthesis, request: dict) -> dict:
    segments = copy.deepcopy(syn.segments)
    return {
        "text": str(request["text"]),
        "rendered_text": str(syn.text or request["text"]),
        "input_mode": "text",
        "language": "Asaxi",
        "lang_code": "asaxi",
        "engine": "festival_wsl",
        "voicebank": str(request["voicebank"]),
        "speed": float(request["speed"]),
        "pitch_hz": float(request["pitch_hz"]),
        "pitch_manual": False,
        "fall_pct": float(request["fall_pct"]),
        "output_gain_db": 0.0,
        "applied_gain_db": 0.0,
        "pre_gain_peak": (
            float(np.max(np.abs(syn.samples)))
            if np.asarray(syn.samples).size else 0.0
        ),
        "vocal_tract_length_ratio": 1.0,
        "chipmunk_range": False,
        "fault_mode": {},
        "join_settings": {},
        "effective_join_settings": dict(syn.join_settings or {}),
        "parameter_mode": "pitch",
        "view_mode": "speech",
        "needs_rerender": False,
        "needs_generate": False,
        "pending_reason": "",
        "phrases": [],
        "phones": list(syn.phones),
        "render_phones": list(syn.render_phones),
        "special_phone_realizations": [
            dict(item) for item in syn.special_phone_realizations
        ],
        "segments": segments,
        "editor_segments": copy.deepcopy(segments),
        "timing_factors": [1.0] * len(segments),
        "generated_targets": list(syn.generated_targets),
        "pitch_override": list(syn.pitch_override),
        "intonation_blocks": [dict(item) for item in syn.intonation_blocks],
        "pitch_mode": str(syn.pitch_mode or ""),
        "unit_overrides": dict(syn.unit_overrides),
        "selected_units": dict(syn.selected_units),
        "target_pitchmarks": list(syn.target_pitchmarks),
        "splice_records": [dict(item) for item in syn.splice_records],
        "frame_trajectory_records": [
            dict(item) for item in syn.frame_trajectory_records
        ],
        "vowel_realizations": [
            dict(item) for item in syn.vowel_realizations
        ],
        "source_voicing_targets": list(syn.source_voicing_targets),
        "generated_voicing_targets": list(syn.generated_voicing_targets),
        "voicing_override": list(syn.voicing_override),
        "voicing_mode": str(syn.voicing_mode or ""),
        "voicing_diagnostics": [
            dict(item) for item in syn.voicing_diagnostics
        ],
        "vocal_tract_requested_ratio": 1.0,
        "generated_vocal_tract_targets": list(
            syn.generated_vocal_tract_targets
        ),
        "vocal_tract_override": list(syn.vocal_tract_override),
        "applied_vocal_tract_targets": list(
            syn.applied_vocal_tract_targets
        ),
        "vocal_tract_mode": str(syn.vocal_tract_mode or ""),
        "vocal_tract_diagnostics": copy.deepcopy(
            syn.vocal_tract_diagnostics
        ),
        "asaxi_prosody": copy.deepcopy(syn.asaxi_prosody),
        "cache_wav": "cache/sentence_0001.wav",
    }


def render_project(
    project_root: str | Path,
    request: dict,
    config_path: str | Path | None = None,
) -> dict:
    text = str(request.get("text") or "").strip()
    voicebank = str(request.get("voicebank") or "").strip()
    if not text:
        raise ValueError("No Asaxi text was provided.")
    if not voicebank:
        raise ValueError("No FestVox voice was selected.")

    cfg = _config(config_path)
    backend = fc.FestivalWSLBackend(cfg)
    try:
        compatibility = backend.voice_compatibility(voicebank)
        if not compatibility.supports("asaxi"):
            raise ValueError(
                f"{voicebank!r} is not an Asaxi-enabled Festival voice."
            )
        pitch = request.get("pitch_hz")
        if pitch is None:
            pitch = backend.voice_pitch_hz(voicebank)
        if pitch is None:
            pitch = cfg.get("pitch_hz") or 160.0
        speed = float(request.get("speed") or 1.0)
        fall = float(
            request.get("fall_pct")
            if request.get("fall_pct") is not None
            else cfg.get("pitch_fall_pct") or 18.0
        )
        normalized_request = {
            "text": text,
            "voicebank": voicebank,
            "speed": speed,
            "pitch_hz": float(pitch),
            "fall_pct": fall,
        }
        syn = backend.synth(
            text,
            "asaxi",
            voicebank,
            speed=speed,
            pitch=float(pitch),
            fall=fall,
        )
        syn = fc.apply_active_speech_calibration(
            syn,
            fc.generated_voice_output_calibration(
                backend.voice_metadata(voicebank)
            ),
        )
        root = fc.prepare_project_folder(project_root).resolve()
        cache = root / "cache" / "sentence_0001.wav"
        fc.write_wav(cache, syn.samples, syn.sr)
        manifest = fc.save_project_folder(
            root,
            [_project_row(syn, normalized_request)],
            active_sentence=0,
        )
        return {
            "ok": True,
            "project_manifest": str(manifest),
            "voicebank": voicebank,
            "text": text,
            "phones": list(syn.phones),
            "seconds": round(float(syn.duration), 4),
            "warning": str(syn.warning or ""),
            "asaxi_prosody_summary": {
                "phrase_count": int(
                    (syn.asaxi_prosody or {}).get("phrase_count") or 0
                ),
                "word_count": int(
                    (syn.asaxi_prosody or {}).get("word_count") or 0
                ),
                "mora_count": int(
                    (syn.asaxi_prosody or {}).get("mora_count") or 0
                ),
            },
        }
    finally:
        backend.shutdown()


def _read_request(stream=None) -> dict:
    """Decode the browser bridge request independently of console code pages."""

    source = stream if stream is not None else sys.stdin.buffer
    raw = source.read()
    if isinstance(raw, str):
        text = raw
    else:
        text = bytes(raw).decode("utf-8-sig")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("The FestVox bridge request must be a JSON object.")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(GUI_DIR / "config.json"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    voices = subparsers.add_parser("voices")
    voices.add_argument("--language", default="asaxi")
    render = subparsers.add_parser("render-project")
    render.add_argument("--project", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "voices":
            result = available_voices(args.config, args.language)
        else:
            request = _read_request()
            result = render_project(args.project, request, args.config)
        print(json.dumps(result, ensure_ascii=True))
        return 0
    except Exception as error:
        print(json.dumps({
            "ok": False,
            "error": str(error),
            "error_type": type(error).__name__,
        }, ensure_ascii=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
