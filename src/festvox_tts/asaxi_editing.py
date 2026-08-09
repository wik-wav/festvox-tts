# -*- coding: utf-8 -*-
"""Persistent, language-specific edit state for Asaxi synthesis.

The dictionary/morphology analyzer remains the source of the automatic H/L
contour. This module stores only user overlays. Continuous Pitch and Voicing
curves are intentionally outside this state and remain the final authorities.
"""

from __future__ import annotations

import copy
import math
import unicodedata
from typing import Mapping


SCHEMA_VERSION = 2
PITCH_OFFSET_MIN_CENTS = -1200
PITCH_OFFSET_MAX_CENTS = 1200
BREATHINESS_HARMONIC_DEPTH = 0.48


def new_edit_state(source_text: str = "") -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_text": str(source_text or ""),
        "last_plan": {},
        "mora_tone_overrides": {},
        "mora_pitch_offsets_cents": {},
        "mora_voicing_overrides": {},
    }


def _normal_text(value: object) -> str:
    return unicodedata.normalize("NFC", str(value or "").strip().lower())


def _number_map(value: object, *, integer: bool, lower: float,
                upper: float, keep_zero: bool = True) -> dict[str, object]:
    result: dict[str, object] = {}
    if not isinstance(value, Mapping):
        return result
    for raw_key, raw_value in value.items():
        try:
            key = str(int(raw_key))
            number = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        number = max(lower, min(upper, number))
        if integer:
            number = int(round(number))
        if keep_zero or abs(float(number)) > 1.0e-9:
            result[key] = number
    return result


def _tone_map(value: object) -> dict[str, str]:
    result: dict[str, str] = {}
    if not isinstance(value, Mapping):
        return result
    for raw_key, raw_value in value.items():
        try:
            key = str(int(raw_key))
        except (TypeError, ValueError):
            continue
        tone = str(raw_value or "").strip().upper()
        if tone in {"H", "L"}:
            result[key] = tone
    return result


def normalize_edit_state(value: object) -> dict[str, object]:
    source = dict(value) if isinstance(value, Mapping) else {}
    state = new_edit_state(str(source.get("source_text") or ""))
    plan = source.get("last_plan")
    state["last_plan"] = copy.deepcopy(
        dict(plan) if isinstance(plan, Mapping) else {}
    )
    state["mora_tone_overrides"] = _tone_map(
        source.get("mora_tone_overrides")
    )
    state["mora_pitch_offsets_cents"] = _number_map(
        source.get("mora_pitch_offsets_cents"),
        integer=True,
        lower=PITCH_OFFSET_MIN_CENTS,
        upper=PITCH_OFFSET_MAX_CENTS,
        keep_zero=False,
    )
    state["mora_voicing_overrides"] = _number_map(
        source.get("mora_voicing_overrides"),
        integer=False,
        lower=0.0,
        upper=1.0,
    )
    legacy_breathiness = _number_map(
        source.get("mora_breathiness_overrides"),
        integer=False,
        lower=0.0,
        upper=1.0,
    )
    # Version 1 exposed breathiness as a second manual dimension.  The shared
    # Mora voicing editor now has one final harmonic-share control, so retain
    # the old audible intent by folding that ceiling into its voicing map.
    for key, amount in legacy_breathiness.items():
        ceiling = 1.0 - float(amount) * BREATHINESS_HARMONIC_DEPTH
        state["mora_voicing_overrides"][key] = min(
            float(state["mora_voicing_overrides"].get(key, 1.0)),
            max(0.0, min(1.0, ceiling)),
        )
    return state


def mora_rows(metadata: object) -> list[dict[str, object]]:
    """Return sentence-level mora rows from current or phrase-sequence data."""

    if not isinstance(metadata, Mapping):
        return []
    direct = metadata.get("moras")
    if isinstance(direct, list):
        return [dict(row) for row in direct if isinstance(row, Mapping)]
    result: list[dict[str, object]] = []
    for phrase in metadata.get("phrases") or []:
        if not isinstance(phrase, Mapping):
            continue
        nested = mora_rows(phrase)
        result.extend(nested)
    return result


def _refined_mora_index_map(
    old_metadata: object,
    new_metadata: object,
) -> dict[str, tuple[str, ...]] | None:
    """Map old indices through an exact per-word mora-boundary refinement."""

    old_rows = mora_rows(old_metadata)
    new_rows = mora_rows(new_metadata)
    required = {
        "mora_index", "phrase_index", "word_index", "word", "text",
    }
    if (
        not old_rows
        or not new_rows
        or any(not required.issubset(row) for row in old_rows + new_rows)
    ):
        return None

    def grouped(rows):
        result: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for row in rows:
            key = (
                int(row["phrase_index"]),
                int(row["word_index"]),
                _normal_text(row["word"]),
            )
            result.setdefault(key, []).append(row)
        return result

    old_groups = grouped(old_rows)
    new_groups = grouped(new_rows)
    if old_groups.keys() != new_groups.keys():
        return None

    mapping: dict[str, tuple[str, ...]] = {}
    for key, previous in old_groups.items():
        current = new_groups[key]
        cursor = 0
        for old_row in previous:
            target = _normal_text(old_row["text"])
            combined = ""
            indices = []
            while cursor < len(current) and len(combined) < len(target):
                combined += _normal_text(current[cursor]["text"])
                indices.append(str(int(current[cursor]["mora_index"])))
                cursor += 1
            if not indices or combined != target:
                return None
            mapping[str(int(old_row["mora_index"]))] = tuple(indices)
        if cursor != len(current):
            return None
    return mapping


def reconcile_plan(
    value: object,
    source_text: str,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Attach fresh analysis and retain overlays only for unchanged text."""

    state = normalize_edit_state(value)
    same_text = (
        _normal_text(state.get("source_text")) == _normal_text(source_text)
    )
    if not same_text:
        state["mora_tone_overrides"] = {}
        state["mora_pitch_offsets_cents"] = {}
        state["mora_voicing_overrides"] = {}
    remap = (
        _refined_mora_index_map(state.get("last_plan"), metadata)
        if same_text else None
    )
    state["source_text"] = str(source_text or "")
    state["last_plan"] = copy.deepcopy(dict(metadata or {}))
    valid = {
        str(int(row["mora_index"]))
        for row in mora_rows(metadata)
        if row.get("mora_index") is not None
    }
    for key in (
        "mora_tone_overrides",
        "mora_pitch_offsets_cents",
        "mora_voicing_overrides",
    ):
        current = dict(state.get(key) or {})
        if remap is not None:
            migrated = {}
            for old_index, amount in current.items():
                for new_index in remap.get(old_index, ()):
                    if new_index in valid:
                        migrated[new_index] = amount
            state[key] = migrated
        else:
            state[key] = {
                index: amount
                for index, amount in current.items()
                if index in valid
            }
    return normalize_edit_state(state)


def with_mora_edit(
    value: object,
    kind: str,
    mora_indices,
    amount,
) -> dict[str, object]:
    state = normalize_edit_state(value)
    field = {
        "tone": "mora_tone_overrides",
        "pitch": "mora_pitch_offsets_cents",
        "voicing": "mora_voicing_overrides",
    }.get(str(kind))
    if field is None:
        raise ValueError(f"Unknown Asaxi mora edit: {kind!r}")
    target = dict(state[field])
    if isinstance(mora_indices, (str, int)):
        mora_indices = [mora_indices]
    for raw_index in mora_indices or []:
        index = str(int(raw_index))
        if amount is None:
            target.pop(index, None)
        elif field == "mora_tone_overrides":
            tone = str(amount or "").strip().upper()
            if tone not in {"H", "L"}:
                raise ValueError(f"invalid Asaxi mora tone: {amount!r}")
            target[index] = tone
        elif field == "mora_pitch_offsets_cents":
            target[index] = max(
                PITCH_OFFSET_MIN_CENTS,
                min(PITCH_OFFSET_MAX_CENTS, int(round(float(amount)))),
            )
            if target[index] == 0:
                target.pop(index)
        else:
            target[index] = max(0.0, min(1.0, float(amount)))
    state[field] = target
    return normalize_edit_state(state)
