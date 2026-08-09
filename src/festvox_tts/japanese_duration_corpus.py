"""Offline Japanese duration-corpus fitting and evaluation.

Runtime synthesis never imports this module.  Corpus timings are used to fit
dimensionless context residuals; the generated voice's selected source-unit
geometry remains the absolute duration baseline in :mod:`japanese_duration`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Iterable, Mapping, Sequence
import wave

import numpy as np

from japanese_devoicing import periodicity_score
from japanese_duration import (
    DEFAULT_PRIORS_PATH,
    JapaneseDurationPriors,
    load_duration_priors,
    phone_class,
)


DEFAULT_VALIDATION_PATH = (
    Path(__file__).resolve().parent
    / "profiles" / "japanese_duration_validation_v1.json"
)


def load_fixed_validation_ids(path: Path | str | None = None) \
        -> tuple[str, ...]:
    source = Path(path) if path is not None else DEFAULT_VALIDATION_PATH
    data = json.loads(source.read_text(encoding="utf-8"))
    if int(data.get("schema_version") or 0) != 1:
        raise ValueError("unsupported Japanese duration validation schema")
    identifiers = tuple(str(item) for item in
                        (data.get("fixed_jsut_ids") or ()))
    if not identifiers or any(not item for item in identifiers):
        raise ValueError("fixed Japanese validation IDs must be nonempty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate fixed Japanese validation ID")
    return identifiers


FIXED_VALIDATION_IDS = load_fixed_validation_ids()

_SILENCE = {"sil", "pau", "sp"}
_PHRASE_BOUNDARY = {"sil", "pau"}
_VOWELS = {"a", "i", "u", "e", "o"}
_FIT_FEATURES = (
    "devoiced_high_vowel",
    "geminate_closure",
    "long_vowel",
    "moraic_nasal",
    "phrase_final_rhyme",
    "utterance_final_rhyme",
)
_FIT_COEFFICIENT_BOUNDS = {
    "devoiced_high_vowel": (-0.70, 0.10),
    "geminate_closure": (-0.25, 0.25),
    "long_vowel": (-0.20, 0.30),
    "moraic_nasal": (-0.25, 0.25),
    # Phrase-final and utterance-final flags overlap heavily in short corpus
    # sentences. Feature-specific bounds prevent a numerically cancelling
    # +0.65/-0.65 pair from entering production.
    "phrase_final_rhyme": (-0.15, 0.25),
    "utterance_final_rhyme": (-0.15, 0.15),
}
_FULL_CONTEXT_PHONE = re.compile(r"^[^^]+\^[^-]+-([^+]+)\+")


@dataclass(frozen=True)
class TimedPhone:
    start_100ns: int
    end_100ns: int
    phone: str
    raw_label: str
    devoiced: bool = False

    @property
    def start_seconds(self) -> float:
        return self.start_100ns / 10_000_000.0

    @property
    def end_seconds(self) -> float:
        return self.end_100ns / 10_000_000.0

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.end_100ns - self.start_100ns) / 10_000_000.0)

    @property
    def duration_ms(self) -> float:
        # HTK labels use 100-nanosecond units.
        return max(0.0, (self.end_100ns - self.start_100ns) / 10_000.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "start_100ns": self.start_100ns,
            "end_100ns": self.end_100ns,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_ms": self.duration_ms,
            "phone": self.phone,
            "devoiced": self.devoiced,
            "raw_label": self.raw_label,
        }


@dataclass(frozen=True)
class CorpusUtterance:
    utterance_id: str
    phones: tuple[TimedPhone, ...]
    label_path: str = ""
    wav_path: str = ""
    corpus: str = "jsut"
    diagnostics: tuple[str, ...] = ()

    @property
    def spoken_phones(self) -> tuple[TimedPhone, ...]:
        return tuple(item for item in self.phones if item.phone not in _SILENCE)

    def to_dict(self) -> dict[str, object]:
        return {
            "utterance_id": self.utterance_id,
            "corpus": self.corpus,
            "phones": [item.to_dict() for item in self.phones],
            "diagnostics": list(self.diagnostics),
            # Reports deliberately retain only path-neutral names.
            "label_name": Path(self.label_path).name if self.label_path else "",
            "wav_name": Path(self.wav_path).name if self.wav_path else "",
        }


@dataclass(frozen=True)
class AlignmentResult:
    pairs: tuple[tuple[int | None, int | None], ...]
    cost: int
    diagnostics: tuple[str, ...] = ()

    @property
    def exact(self) -> bool:
        return self.cost == 0


@dataclass
class DurationFitResult:
    priors: JapaneseDurationPriors
    report: dict[str, object]
    phone_medians: dict[str, float] = field(default_factory=dict)


def normalize_label_phone(label: str) -> tuple[str, bool]:
    raw = str(label).strip()
    match = _FULL_CONTEXT_PHONE.search(raw)
    phone = match.group(1) if match else raw.split()[0]
    phone = phone.strip()
    if phone in {"q", "Q", "っ"}:
        return "cl", False
    if phone.casefold() in {"sil", "pau", "sp"}:
        return phone.casefold(), False
    if phone in {"A", "I", "U", "E", "O"}:
        return phone.casefold(), True
    if phone in {"N", "cl"}:
        return phone, False
    return phone.casefold(), False


def parse_htk_label_line(line: str, *, line_number: int = 0) -> TimedPhone:
    fields = str(line).strip().split(maxsplit=2)
    if len(fields) != 3:
        raise ValueError(
            f"HTK label line {line_number or '?'} needs start, end, and label"
        )
    try:
        start = int(fields[0])
        end = int(fields[1])
    except ValueError as error:
        raise ValueError(
            f"HTK label line {line_number or '?'} has non-integer timestamps"
        ) from error
    if start < 0 or end <= start:
        raise ValueError(
            f"HTK label line {line_number or '?'} has an invalid interval"
        )
    phone, devoiced = normalize_label_phone(fields[2])
    if not phone:
        raise ValueError(f"HTK label line {line_number or '?'} has no phone")
    return TimedPhone(start, end, phone, fields[2], devoiced)


def parse_htk_lab(path: Path | str, *, utterance_id: str | None = None,
                  corpus: str = "jsut") -> CorpusUtterance:
    source = Path(path)
    rows = []
    for line_number, line in enumerate(
            source.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        rows.append(parse_htk_label_line(line, line_number=line_number))
    if not rows:
        raise ValueError(f"label file contains no intervals: {source}")
    previous = -1
    for row in rows:
        if row.start_100ns < previous:
            raise ValueError(f"label intervals are not ordered: {source}")
        previous = row.end_100ns
    return CorpusUtterance(
        utterance_id=utterance_id or source.stem,
        phones=tuple(rows),
        label_path=str(source),
        wav_path=str(_find_wav(source) or ""),
        corpus=corpus,
    )


def _find_wav(label: Path) -> Path | None:
    direct = label.with_suffix(".wav")
    if direct.is_file():
        return direct
    candidates = sorted(label.parent.parent.glob(f"**/{label.stem}.wav"))
    return candidates[0] if candidates else None


def discover_jsut_labels(root: Path | str) -> list[Path]:
    source = Path(root)
    if not source.is_dir():
        raise FileNotFoundError(
            f"JSUT root does not exist: {source}. Supply an extracted dataset path."
        )
    labels = sorted(source.rglob("*.lab"), key=lambda item: item.as_posix())
    if not labels:
        raise FileNotFoundError(
            f"no .lab files were found under JSUT root: {source}"
        )
    return labels


def load_jsut(root: Path | str) -> tuple[CorpusUtterance, ...]:
    return tuple(parse_htk_lab(path, corpus="jsut")
                 for path in discover_jsut_labels(root))


def parse_textgrid(path: Path | str, *, utterance_id: str | None = None) \
        -> CorpusUtterance:
    """Parse a conventional long TextGrid phone tier without a dependency."""
    source = Path(path)
    lines = source.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    rows = []
    current: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("intervals ["):
            if {"xmin", "xmax", "text"} <= current.keys():
                rows.append(current)
            current = {}
        elif " = " in stripped and stripped.split(" = ", 1)[0] in {
                "xmin", "xmax", "text"}:
            key, value = stripped.split(" = ", 1)
            current[key] = value.strip().strip('"')
    if {"xmin", "xmax", "text"} <= current.keys():
        rows.append(current)
    phones = []
    for row in rows:
        if not row["text"].strip():
            continue
        start = int(round(float(row["xmin"]) * 10_000_000))
        end = int(round(float(row["xmax"]) * 10_000_000))
        phone, devoiced = normalize_label_phone(row["text"])
        if end > start:
            phones.append(TimedPhone(
                start, end, phone, row["text"], devoiced
            ))
    if not phones:
        raise ValueError(f"no labeled phone intervals found in TextGrid: {source}")
    return CorpusUtterance(
        utterance_id=utterance_id or source.stem,
        phones=tuple(phones),
        label_path=str(source),
        corpus="csj-core",
    )


def load_csj(root: Path | str | None) -> tuple[CorpusUtterance, ...]:
    if not root:
        return ()
    source = Path(root)
    if not source.is_dir():
        raise FileNotFoundError(
            f"CSJ root does not exist: {source}; omit --csj when unavailable"
        )
    paths = sorted(source.rglob("*.TextGrid"), key=lambda item: item.as_posix())
    if not paths:
        raise FileNotFoundError(f"no TextGrid files found under CSJ root: {source}")
    return tuple(parse_textgrid(path) for path in paths)


def align_phone_sequences(reference: Sequence[str], predicted: Sequence[str]) \
        -> AlignmentResult:
    left = [normalize_label_phone(item)[0] for item in reference]
    right = [normalize_label_phone(item)[0] for item in predicted]
    rows, cols = len(left) + 1, len(right) + 1
    costs = np.zeros((rows, cols), np.int32)
    back: dict[tuple[int, int], tuple[int, int]] = {}
    costs[:, 0] = np.arange(rows)
    costs[0, :] = np.arange(cols)
    for i in range(1, rows):
        back[(i, 0)] = (i - 1, 0)
    for j in range(1, cols):
        back[(0, j)] = (0, j - 1)
    for i in range(1, rows):
        for j in range(1, cols):
            choices = (
                (int(costs[i - 1, j - 1]) + (left[i - 1] != right[j - 1]),
                 i - 1, j - 1),
                (int(costs[i - 1, j]) + 1, i - 1, j),
                (int(costs[i, j - 1]) + 1, i, j - 1),
            )
            cost, pi, pj = min(choices, key=lambda item: item)
            costs[i, j] = cost
            back[(i, j)] = (pi, pj)
    pairs = []
    i, j = len(left), len(right)
    while i or j:
        pi, pj = back[(i, j)]
        pairs.append((i - 1 if pi < i else None,
                      j - 1 if pj < j else None))
        i, j = pi, pj
    pairs.reverse()
    cost = int(costs[-1, -1])
    diagnostics = (() if cost == 0 else (
        f"phone alignment required {cost} insertion/deletion/substitution edits",
    ))
    return AlignmentResult(tuple(pairs), cost, diagnostics)


def _features(phones: Sequence[TimedPhone], index: int) -> dict[str, float]:
    row = phones[index]
    previous = phones[index - 1].phone if index else None
    following = next(
        (item.phone for item in phones[index + 1:] if item.phone != "sp"),
        None,
    )
    is_rhyme = row.phone in _VOWELS or row.phone == "N"
    previous_spoken = next((item.phone for item in reversed(phones[:index])
                            if item.phone not in _SILENCE), None)
    next_spoken = next((item.phone for item in phones[index + 1:]
                        if item.phone not in _SILENCE), None)
    return {
        "devoiced_high_vowel": float(row.devoiced and row.phone in {"i", "u"}),
        "geminate_closure": float(row.phone == "cl"),
        "long_vowel": float(row.phone in _VOWELS and previous_spoken == row.phone),
        "moraic_nasal": float(row.phone == "N"),
        "phrase_final_rhyme": float(
            is_rhyme and following in _PHRASE_BOUNDARY
        ),
        "utterance_final_rhyme": float(is_rhyme and next_spoken is None),
    }


def utterance_phenomena(utterance: CorpusUtterance) -> tuple[str, ...]:
    names = set()
    for index, row in enumerate(utterance.phones):
        for name, value in _features(utterance.phones, index).items():
            if value:
                names.add(name)
        if phone_class(row.phone) in {"stop", "fricative"}:
            names.add(phone_class(row.phone))
    return tuple(sorted(names))


def select_heldout_ids(utterances: Sequence[CorpusUtterance], fraction=0.15) \
        -> tuple[str, ...]:
    eligible = [item for item in utterances
                if item.utterance_id not in FIXED_VALIDATION_IDS]
    count = max(1, int(round(len(eligible) * max(0.0, min(0.5, fraction))))) \
        if eligible else 0
    ordered = sorted(
        eligible,
        key=lambda item: hashlib.sha256(
            item.utterance_id.encode("utf-8")
        ).hexdigest(),
    )
    chosen = {item.utterance_id for item in ordered[:count]}
    # Ensure every observed duration phenomenon has at least one deterministic
    # held-out utterance, without ever moving a fixed validation item to train.
    for phenomenon in _FIT_FEATURES:
        candidates = [item for item in ordered
                      if phenomenon in utterance_phenomena(item)]
        if candidates and not any(item.utterance_id in chosen
                                  for item in candidates):
            chosen.add(candidates[0].utterance_id)
    return tuple(sorted(chosen))


def _median_mad(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    return median, mad


def _training_rows(utterances: Sequence[CorpusUtterance]):
    durations: dict[str, list[float]] = {}
    for utterance in utterances:
        for row in utterance.spoken_phones:
            if 0.010 <= row.duration_seconds <= 0.500:
                durations.setdefault(row.phone, []).append(row.duration_seconds)
    medians = {phone: _median_mad(values)[0]
               for phone, values in sorted(durations.items())}
    rows = []
    for utterance in utterances:
        valid = [item for item in utterance.phones
                 if item.phone not in _SILENCE
                 and item.phone in medians
                 and 0.010 <= item.duration_seconds <= 0.500]
        raw = [math.log(item.duration_seconds / medians[item.phone])
               for item in valid]
        utterance_rate = float(statistics.median(raw)) if raw else 0.0
        for index, item in enumerate(utterance.phones):
            if item not in valid:
                continue
            rows.append({
                "utterance_id": utterance.utterance_id,
                "phone": item.phone,
                "duration": item.duration_seconds,
                "target": math.log(item.duration_seconds / medians[item.phone])
                          - utterance_rate,
                "features": _features(utterance.phones, index),
            })
    return rows, medians, durations


def _robust_fit(matrix: np.ndarray, target: np.ndarray, ridge=0.05) \
        -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(matrix.shape[1] if matrix.ndim == 2 else 0)
    weights = np.ones(len(target), np.float64)
    coefficients = np.zeros(matrix.shape[1], np.float64)
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * float(ridge)
    for _ in range(8):
        weighted = matrix * np.sqrt(weights)[:, None]
        response = target * np.sqrt(weights)
        coefficients = np.linalg.solve(
            weighted.T @ weighted + penalty,
            weighted.T @ response,
        )
        residual = target - matrix @ coefficients
        _center, mad = _median_mad(residual.tolist())
        scale = max(1e-4, 1.4826 * mad)
        threshold = 1.5 * scale
        weights = np.minimum(1.0, threshold / (np.abs(residual) + 1e-12))
    return np.clip(coefficients, -0.65, 0.65)


def fit_duration_priors(
    utterances: Sequence[CorpusUtterance],
    *,
    heldout_fraction: float = 0.15,
    seed_priors: JapaneseDurationPriors | None = None,
) -> DurationFitResult:
    fixed = {item.utterance_id for item in utterances
             if item.utterance_id in FIXED_VALIDATION_IDS}
    heldout = set(select_heldout_ids(utterances, heldout_fraction))
    train = [item for item in utterances
             if item.utterance_id not in fixed | heldout]
    if not train:
        raise ValueError("no training utterances remain after validation exclusion")
    rows, medians, distributions = _training_rows(train)
    if not rows:
        raise ValueError("training labels contain no plausible spoken intervals")
    matrix = np.asarray([
        [float(row["features"].get(name, 0.0)) for name in _FIT_FEATURES]
        for row in rows
    ], np.float64)
    target = np.asarray([float(row["target"]) for row in rows], np.float64)
    fitted = _robust_fit(matrix, target)
    seed = seed_priors or load_duration_priors()
    coefficients = dict(seed.coefficients)
    coefficients.update({
        name: round(float(max(
            _FIT_COEFFICIENT_BOUNDS[name][0],
            min(_FIT_COEFFICIENT_BOUNDS[name][1], value),
        )), 9)
        for name, value in zip(_FIT_FEATURES, fitted)
    })
    class_distributions: dict[str, list[float]] = {}
    for phone, values in distributions.items():
        class_distributions.setdefault(phone_class(phone), []).extend(values)
    observed_class_medians = {
        name: _median_mad(values)[0]
        for name, values in sorted(class_distributions.items()) if values
    }
    anchor_class = "vowel"
    anchor_observed = observed_class_medians.get(anchor_class)
    anchor_target = seed.class_reference_seconds.get(anchor_class)
    class_scale = (
        float(anchor_target) / float(anchor_observed)
        if anchor_observed and anchor_target else 1.0
    )
    class_references = dict(seed.class_reference_seconds)
    class_references.update({
        name: round(float(value) * class_scale, 9)
        for name, value in observed_class_medians.items()
    })
    file_ids = [item.utterance_id for item in sorted(
        train, key=lambda value: value.utterance_id)]
    corpora = sorted({item.corpus for item in train})
    corpus_tag = (
        "kokoro" if len(corpora) == 1 and "kokoro" in corpora[0].casefold()
        else "jsut" if len(corpora) == 1 and "jsut" in corpora[0].casefold()
        else "csj" if len(corpora) == 1 and "csj" in corpora[0].casefold()
        else "mixed"
    )
    fingerprint = hashlib.sha256(
        "\n".join(file_ids).encode("utf-8")
    ).hexdigest()[:12]
    priors = JapaneseDurationPriors(
        model_id=(f"japanese_contextual_source_residual_{corpus_tag}_"
                  f"{fingerprint}_v2"),
        schema_version=seed.schema_version,
        coefficients=coefficients,
        speed_elasticity=dict(seed.speed_elasticity),
        class_reference_seconds=class_references,
        minimum_context_ratio=seed.minimum_context_ratio,
        maximum_context_ratio=seed.maximum_context_ratio,
        cv_compensation_fraction=seed.cv_compensation_fraction,
        source_geometry_ratio_bounds=dict(
            seed.source_geometry_ratio_bounds
        ),
        class_target_ratio_bounds=dict(seed.class_target_ratio_bounds),
        # Corpus fitting updates normalized contextual residuals and
        # phone-class allocation, but must retain the project speaker's
        # absolute mora timing anchors.  These became required profile fields
        # in Prompt 20 and are intentionally copied from the seed profile.
        mora_allocation_seconds=dict(seed.mora_allocation_seconds),
        mora_anchor_seconds=dict(seed.mora_anchor_seconds),
        phrase_pauses_ms=dict(seed.phrase_pauses_ms),
        fit_provenance={
            "absolute_scale": "selected_source_unit_geometry",
            "corpus_speaker_scale_used": False,
            "corpus": corpora,
            "feature_schema_version": 1,
            "fit_method": "deterministic_huber_ridge_log_residual",
            "class_duration_normalization": {
                "method": "Kokoro class medians scaled to project vowel anchor",
                "anchor_class": anchor_class,
                "anchor_seconds": anchor_target,
                "corpus_speaker_scale_used": False,
                "scale": round(class_scale, 9),
            },
            "training_utterance_count": len(train),
            "training_phone_count": len(rows),
            "training_ids_sha256_12": fingerprint,
            "fixed_validation_exclusions": sorted(fixed),
            "heldout_exclusions": sorted(heldout),
            "source_file_ids": file_ids,
            "timestamp_note": "No wall-clock field; identical inputs are byte-stable.",
        },
    )
    phone_stats = {
        phone: {
            "count": len(values),
            "median_seconds": round(_median_mad(values)[0], 9),
            "mad_seconds": round(_median_mad(values)[1], 9),
        }
        for phone, values in sorted(distributions.items())
    }
    report = {
        "schema_version": 1,
        "kind": "japanese_duration_fit",
        "model_id": priors.model_id,
        "training_utterance_count": len(train),
        "training_phone_count": len(rows),
        "fixed_validation_ids": list(FIXED_VALIDATION_IDS),
        "fixed_validation_present": sorted(fixed),
        "heldout_ids": sorted(heldout),
        "coefficients": {name: coefficients[name] for name in _FIT_FEATURES},
        "class_reference_seconds": class_references,
        "phone_statistics": phone_stats,
    }
    return DurationFitResult(priors, report, medians)


def _rank(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(order), np.float64)
    start = 0
    data = np.asarray(values, np.float64)
    while start < len(order):
        end = start + 1
        while end < len(order) and data[order[end]] == data[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float], *, rank=False):
    if len(left) < 2:
        return None
    a = _rank(left) if rank else np.asarray(left, np.float64)
    b = _rank(right) if rank else np.asarray(right, np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def duration_metrics(reference: Sequence[float], predicted: Sequence[float]) \
        -> dict[str, float | int | None]:
    if len(reference) != len(predicted):
        raise ValueError("reference and prediction lengths differ")
    if not reference:
        return {"count": 0, "mae_ms": None, "rmse_ms": None,
                "log_rmse": None, "median_absolute_percentage_error": None,
                "pearson": None, "spearman": None}
    actual = np.asarray(reference, np.float64)
    estimate = np.asarray(predicted, np.float64)
    error = estimate - actual
    usable = actual > 1e-5
    return {
        "count": len(actual),
        "mae_ms": round(float(np.mean(np.abs(error))) * 1000.0, 6),
        "rmse_ms": round(float(np.sqrt(np.mean(error * error))) * 1000.0, 6),
        "log_rmse": round(float(np.sqrt(np.mean(
            (np.log(np.maximum(estimate, 1e-5))
             - np.log(np.maximum(actual, 1e-5))) ** 2
        ))), 9),
        "median_absolute_percentage_error": round(float(np.median(
            np.abs(error[usable]) / actual[usable]
        )) * 100.0, 6) if np.any(usable) else None,
        "pearson": _correlation(actual, estimate),
        "spearman": _correlation(actual, estimate, rank=True),
    }


def _predict_rows(utterances: Sequence[CorpusUtterance], medians,
                  priors: JapaneseDurationPriors):
    rows = []
    for utterance in utterances:
        phrase_index = 0
        mora_index = 0
        for index, item in enumerate(utterance.phones):
            if item.phone in _SILENCE:
                if (item.phone in _PHRASE_BOUNDARY and rows and
                        rows[-1].get("utterance_id") ==
                        utterance.utterance_id):
                    phrase_index += 1
                continue
            if item.phone in _SILENCE or not 0.005 <= item.duration_seconds <= 0.8:
                continue
            baseline = medians.get(item.phone)
            if baseline is None:
                baseline = priors.class_reference_seconds.get(
                    phone_class(item.phone), 0.070
                )
            features = _features(utterance.phones, index)
            effect = sum(float(priors.coefficients.get(name, 0.0)) * value
                         for name, value in features.items())
            effect = max(math.log(priors.minimum_context_ratio),
                         min(math.log(priors.maximum_context_ratio), effect))
            rows.append({
                "utterance_id": utterance.utterance_id,
                "phone_index": index,
                "phrase_index": phrase_index,
                "mora_index": mora_index,
                "phone": item.phone,
                "phone_class": phone_class(item.phone),
                "reference": item.duration_seconds,
                "legacy": float(baseline),
                "contextual": float(baseline) * math.exp(effect),
                "phenomena": [name for name, value in features.items() if value],
                "devoiced": item.devoiced,
            })
            if item.phone in _VOWELS or item.phone in {"N", "cl"}:
                mora_index += 1
    return rows


def _grouped_metrics(rows: Sequence[Mapping[str, object]], keys: Sequence[str],
                     system: str) -> dict[str, float | int | None]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        grouped.setdefault(key, []).append(row)
    reference = [sum(float(item["reference"]) for item in group)
                 for group in grouped.values()]
    predicted = [sum(float(item[system]) for item in group)
                 for group in grouped.values()]
    return duration_metrics(reference, predicted)


def _rate_normalized_log_rmse(rows: Sequence[Mapping[str, object]],
                              system: str) -> float | None:
    residuals = []
    utterance_ids = sorted({str(row["utterance_id"]) for row in rows})
    for utterance_id in utterance_ids:
        selected = [row for row in rows
                    if str(row["utterance_id"]) == utterance_id]
        raw = [math.log(max(1e-5, float(row[system])))
               - math.log(max(1e-5, float(row["reference"])))
               for row in selected]
        center = float(statistics.median(raw)) if raw else 0.0
        residuals.extend(value - center for value in raw)
    return (round(float(np.sqrt(np.mean(np.square(residuals)))), 9)
            if residuals else None)


def _contrast_statistics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    categories = {
        "long_vowel": lambda row: "long_vowel" in row["phenomena"],
        "short_vowel": lambda row: (row["phone"] in _VOWELS and
                                      "long_vowel" not in row["phenomena"]),
        "geminate_closure": lambda row: row["phone"] == "cl",
        "singleton_consonant": lambda row: row["phone_class"] in {
            "stop", "fricative", "affricate"},
        "moraic_nasal": lambda row: row["phone"] == "N",
        "devoiced_high_vowel": lambda row: bool(row["devoiced"]),
        "voiced_high_vowel": lambda row: (row["phone"] in {"i", "u"}
                                            and not row["devoiced"]),
        "phrase_final_rhyme": lambda row: (
            "phrase_final_rhyme" in row["phenomena"]),
        "phrase_medial_vowel": lambda row: (row["phone"] in _VOWELS and
                                              "phrase_final_rhyme"
                                              not in row["phenomena"]),
    }
    result = {}
    for name, predicate in categories.items():
        selected = [row for row in rows if predicate(row)]
        result[name] = {
            "count": len(selected),
            "reference_median_ms": (round(float(statistics.median(
                float(row["reference"]) for row in selected)) * 1000.0, 6)
                if selected else None),
            "legacy_median_ms": (round(float(statistics.median(
                float(row["legacy"]) for row in selected)) * 1000.0, 6)
                if selected else None),
            "contextual_median_ms": (round(float(statistics.median(
                float(row["contextual"]) for row in selected)) * 1000.0, 6)
                if selected else None),
        }
    return result


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    source = Path(path)
    if source.suffix.casefold() != ".wav":
        # Stage A already has a deterministic ffmpeg-backed FLAC reader. Use
        # the same path so held-out devoicing evaluation is not silently empty.
        from formant_analysis import read_audio
        audio = read_audio(source, expected_sample_rate=22050)
        return np.asarray(audio.samples, np.float32), int(audio.sample_rate)
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        data = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"periodicity evaluation expects 16-bit PCM: {path}")
    values = np.frombuffer(data, "<i2").astype(np.float32) / 32768.0
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    return values, rate


def evaluate_voicing(utterances: Sequence[CorpusUtterance]) -> dict[str, object]:
    rows = []
    missing = 0
    for utterance in utterances:
        if not utterance.wav_path or not Path(utterance.wav_path).is_file():
            missing += 1
            continue
        try:
            samples, rate = _read_wav_mono(utterance.wav_path)
        except (OSError, RuntimeError, ValueError, wave.Error):
            missing += 1
            continue
        for item in utterance.phones:
            if item.phone not in {"i", "u"}:
                continue
            start = max(0, int(round(item.start_seconds * rate)))
            end = min(len(samples), int(round(item.end_seconds * rate)))
            score = periodicity_score(samples[start:end], rate)
            if score is not None:
                rows.append({
                    "utterance_id": utterance.utterance_id,
                    "phone": item.phone,
                    "label_devoiced": item.devoiced,
                    "periodicity": round(float(score), 9),
                })
    predicted = [row["periodicity"] < 0.42 for row in rows]
    truth = [bool(row["label_devoiced"]) for row in rows]
    accuracy = (sum(a == b for a, b in zip(predicted, truth)) / len(rows)
                if rows else None)
    grouped = {}
    for label in (False, True):
        values = [row["periodicity"] for row in rows
                  if bool(row["label_devoiced"]) == label]
        grouped["devoiced" if label else "voiced"] = {
            "count": len(values),
            "median_periodicity": (round(float(statistics.median(values)), 9)
                                     if values else None),
        }
    return {
        "available": bool(rows),
        "wav_missing_or_unreadable_count": missing,
        "classification_threshold": 0.42,
        "classification_accuracy": accuracy,
        "groups": grouped,
        "tokens": rows,
        "note": "Periodicity is reported separately from duration and is not itself ground truth.",
    }


def evaluate_duration_model(
    training: Sequence[CorpusUtterance],
    evaluation: Sequence[CorpusUtterance],
    priors: JapaneseDurationPriors,
) -> dict[str, object]:
    _unused, medians, _distributions = _training_rows(training)
    rows = _predict_rows(evaluation, medians, priors)
    reference = [row["reference"] for row in rows]
    systems = {}
    for system in ("legacy", "contextual"):
        systems[system] = duration_metrics(
            reference, [row[system] for row in rows]
        )
    by_class = {}
    for class_name in sorted({row["phone_class"] for row in rows}):
        selected = [row for row in rows if row["phone_class"] == class_name]
        by_class[class_name] = {
            system: duration_metrics(
                [row["reference"] for row in selected],
                [row[system] for row in selected],
            ) for system in ("legacy", "contextual")
        }
    by_phenomenon = {}
    for phenomenon in _FIT_FEATURES:
        selected = [row for row in rows if phenomenon in row["phenomena"]]
        by_phenomenon[phenomenon] = {
            system: duration_metrics(
                [row["reference"] for row in selected],
                [row[system] for row in selected],
            ) for system in ("legacy", "contextual")
        }
    totals = {}
    mora_totals = {}
    phrase_totals = {}
    rate_normalized = {}
    for system in ("legacy", "contextual"):
        actual_totals = []
        predicted_totals = []
        for utterance_id in sorted({row["utterance_id"] for row in rows}):
            selected = [row for row in rows if row["utterance_id"] == utterance_id]
            actual_totals.append(sum(row["reference"] for row in selected))
            predicted_totals.append(sum(row[system] for row in selected))
        totals[system] = duration_metrics(actual_totals, predicted_totals)
        mora_totals[system] = _grouped_metrics(
            rows, ("utterance_id", "mora_index"), system
        )
        phrase_totals[system] = _grouped_metrics(
            rows, ("utterance_id", "phrase_index"), system
        )
        rate_normalized[system] = {
            "log_duration_rmse": _rate_normalized_log_rmse(rows, system)
        }
    return {
        "schema_version": 1,
        "kind": "japanese_duration_evaluation",
        "model_id": priors.model_id,
        "evaluation_utterance_count": len(evaluation),
        "phone_metrics": systems,
        "mora_total_metrics": mora_totals,
        "accent_phrase_total_metrics": phrase_totals,
        "utterance_total_metrics": totals,
        "rate_normalized_residual_metrics": rate_normalized,
        "metrics_by_phone_class": by_class,
        "metrics_by_phenomenon": by_phenomenon,
        "contrast_statistics": _contrast_statistics(rows),
        "voicing_periodicity": evaluate_voicing(evaluation),
        "alignment_failures": [],
        "timing_reference_note": "JSUT/Julius phone boundaries are silver references; CSJ Core is stronger when supplied.",
    }


def write_json(path: Path | str, value: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def write_priors(path: Path | str, priors: JapaneseDurationPriors) -> Path:
    return write_json(path, priors.to_dict())


def markdown_report(report: Mapping[str, object]) -> str:
    lines = ["# Japanese Duration Benchmark", ""]
    lines.append(f"Model: `{report.get('model_id', 'unknown')}`")
    lines.append("")
    metrics = dict(report.get("phone_metrics") or {})
    if metrics:
        lines.extend([
            "## Phone Timing", "",
            "| System | Count | MAE (ms) | RMSE (ms) | Log RMSE | Median APE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for system in ("legacy", "contextual"):
            row = dict(metrics.get(system) or {})
            lines.append(
                "| %s | %s | %s | %s | %s | %s%% |" % (
                    system, row.get("count", 0), row.get("mae_ms"),
                    row.get("rmse_ms"), row.get("log_rmse"),
                    row.get("median_absolute_percentage_error"),
                )
            )
    else:
        lines.extend([
            "## Fit Summary", "",
            f"Training utterances: {report.get('training_utterance_count', 0)}",
            f"Training phones: {report.get('training_phone_count', 0)}",
            "",
            "Fixed validation IDs were excluded before fitting.",
        ])
    lines.extend([
        "", "## Interpretation", "",
        "Duration metrics and periodicity/voicing metrics are intentionally separate.",
        "No numerical timing result establishes acoustic naturalness.",
        "JSUT forced-alignment boundaries are a silver reference; human listening remains required.",
        "",
    ])
    return "\n".join(lines)


def _write_markdown(path: Path | str, report: Mapping[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown_report(report), encoding="utf-8")
    return destination


def _cli_fit(args) -> int:
    utterances = list(load_jsut(args.jsut)) + list(load_csj(args.csj))
    result = fit_duration_priors(
        utterances, heldout_fraction=args.heldout_fraction,
        seed_priors=load_duration_priors(args.seed_priors),
    )
    write_priors(args.output, result.priors)
    write_json(args.report_json, result.report)
    _write_markdown(args.report_markdown, result.report)
    print(f"wrote {result.priors.model_id} from "
          f"{result.report['training_utterance_count']} utterances")
    return 0


def _cli_evaluate(args) -> int:
    utterances = list(load_jsut(args.jsut)) + list(load_csj(args.csj))
    fixed = [item for item in utterances
             if item.utterance_id in FIXED_VALIDATION_IDS]
    heldout = set(select_heldout_ids(utterances, args.heldout_fraction))
    evaluation = fixed + [item for item in utterances
                          if item.utterance_id in heldout]
    excluded = {item.utterance_id for item in evaluation}
    training = [item for item in utterances
                if item.utterance_id not in excluded]
    if not evaluation:
        raise ValueError("no fixed or held-out evaluation utterances were found")
    if not training:
        raise ValueError("no independent training utterances remain")
    priors = load_duration_priors(args.priors)
    report = evaluate_duration_model(training, evaluation, priors)
    report["fixed_validation_ids"] = list(FIXED_VALIDATION_IDS)
    report["evaluated_ids"] = sorted(item.utterance_id for item in evaluation)
    write_json(args.output_json, report)
    _write_markdown(args.output_markdown, report)
    print(f"evaluated {len(evaluation)} utterances with {priors.model_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit/evaluate source-relative Japanese duration residuals"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit deterministic residual priors")
    fit.add_argument("--jsut", required=True)
    fit.add_argument("--csj", default="")
    fit.add_argument("--seed-priors", default=str(DEFAULT_PRIORS_PATH))
    fit.add_argument("--heldout-fraction", type=float, default=0.15)
    fit.add_argument("--output", required=True)
    fit.add_argument("--report-json", required=True)
    fit.add_argument("--report-markdown", required=True)
    fit.set_defaults(function=_cli_fit)
    evaluate = commands.add_parser("evaluate", help="compare legacy/contextual")
    evaluate.add_argument("--jsut", required=True)
    evaluate.add_argument("--csj", default="")
    evaluate.add_argument("--priors", required=True)
    evaluate.add_argument("--heldout-fraction", type=float, default=0.15)
    evaluate.add_argument("--output-json", required=True)
    evaluate.add_argument("--output-markdown", required=True)
    evaluate.set_defaults(function=_cli_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
