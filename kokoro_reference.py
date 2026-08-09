"""Safe Kokoro corpus access and deterministic silver phone alignment.

The Kokoro xlarge archive distributed with this project is a gzip stream
inside another gzip stream.  This module never calls ``TarFile.extract``:
every member is validated, only regular files selected by the caller are
written, and destination paths are checked after resolution.

Kokoro supplies exact phone strings but no authoritative phone boundaries.
The dependency-free aligner below treats those strings as linguistic truth
and refines duration-prior boundaries with local energy and spectral novelty.
Its output is explicitly labelled *silver* and carries confidence/rejection
data.  A Kokoro-Align CTC adapter can supersede these boundaries later without
changing the serialized record shape.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np


KOKORO_REFERENCE_SCHEMA_VERSION = 1
ALIGNMENT_METHOD = "metadata_phone_sequence_acoustic_refinement_v1"
KOKORO_ALIGN_METHOD = "kokoro_align_ctc_20221201_acoustic_refinement_v1"
PAUSE_REFINEMENT_METHOD = "punctuation_pause_energy_v2"
KOKORO_ALIGN_VOCAB = (
    "_", "N", "a", "a:", "b", "by", "ch", "d", "e", "e:", "f",
    "g", "gy", "h", "hy", "i", "i:", "j", "k", "ky", "m", "my",
    "n", "ny", "o", "o:", "p", "py", "r", "ry", "s", "sh", "t",
    "ts", "u", "u:", "w", "y", "z",
)
_VOWELS = {"a", "i", "u", "e", "o"}
_VOICELESS = {"k", "ky", "s", "sh", "t", "ts", "ch", "h", "hy", "f", "p", "py", "cl"}
_PUNCTUATION = {"!", ",", ".", "?", "pau", "sil"}


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class KokoroRecord:
    utterance_id: str
    transcript: str
    reading: str
    phones: tuple[str, ...]
    partition: str
    strata: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "utterance_id": self.utterance_id,
            "transcript": self.transcript,
            "reading": self.reading,
            "phones": list(self.phones),
            "partition": self.partition,
            "strata": list(self.strata),
        }


@dataclass(frozen=True)
class SilverPhoneAlignment:
    index: int
    raw_phone: str
    phone: str
    start_seconds: float
    end_seconds: float
    confidence: float
    boundary_confidence_left: float
    boundary_confidence_right: float
    mora_index: int
    phrase_index: int
    long_vowel: bool = False
    moraic_nasal: bool = False
    geminate: bool = False
    probable_devoicing: bool = False
    rejection_reasons: tuple[str, ...] = ()

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "raw_phone": self.raw_phone,
            "phone": self.phone,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": round(self.duration_seconds, 6),
            "confidence": self.confidence,
            "boundary_confidence_left": self.boundary_confidence_left,
            "boundary_confidence_right": self.boundary_confidence_right,
            "mora_index": self.mora_index,
            "phrase_index": self.phrase_index,
            "long_vowel": self.long_vowel,
            "moraic_nasal": self.moraic_nasal,
            "geminate": self.geminate,
            "probable_devoicing": self.probable_devoicing,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class SilverUtteranceAlignment:
    utterance_id: str
    sample_rate: int
    sample_count: int
    phones: tuple[SilverPhoneAlignment, ...]
    confidence: float
    accepted: bool
    method: str = ALIGNMENT_METHOD
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": KOKORO_REFERENCE_SCHEMA_VERSION,
            "kind": "kokoro_silver_phone_alignment",
            "utterance_id": self.utterance_id,
            "sample_rate": self.sample_rate,
            "sample_count": self.sample_count,
            "duration_seconds": round(
                self.sample_count / max(1, self.sample_rate), 6
            ),
            "method": self.method,
            "confidence": self.confidence,
            "accepted": self.accepted,
            "diagnostics": list(self.diagnostics),
            "phones": [phone.to_dict() for phone in self.phones],
        }


@dataclass(frozen=True)
class KokoroAlignHints:
    boundaries: tuple[float, ...]
    boundary_confidences: tuple[float, ...]
    token_confidences: tuple[float | None, ...]
    diagnostics: tuple[str, ...]
    checkpoint_sha256: str


def _hz_to_mel(frequency: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _kokoro_mfcc(
    samples: np.ndarray,
    sample_rate: int,
    *,
    target_sample_rate: int = 22050,
    n_fft: int = 512,
    n_mels: int = 40,
    n_mfcc: int = 40,
) -> np.ndarray:
    """Reproduce Kokoro-Align's torchaudio MFCC defaults with NumPy.

    The original ``prepare.py`` uses ``torchaudio.transforms.MFCC`` with a
    512-point FFT, 40 HTK mel bands, a 256-sample hop, power spectra, and the
    orthonormal type-II DCT.  Keeping this implementation local makes the
    released checkpoint usable when torchaudio has no wheel for the host
    Python version.
    """
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if sample_rate != target_sample_rate:
        output_length = max(1, int(round(
            values.size * target_sample_rate / float(sample_rate)
        )))
        source_positions = np.arange(values.size, dtype=np.float64)
        target_positions = np.linspace(
            0.0, max(0.0, values.size - 1.0), output_length
        )
        values = np.interp(target_positions, source_positions, values)
    pad = n_fft // 2
    if values.size > 1:
        values = np.pad(values, (pad, pad), mode="reflect")
    else:
        values = np.pad(values, (pad, pad), mode="constant")
    hop = n_fft // 2
    frame_count = max(1, 1 + (values.size - n_fft) // hop)
    frames = np.empty((frame_count, n_fft), dtype=np.float64)
    for index in range(frame_count):
        frames[index] = values[index * hop:index * hop + n_fft]
    # torch.hann_window(periodic=True)
    window = np.hanning(n_fft + 1)[:-1]
    power = np.abs(np.fft.rfft(frames * window, n=n_fft, axis=1)) ** 2
    frequencies = np.linspace(0.0, target_sample_rate / 2.0,
                              n_fft // 2 + 1)
    mel_points = np.linspace(
        float(_hz_to_mel(0.0)),
        float(_hz_to_mel(target_sample_rate / 2.0)),
        n_mels + 2,
    )
    hz_points = _mel_to_hz(mel_points)
    filterbank = np.zeros((n_mels, frequencies.size), dtype=np.float64)
    for index in range(n_mels):
        left, center, right = hz_points[index:index + 3]
        rising = (frequencies - left) / max(center - left, 1e-12)
        falling = (right - frequencies) / max(right - center, 1e-12)
        filterbank[index] = np.maximum(0.0, np.minimum(rising, falling))
    mel_power = np.maximum(1e-10, power @ filterbank.T)
    mel_db = 10.0 * np.log10(mel_power)
    maximum = float(np.max(mel_db))
    mel_db = np.maximum(mel_db, maximum - 80.0)
    bands = np.arange(n_mels, dtype=np.float64)
    coefficients = np.arange(n_mfcc, dtype=np.float64)[:, None]
    dct = np.cos(math.pi / n_mels * (bands + 0.5) * coefficients)
    dct[0] *= math.sqrt(1.0 / n_mels)
    if n_mfcc > 1:
        dct[1:] *= math.sqrt(2.0 / n_mels)
    return (mel_db @ dct.T).astype(np.float32)


def _ctc_viterbi_path(
    log_probs: np.ndarray, target: Sequence[int]
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return the forced CTC state path and frames for every target token."""
    probabilities = np.asarray(log_probs, dtype=np.float64)
    labels = np.asarray(tuple(int(value) for value in target), dtype=np.int64)
    if probabilities.ndim != 2 or not labels.size:
        raise ValueError("CTC alignment requires 2-D logits and target labels")
    expanded = np.zeros(labels.size * 2 + 1, dtype=np.int64)
    expanded[1::2] = labels
    frame_count, _vocabulary_size = probabilities.shape
    state_count = expanded.size
    if frame_count < labels.size:
        raise ValueError("audio has fewer CTC frames than target phones")
    negative = -1.0e30
    previous = np.full(state_count, negative, dtype=np.float64)
    previous[0] = probabilities[0, 0]
    if state_count > 1:
        previous[1] = probabilities[0, expanded[1]]
    back = np.zeros((frame_count, state_count), dtype=np.int8)
    for frame_index in range(1, frame_count):
        current = np.full(state_count, negative, dtype=np.float64)
        for state in range(state_count):
            choices = [(previous[state], 0)]
            if state > 0:
                choices.append((previous[state - 1], 1))
            if (state > 1 and expanded[state] != 0 and
                    expanded[state] != expanded[state - 2]):
                choices.append((previous[state - 2], 2))
            best_score, step = max(choices, key=lambda item: item[0])
            current[state] = best_score + probabilities[
                frame_index, expanded[state]
            ]
            back[frame_index, state] = step
        previous = current
    final_states = [state_count - 1]
    if state_count > 1:
        final_states.append(state_count - 2)
    state = max(final_states, key=lambda value: previous[value])
    states = np.empty(frame_count, dtype=np.int32)
    states[-1] = state
    for frame_index in range(frame_count - 1, 0, -1):
        state -= int(back[frame_index, state])
        states[frame_index - 1] = state
    frames = tuple(
        np.flatnonzero(states == (2 * index + 1))
        for index in range(labels.size)
    )
    if any(item.size == 0 for item in frames):
        raise ValueError("CTC best path omitted one or more target phones")
    return states, frames


class KokoroAlignCheckpoint:
    """Optional inference wrapper for the official epoch-200 checkpoint."""

    def __init__(self, checkpoint_path: Path | str):
        self.path = Path(checkpoint_path).resolve()
        self.sha256 = sha256_file(self.path)
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Kokoro-Align checkpoint inference requires PyTorch; "
                "the dependency-free acoustic aligner remains available"
            ) from exc

        class AudioToChar(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    40, 128, num_layers=2, dropout=0.5,
                    bidirectional=True,
                )
                self.dense = nn.Linear(256, len(KOKORO_ALIGN_VOCAB))

            def forward(self, audio):
                output, _state = self.lstm(audio)
                return self.dense(output)

        payload = torch.load(
            self.path, map_location="cpu", weights_only=True
        )
        state = payload.get("model") if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping):
            raise ValueError("Kokoro-Align checkpoint has no model state")
        model = AudioToChar()
        model.load_state_dict(state, strict=True)
        model.eval()
        self._model = model
        return model

    def alignment_hints(
        self,
        record: KokoroRecord,
        samples: Sequence[float] | np.ndarray,
        sample_rate: int,
    ) -> KokoroAlignHints:
        values = np.asarray(samples, dtype=np.float64).reshape(-1)
        if not values.size:
            raise ValueError("cannot align empty audio")
        model = self._load_model()
        import torch

        mfcc = _kokoro_mfcc(values, sample_rate)
        tensor = torch.from_numpy(mfcc).unsqueeze(1)
        with torch.no_grad():
            logits = model(tensor).squeeze(1)
            log_probs_tensor = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs_tensor.cpu().numpy()
        vocab_to_id = {phone: index for index, phone in
                       enumerate(KOKORO_ALIGN_VOCAB)}
        retained: list[tuple[int, int]] = []
        for original_index, raw_phone in enumerate(record.phones):
            if raw_phone in vocab_to_id and raw_phone != "_":
                retained.append((original_index, vocab_to_id[raw_phone]))
        if not retained:
            raise ValueError("phone sequence has no Kokoro-Align target labels")
        _states, token_frames = _ctc_viterbi_path(
            log_probs, [label for _index, label in retained]
        )
        frame_hop = 256.0 / 22050.0
        duration = values.size / float(sample_rate)
        times, energy, _flux = _acoustic_boundary_features(values, sample_rate)
        active_start, active_end, active_diagnostics = _active_interval(
            times, energy, duration
        )
        weights = np.asarray([_phone_weight(phone) for phone in record.phones],
                             dtype=np.float64)
        coordinates = np.cumsum(weights) - weights / 2.0
        anchor_x = [0.0]
        anchor_time = [active_start]
        token_confidence: list[float | None] = [None] * len(record.phones)
        for (original_index, label), frames in zip(retained, token_frames):
            center = float(np.mean(frames) * frame_hop)
            center = max(active_start, min(active_end, center))
            posterior = np.exp(log_probs[frames, label])
            confidence = float(np.clip(np.median(posterior), 0.0, 1.0))
            anchor_x.append(float(coordinates[original_index]))
            anchor_time.append(center)
            token_confidence[original_index] = confidence
        anchor_x.append(float(np.sum(weights)))
        anchor_time.append(active_end)
        order = np.argsort(anchor_x)
        anchor_x_array = np.asarray(anchor_x)[order]
        anchor_time_array = np.maximum.accumulate(
            np.asarray(anchor_time)[order]
        )
        centers = np.interp(coordinates, anchor_x_array, anchor_time_array)
        boundaries = [active_start]
        boundaries.extend(
            float((centers[index - 1] + centers[index]) / 2.0)
            for index in range(1, len(centers))
        )
        boundaries.append(active_end)
        confidences = [0.60]
        for index in range(1, len(record.phones)):
            adjacent = [token_confidence[index - 1], token_confidence[index]]
            finite = [value for value in adjacent if value is not None]
            confidences.append(
                float(np.mean(finite)) if finite else 0.30
            )
        confidences.append(0.60)
        dropped = [
            record.phones[index] for index, value in enumerate(token_confidence)
            if value is None
        ]
        diagnostics = list(active_diagnostics)
        diagnostics.append(
            "Kokoro-Align epoch-200 CTC checkpoint supplied primary phone anchors."
        )
        if dropped:
            diagnostics.append(
                "The official 39-symbol encoder drops q and punctuation; "
                "their boundaries were interpolated and acoustically refined: "
                + " ".join(dropped)
            )
        return KokoroAlignHints(
            boundaries=tuple(round(value, 6) for value in boundaries),
            boundary_confidences=tuple(
                round(float(value), 6) for value in confidences
            ),
            token_confidences=tuple(token_confidence),
            diagnostics=tuple(diagnostics),
            checkpoint_sha256=self.sha256,
        )


def partition_for_id(utterance_id: str, seed: str = "kokoro-prompt20-v1") -> str:
    value = int.from_bytes(
        hashlib.sha256(f"{seed}\0{utterance_id}".encode("utf-8")).digest()[:8],
        "big",
    ) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "validation"
    return "test"


def canonical_phone(raw_phone: str) -> tuple[str, bool]:
    value = str(raw_phone).strip()
    # Kokoro uses `_` as an orthographic word separator. It is not one of
    # the checkpoint's acoustic phone labels and must not consume an ordinary
    # consonant-sized interval or create a Japanese phrase boundary.
    if value == "_":
        return "sp", False
    if value in {"q", "Q"}:
        return "cl", False
    if value.endswith(":") and value[:-1] in _VOWELS:
        return value[:-1], True
    if value in {"I", "U"}:
        return value.lower(), False
    if value in _PUNCTUATION:
        return "pau", False
    return value, False


def _record_strata(tokens: Sequence[str], transcript: str) -> tuple[str, ...]:
    canonical = [canonical_phone(token)[0] for token in tokens]
    result = {f"vowel_{phone}" for phone in canonical if phone in _VOWELS}
    if any(str(token).endswith(":") for token in tokens):
        result.add("long_vowel")
    if "N" in canonical:
        result.add("moraic_nasal")
    if "cl" in canonical:
        result.add("geminate")
    if any(token in {",", ".", "!", "?"} for token in tokens):
        result.add("phrase_boundary")
    if "?" in tokens or "？" in transcript:
        result.add("interrogative")
    for index, phone in enumerate(canonical):
        if phone not in {"i", "u"}:
            continue
        previous = canonical[index - 1] if index else "pau"
        following = canonical[index + 1] if index + 1 < len(canonical) else "pau"
        if previous in _VOICELESS and following in _VOICELESS | {"pau"}:
            result.add("devoicing_context")
    return tuple(sorted(result))


def parse_metadata_text(text: str, *, seed: str = "kokoro-prompt20-v1") \
        -> tuple[KokoroRecord, ...]:
    records: list[KokoroRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.rstrip("\r\n").split("|")
        if len(fields) != 3:
            raise ValueError(
                f"Kokoro metadata line {line_number} has {len(fields)} fields"
            )
        utterance_id, transcript, reading = fields
        if not utterance_id or utterance_id in seen:
            raise ValueError(
                f"duplicate or empty Kokoro ID at line {line_number}: "
                f"{utterance_id!r}"
            )
        if not _safe_relative_name(utterance_id):
            raise ValueError(
                f"unsafe Kokoro ID at line {line_number}: {utterance_id!r}"
            )
        seen.add(utterance_id)
        phones = tuple(token for token in reading.split() if token)
        records.append(KokoroRecord(
            utterance_id=utterance_id,
            transcript=transcript,
            reading=reading,
            phones=phones,
            partition=partition_for_id(utterance_id, seed),
            strata=_record_strata(phones, transcript),
        ))
    return tuple(records)


def select_stratified_records(
    records: Sequence[KokoroRecord],
    *,
    train_count: int = 36,
    validation_count: int = 12,
    test_count: int = 12,
) -> tuple[KokoroRecord, ...]:
    """Select early deterministic records while covering every phenomenon.

    Keeping the earliest suitable IDs is intentional: Kokoro's archive and
    metadata are ordered similarly, so the streaming extractor can stop after
    finding the sample instead of inflating the complete 4 GB archive.
    """
    limits = {
        "train": max(0, int(train_count)),
        "validation": max(0, int(validation_count)),
        "test": max(0, int(test_count)),
    }
    chosen: list[KokoroRecord] = []
    counts = {key: 0 for key in limits}
    covered = {key: set() for key in limits}
    required = {
        "vowel_a", "vowel_i", "vowel_u", "vowel_e", "vowel_o",
        "long_vowel", "moraic_nasal", "geminate", "phrase_boundary",
        "devoicing_context",
    }
    ordered = sorted(records, key=lambda item: item.utterance_id)
    for record in ordered:
        partition = record.partition
        if counts[partition] >= limits[partition]:
            continue
        contributes = (set(record.strata) & required) - covered[partition]
        if contributes:
            chosen.append(record)
            counts[partition] += 1
            covered[partition].update(record.strata)
    selected_ids = {item.utterance_id for item in chosen}
    for record in ordered:
        partition = record.partition
        if record.utterance_id in selected_ids:
            continue
        if counts[partition] >= limits[partition]:
            continue
        chosen.append(record)
        selected_ids.add(record.utterance_id)
        counts[partition] += 1
        if all(counts[key] >= limits[key] for key in limits):
            break
    return tuple(sorted(chosen, key=lambda item: item.utterance_id))


def _safe_relative_name(name: str) -> bool:
    normalized = str(name).replace("\\", "/")
    if not normalized or normalized.startswith("/"):
        return False
    if re.match(r"^[A-Za-z]:", normalized):
        return False
    path = PurePosixPath(normalized)
    return not path.is_absolute() and all(
        part not in {"", ".", ".."} for part in path.parts
    )


def _safe_destination(root: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not _safe_relative_name(normalized):
        raise ValueError(f"unsafe archive member path: {member_name!r}")
    destination = (root / PurePosixPath(normalized)).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"archive member escapes destination: {member_name!r}"
        ) from exc
    return destination


@contextmanager
def open_kokoro_tar(path: Path | str) -> Iterator[tarfile.TarFile]:
    """Open ordinary or double-gzipped Kokoro tar streams."""
    with ExitStack() as stack:
        raw = stack.enter_context(Path(path).open("rb"))
        first = stack.enter_context(gzip.GzipFile(fileobj=raw, mode="rb"))
        stream: io.BufferedIOBase = first
        if first.peek(2)[:2] == b"\x1f\x8b":
            stream = stack.enter_context(gzip.GzipFile(fileobj=first, mode="rb"))
        archive = stack.enter_context(tarfile.open(fileobj=stream, mode="r|"))
        yield archive


def inventory_kokoro_prefix(
    archive_path: Path | str,
    *,
    maximum_audio_members: int = 800,
    seed: str = "kokoro-prompt20-v1",
) -> dict[str, object]:
    """Inventory a bounded, deterministic prefix without extracting files.

    Kokoro's FLAC members are shuffled.  Sampling only IDs known to occur in
    an early archive prefix lets the subsequent streaming extraction stop
    promptly while partition assignment and phenomenon selection still come
    from the complete metadata table.
    """
    limit = max(1, int(maximum_audio_members))
    metadata_bytes: bytes | None = None
    audio_ids: list[str] = []
    rejected_members: list[dict[str, str]] = []
    members_visited = 0
    with open_kokoro_tar(archive_path) as archive:
        for member in archive:
            members_visited += 1
            normalized = member.name.replace("\\", "/")
            while normalized.startswith("./"):
                normalized = normalized[2:]
            if not _safe_relative_name(normalized):
                rejected_members.append({
                    "member": member.name,
                    "reason": "unsafe archive member path",
                })
                continue
            if member.issym() or member.islnk() or member.isdev():
                rejected_members.append({
                    "member": member.name,
                    "reason": "links and device members are never inspected",
                })
                continue
            if normalized == "metadata.csv" and member.isfile():
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("Kokoro metadata member has no data")
                with source:
                    metadata_bytes = source.read()
                continue
            match = re.fullmatch(r"wavs/([^/]+)\.flac", normalized)
            if match and member.isfile():
                audio_ids.append(match.group(1))
                if len(audio_ids) >= limit:
                    break
    if metadata_bytes is None:
        raise ValueError("metadata.csv was not found in the Kokoro archive")
    records = parse_metadata_text(metadata_bytes.decode("utf-8"), seed=seed)
    by_id = {record.utterance_id: record for record in records}
    absent = [identifier for identifier in audio_ids if identifier not in by_id]
    if absent:
        raise ValueError(
            "archive FLAC IDs are absent from metadata: "
            + ", ".join(absent[:20])
        )
    candidate_records = tuple(by_id[identifier] for identifier in audio_ids)
    return {
        "schema_version": KOKORO_REFERENCE_SCHEMA_VERSION,
        "kind": "kokoro_bounded_archive_inventory",
        "members_visited": members_visited,
        "maximum_audio_members": limit,
        "candidate_record_count": len(candidate_records),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "partition_counts": {
            name: sum(record.partition == name for record in candidate_records)
            for name in ("train", "validation", "test")
        },
        "candidate_records": [record.to_dict() for record in candidate_records],
        "rejected_members": rejected_members,
    }


def _copy_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
) -> dict[str, object]:
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"regular archive member has no stream: {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    digest = hashlib.sha256()
    size = 0
    try:
        with source, temporary.open("wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": destination.name,
        "byte_length": size,
        "sha256": digest.hexdigest(),
    }


def safe_extract_kokoro_archive(
    archive_path: Path | str,
    destination: Path | str,
    *,
    train_count: int = 36,
    validation_count: int = 12,
    test_count: int = 12,
    seed: str = "kokoro-prompt20-v1",
    record_ids: Sequence[str] | None = None,
    archive_sha256: str | None = None,
) -> dict[str, object]:
    """Safely extract metadata and a deterministic FLAC sample.

    ``record_ids`` is intended for bounded smoke tests where an archive member
    has already been named during an inventory pass.  Omitting it keeps the
    deterministic stratified partition selection used for corpus analysis.
    Supplying a previously calculated archive digest avoids reading a large
    compressed archive twice; an omitted digest is reported as unverified
    rather than silently claiming that hashing occurred.
    """
    archive_path = Path(archive_path).resolve()
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata_bytes: bytes | None = None
    selected: tuple[KokoroRecord, ...] = ()
    remaining: set[str] = set()
    extracted: list[dict[str, object]] = []
    rejected_members: list[dict[str, str]] = []
    visited = 0
    with open_kokoro_tar(archive_path) as archive:
        for member in archive:
            visited += 1
            try:
                destination_path = _safe_destination(root, member.name)
            except ValueError as exc:
                rejected_members.append({
                    "member": member.name,
                    "reason": str(exc),
                })
                continue
            if member.issym() or member.islnk() or member.isdev():
                rejected_members.append({
                    "member": member.name,
                    "reason": "links and device members are never extracted",
                })
                continue
            if member.isdir():
                continue
            if not member.isfile():
                rejected_members.append({
                    "member": member.name,
                    "reason": "unsupported non-regular archive member",
                })
                continue
            relative = destination_path.relative_to(root).as_posix()
            if relative == "metadata.csv":
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("Kokoro metadata member has no data")
                with source:
                    metadata_bytes = source.read()
                metadata_text = metadata_bytes.decode("utf-8")
                records = parse_metadata_text(metadata_text, seed=seed)
                if record_ids is None:
                    selected = select_stratified_records(
                        records,
                        train_count=train_count,
                        validation_count=validation_count,
                        test_count=test_count,
                    )
                else:
                    by_id = {record.utterance_id: record for record in records}
                    missing_ids = sorted(set(record_ids) - set(by_id))
                    if missing_ids:
                        raise ValueError(
                            "requested Kokoro IDs are absent from metadata: "
                            + ", ".join(missing_ids[:20])
                        )
                    selected = tuple(
                        by_id[utterance_id]
                        for utterance_id in dict.fromkeys(record_ids)
                    )
                remaining = {
                    f"wavs/{record.utterance_id}.flac" for record in selected
                }
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                destination_path.write_bytes(metadata_bytes)
                extracted.append({
                    "path": "metadata.csv",
                    "byte_length": len(metadata_bytes),
                    "sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                })
                continue
            if relative in remaining:
                details = _copy_member(archive, member, destination_path)
                details["path"] = relative
                extracted.append(details)
                remaining.remove(relative)
                if not remaining:
                    break
    if metadata_bytes is None:
        raise ValueError("metadata.csv was not found in the Kokoro archive")
    if remaining:
        missing = sorted(remaining)
        raise ValueError(
            "selected Kokoro recordings were absent from the archive: "
            + ", ".join(missing[:20])
        )
    selection_payload = {
        "schema_version": KOKORO_REFERENCE_SCHEMA_VERSION,
        "kind": "kokoro_stratified_selection",
        "seed": seed,
        "records": [record.to_dict() for record in selected],
    }
    selection_path = root / "partitions.json"
    selection_path.write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": KOKORO_REFERENCE_SCHEMA_VERSION,
        "kind": "kokoro_safe_extraction_report",
        "archive": {
            "file_name": archive_path.name,
            "byte_length": archive_path.stat().st_size,
            "sha256": archive_sha256,
            "sha256_verified": archive_sha256 is not None,
            "container": "tar inside two gzip streams",
        },
        "members_visited": visited,
        "selected_record_count": len(selected),
        "partitions": {
            name: sum(record.partition == name for record in selected)
            for name in ("train", "validation", "test")
        },
        "extracted_files": sorted(extracted, key=lambda item: str(item["path"])),
        "rejected_members": rejected_members,
        "source_archive_modified": False,
    }
    (root / "extraction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def load_selection(path: Path | str) -> tuple[KokoroRecord, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != KOKORO_REFERENCE_SCHEMA_VERSION:
        raise ValueError("unsupported Kokoro selection schema")
    records = []
    for row in payload.get("records") or ():
        records.append(KokoroRecord(
            utterance_id=str(row["utterance_id"]),
            transcript=str(row.get("transcript") or ""),
            reading=str(row.get("reading") or ""),
            phones=tuple(str(item) for item in row.get("phones") or ()),
            partition=str(row["partition"]),
            strata=tuple(str(item) for item in row.get("strata") or ()),
        ))
    return tuple(records)


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median, max(1e-6, 1.4826 * mad)


def _acoustic_boundary_features(
    samples: np.ndarray, sample_rate: int, hop_seconds: float = 0.005
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hop = max(1, int(round(sample_rate * hop_seconds)))
    window_length = max(hop * 2, int(round(sample_rate * 0.020)))
    nfft = 1 << max(8, (window_length - 1).bit_length())
    window = np.hanning(window_length)
    frame_count = max(1, 1 + max(0, len(samples) - window_length) // hop)
    energy = np.empty(frame_count, dtype=np.float64)
    spectra = np.empty((frame_count, nfft // 2 + 1), dtype=np.float64)
    for index in range(frame_count):
        start = index * hop
        frame = np.zeros(window_length, dtype=np.float64)
        chunk = samples[start:start + window_length]
        frame[:len(chunk)] = chunk
        energy[index] = 10.0 * math.log10(
            max(1e-12, float(np.mean(frame * frame)))
        )
        magnitude = np.abs(np.fft.rfft(frame * window, nfft))
        spectra[index] = magnitude / max(1e-12, float(np.linalg.norm(magnitude)))
    flux = np.zeros(frame_count, dtype=np.float64)
    if frame_count > 1:
        difference = np.maximum(0.0, spectra[1:] - spectra[:-1])
        flux[1:] = np.sqrt(np.mean(difference * difference, axis=1))
    times = (np.arange(frame_count, dtype=np.float64) * hop
             + window_length / 2.0) / sample_rate
    return times, energy, flux


def _active_interval(
    times: np.ndarray, energy_db: np.ndarray, duration: float
) -> tuple[float, float, list[str]]:
    diagnostics: list[str] = []
    if energy_db.size < 3:
        return 0.0, duration, ["too_few_energy_frames"]
    noise = float(np.percentile(energy_db, 15.0))
    peak = float(np.percentile(energy_db, 95.0))
    threshold = max(noise + 8.0, peak - 38.0)
    active = np.flatnonzero(energy_db >= threshold)
    if active.size == 0:
        return 0.0, duration, ["no_active_speech_detected"]
    start = max(0.0, float(times[active[0]]) - 0.015)
    end = min(duration, float(times[active[-1]]) + 0.015)
    if end - start < duration * 0.35:
        diagnostics.append("active_interval_unusually_short")
    return start, end, diagnostics


def _contiguous_true_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    indexes = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not indexes.size:
        return ()
    runs = []
    start = previous = int(indexes[0])
    for value in indexes[1:]:
        current = int(value)
        if current > previous + 1:
            runs.append((start, previous))
            start = current
        previous = current
    runs.append((start, previous))
    return tuple(runs)


def refine_phrase_pauses(
    alignment: SilverUtteranceAlignment,
    samples: Sequence[float] | np.ndarray,
    sample_rate: int,
    *,
    minimum_pause_seconds: float = 0.055,
    search_radius_seconds: float = 0.90,
) -> SilverUtteranceAlignment:
    """Expand internal punctuation pauses to the observed low-energy run.

    Kokoro-Align has no punctuation class. Its interpolated punctuation token
    can therefore leave real silence attached to the preceding vowel or the
    following onset. Punctuation supplies the existence and approximate
    location of a phrase break; smoothed frame energy supplies both edges.
    Word separators are deliberately excluded.
    """
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if sample_rate <= 0 or not values.size or len(alignment.phones) < 3:
        return alignment
    times, energy, _flux = _acoustic_boundary_features(values, sample_rate)
    if energy.size < 5:
        return alignment
    padded = np.pad(energy, (2, 2), mode="edge")
    smooth = np.convolve(padded, np.full(5, 0.2), mode="valid")
    noise = float(np.percentile(smooth, 10.0))
    speech = float(np.percentile(smooth, 75.0))
    contrast = speech - noise
    if not math.isfinite(contrast) or contrast < 8.0:
        return replace(
            alignment,
            diagnostics=alignment.diagnostics + (
                f"{PAUSE_REFINEMENT_METHOD}=insufficient_energy_contrast",
            ),
        )
    threshold = min(speech - 10.0, noise + 0.68 * contrast)
    hop = (float(np.median(np.diff(times))) if times.size > 1
           else 0.005)
    phones = list(alignment.phones)
    refinements = []

    for pause_index, pause in enumerate(tuple(phones)):
        if pause.phone != "pau" or pause_index <= 0 \
                or pause_index >= len(phones) - 1:
            continue
        previous_index = next((
            index for index in range(pause_index - 1, -1, -1)
            if phones[index].phone not in {"pau", "sp", "sil"}
        ), None)
        following_index = next((
            index for index in range(pause_index + 1, len(phones))
            if phones[index].phone not in {"pau", "sp", "sil"}
        ), None)
        if previous_index is None or following_index is None:
            continue
        previous = phones[previous_index]
        following = phones[following_index]
        coarse_center = 0.5 * (pause.start_seconds + pause.end_seconds)
        radius = max(
            float(search_radius_seconds),
            1.35 * max(float(minimum_pause_seconds), pause.duration_seconds),
        )
        lower = max(
            previous.start_seconds + 0.015,
            coarse_center - radius,
        )
        upper = min(
            following.end_seconds - 0.015,
            coarse_center + radius,
        )
        if upper - lower < minimum_pause_seconds:
            continue
        in_search = (times >= lower) & (times <= upper)
        runs = []
        for first, last in _contiguous_true_runs(
                in_search & (smooth <= threshold)):
            start = max(lower, float(times[first]) - hop / 2.0)
            end = min(upper, float(times[last]) + hop / 2.0)
            duration = end - start
            if duration < minimum_pause_seconds:
                continue
            expanded_left = pause.start_seconds - 0.12
            expanded_right = pause.end_seconds + 0.12
            overlap = max(
                0.0,
                min(end, expanded_right) - max(start, expanded_left),
            )
            if overlap <= 0.0 and abs(
                    0.5 * (start + end) - coarse_center) > 0.25:
                continue
            distance = abs(0.5 * (start + end) - coarse_center)
            score = duration + 0.75 * overlap - 0.20 * distance
            runs.append((score, duration, -distance, start, end))
        if not runs:
            continue
        _score, duration, _distance, start, end = max(runs)
        if end <= start:
            continue
        confidence = min(
            0.98,
            0.58 + 0.22 * min(1.0, contrast / 40.0)
            + 0.18 * min(1.0, duration / 0.35),
        )

        def corrected_rejections(item, new_start, new_end):
            reasons = [reason for reason in item.rejection_reasons
                       if reason not in {"phone_too_short", "phone_too_long"}]
            actual = new_end - new_start
            if actual < 0.010:
                reasons.append("phone_too_short")
            if actual > 0.600 and item.phone not in {"pau", "sp"}:
                reasons.append("phone_too_long")
            return tuple(reasons)

        phones[previous_index] = replace(
            previous,
            end_seconds=round(start, 6),
            boundary_confidence_right=round(confidence, 6),
            rejection_reasons=corrected_rejections(
                previous, previous.start_seconds, start),
        )
        phones[pause_index] = replace(
            pause,
            start_seconds=round(start, 6),
            end_seconds=round(end, 6),
            confidence=round(confidence, 6),
            boundary_confidence_left=round(confidence, 6),
            boundary_confidence_right=round(confidence, 6),
            rejection_reasons=(),
        )
        phones[following_index] = replace(
            following,
            start_seconds=round(end, 6),
            boundary_confidence_left=round(confidence, 6),
            rejection_reasons=corrected_rejections(
                following, end, following.end_seconds),
        )
        refinements.append(
            f"{pause.index}:{pause.duration_seconds:.3f}->{duration:.3f}s"
        )

    if not refinements:
        return replace(
            alignment,
            diagnostics=alignment.diagnostics + (
                f"{PAUSE_REFINEMENT_METHOD}=no_internal_pause_refined",
            ),
        )
    return replace(
        alignment,
        phones=tuple(phones),
        diagnostics=alignment.diagnostics + (
            f"{PAUSE_REFINEMENT_METHOD}=" + ",".join(refinements),
        ),
    )


def _phone_weight(raw_phone: str) -> float:
    phone, long_vowel = canonical_phone(raw_phone)
    if phone == "sp":
        # Preserve an inspectable boundary token while assigning almost all
        # acoustically aligned time to its neighbouring spoken phones.
        return 0.08
    if phone == "pau":
        return 1.6 if raw_phone in {".", "!", "?"} else 0.8
    if long_vowel:
        return 1.25
    if phone in _VOWELS:
        return 1.0
    if phone == "N":
        return 0.85
    if phone == "cl":
        return 0.70
    if phone in {"s", "sh", "z", "h", "hy", "f"}:
        return 0.72
    if phone in {"ch", "ts", "j"}:
        return 0.78
    if phone in {"k", "ky", "g", "gy", "t", "d", "p", "py", "b", "by"}:
        return 0.52
    return 0.58


def _mora_indices(raw_phones: Sequence[str]) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    mora = -1
    phrase = 0
    pending_consonant = False
    for raw in raw_phones:
        phone, long_vowel = canonical_phone(raw)
        if phone == "sp":
            result.append((max(0, mora), phrase))
            pending_consonant = False
            continue
        if phone == "pau":
            result.append((max(0, mora), phrase))
            phrase += 1
            pending_consonant = False
            continue
        if phone in _VOWELS or phone in {"N", "cl"} or long_vowel:
            mora += 1
            pending_consonant = False
        elif not pending_consonant:
            mora += 1
            pending_consonant = True
        result.append((max(0, mora), phrase))
    return tuple(result)


def _probable_devoicing(raw_phones: Sequence[str], index: int) -> bool:
    phone, long_vowel = canonical_phone(raw_phones[index])
    if phone not in {"i", "u"} or long_vowel:
        return False
    canonical = [canonical_phone(item)[0] for item in raw_phones]
    previous = next(
        (value for value in reversed(canonical[:index]) if value != "sp"),
        "pau",
    )
    following = next(
        (value for value in canonical[index + 1:] if value != "sp"),
        "pau",
    )
    return previous in _VOICELESS and following in _VOICELESS | {"pau"}


def align_kokoro_record(
    record: KokoroRecord,
    samples: Sequence[float] | np.ndarray,
    sample_rate: int,
    *,
    minimum_confidence: float = 0.42,
    kokoro_align_hints: KokoroAlignHints | None = None,
) -> SilverUtteranceAlignment:
    """Refine metadata-phone boundaries without claiming manual truth."""
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    diagnostics: list[str] = []
    if sample_rate <= 0 or values.size == 0:
        return SilverUtteranceAlignment(
            utterance_id=record.utterance_id,
            sample_rate=max(0, int(sample_rate)),
            sample_count=int(values.size),
            phones=(),
            confidence=0.0,
            accepted=False,
            diagnostics=("empty_or_invalid_audio",),
        )
    tokens = tuple(record.phones)
    if not tokens:
        return SilverUtteranceAlignment(
            utterance_id=record.utterance_id,
            sample_rate=int(sample_rate),
            sample_count=int(values.size),
            phones=(),
            confidence=0.0,
            accepted=False,
            diagnostics=("empty_phone_sequence",),
        )
    duration = values.size / float(sample_rate)
    times, energy, flux = _acoustic_boundary_features(values, sample_rate)
    active_start, active_end, active_diagnostics = _active_interval(
        times, energy, duration
    )
    diagnostics.extend(active_diagnostics)
    active_duration = max(0.02, active_end - active_start)
    weights = np.asarray([_phone_weight(token) for token in tokens],
                         dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(weights)))
    predicted = active_start + active_duration * cumulative / cumulative[-1]
    method = ALIGNMENT_METHOD
    hint_confidences = np.zeros(len(tokens) + 1, dtype=np.float64)
    if kokoro_align_hints is not None:
        if len(kokoro_align_hints.boundaries) != len(tokens) + 1:
            raise ValueError("Kokoro-Align hint count does not match phones")
        predicted = np.asarray(
            kokoro_align_hints.boundaries, dtype=np.float64
        )
        hint_confidences = np.asarray(
            kokoro_align_hints.boundary_confidences, dtype=np.float64
        )
        if hint_confidences.size != len(tokens) + 1:
            raise ValueError(
                "Kokoro-Align confidence count does not match phones"
            )
        diagnostics.extend(kokoro_align_hints.diagnostics)
        diagnostics.append(
            "kokoro_align_checkpoint_sha256="
            + kokoro_align_hints.checkpoint_sha256
        )
        method = KOKORO_ALIGN_METHOD
    flux_median, flux_scale = _robust_scale(flux)
    energy_median, energy_scale = _robust_scale(energy)
    novelty = ((flux - flux_median) / flux_scale
               - 0.28 * (energy - energy_median) / energy_scale)
    boundaries = [float(predicted[0])]
    boundary_confidence = [max(0.55, float(hint_confidences[0]))]
    minimum_gap = max(0.008, active_duration / max(1, len(tokens)) * 0.18)
    for index in range(1, len(tokens)):
        target = float(predicted[index])
        expected_local = active_duration * (
            weights[index - 1] + weights[index]
        ) / (2.0 * cumulative[-1])
        radius = min(
            0.035 if kokoro_align_hints is not None else 0.060,
            max(0.015, expected_local * 0.55),
        )
        lower = max(boundaries[-1] + minimum_gap, target - radius)
        remaining = len(tokens) - index
        upper = min(
            active_end - remaining * minimum_gap,
            target + radius,
        )
        candidates = np.flatnonzero((times >= lower) & (times <= upper))
        if candidates.size == 0:
            chosen_time = max(lower, min(upper, target))
            confidence = 0.15
        else:
            distance_penalty = np.abs(times[candidates] - target) / max(radius, 1e-6)
            scores = novelty[candidates] - 0.45 * distance_penalty
            local = int(np.argmax(scores))
            chosen = int(candidates[local])
            chosen_time = float(times[chosen])
            z = float((novelty[chosen] - np.median(novelty[candidates])) /
                      max(1e-6, 1.4826 * np.median(np.abs(
                          novelty[candidates] - np.median(novelty[candidates])
                      ))))
            acoustic_confidence = 1.0 / (
                1.0 + math.exp(-max(-6.0, min(6.0, z)))
            )
            confidence = (
                0.65 * float(hint_confidences[index])
                + 0.35 * acoustic_confidence
                if kokoro_align_hints is not None
                else acoustic_confidence
            )
        boundaries.append(chosen_time)
        boundary_confidence.append(confidence)
    boundaries.append(active_end)
    boundary_confidence.append(max(
        0.55, float(hint_confidences[-1])
    ))
    # The checkpoint has no acoustic `_` class. Keep its interval visible as
    # an 8 ms alignment marker, then return the interpolated time to the two
    # neighbouring phones instead of fitting a fictitious spoken segment.
    separator_width = max(2.0 / sample_rate, 0.008)
    for index, raw_phone in enumerate(tokens):
        if canonical_phone(raw_phone)[0] != "sp":
            continue
        left_limit = (
            boundaries[index - 1] + 1.0 / sample_rate
            if index > 0 else boundaries[index]
        )
        right_limit = (
            boundaries[index + 2] - 1.0 / sample_rate
            if index + 2 < len(boundaries) else boundaries[index + 1]
        )
        if right_limit - left_limit < separator_width:
            continue
        center = (boundaries[index] + boundaries[index + 1]) / 2.0
        left = max(
            left_limit,
            min(center - separator_width / 2.0,
                right_limit - separator_width),
        )
        boundaries[index] = left
        boundaries[index + 1] = left + separator_width
    mora_positions = _mora_indices(tokens)
    aligned: list[SilverPhoneAlignment] = []
    expected_unit = active_duration / float(cumulative[-1])
    for index, raw_phone in enumerate(tokens):
        phone, long_vowel = canonical_phone(raw_phone)
        start = float(boundaries[index])
        end = max(start + 1.0 / sample_rate, float(boundaries[index + 1]))
        actual = end - start
        expected = expected_unit * weights[index]
        duration_confidence = math.exp(-abs(math.log(
            max(actual, 1e-6) / max(expected, 1e-6)
        )))
        left_confidence = float(boundary_confidence[index])
        right_confidence = float(boundary_confidence[index + 1])
        confidence = (
            0.50 * duration_confidence
            + 0.25 * left_confidence
            + 0.25 * right_confidence
        )
        reasons: list[str] = []
        if actual < 0.010:
            reasons.append("phone_too_short")
        if actual > 0.600 and phone not in {"pau", "sp"}:
            reasons.append("phone_too_long")
        if confidence < minimum_confidence:
            reasons.append("low_alignment_confidence")
        mora_index, phrase_index = mora_positions[index]
        aligned.append(SilverPhoneAlignment(
            index=index,
            raw_phone=raw_phone,
            phone=phone,
            start_seconds=round(start, 6),
            end_seconds=round(end, 6),
            confidence=round(float(confidence), 6),
            boundary_confidence_left=round(left_confidence, 6),
            boundary_confidence_right=round(right_confidence, 6),
            mora_index=mora_index,
            phrase_index=phrase_index,
            long_vowel=long_vowel,
            moraic_nasal=phone == "N",
            geminate=phone == "cl",
            probable_devoicing=_probable_devoicing(tokens, index),
            rejection_reasons=tuple(reasons),
        ))
    ordinary = [
        item.confidence for item in aligned
        if item.phone not in {"pau", "sp"}
    ]
    confidence = float(np.median(ordinary)) if ordinary else 0.0
    accepted_count = sum(
        not item.rejection_reasons for item in aligned
        if item.phone not in {"pau", "sp"}
    )
    ordinary_count = sum(
        item.phone not in {"pau", "sp"} for item in aligned
    )
    accepted = (
        confidence >= minimum_confidence
        and accepted_count >= max(1, math.ceil(ordinary_count * 0.75))
    )
    if not accepted:
        diagnostics.append("utterance_rejected_by_silver_alignment_gate")
    diagnostics.append(
        "Boundaries are silver references derived from metadata phones, "
        "duration priors, energy and spectral novelty; they are not manual labels."
    )
    result = SilverUtteranceAlignment(
        utterance_id=record.utterance_id,
        sample_rate=int(sample_rate),
        sample_count=int(values.size),
        phones=tuple(aligned),
        confidence=round(confidence, 6),
        accepted=accepted,
        method=method,
        diagnostics=tuple(diagnostics),
    )
    return refine_phrase_pauses(result, values, sample_rate)


def align_kokoro_with_checkpoint(
    checkpoint: KokoroAlignCheckpoint,
    record: KokoroRecord,
    samples: Sequence[float] | np.ndarray,
    sample_rate: int,
    *,
    minimum_confidence: float = 0.42,
) -> SilverUtteranceAlignment:
    hints = checkpoint.alignment_hints(record, samples, sample_rate)
    return align_kokoro_record(
        record,
        samples,
        sample_rate,
        minimum_confidence=minimum_confidence,
        kokoro_align_hints=hints,
    )


def write_alignment(path: Path | str, alignment: SilverUtteranceAlignment) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(alignment.to_dict(), ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_alignment(path: Path | str) -> SilverUtteranceAlignment:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != KOKORO_REFERENCE_SCHEMA_VERSION:
        raise ValueError("unsupported Kokoro alignment schema")
    phones = tuple(SilverPhoneAlignment(
        index=int(row["index"]),
        raw_phone=str(row["raw_phone"]),
        phone=str(row["phone"]),
        start_seconds=float(row["start_seconds"]),
        end_seconds=float(row["end_seconds"]),
        confidence=float(row["confidence"]),
        boundary_confidence_left=float(row["boundary_confidence_left"]),
        boundary_confidence_right=float(row["boundary_confidence_right"]),
        mora_index=int(row["mora_index"]),
        phrase_index=int(row["phrase_index"]),
        long_vowel=bool(row.get("long_vowel")),
        moraic_nasal=bool(row.get("moraic_nasal")),
        geminate=bool(row.get("geminate")),
        probable_devoicing=bool(row.get("probable_devoicing")),
        rejection_reasons=tuple(str(item) for item in
                                (row.get("rejection_reasons") or ())),
    ) for row in (payload.get("phones") or ()))
    return SilverUtteranceAlignment(
        utterance_id=str(payload["utterance_id"]),
        sample_rate=int(payload["sample_rate"]),
        sample_count=int(payload["sample_count"]),
        phones=phones,
        confidence=float(payload.get("confidence") or 0.0),
        accepted=bool(payload.get("accepted")),
        method=str(payload.get("method") or ALIGNMENT_METHOD),
        diagnostics=tuple(str(item) for item in
                          (payload.get("diagnostics") or ())),
    )
