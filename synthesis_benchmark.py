"""Matched legacy/optimized benchmark for Prompt 0a.

The legacy side disables slice caching, uses the dependency-free Python
normalizer, emits an in-memory WAV, and decodes it immediately.  The optimized
side uses bounded slice caching, NumPy normalization when already available,
and direct PCM handoff.  Both sides render the same phones and must produce
byte-identical PCM before timings are reported.
"""

from __future__ import annotations

import argparse
import array
import gc
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import time
import wave

import numpy as np

import synth_diphone as sd


TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_VOICE = TOOL_DIR / "generated_voices" / "lem_v4bi_integrated"
DEFAULT_PHRASE = "dh ih s ih z ah t eh s t".split()


def _decode_wav(data: bytes) -> array.array:
    result = array.array("h")
    with wave.open(io.BytesIO(data), "rb") as source:
        result.frombytes(source.readframes(source.getnframes()))
    return result


def _legacy_render(database, phones):
    # synth_diphone intentionally falls back to its standard-library loop when
    # NumPy is not already owned by the host process.
    numpy_module = sys.modules.pop("numpy", None)
    try:
        result = sd.render(database, phones, encode_wav=True)
    finally:
        if numpy_module is not None:
            sys.modules["numpy"] = numpy_module
    return result, _decode_wav(result["wav"])


def _optimized_render(database, phones):
    result = sd.render(
        database, phones, encode_wav=False, return_pcm=True)
    return result, result["pcm16"]


def _timed_runs(function, database, phones, runs):
    elapsed = []
    digest = None
    for _index in range(runs):
        started = time.perf_counter()
        _result, pcm = function(database, phones)
        elapsed.append(time.perf_counter() - started)
        current = hashlib.sha256(pcm.tobytes()).hexdigest()
        if digest is not None and current != digest:
            raise RuntimeError("a repeated benchmark render changed PCM")
        digest = current
    ordered = sorted(elapsed)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "runs": runs,
        "median_seconds": statistics.median(elapsed),
        "minimum_seconds": min(elapsed),
        "p95_seconds": ordered[p95_index],
        "pcm_sha256": digest,
    }


def benchmark(voice: Path, *, runs: int = 12, repetitions: int = 8) -> dict:
    voice = voice.expanduser().resolve()
    if not (voice / "dic" / "diphone_index.json").is_file():
        raise FileNotFoundError("voice has no dic/diphone_index.json: " +
                                str(voice))
    phones = []
    for index in range(max(1, repetitions)):
        if index:
            phones.append("pau")
        phones.extend(DEFAULT_PHRASE)

    started = time.perf_counter()
    legacy_db = sd.DiphoneDB(
        voice,
        slice_cache_max_entries=0,
        slice_cache_max_bytes=0,
    )
    legacy_init = time.perf_counter() - started
    started = time.perf_counter()
    optimized_db = sd.DiphoneDB(voice)
    optimized_init = time.perf_counter() - started

    # One cold result is retained for the cross-mode identity assertion.
    cold_started = time.perf_counter()
    _legacy_result, legacy_pcm = _legacy_render(legacy_db, phones)
    legacy_cold = time.perf_counter() - cold_started
    cold_started = time.perf_counter()
    _optimized_result, optimized_pcm = _optimized_render(
        optimized_db, phones)
    optimized_cold = time.perf_counter() - cold_started
    if legacy_pcm.tobytes() != optimized_pcm.tobytes():
        raise RuntimeError("legacy and optimized benchmark PCM differ")

    gc.collect()
    legacy_misses_before = legacy_db.cache_info()["decode_misses"]
    legacy = _timed_runs(_legacy_render, legacy_db, phones, runs)
    legacy_misses_after = legacy_db.cache_info()["decode_misses"]
    optimized_misses_before = optimized_db.cache_info()["decode_misses"]
    optimized = _timed_runs(_optimized_render, optimized_db, phones, runs)
    optimized_misses_after = optimized_db.cache_info()["decode_misses"]
    if legacy["pcm_sha256"] != optimized["pcm_sha256"]:
        raise RuntimeError("legacy and optimized warm PCM differ")
    legacy["constructor_seconds"] = legacy_init
    legacy["cold_seconds"] = legacy_cold
    legacy["warm_decode_misses"] = (
        legacy_misses_after - legacy_misses_before)
    legacy["cache"] = legacy_db.cache_info()
    optimized["constructor_seconds"] = optimized_init
    optimized["cold_seconds"] = optimized_cold
    optimized["warm_decode_misses"] = (
        optimized_misses_after - optimized_misses_before)
    optimized["cache"] = optimized_db.cache_info()
    return {
        "schema_version": 1,
        "voice_name": voice.name,
        "workload": {
            "phrase_phones": DEFAULT_PHRASE,
            "phrase_repetitions": repetitions,
            "phone_count": len(phones),
            "warm_runs": runs,
        },
        "legacy_emulation": legacy,
        "optimized": optimized,
        "comparison": {
            "warm_speedup": (
                legacy["median_seconds"] / optimized["median_seconds"]
            ),
            "cold_speedup": legacy_cold / optimized_cold,
            "pcm_identical": True,
            "optimized_warm_decode_misses": (
                optimized_misses_after - optimized_misses_before),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = benchmark(
        args.voice,
        runs=max(2, int(args.runs)),
        repetitions=max(1, int(args.repetitions)),
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
