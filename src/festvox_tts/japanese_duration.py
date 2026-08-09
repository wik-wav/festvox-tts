"""Deterministic context-sensitive Japanese phone duration prediction.

The selected UTAU/Festival unit geometry is the absolute timing baseline.
Open JTalk and canonical mora context supply bounded log-ratio adjustments;
they never replace the source speaker with a corpus speaker's millisecond
scale.  The coefficients are versioned data so corpus fitting can update the
model without changing runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

from cache_support import FileIdentityCache, deep_freeze
from japanese_models import JapaneseMora, JapaneseUtterance
from japanese_openjtalk import parse_full_context_label


DURATION_MODEL_SCHEMA_VERSION = 1
DEFAULT_PRIORS_PATH = (
    Path(__file__).resolve().parent
    / "profiles" / "japanese_duration_priors_v1.json"
)
_DURATION_PRIORS_CACHE = FileIdentityCache(
    "japanese-duration-priors", max_entries=4, max_bytes=8 * 1024 * 1024
)

_VOWELS = {"a", "i", "u", "e", "o"}
_VOICELESS = {
    "k", "ky", "p", "py", "t", "ch", "ts", "s", "sh", "f", "h",
    "hy", "cl",
}
_STOPS = {"k", "ky", "g", "gy", "t", "d", "p", "py", "b", "by"}
_AFFRICATES = {"ch", "ts", "j"}
_FRICATIVES = {"s", "sh", "z", "f", "h", "hy"}
_NASALS = {"m", "my", "n", "ny", "N"}


# Backward-compatible allocator values used when an older profile does not
# yet carry the mora allocation section.  These are source-speaker engineering
# anchors; contextual coefficients and bounded source geometry are applied
# afterwards by ``predict_mora_durations``.
DEFAULT_MORA_ALLOCATION_SECONDS = {
    "vowel_only": 0.095,
    "cv_vowel": 0.069,
    "affricate": 0.050,
    "fricative": 0.061,
    "stop": 0.031,
    "nasal": 0.046,
    "approximant": 0.046,
    "other": 0.040,
    "geminate_closure": 0.073,
    "moraic_nasal": 0.077,
    "long_vowel": 0.095,
    "devoiced_high_vowel": 0.092,
}

# Kokoro contributes relative phone allocation, not its speaker's absolute
# rate.  These mora totals retain the pre-corpus project-speaker engineering
# register and are divided among phones using the fitted allocation values
# above.  Keeping the two concepts separate prevents a faster corpus speaker
# from silently becoming the runtime voice's timing baseline.
DEFAULT_MORA_ANCHOR_SECONDS = {
    "vowel_only": 0.110,
    "cv": 0.118,
    "obstruent_cv": 0.122,
    "geminate_closure": 0.082,
    "moraic_nasal": 0.088,
    "long_vowel": 0.108,
    "devoiced_high_vowel": 0.092,
    "other": 0.095,
}

# Total observed break targets are represented by the editable middle-gap
# value used by Japanese synthesis. The two protected 80 ms edge guards add
# 80 ms net to this number (the middle-gap helper subtracts one guard).
DEFAULT_PHRASE_PAUSES_MS = {
    "minor": 120.0,
    "major": 300.0,
    "sentence": 500.0,
}
DEFAULT_ACOUSTIC_EDGE_COMPENSATION_MS = {
    "phrase_initial_vowel": 0.0,
    "phrase_final_vowel": 0.0,
}


def phone_class(phone: str) -> str:
    symbol = str(phone)
    if symbol in _VOWELS:
        return "vowel"
    if symbol == "cl":
        return "geminate_closure"
    if symbol == "N":
        return "moraic_nasal"
    if symbol in _STOPS:
        return "stop"
    if symbol in _AFFRICATES:
        return "affricate"
    if symbol in _FRICATIVES:
        return "fricative"
    if symbol in _NASALS:
        return "nasal"
    if symbol in {"r", "w", "y"} or symbol.endswith("y"):
        return "approximant"
    return "other"


@dataclass(frozen=True)
class JapaneseDurationContext:
    phone: str
    phone_class: str
    phone_position_in_mora: int
    phone_count_in_mora: int
    previous_phone: Optional[str]
    following_phone: Optional[str]
    mora_index: int
    mora_position_in_accent_phrase: int
    mora_count_in_accent_phrase: int
    mora_position_in_phrase: int
    mora_count_in_phrase: int
    mora_position_in_utterance: int
    mora_count_in_utterance: int
    special_mora: Optional[str]
    accent_state: str
    accent_nucleus: Optional[int]
    accent_distance: Optional[int]
    accent_phrase_final: bool
    phrase_final: bool
    utterance_final: bool
    boundary_strength: int
    interrogative: bool
    openjtalk_devoiced: Optional[bool]
    likely_devoicing_environment: bool
    previous_mora_devoiced: bool
    raw_a: Optional[str] = None
    raw_f: Optional[str] = None
    raw_i: Optional[str] = None
    raw_k: Optional[str] = None
    openjtalk_mora_forward: Optional[int] = None
    openjtalk_mora_backward: Optional[int] = None
    openjtalk_breath_group_forward: Optional[int] = None
    openjtalk_breath_group_backward: Optional[int] = None
    lexical_surface: Optional[str] = None
    part_of_speech: Optional[str] = None
    grammatical_role: Optional[str] = None
    conjugation_type: Optional[str] = None
    conjugation_form: Optional[str] = None
    mora_position_in_node: Optional[int] = None
    mora_count_in_node: Optional[int] = None
    function_word: bool = False

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class JapaneseDurationPriors:
    model_id: str
    schema_version: int
    coefficients: Mapping[str, float]
    speed_elasticity: Mapping[str, float]
    class_reference_seconds: Mapping[str, float]
    minimum_context_ratio: float
    maximum_context_ratio: float
    cv_compensation_fraction: float
    source_geometry_ratio_bounds: Mapping[str, tuple[float, float]]
    class_target_ratio_bounds: Mapping[str, tuple[float, float]]
    mora_allocation_seconds: Mapping[str, float]
    mora_anchor_seconds: Mapping[str, float]
    phrase_pauses_ms: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PHRASE_PAUSES_MS))
    acoustic_edge_compensation_ms: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ACOUSTIC_EDGE_COMPENSATION_MS))
    fit_provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
                "coefficients", "speed_elasticity",
                "class_reference_seconds", "source_geometry_ratio_bounds",
                "class_target_ratio_bounds", "mora_allocation_seconds",
                "mora_anchor_seconds", "phrase_pauses_ms",
                "acoustic_edge_compensation_ms", "fit_provenance"):
            object.__setattr__(self, name, deep_freeze(getattr(self, name)))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "schema_version": self.schema_version,
            "coefficients": dict(sorted(self.coefficients.items())),
            "speed_elasticity": dict(sorted(self.speed_elasticity.items())),
            "class_reference_seconds": dict(sorted(
                self.class_reference_seconds.items()
            )),
            "minimum_context_ratio": self.minimum_context_ratio,
            "maximum_context_ratio": self.maximum_context_ratio,
            "cv_compensation_fraction": self.cv_compensation_fraction,
            "source_geometry_ratio_bounds": {
                key: list(value) for key, value in sorted(
                    self.source_geometry_ratio_bounds.items())
            },
            "class_target_ratio_bounds": {
                key: list(value) for key, value in sorted(
                    self.class_target_ratio_bounds.items())
            },
            "mora_allocation_seconds": dict(sorted(
                self.mora_allocation_seconds.items()
            )),
            "mora_anchor_seconds": dict(sorted(
                self.mora_anchor_seconds.items()
            )),
            "phrase_pauses_ms": dict(sorted(
                self.phrase_pauses_ms.items()
            )),
            "acoustic_edge_compensation_ms": dict(sorted(
                self.acoustic_edge_compensation_ms.items()
            )),
            "fit_provenance": dict(self.fit_provenance),
        }


@dataclass(frozen=True)
class JapaneseDurationPrediction:
    phone: str
    source_baseline_duration: float
    baseline_source: str
    source_profile_reference_duration: Optional[float]
    source_geometry_ratio: Optional[float]
    source_geometry_ratio_bounded: Optional[float]
    context_log_ratio: float
    speed_ratio: float
    predicted_duration: float
    effects: Mapping[str, float]
    context: JapaneseDurationContext
    model_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "phone": self.phone,
            "source_baseline_duration": self.source_baseline_duration,
            "baseline_source": self.baseline_source,
            "source_profile_reference_duration": (
                self.source_profile_reference_duration
            ),
            "source_geometry_ratio": self.source_geometry_ratio,
            "source_geometry_ratio_bounded": (
                self.source_geometry_ratio_bounded
            ),
            "context_log_ratio": self.context_log_ratio,
            "speed_ratio": self.speed_ratio,
            "predicted_duration": self.predicted_duration,
            "effects": dict(sorted(self.effects.items())),
            "context": self.context.to_dict(),
            "model_id": self.model_id,
        }


def _read_duration_priors(source: Path) -> JapaneseDurationPriors:
    data = json.loads(source.read_text(encoding="utf-8"))
    version = int(data.get("schema_version") or 0)
    if version != DURATION_MODEL_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Japanese duration-prior schema {version}"
        )
    geometry_bounds = {}
    for key, value in dict(data.get(
            "source_geometry_ratio_bounds") or {}).items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        low, high = float(value[0]), float(value[1])
        if 0.05 <= low <= high <= 10.0:
            geometry_bounds[str(key)] = (low, high)
    target_bounds = {}
    for key, value in dict(data.get(
            "class_target_ratio_bounds") or {}).items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        low, high = float(value[0]), float(value[1])
        if 0.05 <= low <= high <= 10.0:
            target_bounds[str(key)] = (low, high)
    allocations = dict(DEFAULT_MORA_ALLOCATION_SECONDS)
    for key, value in dict(data.get(
            "mora_allocation_seconds") or {}).items():
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if 0.005 <= seconds <= 1.0:
            allocations[str(key)] = seconds
    anchors = dict(DEFAULT_MORA_ANCHOR_SECONDS)
    for key, value in dict(data.get("mora_anchor_seconds") or {}).items():
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if 0.020 <= seconds <= 1.0:
            anchors[str(key)] = seconds
    phrase_pauses = dict(DEFAULT_PHRASE_PAUSES_MS)
    for key, value in dict(data.get("phrase_pauses_ms") or {}).items():
        if str(key) not in phrase_pauses:
            continue
        try:
            milliseconds = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= milliseconds <= 2000.0:
            phrase_pauses[str(key)] = milliseconds
    edge_compensation = dict(DEFAULT_ACOUSTIC_EDGE_COMPENSATION_MS)
    for key, value in dict(data.get(
            "acoustic_edge_compensation_ms") or {}).items():
        if str(key) not in edge_compensation:
            continue
        try:
            milliseconds = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= milliseconds <= 150.0:
            edge_compensation[str(key)] = milliseconds
    return JapaneseDurationPriors(
        model_id=str(data["model_id"]),
        schema_version=version,
        coefficients={str(key): float(value) for key, value in
                      dict(data.get("coefficients") or {}).items()},
        speed_elasticity={str(key): float(value) for key, value in
                          dict(data.get("speed_elasticity") or {}).items()},
        class_reference_seconds={str(key): float(value) for key, value in
                                 dict(data.get("class_reference_seconds") or
                                      {}).items()},
        minimum_context_ratio=float(data.get("minimum_context_ratio", 0.45)),
        maximum_context_ratio=float(data.get("maximum_context_ratio", 1.45)),
        cv_compensation_fraction=float(data.get(
            "cv_compensation_fraction", 0.22
        )),
        source_geometry_ratio_bounds=geometry_bounds,
        class_target_ratio_bounds=target_bounds,
        mora_allocation_seconds=allocations,
        mora_anchor_seconds=anchors,
        phrase_pauses_ms=phrase_pauses,
        acoustic_edge_compensation_ms=edge_compensation,
        fit_provenance=dict(data.get("fit_provenance") or {}),
    )


def load_duration_priors(path: Path | str | None = None) \
        -> JapaneseDurationPriors:
    """Load immutable priors through a file-identity-aware bounded cache."""
    source = Path(path) if path is not None else DEFAULT_PRIORS_PATH
    return _DURATION_PRIORS_CACHE.get(source, _read_duration_priors)


def duration_model_cache_info() -> dict[str, int | str]:
    return _DURATION_PRIORS_CACHE.info()


def clear_duration_model_cache() -> dict[str, int | str]:
    return _DURATION_PRIORS_CACHE.clear()


def _raw_context(mora: JapaneseMora):
    labels = [phone.raw_label for phone in mora.phones if phone.raw_label]
    if not labels:
        return None
    preferred = next((
        phone.raw_label for phone in reversed(mora.phones)
        if phone.raw_label and phone.symbol in _VOWELS
    ), labels[-1])
    try:
        return parse_full_context_label(str(preferred))
    except (TypeError, ValueError, IndexError):
        return None


def build_duration_contexts(
    utterance: JapaneseUtterance,
    mora: JapaneseMora,
    phones: Sequence[str],
) -> tuple[JapaneseDurationContext, ...]:
    all_moras = list(utterance.moras)
    mora_position = next(
        (index for index, item in enumerate(all_moras)
         if item.index == mora.index),
        mora.index,
    )
    phrase = next(item for item in utterance.phrases
                  if item.index == mora.phrase_index)
    accent = next(item for item in phrase.accent_phrases
                  if item.index == mora.accent_phrase_index)
    phrase_moras = list(phrase.moras)
    accent_moras = list(accent.moras)
    phrase_position = next(index for index, item in enumerate(phrase_moras)
                           if item.index == mora.index)
    accent_position = next(index for index, item in enumerate(accent_moras)
                           if item.index == mora.index)
    previous_mora = all_moras[mora_position - 1] if mora_position else None
    following_mora = (all_moras[mora_position + 1]
                      if mora_position + 1 < len(all_moras) else None)
    previous_symbol = next((
        phone.symbol for phone in reversed(previous_mora.phones)
        if phone.symbol not in {"pau", "sil"}
    ), None) if previous_mora else None
    following_symbol = next((
        phone.symbol for phone in following_mora.phones
        if phone.symbol not in {"pau", "sil"}
    ), None) if following_mora else None
    sequence = tuple(str(item) for item in phones)
    previous_symbols = tuple(
        phone.symbol for phone in (previous_mora.phones if previous_mora else ())
        if phone.symbol not in {"pau", "sil"}
    )
    previous_predicted_devoicing = bool(
        previous_mora
        and previous_mora.special_mora != "long_vowel"
        and any(symbol in {"i", "u"} for symbol in previous_symbols)
        and previous_symbols
        and previous_symbols[0] in _VOICELESS
        and sequence
        and sequence[0] in _VOICELESS
    )
    raw = _raw_context(mora)
    morphology = dict(mora.provenance.get("morphology") or {})
    openjtalk_devoiced = True if mora.devoiced is True else (
        False if mora.devoiced is False else None
    )
    result = []
    for local_index, phone in enumerate(sequence):
        previous = sequence[local_index - 1] if local_index else previous_symbol
        following = (sequence[local_index + 1]
                     if local_index + 1 < len(sequence) else following_symbol)
        likely = (
            phone in {"i", "u"}
            and previous in _VOICELESS
            and (following in _VOICELESS or following is None)
        )
        # Runtime profiles may map canonical N to bank-specific symbols such
        # as nn, mm, nng, or xn.  Its moraic role must survive that mapping;
        # classifying the rendered alias alone turns it into a consonant and
        # lets held source geometry dominate the target duration.
        semantic_class = (
            "moraic_nasal"
            if mora.special_mora == "moraic_nasal"
            else phone_class(phone)
        )
        result.append(JapaneseDurationContext(
            phone=phone,
            phone_class=semantic_class,
            phone_position_in_mora=local_index,
            phone_count_in_mora=len(sequence),
            previous_phone=previous,
            following_phone=following,
            mora_index=mora.index,
            mora_position_in_accent_phrase=accent_position,
            mora_count_in_accent_phrase=len(accent_moras),
            mora_position_in_phrase=phrase_position,
            mora_count_in_phrase=len(phrase_moras),
            mora_position_in_utterance=mora_position,
            mora_count_in_utterance=len(all_moras),
            special_mora=mora.special_mora,
            accent_state=accent.accent_state,
            accent_nucleus=accent.accent_nucleus,
            accent_distance=(
                accent_position - accent.accent_nucleus
                if accent.accent_nucleus is not None else None
            ),
            accent_phrase_final=accent_position == len(accent_moras) - 1,
            phrase_final=phrase_position == len(phrase_moras) - 1,
            utterance_final=mora_position == len(all_moras) - 1,
            boundary_strength=phrase.boundary_strength,
            interrogative=phrase.interrogative or accent.interrogative,
            openjtalk_devoiced=openjtalk_devoiced,
            likely_devoicing_environment=likely,
            previous_mora_devoiced=bool(
                previous_mora and (
                    previous_mora.devoiced is True
                    or previous_predicted_devoicing
                )
            ),
            raw_a=raw.mora.raw if raw and raw.mora else None,
            raw_f=(raw.accent_phrase.raw
                   if raw and raw.accent_phrase else None),
            raw_i=(raw.breath_group.raw
                   if raw and raw.breath_group else None),
            raw_k=raw.utterance.raw if raw and raw.utterance else None,
            openjtalk_mora_forward=(
                raw.mora.position_forward if raw and raw.mora else None
            ),
            openjtalk_mora_backward=(
                raw.mora.position_backward if raw and raw.mora else None
            ),
            openjtalk_breath_group_forward=(
                raw.breath_group.position_forward_in_utterance
                if raw and raw.breath_group else None
            ),
            openjtalk_breath_group_backward=(
                raw.breath_group.position_backward_in_utterance
                if raw and raw.breath_group else None
            ),
            lexical_surface=(
                str(morphology.get("string") or morphology.get("orig"))
                if morphology.get("string") or morphology.get("orig")
                else None
            ),
            part_of_speech=(
                str(morphology["pos"]) if morphology.get("pos") else None
            ),
            grammatical_role=(
                str(morphology["grammatical_role"])
                if morphology.get("grammatical_role") else None
            ),
            conjugation_type=(
                str(morphology["ctype"])
                if morphology.get("ctype") else None
            ),
            conjugation_form=(
                str(morphology["cform"])
                if morphology.get("cform") else None
            ),
            mora_position_in_node=(
                int(morphology["mora_position_in_node_zero_based"])
                if morphology.get("mora_position_in_node_zero_based")
                is not None else None
            ),
            mora_count_in_node=(
                int(morphology["mora_count_in_node"])
                if morphology.get("mora_count_in_node") is not None else None
            ),
            function_word=bool(morphology.get("function_word", False)),
        ))
    return tuple(result)


def _effect_map(context: JapaneseDurationContext,
                priors: JapaneseDurationPriors) -> dict[str, float]:
    c = priors.coefficients
    effects: dict[str, float] = {}

    def add(name: str, condition: bool, scale: float = 1.0):
        if condition and name in c:
            effects[name] = float(c[name]) * float(scale)

    is_rhyme = context.phone_class in {"vowel", "moraic_nasal"}
    add("accent_phrase_final_rhyme", is_rhyme and context.accent_phrase_final)
    add(
        "phrase_final_rhyme",
        is_rhyme and context.phrase_final,
        max(0.0, min(1.0, context.boundary_strength / 3.0)),
    )
    add("utterance_final_rhyme", is_rhyme and context.utterance_final)
    add("interrogative_final_rhyme",
        is_rhyme and context.phrase_final and context.interrogative)
    add("accent_nucleus_vowel", context.phone_class == "vowel" and
        context.accent_distance == 0)
    add("geminate_closure", context.phone_class == "geminate_closure")
    add("moraic_nasal", context.phone_class == "moraic_nasal")
    add(
        "moraic_nasal_before_voiceless",
        context.phone_class == "moraic_nasal"
        and context.following_phone in _VOICELESS,
    )
    add("long_vowel", context.special_mora == "long_vowel" and
        context.phone_class == "vowel")
    should_devoice = (
        context.phone in {"i", "u"}
        and (context.openjtalk_devoiced is True or
             context.likely_devoicing_environment)
        and not context.previous_mora_devoiced
        and context.special_mora != "long_vowel"
    )
    add("devoiced_high_vowel", should_devoice)
    add("consecutive_devoicing_avoidance", context.phone in {"i", "u"}
        and context.previous_mora_devoiced)
    add("auxiliary", context.grammatical_role == "auxiliary")
    add("negative_auxiliary",
        context.grammatical_role == "negative_auxiliary")
    return effects


def predict_mora_durations(
    contexts: Sequence[JapaneseDurationContext],
    source_references: Sequence[Optional[float]],
    fallback_durations: Sequence[float],
    *,
    speed: float,
    source_profile_references: Sequence[Optional[float]] | None = None,
    priors: JapaneseDurationPriors | None = None,
) -> tuple[JapaneseDurationPrediction, ...]:
    """Predict one mora while retaining source-speaker timing variation.

    Diphone half-spans are source material, not literal speech durations.  A
    bank profile removes each phone's held-note/recording-collar scale.  The
    remaining occurrence ratio modulates the existing source-derived speaker
    baseline before bounded context residuals are applied.
    """
    model = priors or load_duration_priors()
    if not (len(contexts) == len(source_references) == len(fallback_durations)):
        raise ValueError("duration context/reference lengths do not match")
    if source_profile_references is None:
        source_profile_references = tuple(source_references)
    if len(source_profile_references) != len(contexts):
        raise ValueError("duration source-profile length does not match")
    speed = max(0.25, min(4.0, float(speed)))
    rows = []
    for context, reference, profile_reference, fallback in zip(
            contexts, source_references, source_profile_references,
            fallback_durations):
        has_source = reference is not None and float(reference) > 0.001
        source_anchor = max(0.010, float(fallback) * speed)
        raw_geometry_ratio = None
        bounded_geometry_ratio = None
        if has_source:
            expected = (
                float(profile_reference)
                if profile_reference is not None
                and float(profile_reference) > 0.001
                else float(reference)
            )
            raw_geometry_ratio = float(reference) / expected
            low, high = model.source_geometry_ratio_bounds.get(
                context.phone_class, (0.60, 1.70)
            )
            bounded_geometry_ratio = max(
                float(low), min(float(high), raw_geometry_ratio)
            )
            baseline = source_anchor * bounded_geometry_ratio
        else:
            expected = None
            baseline = source_anchor
        effects = _effect_map(context, model)
        log_ratio = sum(effects.values())
        ratio = math.exp(log_ratio)
        ratio = max(model.minimum_context_ratio,
                    min(model.maximum_context_ratio, ratio))
        elasticity = max(0.0, min(
            1.2, float(model.speed_elasticity.get(context.phone_class, 0.8))
        ))
        speed_ratio = speed ** (-elasticity)
        target = baseline * ratio * speed_ratio
        target_bounds = model.class_target_ratio_bounds.get(
            context.phone_class
        )
        if target_bounds is not None:
            class_reference = float(model.class_reference_seconds.get(
                context.phone_class, source_anchor
            ))
            minimum_target = (class_reference * float(target_bounds[0])
                              * speed_ratio)
            maximum_target = (class_reference * float(target_bounds[1])
                              * speed_ratio)
            bounded_target = max(minimum_target, min(maximum_target, target))
            if abs(bounded_target - target) > 1e-9:
                bound_effect = math.log(
                    max(1e-9, bounded_target) / max(1e-9, target)
                )
                effects = {
                    **effects,
                    "class_duration_bound": bound_effect,
                }
                log_ratio += bound_effect
                target = bounded_target
        rows.append({
            "context": context,
            "baseline": baseline,
            "baseline_source": "source_unit_geometry_profiled" if has_source
            else "legacy_class_fallback",
            "profile_reference": expected,
            "geometry_ratio": raw_geometry_ratio,
            "geometry_ratio_bounded": bounded_geometry_ratio,
            "effects": effects,
            "log_ratio": math.log(ratio),
            "speed_ratio": speed_ratio,
            "target": target,
        })

    vowel_positions = [index for index, row in enumerate(rows)
                       if row["context"].phone_class == "vowel"]
    consonant_positions = [index for index, row in enumerate(rows)
                           if row["context"].phone_class not in
                           {"vowel", "moraic_nasal", "geminate_closure"}]
    if vowel_positions and consonant_positions:
        consonant_baseline = sum(rows[index]["baseline"]
                                 for index in consonant_positions)
        expected = sum(model.class_reference_seconds.get(
            rows[index]["context"].phone_class, rows[index]["baseline"]
        ) for index in consonant_positions)
        delta = max(-0.060, min(0.060, consonant_baseline - expected))
        compensation = model.cv_compensation_fraction * delta
        per_vowel = compensation / len(vowel_positions)
        for index in vowel_positions:
            before = float(rows[index]["target"])
            after = max(0.012, before - per_vowel)
            if abs(after - before) > 1e-9:
                rows[index]["effects"] = {
                    **rows[index]["effects"],
                    "cv_partial_compensation_seconds": -per_vowel,
                }
                rows[index]["target"] = after

    # Phrase-edge OTO material can be audible outside Festival's requested
    # Segment boundary.  These are additive acoustic calibrations, measured
    # source-relative on the fixed training corpus, rather than grammatical
    # duration coefficients.  A proportional cap keeps one-mora phrases and
    # high-speed speech from collapsing when both edge conditions apply.
    for row in rows:
        context = row["context"]
        requested = 0.0
        names = []
        if (context.phone_class == "vowel"
                and context.mora_position_in_phrase == 0
                and context.phone_position_in_mora == 0):
            value = float(model.acoustic_edge_compensation_ms.get(
                "phrase_initial_vowel", 0.0)) / 1000.0
            requested += value
            if value > 0.0:
                names.append("phrase_initial_vowel")
        if context.phone_class == "vowel" and context.phrase_final:
            value = float(model.acoustic_edge_compensation_ms.get(
                "phrase_final_vowel", 0.0)) / 1000.0
            requested += value
            if value > 0.0:
                names.append("phrase_final_vowel")
        before = float(row["target"])
        maximum = max(0.0, min(before * 0.55, before - 0.030))
        reduction = min(requested, maximum)
        if reduction > 1e-9:
            row["target"] = before - reduction
            scale = reduction / max(1e-12, requested)
            row["effects"] = {
                **row["effects"],
                **{
                    f"{name}_acoustic_compensation_seconds": -(
                        float(model.acoustic_edge_compensation_ms[name])
                        / 1000.0 * scale
                    )
                    for name in names
                },
            }

    return tuple(JapaneseDurationPrediction(
        phone=row["context"].phone,
        source_baseline_duration=round(float(row["baseline"]), 6),
        baseline_source=str(row["baseline_source"]),
        source_profile_reference_duration=(
            round(float(row["profile_reference"]), 6)
            if row["profile_reference"] is not None else None
        ),
        source_geometry_ratio=(
            round(float(row["geometry_ratio"]), 6)
            if row["geometry_ratio"] is not None else None
        ),
        source_geometry_ratio_bounded=(
            round(float(row["geometry_ratio_bounded"]), 6)
            if row["geometry_ratio_bounded"] is not None else None
        ),
        context_log_ratio=round(float(row["log_ratio"]), 9),
        speed_ratio=round(float(row["speed_ratio"]), 9),
        predicted_duration=round(max(0.010, float(row["target"])), 6),
        effects=dict(row["effects"]),
        context=row["context"],
        model_id=model.model_id,
    ) for row in rows)
