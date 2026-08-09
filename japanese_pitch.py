"""Versioned, speaker-relative log-F0 model for Japanese prosody."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from cache_support import FileIdentityCache, deep_freeze
from japanese_models import JapaneseAccentPhrase, JapaneseUtterance


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "profiles"
    / "japanese_pitch_model_v1.json"
)
_PITCH_MODEL_CACHE = FileIdentityCache(
    "japanese-pitch-model", max_entries=4, max_bytes=8 * 1024 * 1024
)


@dataclass(frozen=True)
class JapanesePitchModel:
    model_id: str
    model_version: int
    components: Mapping[str, float]
    headroom_below_semitones: float
    headroom_above_semitones: float
    maximum_downstep_accents: int
    later_phrase_declination_delta_semitones: float
    later_phrase_accent_contrast_scale: float
    later_phrase_boundary_delta_semitones: float
    maximum_contextual_phrase_index: int
    analysis_evidence: Mapping[str, object]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "components", deep_freeze(self.components))
        object.__setattr__(
            self, "analysis_evidence", deep_freeze(self.analysis_evidence))
        object.__setattr__(
            self, "provenance", deep_freeze(self.provenance))

    def component(self, name: str) -> float:
        if name not in self.components:
            raise KeyError(f"Japanese pitch component is missing: {name}")
        return float(self.components[name])


@dataclass(frozen=True)
class JapaneseMoraPitchTarget:
    """One speaker-relative structural target before conversion to Hz."""

    mora_index: int
    phrase_index: int
    accent_phrase_index: int
    semitones_from_baseline: float
    kind: str
    components_semitones: Mapping[str, float]


def _accent_semitones(
    accent_phrase: JapaneseAccentPhrase,
    local_mora_index: int,
    model: JapanesePitchModel,
) -> tuple[float, str]:
    if accent_phrase.accent_state == "accented":
        nucleus = int(accent_phrase.accent_nucleus or 0)
        if nucleus == 0:
            return (
                (model.component("initial_accent_nucleus"), "accent_nucleus")
                if local_mora_index == 0
                else (model.component("post_accent_drop"), "post_accent_drop")
            )
        if local_mora_index == 0:
            return model.component("initial_low"), "initial_low"
        if local_mora_index <= nucleus:
            return (
                (model.component("accent_nucleus"), "accent_nucleus")
                if local_mora_index == nucleus
                else (model.component("pre_accent_high"), "pre_accent_high")
            )
        return model.component("post_accent_drop"), "post_accent_drop"
    if accent_phrase.accent_state == "unaccented":
        if local_mora_index == 0 and len(accent_phrase.moras) > 1:
            return (
                model.component("unaccented_initial_low"),
                "unaccented_initial_low",
            )
        return model.component("unaccented_high"), "unaccented_high"
    if local_mora_index == 0:
        return model.component("neutral_initial"), "neutral_initial"
    return model.component("neutral"), "neutral"


def mora_pitch_contour(
    utterance: JapaneseUtterance,
    model: JapanesePitchModel | None = None,
    mora_times_seconds: Mapping[int, float] | None = None,
) -> tuple[JapaneseMoraPitchTarget, ...]:
    """Build one utterance-scoped Japanese contour in semitones.

    Lexical accent and accent-phrase downstep restart inside each breath
    group.  Later phrases vary in *shape*, as Kal's learned start/mid/end F0
    models do through phrase-position features, but never through a cumulative
    register offset.  Every later-phrase component is mean-centred over that
    phrase, so repeated material can differ in reset, accent contrast, and
    boundary realization without frequency drift.

    ``mora_times_seconds`` remains part of the public API because synthesis
    supplies it and older callers may rely on the signature.  Utterance time
    is deliberately not used as an F0-offset predictor.
    """
    pitch_model = model or load_japanese_pitch_model()
    _ = mora_times_seconds
    result: list[JapaneseMoraPitchTarget] = []
    for phrase_position, phrase in enumerate(utterance.phrases):
        phrase_moras = tuple(phrase.moras)
        phrase_lexical = {}
        for accent_phrase in phrase.accent_phrases:
            for local_index, mora in enumerate(accent_phrase.moras):
                phrase_lexical[mora.index] = _accent_semitones(
                    accent_phrase, local_index, pitch_model
                )[0]
        lexical_mean = (
            sum(phrase_lexical.values()) / len(phrase_lexical)
            if phrase_lexical else 0.0
        )
        contextual_phrase_index = min(
            phrase_position, pitch_model.maximum_contextual_phrase_index
        )
        accented_before = 0
        local_phrase_position = 0
        for accent_position, accent_phrase in enumerate(
                phrase.accent_phrases):
            local_count = len(accent_phrase.moras)
            for local_index, mora in enumerate(accent_phrase.moras):
                lexical, lexical_kind = _accent_semitones(
                    accent_phrase, local_index, pitch_model
                )
                phrase_progress = local_phrase_position / max(
                    1, len(phrase_moras) - 1
                )
                local_progress = local_index / max(1, local_count - 1)
                centered_phrase_progress = (
                    phrase_progress - 0.5
                    if len(phrase_moras) > 1 else 0.0
                )
                centered_boundary = (
                    (1.0 if local_phrase_position == len(phrase_moras) - 1
                     else 0.0)
                    - (1.0 / max(1, len(phrase_moras)))
                )
                components = {
                    "lexical_accent": lexical,
                    "breath_group_reset": pitch_model.component(
                        "breath_group_reset"
                    ) * (1.0 - phrase_progress),
                    # Preserve the established phrase-local shape separately
                    # from the gentler sentence-time trend. That distinction
                    # keeps later repeated phrases different without forcing
                    # every individual phrase into a steep global fall.
                    "phrase_declination": pitch_model.component(
                        "phrase_declination_total"
                    ) * phrase_progress,
                    "later_phrase_declination_shape": (
                        pitch_model.later_phrase_declination_delta_semitones
                        * contextual_phrase_index
                        * centered_phrase_progress
                    ),
                    "later_phrase_accent_shape": (
                        pitch_model.later_phrase_accent_contrast_scale
                        * contextual_phrase_index
                        * (lexical - lexical_mean)
                    ),
                    "later_phrase_boundary_shape": (
                        pitch_model.later_phrase_boundary_delta_semitones
                        * contextual_phrase_index
                        * centered_boundary
                    ),
                    "accent_phrase_declination": pitch_model.component(
                        "local_declination_total"
                    ) * local_progress,
                    "accent_phrase_reset": (
                        pitch_model.component("accent_phrase_reset")
                        if accent_position > 0 and local_index == 0 else 0.0
                    ),
                    "downstep": pitch_model.component(
                        "downstep_per_accent"
                    ) * min(
                        accented_before,
                        pitch_model.maximum_downstep_accents,
                    ),
                    "phrase_boundary": (
                        pitch_model.component("phrase_final_base")
                        + pitch_model.component(
                            "phrase_final_per_boundary_strength"
                        ) * phrase.boundary_strength
                        if local_phrase_position == len(phrase_moras) - 1
                        else 0.0
                    ),
                }
                kinds = [lexical_kind, "breath_group_contour"]
                if contextual_phrase_index:
                    kinds.append("later_phrase_shape")
                if components["accent_phrase_reset"]:
                    kinds.append("accent_phrase_reset")
                if components["downstep"]:
                    kinds.append("downstep")
                if components["phrase_boundary"]:
                    kinds.append("phrase_final_lowering")
                result.append(JapaneseMoraPitchTarget(
                    mora_index=mora.index,
                    phrase_index=phrase.index,
                    accent_phrase_index=accent_phrase.index,
                    semitones_from_baseline=round(
                        sum(components.values()), 9
                    ),
                    kind="+".join(kinds),
                    components_semitones={
                        key: round(value, 9)
                        for key, value in components.items()
                    },
                ))
                local_phrase_position += 1
            if accent_phrase.accent_state == "accented":
                accented_before += 1
    return tuple(result)


def _read_japanese_pitch_model(source: Path) -> JapanesePitchModel:
    data = json.loads(source.read_text(encoding="utf-8"))
    if int(data.get("schema_version") or 0) != 1:
        raise ValueError("unsupported Japanese pitch model schema")
    if str(data.get("language") or "") != "ja":
        raise ValueError("Japanese pitch model language must be ja")
    components = {
        str(key): float(value)
        for key, value in dict(data.get("components_semitones") or {}).items()
    }
    required = {
        "accent_nucleus", "accent_phrase_reset", "breath_group_reset",
        "downstep_per_accent", "initial_accent_nucleus", "initial_low",
        "local_declination_total", "neutral", "neutral_initial",
        "phrase_final_base", "phrase_final_per_boundary_strength",
        "post_accent_drop", "pre_accent_high", "unaccented_high",
        "unaccented_initial_low", "phrase_declination_total",
    }
    missing = sorted(required.difference(components))
    if missing:
        raise ValueError("Japanese pitch components missing: " +
                         ", ".join(missing))
    headroom = dict(data.get("psola_safe_headroom_semitones") or {})
    below = float(headroom.get("below_baseline") or 0.0)
    above = float(headroom.get("above_baseline") or 0.0)
    if not 1.0 <= below <= 12.0 or not 1.0 <= above <= 12.0:
        raise ValueError("Japanese pitch headroom must be 1..12 semitones")
    temporal = dict(data.get("temporal_model") or {})
    legacy_drift_keys = {
        "utterance_declination_semitones_per_second",
        "maximum_utterance_declination_semitones",
        "phrase_register_drop_semitones",
        "maximum_phrase_register_drops",
    }
    active_legacy_drift = {
        key: temporal[key] for key in legacy_drift_keys
        if key in temporal and float(temporal[key]) != 0.0
    }
    if active_legacy_drift:
        raise ValueError(
            "Japanese pitch profiles may not contain cumulative frequency "
            "drift; use mean-centred later-phrase shape parameters"
        )
    later_declination = float(temporal.get(
        "later_phrase_declination_delta_semitones", 0.0))
    later_accent_scale = float(temporal.get(
        "later_phrase_accent_contrast_scale", 0.0))
    later_boundary = float(temporal.get(
        "later_phrase_boundary_delta_semitones", 0.0))
    maximum_contextual_phrase_index = int(temporal.get(
        "maximum_contextual_phrase_index", 0))
    if not -2.0 <= later_declination <= 2.0:
        raise ValueError("Japanese later-phrase decline delta must be -2..2")
    if not -0.6 <= later_accent_scale <= 0.6:
        raise ValueError("Japanese later-phrase accent scale must be -0.6..0.6")
    if not -3.0 <= later_boundary <= 3.0:
        raise ValueError("Japanese later-phrase boundary delta must be -3..3")
    if not 0 <= maximum_contextual_phrase_index <= 8:
        raise ValueError("Japanese contextual phrase index cap must be 0..8")
    return JapanesePitchModel(
        model_id=str(data.get("model_id") or ""),
        model_version=int(data.get("model_version") or 0),
        components=components,
        headroom_below_semitones=below,
        headroom_above_semitones=above,
        maximum_downstep_accents=max(
            0, int(data.get("maximum_downstep_accents") or 0)),
        later_phrase_declination_delta_semitones=later_declination,
        later_phrase_accent_contrast_scale=later_accent_scale,
        later_phrase_boundary_delta_semitones=later_boundary,
        maximum_contextual_phrase_index=maximum_contextual_phrase_index,
        analysis_evidence=dict(data.get("analysis_evidence") or {}),
        provenance=dict(data.get("provenance") or {}),
    )


def load_japanese_pitch_model(
    path: Path | str | None = None,
) -> JapanesePitchModel:
    """Load an immutable pitch profile with automatic file invalidation."""
    source = Path(path) if path is not None else DEFAULT_MODEL_PATH
    return _PITCH_MODEL_CACHE.get(source, _read_japanese_pitch_model)


def pitch_model_cache_info() -> dict[str, int | str]:
    return _PITCH_MODEL_CACHE.info()


def clear_pitch_model_cache() -> dict[str, int | str]:
    return _PITCH_MODEL_CACHE.clear()
