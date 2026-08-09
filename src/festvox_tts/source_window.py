"""Deterministic source-window plans for generated FestVox diphones.

An OTO region can be much longer than the target phone.  Mapping that entire
region into a short Segment makes UniSyn compress unrelated source trajectory
into conversational timing.  These plans keep a bounded primary window while
retaining optional full-side variants for genuinely long target phones.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


SOURCE_WINDOW_MODES = ("adaptive", "bounded", "full")
DEFAULT_SOURCE_WINDOW_MS = 60.0
DEFAULT_ZERO_OVERLAP_GUARD_MS = 0.0
_MIN_HALF_SECONDS = 0.002


def normalize_zero_overlap_guard_ms(value: float) -> float:
    guard = float(value)
    if not math.isfinite(guard) or not 0.0 <= guard <= 60.0:
        raise ValueError("zero-overlap guard must be between 0 and 60 ms")
    return guard


def effective_oto_overlap_ms(
    preutterance_ms: float,
    overlap_ms: float,
    *,
    zero_overlap_guard_ms: float = DEFAULT_ZERO_OVERLAP_GUARD_MS,
) -> float:
    """Return the source-safe overlap anchor without rewriting the OTO.

    A positive OTO overlap is authoritative. A nonzero guard is an explicit
    diagnostic experiment which moves a zero/negative-overlap source onset by
    at most one quarter of preutterance. It is disabled by default because
    moving the OTO cut can damage the following handoff; a true repair belongs
    in overlap synthesis rather than source geometry.
    """
    preutterance = max(0.0, float(preutterance_ms))
    overlap = float(overlap_ms)
    guard = normalize_zero_overlap_guard_ms(zero_overlap_guard_ms)
    maximum = max(0.0, preutterance - 2.0)
    if overlap > 0.0:
        return min(overlap, maximum)
    if guard <= 0.0 or maximum <= 0.0:
        return 0.0
    return min(guard, preutterance * 0.25, maximum)


def normalize_source_window_mode(value: str) -> str:
    mode = str(value or "adaptive").strip().casefold()
    if mode not in SOURCE_WINDOW_MODES:
        raise ValueError(
            "source window mode must be one of: "
            + ", ".join(SOURCE_WINDOW_MODES)
        )
    return mode


@dataclass(frozen=True)
class SourceWindowPlan:
    mode: str
    half_window_seconds: float
    full_start: float
    midpoint: float
    full_end: float
    primary_start: float
    primary_end: float
    left_activation_duration: Optional[float]
    right_activation_duration: Optional[float]

    def geometry(self, kind: str = "base") -> tuple[float, float, float]:
        kind = str(kind or "base").casefold()
        if kind not in {"base", "left", "right", "both"}:
            raise ValueError("source window kind must be base, left, right, or both")
        # Only adaptive mode exposes hidden full-side variants. Bounded mode
        # retains the original OTO span as provenance but never selects it.
        # Full mode already uses that span as its primary geometry.
        use_left = self.mode != "bounded" and kind in {"left", "both"}
        use_right = self.mode != "bounded" and kind in {"right", "both"}
        start = self.full_start if use_left else self.primary_start
        end = self.full_end if use_right else self.primary_end
        return start, self.midpoint, end

    def variant_kind(
        self,
        left_phone_duration: float,
        right_phone_duration: float,
    ) -> str:
        """Choose a window without changing the selected recording.

        A source half is exposed in full only when one half of the requested
        phone is at least as long as that source half.  This avoids selecting
        a full recording merely because a phone is modestly lengthened.
        """
        if self.mode != "adaptive":
            return "base"
        left = (
            self.left_activation_duration is not None
            and float(left_phone_duration) >= self.left_activation_duration
        )
        right = (
            self.right_activation_duration is not None
            and float(right_phone_duration) >= self.right_activation_duration
        )
        if left and right:
            return "both"
        if left:
            return "left"
        if right:
            return "right"
        return "base"

    def to_dict(self) -> dict[str, object]:
        def region(kind: str) -> dict[str, float]:
            start, midpoint, end = self.geometry(kind)
            return {
                "start": round(start, 6),
                "phone_boundary": round(midpoint, 6),
                "end": round(end, 6),
            }

        full_region = {
            "start": round(self.full_start, 6),
            "phone_boundary": round(self.midpoint, 6),
            "end": round(self.full_end, 6),
        }

        return {
            "mode": self.mode,
            "half_window_ms": round(self.half_window_seconds * 1000.0, 6),
            "primary": region("base"),
            "full": full_region,
            "variants": {
                kind: region(kind)
                for kind in ("base", "left", "right", "both")
            },
            "activation_phone_durations": {
                "left": (
                    round(self.left_activation_duration, 6)
                    if self.left_activation_duration is not None else None
                ),
                "right": (
                    round(self.right_activation_duration, 6)
                    if self.right_activation_duration is not None else None
                ),
            },
        }


def build_source_window_plan(
    start: float,
    midpoint: float,
    end: float,
    *,
    mode: str = "adaptive",
    half_window_ms: float = DEFAULT_SOURCE_WINDOW_MS,
) -> SourceWindowPlan:
    mode = normalize_source_window_mode(mode)
    values = (float(start), float(midpoint), float(end))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("source window geometry must be finite")
    start, midpoint, end = values
    if not start < midpoint < end:
        raise ValueError("source window geometry must satisfy start < mid < end")
    half_seconds = max(
        _MIN_HALF_SECONDS, float(half_window_ms) / 1000.0
    )
    if not math.isfinite(half_seconds):
        raise ValueError("source window size must be finite")

    if mode == "full":
        primary_start, primary_end = start, end
    else:
        primary_start = max(start, midpoint - half_seconds)
        primary_end = min(end, midpoint + half_seconds)

    left_bounded = primary_start > start + 1.0e-9
    right_bounded = primary_end < end - 1.0e-9
    return SourceWindowPlan(
        mode=mode,
        half_window_seconds=half_seconds,
        full_start=start,
        midpoint=midpoint,
        full_end=end,
        primary_start=primary_start,
        primary_end=primary_end,
        left_activation_duration=(
            2.0 * (midpoint - start)
            if mode == "adaptive" and left_bounded else None
        ),
        right_activation_duration=(
            2.0 * (end - midpoint)
            if mode == "adaptive" and right_bounded else None
        ),
    )


def source_window_variant_names(
    base_name: str,
    plan: SourceWindowPlan,
) -> dict[str, str]:
    """Assign stable hidden unit names, coalescing duplicate geometries."""
    names = {"base": str(base_name)}
    geometry_names = {plan.geometry("base"): str(base_name)}
    suffixes = {"left": "__wl", "right": "__wr", "both": "__wb"}
    for kind in ("left", "right", "both"):
        geometry = plan.geometry(kind)
        name = geometry_names.get(geometry)
        if name is None:
            name = str(base_name) + suffixes[kind]
            geometry_names[geometry] = name
        names[kind] = name
    return names
