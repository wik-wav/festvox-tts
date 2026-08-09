"""Small, explicit dispatcher for Phase 1 Japanese linguistic frontends."""

from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import replace
import threading
from typing import Protocol, runtime_checkable

from cache_support import estimate_size_bytes
from japanese_kana_frontend import KanaJapaneseFrontend
from japanese_models import JapaneseFrontendDiagnostic, JapaneseUtterance
from japanese_openjtalk import (
    OpenJTalkFrontendError,
    OpenJTalkJapaneseFrontend,
    OpenJTalkUnavailableError,
    is_pyopenjtalk_available,
)


FRONTEND_MODES = ("auto", "openjtalk", "kana")
_FRONTEND_CACHE_MAX_ENTRIES = 128
_FRONTEND_CACHE_MAX_BYTES = 16 * 1024 * 1024
_FRONTEND_CACHE = OrderedDict()
_FRONTEND_CACHE_BYTES = 0
_FRONTEND_CACHE_HITS = 0
_FRONTEND_CACHE_MISSES = 0
_FRONTEND_CACHE_LOCK = threading.RLock()


@runtime_checkable
class JapaneseFrontend(Protocol):
    name: str

    def analyze(self, text: str) -> JapaneseUtterance:
        ...


def _validate_mode(mode: str) -> str:
    normalized = mode.casefold().strip()
    if normalized not in FRONTEND_MODES:
        raise ValueError(
            f"Japanese frontend mode must be one of {FRONTEND_MODES}, "
            f"got {mode!r}"
        )
    return normalized


def resolve_japanese_frontend(mode: str = "auto") -> JapaneseFrontend:
    """Resolve a frontend deterministically without analyzing any text."""
    normalized = _validate_mode(mode)
    if normalized == "kana":
        return KanaJapaneseFrontend()
    if normalized == "openjtalk":
        if not is_pyopenjtalk_available():
            raise OpenJTalkUnavailableError(JapaneseFrontendDiagnostic(
                code="pyopenjtalk_unavailable",
                message=(
                    "The Open JTalk frontend was requested, but pyopenjtalk "
                    "is not installed."
                ),
                severity="error",
                action="Install pyopenjtalk locally or select kana.",
                frontend="openjtalk",
                confidence=1.0,
            ))
        return OpenJTalkJapaneseFrontend()
    if is_pyopenjtalk_available():
        return OpenJTalkJapaneseFrontend()
    return KanaJapaneseFrontend()


def _prepend_diagnostic(
    utterance: JapaneseUtterance,
    diagnostic: JapaneseFrontendDiagnostic,
) -> JapaneseUtterance:
    return replace(
        utterance,
        diagnostics=(diagnostic,) + utterance.diagnostics,
        provenance={
            **utterance.provenance,
            "dispatcher_mode": "auto",
            "dispatcher_selected": utterance.frontend_name,
        },
    )


def _analyze_japanese_uncached(text: str, mode: str) -> JapaneseUtterance:
    """Run one frontend analysis without consulting the process cache."""
    normalized = mode
    if normalized == "kana":
        return KanaJapaneseFrontend().analyze(text)
    if normalized == "openjtalk":
        return resolve_japanese_frontend("openjtalk").analyze(text)

    if not is_pyopenjtalk_available():
        fallback = KanaJapaneseFrontend().analyze(text)
        return _prepend_diagnostic(fallback, JapaneseFrontendDiagnostic(
            code="openjtalk_unavailable_auto_fallback",
            message=(
                "Auto mode selected the dependency-free kana frontend because "
                "pyopenjtalk is unavailable."
            ),
            severity="info",
            action="Install pyopenjtalk locally to enable kanji and accent analysis.",
            frontend="auto",
            confidence=1.0,
        ))

    try:
        utterance = OpenJTalkJapaneseFrontend().analyze(text)
        return replace(
            utterance,
            provenance={
                **utterance.provenance,
                "dispatcher_mode": "auto",
                "dispatcher_selected": "openjtalk",
            },
        )
    except OpenJTalkFrontendError as error:
        fallback = KanaJapaneseFrontend().analyze(text)
        return _prepend_diagnostic(fallback, JapaneseFrontendDiagnostic(
            code="openjtalk_failed_auto_fallback",
            message=(
                f"Auto mode fell back to kana after Open JTalk failed: "
                f"{error.diagnostic.message}"
            ),
            severity="warning",
            action=error.diagnostic.action,
            frontend="auto",
            confidence=0.8,
            raw_data={"openjtalk_diagnostic": error.diagnostic.to_dict()},
        ))


def analyze_japanese(text: str, mode: str = "auto") -> JapaneseUtterance:
    """Analyze text through a bounded cache of private utterance snapshots."""
    global _FRONTEND_CACHE_BYTES, _FRONTEND_CACHE_HITS
    global _FRONTEND_CACHE_MISSES
    normalized = _validate_mode(mode)
    available = (False if normalized == "kana" else
                 bool(is_pyopenjtalk_available()))
    key = (str(text), normalized, available)
    with _FRONTEND_CACHE_LOCK:
        cached = _FRONTEND_CACHE.pop(key, None)
        if cached is not None:
            _FRONTEND_CACHE[key] = cached
            _FRONTEND_CACHE_HITS += 1
            return copy.deepcopy(cached[0])
        _FRONTEND_CACHE_MISSES += 1
        utterance = _analyze_japanese_uncached(str(text), normalized)
        cached_utterance = copy.deepcopy(utterance)
        byte_count = estimate_size_bytes(cached_utterance)
        if byte_count <= _FRONTEND_CACHE_MAX_BYTES:
            _FRONTEND_CACHE[key] = (cached_utterance, byte_count)
            _FRONTEND_CACHE_BYTES += byte_count
            while (len(_FRONTEND_CACHE) > _FRONTEND_CACHE_MAX_ENTRIES or
                   _FRONTEND_CACHE_BYTES > _FRONTEND_CACHE_MAX_BYTES):
                _old_key, (_old_value, old_bytes) = \
                    _FRONTEND_CACHE.popitem(last=False)
                _FRONTEND_CACHE_BYTES -= old_bytes
        return utterance


def japanese_frontend_cache_info() -> dict[str, int | str]:
    with _FRONTEND_CACHE_LOCK:
        return {
            "owner": "japanese-frontend-utterances",
            "entries": len(_FRONTEND_CACHE),
            "bytes": _FRONTEND_CACHE_BYTES,
            "max_entries": _FRONTEND_CACHE_MAX_ENTRIES,
            "max_bytes": _FRONTEND_CACHE_MAX_BYTES,
            "hits": _FRONTEND_CACHE_HITS,
            "misses": _FRONTEND_CACHE_MISSES,
        }


def clear_japanese_frontend_cache() -> dict[str, int | str]:
    global _FRONTEND_CACHE_BYTES
    with _FRONTEND_CACHE_LOCK:
        removed = {
            "owner": "japanese-frontend-utterances",
            "entries": len(_FRONTEND_CACHE),
            "bytes": _FRONTEND_CACHE_BYTES,
        }
        _FRONTEND_CACHE.clear()
        _FRONTEND_CACHE_BYTES = 0
        return removed
