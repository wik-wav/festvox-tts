"""Inventory-aware realization fallbacks for Asaxi phone plans.

The canonical Asaxi frontend uses compact palatalized phones such as ``hy``.
Integrated ARPAsing banks commonly provide those phones before the five
Japanese vowels, but not before ARPAsing-specific vowels such as ``ao``.
Changing the canonical G2P would make good native transitions worse, so this
module adapts only transitions that the selected generated voice cannot
render.

For a missing transition involving ``Cy``, the conservative repair is::

    Cy V  ->  C y y V

The repeated glide gives the concatenative renderer the three explicit
transitions ``C-y``, ``y-y``, and ``y-V``.  A repair is accepted only when
every affected diphone, including the preceding context, exists in the
selected bank.  Otherwise the canonical phones are retained and an
actionable diagnostic is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import asaxi_frontend
import asaxi_prosody


ASAXI_PHONE_FALLBACK_MODEL_ID = "asaxi-inventory-fallback-v1"


@dataclass(frozen=True)
class AsaxiPhoneFallbackRecord:
    """One canonical-to-rendered phone expansion."""

    word_index: int
    mora_index: int
    surface: str
    canonical_phone: str
    rendered_phones: tuple[str, ...]
    missing_canonical_diphones: tuple[str, ...]
    validated_diphones: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "word_index": self.word_index,
            "mora_index": self.mora_index,
            "surface": self.surface,
            "canonical_phone": self.canonical_phone,
            "rendered_phones": list(self.rendered_phones),
            "missing_canonical_diphones": list(
                self.missing_canonical_diphones
            ),
            "validated_diphones": list(self.validated_diphones),
        }


@dataclass(frozen=True)
class AsaxiPhoneFallbackResult:
    """Voice-adapted plan plus audit records and new diagnostics."""

    plan: asaxi_prosody.AsaxiProsodyPlan
    records: tuple[AsaxiPhoneFallbackRecord, ...] = ()
    diagnostics: tuple[asaxi_prosody.AsaxiProsodyDiagnostic, ...] = ()
    available_diphone_count: int = 0


@dataclass(frozen=True)
class _PhoneSlot:
    phone: str
    mora_index: int
    word_index: int


def available_diphones(
    metadata_or_index: Mapping[str, object] | Iterable[str] | None,
) -> frozenset[tuple[str, str]]:
    """Return exact canonical diphone pairs from runtime voice metadata."""

    source: object = metadata_or_index
    if isinstance(source, Mapping) and "index" in source:
        source = source.get("index")
    if isinstance(source, Mapping):
        keys: Iterable[object] = source.keys()
    elif isinstance(source, (str, bytes)) or source is None:
        keys = ()
    else:
        try:
            keys = iter(source)
        except TypeError:
            keys = ()

    pairs: set[tuple[str, str]] = set()
    for raw_key in keys:
        key = str(raw_key)
        if "-" not in key:
            continue
        left, right = key.split("-", 1)
        if left and right:
            pairs.add((left, right))
    return frozenset(pairs)


def _pair_name(pair: tuple[str, str]) -> str:
    return f"{pair[0]}-{pair[1]}"


def _local_pairs(
    slots: Sequence[_PhoneSlot],
    start: int,
    end: int,
) -> tuple[tuple[str, str], ...]:
    left = max(0, start - 1)
    right = min(len(slots), end + 1)
    return tuple(
        (slots[index].phone, slots[index + 1].phone)
        for index in range(left, right - 1)
    )


def _expanded_plan(
    plan: asaxi_prosody.AsaxiProsodyPlan,
    slots: Sequence[_PhoneSlot],
    diagnostics: Sequence[asaxi_prosody.AsaxiProsodyDiagnostic],
) -> asaxi_prosody.AsaxiProsodyPlan:
    """Rebuild mora and word spans after phone expansion."""

    phones_by_mora: dict[int, list[str]] = {
        mora.index: [] for mora in plan.moras
    }
    for slot in slots:
        phones_by_mora.setdefault(slot.mora_index, []).append(slot.phone)

    moras = []
    cursor = 0
    for mora in plan.moras:
        phones = tuple(phones_by_mora.get(mora.index, ()))
        moras.append(replace(
            mora,
            phones=phones,
            phone_start=cursor,
            phone_end=cursor + len(phones),
        ))
        cursor += len(phones)

    words = []
    for word in plan.words:
        word_phones = tuple(
            phone
            for mora in moras[word.mora_start:word.mora_end]
            for phone in mora.phones
        )
        words.append(replace(word, phones=word_phones))

    return replace(
        plan,
        words=tuple(words),
        moras=tuple(moras),
        phones=tuple(slot.phone for slot in slots),
        diagnostics=tuple(plan.diagnostics) + tuple(diagnostics),
    )


def adapt_plan_for_inventory(
    plan: asaxi_prosody.AsaxiProsodyPlan,
    metadata_or_index: Mapping[str, object] | Iterable[str] | None,
    *,
    protected_word_indices: Iterable[int] = (),
) -> AsaxiPhoneFallbackResult:
    """Adapt unsupported compound phones without changing canonical G2P.

    ``protected_word_indices`` identifies explicit project/user
    pronunciations.  Those words remain byte-for-byte authoritative even when
    their transitions are absent from the selected voice.
    """

    inventory = available_diphones(metadata_or_index)
    if not inventory:
        return AsaxiPhoneFallbackResult(plan=plan)

    protected = {int(index) for index in protected_word_indices}
    slots = [
        _PhoneSlot(phone, mora.index, mora.word_index)
        for mora in plan.moras
        for phone in mora.phones
    ]
    if tuple(slot.phone for slot in slots) != tuple(plan.phones):
        diagnostic = asaxi_prosody.AsaxiProsodyDiagnostic(
            code="asaxi_phone_fallback_alignment",
            message=(
                "The canonical phone sequence does not match its mora spans; "
                "bank-specific phone fallbacks were not applied."
            ),
            severity="warning",
        )
        return AsaxiPhoneFallbackResult(
            plan=replace(
                plan,
                diagnostics=tuple(plan.diagnostics) + (diagnostic,),
            ),
            diagnostics=(diagnostic,),
            available_diphone_count=len(inventory),
        )

    records: list[AsaxiPhoneFallbackRecord] = []
    diagnostics: list[asaxi_prosody.AsaxiProsodyDiagnostic] = []
    index = 0
    while index < len(slots):
        slot = slots[index]
        phone = slot.phone
        if (
            slot.word_index in protected
            or phone not in asaxi_frontend.PALATAL_PHONES
            or not phone.endswith("y")
        ):
            index += 1
            continue

        canonical_pairs = _local_pairs(slots, index, index + 1)
        missing_canonical = tuple(
            pair for pair in canonical_pairs if pair not in inventory
        )
        if not missing_canonical:
            index += 1
            continue

        base = phone[:-1]
        replacement = (
            _PhoneSlot(base, slot.mora_index, slot.word_index),
            _PhoneSlot("y", slot.mora_index, slot.word_index),
            _PhoneSlot("y", slot.mora_index, slot.word_index),
        )
        candidate = slots[:index] + list(replacement) + slots[index + 1:]
        replacement_pairs = _local_pairs(
            candidate, index, index + len(replacement)
        )
        missing_replacement = tuple(
            pair for pair in replacement_pairs if pair not in inventory
        )
        word = plan.words[slot.word_index]
        if missing_replacement:
            diagnostics.append(asaxi_prosody.AsaxiProsodyDiagnostic(
                code="asaxi_phone_fallback_unavailable",
                message=(
                    f"{word.surface}: selected voice lacks "
                    f"{', '.join(_pair_name(pair) for pair in missing_canonical)} "
                    f"and the complete {base} y y fallback "
                    f"({', '.join(_pair_name(pair) for pair in missing_replacement)} "
                    "also missing). Canonical phones were retained."
                ),
                severity="warning",
                word_index=slot.word_index,
            ))
            index += 1
            continue

        slots = candidate
        record = AsaxiPhoneFallbackRecord(
            word_index=slot.word_index,
            mora_index=slot.mora_index,
            surface=word.surface,
            canonical_phone=phone,
            rendered_phones=(base, "y", "y"),
            missing_canonical_diphones=tuple(
                _pair_name(pair) for pair in missing_canonical
            ),
            validated_diphones=tuple(
                _pair_name(pair) for pair in replacement_pairs
            ),
        )
        records.append(record)
        diagnostics.append(asaxi_prosody.AsaxiProsodyDiagnostic(
            code="asaxi_phone_fallback_applied",
            message=(
                f"{word.surface}: expanded {phone} to {base} y y because "
                f"the selected voice lacks "
                f"{', '.join(record.missing_canonical_diphones)}."
            ),
            severity="info",
            word_index=slot.word_index,
        ))
        index += len(replacement)

    adapted = _expanded_plan(plan, slots, diagnostics)
    return AsaxiPhoneFallbackResult(
        plan=adapted,
        records=tuple(records),
        diagnostics=tuple(diagnostics),
        available_diphone_count=len(inventory),
    )
