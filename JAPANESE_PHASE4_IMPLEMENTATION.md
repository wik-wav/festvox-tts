# Japanese UTAU Integration: Phase 4

> **Corrective workflow note:** the later remediation keeps this project-state
> compatibility layer but replaces the equal-width scroller with a waveform-
> time mora strip. Nuclei and accent-phrase boundaries are directly draggable;
> split/merge and unaccented state persist in `accent_phrase_boundaries` and
> `accent_overrides`. Question editing moved to Intonation blocks, semantic
> pause totals moved to Options, and source inspection moved to Recordings.
> See `JAPANESE_IMPLEMENTATION_REPORT.md` under Corrective Stage 5.

Phase 4 exposes the canonical Japanese model and Phase 3 rendering plan in the
Windows editor. It extends the existing Speech workflow; it does not add a
second waveform or pitch editor and does not change the English path.

## Files and APIs

- `japanese_editing.py`
  - `new_edit_state()` and `normalize_edit_state()` define and migrate the
    provisional project overlay;
  - `reconcile_analyzed_utterance()` keeps overlays for identical text and
    clears occurrence-indexed edits when the sentence itself changes;
  - `utterance_from_dict()` restores the immutable Phase 1 model;
  - `apply_linguistic_edits()` applies phrase/accent overrides structurally;
  - `create_edited_plan()` combines Phase 1 analysis, Phase 4 overlays, and the
    Phase 3 planner;
  - `apply_mora_pitch_offsets()` applies bounded cent offsets to the generated
    baseline;
  - `analyze_bank()` provides a read-only profile, coverage, and unresolved
    alias preview;
  - `invalidation_for_edit()` distinguishes voice rebuilds from re-renders.
- `festvox_gui/festvox_core.py`
  - `FestivalWSLBackend.voice_metadata()` reads a generated runtime index;
  - `japanese_runtime_metadata()` accepts only `language="ja"` and the
    isolated Japanese entry point;
  - recorded OTO context is read compatibly from both established English and
    Phase 3 Japanese metadata field names;
  - project rows preserve `japanese_state` without sharing mutable data.
- `festvox_gui/festvox_gui.py`
  - `JapaneseMoraGrid`, `JapaneseEditorPanel`, and
    `JapaneseBankAnalysisDialog` implement the editing workflow;
  - Japanese text generation uses the optional Open JTalk/kana dispatcher and
    the explicit Phase 3 plan;
  - Japanese re-rendering rebuilds the structural F0 baseline while retaining
    timing edits, continuous pitch points, and manual units.
- `test_japanese_editing.py` and the expanded core/GUI suites cover state,
  migration, safety, planning, persistence, invalidation, and controls.

## Project State

Each sentence may contain a provisional `japanese_state` object:

```text
schema_version: 1
schema_status: phase4-provisional
frontend_mode: auto | openjtalk | kana
utterance: canonical Phase 1 JapaneseUtterance
accent_overrides: global accent-phrase index -> structural edit
phrase_overrides: global phrase index -> question/boundary edit
mora_pitch_offsets_cents: global mora index -> bounded cent offset
manual_candidate_overrides: global mora index -> stable candidate ID
continuous_pitch_authority: pitch_override
last_plan: most recent explicit Phase 3 plan
bank_analysis: coverage/unresolved-alias preview
profile_path: optional external generated profile
needs_voice_rebuild: profile changes not yet rebuilt
```

Indexes are zero-based internally. Old project rows without Japanese data load
with an empty overlay. Experimental `accent_edits` and
`mora_pitch_offsets` names migrate to the current names. The surrounding
project remains version 4, so existing version-4 projects load unchanged.

## Speech Workflow

The Parameter menu enables **Pitch accent** only when the active sentence is
Japanese, Festival/WSL is active, and the selected voice metadata advertises
Japanese support. Shared multilingual ARPAsing voices meet the same explicit
compatibility check; non-Japanese sentences never expose the editor. The page
provides:

- a horizontally scrollable mora grid;
- accent-phrase brackets and visible nucleus markers;
- analyzed, unaccented, and per-mora accent choices;
- phrase question-rise and boundary-strength controls;
- a per-mora pitch offset in cents;
- source-recording inspection for the rendered mora;
- a visible voice-rebuild requirement after profile changes.

The existing Timing, Pitch curve, Intonation blocks, and Recordings pages stay
available. Undo/Redo stores complete Japanese overlay snapshots. Save/load
persists the canonical model, overlays, stable candidate IDs, and last plan.

## Pitch Precedence

The explicit order is:

1. Phase 3 structural baseline from the analyzed accent model;
2. manual accent, phrase, question, and per-mora cent overlays;
3. the existing continuous Pitch curve, when present;
4. the backend's bounded 50-500 Hz safety clamp.

Re-render recalculates steps 1 and 2. It never silently deletes or replaces
continuous pitch points. A manual recording/unit choice remains final over
automatic context selection and survives unrelated accent or pitch edits.

## Bank Analysis

**Voicebank > Analyze Japanese UTAU bank...** reads OTO and metadata files
without opening any source file for write access. It displays inferred bank
type, confidence, total entries, candidate count, traceability, role/family
coverage, and every unresolved alias. Source alias, OTO location, WAV identity,
encoding, and classification reasons are inspectable.

Exact alias overrides are stored in a Japanese bank profile. The profile API
refuses an output path inside the source UTAU bank. Changing bank type or alias
interpretation sets **Voice rebuild required**; it does not pretend that an
ordinary audio re-render can update compiled units.

## Invalidation

- Text/language changes require **Generate**.
- Accent, question, phrase boundary, mora pitch, and candidate changes require
  **Re-render** when audio exists; an unrendered sentence correctly requires
  **Generate**.
- Bank configuration and alias-profile changes require a generated-voice
  **rebuild**.
- Continuous pitch edits continue to use the existing **Re-render** path.

## Safety and Compatibility

- Source UTAU banks are never written, moved, staged, or deleted.
- Generated Japanese metadata is read through the registered generated voice.
- A non-Japanese runtime index is rejected for Japanese text.
- Japanese remains behind explicit `ja` routing and its own `*_ja` voice entry
  point. CMU, ARPAbet, ARPAsing, English phonesets, and English Festival text
  entry points are unchanged.
- Open JTalk remains optional application behavior. Local `pyopenjtalk 0.4.1`
  supplies kanji/accent analysis when available; kana mode remains dependency
  free and deterministic.

## Verification Boundary

Phase 4 verifies model/state consistency, GUI behavior, project round trips,
undo/redo, invalidation, runtime isolation, source safety, and pitch/override
precedence. It does not claim that Japanese acoustic naturalness has been
verified. Boundary metrics, optional acoustic providers, dynamic multipitch,
licensing inventory, performance/caching, and release checks remain Phase 5.
