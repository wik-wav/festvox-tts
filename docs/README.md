# FestVox documentation map

This index is the documentation entry point. Operating guides and historical
reports live together under `docs/`; new development reports belong under
`docs/development`.

## Current operating documentation

- `../README.md`: product scope, quick setup, and feature overview.
- `GUIDE.md`: complete UTAU-to-database and synthesis walkthrough.
- `UNIFIED_VOICE_BUILDER.md`: authoritative voice-build CLI and safety
  boundary.
- `../src/festvox_tts/festvox_gui/README.md`: desktop GUI operation and
  Festival/WSL setup.
- `ASAXI_PITCH_INTEGRATION.md`: Asaxi dictionary, lexical and phrase
  accent, morphological/utterance prosody, Vocab Forge review, and the
  deterministic schema-v2 corpus with separate prediction and reference
  evidence. It also documents `asaxi_reading_guide.py`, which generates
  reader-facing per-word pitch-accent Markdown from plain Asaxi or a clean
  Markdown reader.
- `ASAXI_DURATION_MODEL.md`: provisional mora/phone duration rules,
  research basis, live Festival/WSL handoff, diagnostics, tests, and the
  boundary for a future recording-fitted model.
- `ASAXI_PHONE_FALLBACKS.md`: selected-bank diphone inventory checks and
  the auditable compact-palatal to consonant/glide realization fallback.
- `GUI_STATE_AND_SYNTHESIS.txt`: dated implementation snapshot; verify
  behavior against code before treating it as current.
- `development/PROMPT0_LONG_TERM_PERFORMANCE.md`: resource lifecycle audit,
  fixes, soak procedure, evidence, and remaining risks.
- `development/PROMPT0A_SYNTHESIS_EFFICIENCY.md`: synthesis profiling,
  bounded cache ownership, GUI cache controls, matched benchmark, and safety
  tests.
- `development/SOURCE_WINDOWS_AND_VCV_ONSETS.md`: VCV phrase-start alignment,
  adaptive source-window policy, reversible builder modes, and validation.
- `development/SOURCE_WINDOW_LABEL_AUDIT_2026-07-24.md`: human-labelled Lem
  trajectory audit, rejected runtime bridge, and the validated 60 ms default.
- `development/STRICT_CVVC_RUNTIME_SELECTION.md`: explicit CVVC family gate,
  mixed-bank alias interpretation, metadata, and real-bank sentence audit.
- `development/NATIVE_JOIN_CROSSOVER_AND_E3_PITCH_2026-07-23.md`: native
  per-occurrence crossover architecture, UI controls, persistent-worker
  measurements, E3 pitch correction, tests, and source-bank safety evidence.
- `development/RUNTIME_CACHE_Q_AND_E3_2026-07-24.md`: generated-voice
  metadata-cache bottleneck, measured WSL latency fix, integrated `q` routing,
  lazy diagnostic plots, and E3 manifest verification.
- `development/JOIN_CONTEXT_AND_RERENDER_VIEW_2026-07-24.md`: explicit runtime
  phone classes, voiced-continuant crossover policy, join defocus behavior,
  re-render viewport preservation, and real Lem/Kal verification.

## Architecture and synthesis notes

- `MULTISYN.md`: Multisyn recording and label workflow.
- `JAPANESE_UTAU_INTEGRATION_DESIGN.md`: Japanese architecture and phase
  boundaries.
- `JAPANESE_ASSEMBLY_REMEDIATION.md`: CV/VCV/CVVC source assembly.
- `SPECIAL_PHONE_REALIZATION.md`: language-neutral canonical/source phone
  mapping, structural `cl`, and explicit coexisting literal-phone mappings.
- `JOIN_DISCONTINUITY_DIAGNOSTICS.md`: acoustic join validation.
- `japanese_duration_model.md`: Japanese duration model.
- `SONG_MODE_FUTURE.md`: deferred song-mode design, not current runtime.

## Historical implementation reports

- `JAPANESE_PHASE1_IMPLEMENTATION.md` through
  `JAPANESE_PHASE5_IMPLEMENTATION.md`.
- `JAPANESE_IMPLEMENTATION_REPORT.md` and
  `JAPANESE_DEPENDENCIES_AND_LICENSES.md`.
- `PROMPT19_ARCHITECTURE.md`, `PROMPT19_IMPLEMENTATION.md`, and
  `PROMPT19_BENCHMARK_REPORT.md`.
- `PROMPT20_STAGE_A_FORMANT_ANALYSIS.md` and
  `PROMPT20_IMPLEMENTATION_REPORT.md`.

Generated listening sets, soak JSON, plots, and exported WAV files belong in
ignored `rendered_audio` or other generated-output folders, not in this
documentation tree.
