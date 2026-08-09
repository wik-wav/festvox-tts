# Prompt 19 Architecture Baseline

This note records the production path before Prompt 19 synthesis changes. It
is intentionally descriptive: later implementation notes must state where the
path changed instead of silently creating a parallel synthesizer.

## Runtime path

1. `festvox_gui/festvox_gui.py` selects the explicit Japanese route only when
   the active language is Japanese and the selected voice exposes Japanese
   runtime metadata.
2. `japanese_frontend.py` selects `openjtalk`, `kana`, or deterministic `auto`.
   `japanese_openjtalk.py` calls only `g2p(..., kana=True)` and
   `extract_fullcontext()`. It preserves every raw HTS label and parses named
   A, F, I, and K context groups. Open JTalk audio, durations, and HTS F0 are
   not used.
3. `japanese_editing.py` applies saved accent, phrase, mora-pitch, and manual
   candidate edits, then calls `japanese_synthesis.create_synthesis_plan()`.
4. The planner creates explicit `(phone, duration)` rows, structural F0
   targets, and per-occurrence unit overrides. Before Prompt 19, phone timing
   comes from hard-coded mora totals and consonant budgets in
   `_mora_phone_durations()`, then `_apply_source_timing_constraints()` clamps
   them against source-unit geometry.
5. `MainWindow._render_japanese_plan()` overlays punctuation intonation and
   calls `FestivalWSLBackend.synth_phones()`.
6. `festvox_core.py` constructs a Festival `Utterance Segments` relation. Each
   row carries its exact duration and any F0 daughters. The generated voice's
   UniSyn hook selects automatic or manual units before `(utt.synth u)`.
7. The generated Japanese Scheme selects `Synth_Method 'UniSyn` and
   `us_sigpr 'psola`. It uses waveform and EST pitchmark files directly. The
   generated voice has no LPC coefficient or residual database, so this is
   direct waveform TD-PSOLA, not LPC-residual synthesis.
8. Festival returns PCM WAV plus Segment, TargetCoef, SourceCoef, US_map, Unit,
   and Target diagnostics. Python stores mono `float32` samples nominally in
   `[-1, 1]`, explicit segment boundaries, rendered target pitchmarks, selected
   unit names, and exact/estimated unit-handoff collars in `Synthesis`.

## Source timing and ownership

- Source OTO rows and WAVs remain read-only. Builders copy selected WAVs into
  generated output and write all metadata outside the source bank.
- Japanese unit metadata retains OTO offset, overlap, preutterance, consonant,
  cutoff, source slice start/mid/end, alias, source row, candidate ID, role,
  context, and source pitch/subbank provenance.
- A target phone receives contributions from the right half of its incoming
  diphone and the left half of its outgoing diphone. Existing timing safety
  uses those source half lengths, but no fitted contextual residual model is
  present yet.
- Manual candidate overrides are keyed to one mora occurrence and become
  indexed `us_diphone_left` choices. They are final and must survive duration,
  pitch, and vowel-realization changes.

## Mora and context mapping

- `JapanesePhone`, `JapaneseMora`, `JapaneseAccentPhrase`, `JapanesePhrase`,
  and `JapaneseUtterance` are the canonical linguistic model.
- A phone retains its raw Open JTalk symbol and full-context label. A mora
  retains canonical phones, special-mora class, devoicing flag, phrase and
  accent-phrase membership, and one-based Open JTalk mora-position provenance.
- Accent nuclei are zero-based internally. Open JTalk F-context accent types
  are one-based, with zero representing unaccented.
- `cl`, `N`, long-vowel continuations, pauses, and devoiced high vowels are
  structurally distinguishable, but their pre-Prompt-19 timings are heuristic.

## Duration and F0 application

- Exact target durations are passed through Festival's Segments utterance;
  UniSyn maps analysis frames onto target pitchmarks and overlap-adds them.
- Vowels and sonorants are generally safer to stretch than closures, bursts,
  taps, or short transitions. The current planner has broad class bounds but
  no local duration-error redistribution.
- Structural Japanese F0 is speaker-relative and later overlaid by punctuation
  blocks or continuous manual points. Removing target F0 does not remove
  periodic excitation from a recorded voiced vowel.
- Because this voice uses direct waveform PSOLA, genuine voiced-to-unvoiced
  conversion requires a naturally aperiodic source or a separate excitation
  transformation. TD-PSOLA remains useful for duration mapping of material
  that is already aperiodic.

## Join diagnostics

- `festvox_core.parse_unisyn_render_diagnostics()` derives target epochs and
  unit handoff collars from TargetCoef/US_map instead of guessing joins from
  phone centers when Festival exposes them.
- `join_discontinuity.py` already reports independent level, sample/slope,
  broadband impulse, period/F0, phase, period shape, cepstral envelope, and
  local-novelty metrics. It is read-only.
- `join_spectrogram.py` renders waveform, STFT, issue triangles, handoff bars,
  a time-aligned phone strip, and the complete phone sequence under the plot.
  Stop/closure contexts are visually distinct because their broadband bursts
  can be legitimate.
- F1-F4 trajectories, formant prominence/bandwidth, and normalized formant
  balance are not yet measured.

## Test baseline

- GUI/core suite: 153 tests pass.
- Main FestVox suite: 216 executable tests pass, two optional tests skip, and
  one optional local `pyopenjtalk` integration test errors because the package
  exists while its MeCab dictionary directory does not. Operational dependency
  detection is the first narrow fix.
- The source-bank tree digests were recorded outside committed reports before
  edits and will be compared after real-bank validation.

## Prompt 19 extension points

- Add contextual feature and duration modules beside `japanese_synthesis.py`;
  preserve `create_synthesis_plan()` as the normal entry point.
- Serialize build-derived speaker baselines and versioned corpus residuals in
  generated runtime metadata. Runtime must not need a corpus or ML package.
- Keep duration prediction and vowel realization as separate typed decisions.
- Apply any aperiodic rendering after UniSyn has produced explicit segment
  boundaries, unless a naturally devoiced source candidate can be selected.
- Extend the existing join analyzer and plotting schema; do not replace it.
- Put corpus fitting, evaluation, listening WAVs, and diagnostic images in
  ignored output directories. Commit only code, fixed configuration, model
  parameters, reports without private absolute paths, and tests.

## Implemented extension

The extension points above are now active in the same production route:

- `japanese_duration.py` predicts source-relative contextual phone durations
  from typed Open JTalk/canonical context and versioned schema-1 priors.
- `japanese_synthesis.create_synthesis_plan()` defaults to contextual timing,
  retains `legacy`, and emits explicit `timing_role` values. Japanese `/N/`
  aliases are bounded moraic nuclei rather than generic consonants.
- `japanese_devoicing.py` keeps duration and realization separate. It prefers
  already-aperiodic source material, then the deterministic source/filter
  implementation in `source_filter_voicing.py`, then a clearly reported
  shortened-voiced fallback.
- The Speech editor exposes continuous Voicing targets for every language;
  automatic analysis remains a stable dashed reference and manual points are
  final.
- `japanese_duration_corpus.py`, `japanese_duration_ab.py`, and
  `voicing_validation.py` provide fitting, held-out evaluation, A/B rendering,
  and spectrogram validation outside source banks and generated voices.
- `join_discontinuity.py` now adds formant frequency, bandwidth, prominence,
  balance, spectral-envelope, and trajectory evidence without modifying audio
  or unit selection.

See `PROMPT19_IMPLEMENTATION.md` and `PROMPT19_BENCHMARK_REPORT.md` for the
implemented decision path and measured results.
