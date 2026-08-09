# Native join crossover and E3 pitch checkpoint

Date: 2026-07-23

This checkpoint replaces the experimental sentence-wide join window with an
inspectable per-occurrence crossover while preserving Festival's phone plan,
F0 targets, contextual choices, and manual recording overrides. It also removes
the former automatic pitch headroom that made an E3-only voice default to about
202 Hz.

## Runtime architecture

- Normal Festival/WSL synthesis uses the project-local
  `native_unisyn/festvox-festival` helper.
- Unit selection and target Segment/F0 relations are completed before
  concatenation. The crossover renderer does not choose a different recording.
- The crossover request is expressed in milliseconds, not a fixed period count.
  The default is 40 ms and the supported range is 0-100 ms.
- Each occurrence can override its left and right crossover extent.
- A phone-class and available-context cap prevents a request from consuming
  unsafe consonant or edge material. Source and target ends then snap inward to
  usable pitchmarks.
- Local gain matching is gradual and bounded. Complementary raised-cosine
  windows perform the overlap; no post-render PCM repair is applied.
- **Legacy joins** bypasses the helper and uses a fresh stock-Festival process.
  It is the exact old waveform path, not a visual approximation.

Normal native renders reuse a persistent Festival process. The worker uses a
line-oriented request protocol, is invalidated when voice metadata or the
runtime binary changes, and is recycled after 32 jobs by default. Legacy and
non-native Scheme requests remain one-shot.

Windows-hosted metadata uses file-change tokens. WSL-only registrations use
cheap stats through `\\wsl.localhost\<distro>` so an in-place rebuild
invalidates Python caches and the warm interpreter before the next render
without adding a WSL-process launch to the hot path. A custom native executable
is fingerprinted at its configured path.

A `0 ms` sentence default with no positive occurrence override deliberately
uses stock one-shot Festival as a no-crossover control. A positive occurrence
override still selects the native helper. This is not the same as **Legacy
joins**, which also restores historical source windows and discards overrides.

## GUI authority

**View > Rendered joins in waveform** shows rendered join spans and a selected
join's left/right handles in the main Speech waveform. The inspector exposes
the same values and lists joins in phone render order by default. Selecting a
Recordings block focuses the handoff in the middle of that phone.

The selected-join label reports:

- requested crossover milliseconds;
- actual rendered milliseconds after pitchmark snapping;
- the current context cap.

Dragging either handle changes only that occurrence, creates undo state, and
marks the waveform for Re-render. Phone timing edits remap the displayed span
by relative phone position. Join edits do not regenerate duration, F0, or
recording selection.

## Voice pitch rule

Without `--f0`, the builder now writes the measured source median for the
selected OTO scope and records zero automatic headroom. A fresh E3-only Lem
validation build produced:

```text
average_pitch_hz: 164.81
default_pitch_source: speaker_median
automatic_pitch_floor_hz: 164.81
automatic_pitch_headroom_semitones: 0.0
```

An explicit `--f0` remains final. For backward compatibility, a manifest tagged
`speaker_median_plus_headroom` resolves through its stored source median; an
explicit builder override is not reinterpreted.

The previously generated `lem_v4bi_integrated_TEST` artifact was also migrated
in place: its active manifest, diphone index, and alternatives metadata now all
record `164.81 Hz`, `speaker_median`, and zero automatic headroom. No generated
audio, unit-selection data, pitchmarks, or source-bank files were changed.

## Validation

### Structural and acoustic checks

- Built-in Kal rendered through both normal and Legacy routes.
- A generated integrated Lem voice rendered at a corrected 164.81 Hz default.
- Normal and Legacy Lem renders retained identical segment boundaries, F0
  targets, selected unit identities, sample count, and 1.1461 s duration.
- Normal Lem rendering used eight active crossovers. The discontinuity analyzer
  ranked four joins for inspection, versus six in the matched Legacy render.
- Repeated Legacy renders were byte-identical.
- Acoustic naturalness remains unverified by human listening.

### Performance

Measured on the same local WSL installation:

```text
cold native render:             about 2.486 s
warm native median:             about 0.0295 s
stock one-shot Legacy median:   about 0.2024 s
warm native startup advantage:  about 6.9x
```

A 70-render soak recycled the worker at jobs 32 and 64, used exactly three
process IDs, and produced byte-identical output for every request. Its warm
median was about 0.0311 s. A 657-phone, 56.02 s utterance measured 2.5948 s
native versus 2.6034 s Legacy, showing no long-render regression.

### Automated tests

Run from `99_Tools/festvox`:

```powershell
$root = (Get-Location).Path
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
$env:PYTHONPATH = $root
& $python -m unittest discover -s $root -p "test_*.py"

Push-Location .\festvox_gui
$env:PYTHONPATH = (Get-Location).Path
& $python -m unittest test_festvox_core
$env:QT_QPA_PLATFORM = "offscreen"
& $python -m unittest test_festvox_gui
Pop-Location
```

Results:

- root suite: 442 passed, 5 optional integrations skipped;
- Festival core: 108 passed;
- GUI: 140 passed;
- total: 690 passed, 5 optional integrations skipped.

The skipped integrations require local WSL Festival tools (two tests),
`pyworld`, `pyopenjtalk`, or a `KOKORO_ALIGN_CHECKPOINT`. Their deterministic
fallback and fixture tests still ran.

The native C++ helper was rebuilt in Ubuntu before the real-render checks.

## Source-bank safety

The validation builds and renders read source UTAU banks only. No source-bank
file was staged or modified. Post-validation hashes were:

```text
Lem_V4Bi_Civet/3_E3/oto.ini
4B1848F2E4CF5BAA3329B81C5B4467348BEB6212F5DF868693B347C73B95DB70

uta/oto.ini
9836404BFDCAC094ACEFE7C87A7AE8923B5DEF8607A87B38AB0572549246D0E1
```

Generated WAVs, manifests, plots, benchmark scripts, and soak output remain in
ignored temporary or generated-output directories.

## Relevant implementation

- `native_unisyn/festvox_festival.cc`
- `native_unisyn/build_wsl_runtime.py`
- `join_synthesis.py`
- `unisyn_runtime.py`
- `festvox_gui/festvox_core.py`
- `festvox_gui/festvox_gui.py`
- `speaker_pitch.py`
- `build_festival_voice.py`
- `test_join_synthesis.py`
- `test_unisyn_runtime.py`
- `test_speaker_pitch.py`
- `test_unified_voice_builder.py`
- `festvox_gui/test_festvox_core.py`
- `festvox_gui/test_festvox_gui.py`

## Remaining limits

- A long requested crossover may render shorter when source or phone context is
  unsafe; the UI exposes this rather than silently pretending the request fit.
- The first normal render still pays Festival startup cost.
- Automated discontinuity scores rank suspicious joins but do not establish
  perceptual naturalness.
- The native helper currently targets the supported Ubuntu Festival/EST
  environment; direct Festival and Legacy remain available for comparison.
