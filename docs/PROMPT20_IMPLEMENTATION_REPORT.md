# Prompt 20 Implementation Report

## Completion status

Prompt 20 is implemented in the normal Festival/UniSyn and GUI paths. The
work is deliberately split into two stages:

1. Stage A reference analysis and validation infrastructure was completed and
   committed first as `135fb16 feat: add Prompt 20 Stage A reference analysis`.
2. Stage B consumes the versioned Stage A profiles and adds production
   multilingual pitch-domain helpers, Japanese pitch and duration models,
   automatic mora voicing, source/filter vocal-tract transformation, GUI
   editing, final-waveform diagnostics, benchmarks, and listening fixtures.

Stage B was not enabled before the Stage A gate passed. Source UTAU files are
read-only inputs. Generated reports and listening WAVs live under ignored
`rendered_audio/` directories and are not release assets.

Acoustic naturalness, speaker-identity preservation, and the perceptual quality
of the deliberately expanded vocal-tract range still require human listening.
The automated results below establish structural and signal-processing
invariants; they do not make a naturalness claim.

## Production architecture

### Shared pitch domain

`pitch_domain.py` is the language-neutral boundary between model-space pitch
and Festival/PSOLA Hz. Pitch models retain language-specific behavior, but use
log F0 or semitone deltas internally. Conversion to Hz is delayed until the
Festival target relation or waveform-processing boundary. The helpers reject
non-positive Hz values and make clamping explicit.

English and Asaxi retain their existing language models and generated-voice
entry points. Japanese remains behind the canonical Japanese frontend and
Japanese synthesis planner. Nothing maps Japanese phones through ARPAbet.

### Japanese pitch model

`japanese_pitch.py` loads
`profiles/japanese_pitch_model_v1.json`. The active model ID is:

```text
japanese_speaker_relative_log_f0_kokoro_b453f6caf042_v5
```

The requested Festival pitch remains the source-speaker register anchor. The
model then adds bounded semitone components for initial low tone, pre-accent
high, accent nucleus, post-accent drop, unaccented phrases, downstep, phrase
reset, local declination, final boundary strength, and interrogation.

Kokoro supplies normalized contour-shape evidence only. Its absolute speaker
register is never copied. The runtime PSOLA envelope is limited to five
semitones above and below the requested baseline.

Repeated phrases do not use elapsed-time drift or cumulative phrase-index
register drift. Later phrases instead receive a mean-centered shape change:

```text
later phrase declination delta:       +0.13 semitones
later phrase accent contrast delta:   -0.13 scale
later phrase boundary delta:          -1.25 semitones
maximum contextual phrase index:       3
```

Each phrase-position contribution has zero mean over that phrase. Repetition
therefore changes contour shape without forcing a steadily falling register.
Render Details exposes `phrase_position_model: mean_centered_shape` and
`cumulative_register_drift_enabled: false`.

### Japanese duration model

`japanese_duration.py` and `japanese_duration_corpus.py` load
`profiles/japanese_duration_priors_v1.json`. The active model ID is:

```text
japanese_contextual_source_anchor_kokoro_b453f6caf042_v7
```

Project-speaker recordings provide absolute mora anchors. Kokoro contributes
only normalized phone-class allocation and bounded contextual residuals. The
model has explicit treatment for ordinary CV morae, obstruent CV morae,
vowel-only morae, long vowels, geminate closure, moraic nasal, devoiced high
vowels, phrase-final rhyme, interrogation, grammatical auxiliaries, acoustic
phrase edges, and speaking-rate elasticity.

Current project-speaker anchors are:

| Class | Seconds |
|---|---:|
| vowel-only mora | 0.110 |
| ordinary CV mora | 0.118 |
| obstruent CV mora | 0.122 |
| geminate closure | 0.082 |
| moraic nasal | 0.088 |
| long vowel | 0.108 |
| devoiced high vowel | 0.092 |
| other | 0.095 |

Kokoro-Align does not emit punctuation as a pause class. A read-only energy
refinement therefore expands each interpolated punctuation boundary to the
local low-energy run without moving any source file. On the fixed 36/12/12
sample, 33 minor breaks have a `419.0 ms` observed median and 14 sentence
breaks have an `880.5 ms` median. The production pause settings are `340`,
`530`, and `800 ms` for minor, major, and sentence breaks. Each is rendered as
two protected 80 ms guards plus an editable middle interval 80 ms shorter, for
total spans of `420`, `610`, and `880 ms`; resizing cannot stretch a neighboring
phone. Explicit punctuation is authoritative over a generic Open
JTalk `pau`, so `、` is no longer promoted to a major break.

Open JTalk `run_frontend` nodes are preserved on canonical morae with surface,
reading, pronunciation, part of speech, conjugation, chain, accent, node
position, and coarse grammatical role. The train-only residual fit retained
ordinary auxiliary `-0.045` and negative auxiliary `-0.075` log-duration
effects because both improved held-out error. Particle, polite-copula, and
polite-auxiliary categories remain inspectable but have no fitted correction:
their estimates were unstable or worsened absolute held-out timing. No grammar
effect is inferred when the optional Open JTalk morphology is unavailable.

Logical phone labels did not explain the reported phrase-initial vowel length.
`japanese_phrase_edges.py` measures sustained activity around rendered phrase
boundaries and compares it with the same edge in the Kokoro source. On exact
training renders, vowel-initial phrases had `51.791 ms` median extra lead-in
(`n=8`) versus `7.921 ms` for consonant starts (`n=40`). A versioned, bounded
`50 ms` vowel-initial compensation reduces held-out effective first-mora
excess from `65.404` to `15.404 ms`; it never changes contextual unit choice.
A proposed final-vowel compensation was rejected: the unmodified effective
final-mora median was already within `2.684 ms` of source and the correction
made it too short. Polite-copula endings remain a named audit category rather
than receiving an unsupported blanket rule.

Moraic `/N/` is a rhyme/timing nucleus, not an ordinary consonant. Integrated
aliases such as `nn`, `nng`, `mm`, and `xn` therefore follow Japanese
moraic-nasal timing when and only when the canonical Japanese plan identifies
that role.

Generate Audio may construct a fresh modeled timeline. Re-render Phonemes does
not: it takes the exact current editor segment durations, including every
manual boundary edit, and only regenerates F0, voicing, unit-selection, and
other synthesis metadata. A dedicated GUI regression compares each duration
exactly rather than with a tolerance.

### Automatic mora voicing

`japanese_devoicing.py` predicts high-vowel devoicing from canonical mora,
consonant manner and voicing, neighboring morae, accent, phrase position,
boundaries, geminates, long vowels, consecutive-devoicing avoidance, and
speaking rate. Open JTalk information participates when available.

The prediction remains separate from duration and underlying F0. The normal
render path converts predictions into a continuous source/filter voicing
trajectory. Per-mora overrides are final and may select multiple morae in one
accent phrase. Diagnostics preserve the automatic decision, final value,
override status, and reasons.

The GUI offers both the phrase-level `Mora voicing` editor and the detailed
continuous `Voicing` parameter curve. The measured/generated dashed curve is
stable across regeneration; the detailed user curve remains final authority.

### Source/filter vocal-tract transform

`vocal_tract.py` models the canonical control as:

```text
vocal_tract_length_ratio = target apparent length / source apparent length
formant multiplier       = 1 / vocal_tract_length_ratio
formant shift semitones  = -12 * log2(vocal_tract_length_ratio)
```

It does not resample the waveform. The production transform uses one
overlap-add spectral pass with an F0-adaptive cepstral/true-envelope estimate.
The complex spectrum retains its source phase and excitation structure while
the magnitude receives the ratio between the warped and original envelopes.
Implementation safeguards include:

* exact identity bypass;
* 40 ms analysis windows and 10 ms hops;
* source-RMS preservation;
* phone-dependent transformation strength;
* conservative treatment of fricatives and transients;
* silence bypass;
* smooth invalid-band and Nyquist tapering;
* denominator floors and a 30 dB per-frame envelope-gain limit;
* deterministic overlap-add normalization.

The transform accepts a continuous time curve. GUI editing is intentionally
coarser than voicing, with at most two editable points per phone. Pitch,
duration, voicing, creak placement, and vocal-tract ratio are persisted as
independent controls.

The retained baseline is a uniform spectral-envelope warp. No arbitrary
formant-by-formant or vowel-conditioned correction was added because the
supplied data did not justify one robustly. The uniform model is an acoustic
approximation, not an anatomical larynx simulator or complete gender
conversion.

## Reference analysis

### Data and alignment

Stage A analyzed the project source voicebank, the supplied Prompt 20 formant
references, and a deterministic 60-utterance sample from
Kokoro-Speech-Dataset v1.3 xlarge. Kokoro was split 36/12/12 for
training/validation/test. Kokoro-Align's epoch-200 CTC checkpoint was actually
run; 60 alignments were accepted and none silently discarded. The median
alignment confidence was `0.776531`. These boundaries remain silver reference,
not hand-labelled ground truth.

The supplied formant-reference files actually analyzed were
`neutral.wav`, `static-neutral.wav`, `high.wav`, `low.wav`,
`change-neutral-to-high.wav`, `change-neutral-to-low.wav`, and `sweep.wav`;
`sweep.png` was retained as visual provenance. Their individual SHA-256 values,
along with the Kokoro archive hash, are stored in
`profiles/reference_voice_space_v1.json`. The source-bank analysis covered 120
stable vowel segments selected from the registered generated voice's diphone
index while keeping the source bank read-only.

The source-bank bundle hash recorded at Stage A was:

```text
9bf4cbcf9b2fbdbce060b4243a0965eabf49a9a2a6fc87cc6e40db465d6250ec
```

The Kokoro archive SHA-256 is:

```text
71b019f1bc9489e303fb70ea51000dfd64c8646a2ad6ce660f7b21c763c01f77
```

Supplied-reference hashes are stored in
`profiles/reference_voice_space_v1.json`. The supplied WAV evidence covers
`/e/`; it is not presented as five-vowel population evidence.

### Formant estimator

The primary estimator is an F0-adaptive iterative true envelope informed by
Roebel and Rodet. A dynamically ordered Burg LPC tracker is retained as an
independent cross-check. This avoids treating individual harmonics as formants,
particularly at high F0.

Of 5,294 retained frames, 4,562 passed reliability filters. Rejected frames
remain inspectable with reasons. Median true-envelope/Burg disagreement on
source material was `33.9`, `35.2`, `96.6`, and `52.2 Hz` for F1-F4; 90th
percentiles were `164.8`, `99.3`, `499.5`, and `134.5 Hz`.

Project-source medians were:

| Vowel | Accepted segments | F1 | F2 | F3 | F4 |
|---|---:|---:|---:|---:|---:|
| `/a/` | 24/24 | 775.2 | 1615.0 | 3143.8 | 3572.5 |
| `/i/` | 23/24 | 335.3 | 2581.3 | 3100.3 | 3477.2 |
| `/u/` | 24/24 | 452.2 | 1501.9 | 2947.4 | 3507.2 |
| `/e/` | 23/24 | 581.4 | 1997.2 | 3065.8 | 3478.9 |
| `/o/` | 24/24 | 635.2 | 1270.5 | 3071.2 | 3645.6 |

These are source-speaker acoustic resonance baselines, not anatomical lengths.

### Control ranges

`profiles/reference_voice_space_v1.json` supplies every runtime range:

| Range | Minimum | Maximum | Meaning |
|---|---:|---:|---|
| ordinary | 0.93363995 | 1.05740237 | robust reference-derived default |
| observed supplied-reference span | 0.88083696 | 1.16684161 | evidence only, not a UI range |
| expanded engineering | 0.65 | 1.50 | post-waveform validated Chipmunk range |

The reference evidence is too narrow for a population or binary-gender claim.
The ordinary range is therefore a conservative reference-grounded engineering
range. The expanded endpoints were retained only after all five project-source
vowels passed final-waveform direction, identity, duration, F0, clipping, and
numerical-safety checks. `Chipmunk range` defaults off.

## Objective evaluation

### Duration and pitch against Kokoro

The benchmark aligns the first equal phone and compares final synthesis plans
against Kokoro-Align phone boundaries. It retains train and held-out results
separately.

| Duration metric | Train contextual | Train legacy | Held-out contextual | Held-out legacy |
|---|---:|---:|---:|---:|
| phone MAE (ms) | 31.126 | 33.271 | 32.209 | 34.149 |
| median absolute error (ms) | 19.263 | 21.089 | 19.460 | 21.332 |
| p90 absolute error (ms) | 61.322 | 66.177 | 58.491 | 67.396 |
| median boundary drift (ms) | 209.154 | 236.692 | 237.834 | 220.595 |
| rate-normalized log RMSE | 0.5844 | 0.6310 | 0.5926 | 0.6331 |

| Pitch metric | Train contextual | Train legacy | Held-out contextual | Held-out legacy |
|---|---:|---:|---:|---:|
| F0 MAE (semitones) | 1.850 | 1.905 | 1.993 | 2.105 |
| median contour correlation | 0.553 | 0.482 | 0.435 | 0.359 |
| declination error (st/s) | 0.218 | 0.343 | 0.287 | 0.340 |
| phrase-range error (st) | 2.209 | 2.577 | 2.803 | 3.129 |

The repeated-phrase regression verifies zero paired mean register difference
while requiring a nonzero later-phrase shape difference. This prevents the
previous cumulative frequency drift from returning.

Live Festival A/B output is under ignored directories:

```text
rendered_audio/prompt20_japanese_pitch_ab_no_drift_v5
rendered_audio/prompt20_japanese_duration_ab_source_anchor_v5
```

The pitch set rendered 6/6 WAVs. The duration/devoicing set rendered 45/45
clips with no failures. These clips are for listening; passing render checks do
not establish naturalness.

### Source-versus-synthesis waveform alignment audit

`japanese_alignment_verification.py` applies the same first-equal-phone
timeline alignment as the numeric benchmark, then plots the actual Kokoro
target waveform and the actual Festival output waveform on a shared time axis.
The visual order is target waveform, target-phone strip, correspondence panel,
synthesized-phone strip, synthesized waveform. Each red connector joins an
identified target boundary to its exact canonical synthesis counterpart, and
every matched phone is labelled with `synth duration - target duration` in
milliseconds. Waveforms use a per-pixel min/max envelope so short transients
remain visible when zoomed out.

The completed held-out audit requires exact canonical phone sequences, rejects
visual fixtures over two phrases, splits two-phrase utterances, and gives each
phrase its own first-phone origin. It selected five deterministic timing-error
strata from each of `test` and `validation`: ten global SVGs with raster PNG
companions, phrase-local SVGs, an overview SVG/PNG, and ten Festival WAVs, with
zero render failures. Every plot embeds the
transcript, corpus and synthesis readings, both real waveforms, both phone
strips, separate active-speech/pause timing, and source-relative acoustic-edge
measurements. All ten source hashes were unchanged. Median absolute boundary
drift ranged from `105.152 ms` to `689.342 ms`; the selected median examples
were `117.886 ms` (`test`) and `209.240 ms` (`validation`). The plots therefore
make a remaining limitation explicit: local phone-duration error improved over
the legacy model, but errors can still accumulate substantially across long
utterances. The images are verification evidence, not a naturalness claim.

Outputs are under the ignored directory:

```text
rendered_audio/prompt20_alignment_verification_short_v8
```

### Final-waveform vocal-tract validation

`vocal_tract_validation.py --source-root` uses stable source fixtures for all
five vowels and hashes them before and after analysis. Nine ratios cover both
ordinary endpoints, identity, intermediate points, and both expanded
endpoints. Results are under:

```text
rendered_audio/prompt20_vocal_tract_validation_final_v2
```

| Metric | Result |
|---|---:|
| vowels | 5 |
| transformed points | 45 |
| direction checks | 40/40 passed |
| identity waveform | exact for all vowels |
| maximum duration drift | 0 samples |
| maximum absolute F0 drift | 0.0644 semitones |
| clipped samples | 0 |
| median absolute formant error | 69.22 Hz |
| median absolute formant-ratio error | 0.02754 |
| median absolute formant target error | 49.24 cents |
| worst per-point median formant target error | 113.19 cents |
| formant-tracking rejected-frame rate | 0.0% |
| realistic-range compensated F1/F2 identity | 20/20 passed |
| expanded-range nearest-centroid warnings | 3/20 |
| median processing real-time factor | 0.476 |

The validator measures final transformed WAVs rather than trusting the internal
target envelope. Identity is a byte-equivalent sample bypass. F0 results are
reported per vowel; the maximum occurred on `/u/` at the expanded endpoint.
Nearest-vowel identity is measured after compensating F1/F2 for the requested
global `1 / ratio` movement, so it tests residual vowel-shape preservation
rather than penalizing the intended apparent-size shift. All ordinary-range
points retained their identity. Three deliberately expanded points crossed a
nearest-centroid boundary and remain explicit listening warnings; they are not
silently promoted to a naturalness pass.

The implementation uses bounded frame buffers and overlap-add arrays linear in
the input duration. Peak memory was not separately instrumented, so no numeric
memory claim is made.

### Listening and join fixtures

`vocal_tract_listening.py` renders 15 fixed Japanese fixtures covering isolated
and connected vowels, long vowels, pitch/accent independence, and creak. It
also creates blind sets for source, identity, realistic longer/shorter tract,
both expanded tract endpoints, pitch-only, tract-only, and combined conditions.

The completed current-model set contains 93 WAVs and 18 JSON reports under:

```text
rendered_audio/prompt20_vocal_tract_listening_v2
```

All 15 fixtures rendered. Identity remained exact and duration drift remained
zero. Median-F0 computed on independently accepted whole-utterance frames
shifted by at most `0.3404 st` in the realistic range and `0.2501 st` in the
expanded range. Comparing the same accepted frames before and after
transformation reduced those maxima to `0.00581 st` and `0.02790 st`,
respectively, identifying most of the larger values as estimator
frame-selection sensitivity rather than pulse-timing movement. Fifteen
join-discontinuity reports cover pre-transform, realistic longer/shorter, and
expanded longer/shorter conditions for connected, accent, and creak fixtures.

The blind manifest deliberately records `naturalness_verified: false`.

## GUI and persistence

The Speech parameter menu now includes `Vocal tract length`. Its generated
curve remains visible beneath an editable curve with at most two control points
per phone. The editor provides:

* a visible ratio and resonance-shift readout;
* keyboard editing and exact identity reset;
* reference-derived ordinary bounds;
* `Chipmunk range`, disabled by default;
* clamping when expanded mode is disabled;
* deterministic undo/redo and project persistence;
* final-waveform formant and spectral-envelope inspection;
* potential formant-jump markers for join investigation.

Changing pitch does not change the tract curve, and changing the tract curve
does not rewrite pitch targets. Manual voicing, mora voicing, pitch, duration,
unit choices, and tract edits preserve their established invalidation rules.

The Japanese Render Details view records active duration and pitch model IDs,
timeline/F0 statistics, mora-voicing decisions, phrase-shape behavior, and the
absence of cumulative register drift.

## Research basis and rejected approaches

The implementation follows the source/filter separation and envelope work in:

* A. Roebel and X. Rodet, *Efficient Spectral Envelope Estimation and Its
  Application to Pitch Shifting and Envelope Preservation*, DAFx 2005.
* D. Schwarz and X. Rodet, *Spectral Envelope Estimation and Representation for
  Sound Analysis-Synthesis*, ICMC 1999.
* IRCAM work by Degottex, Roebel, Rodet, Lanchantin, Farner, and collaborators
  on separately transformable excitation, envelope, pitch, duration, and voice
  quality.
* Hatano et al., *Correlation between Vocal Tract Length, Body Height, Formant
  Frequencies, and Pitch Frequency for the Five Japanese Vowels*, Interspeech
  2012.

Rejected production shortcuts:

* waveform resampling, because it couples pitch, formants, and duration;
* a fixed binary male/female formant shift;
* one fixed LPC order for all F0 and sample rates;
* moving hand-picked formants independently;
* copying Kokoro's absolute speaker pitch or duration scale;
* cumulative Japanese phrase-register drift;
* replacing voiced vowels with unfiltered white noise;
* claiming that a spectrogram alone establishes naturalness.

## Reproduction commands

The examples assume PowerShell variables rather than embedding private machine
paths:

```powershell
$Vault = Resolve-Path .
$Festvox = Join-Path $Vault "99_Tools/festvox"
$Python = Join-Path $Vault "tmp/festvox-test-runtime/Scripts/python.exe"
$SourceBank = Resolve-Path $env:FESTVOX_SOURCE_BANK
$ReferenceRoot = Resolve-Path $env:FESTVOX_FORMANT_REFERENCES
$KokoroArchive = Resolve-Path $env:FESTVOX_KOKORO_ARCHIVE
$KokoroCheckpoint = Resolve-Path $env:FESTVOX_KOKORO_CHECKPOINT
$Voice = Resolve-Path $env:FESTVOX_JAPANESE_VOICE
$env:PYTHONPATH = "$(Join-Path $Vault 'tmp/kokoro-align-runtime');$Festvox"
```

Stage A:

```powershell
& $Python (Join-Path $Festvox "prompt20_pipeline.py") run-stage-a `
  --archive $KokoroArchive --candidate-count 800 `
  --train-count 36 --validation-count 12 --test-count 12 --hash-archive `
  --checkpoint $KokoroCheckpoint --source-root $SourceBank `
  --diphone-index (Join-Path $Voice "dic/diphone_index.json") `
  --voice-manifest (Join-Path $Voice "dic/voice_manifest.json") `
  --references $ReferenceRoot --maximum-source-per-vowel 24 `
  --output (Join-Path $Festvox "rendered_audio/prompt20_stage_a")
```

Held-out prosody benchmark:

```powershell
$StageA = Join-Path $Festvox "rendered_audio/prompt20_stage_a"
& $Python (Join-Path $Festvox "japanese_prosody_benchmark.py") `
  --selection (Join-Path $StageA "kokoro_sample/partitions.json") `
  --alignments (Join-Path $StageA "kokoro_alignments") `
  --audio (Join-Path $StageA "kokoro_sample/wavs") --voice $Voice `
  --partition validation --partition test --frontend openjtalk `
  --output (Join-Path $Festvox "rendered_audio/prompt20_prosody_heldout_final_v7")
```

Held-out source-versus-synthesis waveform alignment images:

```powershell
& $Python (Join-Path $Festvox "japanese_alignment_verification.py") `
  --selection (Join-Path $StageA "kokoro_sample/partitions.json") `
  --alignments (Join-Path $StageA "kokoro_alignments") `
  --audio (Join-Path $StageA "kokoro_sample/wavs") --voice $Voice `
  --partition validation --partition test --per-partition 5 --max-phrases 2 `
  --frontend openjtalk --wsl-distro Ubuntu `
  --output (Join-Path $Festvox "rendered_audio/prompt20_alignment_verification_short_v8")
```

Live pitch and duration A/B sets:

```powershell
& $Python (Join-Path $Festvox "japanese_prosody_ab.py") $Voice `
  (Join-Path $Festvox "rendered_audio/prompt20_japanese_pitch_ab") `
  --frontend openjtalk --wsl-distro Ubuntu

& $Python (Join-Path $Festvox "japanese_duration_ab.py") $Voice `
  (Join-Path $Festvox "rendered_audio/prompt20_japanese_duration_ab") `
  --frontend openjtalk --wsl-distro Ubuntu
```

Five-vowel Stage B validation and listening set:

```powershell
& $Python (Join-Path $Festvox "vocal_tract_validation.py") `
  --source-root $SourceBank `
  --output (Join-Path $Festvox "rendered_audio/prompt20_vocal_tract_validation")

& $Python (Join-Path $Festvox "vocal_tract_listening.py") $Voice `
  (Join-Path $Festvox "rendered_audio/prompt20_vocal_tract_listening") `
  --frontend openjtalk --wsl-distro Ubuntu
```

Full deterministic Python test gate:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:PYTHONPATH = "$Festvox;$(Join-Path $Festvox 'festvox_gui')"
& $Python -m unittest discover -s $Festvox -p "test_*.py"

& $Python -m unittest discover `
  -s (Join-Path $Festvox "festvox_gui") -p "test_*.py"
```

Final results after the completed Stage B audit:

* repository suite: 355 tests passed in 44.891 s; 3 optional tests skipped
  (local Kokoro-Align checkpoint, PyWorld, and direct WSL Festival fixture);
* independent GUI suite: 172 tests passed in 245.129 s;
* real held-out correspondence audit: 10/10 Festival renders, zero failures,
  exact canonical phone sequences, and no example over two phrases.

Stage A verified all 754 source-bank files unchanged before and after analysis.
At final checkpoint verification, 753 of 754 files still matched the generated
voice manifest, but source-root `oto.ini` was edited by the user on July 17
after Stage A: expected SHA-256
`71950a4bc6d8a3a8099599157923ffde7bc331c70839832bb9a39deee14ac9ab`,
current SHA-256
`4b1848f2e4cf5baa3329b81c5b4467348beb6212f5df868693b347c73b95db70`.
The user explicitly approved preserving this mismatch for the Stage B
checkpoint. No Prompt 20 tool writes to or repairs the source bank. All source
WAV hashes used by the ten final audits remained unchanged before and after
rendering.

## Known limitations

* Kokoro boundaries and extracted F0 are silver references.
* The source speaker remains relatively monotone; the PSOLA-safe model permits
  more contour range than the recordings demonstrate, but does not guarantee
  natural expressive range.
* Automatic pitch error remains around two semitones on held-out vowel frames,
  and median contour correlation is moderate rather than high.
* First-phone-aligned waveform audits show that duration errors can accumulate
  to substantial boundary drift on long utterances, despite lower aggregate
  phone-duration error than the legacy model.
* The vowel-initial edge calibration has eight exact training phrases and four
  exact held-out phrases. It is bounded and versioned, but should be refit for
  banks whose pause-vowel OTO geometry differs materially.
* Polite-copula timing is structurally isolated, but the fixed sample did not
  support a stable `です`-specific coefficient. A blanket final-vowel rule was
  measured and rejected.
* The supplied vocal-tract references are `/e/` only and do not define a
  population range.
* Uniform spectral warping cannot reproduce all articulatory vowel changes.
* Final formant tracking is uncertain at high F0 and near Nyquist; both the
  identity-anchored envelope measurement and independent tracker are retained.
* Creak placement is structurally preserved and represented in listening
  fixtures, but its perceptual preservation is not yet human-verified.
* The blind outputs require human ratings for intelligibility, naturalness,
  speaker identity, vocal size, brightness, consonant damage, and source-quality
  changes.

## Principal files

Pitch and prosody:

```text
pitch_domain.py
japanese_pitch.py
japanese_duration.py
japanese_duration_corpus.py
japanese_devoicing.py
japanese_synthesis.py
japanese_prosody_benchmark.py
japanese_prosody_ab.py
japanese_alignment_verification.py
japanese_phrase_edges.py
profiles/japanese_pitch_model_v1.json
profiles/japanese_duration_priors_v1.json
```

Reference analysis and vocal-tract processing:

```text
formant_analysis.py
kokoro_reference.py
prompt20_pipeline.py
vocal_tract.py
vocal_tract_validation.py
vocal_tract_listening.py
rendered_formant_diagnostic.py
profiles/reference_voice_space_v1.json
PROMPT20_STAGE_A_FORMANT_ANALYSIS.md
```

Production and GUI integration:

```text
festvox_gui/festvox_core.py
festvox_gui/festvox_gui.py
source_filter_voicing.py
join_discontinuity.py
japanese_editing.py
japanese_refinements.py
```

Each module has a corresponding deterministic unit or integration test under
`test_*.py`; GUI tests run with `QT_QPA_PLATFORM=offscreen`.
