# Prompt 20 Stage A: Reference Analysis

Stage A completed before any production vocal-tract transform was written. The
machine-readable gate reported `passed: true` and
`production_transform_present: false`.

## Inputs and safety

The analysis used the project speaker's generated-voice provenance to locate
120 stable source-vowel segments, the seven supplied formant-shift WAV files,
and a deterministic 60-utterance Kokoro sample split into 36 training, 12
validation, and 12 test records. The supplied recordings are all the vowel
`/e/`; their ratios are not projected onto `/a i u o/`.

The nested Kokoro archive is read with path, link, device, size, and member
count checks. It is never extracted with `TarFile.extract`. The complete archive
hash is recorded in the profile. The source UTAU bank is opened read-only: 754
files were verified before and after analysis with the same content-inventory
SHA-256, `9bf4cbcf9b2fbdbce060b4243a0965eabf49a9a2a6fc87cc6e40db465d6250ec`.

Kokoro-Align's epoch-200 CTC checkpoint was actually run. All 60 sampled
utterances were accepted, with median alignment confidence `0.745323`.
Punctuation and spaces absent from its 39-label encoder remain represented but
are explicitly marked as interpolated, lower-confidence boundaries. The labels
are silver timing evidence, not perfect ground truth.

## Formant method

The primary estimator is an iterative F0-adaptive cepstral true envelope. Its
max-and-resmooth loop uses a 2 dB convergence tolerance, and its cepstral order
changes with F0 so sparse high-F0 harmonics are not mistaken for the vocal-tract
envelope. A dynamically ordered Burg LPC tracker is retained independently for
every frame. The implementation follows the estimator concerns in Roebel and
Rodet's [DAFx 2005 paper](https://www.dafx.de/paper-archive/2005/P_030.pdf) and
the separation of excitation and spectral envelope described by Schwarz and
Rodet's [IRCAM work](https://hal.science/hal-01161231v1/document).

Frames remain inspectable when rejected. Reasons cover missing or ambiguous
F0, low voicing, probable devoicing, creak, short stable bodies, implausible
bandwidth, near-Nyquist peaks, estimator disagreement, and trajectory jumps.
Of 5,294 retained frames, 4,562 were accepted. Median true-envelope/Burg
disagreement on source material was 33.9, 35.2, 96.6, and 52.2 Hz for F1-F4;
the respective 90th percentiles were 164.8, 99.3, 499.5, and 134.5 Hz.

Source-speaker segment medians were:

| Vowel | Accepted | F1 | F2 | F3 | F4 |
|---|---:|---:|---:|---:|---:|
| `/a/` | 24/24 | 775.2 | 1615.0 | 3143.8 | 3572.5 |
| `/i/` | 23/24 | 335.3 | 2581.3 | 3100.3 | 3477.2 |
| `/u/` | 24/24 | 452.2 | 1501.9 | 2947.4 | 3507.2 |
| `/e/` | 23/24 | 581.4 | 1997.2 | 3065.8 | 3478.9 |
| `/o/` | 24/24 | 635.2 | 1270.5 | 3071.2 | 3645.6 |

These are acoustic estimates, not anatomical measurements. Hatano et al.'s
[Japanese-vowel study](https://www.isca-archive.org/interspeech_2012/hatano12_interspeech.pdf)
is the reason each vowel must be validated separately after transformation.

## Derived control range

The versioned runtime profile is
`profiles/reference_voice_space_v1.json`:

* Identity: `1.0`
* Ordinary range: `0.93363995` to `1.05740237`
* Observed supplied-reference span: `0.88083696` to `1.16684161`
* Stage B validated expanded engineering range: `0.65` to `1.50`

The ratio is `target_length / source_length`; values below one raise
resonances and values above one lower them. This is a robust, bounded acoustic
engineering range, not a sex or gender threshold. The `/e/` sweep informs the
observed span. Stage B subsequently tested `0.65` through `1.50` on final
waveforms for all five project-source vowels and retained those bounded
engineering endpoints after direction, identity, duration, F0, clipping, and
Nyquist-safety checks. See `PROMPT20_IMPLEMENTATION_REPORT.md`.

Two independent reruns generated 21 byte-identical CSV, JSON, report, and SVG
artifacts. Runtime defaults are loaded from the profile rather than duplicated
in code.

## Outputs

`prompt20_pipeline.py run-stage-a` creates ignored analysis artifacts beneath a
chosen output directory:

* `formant_frames.csv` and `formant_segments.csv`
* `speaker_formant_summary.json` and `reference_voice_space.json`
* `formant_analysis_report.md` and `reference_manifest.json`
* fitted Kokoro duration priors and held-out benchmark JSON
* source-safety and Stage A gate JSON
* ten SVG formant plots plus per-utterance alignment plots

The committed source and profile contain no private absolute paths. A
reproducible invocation is:

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

& $Python (Join-Path $Festvox "prompt20_pipeline.py") run-stage-a `
  --archive $KokoroArchive `
  --candidate-count 800 --train-count 36 --validation-count 12 --test-count 12 `
  --hash-archive `
  --checkpoint $KokoroCheckpoint `
  --source-root $SourceBank `
  --diphone-index (Join-Path $Voice "dic/diphone_index.json") `
  --voice-manifest (Join-Path $Voice "dic/voice_manifest.json") `
  --references $ReferenceRoot `
  --maximum-source-per-vowel 24 `
  --output (Join-Path $Festvox "rendered_audio/prompt20_stage_a")
```

## Stage B boundary

Stage B may implement one source-filter transform only after loading and
validating this profile. It must reanalyze the final rendered waveform, preserve
F0 and duration independently, test all five vowels, retain consonant
transients, and report identity equivalence, formant-direction accuracy,
Nyquist behavior, creak effects, joins, clipping, and processing cost. A global
uniform warp remains an approximation unless a measured ablation justifies a
smooth vowel-conditioned correction.
