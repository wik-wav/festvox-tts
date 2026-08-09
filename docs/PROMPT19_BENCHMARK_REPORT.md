# Prompt 19 Benchmark Report

This report records the completed deterministic benchmark. Paths are relative
to `99_Tools/festvox`; generated artifacts live under ignored
`rendered_audio/` and are not committed.

## Reproduction

```powershell
python japanese_duration_ab.py `
  generated_voices/<integrated-voice> `
  rendered_audio/prompt19_duration_ab `
  --frontend auto `
  --wsl-distro Ubuntu

python voicing_validation.py `
  <real-voiced-reference.wav> `
  rendered_audio/prompt19_reference_validation `
  --prefix e_reference
```

The A/B systems are fixed in
`profiles/japanese_duration_validation_v1.json`: legacy timing/realization,
contextual timing with shortened-voiced fallback, and contextual timing with
automatic source-filter realization.

## A/B render results

- 15 fixed utterances, 3 systems, 45/45 synthesis clips rendered.
- 49 WAV files, 2 JSON files, 1 Markdown summary, and 1 spectrogram PNG were
  produced (the extra WAVs are controlled renderer experiments).
- Mean contextual-minus-legacy utterance duration: `-0.033333 s`.
- Median contextual-minus-legacy duration: `-0.020000 s`.
- Mean absolute duration change: `0.045333 s`.
- 23 eligible high-vowel decisions: 20 source-filter applications and 3
  accurately labelled shortened-voiced fallbacks.
- Coverage: 4 `/i/`, 19 `/u/`, 3 stop fixtures, 4 fricative fixtures, and one
  deliberately retained ambiguous fixture.
- Median local periodicity changed from `0.738249` to `0.439220`; mean drop was
  `0.248523`.
- Median tract-envelope distance for applied cases: `0.299304`.
- Maximum absolute local level change: `11.535651 dB`. This is retained as a
  separate measurement: natural devoicing is not required to preserve voiced
  RMS, and no naive global level normalization is applied.

The benchmark is an objective structural check, not a naturalness score.

## Supplied-reference calibration

The same 220 ms analysis was applied to a supplied real voiced vowel, real
devoiced vowel, the previous synthetic endpoint, and the current zero-voicing
endpoint:

| Signal | RMS dBFS | Periodicity | Harmonic contrast dB | Centroid Hz |
| --- | ---: | ---: | ---: | ---: |
| Real voiced | -19.3124 | 0.983737 | 8.5954 | 474.3 |
| Real devoiced | -34.5706 | 0.185592 | 1.7549 | 1020.5 |
| Previous synthetic devoiced | -25.5106 | 0.133886 | 1.0264 | 1542.5 |
| Current zero-voicing endpoint | -34.8657 | 0.286993 | 1.2195 | 924.4 |

The current endpoint matches the real reference level within about `0.30 dB`
and is much closer in spectral centroid than the previous synthetic result.
Its tract-envelope correlation to the voiced source is `0.995172` (the real
devoiced reference is `0.900224`). Periodicity is still higher than the real
reference; this remains a concrete limitation.

The committed validation implementation also reports a harmonic-contrast drop
of `7.381885 dB`, contrast ratio `0.141183`, tract-envelope correlation
`0.995169`, and envelope distance `2.475885 dB`. It passed the deterministic
"periodicity removed", "harmonic ridges reduced", and "tract envelope
retained" gates.

## Direct PSOLA experiment

The generated voice uses direct waveform TD-PSOLA, not an LPC residual voice.
The installed Festival path rejected an utterance with absent explicit F0.
When a controlled 300 ms high vowel was forced to a 40 Hz target, the rendered
vowel remained unmistakably periodic: longest-vowel periodicity `0.959286`,
median periodicity `0.903006` over a 25-450 Hz search. A normal duration-only
render measured `0.672091`; contextual source-filter output measured
`0.516116` in the same experiment.

Conclusion: this TD-PSOLA path can time-map periodic or aperiodic source frames
but does not itself remove periodic excitation. No code path reports F0 removal
as successful devoicing.

## Moraic nasal probe

Read-only plans were generated for phrase-final, pre-nasal, pre-velar,
pre-labial, and loanword contexts in both a Japanese-only CVVC voice and an
integrated ARPAsing/Japanese voice.

- Japanese-only canonical `/N/`: `108.3-135.6 ms` in the tested contexts.
- Integrated mapped aliases (`nn`, `nng`, `mm`, `xn`): `85.5-143.8 ms`.
- Every mapped occurrence carried `timing_role="moraic_nasal"`.
- Consonant-only timing edits excluded these regions; rhyme/vowel timing edits
  included them.
- Non-Japanese `nn` retained consonant behavior.

This closes the reported pathological long-`nn` behavior without applying one
universal allophone spelling or changing bank-specific routing.

## Listening status and risks

No human listening judgment has been recorded for the final clips. Acoustic
naturalness is therefore unverified. The generated set is ready for randomized
A/B listening under `rendered_audio/prompt19_duration_ab/`.

Known risks:

- three short/low-confidence high-vowel cases used the shortened-voiced
  fallback;
- zero-voicing periodicity remains above the supplied real reference;
- source-filter analysis is an engineering approximation, not full glottal
  inverse filtering;
- JSUT and licensed CSJ corpora were not locally supplied for a speaker-matched
  fit, so the shipped contextual coefficients remain conservative priors;
- objective timing and spectral metrics cannot establish intelligibility,
  voice identity, or perceived naturalness.

## Verification

- Focused Prompt 19 suite: 42 tests passed.
- Repository root suite: 269 tests passed; 2 optional tests skipped because
  `pyworld` and local WSL Festival/EST tools were unavailable to that test
  process. The installed `pyopenjtalk` integration test ran and passed.
- Synthesis core suite: 72 tests passed.
- GUI suite: 88 tests passed. A single combined host-runner invocation stalled;
  the same suite then completed in four bounded groups of 22 tests with no
  failures.
- Changed Python modules passed `compileall`.
- Offscreen visual inspection passed for the Speech waveform, red phoneme
  boundaries, and the denser automatic/manual voicing curves. The ignored
  screenshot is `rendered_audio/prompt19_gui_visual.png`.

The two source subbanks were audited read-only using the repository's
relative-path, NUL separator, and file-byte SHA-256 convention. Windows
`desktop.ini` metadata is excluded from the content manifest; the one present
predates this work.

| Source scope | Files | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| Integrated single-pitch source | 1,738 | 419,583,059 | `579b16800bc25b04a183b5d62cf836d71ab7a643d89e964657eed75a016cd77e` |
| Japanese single-pitch source | 467 | 111,806,607 | `b6a874a371397e0825d05ed5bee87085befd5fe174f18d435b591d5059458f80` |

No task-era source-bank modification was observed. Generated voices,
benchmark WAVs, plots, JSON diagnostics, caches, and screenshots remain in
ignored output directories.
