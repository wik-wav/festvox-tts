# Source-window label audit (2026-07-24)

## Problem

A human-labelled English render from Lem V4Bi contained seven abrupt acoustic
events. The same events occurred with both the older separate-file build and
the newer grouped-cache build. Their shared selected recordings, phone timing,
target F0, and source pitchmarks showed that the remaining problem was not a
grouped-cache change or contextual-selection regression.

The epoch trace placed several events inside one selected unit. A normal
120 ms source half could traverse multiple source periods during one short
target period, compressing a natural source trajectory into an abrupt output
change.

## Controlled build

An isolated one-pitch Lem V4Bi build changed only the adaptive normal
source-window cap from 120 ms to 60 ms. It retained:

- the same source OTO scope;
- the same 164.8 Hz speaker estimate;
- all 17,340 indexed diphones and alternative takes;
- recording-first contextual/manual selection;
- hidden full-side variants for stretched phones;
- grouped UniSyn runtime storage;
- read-only source-bank access.

The exact labelled sentence rendered in 2.22 seconds. Phone-relative comparison
against the current 120 ms build found:

| Label | Period-shape mismatch | Spectral step |
| --- | ---: | ---: |
| 1 | 0.448 -> 0.309 | 1.768 -> 1.533 |
| 2 | 0.508 -> 0.176 | 1.374 -> 1.381 |
| 3 | 0.069 -> 0.027 | 0.502 -> 0.398 |
| 4 | 0.204 -> 0.205 | 1.310 -> 1.023 |
| 5 | 0.137 -> 0.160 | 1.380 -> 1.293 |
| 6 | 0.138 -> 0.076 | 0.770 -> 0.492 |
| 7 | 0.204 -> 0.088 | 0.997 -> 0.480 |

No contextual recording selection changed after source-window suffixes were
normalized. One broadband detector increased at label 6 while both
period-shape and spectral metrics improved; that detector remains supporting
evidence rather than an automatic repair authority.

## Decision

`DEFAULT_SOURCE_WINDOW_MS` is now 60 ms for the shared English, Asaxi, and
Japanese builder path. This changes future generated voices only. Existing
voices retain their stored geometry until rebuilt.

Japanese paired CVVC/VCV units may widen an individual primary half beyond
60 ms when that is required to retain the compiler's declared phone-center
anchor. This is a structural minimum, not a return to whole-sample mapping:
ordinary units remain capped at 60 ms and the generated metadata records the
effective per-unit window.

The setting remains reversible:

```text
--source-window-mode adaptive --source-window-ms 120
--source-window-mode full
```

Adaptive mode still chooses a full-side variant of the same recording when a
stretched target phone can accommodate it. The change does not re-rank
contextual candidates, replace a manual take, normalize gain, modify F0, or
post-process rendered PCM.

## Rejected runtime experiment

A bounded two-frame runtime bridge was prototyped against the same labels. It
activated only at unrelated epochs and left all seven labelled measurements
unchanged. The prototype was removed. The existing conservative same-unit
phase-reference correction remains; broad source-area resampling remains
disabled by default.

Ignored working evidence is under `tmp/join-user-label-audit/`, including the
60 ms test WAV/JSON, label comparison data, and build logs.

## Verification

The final repository gate used the project Python environment for all
top-level signal-processing, builder, English, Asaxi, and Japanese tests, then
the installed Python 3.14 Qt environment for the offscreen GUI suite:

- 445 top-level tests passed; four optional tests skipped cleanly;
- 262 GUI tests passed;
- 707 tests passed in total;
- the focused source-window/Japanese-assembly gate passed 46 tests;
- Kal rendered nonempty audio with identical Segment timing, F0 targets,
  duration, and selected units with the phase correction enabled or disabled;
- the audited Lem `3_E3/oto.ini` SHA-256 remained
  `4B1848F2E4CF5BAA3329B81C5B4467348BEB6212F5DF868693B347C73B95DB70`.

These checks establish structural compatibility and deterministic diagnostics.
They do not claim that the remaining labelled artifacts are inaudible; the
60 ms listening render remains available for human comparison.
