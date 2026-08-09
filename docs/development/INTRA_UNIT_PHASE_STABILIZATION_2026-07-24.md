# Intra-unit phase stabilization checkpoint

Date: 2026-07-24

## Finding

A user-labelled English render produced the same audible problem locations
with `lem_v4bi_integrated` and `lem_v4bi_integrated-new`. Their relevant source
WAV and pitchmark files were byte-identical. Segment timing, target F0, selected
units, manual overrides, and the UniSyn map were also identical between normal
and control renders.

Only one labelled location was clearly inside a rendered crossover. Several
others landed on nonuniform integer source-frame steps inside an already
selected unit. Adjacent source periods at the strongest examples had poor
zero-lag waveform correlation but high correlation after a small local lag.
Wholesale pitchmark replacement was not supported by the local F0 evidence.

## Runtime change

The project-local native UniSyn renderer now applies a conservative
same-recording phase correction in periodic contexts:

- only vowels, sonorants, and voiced fricatives are eligible;
- unit boundaries and declared crossover epochs are excluded;
- correlation is non-wrapped and the search is limited to one quarter of the
  smaller local source period;
- the original correlation must be below `0.82`;
- corrected correlation must reach `0.90`;
- improvement must be at least `0.15`;
- short correction runs touching a crossover are rejected, while longer runs
  taper over their two edge epochs.

The renderer moves the source-frame center only. It does not change the
selected recording, target epoch, Segment duration, target F0, gain, or
waveform after rendering. An experimental multi-frame source-area resampler is
present for controlled diagnostics but defaults off because it changed a
wider acoustic neighborhood without a direct labelled benefit.

`Fault Mode > Legacy joins` remains an exact bypass through stock Festival.

## Diagnostic provenance

Native output records accepted changes as `GUIFRAMEFIX`. The core parser stores
target time/index, source-frame pair, center shift, correlation before/after,
correction kind, and reason in `Synthesis.frame_trajectory_records`.

Phrase concatenation offsets this provenance, and project JSON preserves it.
The on-demand rendered join inspector shows teal epoch markers and a
**Source trajectory** table in render order. No diagnostic graph is generated
until the user opens or exports a diagnostic.

## Verification

The paired Lem sentence retained 80 segments, 78 exact handoffs, 1,102 target
pitchmarks, identical duration, and unchanged recording choices. The normal
render reported 72 stabilized epochs for the new bank and 66 for the old bank;
Legacy reported zero.

At the user-labelled transitions, period-shape mismatch changed as follows:

- `0.231 -> 0.204`;
- `0.537 -> 0.138`;
- `0.510 -> 0.204`.

Unchanged labels stayed unchanged. One corrected location improved period shape
while its compact spectral metric worsened slightly; this remains visible and
requires human listening rather than being presented as resolved.

A matched five-run warm benchmark used the same Lem voice and native crossover
with only phase stabilization toggled:

- disabled median: `0.1634 s`;
- enabled median: `0.1872 s`;
- measured kernel overhead: `14.5%`;
- repeated PCM and selected-unit maps were deterministic.

Built-in Kal rendered nonempty audio with one accepted correction. With
correction off/on it retained identical Segment boundaries, target F0,
duration, and join-diagnostic summary.

Focused parser, phrase-combination, GUI, and project-round-trip tests cover the
new provenance path. The complete root suite passed 444 tests with five
optional integrations skipped; the GUI-directory suite passed 262 tests.

The read-only `Lem_V4Bi_Civet/3_E3/oto.ini` retained SHA-256
`4B1848F2E4CF5BAA3329B81C5B4467348BEB6212F5DF868693B347C73B95DB70`,
matching the pre-audit checkpoint. Acoustic naturalness remains a listening
judgement.

## Temporary audit artifacts

Ignored evidence and scripts are under `tmp/join-user-label-audit/`. They
include the saved render, controls, mapped labels, per-label measurements, and
`label_comparison.bmp`. They are deliberately not project fixtures.
