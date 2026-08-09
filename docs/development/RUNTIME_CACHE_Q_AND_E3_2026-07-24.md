# Runtime cache, integrated q, and E3 pitch checkpoint

Date: 2026-07-24

This checkpoint fixes a newly generated integrated voice taking several
seconds to render a short utterance even though Festival/UniSyn itself needed
only a few milliseconds. It also restores the integrated `q` phone to the
English entry point and ordinary English inline-phone input.

## Render latency root cause

`dic/diphone_index.json` contains both runtime policy/index data and a complete
copy of the contextual alternatives audit graph. A current integrated Lem
build has thousands of indexed units and tens of megabytes of nested choice
metadata. The GUI previously:

1. parsed that entire file for every `voice_metadata()` query;
2. recursively froze and sized the complete graph;
3. rejected it from the bounded metadata cache because it exceeded that
   cache's half of the 64 MiB voice budget; and
4. repeated the work several times during one render.

The runtime metadata publisher now omits the duplicate `alternatives` and
`alias_metadata` build graphs. The complete immutable contextual records remain
available through `dic/unit_alternatives.json` and `unit_alternatives()`.
Runtime policy, phones, language entry points, source-window policy, speaker
pitch, and the compact diphone index remain available through
`voice_metadata()`.

The existing 64 MiB total voice-cache limit is unchanged. Compact metadata
receives one quarter and contextual alternatives receive three quarters, which
fits one current integrated Lem inventory while allowing normal LRU eviction
when the user changes voices. Rebuild fingerprints and explicit cache clearing
still invalidate both owners together. The fingerprint also stats the active
and legacy UniSyn `.est` indexes and packed `.group` payload. Rebuilding those
runtime files in place therefore restarts the persistent worker instead of
leaving its loaded database stale. This is a metadata-only stat check; it does
not read or hash the large group file during rendering.

## Measured result

The same short `aa q aa` Festival/WSL render measured:

```text
pre-fix full-metadata path:  8.690 s
optimized cold path:         0.602 s  (93.1% less time, 14.4x)
optimized warm path:         0.039 s  (99.6% less time, 225.2x)
older voice warm median:     0.031 s
```

The matched pre-fix/optimized render used one voice, phone list, duration plan,
pitch, and join configuration. Its PCM SHA-256, sample count, Segment
boundaries, F0 targets, selected units (including `q__u3`), and skipped-unit
list were identical. In an alternating three-language check, the new bank
matched the old bank for ten phones and was 7-12% faster for 80- and 400-phone
sequences. These are local engineering measurements, not latency guarantees.
After adding the runtime-file fingerprint, a final four-render check measured
0.526 s cold and a 0.0324 s warm median, again selecting `q__u3` with no
skipped units.

The cache projection cannot alter Scheme, selected units, Segment timing, F0
targets, pitchmarks, or PCM.

## Integrated q behavior

Generated integrated English entry points call Kal's text frontend and then
restore the generated voice's phoneset. This keeps Kal's English lexicon and
prosody while allowing generated superset symbols such as `q`.

The GUI's inline-phone filter now reads the selected generated voice's declared
`phones` inventory. It falls back to Kal's narrow `radio` inventory only for
Kal or legacy English voices without current metadata. Therefore:

- direct phoneme `q` renders in English, Asaxi, and Japanese;
- English text `[q]` is retained for an integrated voice;
- Asaxi's ordinary glottal-stop route remains unchanged; and
- built-in Kal still rejects or omits unsupported `q` instead of crashing.

Contextual and manual recording choices are not rewritten.

## E3 pitch authority

Without explicit `--f0`, the builder stores the measured median of only the
selected OTO/WAV scope. It applies no automatic melodic-headroom
transposition. Current E3 Lem artifacts agree across:

```text
dic/voice_manifest.json       164.81 Hz
dic/diphone_index.json        164.81 Hz
dic/unit_alternatives.json    164.81 Hz
generated Scheme target mean  165 Hz (display-rounded)
```

New manifests use `default_pitch_source: speaker_median` and
`automatic_pitch_headroom_semitones: 0.0`. An explicit builder `--f0` remains
final. The GUI reads an older `speaker_median_plus_headroom` manifest through
its stored source median rather than displaying the transposed value.

## Diagnostic plots

Join-discontinuity analysis, broadband spectrogram rendering, source
pitchmark plots, and final-waveform formant analysis remain strictly
on-demand. Ordinary Generate, Re-render, and waveform display do not calculate
or draw those diagnostics. A GUI regression test enforces this boundary.

## Generation progress

Every Generate and Re-render operation exposes one indeterminate
current-sentence bar for its complete backend and signal-processing lifetime.
Batch operations add a determinate completed/total bar directly beneath it.
The pair occupies one fixed 26-pixel progress area, divided into two 13-pixel
bars with no gap; the status bar remained 31 pixels high in baseline,
single-render, and batch visual checks. A lone bar expands to the full reserved
height. Success, failure, and cancellation paths all remove the current bar.

## Focused verification

- compact metadata is recursively immutable and remains cached;
- the alternatives cache receives the documented budget;
- ordinary render display invokes no diagnostic analyzer or plot renderer;
- generated English inline `q` follows the declared voice inventory;
- Kal retains its compatibility filter;
- real Festival/WSL `aa q aa` renders for both integrated voices under all
  three language entry points with no skipped units;
- source UTAU files remain read only.

Final automated results:

```text
root suite:          442 passed, 5 optional integrations skipped
Festival core:      112 passed
offscreen GUI:      147 passed
total:              701 passed, 5 optional integrations skipped
```
