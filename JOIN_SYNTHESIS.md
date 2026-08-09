# Join synthesis

This note describes the active join policy shared by the WSL/Festival and
pure-Python synthesis routes. Join processing does not reinterpret text,
change phoneme timing, substitute contextual candidates, or replace a manual
per-occurrence recording choice.

## Design basis

Festival documents a UniSyn database as waveform, pitchmark, and index data;
its signal processing is pitch-synchronous and its diphone index explicitly
stores start, middle, and end landmarks:
<https://www.cstr.ed.ac.uk/projects/festival/manual/festival_20.html>.

Moulines and Charpentier describe pitch-synchronous Hanning windows longer
than one period, normally two to four periods with 50-75% overlap:
<https://doi.org/10.1016/0167-6393(90)90021-Z>. Accurate and consistently
interpreted epochs are therefore more useful than choosing a cut from zero
crossings alone.

Yamaha's public VOCALOID description confirms a concatenative architecture,
but does not expose a reproducible VOCALOID 2 join implementation. No
proprietary behavior is claimed or copied:
<https://ipsj.ixsq.nii.ac.jp/records/55748>.

## Pure-Python adaptive join

`join_synthesis.py` implements the measured join used by
`synth_diphone.py` after unit selection. Its conditioning schema is version 3.

1. Estimate the final left and first right periods independently. No period
   analysis window crosses the source cut. Phone hints force pauses, stop
   closures, and periodic-looking fricatives through the aperiodic path.
2. Search at most 12 ms into each already selected source around its annotated
   cut. The search cannot choose a different diphone or numbered take.
3. Rank candidate pairs by period mismatch, local RMS mismatch, normalized
   waveform correlation, amplitude-independent spectral-envelope distance,
   boundary novelty, and source movement. A small movement cost favors the
   annotated cut when acoustic evidence ties.
4. Match local gain gradually within the bounded 0.80-1.25 range.
5. Mix voiced material over approximately three pitch periods with
   complementary raised-cosine ramps. Aperiodic joins use a 10 ms
   raised-cosine overlap and omit period-only metrics.
6. Preserve the exact source trim and skip as `left_trim_samples` and
   `right_skip_samples`, then report the rendered handoff interval and every
   validation result in `splice_records`.

### Validation and fail-safe behavior

Candidate ranking is not sufficient by itself. The selected candidate must
also pass separate, configurable acoustic gates for:

- level step (6 dB by default);
- F0 step when both sides are periodic (1.5 semitones);
- best-lag period correlation (at least 0.45);
- pitch-period shape mismatch (at most 0.75);
- compact spectral-envelope distance (at most 1.35).

An independent content gate measures source-side retention and energy retained
inside the mixed region. This prevents a phase-cancelling or otherwise broken
join from appearing successful merely because it erased audible material.
Silence is allowed only where the supplied phone context identifies a pause or
a closure that can legitimately be quiet. The current defaults require at
least 30% source-collar retention and 45% retained phase-mix energy whenever
the corresponding audibility gate is active.

If content retention or an acoustic gate fails, the renderer discards the
moved cut, returns to the annotated source boundaries, and applies the exact
pre-fix linear mixing law over the already bounded overlap to the same selected
units. This validation fallback is distinct from enabling Fault Mode, which
restores the complete old fixed-duration route. The decision records
`validation_failures`, `content_fallback_used`,
`acoustic_validation_passed`, and `legacy_fallback_used`. Boundary-impulse
novelty is measured after rendering and remains visible even when the
fail-safe path was used; it does not silently choose another unit.

## Generated Festival voices

Normal Festival/WSL rendering uses the project-local
`native_unisyn/build/festvox-festival` helper. It embeds the same Festival
interpreter and loads the same voice Scheme, then replaces only the final
`us_generate_wave` step with `festvox_us_generate_wave`. Text analysis,
phonesets, Segment timing, TargetCoef/F0, the monotonic source/target map, and
the already selected Unit items are left intact. There is no post-render PCM
repair and the renderer never asks the selector for a different recording.

**Fault Mode > Legacy joins** bypasses this helper completely. It invokes the
installed stock Festival executable and stock `us_generate_wave`, so the fault
is an exact historical control rather than another setting in the new mixer.

### Source pitchmarks

`make_pitchmarks()` obtains the F0 guide from UTAU FRQ data when available and
otherwise from the configured WORLD fallback. It places both `pm/*.pm` and
`pm/*.legacy.pm` at the same negative-going, low-pass zero-crossing epochs.
The two files are intentionally identical in the current implementation.

The earlier residual-epoch experiment was rejected by real-bank A/B testing
because independent recordings did not retain one reliable common phase
convention. Join improvement therefore does **not** come from a new pitchmark
phase tracker. `pm/pitchmark_sources.json` schema 2 records the shared
negative-zero-crossing method, F0 provenance, and the normal and Legacy window
policies.

### Millisecond crossover runtime

The normal runtime specifies crossover length in milliseconds, not a fixed
period count. The sentence default is 40 ms and the supported request range is
0-100 ms. Each rendered occurrence may override the left and right side
independently.

For every eligible handoff the helper:

1. finds the handoff inside the shared phone from Festival's Unit map;
2. reads the GUI's explicit canonical phone class, falling back to the active
   Festival phoneset only for an unknown/third-party phone;
3. limits the request to that phone and to a conservative phone-class cap;
4. snaps voiced edges inward to complete target pitchmark frames;
5. traverses outgoing and incoming source frames across that interval;
6. mixes the two trajectories with complementary raised-cosine weights; and
7. emits a `GUIXOVER` record containing the requested sides, context cap,
   rendered interval, epoch count, retention check, context, and bypass reason.

The explicit class is attached to each Segment after the voice's UniSyn hooks
and before unit lookup. This corrects stale generated phonesets without
rebuilding the bank and also applies to built-in Kal. New English/Asaxi and
Japanese builds write accurate Festival `ctype`/`cvox` features.

| Shared phone | Maximum | Phone fraction | Minimum target intervals |
| --- | ---: | ---: | ---: |
| vowel | 80 ms | 60% | 3 |
| nasal, liquid, glide | 60 ms | 70% | 2 |
| voiced fricative | 50 ms | 65% | 2 |
| unvoiced fricative | 16 ms | 25% | 1 |
| stop or affricate | 8 ms | 20% | 1 |

The higher *fraction* for voiced continuants gives short sonorants and voiced
fricatives more of their available phone than vowels. The crossover never
extends into a neighboring phone merely to reach a target duration; very short
phones can therefore still render fewer intervals or report
`insufficient-target-context`.

The elapsed-time request therefore remains stable when F0 changes while the
number of epochs inside it varies naturally. A request can render shorter than
its nominal value because of the phone boundary, context cap, target-pitchmark
snap, an overlap with another crossover, or a safety bypass. Those limits are
reported rather than hidden.

Silence and unsafe phase-cancelling mixtures are bypassed. For built-in Kal,
the helper blends LPC envelopes through stable reflection coefficients while
mixing the corresponding residual frames. Generated waveform voices use the
same dual-frame target-epoch mixer without changing their source data.

The older sentence-wide UniSyn source-window controls remain available:
**Voice policy**, **Symmetric pitch periods**, **Asymmetric source periods**,
and a bounded `1.00`-`1.25` window factor. They affect the source frames fed to
the crossover and are distinct from the crossover's millisecond duration.

### Direct join editing

**View > Rendered joins in waveform** overlays every rendered handoff and
effective crossover span in the Speech waveform. Clicking a diamond selects
that occurrence and exposes green left/right handles. The inline readout keeps
three values separate:

- **requested**: the sentence default or per-occurrence override;
- **rendered**: the interval used by the last render; and
- **cap**: the current renderer context limit.

Dragging a handle stores only that occurrence's left/right millisecond
override. The waveform enters the normal Re-render-pending state, Undo restores
both the prior number and prior pending state, and Re-render applies the edit.
Timing edits remap the displayed join by its relative position in the edited
phone, so the span follows stretched or shortened phonemes.

Clicking the selected diamond again, clicking/dragging the waveform, hiding
the overlay, changing synthesis, or pressing Escape dismisses the join editor.
Dismissal is cosmetic and does not remove the occurrence override.

**Generate > Inspect joins and UniSyn windows...** provides the same controls
beside the read-only acoustic evidence. Its join table defaults to rendered
phone order; **Worst first** is optional. Opening it from a Recordings block
focuses the handoff rendered inside that phone. Requested settings and rendered
telemetry are persisted separately, and neither route changes phone timing,
F0, contextual selection, or an explicit recording override.

Legacy joins disables both editors and ignores saved crossover requests.

Re-render replaces the audio while restoring the exact visible waveform range
and playhead position. A fresh Generate retains its existing full-render
framing behavior.

### Persistent Festival worker

Normal native renders keep one Festival interpreter warm through the helper's
line-oriented `--server` protocol. The worker is serialized per backend,
restarted when the native executable or voice metadata changes, and recycled
after 32 jobs by default to bound Scheme/process state. Legacy joins remains a
stock one-shot Festival invocation so its exact comparison path is unchanged.
Windows-hosted metadata uses file-change tokens. A WSL-only voice is checked
through its `\\wsl.localhost\<distro>` view, using cheap metadata stats rather
than a new WSL process or a large-file digest. A configured native executable
is fingerprinted at its configured Windows/WSL path, not against the default
project binary.

Setting the sentence crossover to `0 ms` with no positive per-occurrence
override is an explicit crossover bypass and uses stock one-shot Festival.
A positive occurrence override still selects the native renderer even when the
sentence default is zero. This bypass is separate from **Legacy joins**:
Legacy also restores the historical source-window policy and ignores every
saved crossover override.

### Generated bridge alternatives

When a recorded transition is unavailable, the builder may compile a bounded
bridge from a stable left-phone tail and an onset from an existing recording.
The normal bridge is made with the same adaptive measured join described
above. The builder also writes a paired `_legacy_*.wav` using the exact old
fixed linear overlap and records sample-bounded Legacy EST geometry.

For missing vowel-to-consonant and pause-to-consonant transitions, the builder
retains a generic onset bridge and deterministic context-specific alternatives
keyed by the phone after the consonant. A matching alternative reuses the
consonant onset from a source recording that also supplies that following C-V
transition, making a source-continuous option available without binding the
two runtime choices. Runtime scoring prefers an exact
`recorded_right_context` over the generic `*` alternative.

These alternatives are added only when the transition itself is missing.
They do not replace recorded CVVC/VCV transitions, do not remove secondary
families, and never override an explicit manual variant. Stable candidate IDs,
source components, used slices, and join-conditioning telemetry remain in the
generated metadata. If the following C-V occurrence is manually changed to a
different numbered take, that manual choice remains final; the preceding
bridge is not allowed to rewrite it.

### Builder validation metadata

Every generated bridge retains its component-level join decision. The builder
also writes a deterministic `generated_bridge_validation` summary to both
`dic/diphone_index.json` and `dic/unit_alternatives.json`. It reports counts of
passed and failed candidates, fallback use, named validation failures, and the
conditioning schema version. A bridge that fell back remains traceable and is
not reported as an adaptive validation success.

## Fault Mode: Legacy joins

**Fault Mode > Legacy joins** persists with each sentence and project and is
available for both renderers:

- pure Python: exact pre-fix fixed-duration linear overlap;
- rebuilt generated Festival voice: the Legacy EST database, shared
  negative-zero-crossing pitchmarks, paired pre-fix bridge WAVs where bridges
  were generated, and symmetric UniSyn windows;
- built-in or older Festival voice: the same source database with the
  historical symmetric UniSyn window. The option is not ignored.

Changing this fault requires Re-render. It does not regenerate text, timing,
F0, contextual selection, or manual unit choices. A voice must be rebuilt with
the current builder to contain paired Legacy bridge WAV/index geometry; the
symmetric-window comparison still works for older voices.

An unchecked phrase fault means "inherit the sentence setting"; it is no
longer persisted as an explicit false override. Applying a project fault to
all sentences cannot therefore be silently defeated by stale phrase data, and
clearing all project faults also clears phrase-local faults.

## Real WSL verification

The bounded matrix in `tmp/join-real-validation` renders normal and Legacy
audio independently and compares candidate identities (or exact UniSyn unit
pair signatures for Kal), Segment boundaries, F0 targets, content retention,
broadband impulses, dropouts, and join diagnostics.

The July 23 matrix passed all ten cases:

- Japanese CV phrase and long-vowel fixtures;
- `lem_v4bi_integrated` English, Asaxi, and Japanese phrase/long-vowel routes;
- built-in Kal English phrase and long-vowel routes.

The production generated voices remained byte-for-byte unchanged during the
matrix. Kal and the pre-upgrade production Lem bank intentionally render with
their authored symmetric geometry. The current Japanese CV fixture uses its
new bridge database and qualified asymmetric windows. These checks establish
structural and acoustic non-regression; they do not replace human listening.

A July 24 focused runtime check used 20 ms `r/z/m/w` phones and natural English
text. Lem classified `r/m/w` as sonorants and `z` as a voiced fricative; the
short controlled phones rendered 12.12 ms (two intervals) rather than the old
4-6 ms stop cap. Natural Kal and Lem continuants used roughly 24-38 ms (four
to six intervals) where the phase-retention gate accepted the blend. Normal
and Legacy results retained identical Segment boundaries, F0 targets, and
selected units.

## Read-only diagnostics

`join_discontinuity.py` is the authoritative read-only analysis of final
audio. Synthesis-side gates protect bridge construction, while the rendered
diagnostic independently measures sample, level, F0, phase, period shape,
spectral trajectory, broadband impulse, and content-dropout evidence at the
actual handoff. See
[JOIN_DISCONTINUITY_DIAGNOSTICS.md](JOIN_DISCONTINUITY_DIAGNOSTICS.md).

Neither the component gates nor the optional severity ranking are validated
perceptual quality scores. Crossfading cannot conceal every large F0 or timbre
mismatch, and automated structure checks cannot establish naturalness.

## Focused tests

Run from `99_Tools/festvox`:

```powershell
py -3.14 -m unittest test_join_synthesis test_synthesis_efficiency `
  test_japanese_festival test_unified_voice_builder
py -3.14 -m unittest discover -s festvox_gui -p "test_*.py"
```

The focused suite covers same-phase and phase-shifted periodic joins, gain and
F0 mismatch, unvoiced input, long voiced phones, silence-as-a-false-fix
rejection, acoustic-gate fallback, deterministic bridge metadata,
context-specific consonant bridges, Legacy byte behavior and geometry, dynamic
UniSyn policy metadata, phrase-fault inheritance, and unchanged
contextual/manual unit selection.
