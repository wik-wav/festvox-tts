# Rendered Join Discontinuity Diagnostics

## Scope

`join_discontinuity.py` performs deterministic, read-only analysis of the
rendered waveform at each known concatenative handoff. It does not normalize,
crossfade, repair, or replace audio. The BS.1770 K-weighted loudness curves
remain available, but loudness is only one independent measurement.

The analyzer supports both renderers:

- Festival/UniSyn exports rendered `TargetCoef` epochs and the monotonic
  `US_map` source-to-target frame mapping. The nominal splice is the midpoint
  between the final target epoch mapped to the incoming unit and the first
  target epoch mapped to the outgoing unit. Both handoff epochs are retained.
- The pure-Python renderer records its exact crossfade start, end, and nominal
  midpoint.
- Old projects and other renderers remain compatible. If exact metadata is
  unavailable, the analyzer uses the shared-phone center and marks the
  position as estimated.

Festival 2.5 appends one or more endpoint rows that wrap the last target epoch
back to source frame zero. `parse_unisyn_render_diagnostics()` reconstructs the
monotonic map and excludes only that trailing relation sentinel.

## API and schema

```python
from join_discontinuity import JoinAnalysisConfig, JoinDiscontinuityAnalyzer

report = JoinDiscontinuityAnalyzer(
    synthesis.samples,
    synthesis.sr,
    synthesis.segments,
    target_pitchmarks=synthesis.target_pitchmarks,
    splice_records=synthesis.splice_records,
    selected_units=synthesis.selected_units,
    alternatives=backend.unit_alternatives(synthesis.voicebank),
    config=JoinAnalysisConfig(),
).analyze()
```

`diphone_loudness.analyze_rendered_joins()` is a backward-compatible wrapper
around the same analyzer. Returned data is strict JSON: unavailable metrics are
`null`, never a misleading zero or NaN.

The current report is `schema_version: 6` with method
`continuous-pitch-synchronous-formant-content-v6`. Schema 6 adds rendered
content-retention/dropout evidence and the stricter broadband spectral-shape
evidence described below. Each join retains raw metrics, local baselines,
component scores, issue labels, source provenance, and plot data. Summary
fields include unexpected broadband-event and content-dropout counts as well
as expected low-energy handoff counts.

## Measurements

Every join retains separate raw measurements and local baselines.

1. **Level discontinuity**
   `left_rms`, `right_rms`, and `level_step_db` use the final complete period
   on the left and first complete period on the right for voiced joins. Other
   joins use independent 12 ms windows. The waveform is not normalized first.

2. **Immediate discontinuity**
   `sample_value_jump` is the exact cross-splice sample step.
   `cross_splice_slope` is compared with `left_local_slope` and
   `right_local_slope`; `slope_jump` therefore exposes the derivative impulse
   created by a click or DC step. Central second-derivative novelty is also
   retained. Ranking can compare a gain-compensated jump, but the original raw
   values are never replaced.

3. **Pitch period and F0**
   Rendered target epochs are preferred. Fractional-sample interpolation avoids
   changing harmonic shape when a period is not an integer number of samples.
   Separate-side autocorrelation is the fallback and never crosses the splice.
   The report includes period samples/seconds, left/right F0, semitones, cents,
   periodicity, and period provenance.

4. **Phase**
   The two boundary periods are resampled to a common length, DC-removed, and
   RMS-normalized. The analyzer reports actual zero-lag correlation, best
   correlation within the configured local circular-lag range, and the best
   offset in samples and cycles. Best alignment never replaces the zero-lag
   evidence.

5. **Period shape**
   `period_shape_mismatch = 1 - best_lag_period_correlation`.
   `period_waveform_error` provides a second normalized error measure after
   best alignment.

6. **Spectral and timbral trajectory**
   Voiced periods use amplitude-independent log harmonic-shape values. The DC
   term and overall peak are removed, and harmonics below the configurable 1%
   floor are ignored to prevent fractional-sample leakage from becoming false
   timbre evidence. Unvoiced frames use low-order real cepstral coefficients
   from a Hann-windowed log spectrum, excluding energy.

   Separate least-squares trajectories are fitted over up to five left and
   five right periods/frames. `spectral_step` compares their values
   extrapolated to the splice. `spectral_slope_break` compares their local
   slopes over one reference period. Boundary flux and short-versus-medium
   window differences remain separate.

7. **Local novelty**
   Major boundary values are compared with adjacent changes wholly inside the
   incoming and outgoing units. The robust score uses the local median and MAD
   with deterministic absolute/relative floors. The report keeps the raw
   value, novelty, baseline median, baseline MAD, and baseline count.

8. **Broadband impulse/crackle evidence**
   A multiresolution 0.5, 1, 2, and 3 ms scan covers the reported handoff
   collar. For each frame it measures the median spectral floor in frequency
   bands, band and bin flatness, spectral tilt, inter-band uniformity, and the
   local full-band energy ratio.

   A large spectral-floor novelty is only corroborating evidence. It cannot by
   itself produce `BROADBAND_IMPULSE`. The event must have independent
   amplitude-normalized spectral-shape support: either an absolutely flat,
   weakly tilted, uniform full-band shape or at least two locally novel signs
   among increased flatness, flattened tilt, and increased band uniformity.
   That shape evidence is then gated by the local energy ratio; floor novelty
   only adjusts ranking after the shape test succeeds. This prevents ordinary
   steeply tilted voiced spectra from being labeled as crackles merely because
   their energy changes quickly.

   Sustained frication is not treated as a crackle solely because it contains
   high-frequency energy. Stop, affricate, and closure phones (`p b t d k g q
   cl ch jh ts dz dx`, including palatalized stops such as `ky`) annotate a
   detected event as a potentially expected release burst. The annotation
   never removes raw evidence or changes the selected unit.

9. **Rendered content retention/dropout**
   Schema 6 scans short RMS frames across the complete declared handoff and
   compares them with references wholly before and after it. For voiced joins,
   the frame is at least one estimated pitch period and the scan hop is
   pitch-synchronous; other joins use the configured fixed frame and hop. The
   report preserves handoff RMS, median and minimum frame RMS, both
   reference RMS values, retention ratio, attenuation in dB, frame geometry,
   and whether pitch-synchronous analysis was used.

   A dropout contributes to severity only when both source-side references are
   audible. Pauses and real stop/affricate closures retain their raw dropout
   measurements but are marked `content_dropout_expected` and excluded from
   the dropout ranking component. The flap `dx` is intentionally not granted a
   blanket low-energy exemption. This is the anti-silence check: a join cannot
   look improved simply by cancelling an audible vowel or sonorant, while
   silence over a legitimate closure is not automatically treated as damage.

10. **Formants and full spectral envelope**
    When both sides are voiced and above the energy/confidence floor, separate
    LPC analyses estimate up to F1-F4, bandwidth, prominence, and normalized
    energy. Left and right observations never cross the splice. Independent
    robust trajectories are extrapolated to the boundary, and ambiguous or
    implausible tracks are rejected rather than forced into nearest-frequency
    matches.

    The report keeps frequency jump, slope break, bandwidth jump, prominence
    jump, normalized formant-balance jump/novelty, full log-envelope step, and
    full-envelope trajectory break. Energy-normalized formant balance remains
    separate from RMS. Near silence, raw ratios remain visible but their
    perceptual severity is gated. Unvoiced joins explicitly mark formant fields
    unavailable and continue through frame spectral analysis.

Pitch, phase, and period-shape metrics are `null` for unvoiced, mixed, silent,
or insufficiently periodic joins. Those joins are classified from level,
sample/derivative evidence, spectral trajectories, flux, short/medium spectra,
broadband evidence, and eligible content-dropout evidence instead.

## Labels and ranking

Possible dominant labels are:

- `OK`
- `LEVEL_STEP`
- `SAMPLE_DISCONTINUITY`
- `CONTENT_DROPOUT`
- `PHASE_MISMATCH`
- `F0_STEP`
- `PERIOD_SHAPE_MISMATCH`
- `SPECTRAL_STEP`
- `SPECTRAL_TRAJECTORY_BREAK`
- `UNVOICED_SPECTRAL_BREAK`
- `BROADBAND_IMPULSE`
- `FORMANT_FREQUENCY_BREAK`
- `FORMANT_BALANCE_BREAK`
- `FORMANT_PROMINENCE_BREAK`
- `FORMANT_TRAJECTORY_BREAK`
- `SPECTRAL_ENVELOPE_BREAK`
- `INSUFFICIENT_CONTEXT`

`severity_score` is only a sorting aid. Each component has a continuous
absolute-mismatch score and, where context exists, a separate local-novelty
score. A smooth energy gate suppresses ranking contributions near silence
without deleting raw measurements. Changes already normal inside both source
units receive a configurable discount; the undiscounted value, novelty
support, discount, and final contribution remain inspectable. Weighted
components are combined by root-sum-square. The weighting is configurable and
has not been established as a perceptual or scientific quality score.

Repair recommendations are text stored separately from measurements. The
analyzer itself never changes audio.

## Audio-repair status and join modes

Rendered PCM repair was tested and retired: even bounded period morphs could
create crackle and corrupt an entire voiced syllable. Festival output is
returned byte-for-byte apart from normal WAV encoding. The compatibility field
`Synthesis.join_repairs` remains present and empty for older project data.

Generated voices use Festival's stock UniSyn synthesis type. Both normal and
Legacy source pitchmark tracks currently use the same FRQ/WORLD-guided,
low-pass, negative-going zero-crossing epochs. Normal versus Legacy behavior
is not a new epoch detector:

- metadata-qualified Japanese-only bridge voices may use asymmetric UniSyn
  windows in normal mode;
- integrated ARPAsing voices, built-in Kal, older generated voices, and all
  Legacy renders use symmetric UniSyn windows;
- generated fallback bridges also have paired normal adaptive and pre-fix
  linear-overlap WAV/index geometry.

Festival's `segment_single` map assigns each target pitchmark to one source
frame, and `td_synthesis` Hann-windows and overlap-adds neighboring frames
across a handoff. The selector receives no diagnostic repair costs or
overrides, so contextual and manual unit choices remain authoritative.

The pure-Python renderer has its own measured raised-cosine join and exact old
linear-overlap Legacy route. See [JOIN_SYNTHESIS.md](JOIN_SYNTHESIS.md).

## GUI

After rendering, use **Generate > Inspect joins and UniSyn windows...**. The
same command is available from a recording occurrence menu.

The Overview tab contains:

- peak-cached rendered waveform and exact handoff markers;
- phone and selected-unit spans;
- retained short and momentary loudness curves;
- a severity-ranked virtual table with level, F0, phase, spectrum, content,
  voicing, and exact/estimated position status.

Selecting a row updates the Selected Join tab. It shows:

- local raw waveform, handoff interval, splice, and pitch marks;
- the requested rendered crossover as green draggable handles and shading,
  plus the actual pitchmark-snapped span in dark green;
- the independent UniSyn source-window extent as blue draggable handles and
  shading;
- first difference;
- raw, normalized, and best-aligned boundary periods;
- local per-period/per-frame RMS;
- local F0 when voiced;
- left and right spectral observations, fitted trajectories, and extrapolated
  value gap at the splice.

The measurements and JSON remain read-only. For normal Festival/WSL renders,
the green handles set a 0-100 ms per-occurrence crossover request; the default
is 40 ms. The renderer reports its phone/context cap and pitchmark-snapped
effective duration. The separate **Voice policy**, **Symmetric pitch periods**,
and **Asymmetric source periods (experimental)** controls configure the
bounded `1.00`-`1.25` source-window factor used across the sentence. Neither
control moves the measured splice or alters timing, F0, or unit selection.
Apply marks the sentence for Re-render. Legacy joins disables both editors and
retains the exact stock-Festival path.

Use **Save JSON...** to write the complete structured report. Opening the
dialog analyzes the unmodified waveform delivered to the Speech tab; changing
a control does not modify that already-rendered waveform.

From the GUI, select **Recordings** and press **Export Broadband Impulse Join
Audit...**. The image is written under
`diagnostic_images/broadband_impulse_join_audit/`. For a shareable static audit
from saved files, run:

```powershell
py -3.14 join_spectrogram.py --wav rendered.wav `
  --json join_discontinuities.json --output join_spectrogram.png
```

The image contains the waveform, STFT, issue triangles above the STFT,
handoff-span bars, an aligned rendered-phone strip, the complete rendered-phone
sequence, and an embedded legend. Red marks an unexplained issue, violet marks
a broadband event in an expected stop/affricate context, and amber marks a
measured join below the flag threshold. Stop/closure phone cells are
blue-tinted so a legitimate release can be interpreted beside the evidence.

## Tests

Run from `99_Tools/festvox`:

```powershell
py -3.14 -m unittest -v test_join_discontinuity `
  test_diphone_loudness test_join_spectrogram
```

The synthetic suite isolates same-phase continuity, phase-only mismatch,
harmonic shape, a 6 dB gain step, sample/DC steps, F0 changes, smooth and
abrupt spectral trajectories, stationary and changed unvoiced noise,
insufficient context, displaced broadband impulses, sustained frication,
steeply tilted voiced spectra, expected stop releases, audible-content
cancellation, legitimate silence/closure cases, strict JSON, marker placement,
phone-strip rendering, and corrupt-font pixel detection.

Real-bank audits live only in ignored output directories. Their results are
fixture-specific structural evidence and are not a claim of naturalness.

## Abbreviated bad-join example

The complete schema also includes local baselines, component details,
source-unit provenance, and plot data.

```json
{
  "schema_version": 6,
  "method": "continuous-pitch-synchronous-formant-content-v6",
  "joins": [{
    "splice_sample": 6400,
    "splice_time_seconds": 0.4,
    "position_source": "festival-us-map",
    "position_estimated": false,
    "voicing": "voiced",
    "left_rms": 0.1414,
    "right_rms": 0.1414,
    "level_step_db": 0.0,
    "sample_value_jump": 0.1177407,
    "sample_jump_novelty": 15.0348329,
    "left_f0_hz": 200.0,
    "right_f0_hz": 200.0,
    "f0_step_cents": 0.0,
    "zero_lag_period_correlation": 0.0,
    "best_lag_period_correlation": 0.7808688,
    "best_phase_offset_cycles": 0.25,
    "period_shape_mismatch": 0.2191312,
    "content_retention_ratio": 0.97,
    "content_attenuation_db": 0.26,
    "content_dropout_expected": false,
    "broadband_absolute_shape_score": 0.08,
    "broadband_relative_shape_score": 0.14,
    "dominant_issue": "PHASE_MISMATCH"
  }]
}
```

## Limits

- Automated tests establish deterministic structure and metric separation,
  not perceptual validity or naturalness.
- Target pitchmarks describe rendered PSOLA epochs. The handoff is an interval
  under overlap-add, so its midpoint is a diagnostic convention.
- Autocorrelation fallback is deliberately lightweight and less reliable than
  rendered target epochs.
- Rapid but legitimate source changes can still exceed a threshold. Local MAD
  normalization reduces this risk but does not eliminate it.
- Spectral and formant features are compact diagnostics, not a full auditory
  model.
- Threshold defaults require listening-led calibration across more voices.
- Zero flagged joins is an automated structural result, not proof of acoustic
  naturalness. The natively rendered WAV still requires human listening.
