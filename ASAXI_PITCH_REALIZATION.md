# Asaxi Pitch Realization

## Purpose

Asaxi lexical and morphological analysis still produces categorical `H` and
`L` values per mora. Those labels are phonological instructions, not two fixed
frequencies. `asaxi_pitch.py` is the separate phonetic realization layer that
turns the complete, finally timed sentence into a continuous speaker-relative
log-F0 plan.

This is version 1 of the acoustic realization. Its conservative parameters
live in `profiles/asaxi_pitch_model_v1.json`; they have not yet been fitted to
a recorded Asaxi prosody corpus.

## Runtime Order

1. `asaxi_prosody.py` resolves dictionary, morphology, phrase expressions,
   utterance H/L, speech act, and boundary tone.
2. `asaxi_duration.py` assigns phone durations.
3. Phrase assembly creates the final sentence timeline, including all pause
   regions and any timing fault.
4. `asaxi_prosody.realize_pitch_for_plans()` aligns sentence-level mora indexes
   to that final timeline.
5. `asaxi_pitch.realize_pitch()` creates one sentence-level log-F0 contour.
6. Festival/UniSyn receives those exact targets. The shared pitch editor and
   other acoustic parameters continue through the normal multilingual path.

Generate and Re-render both call step 4 after final timing is known. Re-render
uses the existing editor durations and does not request new phone lengths.

## Model

The model works in semitones relative to the selected speaker pitch:

- H and L select relative lexical goals inside the current phrase shape.
- Phrase declination is centred around zero, so it changes early-versus-late
  shape without translating the sentence.
- Later-phrase declination, contrast, and boundary components are also
  mean-centred within each phrase.
- The realized F0 value and slope carry across phrase boundaries.
- Boundary type controls a bounded partial reset of that carried state.
- `L%`, `H-`, and `H%` occupy a final region rather than one endpoint.
- `LH%` contains a low goal, an earlier high goal, and a held high region so a
  short final mora is not given an impossible last-sample jump.
- A critically damped, rate-limited target tracker creates
  duration-dependent target approximation. Short morae naturally undershoot
  more than long morae.
- The latent state continues across pauses, but no render targets are emitted
  inside pause spans.

No cumulative elapsed-time or phrase-index frequency drift is allowed. The
profile loader rejects legacy-style nonzero drift keys. Repeated phrases may
differ in contour shape and boundary state, but there is no automatic global
frequency slide.

Existing symbolic statement/question/directive and H/L rules remain in
`asaxi_prosody.py`. This version does not invent focus, prominence,
microprosody, or a new downstep rule beyond those established symbolic rules.

## Manual Authority

Mora tone and cents edits are localized with a raised-cosine envelope contained
inside the selected mora. The edit reaches full strength across the mora
centre, including the linguistic 58-percent target anchor, and is zero at both
edges. A +1200-cent edit therefore doubles that mora's target without shifting
unrelated targets.

The detailed continuous Pitch curve remains the final manual authority after
the automatic Asaxi realization.

## Diagnostic Trace

`Synthesis.asaxi_prosody.prosody_trace` is deterministic JSON-compatible data
with:

- model ID and version;
- speaker centre, Fall setting, and target-tracker parameters;
- phrase carry-in F0/slope, boundary reset strength, and post-reset state;
- each mora's lexical, utterance, and selected tone;
- phrase-shape and manual semitone contributions;
- temporally extended boundary events;
- every desired and realized target in semitones and Hz.

The trace contains no timestamps of execution or absolute private paths.
`cumulative_frequency_drift` is explicitly recorded as `disabled`. The
`voicing_status` field is also explicit: this layer creates a latent F0 curve,
while Festival applies it only to voiced frames. It does not pretend to have
performed a separate voicing classification.

The trace is retained in project synthesis metadata and can be inspected in
the project JSON. Plot generation remains on demand; ordinary synthesis does
not create diagnostic images.

## Performance

The planner emits targets at the profile's 25 ms interval plus exact linguistic
and boundary anchors. Goal interpolation uses binary lookup and the active mora
is advanced monotonically, so realization scales with emitted points rather
than rescanning all morae for every point.

The model profile uses the bounded shared file-identity cache and participates
in **Options > Cache > Clear model caches**.

## Verification

Focused tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s . -p "test_asaxi_pitch.py"
.\.venv\Scripts\python.exe -m unittest discover -s . -p "test_asaxi_prosody.py"
.\.venv\Scripts\python.exe -m unittest discover -s .\festvox_gui -p "test_festvox_core.py"
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m unittest discover -s .\festvox_gui -p "test_festvox_gui.py"
```

The acceptance tests cover profile safety, deterministic serialization,
mean-centred later-phrase variation, boundary-state carry, timed `LH%`, pause
gaps, duration-dependent undershoot, and local manual authority.

## Known Limits

- Acoustic naturalness and the numeric profile still require human listening
  review and fitting against recorded Asaxi speech.
- Phrase breaking remains dependent on the current Asaxi text/punctuation
  planner.
- Prominence, discourse focus, microprosody, and source-residual retention are
  not part of this version.
- The trace records the targets sent to synthesis; it is not a measurement of
  the F0 that a particular unit waveform ultimately produced.
