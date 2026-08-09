# Song Mode: Future Rebuild Brief

Status: design handoff only. The previous Song implementation was removed in
July 2026 so a new version can be designed without compatibility constraints.

## Product goal

Song mode should turn an already generated FestVox/Festival speech performance
into an editable vocal performance. It must keep the selected speaker's real
recordings, phoneme choices, consonant character, and intelligibility while
regenerating note pitch and detailed F0 through Festival/UniSyn PSOLA. It must
not be a post-render pitch shifter.

Speech and Song should share one time axis, one playhead, and the same phoneme,
recording-take, timing, project, and speaker data. Editing either view must not
silently discard work in the other.

## MIDI import contract

- One nonempty MIDI track creates one sentence.
- A track remains one long phrase. MIDI rests do not split it into phrases.
- If a track contains overlapping notes, each monophonic overlap lane creates
  a separate sentence.
- Before creating overlap-lane sentences, show a clear warning and offer a
  **Fix overlaps** action. The result must remain deterministic and reversible.
- Preserve the source track name, track index, overlap-lane index, tempo map,
  note starts, durations, velocities, and lyrics when present.
- Multiple tracks and overlap lanes should be installable in one operation
  without overwriting existing project sentences unexpectedly.

## Editing experience

- Use an inline piano roll aligned with the rendered waveform and phonemes.
- Support pointer, draw, erase, split, resize, quantize, free timing, and
  horizontal and vertical zoom.
- Notes need readable pitch names, a practical C1-C7 working range, and clear
  drag previews and drop/resize feedback.
- Support signed portamento around note boundaries and melismas over sustained
  vowels.
- Keep consonants independently editable. Note alignment should primarily
  retime vowels and adjacent pauses instead of forcing every phone to a note.
- Show a waveform-aligned vocal-deviation curve in cents. Fast mouse movement
  must interpolate a continuous stroke, and Shift-drag should lock the starting
  value while painting horizontally.
- Playback, selection, undo/redo, keyboard shortcuts, and performance
  virtualization should follow the Speech and Sentences conventions.

## Synthesis and safety

- Generate note targets and vocal deviation as an F0 contour, then pass that
  contour through the same bounded Festival/UniSyn PSOLA path as Speech pitch
  overrides.
- Keep all requested and interpolated targets inside the proven Festival range
  of 50-500 Hz unless a future voice-specific capability check establishes a
  narrower range.
- Preserve the generated speech F0 as a visible reference beneath Song edits.
- Keep phrase-edge pause handling compatible with the four-pause speech model:
  outgoing guard and half-gap, then incoming half-gap and guard.
- Re-rendering must preserve user-selected UTAU recording alternatives and
  remap them only when their phone transition truly changes.
- Long tracks need viewport-only waveform, grid, note, label, and curve
  rendering with level-of-detail simplification when zoomed out.

## Rebuild boundaries

Build the domain model and headless tests before the piano-roll UI. Keep MIDI
parsing, overlap-lane planning, note normalization, lyric alignment, and F0
planning independent of Qt. Do not restore the removed implementation by
copying its classes back into `festvox_gui.py`; start with a small, explicit
state contract and add the interface after round-trip and synthesis tests pass.
