# FestVox Speech GUI Development Guide

## Product priorities

Performance and usability are high-priority acceptance criteria, not optional
polish. A change is incomplete when it is technically correct but leaves the
GUI sluggish, blocks normal editing, loses the user's place, obscures state, or
requires avoidable regeneration.

- Keep synthesis and expensive analysis off the GUI thread. The Speech and
  Sentences tabs must remain responsive during individual and batch renders.
- Exercise long utterances and representative zoom levels. Measure redraw,
  sentence-switching, render, and cache behavior instead of judging only short
  fixtures.
- Avoid repeated full-WAV loads, unnecessary array copies, hidden-widget
  redraws, and rehydrating editors that are not visible.
- Preserve user context across background work: selection, scroll/zoom,
  parameter edits, manual unit choices, and generated sentence metadata.
- Prefer controls and feedback that make the current state and pending action
  immediately understandable. Disabled controls must look disabled, and
  available actions must remain reachable in their relevant context.
- Add focused performance or interaction regressions for diagnosed problems,
  then run the complete test suite. Do not trade synthesis correctness,
  language-specific prosody, or source-bank safety for speed.
