# Join context and re-render view checkpoint

Date: 2026-07-24

## Defect

Old generated Festival phonesets described every non-vowel as a voiced stop.
The native crossover therefore capped `r`, `z`, `m`, and `w` like closures,
typically producing only 4-6 ms even when the sentence requested 40 ms.

The Speech join editor also had no dismissal gesture, and replacing audio
during Re-render reset the user's waveform viewport.

## Implementation

- `festvox_gui/festvox_core.py` attaches a canonical join class to every known
  Segment after voice hooks and before unit lookup.
- `native_unisyn/festvox_festival.cc` consumes that class first and retains
  Festival predicates as the unknown/third-party fallback.
- Sonorants use a 60 ms maximum and 70% of the phone; voiced fricatives use
  50 ms and 65%. Both require two target intervals when context permits.
- `build_festival_voice.py` and `japanese_festival.py` now write accurate
  `ctype` and `cvox` phone features for future banks.
- Join focus is dismissed by toggling the selected marker, selecting the
  waveform, hiding/disabling the overlay, replacing synthesis, or Escape.
- Re-render restores the exact X range and playhead. Generate remains
  unchanged.

None of these paths alters Segment timing, TargetCoef/F0, contextual unit
selection, or a manual per-occurrence unit choice. Legacy joins still bypasses
the project-local helper and uses stock Festival.

## Verification

Focused Python suites:

- core: 112 passed;
- GUI: 149 passed;
- unified builder: 20 passed;
- Japanese Festival compiler: 42 passed, 3 optional integration tests skipped.

The native helper rebuilt successfully in Ubuntu WSL.

A controlled Lem render used 20 ms `r/z/m/w` phones at 165 Hz. Sonorants
received a 14 ms cap, the voiced fricative a 13 ms cap, and all four rendered
12.12 ms/two-interval crossovers instead of the old stop policy.

Natural text in built-in Kal and `lem_v4bi_integrated-new` produced accepted
voiced-continuant spans of roughly 24-38 ms (four to six target intervals);
unsafe phase cancellation remained a reported bypass. The controlled
normal/Legacy comparison retained identical Segment boundaries, F0 targets,
and selected-unit mappings.

The generated Lem manifest remains at 164.81 Hz. No source UTAU files were
written or removed.
