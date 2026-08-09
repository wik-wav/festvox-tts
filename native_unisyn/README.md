# Native UniSyn Crossover Runtime

`festvox_festival.cc` is a small Festival 2.5 batch wrapper used only when
the GUI requests the measured multi-epoch crossover. It does not replace the
installed `festival` executable.

The crossover length is expressed in milliseconds. Voiced edges are snapped
inward to existing target pitchmarks, so the displayed period count changes
with F0 while the requested time span remains stable. The runtime:

- keeps Segment timing, TargetCoef times, F0, and selected Unit items intact;
- limits each crossover to the shared phone;
- applies explicit canonical phone-class duration caps, with Festival
  phoneset predicates as the third-party fallback;
- bypasses silence and unsafe phase-cancelling joins;
- blends Kal LPC envelopes through stable reflection coefficients;
- reports every applied or bypassed join as `GUIXOVER` diagnostics.

`GUIXOVER` keeps the requested left/right milliseconds, the phone-context cap,
the pitchmark-snapped effective duration, and the bypass reason separate. The
GUI uses these fields for the draggable Speech-waveform join handles and the
`requested | rendered | cap` readout. A per-occurrence edit changes only the
corresponding target handoff.

The GUI annotates Segment items after voice hooks and before `us_get_diphones`.
That runtime annotation fixes old generated banks whose phonesets described
every consonant as a stop. Future generated phonesets also emit correct
features. Vowels use an 80 ms/60% cap; nasals, liquids, and glides use
60 ms/70%; voiced fricatives use 50 ms/65%. The voiced-continuant policies
require two target intervals when available, while stop and affricate spans
remain closure-sensitive.

The helper also supports `--server`. `FestivalWSLBackend` uses that protocol
for normal renders so Festival, the voice Scheme, and the grouped UniSyn
database can remain warm between jobs. Jobs are serialized and the process is
recycled after 32 renders by default. A native-binary change, voice metadata
invalidation, or application shutdown also closes the worker.
Generated voices registered by WSL path are fingerprinted through their
`\\wsl.localhost` metadata view, so an in-place rebuild restarts the warm
interpreter before the next render without spawning a metadata probe per job.
If the distro setting is blank and the registry has no default name, one
cached `wsl --list --quiet` probe resolves it.
Configured native binaries are fingerprinted at the configured path.

A `0 ms` sentence crossover with no positive occurrence override deliberately
uses one-shot stock Festival. This provides a no-crossover control without
claiming to be **Legacy joins**, which additionally restores old source-window
policy and ignores saved overrides.

Build in PowerShell:

```powershell
py -3.14 .\native_unisyn\build_wsl_runtime.py --distro Ubuntu
```

WSL build dependencies:

```bash
sudo apt install g++ festival-dev libestools-dev libsystemd-dev libncurses-dev
```

The generated executable is written to
`native_unisyn/build/festvox-festival`. The build directory is local output
and should not be committed.

`Fault Mode > Legacy joins` never invokes this runtime. It continues to run
the installed Festival binary and stock `us_generate_wave`.

## Verified behavior

The focused July 23 validation established:

- exact repeatability for stock Kal under Legacy joins;
- unchanged Segment timing, F0 targets, and selected units for Kal and
  `lem_v4bi_integrated`;
- byte-identical output across 70 normal Kal jobs while the worker recycled
  twice;
- about 31 ms median warm rendering for the short Kal fixture, compared with
  about 202 ms for stock one-shot Festival;
- 2.59 s for a 657-phone, 56-second native render, effectively equal to the
  2.60 s stock path.

These are structural and performance checks, not a claim that acoustic
naturalness has passed human listening.

The July 24 class-policy check additionally established that natural-text
`r/z/m/w/ng` joins in Kal and `lem_v4bi_integrated-new` use voiced-continuant
policies rather than the 4-6 ms stop cap. The controlled normal/Legacy A/B
kept Segment boundaries, F0 targets, and selected units equal.
