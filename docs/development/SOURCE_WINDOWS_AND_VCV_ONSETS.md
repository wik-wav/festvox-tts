# Source windows and VCV phrase-start onset

Implementation checkpoint: 2026-07-18.

## Problem

Japanese VCV phrase-start aliases such as `- V` displayed pre-vowel silence
inside the GUI vowel phone. The compiler used OTO overlap as the `pau-V`
boundary even though UTAU preutterance identifies the vowel onset.

Independently, a long OTO source region could be mapped in full to a short
Festival Segment. UniSyn then compressed an unnecessarily long source
trajectory into ordinary speech timing. The behavior was shared by Japanese
and ARPAsing-generated voices, so the correction is shared as well.

## Implementation

`source_window.py` defines the deterministic source-window policy used by
`japanese_festival.py` and `utau2festvox.py`. Both paths expose matching CLI
options through `build_festival_voice.py`:

```text
--source-window-mode adaptive|bounded|full
--source-window-ms MILLISECONDS
--zero-overlap-guard-ms MILLISECONDS
```

The default is `adaptive` with a 60 ms cap for each normal source half.
Generated metadata retains the complete OTO span. A paired Japanese CVVC/VCV
unit widens only as far as necessary when a 60 ms half would clip its declared
phone-center anchor. Hidden `__wl`, `__wr`, and `__wb` index rows expose a
complete left side, right side, or both when those geometries differ from the
primary row. Festival chooses one of those rows only after it has selected the
contextual or manually overridden recording.

The full-side activation threshold is the complete source-half duration times
two. One diphone side normally receives approximately half of a target phone,
so this avoids exposing a long source trajectory for a modest timing edit.

`bounded` emits only the capped primary geometry. `full` makes the primary
geometry equal to the complete OTO span and therefore reproduces legacy source
windowing. The modes affect source geometry only: they do not normalize gain,
replace a take, alter phone duration, add a fade, or post-process rendered PCM.

Festival's EST index offers discrete rows, not a continuously variable source
slice. Adaptive mode therefore switches from bounded to complete geometry at
the threshold. A continuously expanding intermediate slice remains a possible
future synthesis-engine feature, not behavior claimed here.

## Zero-overlap OTO rows

Some CV banks deliberately declare `overlap=0`. The builder preserves the raw
OTO offset by default. A nonzero `--zero-overlap-guard-ms` remains available as
an explicit diagnostic experiment, capped at one quarter of preutterance and
leaving two milliseconds before the phone boundary. It is not a production
default: listening validation found that shortening a generated consonant-side
clip could make the following handoff worse despite improved numerical join
scores. Positive overlap remains authoritative and contextual candidates are
unchanged.

## VCV boundary rule

For a one-vowel phrase-start VCV candidate:

```text
vowel onset = OTO offset + OTO preutterance
compiled edge = pau-V
```

The compiler no longer creates a synthetic medial `V-V` edge from `- V`.
Ordinary CV or VCV candidates supply sustained and medial vowels. Consonant-
vowel VCV rows already use preutterance as their consonant-vowel boundary and
retain their coherent two-edge source split.

## Files changed

- `source_window.py`: shared source-window model and stable variant names.
- `japanese_festival.py`: VCV onset correction, Japanese window metadata,
  hidden index rows, and generated Scheme selection.
- `utau2festvox.py`: matching ARPAsing window generation and metadata.
- `build_festival_voice.py`: unified and compatibility CLI options plus
  generated Scheme selection.
- `test_source_window.py`: policy unit tests.
- `test_japanese_festival.py`: VCV onset and reversible-mode tests.
- `test_utau2festvox.py`: ARPAsing bounded/full geometry tests.
- Existing GUI and state documentation describing generated-voice behavior.
- Current builder, GUI, state, and documentation-index files.

## Validation

Focused policy, Japanese compiler, Japanese assembly, ARPAsing converter, and
unified builder tests pass. The top-level and GUI suites retain their existing
separate import layouts, so repository validation uses:

```powershell
$TopLevel = Get-ChildItem -File -Filter "test_*.py" |
  Where-Object Name -ne "test_japanese_devoicing.py" |
  ForEach-Object BaseName
$env:PYTHONPATH = "$PWD;$PWD\festvox_gui"
py -3.14 -m unittest $TopLevel

$env:PYTHONPATH = "$PWD"
py -3.14 -m unittest test_japanese_devoicing

$env:PYTHONPATH = "$PWD\festvox_gui;$PWD"
py -3.14 -m unittest discover -s .\festvox_gui -p "test_*.py"
```

Result: 382 top-level tests, 13 Japanese devoicing tests, and 210 GUI/core
tests passed (605 total). One optional integration test in the top-level suite
skipped cleanly because its local dependency was unavailable.

A read-only real-bank check examined 18 phrase-start vowel candidates in a VCV
bank. Every compiled phone boundary equaled `offset + preutterance`. For the
audited `- a` row, offset 807.15 ms plus preutterance 160 ms produced a
967.15 ms boundary. Source OTO bytes were unchanged.

A second read-only CV-bank check examined 148 OTO rows, including 74 with zero
overlap. An inferred 12 ms source-cut experiment improved several numerical
join scores but failed listening validation and visibly perturbed generated
handoffs, so it was withdrawn as the default. The default again reproduces raw
OTO geometry. Stop bursts remain visible because they are legitimate broadband
consonant events. The source `oto.ini` retained SHA-256
`9836404bfdcac094acefe7c87a7ae8923b5def8607a87b38ab0572549246d0e1`.
