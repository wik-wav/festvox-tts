# Prompt 0: long-term performance and resource audit

Date: 2026-07-17

This is the implementation and validation record for Optimization Prompt 0.
It covers the Windows desktop GUI, the bundled pure-Python diphone engine, and
the shared cache and project lifecycles. It does not claim that Festival/WSL
process cancellation or physical audio-device playback was exercised on the
audit machine.

## Outcome

The project-owned runtime is now self-contained under `99_Tools/festvox`.
`synth_diphone.py`, previously loaded implicitly from the neighboring Vocab
Forge project, is bundled beside the builder and GUI. Vocab Forge remains an
optional caller of FestVox output and was not edited.

The repeatable 30-cycle GUI soak passed all configured gates. It exercised
generation, playback start/stop through a no-device player, batch
cancellation, voice switching, generated-voice database loading, and backend
configuration reload. No direct child process, FestVox temporary file,
stopped playback timer, stale phrase preview, or unbounded logical cache
remained after the run.

An initially suspicious native-memory trend disappeared in a matched run with
tracemalloc disabled. Python allocation tracing itself increased runtime five
fold and produced the apparent RSS slope, so uninstrumented measurement is now
the default and `--tracemalloc` is an explicit attribution mode.

## Runtime ownership audit

Project-owned files required by the GUI and builders are under this directory:

- `synth_diphone.py`: bundled pure-Python renderer and G2P frontends.
- `build_festival_voice.py` and `utau2festvox.py`: voice builders.
- `festvox_gui/festvox_gui.py` and `festvox_gui/festvox_core.py`: desktop GUI
  and synthesis backends.
- root Japanese, diagnostics, source-filter, pitch, manifest, and profile
  modules consumed by the GUI.
- `profiles/`: shipped language and model profiles.
- `requirements.txt`: required Python dependency entry point.
- `festvox.example.json`: portable relative-path configuration template.
- `check_environment.py`: read-only installation report.

`corpus_extract.py` now imports the adjacent renderer rather than adding
`99_Tools/vocab_forge` to `sys.path`. The GUI backend searches the explicit
developer override, the FestVox root, and its own directory. It no longer
implicitly searches Vocab Forge.

External runtimes are intentionally not vendored:

- Required Python packages: NumPy, PyQt5, PyQtGraph, and `cmudict`.
- Optional Python providers: sounddevice, librosa, SciPy, pyopenjtalk,
  pyworld, and torch-based F0 providers.
- Festival mode: WSL plus Festival. The pure-Python diphone mode does not
  require either.

Run the audit from `99_Tools/festvox`:

```powershell
python -m pip install -r requirements.txt
python check_environment.py --json
```

The validation environment had NumPy, PyQt5, PyQtGraph, and pyopenjtalk, but
did not have `cmudict`. Its WSL executable was visible, while that execution
context reported no installed distribution. Those are environment gaps, not
missing project-owned files.

## Confirmed findings and fixes

### P0: deferred Qt objects accumulated in repeated window lifecycles

Evidence: the GUI test process retained every `MainWindow` when teardown used
only `deleteLater()` and `processEvents()`. A 104-test run reached about 546
MiB RSS and 1.32 GiB private bytes and took 170.8 seconds. Explicitly flushing
`QEvent.DeferredDelete` released the windows. The expanded 105-test GUI file
then completed in 15.0 seconds.

Fix: GUI teardown and the soak harness close the window, schedule deletion,
flush deferred-delete events, and process resulting events. Production
`closeEvent` now also shuts down playback and owned timers.

Regression coverage: `festvox_gui/test_festvox_gui.py` and
`resource_soak.py` track live Qt widget counts after repeated work and after
idle teardown.

### P1: decoded WAV and voice-database caches were unbounded

Evidence: each loaded generated voice could retain a database, decoded WAV
arrays, and sustain analysis indefinitely. Voice switching and config reload
therefore had growth proportional to the set of visited voices and samples.

Fix:

- The bundled renderer has a 64-file, 64 MiB decoded-WAV LRU.
- `DiphoneBackend` retains at most two generated-voice databases.
- Evicting or reloading a database also clears its decoded files and sustain
  entries.
- Festival auto-discovery and uninstall use centralized metadata and sustain
  cache invalidation.

Regression coverage: `test_utau2festvox.py` and
`festvox_gui/test_festvox_core.py` verify LRU promotion, byte/file eviction,
reload, uninstall, and sustain invalidation.

### P1: undo snapshots duplicated large PCM arrays without a bound

Evidence: structural snapshots copied waveform arrays, then undo lambdas
deep-copied those snapshots again. The command stack had no limit.

Fix: the undo stack defaults to 64 commands. Structural snapshots share the
immutable prior PCM arrays; restoring an old snapshot copies only the array
that becomes active. This preserves undo isolation without duplicating every
waveform at snapshot and closure creation.

Regression coverage: GUI tests verify the command limit, PCM sharing, and
restoration behavior.

### P1: playback timers and temporary WAV files could outlive playback

Evidence: playback completion used independent single-shot callbacks that
could not be cancelled when playback stopped or restarted. Qt multimedia
fallback files had incomplete ownership and cleanup.

Fix: `MainWindow` owns one cancellable playback-finish timer. `Player` stops
the previous backend before playing, owns every temporary WAV path, releases
Qt media, retries file removal, and exposes `shutdown()`.

Regression coverage: GUI tests verify timer cancellation and temporary-file
cleanup.

### P1: stale preview and project-cache audio survived edits

Evidence: phrase preview arrays could remain associated with text that no
longer represented that phrase. Saving a shorter project could leave old
`cache/sentence_*.wav` files in the project folder.

Fix: text edits move previews into explicit revert state and clear active
stale previews; reverting the exact edit restores them. Project save removes
only obsolete sentence-cache WAV files and preserves unrelated cache assets.

Regression coverage: GUI tests cover edit, revert, save, and stale-cache
cleanup.

### P2: transient menus and heavy dialogs retained parent ownership

Evidence: context menus constructed with the long-lived editor as parent and
diagnostic dialogs without delete-on-close could remain reachable after use.

Fix: transient context menus are local parentless objects. Pitchmark, join,
and rendered-formant diagnostic dialogs use `WA_DeleteOnClose`.

## Soak evidence

Command:

```powershell
python resource_soak.py --cycles 30 --warmup 6 --idle-seconds 5 --output rendered_audio/prompt0_resource_soak.json
```

The JSON output is generated evidence and remains ignored. The default run is
uninstrumented so the profiler does not materially change RSS or timing.
Summary:

| Measurement | Result |
| --- | ---: |
| Post-warmup measured cycles | 24 |
| Workflow failures | 0 |
| RSS slope | +0.010 MiB/cycle |
| Private-byte slope | +0.015 MiB/cycle |
| Traced Python slope | not sampled |
| OS-thread slope | 0.000/cycle |
| Handle slope | 0.000/cycle |
| Qt-widget slope | 0.000/cycle |
| Direct child growth | 0 |
| FestVox temp-file/byte growth | 0 / 0 |
| Idle CPU over five seconds | 0.31% |
| Maximum loaded voice databases | 1 |
| Maximum decoded cache | 9 files / 4.07 MiB |
| Stale phrase previews | 0 |
| Active stopped-playback timers | 0 |
| Compute GPU allocation reported for process | 0 |

Startup-to-final-idle deltas are reported separately from steady-state leak
gates. The uninstrumented run ended with +36.50 MiB RSS and +33.23 MiB private
bytes relative to the freshly constructed GUI. This is one-time lazy
initialization and allocator retention: cycle-zero RSS was about 134.02 MiB
and fell to about 118.86 MiB after teardown and idle.

A matched `--tracemalloc` run reported +0.489 MiB RSS/cycle and +0.446 MiB
private bytes/cycle despite only +0.0016 MiB/cycle of live traced Python
allocations. It also took 322 seconds instead of 34 seconds. The difference
identifies tracemalloc's unreported native bookkeeping as the measurement
artifact. Use this slower mode only to attribute a failure found by the
default run:

```powershell
python resource_soak.py --cycles 30 --warmup 6 --idle-seconds 5 --tracemalloc --output rendered_audio/prompt0_resource_soak_traced.json
```

## Test commands and results

Run from `99_Tools/festvox` with `QT_QPA_PLATFORM=offscreen` for headless GUI
testing:

```powershell
python -m unittest discover -s . -p "test_*.py" -q
python -m unittest discover -s festvox_gui -p "test_*.py" -q
python test_resource_soak.py -q
python test_environment.py -q
```

Results on 2026-07-17:

- Root discovery: 368 passed, 3 skipped, 46.1 seconds.
- GUI-directory discovery: 179 passed, 0 skipped, 14.9 seconds.
- Resource analyzer tests: 7 passed.
- Environment checker tests: 3 passed.
- Python compilation and `git diff --check`: passed.

The root discovery run also exposed and fixed a path-sensitive test import in
`test_japanese_devoicing.py`; it now imports the package-qualified GUI core.
No production Japanese or English behavior changed for that fix.

## Measurement boundaries

- GPU reporting uses `nvidia-smi` compute-process memory. It does not include
  desktop compositor or Qt Direct3D allocations.
- Playback cycles use a deterministic no-device player. Temporary WAV and
  timer ownership are tested, but a physical sound driver was not stressed.
- The available execution context did not expose a WSL distribution, so
  Festival process-tree cancellation and Festival's own resident memory need
  a host-side follow-up.
- Backend reload exercises generated-voice metadata and caches. Optional
  third-party frontends may keep documented process-global dictionaries or
  model caches after first use.
- Undo is count-bounded, not byte-budgeted. A command that intentionally owns
  a removed sentence can still retain one large waveform until it leaves the
  64-command history.
- Tracemalloc is intentionally opt-in because its native trace table changes
  both RSS slope and render timing. Its JSON report records that it was active.

## Documentation changes

- `README.md`: self-contained renderer, installation check, config template,
  and this Prompt 0 report.
- `GUIDE.md`: local renderer paths and optional Vocab Forge integration.
- `festvox_gui/README.md`: bundled engine, root requirements, environment
  checker, and soak entry point.
- `docs/README.md`: documentation map separating operating guides,
  architecture, and historical implementation reports.
- This file: Prompt 0 findings, evidence, fixes, tests, and limitations.

Historical reports remain at their existing paths to preserve links. New
development reports belong in `docs/development`.

## Safety

The Prompt 0 run used an existing generated voice and temporary test fixtures.
It did not build from, write to, move, or delete a source UTAU bank. Vocab
Forge was inspected read-only and was not modified.
