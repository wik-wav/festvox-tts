# Prompt 0a: synthesis efficiency and cache management

Date: 2026-07-17

This report records the Prompt 0a profiling, runtime optimizations, cache
ownership rules, GUI controls, tests, and remaining work. It extends the
long-session lifecycle audit in `PROMPT0_LONG_TERM_PERFORMANCE.md`.

## Implemented optimizations

1. `DiphoneDB` now has a second bounded LRU for pre-sliced diphone audio.
   Repeated internal synthesis reuses a read-only source slice instead of
   copying it from a decoded recording on every render. Public slice calls
   receive defensive copies, so a caller cannot poison later synthesis.
2. The GUI requests final PCM directly from `synth_diphone.render`. The
   standalone and file-output API still returns WAV by default, but the GUI no
   longer encodes a WAV into `BytesIO` only to decode it immediately.
3. Peak normalization uses a byte-identical NumPy vector path when NumPy is
   already owned by the host process. The dependency-free Python loop remains
   the standalone fallback.
4. Pure-Python voice pitch and alternatives queries reuse the resident parsed
   `DiphoneDB` metadata. They no longer parse 11-13 MB JSON files separately.
5. CMUdict and the integrated kana map are process-owned singleton model
   caches with explicit admission limits and a clear operation.
6. Japanese frontend utterances are cached by text, explicit frontend mode,
   and Open JTalk availability. The cache owns a private canonical snapshot
   and returns a copy, so mutable provenance cannot poison a later analysis;
   manual GUI edits remain separate project state.
7. Japanese duration, pitch, and vocal-tract profile JSON uses a shared
   file-identity LRU. Resolved path, timestamps, size, filesystem identity,
   and a short content digest invalidate only that value. The digest covers
   same-size replacements whose timestamp was deliberately restored.
8. One Japanese plan now loads duration priors once and passes the parsed
   object through allocation and source-timing constraints. The former second
   parse/load attempt is gone.
9. Generate and Re-render no longer flush every Festival metadata cache.
   Windows voice metadata is checked by per-voice file fingerprint, including
   NTFS ChangeTime; WSL voices are invalidated by Reload/scan/re-registration
   because probing WSL merely to validate a cache would itself launch an
   expensive process.
10. Cache dictionaries and LRUs use locks. Diphone synthesis controls are now
    passed per render instead of mutating renderer globals. A lock-protected
    legacy path remains for an explicitly configured old renderer.
11. Decoded and sliced generated WAV data carries the referenced WAV's file
    identity and cheap OS file-change token. Replacing or rewriting a generated
    WAV invalidates only its old decoded and sliced values even when inode,
    size, mtime, and `diphone_index.json` are unchanged. Windows uses native
    NT ChangeTime, POSIX uses ctime, and an unavailable Windows token falls
    back to a content digest; hot WAV lookups do not hash full recordings.
12. Festival loaders publish recursively read-only metadata only if the
    fingerprint and global invalidation generation still match after I/O.
    This closes the load-versus-rebuild race. Recursive cache accounting now
    enforces the advertised byte limits, and invalidation state is constant
    size rather than one entry per historical voice name.

## Profile findings

The integrated Lem test voice contains 7,499 diphone index entries, 4,249
alternative pairs, and 754 source WAVs. The highest-value findings were:

- Python peak normalization consumed 16.71 ms of a 19.24 ms warm render for a
  103,101-sample result. The matched NumPy operation took 0.376 ms and produced
  identical `int16` bytes.
- A pure-Python alternatives query parsed 12,562,751 bytes in 86.14 ms, while
  a pitch query parsed 11,547,447 bytes in 49.62 ms. Both now reuse the voice
  object loaded for synthesis.
- A 96-phone/96-source-WAV stress path read 45.04 MB cold. With the slice LRU,
  an identical repeat performed no file reads and retained 1.73 MB of slices.
- Recursive in-memory sizing of the 12.56 MB index briefly raised constructor
  time from about 78 ms to 306 ms during development. This was caught during
  profiling and removed. Pure-Python voice-cache reporting uses the serialized
  index size, while Festival's separately loaded graphs are sized once on
  admission. Current pure-Python constructor time is about 77-81 ms.
- Direct PCM avoids a redundant representation and copy. The measured WAV
  decode alone was only about 0.045 ms, so normalization and metadata reuse are
  the larger CPU wins.

## Matched benchmark

Run:

```powershell
python synthesis_benchmark.py --runs 12 --repetitions 8 `
  --output rendered_audio/prompt0a/synthesis_efficiency_benchmark.json
```

The legacy side disables slice caching, uses Python normalization, emits WAV,
and decodes it. The optimized side uses the slice LRU, NumPy normalization,
direct PCM, and the final per-WAV stale-file validation. Both render the same
87-phone workload from the same voice. The script aborts if PCM differs.

| Measurement | Legacy emulation | Optimized | Result |
| --- | ---: | ---: | ---: |
| Constructor | 0.07848 s | 0.07789 s | equivalent |
| Cold render | 0.09131 s | 0.03789 s | 2.41x faster |
| Warm median | 0.08625 s | 0.03371 s | 2.56x faster |
| Warm p95 | 0.08746 s | 0.03419 s | 2.56x faster |
| Warm decode misses | 0 | 0 | no disk reads |

Both sides produced SHA-256
`6a31056abc415b3265cf6fa973211ac53b7b34b71f52e454c1a5e906cf3e2216`.
The JSON result is under ignored `rendered_audio`; it is evidence, not project
or source-bank data.

## Cache owners and limits

All limits are process-local and configurable where noted.

| Domain | Owner | Default limit | Invalidation |
| --- | --- | --- | --- |
| Audio | decoded source WAVs | 64 files / 64 MiB per resident diphone voice | LRU, referenced-WAV identity, voice fingerprint, clear |
| Audio | pre-sliced diphones | 512 entries / 32 MiB per resident diphone voice | LRU, referenced-WAV identity, voice fingerprint, clear |
| Audio | sustain samples | 64 entries / 32 MiB per backend | LRU, voice invalidation, clear |
| Audio | waveform display summaries | two fixed LOD tiers for the active waveform | new waveform, clear |
| Voice | pure-Python `DiphoneDB` | 2 resident voices | LRU, index fingerprint, reload |
| Voice | Festival metadata | 8 voices / 32 MiB exact graph budget | local fingerprint/generation or explicit reload |
| Voice | Festival alternatives | 8 voices / 32 MiB exact graph budget | local fingerprint/generation or explicit reload |
| Voice | GUI alternative lookup | 16 voice/token entries | LRU, token change, reload, clear |
| Model | duration profiles | 4 files / 8 MiB | file identity + content digest, clear |
| Model | pitch profiles | 4 files / 8 MiB | file identity + content digest, clear |
| Model | vocal-tract profiles | 4 files / 4 MiB | file identity + content digest, clear |
| Model | Japanese utterances | 128 entries / 16 MiB | mode/availability key, clear |
| Model | CMUdict | one admitted model / 128 MiB | clear or process exit |
| Model | kana mapping | one admitted model / 8 MiB | clear or process exit |

Relevant configuration keys are `diphone_voice_cache_limit`,
`diphone_wav_cache_files`, `diphone_wav_cache_mib`,
`diphone_slice_cache_entries`, `diphone_slice_cache_mib`,
`sustain_cache_entries`, `sustain_cache_mib`,
`festival_voice_cache_limit`, `festival_voice_cache_mib`, and
`voice_variant_cache_limit`.

## GUI behavior and safety

Open **Options > Application caches**. The menu shows approximate current
in-memory usage and separate actions for Audio, Voice, Model, and All.

- Audio clears decoded recordings, source slices, sustain samples, and
  waveform LOD summaries.
- Voice clears parsed voice indexes, alternatives, compatibility metadata, and
  associated audio working sets. The next render reloads them.
- Model clears pronunciation/front-end results and parsed model profiles.
- All invokes the union of those registered in-memory operations.

The clear API accepts only a fixed category, never a filesystem path. It does
not call recursive deletion and cannot remove a source UTAU bank, generated
voice, installed dictionary, current `Synthesis`, sentence preview, undo
history, project `cache/sentence_NNNN.wav`, export, configuration, or
application file. A GUI regression test creates source/project/export
sentinels and makes any filesystem deletion attempt fail the test.

## Tests added or extended

- bounded file-identity and general memory LRUs;
- same-size/restored-timestamp model invalidation;
- slice-cache hit, eviction, clear, concurrent access, defensive-copy safety,
  referenced-WAV replacement, and same-inode WAV rewrites with restored size
  and mtime;
- direct PCM versus default WAV byte identity;
- canonical Japanese frontend reuse, mutation isolation, and clear;
- exactly one duration-prior load per Japanese plan;
- unchanged voice cache preservation, changed local metadata invalidation,
  same-inode local JSON rewrites with restored size and mtime, concurrent
  invalidation publication safety, exact recursive byte accounting, and
  constant-size invalidation state;
- GUI usage labels and memory-only cache clearing;
- matched benchmark PCM identity;
- a persistent warm-cache resource soak plus an optional explicit reload mode.

Run the focused checks:

```powershell
python test_synthesis_efficiency.py -q
python test_japanese_frontend.py -q
python test_japanese_duration.py -q
python festvox_gui/test_festvox_core.py -q
$env:QT_QPA_PLATFORM='offscreen'
python festvox_gui/test_festvox_gui.py -q
```

The complete repository suites, persistent warm-cache soak, matched benchmark,
and independent adversarial validation are the release gate.

## Final validation

- Complete root discovery: 376 tests passed; one optional real-model Kokoro
  alignment test was skipped because `KOKORO_ALIGN_CHECKPOINT` was not set.
- Full GUI suite: 106 tests passed offscreen with the installed PyQt runtime.
- Focused cache suite: 7 tests passed, including restored-timestamp model
  replacement, public mutation isolation, atomic generated-WAV replacement,
  and same-inode generated-WAV rewrite. Four pitch-model tests also passed,
  including direct cached-mapping mutation rejection.
- Persistent warm-cache soak: 30 cycles, 5 warmup cycles, passed. Measured RSS
  slope was 0.0143 MiB/cycle, private-memory slope was -0.0056 MiB/cycle,
  thread and widget slopes were 0, final handle delta was 1, and there were no
  workflow failures. Two resident databases peaked at 26 decoded files and
  12,899,372 bytes of decoded/sliced audio.
- Explicit reload/invalidation soak: 12 cycles, passed with no failed checks
  or workflow failures.
- Read-only environment audit: ready; every required file and module was
  present, and WSL Festival, pyopenjtalk, and pyworld were available.
- Matched benchmark: identical PCM SHA-256, zero optimized warm decode misses,
  2.41x cold speedup, and 2.56x warm-median speedup.
- Independent adversarial revalidation: clean. All cached pitch and duration
  mapping nodes rejected mutation; same-inode WAV and both local Festival JSON
  rewrites refreshed under restored size/mtime conditions. The reviewer's
  focused suites passed 103/103 with no remaining P1 or P2 finding.

Soak and benchmark JSON live under ignored `rendered_audio/prompt0a/`; they are
reproducible evidence and are not application or project state.

## Follow-up: WSL language feature and level parity

The final-render architecture now applies one ordered acoustic capability
pipeline to English, Asaxi, and Japanese: continuous voicing realization,
vocal-tract transformation, generated-voice completed-phrase calibration,
diagnostic bit depth, then user gain. Language frontends still own their phones,
durations, phrasing, and generated F0. Japanese mora/accent state remains in the
sentence and project while its controls are hidden outside Japanese.

Generated voices use an explicit active-speech policy from their metadata.
Legacy local generated voices without that field are identified by stable
manifest/builder fields and receive the current default; built-in Kal and
unknown external voices do not. The default targets -20 dBFS active RMS with a
-6..+12 dB range and 0.98 peak ceiling, measured outside `pau` and applied once
per completed phrase. This is deliberately not per-phone or per-unit level
normalization.

A real WSL verification used `lem_v4bi_integrated`, pitch 165 Hz, Fall 18, and
matched speed. Before calibration, `this is a test.` measured -22.821257 dBFS
active RMS and `\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059\u3002`
measured -30.045794 dBFS after Japanese linguistic voicing. The shared policy
applied +2.821257 dB and +10.045794 dB respectively; both finished at
-20.000000 dBFS, with peaks 0.531539 and 0.467225. The runtime selected
`voice_lem_v4bi_integrated_ja`, not the integrated voice's English top-level
entry point.

Follow-up validation passed 377 complete repository tests with one optional
Kokoro-model skip and 189 offscreen GUI tests with no failures. The real WSL
comparison completed successfully in 5.9 seconds.

### Sentence playback pause coverage

Phrase-preview capture no longer pairs logical text phrases and acoustic spans
with `zip`. Festival may insert an unpunctuated internal pause, making the
acoustic list longer and previously dropping the unmatched spoken tail. The
new deterministic partition assigns every rendered segment exactly once,
retains every pause sample, and groups extra acoustic spans according to the
relative logical-phrase weight. Internal boundaries now have four pause parts:
the first two are included with the outgoing phrase preview and the final two
with the incoming phrase preview. Legacy three-part edits preserve their outer
guards and split only their middle gap. Sentence-level Play All remains sourced
from the complete canonical sentence waveform and cannot be truncated by
phrase metadata. Synthetic coverage includes the reported English
three-span/two-text-phrase shape followed by a one-phrase Japanese sentence.
Sentence-selection replacement also clears stale phrase keys; this closes the
separate state mismatch where the button displayed Play all but the handler
still queued only previously selected phrases.

After the four-pause ownership migration, the GUI/core suite passes 194 tests
and the root FestVox suite passes 377 tests with one optional dependency skip.

### Runtime source-audio cache and sentence switching

The generated Festival voice now defaults to a UniSyn grouped runtime database.
The builder packs the generated `wav/` and `pm/` files once into
`group/<voice>_diphone.group`, using RIFF input and 16-bit short samples. At
synthesis time Festival opens the indexed group rather than repeatedly opening
the individual source WAV files for each selected unit. The original generated
WAV and pitchmark files remain present, and the generated Scheme falls back to
the separate-file database when the group is absent or cannot be opened. This
changes storage and I/O only: contextual candidate scoring, manual overrides,
phone timing, F0, and the selected unit identities are unchanged.

`--runtime-audio-storage grouped` is the builder default.
`--runtime-audio-storage separate` remains available for diagnosis and
compatibility. A `--skip-pm` build cannot create a usable group and therefore
uses the separate layout until pitchmarks and the group are generated. Runtime
storage metadata is recorded in all applicable manifests and reports.

The direct pure-Python `synth_text()` entry point now owns a bounded two-voice
LRU of `DiphoneDB` instances. Existing decoded-WAV and slice caches therefore
survive repeated one-call synthesis requests; changing the index invalidates
the matching resident database.

Sentence selection no longer hydrates the hidden Speech editor. The Sentences
tab restores lightweight controls and keeps the immutable synthesis state, then
builds the waveform and parameter editor once when Speech becomes visible.
Voice selection reuses the current list instead of rescanning generated banks,
and waveform segments share read-only PCM views until an edit replaces a
buffer. This removes the previous double waveform construction and repeated
full-array copies on every sentence click.

The capture path now verifies both sentence ownership and synthesis-object
identity before copying waveform-derived state back into a sentence. This
prevents a blank or stale hidden editor from erasing a render committed by
Generate All before that sentence has ever been opened in Speech. A regression
generates two background sentence states, changes the Sentences selection, and
then opens Edit; timing, pitch, voicing, and vocal-tract tracks must all hydrate
from the retained synthesis metadata.

### Contextual controls during worker synthesis

The worker lock now records each control's explicit enabled state instead of
Qt's inherited effective state. This matters when synthesis begins with no row
selected in the Sentences tab: selecting a row while the worker is active must
restore its speaker, language, speaking-rate, gain, and render controls when the
request completes. Sentence selection and engine capability are reapplied after
every worker call, so the sidebar remains disabled only when it has no sentence
context and Festival-only controls remain unavailable on the pure-Python
engine.

Unavailable rows in shared dropdowns now use a muted background, darker text,
and a slim leading marker. Native top menus use one item padding geometry for
both enabled and disabled actions, with disabled actions distinguished by color
instead of a misaligned italic font.

Focused regressions cover deterministic/atomic group creation, fallback after
a packing failure, metadata propagation, one-call database reuse, hidden
sentence-editor hydration, voice-list reuse, and shared PCM ownership. A real
Festival probe created the same group hash on two builds and rendered a valid
grouped `(pau a pau)` utterance.

The complete repository suite after this follow-up passes 392 tests with one
optional dependency skip.

### Responsive rendering and editable acoustic pause edges

Festival/WSL synthesis, pure-Python synthesis, and the expensive voicing and
vocal-tract transforms now run through a `QThread` worker. The public GUI
handlers remain synchronous for project and test compatibility, but wait in a
nested Qt event loop: painting, window-manager input, status updates, and batch
cancellation stay responsive while result/error delivery and all widget/state
commits remain on the main thread. Controls capable of invalidating the active
request and duplicate render commands are locked during each call. The
application does not install a global wait cursor or disable either workspace
tab: sentence-list/timeline scrolling, tab switching, panning, zooming, and
inspection remain available. A Festival child is not killed during an
output write; it finishes normally or reaches the configured backend timeout.

The Japanese statement punctuation layer now treats zero Fall as an identity
operation. It neither inserts control points nor pulls a lowered linguistic
endpoint back to the global base pitch. Questions and expressive punctuation
remain active, and nonzero Fall still opts into the shared statement overlay.

Generated/automatic voicing still masks and restores true pause samples.
Explicit continuous curves no longer do so: UTAU source overlap can place
audible speech inside a neighboring `pau`, and the user-authored curve is now
the final authority over those samples. The Voicing editor draws its manual
line continuously through pause-colored regions to make that editability
visible.

### Non-focusing batch generation and multiline sentence editing

Generate All and Re-render All no longer load each target into the global
Speech editor. A batch call captures the target sentence's engine, language,
voice, speed, pitch, faults, dictionary, tract settings, synthesis metadata,
and edit revision, then commits the result directly to that sentence state.
`_active_sentence_index`, the selected row, the active tab, and keyboard focus
remain user-owned. Repeated edits advance the revision even when the pending
reason is unchanged; a render whose revision no longer matches is discarded
instead of overwriting newer text or parameter work.

In Sentences, `Ctrl+R` passes the selected row indices to Generate All and does
not enter the playback path. Sentence-row Enter emits Generate, while
Shift+Enter inserts a retained newline. Core phrase parsing treats retained
newlines as sentence-strength internal phrase boundaries and avoids adding a
duplicate boundary when a newline immediately follows punctuation or `[pau]`.
The row editor grows to a 220-pixel cap and scrolls beyond it.

## Remaining opportunities

These were deliberately left out because they require broader ownership or
behavior changes than Prompt 0a's low-risk pass:

- Festival text currently performs a seed synthesis to obtain relations and a
  second explicit-segment render. A reusable analysis intermediate or one
  Scheme process could remove that duplicate process, but timing, F0, phrase,
  and manual-unit golden tests must protect the change.
- Open JTalk still calls its pronunciation, full-context, and morphology APIs
  separately for a cold analysis. Whole canonical utterances are now cached;
  collapsing the cold calls needs verified API-semantic equivalence.
- Waveform display and sentence switching now use immutable views, but an edit
  still materializes the affected revision. A future revision store could
  deduplicate unchanged regions across long undo histories.
- Join/formant diagnostics can share STFT, cepstral, pitch-period, and formant
  intermediates keyed by an immutable render revision.
- Phrase dictionary file reads and broader Festival runtime indexes can be
  consolidated after their change-notification contract is explicit.

No source UTAU voicebank was written, moved, deleted, or included in a cache.
