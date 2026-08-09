# Japanese UTAU Implementation Report

This file is the running, path-neutral engineering record. Commit hashes for a
phase are added after that phase's commit exists; the final consolidated report
is authoritative for the last commit hash.

## Shared ARPAsing language profile update

ARPAsing configurations may explicitly enable English, Asaxi, and Japanese
frontends over one generated UniSyn database. `arpasing_profile.py` parses the
bundled `profiles/en-jap-mapping.yaml` without PyYAML, preserves conflicting or
invalid rows as diagnostics, and records a path-neutral source checksum.
Existing ASCII ARPAsing aliases retain their established converter meaning.
Japanese graphemes, bank-defined moraic-nasal allophones, and bounded relative
timing weights are routed through the profile only when Japanese is enabled.
Japanese-only CV/VCV/CVVC compilation remains isolated.

The GUI Pitch accent page is sentence-scoped: it is enabled for Japanese
sentences using any generated voice that explicitly advertises Japanese,
including shared ARPAsing voices, and disabled for non-Japanese sentences.
Generated metadata is re-read after a rebuilt voice selection refresh.

Verification for this update passed 167 builder, frontend, synthesis, release,
and safety tests plus 120 GUI/backend tests. The bundled mapping is byte-for-byte
identical to the supplied file at SHA-256
`6356B50F3C25417797130F94F47E9D52C3B4A96B7DC5FFDB511A18358C517A99`.
The protected representative Japanese source remained 467 files with its
established relative-path-plus-bytes SHA-256
`B6A874A371397E0825D05ED5BEE87085BEFD5FE174F18D435B591D5059458F80`.

## Checkpoints

- Phase 0/current analyzer checkpoint:
  `6a39ffe09c7ba967b5b208c204d5841f11ad8fca`
- Phase 1 text frontends:
  `b776c6f2073dd3d3d30c88614a4c499aceccf507`
- Phase 2 candidate compiler:
  `2834f2c2cf11659cf95cc9fdab0d610cdf22d3db`
- Phase 3 Festival synthesis:
  `ec5a1f154758d79e8c44addff56fb32742afcc4e`
- Phase 4 Japanese editing workflow:
  `d12051b21d719ccba3ee6822a2296b9c822cf848`
- Phase 5 Japanese synthesis refinements:
  `706086e`
- Remediation Stage 1, configuration-scoped metadata:
  `05066f6`
- Remediation Stage 2, CV/VCV/CVVC assembly repair:
  `3831a19d24d35d0219dd017e374dfe92e957d472`
- Remediation Stage 3, unified builder and paths:
  `d426d01`
- Remediation Stage 4, shared source-speaker F0 analysis:
  current checkpoint (hash recorded by the following checkpoint)
- No Git remote is configured.

## Remediation Audit (2026-07-14)

The pre-remediation working tree was clean and the complete 223-test baseline
passed. The correction work is intentionally ordered: metadata and scope,
assembly correctness, unified building/paths, shared FRQ pitch analysis, GUI
cleanup, then Japanese duration and prosody. Later stages do not proceed until
the assembly gate is clean.

Human listening found the previous automated quality claim insufficient:

- accent contrast, moraic nasal, long vowels, geminates, palatalized morae,
  phrase boundaries, statements, questions, vowels, and VCV transitions were
  recognizable;
- VCV was the cleanest route;
- CVVC failed because expected VC contributions were missing and joins sounded
  disjointed;
- ordinary-CV and palatalized examples exposed a doubled first consonant;
- geminate, long-vowel, and phrase examples inherited the CVVC join defect.

These are hard Stage 2 acceptance failures, not cosmetic warnings. The build
will not advance beyond assembly while missing VC transitions, doubled onset
consonants, or large gaps remain.

### Metadata and scope checkpoint

`voice_manifest.py` now distinguishes an immutable
`SourceRecordingBundle` from one explicit `VoiceConfiguration`. CV, VCV, and
CVVC configurations have separate IDs, alias namespaces, canonical-phone
namespaces, and candidate IDs even when they share recordings. The Japanese
Festival compiler requires `--bank-type cv|vcv|cvvc`; `auto` and `mixed` are
analysis modes only. Generated runtime metadata declares source/configuration
IDs, primary and supported languages, alias system, scoped namespaces, and a
language-to-entry-point map.

The Festival GUI reads this compatibility metadata, chooses a current voice's
primary language, disables unsupported languages, constrains direct phones to
the declared phoneset, and labels old path-backed metadata as legacy or
unknown. Legacy voices remain readable and receive a rebuild recommendation.

Focused verification commands used the same Python 3.14 environment as the
application. The new manifest suite passed 6/6, candidate tests passed 21/21,
Festival tests passed 16/16 in the full environment, core tests passed 51/51,
and the real Qt suite passed 56/56. The complete post-checkpoint gate passed
230/230 with no failures or skips. The source inventory includes the recording
bytes, so changing a WAV changes the source bundle, configuration, and scoped
candidate identities.

Read-only before/after source tripwires were identical for the CVVC, VCV, CV,
and requested Phascogale CVVC subbanks respectively:

- `FBC44FF56C774DD7200D30B033F16F0C383CBA5174293E63FB524A7C0BA64B9A`
- `7E41BA254C57AB3330F3BEDC7F91E51184B43C64B7E656920B628FCED39229D1`
- `2CFA8881B24E9AEC8E0BD7677671AAD64CE6E8A88B1A63D588BCF03F5F08724A`
- `B6A874A371397E0825D05ED5BEE87085BEFD5FE174F18D435B591D5059458F80`

The CV count includes a pre-existing hidden desktop metadata file; no test or
builder created it. No source bank was written, staged, moved, or deleted.

### Assembly correctness checkpoint

The listening failures were traced to assembly after candidate compilation,
not to lost VC aliases. Phrase-start and VCV pairs overlapped from consonant
onset through preutterance, replaying one consonant. Their halves were also
selected independently, allowing an initial `pau-k` from one vowel recording
and `k-a` from another.

The compiler now splits phrase-start, VCV, ordinary-CV, VC, and release
geometry at bounded phone-center anchors. Paired source intervals meet exactly;
an incoming CV starts at consonant center and cannot replay onset material from
the outgoing VC. Phrase-start and VCV left halves retain following-vowel
context, so both edges select one candidate. Explicit CVVC treats CV and VC as
co-primary while secondary VCV remains visible fallback evidence.

A pure CV bank no longer reaches Festival's hidden `pau-pau` default for a
missing transition. The compiler creates a deterministic generated-output WAV
from a stable left-phone tail and the next CV onset, with an 8 ms crossfade.
The `generated_cv_bridge` row records both source candidates, aliases, OTO rows,
landmarks, and slices, and every use emits a visible fallback diagnostic.

New files and APIs:

- `japanese_assembly.py`: Festival-equivalent automatic selection,
  `JapaneseSourceContributionPlan`, exact source traces, and structural
  validators;
- `japanese_assembly_listening.py`: ignored 12-item human comparison corpus
  with one contribution JSON per WAV;
- `test_japanese_assembly.py`: isolated CV/VCV/CVVC fixture banks, the required
  11-utterance matrix, overlap regression, hidden-gap regression, source trace,
  and byte-deterministic bridge tests;
- `JAPANESE_ASSEMBLY_REMEDIATION.md`: corrected geometry, fallback, API, and
  verification contract.

Real read-only structural builds:

- representative CVVC: 2,033 units, 20 generated bridges, zero bridge-source
  failures;
- representative VCV: 3,183 units, 82 generated bridges, zero bridge-source
  failures;
- representative CV: 548 units, 337 generated bridges, zero bridge-source
  failures.

Each real configuration rendered 12/12 comparison WAVs through WSL Festival.
Across all three corpora there were zero skipped diphones, hidden-silence
fallbacks, structural errors, or automated `poor` joins. Aggregate fallback
uses were 7 for CVVC, 15 for VCV, and 41 for CV; these counts are preserved in
the manifests rather than presented as native recorded transitions. The worst
automated join-risk scores were 35.103, 27.366, and 32.786 respectively.
Acoustic naturalness and the audible improvement over the failed human baseline
still require a new human listening pass.

Focused verification passed 8/8 assembly tests and 16/16 Festival tests. The
Festival integration test ran with no skip and verified rendering, explicit
timing/F0, and manual unit overrides. The final gate passed 238/238: 131
non-GUI tests and 107 offscreen Qt tests, with no failures or skips.

Two fresh real CVVC builds made in different destination directories compared
byte for byte. Generated Scheme now resolves the voice root from the load path
installed by the backend instead of embedding the destination path; this keeps
the complete generated tree deterministic and relocatable. A unit test compares
all generated files, not only metadata and bridge WAVs. The checkpoint commit
hash is recorded in the next running-report update because a commit cannot
contain its own hash.

The same relative-path-plus-bytes source hashes remained unchanged after real
builds, pitchmarking, and all three listening corpora:

- `FBC44FF56C774DD7200D30B033F16F0C383CBA5174293E63FB524A7C0BA64B9A`
- `7E41BA254C57AB3330F3BEDC7F91E51184B43C64B7E656920B628FCED39229D1`
- `2CFA8881B24E9AEC8E0BD7677671AAD64CE6E8A88B1A63D588BCF03F5F08724A`
- `B6A874A371397E0825D05ED5BEE87085BEFD5FE174F18D435B591D5059458F80`

### Assembly follow-up: bank semantics and runtime routing

Human review then identified two additional source-bank conventions. Medial
CVs must not use `- V`/`- CV` phrase-start aliases because those recordings
contain intentional leading silence. Separately, some CV banks define `* V`
rows whose audible vowel begins at OTO offset and whose preutterance/overlap
landmarks are intended for vowel blending. The analyzer and compiler now keep
these concepts separate: phrase starts are hard-limited to pause edges, while
`vowel_blend` is an incoming generated-bridge fallback with a bounded OTO-based
crossfade. Exact VCV or VC transitions remain preferred.

Moraic-nasal allophone names are now explicit `JapaneseBankProfile` data.
Each configured group declares exact mora aliases, exact VC context aliases,
following canonical phones, and an optional default. Numeric aliases declared
there are protected from numbered-take stripping. Both `V-N` and `N-C` edges
select the same configured group; unknown allophone-like aliases remain
traceable with diagnostics, and manual occurrence overrides remain final.

The generated Japanese Scheme already contained the correct role and
allophone selector, but the GUI backend was pre-filling every edge with the
generic English contextual selector. Current Japanese runtime metadata now
causes automatic choices to remain inside the generated UniSyn hook. Only
explicit user choices are forwarded as overrides. English/ARPAsing selection
is unchanged.

The fresh real CVVC corpus selected configured labial (`a んm`, `m b`), velar
(`a んng`, `ng k`), coronal (`a ん`, `n t`), and bank-specific uvular
(`e んn`, `nn s`) pairs. Exact `e い` and the internal transitions for
`関係ないです` no longer use phrase-start or hidden-silence sources. Seventeen
kana fixtures rendered through Festival with zero failures, skipped diphones,
hidden silence, or structural errors; the full 18-item corpus also includes an
Open JTalk kanji fixture. Acoustic naturalness remains unverified pending a new
human pass.

The complete gate passed 259/259 tests with `pyopenjtalk 0.4.1`, its local
dictionary, and offscreen Qt enabled. The single sandbox-gated WSL integration
test then passed separately against Festival. Two fresh profile-driven real
builds each produced 234 files and compared byte for byte. The audited
467-file source subbank remained byte-identical at
`B6A874A371397E0825D05ED5BEE87085BEFD5FE174F18D435B591D5059458F80`.

### Unified builder and path checkpoint

`build_festival_voice.py` is now the shared Windows front door for Japanese,
English, and Asaxi generated voices. It requires one language, one explicit
bank type, a read-only sample root, an exact OTO scope, one generated output
folder, and one voice name. English and Asaxi continue through the established
ARPAsing converter; Japanese continues through the isolated profile,
candidate, assembly, and Festival compiler. The old `--db`/`--utau` interface
remains available for existing workflows.

The new `voice_paths.py` owns canonical Windows paths, WSL boundary conversion,
output nesting/overwrite guards, registration schema version 1, and migration.
Current GUI registrations store a canonical Windows path and a derived runtime
path beneath one configured generated-voice root. Historical `/mnt/...`
registrations migrate to Windows paths. Historical `/home/...` registrations
remain read-only WSL-only entries; no migration moves or deletes them. The
normal WSL-path add action is no longer exposed.

The generated manifest uses one source-recording bundle ID and a separate
voice-configuration ID. Reinterpreting identical recordings through different
language/alias policies therefore keeps source identity while receiving a
different configuration, namespace, and entry point. Current English and
Asaxi outputs define one `voice_<name>`; Japanese defines one
`voice_<name>_ja`. Generated Scheme resolves its root from Festival's load path
and generated provenance is source-relative. A destination-independence test
compares every generated byte.

### Shared speaker-F0 checkpoint

`speaker_pitch.py` is now the only active source-speaker analysis used by the
unified Japanese, English, and Asaxi routes. It parses the complete official
UTAU `FREQ0003` layout recursively, rejects nonfinite/unvoiced frames, retains
malformed-file diagnostics, and requires three usable files. The source base
pitch remains the median of per-file header averages to preserve established
English behavior; voiced frames provide p10/p90 and sample count. If FRQ data
is insufficient, a deterministic spread of source WAV windows is analyzed by
autocorrelation. A fixed fallback is explicit and diagnostic rather than
silent.

The source-recording bundle and top-level runtime metadata carry the same
path-neutral analysis object. User `--f0` and pitchmark-bound flags remain
effective overrides while leaving the measured source object intact. Japanese
planning resolves its default base pitch from this metadata and constrains
structural movement to a speaker-relative range; callers that explicitly pass
a base pitch retain authority.

The audited real subbank yielded 242 valid FRQ files, 128,634 voiced frames, a
165.625498 Hz median, 160.363636/168.320611 Hz p10/p90, and 102.7-281.6 Hz EST
tracking bounds. Synthetic shared-root regression builds prove byte-identical
pitch statistics for English and Japanese configurations.

Explicit multi-OTO scope now propagates through `japanese_utau.analyze_bank`,
`infer_bank_profile`, and `compile_candidate_graph`. Sibling English,
Japanese, or ignored OTO files are not rediscovered. Omitting the optional
scope preserves the earlier recursive analyzer behavior.

The common smoke-test contract uses fresh temporary files so stale WAVs cannot
produce a false pass. English/Asaxi render text and explicit phones. Japanese
analyzes language-specific default kana, creates the canonical explicit
duration/F0/unit plan, and renders that plan through Festival. Temporary Scheme
and failed artifacts are removed; successful test WAV/segment output remains
inside the generated folder. Unique pitchmark handoff filenames also remove a
concurrent/resumed-build race.

Real read-only validation through the shared front door produced:

- English ARPAsing: 7,497 diphones from 752 WAVs and 8,075 OTO entries, with
  3,248 alternatives; the single known `-aw11` token remained visible.
- Japanese CVVC: 1,280 source candidates, 1,205 selectable candidates, 2,033
  compiled units, 20 labeled generated bridges, and zero bridge-source
  failures.
- Real Festival smoke renders returned nonempty English and Japanese WAVs and
  coherent segment relations through their isolated entry points.

Two real English builds in different destinations produced identical 1,515
file, 741,366,586-byte trees with aggregate SHA-256
`E77D3B6BBF1400D4EDE410610719A8971FF51342161A9AEE1BF35144B0B82115`.
Two real Japanese builds produced identical 244-file, 114,761,302-byte trees
with aggregate SHA-256
`C515C2A4156B73F6A49E1277598B3CC3532B5768507CFD7002B4EC4FB3F1211A`.

The final gate passed 246/246 tests with no failures or skips: 138 analyzer,
frontend, candidate, assembly, Festival, refinement, converter, manifest, and
unified-builder tests plus 108 core/offscreen Qt tests. The Qt settings-dialog
visual check rendered at 560 by 400 pixels with controls inside bounds. The
offscreen Windows font plugin omitted glyphs, so labels, path values,
accessibility names, wrapping, and dimensions were verified by widget tests.

Read-only before/after hashes were unchanged:

- representative CVVC: 467 files,
  `FBC44FF56C774DD7200D30B033F16F0C383CBA5174293E63FB524A7C0BA64B9A`;
- representative VCV: 379 files,
  `7E41BA254C57AB3330F3BEDC7F91E51184B43C64B7E656920B628FCED39229D1`;
- representative CV: 230 files,
  `2CFA8881B24E9AEC8E0BD7677671AAD64CE6E8A88B1A63D588BCF03F5F08724A`;
- requested Japanese CVVC: 467 files,
  `B6A874A371397E0825D05ED5BEE87085BEFD5FE174F18D435B591D5059458F80`;
- complete English source bank: 17,395 files,
  `B8E6CDC7EBD8A929D598E531AF02EC8C5F513225551516E2B78E61CE10788DDD`.

No source bank, generated voice, test WAV, log, screenshot, cache, or private
absolute path is staged. The generated-root dialog, all three command forms,
metadata fields, migration behavior, and safety contract are documented in
`UNIFIED_VOICE_BUILDER.md`, `GUIDE.md`, the tool and GUI READMEs, this report,
and the implementation state document.

Checkpoint message reserved for the completed gate:
`refactor: unify Festival voice building and paths`.

## Phase 2 Gate Record

Pre-phase tree: clean. Expected Phase 1 checkpoint: present. Reference source
OTO SHA-256 values matched their Phase 0 values before implementation.

Pre-phase commands:

```text
py -3.14 99_Tools/festvox/test_japanese_utau.py -q
py -3.14 99_Tools/festvox/test_japanese_frontend.py -q
py -3.14 99_Tools/festvox/test_utau2festvox.py -q
py -3.14 99_Tools/festvox/festvox_gui/test_festvox_core.py -q
py -3.14 99_Tools/festvox/festvox_gui/test_festvox_gui.py -q
```

Result before the optional frontend was installed: 145 discovered, 144 passed,
one optional `pyopenjtalk` integration skipped. `pyopenjtalk 0.4.1` was then
installed for Python 3.14 and its dictionary asset initialized. The 29-test
frontend suite subsequently passed with no skip. The adapter remains optional
application behavior.

Phase-specific command:

```text
py -3.14 99_Tools/festvox/test_japanese_candidates.py -q
```

Phase-specific result: 21 passed.

Post-phase result: 166 passed, zero failed, zero skipped. This total includes
13 analyzer, 29 frontend, 21 candidate, 7 converter/builder, 48 synthesis-core,
and 48 GUI tests. The same six commands listed above were used, with the new
candidate test command added.

Real-bank read-only validation:

- Full CVVC profile: 12,760 entries, 750 unresolved (5.88%).
- Full VCV profile: 17,506 entries, 1,133 unresolved (6.47%).
- Full CV profile: 749 entries, 16 unresolved (2.14%).
- Every source entry produced one traceable candidate.
- Mixed-family candidates remained present.
- No source-relative WAV reference escaped a bank or was missing.
- Remaining ambiguity is preserved, not guessed or dropped.

Determinism check: two full 12,760-entry CVVC graph builds produced identical
22,433,150-byte JSON and identical candidate IDs. SHA-256:
`64a6a30ceeabdc829491fd5de9f8cfad82d9d8cd06c5def783a015b5ce07b05b`.

Post-phase reference OTO SHA-256 values exactly matched the pre-phase values:

- `D320B7968F37C5C8BD01692D282D47742C31DDE06DF13B8556920EABF9EDD7AA`
- `320771C0BD3DD64F0009C14299F42C46D063ECFAC529D886BA9FBCAD78F7BA84`
- `9D9B0A56CF89B05ADCE186EC82A62B2BCD48FDAB07720B223EB9DCBB4EA2129D`

Commit message reserved for the completed gate:
`feat: add Phase 2 Japanese UTAU candidate compiler`.

## Phase 3 Gate Record

Pre-phase tree: clean. Expected Phase 2 checkpoint: present. The complete
166-test baseline passed before Phase 3 edits.

Phase 3 adds an isolated Japanese Festival/UniSyn compiler, explicit synthesis
planner, reproducible listening-set command, documentation, and deterministic
tests. No English converter, phoneset, lexicon, entry point, or GUI behavior was
modified.

Phase-specific commands:

```text
py -3.14 -m unittest discover -s 99_Tools/festvox \
  -p test_japanese_festival.py
py -3.14 -m unittest \
  test_japanese_festival.JapaneseFestivalCompilerTests.\
test_wsl_festival_preserves_timing_f0_and_manual_override -v
```

The sandbox-visible run executes deterministic unit tests and cleanly skips the
WSL-only case because WSL registrations are hidden there. The same integration
test was run with narrowly elevated WSL access and passed. Festival returned
nonempty audio, exact explicit durations, coherent F0 targets, the expected
automatic context choices, and the requested per-occurrence numbered take.

Real-bank result for the representative CVVC subbank:

- 1,271 Phase 2 source candidates;
- 1,196 selectable candidates;
- 1,191 linguistic candidates compiled into 2,001 edge units;
- five preserved breath candidates deliberately excluded from phone synthesis;
- no absolute source paths in generated metadata;
- source bank remained read-only.

The ignored listening corpus rendered 12/12 examples with zero missing-diphone
fallbacks, zero Festival warnings, and peaks from 0.09 to 0.18. Categories:
vowels, ordinary CV, VCV, CVVC, moraic nasal, gemination, palatalization, long
vowels, phrase boundaries, statement, question, and accent contrast. Acoustic
naturalness is explicitly unverified pending human listening.

Post-phase result: 181 passed, zero failed, zero skipped. This total includes
13 analyzer, 29 frontend, 21 candidate, 15 Festival/planner, 7 legacy
converter/builder, 48 synthesis-core, and 48 GUI tests.

Two complete real generated-voice builds were byte-identical across 440 files.
Aggregate SHA-256:
`7C832D4EC8F23F56710DAF47F06DFE1CAF2DAED2DD245699D4A1BC37E3EC849B`.

Full source-subbank manifest hashes matched before and after Phase 3:

- `64A27B6823B6222DCFFFE06899B65060DB88CDA9021C352F54C8FC9679117B89`
- `4EF53FB15921003C4B0AABF3EF7D240096BDA95381F0E899B65CCA77899B5D35`
- `301AAC8DBFBEA9E3DC2739A8A5A313E81E60712B848E76E98B723079823CFA78`

Commit message reserved for the completed gate:
`feat: add Phase 3 Japanese Festival synthesis`.

## Phase 4 Gate Record

Pre-phase tree: clean. Expected Phase 3 checkpoint: present. The complete
181-test Phase 3 suite passed before Phase 4 edits.

Phase 4 adds a pure Japanese edit-state layer, explicit project migration,
generated-voice runtime metadata access, canonical Japanese generation in the
GUI, mora/accent controls, read-only bank coverage and unresolved-alias
workflow, source inspection, undo/redo, and rebuild-versus-rerender routing.
The existing continuous Pitch curve remains the final F0 authority. Existing
manual unit and stable candidate choices survive unrelated accent/pitch edits.

Phase-specific commands:

```text
py -3.14 99_Tools/festvox/test_japanese_editing.py -q
py -3.14 99_Tools/festvox/festvox_gui/test_festvox_core.py -q
py -3.14 99_Tools/festvox/festvox_gui/test_festvox_gui.py -q
```

Full gate commands also ran the analyzer, frontend, candidate, Festival,
legacy converter/builder, core, and GUI test files. Result: 199 passed, zero
failed, zero skipped. Totals: 13 analyzer, 29 frontend, 21 candidate, 15
Festival/planner, 9 Japanese editing, 7 legacy converter/builder, 51 core, and
54 GUI tests. The frontend suite used local `pyopenjtalk 0.4.1`; the Festival
suite used the local WSL runtime.

The offscreen 1440x900 visual pass verified that the Japanese page fits the
existing Speech layout, scrolls horizontally for long mora sequences, displays
accent-phrase brackets and nucleus markers, and leaves the compact 1024x680
window contract unchanged. Qt's offscreen Windows font backend did not render
text glyphs in the screenshot; widget text, labels, and accessibility state are
covered by the GUI tests.

Read-only source manifests were computed immediately before and after the full
gate with the same relative-path-plus-file-bytes SHA-256 procedure. Counts and
hashes matched exactly:

- representative CVVC subbank, 467 files:
  `CC8145FE81CDABCBBE061368AF6012B837922A5AC34284BE0492D297ACE3F4BE`
- representative VCV subbank, 379 files:
  `D6989D06A7EE95D3A965BA4768CDC3CCE7189C6A29226A6ED64CB1589A7EEB6B`
- representative CV subbank, 229 files:
  `4BA86CF3D2751796139710E8108F88CF83D018542A3739CE17813DAC49C1A94A`

The three OTO SHA-256 values also remained the Phase 0 values recorded above.
No source bank was written, staged, moved, or deleted. No private absolute path
is part of the implementation or documentation.

Commit message reserved for the completed gate:
`feat: add Phase 4 Japanese editing workflow`.

## Phase 5 Gate Record

Pre-phase tree: clean. Expected Phase 4 checkpoint
`d12051b21d719ccba3ee6822a2296b9c822cf848` was present. The complete
199-test Phase 4 suite passed before Phase 5 edits.

Phase 5 adds generated-copy acoustic join metrics and a content-addressed
cache, optional Open-JTalk-label and external-HTS-JSON baseline providers,
deterministic pitch-subbank and exact voice-color routing, GUI controls and
project persistence, an expanded listening corpus, dependency/package checks,
and a release/license inventory. The default structural baseline is unchanged.
Manual per-occurrence candidate choices and continuous Pitch points remain
final. No Open JTalk or HTS waveform is used.

The full gate ran these path-neutral equivalents from the repository root; the
managed runner expanded each test path to an absolute local path:

```text
py -3.14 99_Tools/festvox/test_japanese_utau.py -q
py -3.14 99_Tools/festvox/test_japanese_frontend.py -q
py -3.14 99_Tools/festvox/test_japanese_candidates.py -q
py -3.14 99_Tools/festvox/test_japanese_festival.py -q
py -3.14 99_Tools/festvox/test_japanese_editing.py -q
py -3.14 99_Tools/festvox/test_japanese_quality.py -q
py -3.14 99_Tools/festvox/test_japanese_refinements.py -q
py -3.14 99_Tools/festvox/test_japanese_release.py -q
py -3.14 99_Tools/festvox/test_utau2festvox.py -q
py -3.14 99_Tools/festvox/festvox_gui/test_festvox_core.py -q
py -3.14 99_Tools/festvox/festvox_gui/test_festvox_gui.py -q
```

Final result: 223 passed, zero failed, zero skipped. Totals: 13 analyzer,
29 frontend, 21 candidate, 16 Festival/planner, 11 Japanese editing,
6 quality, 8 refinement, 6 release, 7 legacy converter/builder, 51 core, and
55 GUI tests. The frontend suite used local `pyopenjtalk 0.4.1`; the Festival
suite used the local WSL runtime. An initial sandbox-only run exposed that the
no-waveform test patched an absent optional module directly. The test now
injects a sentinel module and asserts that neither waveform API is called, so
it passes both with and without pyopenjtalk. The complete real-environment gate
then passed without skips.

Real-bank read-only validation remained identical to Phase 2:

- Full CVVC profile: 12,760 entries, 750 unresolved (5.88%).
- Full VCV profile: 17,506 entries, 1,133 unresolved (6.47%).
- Full CV profile: 749 entries, 16 unresolved (2.14%).
- Every source entry still produced one traceable candidate.
- Unresolved entries and secondary alias families remained visible.

A fresh representative CVVC generated voice compiled 2,001 edge units from
1,191 linguistic candidates out of 1,196 selectable candidates. Five preserved
nonlinguistic/breath candidates were deliberately excluded from phone
synthesis. The expanded corpus rendered 16/16 examples through the separate
Japanese voice entry point with Open JTalk 0.4.1. It measured 169 joins:
36 `review`, zero `poor`, and peaks from 0.099 to 0.222. The declared-color
stress case visibly reported one `o-n` fallback instead of hiding it. Acoustic
naturalness remains unverified pending human listening.

Two independent listening runs produced byte-identical 115-file trees,
including WAVs, manifest, and quality cache: 1,932,477 bytes with aggregate
SHA-256
`B3CEEDF4ED09F2418B45741DE5CF6AD2C3383C2E05EA1E0B1548502A1829EA72`.
The release report also repeated byte-for-byte with SHA-256
`698AA8E5848FCA45B21C453E042134FDA351A78E7780A7ED0632DB19349D39DD`.

The release checker found all tested required and optional dependencies,
including pyopenjtalk 0.4.1, and no bundled Open JTalk dictionary or HTS voice.
Implementation checks passed. Redistribution readiness remains false because
the project has no declared license, every source UTAU bank has independent
terms, and PyQt5 requires a GPLv3 or commercial distribution decision.

Offscreen visual QA covered 1440x900 and 1024x680 layouts. The refinement
controls remain stable without overlap, and the compact layout scrolls rather
than compressing the voice-color control. Qt's offscreen Windows font backend
does not draw text glyphs; widget labels, state, enablement, persistence,
undo/redo, and manual-override precedence are covered by GUI tests.

Read-only source manifests matched before and after tests, the real build, and
both listening renders:

- representative CVVC subbank, 467 files:
  `CC8145FE81CDABCBBE061368AF6012B837922A5AC34284BE0492D297ACE3F4BE`
- representative VCV subbank, 379 files:
  `D6989D06A7EE95D3A965BA4768CDC3CCE7189C6A29226A6ED64CB1589A7EEB6B`
- representative CV subbank, 229 files:
  `4BA86CF3D2751796139710E8108F88CF83D018542A3739CE17813DAC49C1A94A`

No source bank was written, staged, moved, or deleted. Generated voices,
listening WAVs, quality caches, release reports, and screenshots remained in
ignored or temporary output locations. No private absolute path is committed.

Commit message reserved for the completed gate:
`feat: add Phase 5 Japanese synthesis refinements`.

## Open Risks

- Acoustic quality and naturalness are unverified; automated rendering proves
  structure only.
- Ambiguous nasal-color and boundary aliases require profile overrides or later
  acoustic evidence.
- Phase 3 OTO split assumptions have deterministic join metrics but still need
  human listening.
- Open JTalk dictionary redistribution is not implied by local installation;
  the exact inventory is documented, and packaging remains blocked until its
  license checklist and the application's release license are resolved.
- Exact voice-color routing can expose missing color-specific diphones; these
  retain the ordinary candidate with a visible diagnostic rather than failing
  silently.

## Corrective Stage 5: Speech Editing Workflow

The remediation pass replaces the old equal-width Japanese scroller with a
waveform-time mora strip. Mora intervals are derived from the rendered plan and
current Segment boundaries; waveform selection, mora selection, horizontal
view range, and playhead now share that seconds map. Dense utterances use a
label-suppressed overview while the selected mora keeps a readable target.

The Japanese parameter now contains linguistic accent editing only.
Single-click mora selection is cosmetic; double-click places the nucleus and
right-click marks the accent phrase unaccented. Accent phrases support
draggable boundaries plus split and merge
commands. `accent_phrase_boundaries` stores global mora-start indexes in the
project overlay, and rebuilding a phrase also rebuilds every mora and phone's
accent-phrase membership. Accent/nucleus/structure edits use the existing undo
stack, persist in version-4 projects, and invalidate only Re-render.

Question controls were removed from Japanese Accent; punctuation question
shape remains in Intonation blocks. Phrase duration moved to **Options > Phrase
pauses...** as bounded minor/major/sentence millisecond totals. The values are
stored in `config.json` and project settings. Generate applies them in every
Festival text frontend; Re-render changes only internal pause runs and retains
the spoken-phone sequence and indexes.

Source inspection moved to Recordings. The selected-mora command expands the
stored source contribution plan and shows every role, alias, WAV, source slice,
target edge, and fallback. Manual per-edge choices remain on the ordinary
Recordings track.

Stable GUI controls for dynamic pitch-bank and voice-color routing were
removed. Legacy metadata remains readable, while `create_edited_plan` requires
an explicit `allow_experimental_routing=True` API opt-in before either can
affect unit choices. Separate generated voice configurations remain the stable
workflow.

Qt high-DPI attributes are set before QApplication construction. Global text
uses points, and startup verifies Japanese glyph support while preferring Yu
Gothic UI, Meiryo UI, Meiryo, and Noto Sans CJK JP before scanning installed
system fonts. No fonts are bundled.

## Corrective Stage 6: Japanese Timing And Baseline Contour

The Japanese synthesis plan is now provisional schema 2. It records a
mora-first duration plan alongside explicit Segment durations. Ordinary CV,
vowel-only, palatalized, geminate, moraic-nasal, long-vowel, devoiced-vowel,
and phrase-final cases have separate allocations. Every phone row exposes the
predicted and final duration, source reference duration, OTO-derived safe
range, requested stretch, and constraint source. Render Details presents the
same information without making OTO landmarks the linguistic duration model.

When several automatic candidates are available, source safety uses the
shortest valid contributing half. This avoids inflating a normal consonant
because a context-rich alternative contains a longer consonant region. A
manual per-occurrence candidate remains authoritative. CV-bank `* V` aliases
remain OTO-timed vowel-blend fallbacks, distinct from phrase-initial `- V`
aliases; exact VCV or CVVC transitions still outrank that fallback.

The structural Japanese F0 baseline is expressed in semitones around the
shared FRQ speaker analysis. It combines phrase-initial rise, accent-nucleus
fall, bounded downstep, breath-group reset, sentence declination, and
phrase-final lowering. Japanese and full-width punctuation is normalized into
the shared Intonation-block layer. Authority is:

```text
shared speaker analysis
Japanese structural baseline
Pitch-accent edits
general Intonation blocks
continuous Pitch points
safety bounds
```

Question rise is therefore absent from the structural accent generator and is
applied only by Intonation blocks. Speaking-rate changes scale duration without
scaling F0. Optional duration providers update the serialized timing rows as
well as the Segment relation, so diagnostics never describe stale timings.

The stable listening corpus now contains 16 cases: vowels, ordinary CV, VCV,
CVVC, moraic nasal, geminate, palatalized, long vowels, devoiced vowels, phrase
boundaries, statement, question, accent contrast, multiple-accent downstep,
long-phrase declination, and boundary-stress joins. Two independent ignored
runs produced byte-identical 137-file trees: 1,970,154 bytes with aggregate
SHA-256
`8F5C7A1E066D3C01C93C7356B80BADA3D9BC817978CC43933CB97DE1A2C47844`.
All 16 cases rendered. The manifest contains 192 phone timing rows, zero
duration clamps, 190 analyzed joins, 33 `review`, zero `poor`, zero missing
diphones, and zero render warnings. Acoustic naturalness remains unverified
pending human listening.

Final gate commands:

```text
py -3.14 -m unittest discover -s 99_Tools/festvox -p test_*.py
py -3.14 -m unittest discover -s 99_Tools/festvox/festvox_gui -p test_*.py
py -3.14 -m unittest test_japanese_festival.JapaneseFestivalCompilerTests.test_wsl_festival_preserves_timing_f0_and_manual_override -v
```

Results: 160 synthesis, builder, frontend, release, and safety tests passed
with one expected sandbox-hidden WSL skip; all 120 GUI/core tests passed; the
same WSL/Festival integration test then passed in the real local runtime.
English regression behavior remained green.

The protected representative CVVC source subbank remained 467 files with the
same relative-path-plus-bytes SHA-256:
`B6A874A371397E0825D05ED5BEE87085BEFD5FE174F18D435B591D5059458F80`.
No source UTAU file was written, staged, moved, or deleted. Listening WAVs and
manifests remain under ignored `tmp/` output directories.

Commit message reserved for this completed gate:
`fix: improve Japanese duration and baseline contour`.

## Corrective Stage 7: Phrase And RB Selection Guards

The GUI phrase splitters now recognize Japanese full-width punctuation without
requiring trailing whitespace. Western punctuation retains its spaced split
rule. Asaxi input is lowercased before either synthesis frontend. Pitch
middle-pan uses a fixed pixel delta and initial Hz span, Shift-wheel performs
horizontal waveform scrolling without selection movement, and Sentences labels
selected playback explicitly. Single-click mora selection no longer changes
accent; double-click places the nucleus.

The Japanese OTO classifier treats context-prefixed `RB` and numbered `RBn`
as preserved, nonselectable rest-breath material. It does not reinterpret them
as tapped `r`. Read-only validation on the representative Phascogale CVVC scope
found 1,280 candidates: all 31 RB sources were breath/nonselectable, while all
51 tapped-r VC sources remained selectable. The source tree stayed at 467
files with SHA-256
`b6a874a371397e0825d05ed5bee87085befd5fe174f18d435b591d5059458f80`
before and after analysis.

## Corrective Stage 8: Editor Reliability And Built-In Kal

Festival's WSL `kal_diphone` is now the authoritative built-in registration.
An optional or stale Windows mirror cannot shadow it, change its English-only
compatibility, or make it removable. The Voicebank Manager supports extended
selection and one consolidated deletion warning for generated banks while
continuing to reject Festival built-ins and source UTAU folders.

Speech editing now provides one-shot full-duration Auto Adjust, a dedicated
pitch timeline scrollbar and vertical pitch zoom navigator, and neutral pitch
nodes and line sections over `pau`. The Pitch accent page
retains cosmetic single-click selection while allowing the existing nucleus
triangle to be dragged after the normal pointer threshold. Voice changes mark
sentences for fresh Generate. Mixed sentence selections show `-` for Language
and Voicebank; each control applies in bulk without collapsing already
supported mixed languages, and changing tabs clears row and phrase selection.

Japanese alias classification is case-sensitive for CVVC rest material.
Uppercase context tokens `R`, `Rn`, `RB`, and `RBn` remain traceable but
nonselectable, while lowercase tapped `r` remains an eligible VC transition.
Two real-bank graph builds produced identical 2,105,134-byte JSON with SHA-256
`b39d07d279c6f147ea990b3d6752ffcea3c6dc0e653b0b8cb26cbd0f7dad8eca`.
All 1,280 entries remained traceable; all 61 inspected uppercase `R`/`RB`
context candidates were nonselectable, and all 21 inspected lowercase `r`
candidates remained selectable.

Final verification passed 170 repository tests, 64 dedicated core tests, and
75 offscreen GUI tests: 309 total. Native Windows screenshots verified the
pitch footer, pause styling, and mixed-selection controls without overlap. The
screenshots remain in ignored temporary output. The protected representative
source stayed at 467 files with the established relative-path-plus-bytes
SHA-256
`b6a874a371397e0825d05ed5bee87085befd5fe174f18d435b591d5059458f80`
before and after read-only analysis.

## Corrective Stage 9: Shared Source F0 And Prompt 17 Workflow

Source pitch analysis is now shared by every generated UTAU voice. English,
Asaxi, integrated multilingual, and Japanese builders use the same policy:
valid UTAU FRQ data is authoritative; otherwise the builder uses WORLD
Harvest plus StoneMask by default, with DIO plus StoneMask as an explicit
faster option. The generated UniSyn pitchmarks are phase-aligned to waveform
zero crossings and retain the analyzed voiced/unvoiced contour in a
deterministic `pm/*.f0.json` sidecar. `pm/pitchmark_sources.json` records the
method used for each generated WAV. This replaces the old EST `pda` fallback,
which could create locally plausible marks around a broadly corrupt harmonic
region. Rebuilds can select `--f0-estimator harvest` or
`--f0-estimator dio`; the option is language-independent.

The Recordings menu can display the exact generated WAV and pitchmarks used by
UniSyn without opening the source UTAU bank. The inspector overlays the
analyzed F0, preserves unvoiced gaps, separately shows the F0 implied by epoch
spacing, marks local period jumps, and remains compatible with older voices
that have no analyzed-F0 sidecar. Source provenance paths are never
dereferenced by this command.

The default was selected conservatively from both primary documentation and a
read-only bank comparison. WORLD provides DIO and Harvest under a modified BSD
license, while the PyWORLD wrapper is MIT-licensed and recommends Harvest for
lower-SNR speech. On seven representative source recordings, both estimators
tracked FRQ closely: Harvest had complete voiced coverage and 2.39-cent median
absolute disagreement; DIO had 99.5% coverage and 2.37-cent median
disagreement. Harvest therefore remains the robust default and DIO remains a
deterministic speed option. pYIN, CREPE, and RMVPE were assessed but are not
exposed in the stable builder: they add materially larger signal-processing or
model dependencies, and RMVPE is specifically designed for vocal F0 in
polyphonic music rather than this isolated-speech source analysis. References:

- WORLD: <https://github.com/mmorise/World>
- PyWORLD: <https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder>
- Harvest: <https://www.isca-archive.org/interspeech_2017/morise17b_interspeech.pdf>
- pYIN: <https://www.eecs.qmul.ac.uk/~simond/pub/2014/MauchDixon-PYIN-ICASSP2014.pdf>
- CREPE: <https://arxiv.org/abs/1802.06182>
- RMVPE: <https://www.isca-archive.org/interspeech_2023/wei23b_interspeech.pdf>

The Speech pitch editor now places zoom and vertical navigation together in a
vertical control beside the working area. Phrase-final pitch segments survive
Generate and Re-render, including sentences separated by the two-pause phrase
model. Compiled vowel-to-vowel transitions are labelled `VV`; source aliases
may still retain `VCV` as provenance where that is the bank's recording
family.

The Sentences tab can follow the sentence currently being spoken, with a
persistent opt-out. Generate All commits every row waveform before returning,
keeps the user's active tab, and consolidates identical wrong-language or
incompatible-voice failures into one actionable message. A sentence without a
phrase snapshot receives one full-sentence waveform snapshot rather than an
empty row.

Final verification passed 174 repository tests, 66 dedicated core tests, and
80 offscreen GUI tests: 320 total. Native layout renders verified the Speech
pitch navigator, phrase-final contour, source-pitchmark inspector, Sentences
waveforms, and follow control without overlap. Temporary screenshots remain
ignored. The protected representative source scope stayed at 467 files with
the established relative-path-plus-bytes SHA-256
`b6a874a371397e0825d05ed5bee87085befd5fe174f18d435b591d5059458f80`.
No source UTAU file was written, staged, moved, or deleted. Acoustic
naturalness remains unverified pending listening to a rebuilt voice.

## Corrective Stages 10-12: Join Diagnostics and Restored Control

Abrupt artifacts occur at UniSyn handoffs where `A-B` meets `B-C`, which are
inside the displayed shared phone rather than necessarily at a visible phone
boundary. The active builders retain OTO-aware center geometry: positive OTO
overlap can establish an ARPAsing left center, a following same-recording
transition can establish the right center, and Japanese retains its distinct
CV/VCV/CVVC geometry. This metadata remains useful for source inspection and
does not alter contextual candidate scoring.

Three attempted acoustic repairs were rejected after listening and fixed-unit
spectrogram comparison:

- bounded gain tapers on generated WAV copies changed natural envelopes and
  produced a worse rendered result;
- asymmetric Hanning windows changed a single-source UniSyn frame window but
  did not implement a genuine two-source crossfade;
- forcing source pitchmarks onto nearby zero crossings produced broad
  syllable-level corruption rather than a reliable phase match.

All three are removed from the active path. Both builders emit Festival's stock
`UniSyn` synthesis method, generated WAVs remain unconditioned copies, source
pitchmarks use the established FRQ/WORLD-guided waveform method, and the GUI
does not inject a replacement synthesis type. The legacy `join_repairs` field
remains empty for compatibility. No repair-driven cost or override reaches the
unit selector, so contextual and manual occurrence choices stay final.

The replacement is diagnostic-only. `join_discontinuity.py` keeps level,
sample/derivative, F0, phase, period-shape, spectral-value, spectral-slope, and
local novelty evidence. A multiresolution 0.5/1/2/3 ms scan now detects brief,
locally novel, unusually flat full-band impulses. Sustained frication is not
enough to trigger it. Stop/affricate context marks a release burst as possibly
expected without suppressing the raw score. `join_spectrogram.py` points small
triangles at measured event regions, draws handoff spans above rather than over
the STFT, includes an aligned rendered-phone strip, tints stop/closure cells,
and embeds its own legend.

Festival 2.5's stock `segment_single` mapping assigns one source frame to each
target pitchmark. It cannot mix the final incoming and first outgoing source
periods at the same target epoch; the available `interpolate_joins` code is an
unfinished debug path. A true pitch-synchronous crossfade therefore requires a
tested Festival/UniSyn engine extension or a different renderer. No further
pseudo-crossfade is active.

A fresh read-only Japanese CVVC control build loaded successfully and passed
its Festival smoke render. The fixed utterance `あの高いビルはホテルです。`
produced 44.1 kHz, 92,437-sample audio with 26 exact handoffs, 12 diagnostic
flags, and two unexpected broadband events. The earlier combined
gain/window/phase experiment produced 18 flags on the corresponding fixture.
The control OTO remained SHA-256
`18bf9730879a707103785e08dcf5b917623b13bd942c33fa554fd4d76eb031a4`
before and after. This is structural and visual evidence only; acoustic
naturalness still requires human listening.

Final rollback verification passed 215 root tests and 149 offscreen GUI/core
tests (364 executions). The source pitch scope remained 467 files and
111,806,607 bytes with aggregate relative-path-plus-content SHA-256
`316ab2bd7f326bfcb0b1090a4ae519e09f2d72650bc518397d2a584448fc6ffd`.

## Prompt 19: Contextual Duration And Source-Filter Voicing

Prompt 19 extends the normal Japanese planner with a versioned, deterministic
source-relative duration model. `japanese_duration.py` retains selected-source
timing as the absolute speaker baseline and adds bounded log-space context
residuals for phone class, speed, partial CV compensation, long vowels,
geminates, moraic nasals, likely high-vowel devoicing, accent/phrase edges, and
utterance boundaries. `legacy` remains selectable.

Moraic-nasal semantics now survive bank mapping. Integrated `nn`, `nng`, `mm`,
and `xn` occurrences carry the explicit `moraic_nasal` timing role, receive
class/reference and source-geometry caps, and are not stretched by
consonant-only edits. This does not globally reinterpret `nn` outside Japanese.

Direct-waveform UniSyn TD-PSOLA was experimentally confirmed to preserve
periodic source pulses even at an artificial 40 Hz target. The implemented
realization therefore separates timing from voicing and applies deterministic
source/filter residual modification after UniSyn, while preferring already
aperiodic source material and retaining a safe shortened-voiced fallback.
Manual continuous Voicing points remain final over a stable automatic curve.

The fixed validation matrix, objective results, corpus interfaces, licensing
limits, and commands are recorded in `PROMPT19_IMPLEMENTATION.md`,
`PROMPT19_BENCHMARK_REPORT.md`, and `docs/japanese_duration_model.md`. Final
acoustic naturalness is not claimed without human listening.

## Prompt 20: Full Prosody And Vocal-Tract Resonance

Prompt 20 replaces absolute-Hz Japanese contour arithmetic with a
speaker-relative semitone/log-F0 model, calibrates contextual duration and
contour shape against train/held-out Kokoro-Align measurements, and integrates
automatic mora voicing into normal generation. Repeated phrases receive
mean-centered contour-shape variation rather than cumulative frequency drift.
The GUI exposes model IDs and confirms that register drift is disabled.

`Re-render Phonemes` now preserves the editor's exact segment durations. It may
refresh F0, voicing, recording, and vocal-tract metadata, but only a fresh
Generate may replace the current timeline with newly modeled durations.

The analysis-first vocal-tract feature uses the Stage A true-envelope/Burg
reference profile and a Stage B overlap-add source/filter transform. The
continuous `Vocal tract length` curve is independent of pitch and duration;
the ordinary range is reference-derived and the separately validated expanded
range requires `Chipmunk range`. Final-waveform five-vowel validation, blind
listening fixtures, exact commands, model parameters, source hashes, objective
prosody metrics, deterministic held-out source-versus-synthesis waveform
alignment plots, and remaining perceptual risks are recorded in
`PROMPT20_IMPLEMENTATION_REPORT.md`.

The final duration profile is v7. Kokoro punctuation pauses are refined to
low-energy runs before fitting. Open JTalk morphology is mapped onto canonical
morae and retained verbatim alongside coarse grammatical roles. Ordinary and
negative auxiliary shortening survived both train and held-out gates;
particle, polite-copula, and blanket final-vowel effects were explicitly tested
and rejected. A read-only rendered-edge analyzer found that vowel-initial
phrases begin substantially before their logical Festival boundary. The
training-derived 50 ms bounded compensation reduced held-out effective
first-mora excess from 65.4 to 15.4 ms without changing unit selection.

## Shared Measured Join Update

The later shared-renderer update does not reinstate the rejected rendered-PCM
repair or the failed residual-epoch experiment. Both normal and Legacy
pitchmark files use the same FRQ/WORLD-guided negative-going, low-pass
zero-crossing epochs; their separation records policy provenance rather than a
claimed universal excitation phase.

Normal Festival/WSL synthesis now renders a millisecond-authoritative,
pitchmark-snapped crossover in the project-local native helper. Requests are
limited by phone/context safety and can be edited per occurrence. The
discontinuity analyzer continues to report period, level, waveform correlation,
spectral shape, and boundary novelty without overriding contextual or manual
recording choices. Fault Mode > Legacy joins bypasses the helper and restores
the exact stock-Festival waveform path for controlled A/B testing. See
`JOIN_SYNTHESIS.md` for implementation, tests, and limitations. Acoustic
naturalness remains subject to listening.

## Integrated Tap Mapping Correction

Open JTalk 0.4.1 can retain non-spoken Japanese quotation marks in its kana
reading while omitting them from full-context phone labels. Previously those
marks were counted as unknown morae. The resulting reading/label mismatch made
the parser discard all kana mora readings and use label-derived strings such
as `ra` and `ri`; those strings bypassed the integrated ARPAsing profile's
`ら -> dx` and `り -> dxy` entries.

Reading segmentation now ignores non-spoken quotation/bracket punctuation.
The ARPAsing mapper also has a deterministic profile-backed recovery path: it
derives the stable hiragana key from a canonical mora phone tuple and applies
the selected voice's own target mapping. This recovery is active only when a
generated voice supplies `japanese_phoneme_map`; Japanese-only voices continue
to use canonical Japanese `r`.

The regression sentence beginning `24日も東海から九州を中心に` now analyzes to
66 aligned morae. Its integrated Lem plan and Festival-returned Segment
relation contain `dx`, `dxy`, and `dxy` for the three liquid morae, with no
standalone `r`. Deterministic tests cover ignored quotation punctuation and
label-derived `ra/ri/ru/re/ro` recovery. The optional real-pyopenjtalk test
uses the complete news sentence.

## Inline Japanese Pause Reconciliation

Open JTalk 0.4.1 emits the same generic `pau` label for explicit punctuation
and for non-spoken quotation or bracket marks. Promoting every such label to a
phrase boundary made `「熱中症警戒アラート」は` contain an 880 ms three-part
pause between the closing quote and topic particle.

The full-context parser now reconciles each pause with its source mora
boundary. A bracket-only pause remains in the phone list, raw-label provenance,
and accent-phrase structure, but it does not split the rendered phrase.
Explicit punctuation at the same boundary wins: `」、「` remains a minor comma
pause, while `」は` is continuous. The parser records affected label indices
under `inline_bracket_pause_label_indices` and emits the informational
`openjtalk_inline_bracket_pause` diagnostic.

Source punctuation reconciliation also treats decimal `.`/`．` between digits
as spoken `テン`, `・` as a minor list boundary, and `▽` as a major list-item
boundary. The supplied 701-character news fixture now produces 55 source-
matched phrases with no reading-mora or source-phrase mismatch. It retains
42 comma boundaries, three triangle/list boundaries, two middle-dot
boundaries, and seven sentence stops while suppressing 11 bracket-only pauses.

The GUI's normal Japanese path consumes this canonical plan and therefore
contains no quote-to-particle silence. The lower-level Festival/WSL text API
also has a guarded compatibility correction: when its ordered internal pause
runs exactly match ordered inline source symbols, bracket pauses are capped at
60 ms before the explicit Segment/PSOLA pass. Any count mismatch leaves the
Festival plan unchanged, and explicit `[pau]` durations are never shortened.

A real WSL render with `lem_v4bi_integrated` confirmed one phrase and only the
two leading plus two trailing edge pauses for the reported sentence. Festival
returned a nonempty 44.1 kHz waveform with no missing units. Acoustic
naturalness still requires user listening, but the former structural 880 ms
quote gap is absent.
