# Japanese UTAU Integration: Phase 3

Phase 3 is the first Japanese waveform phase. It compiles the read-only source
candidates from Phase 2 into a separate Festival/UniSyn voice and creates an
explicit Japanese rendering plan. It does not route Japanese through the
English phoneset, lexicon, entry point, or ARPAsing converter.

> **Current remediation note:** the original plan schema 1 timing/F0 baseline
> is superseded by additive schema 2 in `japanese_synthesis.py`. The Festival
> entry point and unit compiler remain Phase 3-compatible; schema 2 adds
> mora-first timing diagnostics and speaker-relative phrase/accent contouring.

## Files and APIs

- `japanese_festival.py`
  - `candidate_edge_proposals(candidate, wav_duration_seconds)` converts one
    source candidate into canonical Japanese phone edges.
  - `compile_festival_voice(graph, output, ...)` writes a versioned Japanese
    UniSyn voice.
  - `load_japanese_runtime_metadata(path)` validates the generated runtime
    index.
  - The command-line entry point accepts a source bank/subbank and a generated
    output directory.
- `japanese_synthesis.py`
  - `create_synthesis_plan(utterance, ...)` returns explicit phone durations,
    structural F0 targets, and resolved per-occurrence unit overrides.
  - `JapaneseSynthesisPlan.backend_arguments()` targets the existing
    `FestivalWSLBackend.synth_phones` API.
- `japanese_assembly.py`
  - mirrors the generated Festival selector in Python;
  - exposes one exact source-contribution row per canonical phone edge;
  - validates shared centers, paired candidates, hidden silence, and fallback
    visibility.
- `japanese_assembly_listening.py`
  - renders the corrective CV/VCV/CVVC comparison corpus;
  - writes exact contribution JSON beside every ignored WAV.
- `japanese_listening_set.py`
  - renders the ignored, path-neutral Phase 3 human-listening corpus;
  - records failures, skipped diphones, diagnostics, peaks, and structure in
    `manifest.json`;
  - always records `acoustic_naturalness_verified: false`.
- `test_japanese_festival.py` covers compiler, planner, safety,
  determinism, and real Festival behavior.

The generated runtime schema is `phase3-provisional`, version 1. It contains
the exact diphone index, alternatives, stable candidate-to-unit mappings,
average pitch, and the Japanese voice entry point. The remediation manifest
also separates `source_bundle_id` from `configuration_id` and records
`primary_language`, `supported_languages`, `alias_system`, scoped alias and
phone namespaces, and `voice_entry_points`. It contains source-relative
provenance but no source-bank absolute path.

## Generated Voice

`compile_festival_voice` creates:

```text
wav/                              copied WAVs + bounded bridges + digital silence
pm/                               EST pitchmarks
  *.f0.json                       analyzed voiced-F0 sidecars
  pitchmark_sources.json          per-unit FRQ/WORLD provenance
dic/<name>_ja_diphone.est         UniSyn index
dic/diphone_index.json            runtime index and candidate map
dic/unit_alternatives.json        GUI/backend alternative map
dic/japanese_build_report.json    deterministic build report
festvox/<name>_ja.scm             Japanese-only voice entry point
```

The entry point is `voice_<name>_ja`. The Scheme file defines its own Japanese
phoneset and UniSyn selector. It does not define an English companion entry
point or load an English frontend.

Primary build:

```text
py -3.14 build_festival_voice.py --language ja \
  --bank-type <cv|vcv|cvvc> --samples <source-recording-root> \
  --oto <selected-oto.ini> --output <generated-output> \
  --name <voice-name> --f0 180 --wsl-distro Ubuntu --test
```

The shared front door derives WSL paths, writes the common portable manifest,
and performs a real canonical-plan Festival smoke render. The direct
`japanese_festival.py` entry point remains an expert/test API. See
`UNIFIED_VOICE_BUILDER.md` for the complete command contract.

The bank type is deliberately required. `auto` and `mixed` remain useful
read-only analysis modes, but the compiler refuses to turn either one into a
generated voice. Different explicit interpretations of the same recordings
receive different configuration, alias, phone, and candidate identities.

The compiler refuses an output directory inside the source bank. Source WAVs
are copied byte-for-byte to generated output. Pitchmarks are generated from
those unconditioned copies and the generated voice uses Festival's stock
`UniSyn` synthesis type. OTO overlap and the role-specific CV/VCV/CVVC geometry
below remain alignment metadata; they are not interpreted as permission to
normalise, taper, phase-lock, or post-process source audio. Contextual and
manual candidate selection remains final. Rendered join quality is assessed by
the read-only analyzer documented in `JOIN_DISCONTINUITY_DIAGNOSTICS.md`.

## OTO Geometry

The authoritative OTO region is:

```text
start = offset
vowel alignment = offset + preutterance
end = offset + abs(cutoff), when cutoff is negative
end = WAV duration - cutoff, otherwise
```

The corrective Stage 2 edge rules are explicit in metadata:

- every consonant-bearing CV starts at a bounded consonant center between OTO
  overlap and preutterance;
- every VC ends at a bounded consonant center after its phone boundary;
- one-phone vowels and special morae create a sustain edge;
- phrase-start, VCV, and release pairs end and begin at one identical source
  phone-center anchor, so no consonant interval is replayed;
- canonical `V-cl-C` remains visible for mora timing, but the shared
  special-phone resolver sources it as `V-C-C`;
- the generated `C-C` edge is a bounded hold from the consonant portion of a
  normal `C-V` recording, while the following `C-V` unit remains the only
  consonant release;
- a missing transition uses a visible, bounded generated bridge from a stable
  left-phone region to the next CV onset, never the hidden default silence.

These rules make the geometry testable and reversible. OTO arithmetic alone
does not prove that a join sounds natural. See
`JAPANESE_ASSEMBLY_REMEDIATION.md` for the full contract and audit results.

## Selection and Overrides

Automatic selection is deterministic:

- phrase-start candidates require both pause context and the recorded following
  vowel on their left edge;
- VCV candidates require the recorded vowel context and retain one candidate
  across their `-1` and `0` edges;
- VC candidates are preferred for matching geminate closure context;
- explicit CVVC treats both incoming CV and outgoing VC/release as primary;
- material from another alias family is retained but visibly demoted to
  fallback status;
- Phase 2 selection cost breaks remaining ties.

Every stable candidate ID remains mapped even when two OTO rows share identical
audio and geometry. A manual candidate override is resolved to one or more
exact UniSyn left-unit names and applied only to the selected mora occurrence.
Manual choices are final and bypass automatic scoring.

## Timing and F0

`create_synthesis_plan` builds explicit Festival `Segments` input. It supplies:

- conservative phone-class durations;
- two independently editable pauses at internal phrase boundaries;
- stable two-pause utterance edges;
- structural accent-phrase targets from the canonical model;
- a final interrogative rise when marked by the frontend;
- neutral diagnostics when lexical accent is unknown.

Accent nuclei are structural inputs, not an imported HTS trajectory. The
existing continuous pitch editor remains the final authority because its
explicit target points are passed directly to Festival/UniSyn PSOLA.

Source-waveform F0 estimation is shared with English and Asaxi; it is not a
Japanese frontend behavior. UTAU FRQ is authoritative per recording. Missing
FRQ falls back to WORLD Harvest (quality default) or DIO (fast option), followed
by StoneMask and phase-aligned epoch construction. The generated sidecar keeps
unvoiced frames as zero rather than drawing default-filled PSOLA epochs as a
real voiced contour. Only isolated octave slips and gaps shorter than 35 ms are
sanitized; a sustained corrupt region remains visible and is never smoothed
into a plausible-looking contour.

A compiled vowel-to-vowel diphone is reported as `VV`. A VCV bank alias may be
its source recording family, but the runtime edge itself is not called VCV.

## Verification

The deterministic synthetic suite covers:

- CV, VCV, VC, release, sustain, nasal, geminate, and palatalized units;
- arbitrary numbered takes and byte-identical candidate preservation;
- source-output separation and source hash stability;
- metadata determinism and private-path exclusion;
- two-pause phrase plans, question targets, and stable serialization;
- single- and multi-edge manual candidate overrides;
- Japanese/English entry-point isolation.
- native crossfade Scheme generation, parameter restoration, and unchanged
  contextual/manual unit-selection ordering.

The corrective assembly suite additionally covers isolated CV, VCV, and CVVC
banks; the exact `あ`, `か`, `あか`, `かか`, `あき`, `かさ`, `さか`, `あん`,
`あった`, `きゃ`, and `きょう` matrix; byte-identical generated bridges; and
rejection of the former overlapping-consonant geometry.

The real WSL Festival test verifies nonempty audio, exact returned phone
durations, coherent F0 targets, clipping bounds, context-sensitive automatic
selection, and a numbered manual take changing only its intended occurrence.

A representative real CVVC configuration produced 2,033 indexed units from
1,200 linguistic candidates and 20 bounded generated bridges. Representative
VCV and CV configurations produced 3,183 and 548 units respectively. Their
12-example corrective corpora all rendered with zero skipped diphones, hidden
silence, structural errors, or automated `poor` joins. Fallbacks remain
explicit: a compact CV bank naturally uses more generated transitions than a
recorded CVVC or VCV bank.

Acoustic naturalness remains unverified pending human listening. The generated
WAVs and real voice build remain in ignored output directories.

## Phase Boundary

Phase 3 does not add GUI analysis, mora/accent editing, persistence, migration,
undo/redo, duration-provider experiments, dynamic multipitch routing, or
packaging. Those remain Phase 4 and Phase 5 work.
