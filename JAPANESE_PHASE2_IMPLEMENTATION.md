# Japanese UTAU Integration: Phase 2

Phase 2 adds read-only Japanese bank profiles and a deterministic source-unit
candidate graph. It does not slice audio, generate a Festival voice, alter the
GUI, or route Japanese through the English ARPAsing implementation.

## Files and APIs

`japanese_profiles.py` provides:

- `infer_bank_profile(source, ..., oto_files=None) -> JapaneseBankProfile`
- `profile_json_bytes(profile) -> bytes`
- `write_profile(profile, output)` with a source-bank write guard
- `load_profile(path) -> JapaneseBankProfile`
- `resolve_bank_context(source) -> BankContext`

`japanese_candidates.py` provides:

- `compile_candidate_graph(source, profile=..., oto_files=None) ->
  JapaneseCandidateGraph`
- `candidate_metadata_bytes(graph) -> bytes`
- `write_candidate_metadata(graph, output)` with a source-bank write guard
- `format_coverage_summary(graph) -> str`

Both modules have small command-line entry points. Their imports remain
separate from `utau2festvox.py`, `build_festival_voice.py`, and the English
phoneset/frontends.

When `oto_files` is supplied to profile inference or candidate compilation, it
is an exact configuration scope: sibling OTO files are not rediscovered.
Omitting it preserves the original recursive analysis behavior. The underlying
`japanese_utau.analyze_bank(..., oto_files=...)` adapter follows the same rule.
Every explicit file must be an existing `oto.ini` beneath the resolved source
root.

## Profile Model

`JapaneseBankProfile` stores:

- requested, inferred, and effective bank configuration;
- inference confidence;
- default and per-OTO encoding policy;
- exact alias prefixes and suffixes;
- enabled candidate families and optional voice color;
- exact alias or `relative/oto.ini:line` overrides;
- bank-specific moraic-nasal allophone aliases and following-phone routes;
- discovered OpenUtau/UTAU subbanks;
- source-relative metadata provenance and hashes;
- diagnostics and the mandatory `preserve` unknown-alias policy.

`character.yaml` is considered before `prefix.map`. The dependency-free YAML
reader handles the scalar/list subbank shape used by OpenUtau, including
unquoted sharp pitch suffixes such as `C#4H`. Unknown YAML fields are ignored;
undecodable metadata raises instead of being replaced or silently discarded.
Explicit profile policy can override inferred configuration and alias roles,
but inference evidence remains visible.

The profile schema is version 1 and explicitly labeled
`phase2-provisional`. It does not freeze the future generated-voice metadata
version 2 schema.

## Candidate Graph

Every parsed OTO entry produces exactly one traceable candidate. Roles are
structurally distinct:

- `mora_cv`
- `phrase_start_cv`
- `vowel_blend`
- `vcv_mora`
- `vc_transition`
- `release`
- `special_mora`
- `silence`
- `breath`
- `extra`
- `unresolved`

Targets keep canonical mora phones separate from source aliases. CV, start-CV,
VCV, VC, and release targets therefore cannot collapse into one namespace.
Mixed banks retain candidates from every family. Automatic analysis remains
permissive. An explicit CVVC build keeps clear VCV mora rows in the graph but
marks them non-selectable so they cannot enter the Festival runtime; ordinary
CV and recorded VC/VV rows remain the two halves of CVVC synthesis. Disabled
families and invalid-timing rows likewise remain present and non-selectable.

Numbered alternatives are recognized after declared prefix, suffix, pitch, and
voice-color metadata. The exact OTO alias is always retained. A conservative
Phase 2 compatibility layer also recognizes small-kana OTO spellings, nasal
allophones as moraic `N` context, wildcard VCV starts, explicit releases, and
clearly named breath extras. Ambiguous boundary-dot, nasal-color composite,
consonant-only, and nonstandard aliases stay unresolved until an exact profile
override is supplied.

Moraic-nasal alias meanings are profile-local because UTAU banks do not share
a standard allophone notation. Profile-declared numeric aliases remain nasal
aliases rather than being stripped as numbered takes. Rows that look like
allophones but are not configured remain preserved with an actionable
diagnostic. Likewise, `* V` CV aliases are structurally distinct
`vowel_blend` candidates; `- V` remains a phrase-start candidate.

Filenames are provenance only and never provide phonetic meaning.

## Stable Identity and Provenance

A candidate ID is the SHA-256-derived identity of:

1. source-relative OTO path;
2. exact WAV field;
3. NFC alias identity;
4. duplicate occurrence ordinal.

OTO timing changes do not change the identity of an otherwise identical source
occurrence. Provenance includes relative OTO/WAV paths, OTO hash and encoding,
line and byte offset, exact source/canonical/match aliases, affixes, pitch tags,
alternative numbers, timing, and a raw-line hash. Absolute or escaping WAV paths
are not opened and produce a visible error candidate.

Generated profile and graph JSON contains no timestamp or absolute source path.
Objects and arrays have stable ordering, and JSON keys are sorted.

## Coverage

Coverage reports contain source/candidate counts, role/family/selectable counts,
unresolved IDs and rate, invalid timing, missing/outside WAVs, candidate and
alternative groups, per-mora role coverage, and missing core mora/start/CV
inventories.

Read-only full-bank validation produced:

| Bank profile | OTO entries | Effective type | Unresolved | Rate |
| --- | ---: | --- | ---: | ---: |
| Lem V4JP Civet | 12,760 | CVVC | 750 | 5.88% |
| Lem V3JP Weasel | 17,506 | VCV | 1,133 | 6.47% |
| shizuka KEY JPN | 749 | CV | 16 | 2.14% |

All entries remained traceable. CV, VCV, and CVVC candidates remained present
in every mixed real bank. No WAV escaped its source root and no referenced WAV
was missing in these validation runs. Remaining unresolved classes were visible
and concentrated in boundary-dot extras, nasal-color composites, nonstandard
romanized rows, and consonant-only start forms.

The full 12,760-entry CVVC graph was generated twice. Both 22,433,150-byte JSON
documents had SHA-256
`64a6a30ceeabdc829491fd5de9f8cfad82d9d8cd06c5def783a015b5ce07b05b`,
with identical candidate IDs and no absolute source path.

## Tests

`test_japanese_candidates.py` contains 21 deterministic tests for profile
inference and round trips, OpenUtau metadata, sharp pitch suffixes, distinct
CV/VCV/VC/release roles, mixed-bank retention, numbered alternatives, exact
overrides, unresolved preservation, disabled families, malformed timing,
small-kana and nasal contexts, wildcard starts, breath extras, stable IDs and
bytes, relative provenance, outside-WAV refusal, source-bank write guards,
source immutability, and English-module isolation.

Phase 2 tests establish structural and deterministic behavior. They do not
establish acoustic quality or naturalness.

## Phase 3 Boundary

Phase 3 may consume this graph to compile waveform slices, pitchmarks, UniSyn
indexes, a Japanese Festival phoneset/voice entry point, explicit Segments,
baseline durations, baseline F0 targets, and manual candidate overrides. Phase
2 contains none of those synthesis behaviors.
