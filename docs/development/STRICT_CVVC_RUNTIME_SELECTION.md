# Strict CVVC runtime selection

Implementation checkpoint: 2026-07-18.

## Problem

A Japanese source bank may contain CV, CVVC, and VCV OTO rows together. Before
this correction, `--bank-type cvvc` changed automatic selection costs but left
VCV candidates selectable. A generated CVVC voice could therefore choose VCV
mora recordings such as `a か`, `a の`, or `i な`.

## Runtime policy

An explicit `--bank-type cvvc` now means:

- ordinary CV and phrase-start CV rows provide mora onsets and vowels;
- `* V` rows remain optional vowel-blend fallbacks;
- recorded `V C`, `V V`, and release rows provide CVVC transitions;
- clear VCV mora rows remain in the candidate graph but are non-selectable;
- the Festival compiler repeats the family check as a defensive boundary.

CV rows are not excluded: CV plus VC is the CVVC method. “Strict CVVC” means
that a `V CV` VCV recording cannot replace those two source roles.

Two-token ASCII aliases need a configuration-aware interpretation. Under an
explicit CVVC profile, a single right-hand vowel is a VV transition and a
right-hand romaji token without a vowel letter is a consonant context. Kana or
multi-phone romaji morae remain VCV. A token explicitly declared in the bank
profile as a moraic-nasal mora alias is never reinterpreted as a consonant.

Automatic bank analysis remains permissive. The policy changes runtime
selectability only when the caller explicitly requests CVVC; no OTO row is
deleted or omitted from provenance.

## Inspectability

Candidate graph and generated runtime JSON contain `runtime_family_policy`.
Generated runtime metadata also contains
`configuration_excluded_candidate_count`, and the build report emits
`strict_runtime_family_policy_applied`. Every excluded candidate retains its
stable ID, exact alias, source OTO row, timing, and an explanatory diagnostic.

## Validation

A read-only validation used a single-pitch mixed CVVC/VCV bank and the sentence
`あの場所は怖いなぁ。` with pyopenjtalk 0.4.1. The fresh structural build had:

- 1,942 source entries, all traceable;
- 1,014 ordinary VCV candidates preserved and zero selectable;
- zero VCV families or `vcv_mora` roles in runtime alternatives;
- 461 CVVC incoming special-mora edges retained, including `/N/` and geminate
  transition alternatives rather than treating them as ordinary VCV morae;
- all spoken edges sourced and no hidden-silence fallback;
- recorded CVVC transitions for `a-n`, `o-b`, `a-sh`, `o-w`, `a-k`, and `i-n`;
- ordinary CV rows for the corresponding mora onsets;
- one generated CV bridge for `a-i`, because no CVVC-form recorded VV alias
  existed in the selected OTO scope;
- an unchanged source-bundle fingerprint before and after compilation.

The validation output is generated under the ignored project `tmp` area. It is
not installed as a voice and is not committed. This is a structural source-
selection audit, not a human naturalness claim.

## Tests

`test_japanese_candidates.py` verifies that explicit CVVC profiles keep VCV
rows traceable but non-selectable, retain CV and CVVC rows, and reinterpret
unambiguous ASCII VC/VV aliases with diagnostics.

`test_japanese_festival.py` verifies that no VCV source alias reaches runtime,
recorded VC/VV aliases remain available, excluded counts are serialized, and
the build report exposes the strict policy.

## Files changed

- `japanese_candidates.py`
- `japanese_festival.py`
- `test_japanese_candidates.py`
- `test_japanese_festival.py`
- `UNIFIED_VOICE_BUILDER.md`
- `GUIDE.md`
- `JAPANESE_PHASE2_IMPLEMENTATION.md`
- `docs/README.md`
- this implementation note
