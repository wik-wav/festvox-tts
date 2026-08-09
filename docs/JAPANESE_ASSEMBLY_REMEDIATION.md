# Japanese Assembly Remediation

This note documents the corrective Stage 2 implementation. It supersedes the
original Phase 3 OTO split assumptions where they conflict. The Japanese
linguistic frontend, English ARPAsing path, and source UTAU banks are unchanged.

## Why the old assembly failed

Human review found doubled first consonants, disjoint CVVC transitions, and
apparently missing VCs. The source candidate graph was not losing VC aliases;
the representative CVVC bank contained hundreds of them. Two downstream
assembly defects caused the audible result:

1. A phrase-start or VCV recording generated two adjacent diphones whose
   source intervals both contained the complete consonant. The left unit ended
   at preutterance while the right unit began near overlap, so the consonant
   interval was replayed.
2. Festival selected the two halves independently. For initial `/ka/`, the
   `pau-k` half could come from `-け` while `k-a` came from `-か`, because the
   first half did not retain the following vowel as selection context.

FestVox defines a diphone as the middle of one phone through the middle of the
next phone. Adjacent diphones therefore meet at one phone center, not at two
copies of a phone. See the official [FestVox tutorial](https://festvox.org/festtut/notes/festtut_2.html)
and [database construction notes](https://festvox.org/festvox-2.1/c59.html).

## Corrected source geometry

For a consonant-bearing CV, phrase-start CV, or VCV source:

```text
consonant onset  = bounded OTO overlap position
vowel boundary   = offset + preutterance
consonant center = midpoint(consonant onset, vowel boundary)
```

The two source contributions are now:

```text
V-C or pau-C: source start -> consonant onset -> consonant center
C-V:          consonant center -> vowel boundary -> stable vowel center
```

The first source end, second source start, and declared shared anchor must be
identical after metadata rounding. Release pairs use the same rule: `V-C` ends
at one consonant center and `C-pau` starts there. A standalone CV begins at its
consonant center, so it does not replay onset material already supplied by a
VC. A VCV recording supplies both edges without an added plain-CV consonant.

Canonical `V-cl-C` remains a deliberate, editable geminate closure. The
language-neutral special-phone resolver sources it as `V-C-C`: `V-C` provides
the anticipatory VC edge, a bounded generated `C-C` unit holds the consonant
without replaying its release, and the following `C-V` supplies the sole
release. An OTO alias named `cl` does not change this policy.

## Explicit bank-type selection

The selected configuration controls automatic source priority:

- CV: ordinary and phrase-start CV are primary;
- VCV: VCV and phrase-start sources are primary;
- CVVC: both VC/release and ordinary/phrase-start CV are primary.

CV material is therefore not penalized as foreign inside CVVC. Secondary
families remain selectable fallback evidence, but receive a large enough cost
that they cannot outrank a valid primary source solely because of a context
bonus. Phrase-start and VCV left halves retain their expected following vowel;
both halves consequently select the same stable candidate ID.

Phrase-start aliases are a hard edge constraint, not merely a preference.
`- V` and `- CV` sources are eligible at a real pause edge and are excluded
from medial diphones and generated-bridge source pools. This prevents an alias
such as `- め`, whose leading silence is intentional, from replacing ordinary
medial `め`.

Some CV banks use `* V` for a different purpose. These rows are classified as
`vowel_blend`, never as phrase starts. The OTO offset is the audible vowel
onset, preutterance identifies the stable join region, and overlap supplies the
preferred crossfade (bounded to 4-60 ms). A `* V` row may provide the incoming
half of a generated vowel transition when no real VCV/VC transition exists;
an exact recorded transition always wins.

A separate read-only CV fixture supplied during remediation contains 148 OTO
rows and five paired `* V` aliases. All five classify as `vowel_blend`; their
overlap/preutterance ratios are 1.923-2.000 and their offsets mark the audible
vowel onset. This validates the role and geometry against a real convention
without making that convention mandatory for banks that do not define it.

## Bank-specific moraic nasal allophones

UTAU aliases for Japanese moraic `/N/` are not standardized. Their meaning is
therefore explicit profile data rather than a universal `n`/`nn`/`ng` rule.
Each `moraic_nasal_allophones` entry declares:

- exact mora aliases, including numbered aliases that must not become takes;
- exact context aliases used by VC rows;
- canonical following phones that select the realization;
- at most one default for phrase-final or otherwise unmatched contexts.

The profile used for the audited Phascogale configuration defines coronal,
labial, velar, and bank-specific uvular groups. This makes both sides of `/N/`
agree: for example, a labial context selects the configured labial `V-N`
source and the configured labial `N-C` source. Unconfigured allophone-like
aliases remain traceable with a diagnostic instead of being guessed or
dropped. Manual per-occurrence unit choices remain final.

## Missing-transition bridge

A pure CV bank has no recorded `V-C` unit. Festival previously substituted its
default `pau-pau` silence for such a missing edge. The compiler now creates a
bounded audible bridge in generated output:

1. take at most 80 ms from a stable left-vowel or moraic-nasal region;
2. take at most 80 ms from the next phone's recorded onset-to-center region;
3. apply an 8 ms linear crossfade;
4. index the crossfade center as the phone boundary;
5. generate pitchmarks for this copied/generated WAV like every other unit.

The bridge never writes to the source bank. Its ID and filename are
content-derived, repeated builds are byte-identical, and runtime metadata lists
both source candidates, aliases, WAVs, OTO rows, timing landmarks, and slices.
Its role is `generated_cv_bridge`, and every use carries a visible fallback
diagnostic. A bridge is not described as a recorded VC or as quality-equivalent
to one.

## Source-contribution API

`japanese_assembly.py` provides:

```text
select_automatic_choice(choices, outer_left, outer_right)
create_source_contribution_plan(plan, runtime_metadata, selected_units=...)
```

`JapaneseSourceContributionPlan` contains one row per canonical phone edge:

```text
linguistic phones and mora indices
target-time start, boundary, and end
selected candidate ID, role, and family
source alias, WAV, OTO file, and OTO line
source slice and OTO timing landmarks
shared phone-center anchor
all alternative candidate IDs
selection reason and fallback reason
all source components for a generated bridge
```

Diagnostics make these conditions explicit:

- mismatched shared anchor;
- overlapping/doubled paired consonant source;
- source gap between paired halves;
- different candidates selected for one contextual pair;
- generated or cross-configuration fallback;
- spoken edge that would use hidden default silence.

`japanese_quality.py` now uses the same automatic selector as Festival instead
of assuming the first metadata row. Source-contribution reports can also be
resolved from the actual unit names returned by Festival.

Current Japanese Festival voices perform automatic role, bridge, and
allophone selection in their generated UniSyn hook. The GUI backend sends only
explicit user overrides for these voices; it does not pre-pin every edge with
the English/ARPAsing contextual take selector. English behavior is unchanged.

## Tests and comparison corpus

`test_japanese_assembly.py` builds isolated deterministic CV, VCV, and CVVC
banks. It covers:

```text
あ  か  あか  かか  あき  かさ  さか  あん  あった  きゃ  きょう
```

The suite proves pair-coherent selection, exact shared centers, visible CV
fallback, no hidden spoken-edge silence, rejection of the former overlapping
geometry, deterministic metadata, byte-identical bridge WAVs, and identical
complete generated trees across different destination directories. Generated
Scheme resolves its voice root from Festival's backend-provided load path, so
it remains relocatable without embedding a machine-specific output path.

The corrective follow-up gate passed 259/259 tests with the optional
`pyopenjtalk 0.4.1` frontend and offscreen Qt enabled. The one sandbox-gated
WSL integration test was then run separately against real Festival and passed,
including explicit timing, F0 targets, and a per-occurrence manual unit
override.

`japanese_assembly_listening.py` renders an ignored 18-item human-review corpus
and writes `<id>.contributions.json` beside every WAV. It covers vowels,
ordinary CV, VCV and CVVC transitions, moraic nasal, geminate, palatalized and
long-vowel forms, exact `えい`, four configured nasal contexts, an Open JTalk
kanji vowel sequence, phrase boundaries, statement, question, and an accent
carrier. The manifest always states:

```json
"acoustic_naturalness_verified": false
```

Example:

```text
py -3.14 japanese_assembly_listening.py \
  <generated-voice> <ignored-listening-output> --wsl-distro Ubuntu
```

Representative real CV, VCV, and CVVC configurations each rendered 12/12
items with no skipped diphones, no hidden silence, no structural errors, and no
join rated `poor` by the automated metric. Automated checks do not replace a
new human listening pass, especially for generated CV bridges and joins marked
`review`.
