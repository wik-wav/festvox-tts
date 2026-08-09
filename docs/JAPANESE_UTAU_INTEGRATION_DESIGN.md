# Japanese UTAU Voicebank Integration Design

Status: approved architecture implemented through Phase 5: read-only analysis,
canonical text frontends, source candidate compilation, isolated
Festival/UniSyn synthesis, GUI editing, acoustic diagnostics, optional baseline
experiments, dynamic pitch/color routing, and release checks. Human listening
and redistribution approval remain explicit external gates.

Evidence labels used below:

- **Repository fact**: verified in this working tree or a supplied OTO file.
- **External fact**: supported by a primary source in section 24.
- **Engineering inference**: a proposed design based on those facts.
- **Open question**: requires listening tests, licensing review, or more data.

## 1. Current Repository Findings

- **Repository fact:** `utau2festvox.py` is a production ARPAsing-oriented
  converter. `PHONEME_MAP`, numbered-take handling, directional OTO context,
  `character.yaml`, `prefix.map`, and voice-color support are all designed around
  the existing English/Asaxi inventory. An unknown ASCII token may pass through;
  a kana alias cannot.
- **Repository fact:** the old metadata reader tries UTF-8, CP932, Shift-JIS, and
  finally Latin-1. Latin-1 cannot fail, so it can hide mojibake. Changing that
  shared function would change the English path and is outside this first phase.
- **Repository fact:** `build_festival_voice.py` generates a phoneset, Festival
  Scheme, an EST UniSyn index, pitchmarks, alternatives, provenance, and separate
  base/English voice entry points. Existing generated metadata has no top-level
  schema version.
- **Repository fact:** the GUI's Festival backend can synthesize explicit Segment
  durations and explicit Target F0 points, capture Festival's generated segment
  and F0 relations, and re-render edits through UniSyn. This is already the right
  waveform boundary for Japanese.
- **Repository fact:** current Japanese text handling is a handmade kana/romaji
  path in `synth_diphone.py`. It has no kanji morphology, word boundaries, accent
  phrases, accent nucleus, or Japanese duration model. The GUI sends its phones
  through the explicit-segment path.
- **Repository fact:** the three supplied CP932 OTO files classify from aliases as
  follows. WAV filenames were not used:

  | Supplied reference | Entries | Alias-family evidence | Result |
  |---|---:|---|---|
  | `Lem_V4JP_Civet/3_E3` | 1,271 | CV 410, VCV 151, CVVC 610 | CVVC |
  | `Lem_V3JP_Weasel/F3` | 1,942 | CV 334, VCV 1,040, CVVC 138 | VCV |
  | `shizuka_KEY_JPN/F3` | 187 | CV 132, VCV 35, CVVC 7 | CV |

- **Repository fact:** the CV reference contains five malformed numeric OTO
  fields. They are now retained with precise diagnostics instead of being
  discarded.
- **Repository fact:** `japanese_utau.py` now implements a read-only phase-0
  analyzer. It does not call the converter, copy audio, or write to the bank.

## 2. Existing English Path and Compatibility Boundary

- **Engineering inference:** treat the ARPAsing converter, its `PHONEME_MAP`,
  generated English phoneset, context selector, and English Festival entry point
  as a compatibility boundary. Japanese code must not add language conditionals
  inside English token mapping.
- **Engineering inference:** route Japanese only when a versioned bank profile
  explicitly says `language: ja`. Auto-detection may propose a profile, but must
  not silently reinterpret an existing English build.
- **Engineering inference:** source aliases, source takes, slice times, and manual
  per-occurrence choices remain immutable provenance in both paths.
- **Open question:** common raw OTO and metadata loading can eventually move to a
  shared module, but only after English golden-output tests prove identical JSON,
  EST indexes, Scheme, and WAV slices before and after the move.

## 3. Japanese Voicebank Configurations

- **External fact:** OpenUtau ships distinct Japanese VCV and CVVC phonemizers and
  treats voice color, tone shift, alternatives, Unicode normalization, and alias
  fallback as separate concerns.
- **Repository fact:** real banks are not pure. The supplied CV bank has optional
  vowel-plus-kana aliases, the VCV bank has CV and VC fallbacks, and the CVVC bank
  has some VCV material.
- **Engineering inference:** configuration is therefore a bank-level default plus
  per-alias evidence, not a global cast:
  - CV: standalone mora recordings such as `か`, plus starts such as `- か`.
  - VCV: previous vowel or mora-nasal plus a mora, such as `a か`.
  - CVVC: mora CV recordings plus outgoing VC transitions such as `a k`.
  - Mixed: no family dominates, or a profile intentionally combines families.
  - Extra: breaths, closures, releases, glottal events, and unknown aliases kept
    outside the canonical phone inventory until explicitly mapped.
- **Engineering inference:** `character.yaml` takes priority over `prefix.map`,
  followed by explicit user affixes. `presamp.ini` is useful evidence for a mixed
  bank but must never override exact OTO aliases silently.

## 4. Encoding and Alias Parsing

- **Repository fact:** all three supplied Japanese OTOs fail strict UTF-8 and pass
  strict CP932.
- **External fact:** Unicode NFC preserves compatibility distinctions while
  canonicalizing equivalent sequences. NFKC also folds compatibility forms, such
  as half-width kana, and must not be used blindly as the stored identity.
- **Engineering inference:** retain four separate values:
  1. original bytes and SHA-256;
  2. exact decoded source alias;
  3. NFC canonical alias for stable identity;
  4. NFKC comparison key for matching only.
- **Implemented:** auto-decoding tries strict UTF-8 and strict CP932. A UTF-8 BOM
  is recognized. If both decodes are plausible but differ, the report marks the
  choice ambiguous and asks for `--encoding`. Latin-1 and replacement characters
  are never used.
- **Implemented:** every diagnostic contains source path, line, and byte offset.
  Structurally recognizable aliases survive blank or invalid timing numbers.
- **Engineering inference:** declared pitch, prefix, suffix, and voice-color
  markers are removed iteratively for analysis, so both `ayPE3` and `ayE3P` work.
  The exact source alias remains untouched.

## 5. Canonical Japanese Linguistic Model

- **Engineering inference:** add a language-neutral utterance carrier with a
  Japanese layer, rather than forcing Japanese into ARPAbet:

  ```text
  JapaneseUtterance
    phrases[]
      accent_phrases[]
        moras[]
          surface, reading, consonant?, vowel, special_mora?, devoiced?
          canonical_phones[]
        accent_nucleus?, interrogative?, boundary_strength
    canonical_phones[]
    phone_durations[]
    baseline_f0[]
  ```

- **Engineering inference:** canonical phones describe linguistic intent. Source
  units describe available recordings. They must be linked by candidates rather
  than sharing one string namespace.
- **Engineering inference:** explicitly represent long vowels, mora nasal `ん`,
  geminate `っ`, palatalized morae, devoiced vowels, punctuation, and pauses.
- **Open question:** the canonical phone symbols should be finalized against Open
  JTalk labels and the supplied banks before generated phoneset schema version 2
  is frozen.

## 6. CV Conversion Design

- **Engineering inference:** map each canonical mora to one or more CV candidates,
  with separate start candidates when available. OTO preutterance remains the
  consonant-to-vowel alignment point; overlap and consonant values remain source
  timing evidence.
- **Engineering inference:** CV alone does not record every inter-mora transition.
  Build half-phone boundaries from the CV slice and use a short, bounded
  pitch-synchronous overlap at the preceding vowel tail. Do not stretch a stop or
  closure merely to fill a mora duration.
- **Engineering inference:** phrase-initial aliases (`- か`, `* か`) outrank plain
  `か` only at a real phrase edge. Plain CV remains the fallback.
- **Open question:** acceptable crossfade and stable-vowel windows differ by bank;
  they require waveform inspection and listening tests, not only OTO arithmetic.

## 7. VCV Conversion Design

- **External fact:** OpenUtau's VCV reference implementation looks up the previous
  lyric's trailing vowel, then tries `vowel + current`, `* + current`, current, and
  `- + current`, while respecting mapped tone, color, and alternatives.
- **Engineering inference:** use the same semantic order, but convert each hit into
  a source-unit candidate with explicit provenance and cost. Do not collapse two
  source aliases that normalize to the same mora pair.
- **Engineering inference:** the previous context is a vowel category or mora
  nasal, not the literal previous WAV name. Phrase starts use explicit start/CV
  candidates. Phrase-internal VCV is preferred because it contains the recorded
  transition.
- **Engineering inference:** split each candidate around the OTO preutterance so
  UniSyn can align the onset and vowel while the GUI still edits phone durations.
- **Open question:** bank-specific handling of `ん`, vowel devoicing, and release
  aliases needs profiles and acoustic validation.

## 8. CVVC Conversion Design

- **Engineering inference:** realize a mora as a CV nucleus candidate plus, when
  needed, an outgoing VC transition candidate toward the next consonant. The VC
  candidate supplies the recorded vowel-to-consonant movement; the next CV
  supplies the consonant release and following vowel.
- **Engineering inference:** maintain one canonical consonant boundary even when
  two source units contribute to it. This avoids doubling stops or exposing two
  GUI phonemes for one linguistic consonant.
- **Engineering inference:** release forms such as `u h-` are separate candidates,
  used only where the profile declares their role. A trailing hyphen is evidence,
  not enough by itself to infer semantics globally.
- **Engineering inference:** CV-only fallback remains available for missing VC
  transitions, with an explicit quality warning in coverage reports.

## 9. Mixed-Bank Strategy

- **Repository fact:** all supplied examples contain evidence from more than one
  family.
- **Engineering inference:** compile an alias graph whose nodes are canonical
  mora/transition roles and whose edges are source candidates. A bank-level type
  sets preference weights; it does not delete other valid candidates.
- **Engineering inference:** precedence is exact profile override, exact contextual
  unit, configuration-preferred candidate, compatible fallback, then an explicit
  missing-unit diagnostic. Unknown aliases remain in the report and can be mapped
  later.
- **Engineering inference:** expose a coverage matrix before building so the user
  can see missing morae, VC transitions, starts, releases, and unresolved extras.

## 10. Japanese Text Frontend Options

1. **Festival-native Japanese frontend.** **External fact:** Festival provides the
   framework for a language's phoneset, lexicon, phrasing, duration, and intonation,
   but the inspected manual does not provide a maintained modern Japanese frontend.
   **Inference:** implementing morphology and accent in SIOD would be high-risk.
2. **Current kana/romaji rules.** **Repository fact:** dependency-free and useful
   as a deterministic fallback, but it cannot analyze kanji or lexical accent.
3. **MeCab directly.** **External fact:** MeCab provides morphology, readings, and
   configurable dictionary encodings, but dictionary data is separate. **Inference:**
   it still leaves NJD pronunciation transformations and accent interpretation to us.
4. **Open JTalk or pyopenjtalk frontend.** **External fact:** pyopenjtalk exposes
   G2P and full-context labels on Windows, Linux, and macOS. Those labels include
   phone and linguistic/prosodic context. **Recommendation:** primary optional
   frontend.
5. **HTS/Open JTalk waveform output.** **External fact:** HTS jointly models
   spectrum, excitation/log-F0, and duration. **Inference:** useful as a reference
   or trajectory source, but using its waveform would replace the selected UTAU
   voice and bypass the existing editor.

**Recommended primary architecture:** optional pyopenjtalk/Open JTalk analysis,
converted into the canonical utterance, followed by existing Festival/UniSyn UTAU
unit rendering. **Fallback:** current kana/romaji parsing plus documented rule-based
accent templates. Open JTalk must not become mandatory until packaging, dictionary,
and licensing behavior is settled.

## 11. Prosody Options

- **External fact:** Fujisaki and Hirose model Japanese log-F0 as phrase and accent
  components controlled by a small set of linguistic commands. Standard Japanese
  word accent permits at most one downward transition within a word.
- **External fact:** Open JTalk can emit full-context labels; HTS can emit duration
  and log-F0 trajectories from those labels.
- **Engineering inference:** initial high-quality order:
  1. extract words, morae, accent phrases, nucleus, interrogative state, and
     pronunciation from Open JTalk labels;
  2. create deterministic editable accent templates with phrase declination,
     initial rise, nucleus fall, downstep, boundary lowering, and interrogative rise;
  3. optionally compare or seed durations/F0 from HTS later;
  4. preserve the user's continuous F0 override as the final authority.
- **Engineering inference:** phrase and accent templates are easier to edit and
  debug than importing a dense HTS contour immediately. HTS-derived trajectories
  can be added as another baseline without changing the waveform backend.
- **Open question:** Tokyo accent is a practical default, not a universal Japanese
  prosody model. Dialect and expressive speech require explicit profiles.

## 12. Recommended Architecture

```text
Japanese text
  -> JapaneseFrontend (Open JTalk optional; kana fallback)
  -> CanonicalJapaneseUtterance
  -> editable mora/accent/duration/F0 model
  -> JapaneseUnitSelector + bank profile + source provenance
  -> explicit Festival Segment, Target, and unit-choice relations
  -> existing UniSyn TD-PSOLA UTAU waveform rendering
  -> existing waveform, timing, pitch, and per-occurrence recording editors
```

- **Engineering inference:** this is stronger than a separate Open JTalk waveform
  backend because it preserves the chosen UTAU speaker and all existing GUI edits.
- **Engineering inference:** it is stronger than a pure Festival frontend because
  Open JTalk already supplies Japanese morphology and accent context.
- **Engineering inference:** keep Open JTalk behind a small adapter and retain the
  kana fallback so the GUI can load and edit a Japanese project without the optional
  dependency.

## 13. User Pitch and Accent Controls

- **Repository fact:** the GUI already separates generated ground-truth F0 from an
  editable working contour and re-renders changes through UniSyn.
- **Engineering inference:** add Japanese overlays, not a second pitch editor:
  mora grid, accent-phrase brackets, nucleus marker, phrase command/rise, accent
  fall, downstep, interrogative rise, and optional devoicing state.
- **Engineering inference:** control precedence is generated baseline, accent-block
  edits, continuous F0 edits, then bounded safety clamping. Manual points should
  survive a bank candidate change when timing can be remapped.
- **Engineering inference:** show HTS/Open JTalk estimates as a baseline layer only;
  they must never overwrite user points silently.

## 14. Timing Model

- **Repository fact:** explicit phone durations and editable boundaries already
  exist. The two-pause phrase representation separates the coda transition from
  the freely sized inter-phrase silence.
- **Engineering inference:** preserve that representation for Japanese. A phrase
  boundary is final phone -> first `pau` for acoustic closure, followed by a second
  independently resizable `pau` for silence duration.
- **Engineering inference:** represent mora duration separately from internal phone
  allocation. Long vowels, mora nasal, geminate closure, palatalized onsets, and
  devoiced vowels need dedicated timing rules.
- **Engineering inference:** use Open JTalk/HTS durations only as an optional
  baseline. OTO preutterance, overlap, consonant, and cutoff constrain source-unit
  alignment; they are not linguistic target durations.

## 15. Unit Selection Model

- **Engineering inference:** candidates receive a deterministic cost composed of:
  exact canonical role, left/right mora context, phrase-edge compatibility, bank
  configuration, source color, source pitch/subbank, declared alternative, timing
  validity, and acoustic boundary quality.
- **Repository fact:** existing English selection already preserves alternatives,
  context, manual overrides, and source inspection. Japanese should use the same UI
  contract with a language-specific cost function.
- **Engineering inference:** filenames are provenance only. They must never supply
  phonetic meaning. Exact OTO alias and profile mappings are authoritative.
- **Engineering inference:** per-occurrence user choice remains final and is stored
  by stable source candidate ID, not list index or phone label.

## 16. Metadata and Configuration Schemas

- **Implemented:** analyzer reports use `schema_version: 1` and contain source
  hashes, encoding/confidence, exact aliases, canonical/match forms, removed affixes,
  timing validity, role evidence, family composition, examples, and diagnostics.
- **Engineering inference:** a future explicit bank profile should be JSON so phase
  one adds no YAML dependency, while still reading OpenUtau's `character.yaml`:

  ```json
  {
    "schema_version": 1,
    "language": "ja",
    "bank_configuration": "auto",
    "encoding_overrides": {"F3/oto.ini": "cp932"},
    "alias_prefixes": [],
    "alias_suffixes": ["P"],
    "voice_color": null,
    "alias_overrides": {},
    "unknown_alias_policy": "preserve"
  }
  ```

- **Engineering inference:** generated voice metadata version 2 should separate
  canonical phones, source units, candidate graph, provenance, configuration, and
  frontend requirements. Version 1 remains readable through an adapter.
- **Engineering inference:** store source-relative paths plus hashes. Never store a
  writable operation pointed at the source bank.

## 17. Festival/UniSyn Integration

- **External fact:** UniSyn supports waveform, pitchmark, and index data, runs hooks
  before unit selection, permits per-segment diphone-name features, converts F0 to
  pitchmarks, constructs a unit stream, and performs pitch-synchronous synthesis.
- **Repository fact:** the current backend already sends explicit segments, target
  points, and unit overrides and receives rendered waveform/relations.
- **Engineering inference:** Japanese should end at that existing API boundary.
  Add Scheme hooks that map canonical Japanese segment context to chosen source
  units; do not add a new finished-audio pitch shifter.
- **Engineering inference:** keep the generated Japanese phoneset and voice entry
  point separate from `_en`. No Japanese text should pass through CMU lexicon,
  ARPAbet, or English duration/intonation trees.

## 18. Pure-Python Backend Impact

- **Repository fact:** the pure-Python backend concatenates indexed units and lacks
  Festival's full pitch-synchronous F0 regeneration.
- **Engineering inference:** it should consume the same canonical utterance and
  selected candidate IDs so projects remain inspectable without WSL.
- **Engineering inference:** its Japanese fallback may initially use conservative
  CV/VCV/CVVC concatenation and edited durations, while clearly labeling that
  pitch/prosody quality is below Festival/UniSyn.
- **Open question:** adding an independent PSOLA engine would duplicate difficult
  signal-processing behavior and is not justified in the first Japanese phases.

## 19. Module and API Changes

- **Implemented:** new standalone `japanese_utau.py` API:
  - `decode_text_file()`
  - `normalize_alias()` and `classify_alias()`
  - `parse_oto_file()`
  - `analyze_bank()`
  - `write_report()` with a source-bank write guard
- **Implemented:** CLI usage:

  ```powershell
  python japanese_utau.py "D:\UTAU\voice\Bank"
  python japanese_utau.py "D:\UTAU\voice\Bank\oto.ini" --encoding cp932
  python japanese_utau.py "D:\UTAU\voice\Bank" --json
  ```

- **Repository fact:** no existing converter, builder, core, GUI, or generated
  Scheme API changed in phase 0.
- **Engineering inference:** later modules should be `japanese_frontend.py`,
  `japanese_units.py`, and a small builder adapter. Avoid scattering `lang == ja`
  branches through the English converter.

## 20. Migration and Backward Compatibility

- **Engineering inference:** old English/Asaxi generated voices and project files
  load exactly as now. Their missing metadata schema implies version 1.
- **Engineering inference:** a Japanese bank is enabled only after analysis and an
  explicit Japanese profile. Auto-detection produces a proposal and diagnostics,
  never an in-place migration.
- **Engineering inference:** generated Japanese voice version 2 can be rebuilt from
  the read-only source. Rebuilds may replace generated output only after the normal
  output-path checks; they never modify the UTAU bank.
- **Open question:** schema adapters should be frozen only after one CV, one VCV,
  one CVVC, and one intentionally mixed bank complete end-to-end builds.

## 21. Testing Strategy

- **Implemented:** 13 synthetic tests cover CP932, UTF-8 BOM, ASCII ambiguity,
  hard decode failure, exact alias preservation, NFC/NFKC separation, malformed
  timing retention, affix order, CV/VCV/CVVC detection, explicit overrides, no
  source writes, and report-path refusal.
- **Implemented:** read-only probes against the three supplied banks produce the
  expected CV, VCV, and CVVC bank classifications.
- **Engineering inference:** next test layers:
  1. golden Open JTalk labels -> canonical mora/accent structures;
  2. synthetic profile and coverage graph fixtures;
  3. generated metadata/EST/Scheme snapshots;
  4. Festival integration tests for duration/F0/unit relations;
  5. GUI persistence, undo, re-render, and per-occurrence override tests;
  6. listening sets for starts, V-V, V-C, geminates, mora nasal, devoicing,
     long vowels, phrase boundaries, accent contrasts, and questions.
- **Engineering inference:** test success proves deterministic behavior and coverage,
  not naturalness. Listening results must remain a separate quality gate.

## 22. Phased Implementation Plan

1. **Phase 0 - complete:** repository trace, researched design, strict read-only OTO
   analyzer, diagnostics, bank composition report, synthetic tests, and real-bank
   validation.
2. **Phase 1 - complete:** optional Open JTalk adapter, canonical Japanese
   utterance, named label parser, kana fallback adapter, and golden frontend
   tests. No waveform changes. See `JAPANESE_PHASE1_IMPLEMENTATION.md`.
3. **Phase 2 - complete:** profile serializer, alias override workflow, coverage
   graph, CV/VCV/CVVC candidate compiler, stable candidate IDs, source safety,
   real-bank validation, and deterministic provisional metadata. See
   `JAPANESE_PHASE2_IMPLEMENTATION.md`.
4. **Phase 3 - complete:** separate Japanese Festival phoneset and voice entry
   point, versioned candidate-to-unit compiler, UniSyn selection hooks,
   explicit duration/F0 baseline, stable manual overrides, real WSL synthesis,
   and an ignored 12-example listening corpus. See
   `JAPANESE_PHASE3_IMPLEMENTATION.md`. Acoustic naturalness is unverified.
5. **Phase 4 - complete:** GUI bank-analysis preview, exact unknown-alias
   resolution, mora/accent overlays, phrase/question controls, per-mora pitch,
   persistence, migration, undo/redo, and explicit rebuild/re-render routing.
   See `JAPANESE_PHASE4_IMPLEMENTATION.md`.
6. **Phase 5 - complete:** generated-copy acoustic boundary metrics, optional
   Open-JTalk-label and external HTS-JSON baselines, deterministic multipitch
   and voice-color routing, expanded quality corpus, content-addressed cache,
   and documented release/licensing checks. See
   `JAPANESE_PHASE5_IMPLEMENTATION.md`.

Each phase keeps the English golden suite green and can be reverted without
rewriting source voicebanks.

## 23. Risks, Licensing, and Open Questions

- **External fact:** pyopenjtalk is MIT; Open JTalk is Modified BSD; Open JTalk's
  own notice says MeCab, mecab-naist-jdic, and Open JTalk have separate COPYING
  files. MeCab itself offers GPL/LGPL/BSD choices and ships no dictionary in its
  source package.
- **Engineering inference:** do not bundle an Open JTalk dictionary or HTS voice
  until its exact archive COPYING file is recorded in the application and release
  package. An optional user-installed frontend has lower immediate licensing risk.
- **External fact:** OpenUtau is MIT. Its phonemizers are valuable behavioral
  references, but copying code is unnecessary; this design reimplements only the
  observed alias semantics and cites the source.
- **Open question:** every UTAU bank has its own redistribution and derivative-use
  terms. The builder should record, not invent, bank license metadata.
- **Open question:** OTO timing is not sufficient to guarantee a clean join,
  especially for CV banks. Acoustic checks and listening remain mandatory.
- **Open question:** Open JTalk accent output, dictionary version, names, foreign
  words, and dialect may not match user intent. The GUI needs user overrides.
- **Implemented conservatively:** dynamic multipitch considers declared pitch
  metadata and target F0; voice color is an exact declared-subbank filter. Both
  are opt-in, deterministic, and subordinate to manual occurrence choices.
- **Open question:** unknown aliases must remain visible and lossless. A wrong
  automatic classification is more harmful than an explicit unresolved item.

## 24. Primary Sources

- [Festival manual: Voices](https://www.cstr.ed.ac.uk/projects/festival/manual/festival_24.html)
  - voice boundary, phonesets, lexicons, phrasing, intonation, and duration.
- [Festival manual: UniSyn synthesizer](https://www.cstr.ed.ac.uk/projects/festival/manual/festival_20.html)
  - database structure, pitchmarks, indexes, hooks, and per-segment unit names.
- [Festival manual: function list](https://www.cstr.ed.ac.uk/projects/festival/manual/festival_34.html)
  - `us_f0_to_pitchmarks`, `us_get_synthesis`, and pitch-synchronous synthesis.
- [Open JTalk official README](https://open-jtalk.sourceforge.net/readme_open_jtalk.php)
  - purpose, Modified BSD terms, and separate component license notices.
- [pyopenjtalk repository and README](https://github.com/r9y9/pyopenjtalk)
  - supported platforms, G2P, full-context labels, optional marine accent model,
  and component licenses.
- [MeCab official documentation](https://taku910.github.io/mecab/)
  - morphology, dictionary separation, encoding configuration, and license choices.
- [OpenUtau Japanese VCV phonemizer](https://github.com/openutau/OpenUtau/blob/master/OpenUtau.Plugin.Builtin/JapaneseVCVPhonemizer.cs)
  - Unicode normalization, VCV fallback order, tone/color/alternative lookup.
- [OpenUtau repository](https://github.com/openutau/OpenUtau)
  - Japanese CVVC/VCV phonemizer availability, encoding support, and MIT license.
- [Unicode Standard Annex #15](https://www.unicode.org/reports/tr15/)
  - NFC/NFKC semantics and compatibility-folding cautions.
- [Fujisaki and Hirose, Japanese sentence F0](https://doi.org/10.1250/ast.5.233)
  - phrase and accent components in log-F0.
- [Fujisaki and Sudo, Japanese word accent](https://doi.org/10.20697/jasj.27.9_445)
  - initial transition and at most one downward transition in a standard Japanese
  word accent.
- [Zen et al., HTS overview](https://www.cs.cmu.edu/~awb/papers/apsipa2009/zen_APSIPA2009.pdf)
  - context labels and joint spectrum, excitation/log-F0, and duration modeling.
