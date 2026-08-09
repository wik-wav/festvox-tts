# Asaxi Pitch and Vocab Forge Integration

## Authoritative Data

Each Asaxi lexical note stores:

```yaml
pitch_accent: H.L
pitch_accent_class: lexical
g2p_override:
```

`g2p_override` is optional. Regular entries use the canonical rules in
`asaxi_frontend.py`.

Multiword expressions use one pitch chunk per written word:

```yaml
pitch_accent: H.L | H
pitch_accent_class: phrase
```

Each chunk contains one `H` or `L` per mora. The `|` boundaries are retained
through Vocab Forge, dictionary generation, synthesis planning, project
persistence, and Anki audio review. Expressions may contain any number of
words. Runtime matching is deterministic, left-to-right, and longest-first at
each position, so an embedded shorter expression cannot steal part of a
longer idiom and repeated non-overlapping idioms remain independent.

The visible `### Pitch Accent` block in each Markdown note is reader-facing
data generated from the same frontmatter. Vocab Forge synchronizes the block
when an entry is added or edited.

## Generated Dictionary

Build or verify the deterministic dictionary:

```powershell
py -3.14 ..\vocab_forge\build_asaxi_synthesis_dictionary.py
py -3.14 ..\vocab_forge\build_asaxi_synthesis_dictionary.py --check
```

The runtime file is `dictionaries/asaxi_lexicon.json`. It keeps canonical
phones, mora boundaries, pitch accent, typed homograph variants, source-note
provenance, multiword expression records, canonical grammar morphemes, and
attested interlinear analyses. It contains no source-bank audio and does not
write to UTAU banks.

Mora segmentation recognizes atomic G2P graphemes that contain their own
vowel nucleus. In particular, palatalized `ni` and `si` close their own mora:
`nihè` is `ni · hè`, not one fused block, while its bank-phone sequence remains
`ny i h ax`.

The orthographic dot in a nasal geminate is structural, not punctuation.
Undotted `mm` and `nn` are syllabic nasal nuclei, while dotted `m.m` and `n.n`
preserve two ordinary nasal phones across adjacent syllables. The nasal before
the dot closes the preceding block and the nasal after it begins the next:
`kem.ma` is displayed as `kem · ma` and synthesized as
`k e m | m a`. The dot itself has no duration or pitch target. The canonical
frontend owns this rule, and the standalone renderer, GUI, and Vocab Forge
wrappers delegate to it so dictionary generation and live synthesis cannot
diverge.

When a splitter correction refines older visible blocks, use:

```powershell
py -3.14 ..\vocab_forge\migrate_asaxi_pitch_accent.py `
  --refine-mora-boundaries
py -3.14 ..\vocab_forge\migrate_asaxi_pitch_accent.py `
  --apply --refine-mora-boundaries
```

The first command is a read-only preview. The guarded migration applies only
when the old visible morae exactly partition the corrected morae. It copies
each existing H/L target across the newly exposed sub-blocks and rejects
unrelated malformed patterns rather than inventing an accent.
Saved GUI projects use the same exact-refinement rule: edits on a formerly
fused block are copied to its refined blocks, while edits on later morae move
to their corrected indices instead of silently changing targets.

Dictionary generation has three explicit layers:

1. Vault lexical and standalone grammar notes are the ordinary source of G2P
   and accent data.
2. `../vocab_forge/asaxi_grammar_morphemes.toml` contains productive forms
   documented inside broader grammar chapters rather than standalone notes.
   This canonical layer is always included.
3. `../vocab_forge/asaxi_synthesis_overrides.toml` contains optional manual
   exceptions. Vocab Forge's **keep overrides** rebuild applies it; **fully
   clean (ignore overrides)** omits it without modifying or deleting the file.

The same modes are available through
`vocab_forge.py rebuild-synthesis-dictionary --override-policy
preserve|clean`. The generated `morphemes` index supports direct canonical
lookup (`-b-`) and surface lookup (`b`). The generated
`morphological_analyses` index retains every word-aligned `Morphemes` row and
its vault-relative source line. Conflicting attested analyses are preserved,
not collapsed; the current source has one such word, `nona`
(`no-na` versus `nono-a`).

## Runtime Pipeline

1. `asaxi_frontend.py` normalizes lower-case orthography and maps graphemes to
   the integrated ARPAsing-compatible phone inventory.
2. `asaxi_phone_fallback.py` compares the canonical phone plan with the
   selected generated bank's exact diphone index. Compact palatalized phones
   remain preferred when supported. An unsupported transition such as
   `hy-ao` is expanded to the fully validated `h-y`, `y-y`, `y-ao` path;
   partial repairs are refused and explicit user pronunciations remain final.
3. `asaxi_prosody.py` resolves dictionary entries, typed variants,
   leftmost-longest non-overlapping expression matches, and conservative
   transparent morphology. It compiles a bounded, cached inventory of typed
   lexical stems, explicitly bound dictionary morphemes, canonical grammar
   records, documented productive suffixes, and reversible allomorph rules.
   A documented bound form written independently can resolve through the
   morpheme surface index, but a free particle is not accepted merely because
   it happens to match the edge of an unknown word. Productive bound records
   may declare `host_lexical_types`; the canonical plural allomorphs accept
   nouns only, so a same-spelled verb cannot be mistaken for a plural stem.
4. Festival supplies the exact supported phone sequence. The provisional
   `asaxi-moraic-rules-v1` planner then allocates nonuniform phone durations
   within the canonical morae; punctuation continues to own pause timing.
5. The Asaxi plan aligns its mora targets to the complete final sentence
   timeline. `asaxi-hierarchical-log-f0-v1` realizes categorical H/L inside a
   moving, phrase-local log-F0 shape, carries value and slope across phrase
   boundaries, applies boundary-dependent reset, and creates extended boundary
   events through duration-sensitive target approximation.
6. The existing shared Festival/WSL resynthesis stages apply timing, continuous
   pitch edits, voicing, vocal-tract edits, joins, output calibration, and user
   gain.

The duration planner is documented in
[`ASAXI_DURATION_MODEL.md`](ASAXI_DURATION_MODEL.md). It is a bounded,
recording-independent fallback, not a fitted acoustic model. It replaces the
former `Duration_Default` behavior that made every phone 100 ms while
preserving the editor's current phone lengths during Re-render.

The selected-bank realization adapter is documented in
[`ASAXI_PHONE_FALLBACKS.md`](ASAXI_PHONE_FALLBACKS.md). It does not alter
canonical dictionary G2P, contextual unit choices, or source-bank data.

Before `SynthText`, the GUI overlays the resolved pronunciation of every
planned surface word onto Festival's utterance-local lexicon addenda. This
bridge is required for inferred inflections and structural spellings such as
`kem.ma`, which need not exist as standalone dictionary headwords. Explicit
project or user pronunciation overrides are applied after the inferred forms
and remain final authority.

`asaxi_prosody.realize_pitch_for_plans()` is the sentence-level handoff used by
both Generate and Re-render. Generate calls it only after phrase assembly and
all final phone durations are known. Re-render analyzes the same source text,
then calls it on the exact current editor timeline without requesting new phone
lengths. `targets_for_plans()` remains the backward-compatible target-only
wrapper.

The acoustic model is versioned in
`profiles/asaxi_pitch_model_v1.json` and documented in
[`ASAXI_PITCH_REALIZATION.md`](ASAXI_PITCH_REALIZATION.md). H/L goals are
speaker-relative semitone instructions, not fixed acoustic heights. Phrase and
later-phrase shape components are mean-centred, cumulative frequency drift is
forbidden, boundary tones occupy temporal regions, and the target tracker
creates duration-dependent undershoot. A localized raised-cosine overlay keeps
mora H/L and cents edits final at that linguistic layer without moving
unrelated targets.

Unsupported Unicode letters are rejected before Festival synthesis. They are
never silently dropped or allowed to split a word into partial lookups.

`Synthesis.asaxi_prosody` and project rows retain phrase, word, mora,
diagnostic, source-note, pitch-model, and deterministic `prosody_trace`
provenance. The trace records phrase carry/reset state, mora components,
boundary events, and every desired/realized log-F0 target. Combining
independently rendered phrases keeps each plan under its phrase index and
flattens the rendered mora timeline without applying the source-filter
phonation stage twice.

## Speech Editor Mora Controls

The Speech Parameter Editor uses the existing shared linguistic parameter
types:

- **Pitch accent** shows word-bracketed Asaxi mora blocks, the connected H/L
  contour, direct H/L overrides, and per-mora pitch;
- **Mora voicing** shows the same mora blocks with automatic/manual voicing.

Both parameters are available for Asaxi only when all of these are true:

- Festival/WSL is selected;
- the sentence language is Asaxi;
- input mode is Text;
- the selected generated voice explicitly declares Asaxi compatibility.

The language-specific editor implementation still keeps Asaxi inference and
state separate from Japanese. It uses the same fixed-width block interaction
as the Japanese editor. In **Pitch accent**, double-clicking a pitch-bearing
mora toggles H/L; its context menu and Mora tone menu provide explicit
H/L/inferred choices. Red contour points indicate manual tone overrides. In
**Mora voicing**, click, Ctrl-click, and Shift-click select one or more morae
inside a phrase before changing their shared voiced percentage. Waveform and
block selection navigate in both directions. Edits support Reset, undo/redo,
project save/load, and normal pending Re-render state. Changing the source text
invalidates stale mora-indexed overlays; changing another language merely
hides the controls and preserves the Asaxi state.

Control precedence is:

1. dictionary/morphology H/L and automatic phonation prediction;
2. explicit mora H/L, pitch, and voicing values;
3. the detailed continuous Pitch and Voicing curves, which remain final.

Pitch offsets are clamped to +/-1200 cents. Voicing uses 0..1 internally and
displays a percentage. There is no separate Breathiness control. Aspiration
and breathy lexical predictions are folded into the automatic voicing target,
while a manual Mora voicing value is final at that linguistic layer. Version
1 project breathiness overrides migrate deterministically to equivalent
voicing ceilings. The detailed dashed voicing baseline updates as soon as a
block changes, before the pending audio is re-rendered.

## Automatic Asaxi Phonation

`asaxi_phonation.py` deliberately implements only rules supported by the
current language description or made visible as provisional, editable
defaults:

- a vowel aligned between two voiceless obstruents receives reduced harmonic
  voicing (`shěso` is the documented control);
- `x` following a voiceless obstruent receives an aspiration-biased automatic
  voicing target;
- other aligned `x` realizations receive a lighter aperiodic prediction;
- the interjection `ox` receives the stronger documented breathy default,
  folded into its automatic mora-voicing value.

The H/L mora identity remains intact when its vowel devoices, matching the
prosody note: the tone is phonological and can be carried by surrounding
context. Rules never look across a pause. Non-vowel morae are displayed but
their phonation cells are unavailable rather than silently receiving a zero.
Every prediction records its reason, automatic value, final value, and whether
the user overrode it in `Synthesis.asaxi_prosody`.

The numeric strengths are conservative synthesis defaults, not measured
phonetic constants. They require speaker-specific listening review and remain
independently editable. Consonant place/voice assimilation stays in the G2P
and unit-selection layers; the vowel-phonation predictor does not reinterpret
those phones.

Implementation ownership:

- `asaxi_editing.py`: persistent, reconciled sentence edit state;
- `asaxi_phonation.py`: language prediction and shared source-filter handoff;
- `asaxi_prosody.py`: symbolic H/L planning and rendered mora alignment;
- `asaxi_pitch.py`: versioned sentence-level acoustic F0 realization;
- `festvox_gui/festvox_core.py`: Festival/WSL Asaxi rendering and metadata;
- `festvox_gui/festvox_gui.py`: contextual UI, undo/persistence, Generate and
  Re-render integration.

## Morphological Accent Inference

An inflected word does not need a duplicate lexical entry. The runtime first
finds a dictionary stem with a compatible lexical type, then applies
documented bound morphology and composes the pitch pattern from the resulting
morae.

For an interlinear source word, the runtime constrains inference to its
attested segmentations. Every segment must resolve to a lexical unit, a
documented bound form, or a licensed allomorph. A valid generic rule can still
provide richer internal structure for that same segmentation. Analyses may
nest: `gapỏbifùbiwa` is represented
as `ga- + pỏbi + [fùbi + -wa]`, retaining the plural inside the compound head.
Vowel coalescence such as `ga- + aksami -> gaksami` is aligned at the mora
level rather than rejected as a spelling mismatch. If any claimed unit is
unknown, the runtime emits `no_matching_lexical_units` and keeps regular G2P
plus the explicit default accent; it does not report the misleading
"not in dictionary" condition or invent a root.

For nominal plurals, the analyzer implements all six documented surface rules
from `10_Nominal Pluralization in Asaxi.md`:

| Rule | Analysis example | Pitch behavior |
|---|---|---|
| final `-o` replacement | `shěsa = shěso + -a` | preserve the root mora targets |
| `-a/-á + -ma` | `sháma = shá + -ma` | add an atonal plural mora |
| consonant/diphthong `+ -a` | `dăa = dă + -a` | add an atonal plural mora |
| syllabic-nasal resolution | `kama = kamm + -a` | preserve the resolved root targets |
| pure-vowel `+ -wa` | `gaviwa = gavi + -wa` | add an atonal plural mora |
| reduplicated-diphthong reduction | `pỏpa = pỏpỏ + -a` | preserve the replaced mora target |

The lexical-type gate is load-bearing: only a noun variant can license plural
morphology. A verb such as `ma` therefore cannot cause an unknown `mama` to be
silently classified as a plural.

Atonal suffixes normally contribute L. The attested monomoraic-root plateau
still applies, so lexical `shá` (`H`) plus plural `-ma` surfaces as `H.H`,
whereas disyllabic `shěso` (`H.L`) pluralizes to `shěsa` (`H.L`).

`AsaxiProsodyWord.morphemes` serializes every inferred component with its
surface, lemma, role, accent class, source note, and rule identifier. The
reading guide and schema-v2 corpus expose the same analysis. If equally ranked
analyses survive the grammatical constraints, the runtime preserves the
deterministic choice and emits `ambiguous_morphological_analysis`; it never
hides the ambiguity.

Compiled morphology inventories are cached per dictionary and individual word
analyses use a bounded cache. The richer parsing therefore does not rescan the
full lexicon for every repeated token.

The current reader audit has 34 unresolved occurrences across 20 forms, down
from 280 occurrences across 82 forms before interlinear and grammar-index
resolution. Eight are the foreign name `nana`, eight are the repeated
`hùwaśbiwa`, and the remainder are another name, undocumented lexical pieces,
or source/interlinear spelling mismatches. This stricter total is intentional:
attested rows with a missing unit are no longer replaced by an unrelated
generic segmentation. They remain visible diagnostics pending source
correction or an explicit override.

## Initial Utterance Model

The initial deterministic model now treats lexical pitch and boundary melody
as separate layers:

- ordinary statements insert onset H when the first mora is atonal, apply the
  documented one-mora downstep, and end in `L%`;
- particleless questions deaccent the utterance, while wh-questions restore
  the wh H target and end in `LH%`;
- directives preserve their attested lexical onset and end in `LH%`;
- `wő`/`ő` suppress the ordinary statement onset and produce the insistent
  `H%` tail;
- an initial `ăjo ... ,` phrase preserves the call contour and deaccents the
  following name;
- monomoraic-root plateaus remain high through atonal suffixes;
- boundary tones are serialized explicitly as `L%`, `LH%`, `H%`, or `H-` and
  drive temporally extended events in the same F0-target path used by
  synthesis.

Against the 13 dated elicitation controls, the model exactly matches all 11
controls whose H/L count is compatible with the written mora count. The
preserved shorthand readings for `ŕoŕo daohè!` and `haśùnáhè!` contain one
fewer H/L value than the orthographic/mora analysis, so the corpus reports
`mora_count_mismatch` rather than inventing an alignment. Their current
canonical model readings are `H.L.H.L.L↗` and `L.H.H.L↗`.

## Vocab Forge Review

Vocab Forge remains a standard-library server. `festvox_review.py` invokes
`vocab_review_bridge.py` with the FestVox virtual environment so the bridge can
use NumPy, PyQt, and the real Festival/WSL backend.

- The voice picker lists installed voices that explicitly support Asaxi.
- Word audio uses the complete saved headword, including idioms.
- Sentence audio uses a selected structured Asaxi example, not a translation
  field or an untyped text blob.
- **Edit in FestVox** creates a normal one-sentence v4 project and opens it in
  the GUI.
- **Finalize** refuses projects with pending Generate/Re-render state and only
  copies the reviewed, saved cache WAV into the Anki asset directory.

The bridge reads request bytes as UTF-8 explicitly. This avoids corruption by a
Windows console code page when Asaxi diacritics travel between the two Python
runtimes.

## Prosody Recording Corpus

`asaxi_prosody_corpus.py` builds the deterministic
`corpora/asaxi-prosody-v1` recording corpus from the current dictionary,
grammar examples, dated prosody elicitation appendix, and the interlinear
reader text:

```powershell
.\.venv\Scripts\python.exe .\asaxi_prosody_corpus.py
.\.venv\Scripts\python.exe .\asaxi_prosody_corpus.py --check
```

The corpus currently contains 476 prompts and 743 recommended takes, estimated
at 32.7 minutes. Its strata separate direct prosody controls, fixed
expressions, lexical citations, translated grammar examples, and natural
narrative speech.

- `reader_corpus.md` is the human-readable edition. Every prompt has plain
  Asaxi text, an English translation or explicitly labeled broader source
  context, and a table with one row per written word. The table separates
  dictionary/phrase accent, the current utterance prediction, structured
  morpheme analysis, and reference evidence. A sentence-level line states the
  reference scope and agreement.
- `recording_script.tsv` has one row per requested take and separate English,
  dictionary-pitch, predicted-utterance, morphology, boundary-tone, reference,
  and agreement columns.
- `manifest.json` retains words, morae, phones, boundaries, provenance,
  diagnostics, source hashes, and a schema-v2 `pitch_analysis` object for
  alignment and model fitting.
- `prompts.txt` is a compact speaker-facing list.
- `coverage.json` reports corpus balance and unresolved diagnostics.

H and L are mora-level targets, not measured frequencies.
`pitch_analysis.predicted` is always current model output.
`pitch_analysis.reference` preserves attested utterance evidence or lexical
dictionary evidence with an explicit authority and scope. The agreement record
compares the appropriate layer and never relabels a prediction as attested.
Natural narrative and unresolved forms remain visible as `model_hypothesis`
and `requires_linguistic_review`; generation does not hide or silently repair
them.

The corpus generator writes only inside its generated corpus directory. It
does not record audio, fit parameters, build a Festival voice, or write to a
source UTAU bank. After recording, use the stable recording IDs to align WAVs
against the manifest's word, mora, and phone sequence. Preserve the original
recording and store corrected boundaries or measured F0 as separate annotation
layers so model revisions remain reproducible.

## Reader Pitch-Accent Guides

`asaxi_reading_guide.py` turns either plain Asaxi text or a clean Markdown
reader into a deterministic, reader-facing `.md` guide:

```powershell
.\.venv\Scripts\python.exe .\asaxi_reading_guide.py `
  "path\to\reader.md" `
  --output "path\to\reader (Pitch Accent Guide).md" `
  --title "Reader title: Pitch Accent Guide"
```

Direct text and standard input are also supported:

```powershell
.\.venv\Scripts\python.exe .\asaxi_reading_guide.py `
  --text "sè no txănýj." `
  --output ".\tmp\sample-reading-guide.md"
```

The guide has one section per utterance and one table row per written word. It
shows mora segmentation, dictionary or phrase accent, the predicted
utterance-level H/L pattern, accent class, morpheme analysis, speech act, and
boundary tone.
Asaxi prose remains plain text rather than bold or italic. Native vocabulary
stays lower case, full-cap proper/borrowed terms retain their capitals, and
Polish quotation marks are retained.

Markdown mode excludes frontmatter, navigation, introductory documentation,
lists, tables, blockquotes, and fenced code. It preserves level-two and deeper
section headings. Extraction is intentionally conservative: text must contain
Asaxi-specific graphemes or have at least 70 percent dictionary coverage.

The output stores only the source filename and a SHA-256 digest, never an
absolute private path or generation timestamp. Re-run with `--check` to verify
that an existing guide still matches its source and the current prosody model.
Model warnings remain visible in collapsed callouts; the generator never
silently upgrades a prediction into attested evidence.

The current `onă gaksamipỏpỏ` reader links to its generated pitch-accent guide.
That guide contains 244 utterances across all 15 reader sections.

## Capitalized English G2P Terms

Asaxi's full-cap proper-name and borrowing convention is also an explicit
synthesis route. During Asaxi synthesis, a token such as `JOHN` is sent through
the voice's English frontend rather than lowercased and interpreted as native
Asaxi spelling. The attested project pronunciation is `jh ao n`. Lowercase
`john` retains regular Asaxi G2P, so capitalization is functional rather than
cosmetic.

The WSL backend queries the selected integrated voice's English lexicon/LTS for
other full-cap terms. The pure-Python renderer uses its existing CMU frontend.
An unresolved English term produces an actionable error instead of silently
dropping letters. A project Dictionary pronunciation remains the final
authority and bypasses automatic English G2P.

English-routed phones are grouped into deterministic syllable-sized beats for
Asaxi duration and pitch planning. They participate in the surrounding Asaxi
sentence contour without being mistaken for native Asaxi morphemes. The plan
records `capitalized_english_g2p` provenance and preserves the capitalized
surface spelling in UI metadata.

## Limitations

- Programmatic pitch suggestions are deterministic linguistic defaults, not a
  substitute for elicited lexical accent.
- Typed homographs use the dictionary default unless the caller supplies a
  lexical-type hint.
- The baseline is structural H/L planning. Continuous pitch points in the GUI
  remain the final manual authority.
- Acoustic naturalness still requires listening review per voice.

## Focused Verification

```powershell
.\.venv\Scripts\python.exe test_asaxi_frontend.py
.\.venv\Scripts\python.exe test_asaxi_editing.py
.\.venv\Scripts\python.exe test_asaxi_phonation.py
.\.venv\Scripts\python.exe test_asaxi_pitch.py
.\.venv\Scripts\python.exe test_asaxi_prosody.py
.\.venv\Scripts\python.exe test_asaxi_prosody_corpus.py
.\.venv\Scripts\python.exe test_asaxi_reading_guide.py
.\.venv\Scripts\python.exe festvox_gui\test_festvox_core.py
.\.venv\Scripts\python.exe festvox_gui\test_festvox_gui.py
.\.venv\Scripts\python.exe test_vocab_review_bridge.py
.\.venv\Scripts\python.exe ..\vocab_forge\test_festvox_review.py
```

The real-backend smoke uses an Asaxi-compatible generated voice and compares
`shěso.` with and without a +1200-cent first-mora edit. The durations and
phones must remain identical, the first-mora targets must double, and the
second-mora targets must remain unchanged. Smoke WAVs are written only to the
ignored project `tmp` directory.

The full FestVox and Vocab Forge suites must also pass after changes.
