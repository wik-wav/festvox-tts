# UTAU → FestVox diphone conversion + diphone synthesis

For a dated, implementation-level snapshot of the Windows GUI and its complete
text/phoneme-to-waveform path, see `docs/GUI_STATE_AND_SYNTHESIS.txt`.
The maintained documentation map is [docs/README.md](docs/README.md); it
separates current operating guides from historical phase and prompt reports.
The dependency-free English phone syllabifier, shared opt-in syllable/mora
continuous-curve overlay, serialization contract, and current lexical-boundary
limitation are documented in
[ENGLISH_SYLLABIFICATION.md](docs/ENGLISH_SYLLABIFICATION.md).
The long-session resource audit, lifecycle fixes, soak command, and paired
tracemalloc methodology are documented in
[docs/development/PROMPT0_LONG_TERM_PERFORMANCE.md](docs/development/PROMPT0_LONG_TERM_PERFORMANCE.md).
The synthesis profiler, direct-PCM path, bounded cache architecture, cache
menu, and matched before/after benchmark are documented in
[docs/development/PROMPT0A_SYNTHESIS_EFFICIENCY.md](docs/development/PROMPT0A_SYNTHESIS_EFFICIENCY.md).
For the primary Windows voice-build command, configuration identity, output
protection, and legacy-registration migration, see
[UNIFIED_VOICE_BUILDER.md](docs/UNIFIED_VOICE_BUILDER.md).
For the native millisecond-authoritative crossover renderer, direct
per-occurrence join editing, measured discontinuity diagnostics, and the exact
stock-Festival **Legacy joins** fault, see
[JOIN_SYNTHESIS.md](docs/JOIN_SYNTHESIS.md).
For dictionary-driven Asaxi G2P, lexical and multiword pitch accent, strict
grapheme handling, Vocab Forge's reviewed Festival/WSL asset workflow, and the
standalone `asaxi_reading_guide.py` Markdown pitch-guide generator, see
[ASAXI_PITCH_INTEGRATION.md](docs/ASAXI_PITCH_INTEGRATION.md).
The versioned sentence-level log-F0 realization, boundary-state carry,
duration-sensitive target approximation, deterministic trace, and no-drift
constraint are documented in
[ASAXI_PITCH_REALIZATION.md](docs/ASAXI_PITCH_REALIZATION.md).
The provisional nonuniform mora/phone timing rules, live WSL handoff,
diagnostic metadata, evidence, tests, and recording limitations are documented
in [ASAXI_DURATION_MODEL.md](docs/ASAXI_DURATION_MODEL.md).
Inventory-aware Asaxi realization fallbacks for unsupported compact
palatalized transitions are documented in
[ASAXI_PHONE_FALLBACKS.md](docs/ASAXI_PHONE_FALLBACKS.md).

For the researched Japanese CV/VCV/CVVC architecture, compatibility boundary,
and phased implementation plan, see
[JAPANESE_UTAU_INTEGRATION_DESIGN.md](docs/JAPANESE_UTAU_INTEGRATION_DESIGN.md).
Phase 0 is available as the read-only `japanese_utau.py` analyzer; it reports
strict UTF-8/CP932 decoding, malformed OTO data, and mixed alias-family evidence
without building audio or changing the source bank.
Phase 1 is available through the isolated `japanese_frontend.py` API, with a
typed canonical utterance, dependency-free kana/romaji fallback, and optional
pyopenjtalk full-context analysis. See
[JAPANESE_PHASE1_IMPLEMENTATION.md](docs/JAPANESE_PHASE1_IMPLEMENTATION.md). This is
the linguistic frontend consumed by the isolated Japanese synthesis route; it
does not alter the English frontend.
Phase 2 adds `japanese_profiles.py` and `japanese_candidates.py`: read-only bank
profiles, exact alias overrides, stable source candidates, mixed CV/VCV/CVVC
coverage, and deterministic provisional metadata. See
[JAPANESE_PHASE2_IMPLEMENTATION.md](docs/JAPANESE_PHASE2_IMPLEMENTATION.md).
Read-only inference is advisory. A generated Japanese voice requires an
explicit CV, VCV, or CVVC configuration; its aliases, canonical phones, and
candidate IDs are scoped to that configuration.
For a mixed source bank, explicit CVVC is a strict runtime selection policy:
ordinary VCV mora rows remain auditable but cannot become Festival units. CV,
VC/VV, release, and special-mora transition material remains available. See
[the strict CVVC implementation note](docs/development/STRICT_CVVC_RUNTIME_SELECTION.md).
Phase 3 adds `japanese_festival.py`, `japanese_synthesis.py`, and
`japanese_listening_set.py`: a separate Japanese Festival/UniSyn voice,
explicit duration and F0 plans, stable candidate overrides, and a reproducible
human-listening corpus. See
[JAPANESE_PHASE3_IMPLEMENTATION.md](docs/JAPANESE_PHASE3_IMPLEMENTATION.md).
The corrective assembly layer in `japanese_assembly.py` exposes every selected
alias and source slice, enforces one shared consonant center, and replaces a
missing pure-CV transition with a visible bounded bridge rather than hidden
silence. `japanese_assembly_listening.py` renders the CV/VCV/CVVC comparison
set. See
[JAPANESE_ASSEMBLY_REMEDIATION.md](docs/JAPANESE_ASSEMBLY_REMEDIATION.md).
Generated runtime metadata separates the immutable source recording bundle
from the chosen voice configuration and declares its exact supported
languages and Festival entry points. The GUI constrains language selection
from this manifest and labels older path-backed voices whose metadata should
be rebuilt.
Phase 4 adds `japanese_editing.py` and the Speech-tab mora/accent workflow,
project persistence, undo/redo, and read-only bank coverage/alias resolution.
See [JAPANESE_PHASE4_IMPLEMENTATION.md](docs/JAPANESE_PHASE4_IMPLEMENTATION.md).
Phase 5 adds `japanese_quality.py`, `japanese_refinements.py`, and
`japanese_release.py`: generated-unit join diagnostics, optional label/HTS-JSON
baselines, deterministic multipitch and voice-color routing, cache/release
checks, and a 16-example listening corpus. See
[JAPANESE_PHASE5_IMPLEMENTATION.md](docs/JAPANESE_PHASE5_IMPLEMENTATION.md) and the
[dependency/license inventory](docs/JAPANESE_DEPENDENCIES_AND_LICENSES.md).
Prompt 19 adds the production contextual duration model, continuous
source-filter voicing control, fixed A/B evaluation matrix, and extended join
analysis. See [PROMPT19_IMPLEMENTATION.md](docs/PROMPT19_IMPLEMENTATION.md), the
[duration-model note](docs/japanese_duration_model.md), and the
[measured benchmark report](docs/PROMPT19_BENCHMARK_REPORT.md).

> **Diphone build & synth:** see [GUIDE.md](docs/GUIDE.md) for the full
> record → configure → build → synthesize process.
> **Multisyn (unit-selection) upgrade:** see [MULTISYN.md](docs/MULTISYN.md) —
> data budget, Audacity-vs-oto assessment, recording/boundary protocol, and
> the `corpus_extract.py` (recording-script) + `labels2festvox.py` (label) tools.
> **Languages:** default is **English**; `--lang asaxi|en|ja` per call (Japanese via `en-jap-mapping.yaml`).

Turns a UTAU voicebank into a FestVox-style diphone database and renders it
from the bundled `synth_diphone.py`; no Vocab Forge or Festival runtime is
required for the pure-Python engine.

## Repository layout

- `src/festvox_tts/` contains the synthesis, language, builder, GUI, profile,
  corpus, and native-runtime implementation.
- `tests/` contains the complete headless and GUI regression suites.
- `docs/` contains operating guides, architecture notes, and historical
  implementation reports.
- `build_festival_voice.py` and `run_gui.py` are stable root launchers.
- `festvox.example.json` is the portable local-configuration template.

Install and verify from this directory without modifying any voicebank:

```text
python -m pip install -r requirements.txt
python check_environment.py
python run_tests.py
```

`festvox.example.json` is the tracked relative-path template. Machine-local
`festvox.json`, generated voices, caches, and rendered audio remain ignored.

## 1. Build the database

Config-driven (paths live in `festvox.json`):

```
python src/festvox_tts/utau2festvox.py                     # build every voice in festvox.json
python src/festvox_tts/utau2festvox.py --voice asaxi_lem   # just one
```

One-off (no config):

```
python src/festvox_tts/utau2festvox.py --bank "C:/path/to/voice/F3" --out "./generated_voices/asaxi_lem" --name asaxi
```

OpenUtau and legacy multipitch banks can be given at their top-level folder.
The converter reads every root/nested `oto.ini`, preferring
`character.yaml` subbank declarations and then `prefix.map`. Exact declared
prefixes and suffixes are removed before phone mapping, so aliases such as
`- ayPE3` become `- ay`; an exact declaration works equally well when a bank
uses `E3P` instead. The uncolored/default voice color is built unless another
one is requested:

```
python src/festvox_tts/utau2festvox.py --bank "C:/path/to/MyBank" --voice-color Headvoice
python src/festvox_tts/utau2festvox.py --bank "C:/path/to/MyBank/P3_E3" --character-yaml "C:/path/to/MyBank/character.yaml"
python src/festvox_tts/utau2festvox.py --bank "C:/path/to/LegacyBank" --alias-suffix P
```

`--character-yaml` and `--prefix-map` override auto-discovery. Repeat
`--alias-prefix` or `--alias-suffix` for a bank whose metadata is incomplete;
manual suffix `P` handles both `ayPE3` and `ayE3P` because declared affixes and
ordinary pitch tags are removed iteratively. `--voice-color all` is explicit:
it merges every color into one alternative pool, while current selectors have
no desired-color control and may cross colors during context selection. Treat
that mode as diagnostic/future-facing; separate generated voices per color are
the predictable current path.

Databases build to `output_root/<voice>` (the tracked example uses
`generated_voices/`),
**outside** the voicebank. Each DB dir contains:

```
wav/                         renamed 16-bit source wavs (Scheme-safe names)
dic/asaxi_diphone.scm        Festival index list  (diphone wav start mid end)
dic/asaxi_diphone.est        EST_File index       (for UniSyn / make_lpc)
dic/diphone_index.json       machine index + all OTO takes and contexts
festival/asaxi_diphone_stub.scm   minimal UniSyn voice scaffold
conversion_report.txt        unmapped tokens, preserved takes, bad lines
```

The audited Lem 4_Fis3 bank has **7884 selectable units** from 811 wavs,
including **3632 preserved alternative takes**. Only `-aw11` is unmapped.

### Timing conversion

UTAU stores relative-ms values; FestVox needs absolute seconds:

| FestVox | from UTAU |
| --- | --- |
| **start** | `Offset + bounded positive Overlap`; non-positive overlap retains the raw `Offset` |
| **mid** (phone boundary) | `Offset + Preutterance` |
| **end** | `Offset + |Blank|` if `Blank < 0`, else `file_length − Blank` |

The `wave` module measures each file so the negative-`Blank` case (length
measured *from* the offset) is computable at all. Every produced triple
satisfies `start < mid < end`.

Generated Festival voices apply a second, language-independent source-window
policy after those OTO landmarks are established. The default `adaptive` mode
uses at most 60 ms on either side of a normal diphone boundary, while hidden
full-side variants expose the rest of the same recording when a target phone
is genuinely long. `bounded` keeps only the short window; `full` restores the
legacy complete OTO span. A VCV `- V` row uses `Offset + Preutterance` as the
vowel onset and contributes only the phrase-start `pau-V` edge.

Zero-overlap OTO rows preserve their raw OTO geometry. A nonzero
`--zero-overlap-guard-ms` remains available only as an explicit diagnostic
experiment; it is disabled by default because listening validation showed that
moving the source cut can damage the following handoff. Positive overlap is
always authoritative. Generated Japanese bridges continue to use their
existing bounded crossfade without changing contextual take selection.

> Note: the brief stated the inverse `Blank` sign convention. On this bank's
> real data the inverse overruns the following diphone by whole seconds, so
> standard UTAU semantics are used (documented at the top of the script).

## 2. Defining the phoneme dictionary

Edit **`PHONEME_MAP`** at the top of `utau2festvox.py`. Keys are normalized
UTAU alias *tokens*: declared OpenUtau/UTAU affixes are removed first, then a
plain pitch tag such as `F#3`, and finally the alias is split on spaces.

- `"k": "k"` — map a token to a Festival phone name (diphone `k-a`).
- `"-": "pau"` — silence.
- `"inh": None` — exclude (breaths, etc.).
- A standalone `"b-"` can remain a distinct `b_` sustain/allophone. A
  two-token `V b-` alias is a V-C-silence triphone and is deliberately ignored;
  use the ordinary `V b` plus explicit `b -` transition instead.
- Unlisted plain-ASCII tokens map to themselves. Numbered aliases share the
  base phone spelling, but every take is retained with its recorded outer
  context for automatic or per-occurrence selection.

Everything is sanitized to valid Scheme atoms (ASCII-folded, `[A-Za-z0-9_]`).

## 3. Speaking with the bundled renderer

The renderer in this directory reads `festvox.json` and needs no files from
`99_Tools/vocab_forge`. Render from `99_Tools/festvox`:

```
python src/festvox_tts/synth_diphone.py "Onă Gaksamipỏpỏ"              # standalone, Asaxi
python src/festvox_tts/synth_diphone.py "the velveteen rabbit" --lang en --outdir clips
```

Vocab Forge may separately consume the same generated DB through its own
configuration, but that is an optional integration rather than a FestVox
runtime dependency.

Front ends (`synth_diphone.py`):
- **Asaxi** — romanization→arpasing rules from *00_Phonemes of the Asaxi
  Language* (gemination → held `cl`, `C+y` palatal units, digraphs).
- **English** — CMU dictionary (`pip install cmudict`), stress stripped;
  works because the bank is arpasing.

The renderer is pure-stdlib concatenation: it cuts each diphone around its
`mid` boundary (typically within 150 ms), searches a bounded source-local
neighborhood, and uses a measured pitch-synchronous raised-cosine overlap.
It de-clicks utterance edges and peak-normalizes. Vowel-quality and
Japanese-CV fallbacks cover the
few gaps in the diphone matrix (verified: 0 skipped diphones across the test
set).

## 4. Real Festival voice and Windows editor

`build_festival_voice.py` is the shared Windows front door for English, Asaxi,
and Japanese UTAU sources. The command requires an explicit language, bank
type, OTO scope, source recording root, output folder, and generated name.
English and Asaxi use the existing ARPAsing conversion path; Japanese-only
banks use the isolated CV/VCV/CVVC candidate and assembly path. An ARPAsing
configuration may explicitly enable English, Asaxi, and Japanese over one unit
database with distinct entry points and the bundled
`src/festvox_tts/profiles/en-jap-mapping.yaml`.
Each output declares its exact supported languages, alias namespace,
configuration identity, and Festival entry points.
Festival and EST run locally when available or through WSL using paths derived
at the tool boundary. Generated Scheme is relocatable and does not bake in its
build destination. See `docs/UNIFIED_VOICE_BUILDER.md` for commands.
The standard `kal_diphone` English voice is always registered from Festival's
authoritative WSL installation at
`/usr/share/festival/voices/english/kal_diphone`. On refresh the GUI may mirror
its roughly 6 MB files into `generated_voices/kal_diphone`, but that optional
Windows copy cannot shadow, remove, or change the language of the built-in
entry. Kal remains selectable as English even if the mirror registration is
stale or the mirror folder has been removed.

The builder generates pitchmarks, a UniSyn index, language-scoped Scheme, a
portable voice manifest, and alternative-unit metadata. Generated selectors
use portable Festival SIOD bindings and comparisons (without extended `let*`
or numeric `=`) and score the recorded outer context from the ordered `oto.ini`
aliases only; WAV filenames are source identifiers, not
phonetic evidence. Exact symbols score highest, followed by matching
language-neutral features (vowel, silence, voicing, manner,
liquid/nasal/glide), so a Japanese vowel context can safely support an English
transition. A strict CV context such as `ka` is directional: before a target
its adjacent edge is `/a/`, while after a target its adjacent edge is `/k/`.
When an internal transition is omitted, the converter uses the nearest edge
of the immediately adjacent, time-ordered OTO in the same recording. For
example, `ae s` followed by `t k` records the outer context of `t-k` as `s`.
It never searches past an intervening OTO and never reads the WAV filename as
phonetic evidence. Only a recording edge with no adjacent OTO remains unknown
`*`; unresolved literal aliases remain unclassified instead of being grouped
together. Generated choices record whether each context came from a chained
transition, an adjacent OTO edge, or was unavailable.

Special-phone realization is also language-neutral. Generated voices keep
canonical `cl` visible and editable but source `V-cl-C` as `V-C-C`, using a
bounded generated `C-C` consonant hold with no duplicated release. This applies
equally to frontend output and a `cl` typed manually in Phonemes mode. An OTO
alias named `cl` never enables literal behavior by itself. A creator of a bank
with a genuine linguistic `/cl/` can add
`--literal-phone-map cl_literal=cl`. Structural `cl` and authored
`cl_literal` then coexist; the builder requires non-silence `X-cl` and `cl-X`
source units. See
`docs/SPECIAL_PHONE_REALIZATION.md`.

Unless `--f0` is supplied, the generated voice pitch is the measured median of
the selected OTO scope, with zero automatic headroom. An E3-only Lem build
therefore records approximately `164.81 Hz`, not the former `202 Hz`.
Explicit `--f0` remains authoritative. For compatibility, an older manifest
tagged `speaker_median_plus_headroom` is read using its stored source median.

The English/Asaxi route accepts `--character-yaml`, `--prefix-map`, repeatable
`--alias-prefix`/`--alias-suffix`, and one `--voice-color`.
Nested subbanks may contain the same WAV basename; generated names include the
subbank path and are collision-safe. `dic/diphone_index.json` retains each
take's source OTO, subbank, color, affixes, and tone ranges. Multiple source
pitches remain labeled alternatives; optional Japanese dynamic routing is a
later runtime choice and manual per-occurrence selection remains final. The
legacy `--db`/`--utau` forms remain available for existing workflows, but new
builds should use the language-scoped command.

The WSL editor keeps language planning and final acoustic capabilities as
separate layers. English, Asaxi, and Japanese retain independent phone,
duration, phrasing, and F0 models, then every rendered phrase uses the same
Voicing, vocal-tract, output-calibration, fault, and user-gain stages. Generated
voices declare an active-speech calibration policy; the shared default is
-20 dBFS RMS with -6..+12 dB automatic gain and a 0.98 peak ceiling. It applies
one scalar after synthesis and excludes pauses, never per-unit normalization.
Current defaults also apply safely to older locally generated manifests;
built-in Kal and unknown external voices are unchanged.

Asaxi Text rendering now consumes the generated synthesis dictionary and
morphological H/L inference directly. The categorical pattern is realized by
one final-timing, sentence-level log-F0 planner shared by Generate and
Re-render; it carries F0 and slope across phrases, shapes boundary regions,
and deliberately forbids cumulative frequency drift. The shared **Pitch
accent** parameter
shows word-bracketed mora blocks with adjustable H/L and per-mora pitch;
**Mora voicing** shows selectable automatic/manual voicing blocks. There is no
separate Asaxi Breathiness control. These entries dispatch to
language-specific editors and are
available only for an Asaxi-compatible Festival/WSL voice. They are hidden,
without discarding state, for English and direct-phone sentences. Automatic
vowel devoicing and `x` aspiration are folded into the language-specific
automatic voicing value; realization uses the same source-filter stage as the
continuous Voicing editor. A block edit immediately updates the displayed
dashed baseline, while Re-render applies it to audio on the current edited
phone timeline and never regenerates phone lengths.

For diphones ending in a voiced sibilant (`z`, `zh`, `zi`, `dz`, or `jh`),
automatic selection prefers a verified vowel/sonorant/voiced-continuant
following context, then an unannotated context, over a verified stop or
affricate context that may contain recorded devoicing. If every take is risky,
the base remains selected. This safety tier is followed by ordinary two-sided
context scoring, including English light/dark `l`; manual per-occurrence takes
remain unrestricted. UTAU `V C-` coda triphones are excluded because their
embedded silent tail conflicts with the following `C-pau` unit; explicit
`C -` pause diphones remain available.

The Windows GUI refreshes generated-bank alternative and sustain indexes at
the start of every Generate and Re-render operation. Rebuilding or
re-registering a voice under the same name therefore cannot apply an old
`takeN` meaning to the new bank. Ordinary takes are compared with the explicit
base's real context score: an incoming vowel-class match may beat a base
recorded in a consonant cluster even when the far context differs. A lone
phrase-edge pause match cannot beat unrelated spoken context. If no numbered
take improves safely, fallback retains the explicit unnumbered `base` row
rather than relying on metadata order.
Conversion also caps an OTO slice at the following transition midpoint when a
recorded tail crosses that boundary. The source WAV and `oto.ini` remain
untouched; the generated index records both the original end and the clamp.
The generated voice also receives the final `dic/diphone_index.json`, which
lets the GUI locate unnumbered `X-X` sustains without consulting or modifying
the UTAU source. Existing generated voices remain compatible through a
read-only fallback to `db/dic/diphone_index.json` and `db/wav/`.

Breath-marked OTO aliases are never reduced to their surviving vowel token.
For example, `inh aw` is rejected as a whole instead of becoming an `aw-aw`
sustain that could be selected inside a phrase. Asaxi diphthong graphemes are
likewise planned as vowel-plus-glide sequences (`a y`, `a w`, and their
front/back vowel equivalents), allowing the ordinary transition selector to
choose each edge from its real spoken context.
Japanese CVVC `V R`/`V Rn` and `V RB`/`V RBn` rest-breath aliases are likewise
preserved as nonselectable source metadata and never canonicalized to tapped
`V r`; genuine lowercase `V r` aliases remain eligible VC transitions.
Japanese full-width punctuation
splits editable phrase state without requiring whitespace, and Asaxi text is
case-normalized before frontend processing.

Run `python run_gui.py` to open the Windows editor in
`src/festvox_tts/festvox_gui/`: waveform selection,
vowel-only and indefinite `X-X` sustain stretching, generated-F0 and
punctuation overlays, visible automatic/manual recording takes, four-part
phrase pauses, cleaned voice-local dictionaries, and guarded voicebank removal.
Every internal break exposes two outgoing and two incoming pause regions, so
each phrase owns a protected pair on both sides and changing the gap does not
stretch either neighboring phone. This applies to Japanese plans as well as
English/Asaxi phrase assembly.
Pitch-accent editing uses a horizontally scrollable, uniform-width mora
grid independent of waveform pan and zoom. Deleting or changing a mora's
planned phones removes its stale accent cell rather than attaching that cell to
the following audio. Recording blocks show compact role/alias captions and
keep full generated candidate IDs in their details instead of drawing hashes
over neighboring blocks.
Asaxi uses the same compact block interaction as Japanese, with word brackets
and a connected H/L contour in Pitch accent and multi-select mora blocks in
Mora voicing. Its underlying H/L model remains distinct from Japanese accent
nuclei. Waveform and block selections stay synchronized. Per-mora edits are
undoable and persistent; a detailed continuous Pitch or Voicing curve remains
the final authority.
Long utterances use cached per-pixel waveform peaks plus adaptive phone, pitch,
timing, recording, and intonation overviews; zooming in restores every editable
item without changing the underlying project data.
Runtime audio, voice metadata, and model caches are independently bounded and
thread-safe. **Options > Application caches** reports their approximate
in-memory size and can clear Audio, Voice, Model, or All. These actions do not
delete generated voices, source UTAU banks, projects, project WAV caches,
exports, dictionaries, configuration, or application files.
Festival runtime metadata excludes the duplicate alternatives/audit graph
stored in `diphone_index.json`; complete contextual choices remain in the
separate alternatives cache. This keeps a newly generated integrated voice
hot instead of reparsing and recursively sizing tens of megabytes several
times per render. The same 64 MiB default voice-cache budget is retained.
Source-pitchmark, join, spectrogram, and rendered-formant diagnostics are
calculated only when the user opens or exports the corresponding diagnostic.
Waveform Auto Adjust is a one-shot full-duration fit and immediately releases
auto-ranging, avoiding redraw/range feedback. The Pitch curve has its own
waveform-synchronized horizontal scrollbar, with a compact horizontal slider
for vertical zoom in the same row. Editable nodes and line sections crossing
`pau` are neutral gray while voiced sections retain their active color.
Version-4 projects are dedicated folders containing `project.json`, `cache/`,
and `exports/`; users open the folder's `project.json`, and older project JSON
is currently outside the GUI's supported workflow. Stable per-occurrence
segment IDs preserve duplicated Speech regions
through Re-render, save/load, Undo, and Redo. Generate warns before a fresh
phone plan can reset manual edits. Speech, Sentences, cache, playback, and
export all consume the latest committed sentence preview. Batch export writes
separate files or one merged file. Sentences supports
board-wide rectangle/Ctrl/Shift/Alt selection of rows and phrase blocks,
group drag ordering with insertion markers, selected WAV export/removal,
editable sentence text, per-sentence and per-phrase speakers, speaker
portraits, and highlighted phrase playback. `Ctrl+C`, `Ctrl+X`, `Ctrl+V`, and
`Ctrl+D` copy, cut, paste, and duplicate selected Speech regions or Sentences
items; Shift-drag moves a Speech region with a live insertion marker. In a
sentence row, Enter generates and Shift+Enter inserts a newline phrase while
the editor grows to a bounded scrollable height. `Ctrl+R` in Sentences
generates selected rows, or the whole list with no selection, and never starts
playback. The
playback button reads **Play selected** whenever sentence rows are selected.
Language and voicebank controls show `-` for mixed row selections; choosing a
real value applies it to every selected sentence and requires fresh Generate.
Changing tabs clears row and phrase selection. The Voicebank Manager supports
extended selection and one consolidated, destructive-warning deletion for
multiple generated banks; built-in Kal and source UTAU banks remain protected.
Its all-sentences gain control and
the Speech gain control use measured waveform headroom, warn visually about
pending re-renders, and require an explicit **Allow clipping** choice before
gain can exceed the safe peak limit. Stale audio turns grey with a yellow
Re-render action; boundary timing edits instead use a lighter blue waveform
with red boundaries. Changes that need a fresh Generate keep a stronger
Speech cue. Sentences rows instead become subtly neutral, omit the Generate
pending badge, and retain the yellow Generate button. Reverting text to the
last rendered value restores the normal row. Numbered
OTO choices survive phone edits by diphone identity; every transition remains
inspectable, including grey transitions with no alternatives. Conservative
runtime scoring protects older generated banks, and the inspector reports the
source alias/WAV/OTO line. Fault Mode includes
pinned/random pitch-estimation damage across one or more phones, exact replay
of a heard pinned frequency, sustain-loop comparison, the exact pre-fix join
path, and volume-compensated
8/4/2/1-bit output. Speaker icons can be replaced or removed from the
Voicebank menu; replacement clears stale `speaker.*` files with other image
extensions.
Existing generated folders require destructive confirmations; stale
registrations whose folders are already absent can be forgotten without any
file deletion. The previous Song implementation has been removed; its future
product contract is preserved in `docs/SONG_MODE_FUTURE.md`. See
`docs/GUIDE.md` and `src/festvox_tts/festvox_gui/README.md` for setup and use.

Prompt 20 adds speaker-relative semitone/log-F0 handling, a Kokoro-calibrated
Japanese pitch and duration model, automatic continuous mora voicing, and an
analysis-first vocal-tract resonance control. Japanese repeated phrases use
mean-centered contour-shape variation and never cumulative register drift.
Punctuation pauses are recovered from source energy rather than Kokoro's
interpolated boundary alone. Open JTalk morphology is retained per mora;
ordinary and negative auxiliaries receive only the train/held-out-replicated
timing effects, while unstable particle and polite-copula effects stay
diagnostic-only. A source-relative acoustic-edge audit compensates the extra
audible lead-in of phrase-initial vowel diphones without changing contextual
recording choices; a proposed blanket final-vowel correction was measured and
rejected.
`Re-render Phonemes` preserves the current editor durations exactly; only a
fresh Generate may replace them with a new modeled timeline. The Vocal tract
length parameter loads its ordinary and expanded bounds from
`src/festvox_tts/profiles/reference_voice_space_v1.json`, keeps pitch and
duration independent,
and exposes final-waveform formant diagnostics. Reproduction commands,
objective metrics, source hashes, and short held-out source-versus-synthesis
waveform correspondence plots with per-phone duration deltas are in
`docs/PROMPT20_IMPLEMENTATION_REPORT.md`.

Generated Festival voices now default to a deterministic UniSyn grouped audio
cache. The builder packs generated WAV slices and pitchmarks once, Festival
uses the indexed group instead of repeatedly opening source files for selected
units, and generated Scheme retains a separate-file fallback. This storage
change never alters contextual or manual unit choices. The Sentences tab also
defers hidden Speech-editor waveform hydration and reuses immutable PCM views,
so changing the selected sentence no longer performs duplicate waveform copies
or a generated-voice rescan.

Backend synthesis and expensive source-filter transforms now execute on a Qt
worker while the GUI event loop stays live. The normal pointer, tab navigation,
timeline interaction, and sentence-list scrolling remain available; only
request-defining controls and duplicate render actions are locked for the
duration of each call. Batch targets are committed without changing the
active sentence or tab, so users can navigate and edit while other rows are
generated; a result whose own sentence changed in flight is discarded as
stale. Zero-Fall Japanese statements keep
their generated linguistic endpoint instead of rising toward the global base
pitch; question and expressive punctuation behavior is unchanged. Manual
continuous Voicing edits now remain authoritative inside `pau` spans too, so
audible source speech crossing an acoustic/label boundary can be corrected.

Normal WSL renders use the project-local native Festival helper and keep one
process warm between jobs, recycling it after 32 requests by default. This
removes repeated Festival startup cost without caching stale voice metadata.
Direct Festival and **Legacy joins** remain one-shot stock-Festival paths.
