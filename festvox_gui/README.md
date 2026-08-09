# Festvox Speech Synthesis GUI (PyQt5)

A Windows-XP-styled desktop front-end with **two synthesis engines**, selected
from the Engine dropdown:

1. **Diphone (pure Python)** — the bundled `../synth_diphone.py`,
   rendering the FestVox-style DBs built by `utau2festvox.py`. No Festival
   runtime, runs on plain Windows Python.
2. **Festival via WSL (Multisyn)** — the real Festival running inside WSL.
   This is the path for **Multisyn unit-selection voices** (built per
   `MULTISYN.md`) and any other genuine Festival voice.

Both engines feed the same editor: waveform with draggable phone boundaries
(time-stretch DSP), editable phoneme fields, a waveform-aligned
timing/pitch/intonation/recording parameter editor, multi-sentence project
save/load, individual and batch WAV export, and diagnostic faults.

## Files
- `../GUI_STATE_AND_SYNTHESIS.txt` - dated plain-text snapshot of the current
  product state, architecture, complete synthesis pipeline, safety boundaries,
  caches, and known limitations. Start here when handing the project to a new
  development session.
- `festvox_gui.py` — the PyQt5 + PyQtGraph GUI.
- `festvox_core.py` — no-Qt backends: `DiphoneBackend` (synth_diphone glue)
  and `FestivalWSLBackend` (wsl.exe → festival -b, path translation,
  utt.save.segs parsing), plus DSP and WAV/project IO.
- `config.json` — all GUI settings. Written back automatically.
- `requirements.txt` - required Python packages for synthesis, editing, and
  playback.
- `../SONG_MODE_FUTURE.md` - retained product and MIDI-import brief for a
  future Song implementation. Song is deliberately not part of the current
  runtime.

## Install & run (Windows)
```bat
pip install -r ..\requirements.txt
pip install sounddevice   :: optional, best playback (else winsound is used)
pip install librosa       :: optional, higher-quality time-stretch
python festvox_gui.py
```

Run `python ..\check_environment.py` for a read-only installation report.
Long-session resource checks are documented in
`../docs/development/PROMPT0_LONG_TERM_PERFORMANCE.md` and run with
`python ..\resource_soak.py`.

### Setting up the Festival/WSL engine
```bat
wsl --install             :: once, if you don't have WSL yet
```
inside WSL:
```bash
sudo apt install festival festvox-kallpc16k   # runtime + a test voice
```
Then in the GUI: *Options → WSL / Festival settings...* → **Test connection**.
Defaults (wsl.exe on PATH, default distro, `festival` binary) usually just
work; distro, binary path, an extra site `.scm`, and the timeout are all
settable there.

## Voices
**Diphone engine** — voicebanks come from the toolchain's `festvox.json`
(`voices` + `output_root`) or *Voicebank → Add diphone DB folder...*
(needs `dic/diphone_index.json`).

**Festival engine** - **Options > WSL / Festival settings** configures a
Windows scan root and an optional WSL scan root. Immediate child folders that
contain Festival Scheme are discovered automatically. A successfully scanned
root removes stale auto-discovered entries when their folders disappear; an
unavailable root preserves its entries, and manually added registrations are
never overwritten or removed by discovery. Windows wins duplicate names. The
Windows root is watched while the app runs, so deleting a child voice folder
updates the list without restarting; WSL is refreshed on startup and command.

The standard Kal voice is authoritative at
`/usr/share/festival/voices/english/kal_diphone` inside Ubuntu and appears by
default as an English-only built-in. The GUI may copy it once into
`generated_voices/kal_diphone`, but a stale or deleted Windows mirror cannot
hide the WSL voice and the built-in cannot be uninstalled.

**Voicebank > Voicebank manager...** lists locations and status, refreshes both
roots, adds an individual Windows or WSL folder, and deletes generated voices.
Its list supports Ctrl/Shift multi-selection and one consolidated permanent
file-deletion warning for the selected generated banks. Deletion refuses
Festival built-ins or UTAU source folders. The sidebar list itself collapses
from its heading and its lower grip changes the amount of space it occupies.

### Build a real Festival voice

An `utau2festvox.py` database is not loadable by Festival on its own: its
`festival/` stub has no pitchmarks, complete Scheme, or voice entry point. Use
the shared Windows builder with an explicit language and bank configuration:

```powershell
$Builder = ".\99_Tools\festvox\build_festival_voice.py"
$Source = "X:\UTAU\voice\MyBank\F3"
$Output = ".\99_Tools\festvox\generated_voices\my_bank_en"

py -3.14 $Builder `
  --language en `
  --bank-type arpasing `
  --samples $Source `
  --oto "$Source\oto.ini" `
  --output $Output `
  --name my_bank_en `
  --test
```

Japanese-only banks use `--language ja` and an explicit `cv`, `vcv`, or `cvvc`
type. Asaxi uses `--language asaxi --bank-type arpasing`. An ARPAsing voice can
advertise all three frontends with repeatable `--enable-language` flags and the
default `../profiles/en-jap-mapping.yaml`. Full commands and output identity
rules are in `../UNIFIED_VOICE_BUILDER.md`.

The builder defaults to `--source-window-mode adaptive --source-window-ms
120` for every language. This keeps long OTO tails out of ordinary short
phones but retains full-side variants of the same selected recording for long
stretches. `bounded` is the strict capped A/B mode and `full` restores legacy
whole-region behavior. Japanese VCV `- V` rows align their GUI vowel onset to
OTO preutterance and are never used as medial vowel units. Rebuild an existing
voice to apply either behavior.

For an OpenUtau top-level English/Asaxi bank, the converter reads nested OTO
files and prefers `character.yaml` over `prefix.map` when removing pitch/color
affixes.
Both metadata paths are auto-detected at the selected bank root; explicit paths
also let a user build one subfolder while referencing its parent metadata.
Repeatable `--alias-prefix` and `--alias-suffix` are the fallback for incomplete
metadata. A manual suffix `P` handles either `ayPE3` or `ayE3P`. Default builds
exclude named colors, so tone colors do not silently enter ordinary speech.
The stable front door rejects `--voice-color all`; use separate generated
voices per color. The legacy builder can still create a diagnostic merged
pool, but the GUI does not treat that as a normal current configuration.

Generated alternative metadata records the exact source OTO, subbank, color,
prefix, suffix, and tone ranges. Dynamic pitch/color routing requires an
explicit experimental API opt-in and is absent from the stable GUI; build one
generated voice per pitch/color. Manual per-occurrence unit choices remain final.

It generates pitchmarks (a normalized 16 kHz marking copy plus the FestVox lx
filter chain), the UniSyn `.est` index, portable metadata, and exactly one
declared language entry point. `--test` loads that entry point and requires a
fresh nonempty Festival WAV. Register it with **Voicebank > Add Festival voice
folder...**.

**If you built a voice with an older version of the builder: just rerun the
build command** — the pitchmark recipe is versioned (stale marks regenerate
automatically) and the voice scheme is rewritten. Rebuilding also adds the
newer fixes:

- **Language-scoped entry points** - current English and Asaxi builds each
  expose one `voice_<name>` with the matching front end; Japanese exposes one
  `voice_<name>_ja`. Historical combined builds may still expose
  `voice_<name>_en`, and the GUI retains a read-only compatibility route.
- **Silence fallback unit** — a real `pau-pau` diphone backed by a silence
  wav becomes the default substitute, so bank gaps or typo'd phonemes
  (`hhh`) produce a quiet gap plus a "MISSING" report in the status bar,
  instead of a glitchy blip from whatever unit sorted first.
- **Context-sensitive OTO takes** — numbered duplicates remain selectable,
  `dic/unit_alternatives.json` supplies the GUI menus, and the generated
  UniSyn hook chooses a take from its recorded context. Rebuild an older voice
  before expecting these menus or automatic choices.
- **Runtime sustain index** - `dic/diphone_index.json` is installed after
  silence and onset repairs, so the GUI can locate each unnumbered `X-X` WAV
  region for indefinite vowel previews without reading or changing the UTAU
  source. Voices built before this runtime copy was added remain compatible:
  the GUI reads the embedded `db/dic/diphone_index.json` and `db/wav/`
  locations as a read-only fallback.
- **Guarded OTO tails** - a generated unit whose raw end crosses the following
  transition midpoint is capped at that midpoint. The generated index retains
  `raw_end` and `tail_clamped` for inspection; the source WAV and `oto.ini`
  are never rewritten. Rebuild older generated voices to receive the fix.
- **Class-based default durations** — stops get ~60 ms instead of 90 ms;
  PSOLA-stretching a plosive repeats its burst, which was the "glitched
  blip" on initial consonants like `b`. (Initial-consonant gaps also show up
  as MISSING now — CV-style UTAU aliases don't yield `pau-C` units; the
  Asaxi reclist covers them, English hits more gaps.)

### Multisyn workflow (short version; details in `MULTISYN.md`)
record continuous Asaxi, label it in Audacity, run `labels2festvox.py`, build
the voice with the FestVox multisyn recipe, then use **Add Festival voice
folder...**, select it, and Generate. Current generated voices belong under the
configured Windows root; existing WSL-only registrations remain legacy input.

## What you can do (both engines)
- **Generate Audio** — diphone: g2p + unit concatenation with per-phone
  timing; Festival: `SynthText`, wav via `utt.save.wave`, real phone
  boundaries via `utt.save.segs`, **F0 targets captured** for later
  re-renders. Missing units are reported in the status bar on both engines.
- **Input mode (Text / Phonemes)** — the combo next to the input box. In
  *Phonemes* mode you type space-separated phones and they are synthesized
  directly, e.g. `hh eh l ow pau m ay n ey m ih z l eh m`.
  Special timing phones use the same language-neutral source resolver as Text
  mode. In a generated UTAU voice, manually typed `i cl s o` stays visible and
  editable as written but sources `i s s o`; an OTO alias named `cl` cannot
  silently change that behavior. A creator can expose an authored `/cl/`
  simultaneously under a distinct token such as `cl_literal` with
  `--literal-phone-map cl_literal=cl`.
- **Language picker (works on both engines)** — Diphone: Asaxi /
  English-CMU / Japanese as before. Festival uses the language entry point
  declared by the selected configuration. Legacy combined voices can route
  English through their `voice_*_en` compatibility front end (CMU g2p over the
  generated units); **Asaxi** uses the configuration's own letter rules;
  **Japanese** is analyzed on the Windows side and sent as explicit phones.
  Native Asaxi words are case-normalized before either frontend. Full-cap
  proper names and borrowed terms remain intact as explicit switches to the
  integrated voice's English G2P; accidentally mixed-case native words still
  normalize to lower case.
  Generated Asaxi text uses `dictionaries/asaxi_lexicon.json` for canonical
  phones and one H/L target per mora. Multiword expression records use one
  `|`-separated accent chunk per written word and match left-to-right,
  longest-first even inside a larger sentence. Their source-note provenance is
  retained in project data. Unsupported letters are rejected rather than
  silently removed.
  Phones outside the selected manifest are rejected before Festival runs.
- **Shared Festival acoustic pipeline** - English, Asaxi, and Japanese keep
  their own frontend phones, timing, phrasing, prosody, and F0 plan, but every
  completed WSL render uses the same final capability order: Voicing, Vocal
  tract length, generated-voice phrase calibration, diagnostic bit depth, and
  user Output volume. Japanese-only mora/accent pages are hidden and disabled
  for non-Japanese sentences; their underlying sentence/project state is not
  deleted when the language changes.
- **Japanese planning** - the
  Festival path now uses the explicit auto/Open JTalk/kana frontend, canonical
  mora/accent model, and version-2 phone/timing/F0 plan. The structural contour
  combines FRQ-centered semitone components for phrase reset, lexical accent
  fall, bounded downstep, declination, and final lowering. Mora-first timing
  separately allocates ordinary, palatalized, geminate, moraic-nasal, long,
  devoiced, and phrase-final morae. Contextual mode keeps selected-source
  geometry as the absolute baseline and applies bounded Open JTalk/canonical
  residuals; legacy mora timing remains selectable under Options. Mapped
  Japanese `nn`/`nng`/`mm`/`xn` aliases retain a moraic-nasal timing role
  instead of becoming generic consonants. It accepts only a
  generated Japanese `*_ja` voice; kal and English generated voices are
  rejected with a clear isolation error. Optional label/HTS-JSON baselines
  extend this plan without changing the English path or using an Open JTalk
  waveform. Stable planning ignores stored pitch/color routing metadata.
- **Waveform**: the ruler above it is the only place that moves the playhead;
  Space starts at that position and the marker advances during playback.
  Press Space again to stop and hold the current position, then Space resumes
  from there. Natural completion returns it to zero for a fresh replay.
  Mouse-wheel zooms the shared time axis and the always-visible scrollbar pans
  it. Shift-wheel scrolls horizontally without moving the current selection;
  middle-drag remains viewport panning even while a region is selected. The
  ruler mirrors the waveform's numeric range, keeping clicks and the
  playhead aligned at every zoom, including when panning into the small area
  before the zero mark. **Follow playhead** is enabled by default and jumps
  one page whenever playback leaves the visible range; uncheck it to inspect
  another part of the sentence without the view returning. This preference is
  saved. A left click selects one phone; left-drag
  selects a larger phone-aligned region. Shift-drag inside that selection
  moves it: a translucent copy follows the pointer and a blue line marks the
  insertion point. Right-click opens the selected
  region's pitch-fault menu and never pans the view. The phoneme
  boxes below **follow the zoom/pan**, staying aligned with
  the audio. Red dashed lines are phone boundaries; dragging one is a
  **ripple edit**: this phoneme's length changes and every later phoneme
  slides along — no limit, so any phoneme (incl. the last) can be stretched
  arbitrarily. Automatic Resize performs one full-duration horizontal fit,
  releases auto-ranging immediately, and keeps the waveform locked to its
  stable -1.05..1.05 vertical range.
- **Phoneme fields** — edit any phone (yellow = pending), space-separate to
  insert several, clear to delete — or **right-click for Insert
  before/after / Delete**. Enter or **Re-render Phonemes** applies. On the
  **Festival engine the re-render keeps the original prosody** (durations +
  captured F0 via a `Segments` utterance) — kal stays melodic.
- **Speed** — double-click the slider to reset to x1.00, or type an exact
  factor in the box. Diphone: real x0.25–x4 concatenative pacing. Festival:
  sent as `Duration_Stretch` (Multisyn ignores it — use the timing bars).
- **Parameter Editor / Timing** — one vertical bar per phoneme, aligned
  under the waveform (x0.25–x4). Drag up = longer, down = shorter, and
  **sweep sideways to paint several bars in one gesture**; the WSOLA stretch
  is applied on release. Fast sweeps interpolate every crossed bar instead of
  leaving gaps. Hold Shift while painting to lock the first value and apply it
  horizontally. **Right-click a bar to reset that phoneme to the
  rendered timing.** The in-editor stretch is a preview — **hit Re-render
  Phonemes afterwards for optimal quality** (the engine re-synthesizes at
  your timings; the button lights up to remind you).
- **Parameter Editor / Voicing** — the dashed line is the measured automatic
  harmonic/aperiodic analysis and the editable line is the final continuous
  target. Analysis uses short overlapping frames rather than one point per
  phone and remains stable after regeneration. The source/filter renderer
  separates harmonic and stochastic excitation, retains a shared tract
  envelope, and uses one continuous deterministic shaped-noise source at low
  voicing. This control is available for rendered audio in every language;
  Japanese contextual devoicing can seed its automatic targets. For English,
  the common continuous-curve view can expose the diagnostic syllable parser
  as alternating background bands and dotted boundaries. **Show syllables /
  morae** appears for Pitch curve, Voicing, and Vocal tract length, is off by
  default, and is saved as a local view preference. It shows English syllables
  with phone/stress details, or the existing orthographic mora plan for
  Japanese and Asaxi. Labels are bounded and zoom-dependent for long
  utterances. English inline-phone passages also recognize every
  speech-bearing vowel nucleus in the default integrated profile, including
  `a e i o u`, `rr`, and its syllabic nasal aliases. Additional generated
  profiles can declare their own vowel nuclei. The overlay never changes any
  curve, phone, or synthesis.
- **Play / Export / Projects** — projects preserve every sentence's engine,
  language, voice, input mode, phones, timing, pitch/intonation, recording
  overrides, fault settings, and stable per-occurrence segment IDs. The latest
  sentence preview is the single audio source used by Speech, Sentences,
  playback, project cache, and export. The divider between Waveform and
  Parameter Editor is draggable, so either view can receive more vertical room.
- **Parameter Editor / Pitch accent** - enabled only for a Japanese
  sentence on the Festival engine when the selected generated voice explicitly
  advertises Japanese support. Rebuilt voice metadata is reloaded when the
  selection refreshes, and the page remains inaccessible for English and Asaxi
  sentences. Its mora intervals,
  playhead, selection, and horizontal view share waveform time. A single click
  selects a mora without changing accent; double-click places the nucleus, and
  dragging the existing triangle moves it after the normal drag threshold.
  Right-click sets the accent phrase to unaccented. Bracket
  boundaries drag, and Split/Merge commands edit phrase structure. It also has
  per-mora cent offsets and a structural/Open-JTalk-label/external HTS-JSON
  baseline selector. Question shape belongs only to Intonation blocks; fresh
  Japanese renders create those blocks from ASCII or Japanese full-width
  punctuation. Source
  inspection belongs only to Recordings. Optional provider durations apply on Generate;
  Re-render preserves manually edited boundaries. Accent and mora edits overlay
  the generated baseline, manual recording choices remain final, and the
  ordinary continuous Pitch curve remains the final F0 authority.
- **Sentences playback** - selecting sentence rows changes **Play all** to
  **Play selected**; playback already follows that selection. Sentence-level
  playback always uses each complete canonical waveform. Phrase previews
  partition that waveform without gaps: the initial pause belongs to the first
  phrase, while each canonical four-pause internal boundary is divided two and
  two between the outgoing and incoming phrases. Extra acoustic breaks
  inserted by Festival are grouped into their logical text phrase, so they
  cannot drop a later spoken span from Play selected/all.
  Programmatic sentence selection clears any old phrase selection, keeping the
  Play all/Play selected label and the actual playback branch identical.
- **Japanese timing diagnostics** - **Generate > Render details...** lists each
  mora's predicted and final duration and each phone's OTO-derived source
  reference, source-safe range, requested stretch, and final allocation. The
  same deterministic rows are stored in `last_plan.mora_timings` and the
  ignored listening-set manifest.
- **Japanese bank analysis** - **Voicebank > Analyze Japanese UTAU bank...**
  previews CV/VCV/CVVC/mixed coverage and every unresolved alias read-only.
  Exact overrides are saved in an external profile; the profile guard refuses
  to write inside the source bank, and profile edits visibly require a rebuilt
  generated voice rather than an ordinary Re-render.
- **Keyboard** - Space plays, `R` re-renders edits, and `Ctrl+R` performs a
  fresh Generate, including while the text box has focus. In Sentences,
  `Ctrl+R` generates the selected rows (or all rows when none are selected)
  without starting playback. Plain Enter in a sentence-row editor generates
  that sentence; Shift+Enter inserts a line break, which is synthesized as a
  new phrase inside the same sentence. The editor grows with those lines up to
  a bounded scrollable height. Single-sentence generation moves
  focus to the ruler so Space immediately plays. Defaults also include
  `Ctrl+Z` Undo, `Ctrl+Y`/`Ctrl+Shift+Z` Redo, standard project/export and
  clipboard bindings, `Ctrl+D` Duplicate, and Delete. In Speech these operate
  on the selected phone region, including a single phone, and paste directly
  after the selection. A copied or duplicated phone receives a new segment ID;
  moves, save/load, Re-render, Undo, and Redo retain that occurrence's ID. In
  Sentences the shortcuts operate on selected phrase blocks or sentence rows.
  **Options > Keyboard
  shortcuts...** validates collisions and unsafe bare typing keys. The status
  strip lists the useful bindings for the active or hovered work area and
  updates after customization.
- **Pending render cues** - changes that can reuse the current phone plan turn
  the blue waveform grey and highlight Re-render yellow. Moving phone
  boundaries instead uses a lighter blue waveform and keeps the boundaries
  red. Changes that need a new text/front-end pass, such as text, language,
  engine, input mode, or Output Speed, use a muted red Speech waveform. In the
  Sentences tab, an edited row and its previews become only slightly more
  neutral than their default colors; there is no **Generate pending** badge.
  The Generate button remains yellow. If edited text is restored exactly to
  the last rendered text before Generate, the pending state clears and the row
  returns to its normal color. The same state appears on phrase previews and
  Generate/Re-render All. Repeated volume and clipping-limit synchronization
  is non-reentrant: programmatic ceiling clamps do not feed gain signals back
  into the pending-state painter. Speaker, volume, pitch/Fall, faults, timing, phone,
  intonation, and recording-take edits all use this shared model. Output Speed
  always requires the stronger fresh-Generate state.
- **Output volume** - Speech and Sentences use the same resettable slider,
  exact dB field, and clipping rules. Gain stays disabled until rendered audio
  exists. Editing it turns the control amber until Generate or Re-render
  applies the pending value to preview, cache, playback, and export. By
  default, the upper limit follows the rendered peak so the result cannot
  clip; **Allow clipping** explicitly releases that ceiling. Double-click any
  slider to restore its default. Generated Festival voices first receive one
  active-speech scalar per completed phrase, excluding `pau`: the default is
  -20 dBFS RMS, -6..+12 dB gain, and a 0.98 peak ceiling. This does not
  normalize phones or recording takes. Explicit voice metadata wins; legacy
  voices built by this tool use the current default, while Kal and unknown
  external Festival voices remain untouched.
- **Pitch / Fall controls** - Fall is visibly bounded to 0-40%; normal English
  Text mode runs the same phrase-edge PSOLA pass as phoneme input, so Fall
  changes the rendered contour. Fall is hidden when the selected engine or
  Monotone mode makes it ineffective. The Pitch editor first uses Festival's
  learned `Target` relation. If a Festival voice omits that relation, it
  recovers a compact contour from the rendered UniSyn target pitchmarks; only
  a render with neither source uses a basic Pitch/Fall baseline. Re-render
  carries the recovered contour forward instead of returning to an empty
  track. For English LR intonation, `Fall = 0%` preserves the loaded voice's
  native pitch variance; only Fault Mode > Monotone deliberately flattens it.
  Combo and spin
  controls use painter-drawn arrow indicators rather than style-sheet
  placeholder rectangles.
- **Voice pitch default** - choosing a generated Festival bank reads the
  selected OTO scope's measured speaker median; `kal_diphone` defaults to
  110 Hz. Automatic builds add no melodic-headroom transposition, so Lem V4Bi
  `3_E3` resolves to about 164.81 Hz rather than 202 Hz. Older
  `speaker_median_plus_headroom` manifests recover their stored median, while
  an explicit builder `--f0` remains authoritative. A value typed into Pitch
  becomes a sentence-local manual override and is not replaced by later voice
  changes.
- **Japanese text rendering** - Qt high-DPI behavior is enabled before the app
  starts, text uses point sizes, and the font helper verifies Japanese glyphs
  while preferring Yu Gothic UI, Meiryo UI, Meiryo, or Noto Sans CJK JP. It
  scans installed system fonts as fallback and bundles no font files.

### Multi-sentence projects

The sentence selector above the input field owns an arbitrary list of separate
sentences. Add, duplicate, remove, and switch sentences without losing each
sentence's engine, language, voice, input mode, speed, phones, timing factors,
pitch/intonation edits, selected recordings, Japanese canonical/accent state,
or Fault Mode settings.
When several sentence rows are selected, differing Language or Voicebank
values display `-`; choosing a real value applies it to every selected row and
marks each for fresh Generate. All configured languages remain selectable for
selected rows even when their current speaker supports only one language, so
language can be changed before choosing a compatible speaker. The voice
manifest is still enforced when rendering. Switching between Speech and Sentences clears
both row and phrase selection.

Every Generate and Re-render shows an indeterminate current-sentence progress
bar for the complete backend and signal-processing operation.
**Generate All Sentences** renders the list in order. Generate All and
Re-render All add a determinate completed/total bar immediately beneath the
current-sentence bar. The two bars split one normal progress-bar height evenly,
so batch work does not make the status bar grow. When only one bar is visible,
it expands to the full reserved height. The adjacent stop button requests
cancellation after the current synchronous Festival render and preserves
completed sentence audio. Both progress rows clean up after success, failure,
or cancellation. Batch rendering never selects or opens the sentence currently
being rendered. The active sentence, tab, row selection, and sentence-editor
focus remain user-controlled, so Speech editing and navigation can continue
while other rows render. Each result is committed directly to its target state;
if that sentence changes during its own render, the obsolete result is
discarded and the edit remains pending.
**Export Batch** writes
one numbered WAV per generated sentence into a chosen folder; existing files
are never overwritten (a numeric suffix is added). New projects use the
version-4 folder layout:

```text
Project Name/
  project.json
  cache/
  exports/
```

`project.json` stores the sentence list, active index, phrase routing,
parameters, Japanese mora/accent overlays, stable segment IDs, and the semantic
minor/major/sentence phrase-pause settings. It also stores
`rendered_pitch_hz` and `rendered_fall_pct`, the controls used by the last
successful waveform render; these are compatibility-safe state markers, not a
second pitch contour. `cache/`
stores one WAV per rendered
sentence, while export dialogs default to `exports/`. **Open Project JSON**
opens the `project.json` inside this folder. Pre-version-4 project JSON is not
currently exposed or supported by the GUI.

### New synthesis and safety controls

- **Pitch curve** keeps generated Festival F0 as a dashed reference beneath an
  editable 50-500 Hz contour. Normal mode puts control points at both voiced
  edges of every phrase as well as phone centers; **Raw F0 joins** deliberately
  removes those edge anchors. Left-drag paints without needing to hit a dot;
  right-drag smoothly restores regions to generated F0. The toolbar reset
  restores the whole curve and enables Re-render. Re-render sends targets
  through UniSyn PSOLA; it does not pitch-shift finished audio. Sparse visible
  controls are stored as local log-F0 deviations over the complete generated
  Target relation, so moving one point does not replace or rebase untouched
  detail elsewhere in the sentence. Stretching carries that deviation
  proportionally inside the same segment. Insertion, deletion, movement, and
  Undo/Redo use persistent segment IDs, so removing one phone never scales the
  rest of the sentence's F0 timeline. Re-render consumes the already-retimed
  baseline without a second global recentering pass. Pause controls are rebuilt
  canonically from their neighboring phrase-edge pitches, preventing stale,
  jagged pause targets from accumulating across edits. Projects retain the
  Pitch/Fall values used by the last successful render so a deliberate global
  Pitch edit remains distinct from an ordinary phoneme edit. Vertical zoom
  changes only the view; its horizontal slider shares a footer row with a
  waveform-synchronized pitch scrollbar. Horizontal
  grid lines extend the left-side Hz markers across the editable area. Fast
  paint strokes interpolate the crossed controls, while Shift locks the first
  painted frequency. Nodes and line sections over `pau` are fully neutral gray
  so voiced pitch material remains visually distinct.
- **Intonation blocks** align statement, question, exclamation, continuation,
  colon, and semicolon contours with phrase spans. Changing a block immediately
  updates the Pitch curve preview before Re-render.
- **Phrase edges and gaps** use paired `pau` segments by default. At utterance
  edges one pause can absorb outer padding while the phone-adjacent pause
  stays short; at internal joins the trailing pause protects the preceding
  phone and the leading pause carries the resizable gap. **Options > Phrase
  pauses...** edits semantic minor/major/sentence totals in milliseconds and
  marks rendered Festival text for Re-render. The internal segment count stays
  hidden. **Single phrase pause** in Fault Mode collapses each run for comparison.
- **Recorded alternatives** preserve every numbered OTO take. The
  **Recordings** parameter view shows every rendered transition. Transitions
  with alternatives show the selected `Auto takeN` or manual take; transitions
  with no choice remain as grey, inspect-only blocks instead of disappearing.
  Right-click any block to inspect it; selectable blocks also offer Auto and
  every available take for that occurrence. The phoneme right-click menus
  remain as a second route. English
  `l` prefers light takes before vowels or `/j/`, and dark takes before
  consonants or a pause. Phone insertions and deletions remap manual choices by
  unchanged diphone identity instead of erasing them or applying an old index
  to the wrong phone. **Inspect selected recording** reports the exact source
  alias, WAV, OTO line, slice times, recorded outer context, directional edge
  evidence, automatic reason, and score. For Japanese, **Inspect selected mora
  contributions...** lists every source role and slice used by that mora.
  **Export Broadband Impulse Join Audit...** writes a named waveform/STFT
  diagnostic to
  `diagnostic_images/broadband_impulse_join_audit/`. The image points to
  measured events above the spectrogram, shows handoff spans, aligns rendered
  phone regions under the image, and includes the complete phone sequence.
  Stop and affricate regions such as `k` are tinted separately because a
  broadband release can be legitimate; this annotation never removes the raw
  measurement or its issue label.
  **Inspect joins and UniSyn windows...** opens the same exact-handoff
  evidence plus the real normal-render controls. Its table defaults to
  rendered phone order and can optionally sort Worst first. Blue handles still
  preview the bounded `1.00`-`1.25` source-window radius, while the crossover
  control uses elapsed milliseconds: 40 ms by default, up to 100 ms, with
  asymmetric left/right overrides for an individual rendered occurrence.
  Voiced edges snap inward to target pitchmarks and phone-class context caps
  are reported separately from the requested and rendered durations.
  A **Source trajectory** tab appears when the native renderer found a strong,
  bounded phase improvement between adjacent mapped pitch periods inside one
  selected recording. Teal triangles locate corrected epochs in the overview;
  the table shows source frames, center shift in samples, correlation
  before/after, and the acceptance reason. This evidence is stored with the
  rendered sentence and survives project save/load.
  **View > Rendered joins in waveform** places those occurrences directly on
  the Speech waveform. Click a diamond to expose green left/right handles; the
  readout shows `requested | rendered | cap`. A timing stretch carries the
  overlay with its phone, and a join drag enters the normal Re-render-pending
  state with Undo/Redo support. Opening the inspector from a Recordings block
  focuses the join rendered inside that phone. These controls never move phone
  boundaries, F0, or contextual/manual take choices. Legacy joins disables
  them, bypasses same-unit phase stabilization, and runs exact stock Festival.
  Click the selected diamond again, click
  or drag the waveform, hide the overlay, or press Escape to dismiss the join
  handles without deleting the saved setting.
  Runtime phone classes are explicit: voiced sonorants and voiced fricatives
  receive a larger fraction of their phone than vowels, while stops and
  affricates retain short closure-safe caps. This also corrects existing
  generated voices with old generic consonant phonesets.
  A sentence default of `0 ms` with no positive occurrence override is an
  explicit no-crossover A/B request and also uses one-shot stock Festival. A
  positive occurrence override still invokes the native renderer.
  Classification uses only adjacent,
  ordered OTO aliases; the WAV filename is displayed for location but ignored
  as phonetic evidence. If a transition is omitted between two ordered OTOs,
  the nearest edge of the immediately adjacent OTO supplies the missing outer
  context; `*` means no adjacent OTO evidence exists, not silence. Strict CV
  aliases use the edge nearest the selected diphone (`ka`
  before a target contributes vowel `/a/`; `ka` after it contributes stop
  `/k/`), while unresolved literal aliases remain unclassified.

  Automatic matching is conservative: one matching phrase edge cannot
  outweigh an unrelated spoken context. When literal symbols differ across
  languages, compatible broad classes (vowel, silence, voicing and manner,
  liquid/nasal/glide) receive a smaller positive score. Ordinary numbered
  takes are compared with the base recording's actual score rather than a
  fixed zero floor. A take recorded after a vowel can therefore replace a
  consonant-cluster base when the target consonant is also preceded by a vowel,
  even if the context beyond the diphone's other edge differs. A take whose
  only advantage is an exact phrase-edge `pau` remains rejected when its
  spoken-side context mismatches. Diphones ending in a
  voiced sibilant prefer a verified vowel, sonorant, or voiced-continuant
  following context, then unknown OTO context, over a verified stop or
  affricate context; an all-risky inventory retains its base. Manual takes are
  never restricted. The same guards are applied by the GUI at runtime, so
  older generated voices are protected before they are rebuilt. Generate and
  Re-render also discard in-memory generated-bank indexes and reload the
  current metadata. Rebuilding a bank under the same name therefore cannot
  leave numbered take meanings from the previous build in either automatic
  selection or the Recordings menu. If no numbered take has stronger safe
  evidence, the explicit unnumbered `base` row is kept even when metadata rows
  arrive in a different order. New
  conversion also ignores `V C-` coda triphones; explicit `C -` units remain
  valid.
- **Fault Mode** independently offers no phone timing rules, no learned English
  prosody, raw F0 joins, a single phrase pause, no-sustain stretching, and
  monotone output. **Legacy joins** restores the old fixed linear fade in the
  standalone renderer and the paired pre-fix database/bridge geometry with
  symmetric UniSyn windows in rebuilt Festival voices. Normal and Legacy
  pitchmark files currently share the same negative-going zero-crossing
  epochs; the fault never changes contextual/manual take selection and takes
  precedence over a saved join-window request. **Broken
  pitch estimate** gives every eligible phone a bounded 18% corruption chance,
  guarantees at least one fault, and caps a render at five. Each fault is
  restored at that token's boundaries. Its region menu appears only while the
  fault is active. Selecting an amber or magenta fault can pin the exact heard
  Hz plateau; later renders reuse that value verbatim instead of deriving
  another frequency from the phone index. The **Bit depth** submenu provides
  compensated
  8-, 4-, 2-, and 1-bit output; lower modes are progressively quieter, with
  1-bit capped at 9% full scale. An amber badge and the Fault Mode menu count
  remain visible whenever any fault is active.
- **Voicebank uninstall** uses one explicit destructive confirmation before it
  deletes one or several path-backed generated FestVox folders. UTAU `oto.ini`
  folders and Festival built-ins are rejected. If registered folders were
  already removed outside the GUI, or only a stale Festival registration
  remains, their entries are forgotten without file deletion. The manager's
  Delete button operates on the selected rows in table order.

The English `l` default follows John Wells's UCL allophony formulation:
velarized/dark before a consonant (including `/w/`, excluding `/j/`) or a major
boundary, and clear/light before a vowel or `/j/`. In the selector, an outgoing
`l-V` or `l-y` unit is light; for an incoming `V-l` unit the phone after `l` is
inspected, so consonant/pause contexts are dark. This is a configurable bank
choice rather than a claim that every English dialect has the same split; the
Carter and Local production study documents dialect variation. Sources:
[Wells, *Phonology of English: Allophony*](https://www.phon.ucl.ac.uk/home/wells/p201-2-lecture.PDF)
and [Carter & Local (2007), *Journal of the International Phonetic Association*](https://doi.org/10.1017/S0025100307002939).

Per-occurrence choices are transferred as an indexed list and consumed by the
generated `UniSyn_module_hooks` selector after Festival creates its `Segment`
relation. This ordering matters for direct-phoneme re-renders; attempting to
set a segment feature before `utt.synth` produces Festival's `Feature Segment
not defined` error.

## Version 3 workflow and architecture

The UI keeps editing state separate from synthesis backends:

1. Text mode sends each logical phrase through the selected language front
   end. Festival text first captures learned segment timing and F0, then uses
   an explicit `Segments` utterance for corrected pause boundaries, faults,
   and PSOLA targets. Direct Phonemes mode starts with explicit class-based
   durations and uses the same Festival `Segments` path.
2. The Speech editor stores generated F0 separately from pitch overrides.
   Timing, pitch, intonation, and recording-take edits are previewed in the
   timeline; Re-render rebuilds the utterance instead of pitch-shifting a
   finished WAV. Re-render restores the exact waveform viewport and playhead
   after replacing the audio; a fresh Generate keeps its full-render framing.
3. Sentences may route phrases independently by speaker, installed dictionary,
   fault set, timing factors, and pitch override. When an override is present,
   each affected sentence is rendered by its phrase routes and joined in order.
   Ordinary sentences stay on the punctuation-aware backend planner. Every
   normal join is four `pau` segments: an outgoing guard and half-gap followed
   by an incoming half-gap and guard. The first pair belongs to the preceding
   phrase and the second pair to the following phrase. This isolates pause
   resizing from both neighboring phones. Single phrase pause collapses the
   run to one segment.
4. Diagnostic bit depth and Output volume are applied after synthesis. Preview,
   playback, cache, and export therefore agree on output level.

## Waveform and parameter gestures

- The waveform always has a horizontal scrollbar. The mouse wheel zooms the
  shared time axis; parameter tracks use that same axis. Automatic Resize
  changes horizontal framing while keeping the waveform centered at
  `-1.05..1.05` vertically. Follow playhead performs page jumps only when the
  marker exits the view, avoiding continuous recentering. Its inline button at
  the right of the Speech input turns this behavior on or off. Space stops an
  active playback at the current playhead instead of restarting it.
- Left-click selects one phone and left-drag selects a phone-aligned region.
  `Ctrl+C`, `Ctrl+X`, `Ctrl+V`, and `Ctrl+D` preserve the region's phone names,
  current audio chunks, original stretch sources, and timing baselines.
  Shift-drag inside it performs a move with a live translucent preview and
  insertion marker. Its right-click context menu can cut the region to a new
  phoneme sentence. When Broken pitch estimate is active, the menu can pin one
  or more highlighted faults exactly or return to random locations. Pinned
  damage is magenta; random damage is amber.
- A normal boundary drag is a ripple edit. Hovering a boundary explains the
  two gestures; Shift-drag keeps the two-phone outer span fixed. Selecting one
  phone is cosmetic for timing, so its normal
  boundaries remain editable for consonants and vowels alike. Dragging a
  multi-phone selection's right edge changes only its vowels proportionally;
  Shift-dragging its left edge keeps the right edge fixed and lets the adjacent
  phone absorb the boundary movement. An all-pause selection distributes the
  resize across its pauses, so pause boundaries remain fully editable. Later
  phone boundaries follow the live preview exactly once per delta and cannot
  visually overtake the selection.
- Vowels can be preview-stretched indefinitely. Beyond the ordinary WSOLA
  range, the editor preserves attack and release and loops the voicebank's
  unnumbered `X-X` sustain sample, falling back to the rendered vowel center.
  Fault Mode > Long stretches, no sustain samples disables this path.
- Timing has independent Consonant velocity and Vowel length filters. Disabled
  classes are grey and cannot be painted; pause timing remains a boundary edit.
  Japanese moraic nasals use the vowel/rhyme filter, so consonant-only edits do
  not create an unrealistic held `nn`. A plain non-Japanese `nn` remains a
  consonant.
- Pitch curve shows generated F0 underneath the editable contour. It supports
  horizontal wheel zoom, a synchronized scrollbar and a compact vertical zoom
  navigator beside the plot,
  full-width horizontal pitch guides, regional right-drag restore, and a
  Reset pitch curve button. Editing does not recenter its Y axis. Re-render
  retains every dense generated target outside the locally painted control
  interval. Before the first manual edit, the baseline is Festival's returned
  Target relation from the waveform-producing explicit pass, not the earlier
  pre-recenter text-pass contour. This prevents the first local edit from
  shifting the whole sentence. The reference is retimed onto edited phone
  durations and every pause-delimited phrase is anchored independently,
  including `pau pau` joins.
- Recordings hides labels that cannot fit. Right-drag across blocks clears
  occurrence overrides; click still opens arbitrary numbered alternatives and
  exposes the source-recording inspector.

### Long-waveform rendering

Playback and export always retain every original sample. The display uses a
separate level-of-detail cache inspired by Audacity's waveform renderer:

- raw samples are drawn through two samples per pixel;
- from two through sixteen samples per pixel, chronological minima and maxima
  are connected in sample order so useful editing zooms remain continuous;
- wider views draw continuous lower and upper peak envelopes instead of
  thousands of disconnected vertical peak sticks;
- a lazy 16, 32, 64, 128... min/max pyramid selects the nearest level at every
  zoom, keeping source reduction bounded near viewport width without a dense
  intermediate tier;
- only the visible time range is sent to the graphics scene; and
- offscreen phoneme labels, editable fields, and draggable boundary objects
  are not retained while panning. Their segment-index slots remain stable, so
  zooming back in restores the same editing behavior without carrying one Qt
  widget and one graphics object for every phone in a long utterance.

Dense phone and parameter overlays have a second display-only LOD layer:

- when boundaries have less than 18 px each, draggable lines become one compact
  bottom-edge tick per 6 px bucket; pause-adjacent ticks are taller and true
  phrase-splitting double-pause boundaries are tallest and retained first;
- phone labels and editable phone fields each need 24 px, so neither is
  created or laid out when it cannot be read or used;
- Timing, Recordings, and Intonation combine dense entries into one colored
  summary block per 6 px bucket while preserving edited, manual, and strong
  punctuation states; and
- generated and overridden F0 curves are clipped to the visible range and use
  min/max point summaries. Pitch control dots disappear below 10 px per point;
- rendered join markers and crossover spans switch below 10 px per join to one
  representative per 8 px viewport bucket. Selecting or zooming into a bucket
  exposes the exact join and its draggable crossover handles; and
- optional syllable/mora guides use the same viewport-only strategy, retaining
  exact bands when readable and one band per 8 px bucket in an overview.

Zooming in restores the individual boundaries, labels, fields, pitch points,
and parameter blocks automatically. The overview never removes or merges
project data. Pitch loading also interpolates the generated F0 in one cached,
vectorized pass, so long contours do not repeat the full calculation for every
control point. Pitch paint/reset hit-testing uses the same sorted time data to
find the local phone and control point without scanning the whole utterance.
The fixed-width Japanese and Asaxi mora strips likewise paint only cells and
word/accent brackets intersecting the exposed scroll-area region. Japanese
playback no longer requests a full mora-strip repaint every 30 ms because that
strip does not draw the playhead; the waveform remains the authoritative
playhead display. X-linked parameter pages retain their numeric view while
hidden, but defer LOD reconstruction until selected; zooming no longer redraws
all six parameter modes behind the current one.

A bounded offscreen paint check with 2,000 phones and 1,999 visible join
records reduced the waveform frame from about 10.5 ms to 2.6 ms on the
development machine. The exact number is platform-dependent; regression tests
assert the stronger invariant that live controls and overview items are
bounded by viewport width rather than utterance length.

This follows Audacity's documented per-pixel peak-display approach, extended
here with additional power-of-two levels to avoid visible performance cliffs:
[Audacity waveform manual](https://manual.audacityteam.org/man/audacity_waveform.html#rms),
[Audacity WaveDataCache source](https://doxy.audacityteam.org/_wave_data_cache_8h_source.html).

### Application caches

**Options > Application caches** reports approximate process-memory use and
offers four independent commands:

- **Clear audio cache** drops decoded source WAVs, pre-sliced diphones,
  reusable sustain samples, and waveform LOD summaries;
- **Clear voice cache** drops parsed diphone/Festival indexes, alternatives,
  compatibility metadata, and their audio working sets;
- **Clear model cache** drops CMU/kana pronunciation data, canonical Japanese
  frontend results, and duration/pitch/vocal-tract profiles; and
- **Clear all application caches** invokes those three memory-only operations.

Current synthesis, sentence previews, Undo/Redo, project
`cache/sentence_NNNN.wav`, exports, installed dictionaries, generated voices,
source UTAU banks, configuration, and application files are never cache-menu
targets. Local generated-bank metadata is fingerprinted by path, timestamps,
size, and filesystem identity; a generation check prevents an in-flight old
load from being published after invalidation. Referenced generated WAVs carry
their own identity, so replacing one invalidates its decoded and sliced audio.
An unchanged voice remains hot across Generate and Re-render. WSL path-backed
voices invalidate through Reload voicebanks, scan/re-registration, or the
clear command. Exact owner limits, benchmark results, and safety tests are in
`../docs/development/PROMPT0A_SYNTHESIS_EFFICIENCY.md`.

Rendered waveform arrays are shared by sentence previews, phrase previews,
Undo/Redo snapshots, duplicates, and the in-memory clipboard. Editor metadata
is still deep-copied, and a new render replaces rather than mutates its audio
buffer. This avoids multiplying one long waveform across editor history.

Current Festival metadata publication leaves the duplicate alternatives and
alias-audit graphs on disk and caches the compact runtime/index projection.
The complete contextual choice records are loaded by the separate alternatives
owner. The 64 MiB total limit is unchanged: one quarter is reserved for compact
metadata and three quarters for choices. A newly built integrated Lem voice
therefore remains hot across repeated renders instead of reparsing a large
index several times. See
`../docs/development/RUNTIME_CACHE_Q_AND_E3_2026-07-24.md`.

Acoustic diagnostic graphs are lazy. Showing a render does not analyze joins,
build a spectrogram, inspect source pitchmarks, or track formants. Those costs
begin only after the user opens the corresponding diagnostic or explicitly
exports the broadband impulse audit.

## Sentences, files, and caches

The Sentences tab shows one row per project sentence and one draggable phrase
block per punctuation, line break, or strong explicit pause boundary. One
`[pau]` remains an inline hesitation inside its phrase; a consecutive run of
two or more `[pau]` tokens is an explicit phrase boundary. Rendered phrases
include a waveform preview. Phrase widths follow text and duration within
sensible limits, then wrap into as many balanced rows as needed. The speaker
portrait appears once in each sentence header.

Selection is project-wide. Click selects a row or phrase, Shift extends a
range, and Alt-click adds or removes individual phrases across sentence
rows. Dragging a rectangle through the board selects every intersected phrase;
the rectangle may begin anywhere in the blank working area, not only inside a
row. Alt-drag adds to the existing selection. Play All begins with the earliest
selected phrase or sentence and follows project order. During playback the
current sentence and phrase are highlighted. Clicking any sentence or phrase
Play button immediately restarts playback from that item, even while another
item is already playing.

The six-dot row handle and phrase dragging move the whole selected group while
preserving its internal order. A blue insertion line shows the exact drop
position and a translucent image follows the pointer. Right-click selection
menus export generated selections to separate WAV files, remove them after
confirmation, and move sentence groups to a named position. `Ctrl+A`,
`Ctrl+C`, `Ctrl+X`, `Ctrl+V`, `Ctrl+D`, and Delete operate on selected phrases
or sentences in this tab. Duplicate and Paste insert directly after the
selection. Sentence and phrase moves and removals are undoable.

Sentence text is directly editable in a fitted multi-line field. Editing
invalidates only that sentence's stale render. Click its speaker portrait or
name to assign an installed voice. The left sidebar always edits the selected
sentence; with no selection it is disabled and says **Select a sentence to
edit**. **Add sentence** inserts after the selection, or at the project end
when nothing is selected. **Remove selected** stays disabled until a sentence
is selected. Each row's **Edit** button opens it in Speech. Contextual
**+ Phone** and **- Phone** controls are hidden in Sentences because they
operate on Speech phone fields.

Double-clicking a phrase selects its span in Speech. Contiguous selected phrase
blocks can Merge; selected phrases can also be exported or removed together.
The context menu cuts a phrase to a new sentence and assigns phrase-local
faults, bit depth, speaker, or installed dictionary. Speaker portraits are
square, crop to fill, have a visible border, and are installed into the
selected generated voice folder. The Speech portrait follows the exact
selected engine and voice; an image assigned to one speaker is never reused
for an unmapped speaker. **Voicebank > Replace selected speaker icon** removes
old supported `speaker.*` image extensions before installation; **Remove
selected speaker icon** deletes only those icon files after confirmation and
does not touch voice audio or an UTAU source.

Sentences has an all-sentences gain control with the same peak-safe ceiling,
pending state, reset gesture, and explicit clipping override as Speech. Load
text file creates entries at sentence boundaries and can append or replace the
current list. Generate/Re-render All operate in project order. Batch export
can write numbered separate WAVs without overwriting existing files, or one
merged WAV. Export slugs are capped so long text cannot exceed filesystem name
limits. Clear faults affects the current sentence; Project faults can apply or
clear the current set across all sentences. Clear all removes the list only
after confirmation.

Version-4 projects are directories containing `project.json`, `cache/`, and
`exports/`. Saving writes the live sentence preview, not an older synthesis
buffer, to `cache/sentence_NNNN.wav`. Opening the folder's `project.json`
rebuilds the Sentences board immediately and restores available caches. Missing
audio is reported as needing rendering instead of being regenerated silently. Returning
to Speech reconstructs the full waveform, duplicated phone occurrences,
boundaries, and parameter tracks instead of showing a cache-only preview. A
cached project can still open when a synthesis backend is temporarily
unavailable.

Every editor segment has a persisted unique ID. Duplicate and Paste allocate
new IDs; move, Undo, and Redo preserve them. Re-render transfers IDs by matched
occurrence across the returned phone sequence, so repeated labels are not
treated as the same region. A fresh Generate may rebuild the phone plan. Speech
and Sentences therefore show this confirmation before replacing existing
audio: **Generate may reset manual timing, pitch, segment, or recording edits.
Continue?** Cancelling leaves the current structure and audio untouched.

Generate and Re-render commit their returned waveform to the active sentence
immediately. Speech playback, Sentences playback, the waveform, project cache,
and WAV export then read the same preview state. Local timing edits update that
preview immediately and remain marked for Re-render for optimal PSOLA quality.
Structural phrase edits keep an audition preview but remain marked for
Generate because their full-sentence phone map must be rebuilt. A cached
one- or two-pause internal boundary is upgraded to the four-pause model on its
next Re-render. A legacy three-pause timing edit preserves both outer guards
and splits only its middle gap, preserving its exact total duration.

## Future Song work

The former Song runtime and its MIDI dependency have been removed so the
current application has one dependable editing model. The intended future
workflow, including one MIDI track per sentence, overlap repair, the piano
roll, PSOLA pitch targets, and shared Speech timing, is preserved in
`../SONG_MODE_FUTURE.md`. That document is a design brief, not a description
of currently available controls.

## Dictionaries and shortcuts

Install dictionary into selected voicebank parses a supported source once,
writes deterministic `word phone phone` text under the generated voice's
`dic/` folder, and remembers it by engine, voice, and language. Selecting that
combination auto-loads the cleaned dictionary. System Festival voices without a
writable path are rejected. Clearing the active dictionary removes the saved
association; it does not delete the generated voice or touch an UTAU source.

Space plays and `R` re-renders outside text, combo, and number editors;
`Ctrl+R` is a global fresh Generate even while the text input has focus.
`Ctrl+C`, `Ctrl+X`, `Ctrl+V`, `Ctrl+D`, and Delete operate on the active Speech
region or the selected Sentences items instead of maintaining a second set of
mode-specific shortcuts.
**Options > Keyboard shortcuts...** customizes every command, rejects duplicate
sequences and unsafe bare typing keys, and keeps `Ctrl+Shift+Z` as an alternate
Redo unless reassigned. The status strip follows the active/hovered Speech,
parameter or Sentences context. The sidebar corner button collapses or
restores the speaker controls; clicking the square portrait chooses a
project-local image. At compact window heights the sidebar scrolls
independently, so all work areas keep their full vertical space.

## Intonation rationale

Punctuation blocks overlay the generated contour instead of replacing lexical
pitch accents. In particular, the Question block preserves the generated onset
and approaches a phrase-final high boundary; statement and continuation types
likewise blend only near phrase edges. This follows the English ToBI account of
pitch accents plus phrase accents and boundary tones, including final H-H% for
the canonical rising yes/no question. Sources: [Macquarie University ToBI
introduction](https://www.mq.edu.au/faculty-of-medicine-health-and-human-sciences/departments-and-schools/department-of-linguistics/our-research/phonetics-and-phonology/speech/phonetics-and-phonology/Intonation-tobi-introduction)
and [Pierrehumbert, *The Phonology and Phonetics of English Intonation*
(PDF)](https://www.phon.ox.ac.uk/jpierrehumbert/publications/Pierrehumbert_PhD.pdf).

## config.json keys

Selected user-facing keys are settable from the GUI; runtime paths and process
lifecycle limits remain ordinary configuration-file settings.
`engine` ("diphone" / "festival_wsl"); `festival_wsl` {`wsl_exe`, `distro`,
`festival_bin`, `voices` (name → {dir, scm, voice}; dir may be `E:\...` or
`/home/...`), `installed_voices`, `default_voice`, `extra_scheme`,
`timeout_s`, `native_festival_bin`, `persistent_native_runtime`,
`native_runtime_max_jobs`}; `fault_mode` (including `pitch_glitch`,
`pitch_glitch_pins`, `no_sustain_stretch`, and `bit_depth`; an exact pitch pin
records the heard faulty Hz rather than merely the phone index); `parameter_mode`; `output_gain_db`;
`allow_output_clipping`; `follow_playhead`; `show_rendered_joins`;
`speaker_portrait` (legacy fallback before any per-voice portraits exist);
`voice_portraits` (exact engine/voice to its local display image, also installed
in the generated voice folder); `voice_dictionaries`;
`shortcuts` (one validated key sequence per command, including `duplicate`);
plus the diphone keys
(`festvox_config`, `synth_diphone_dir`, `languages`, `extra_voicebanks`,
`advanced`, `diphone_voice_cache_limit`, `diphone_wav_cache_files`,
`diphone_wav_cache_mib`, `diphone_slice_cache_entries`,
`diphone_slice_cache_mib`, `sustain_cache_entries`, `sustain_cache_mib`,
`festival_voice_cache_limit`, `festival_voice_cache_mib`,
`voice_variant_cache_limit`, ...) documented previously.
(The old `apply_velocity`/`velocity_depth` gain-envelope keys are gone — the
former velocity area is now the Parameter Editor, whose Timing mode is saved
per sentence as `timing_factors`.)

For Japanese, fresh Generate uses the versioned contextual mora model and
fitted punctuation gaps. Open JTalk grammatical roles and the phrase-initial
vowel acoustic-edge decision appear in Render Details. These timing rules do
not override a contextual or manual recording choice. Re-render keeps every
current phone duration exactly. `Mora voicing`, detailed `Voicing`, `Pitch
accent`, and `Vocal tract length` remain independent editable parameters;
continuous pitch points are final over generated accent contours.

Open JTalk's generic pause at a non-spoken quote or bracket no longer becomes
a full phrase gap. `「語」は` remains one rendered phrase, while an explicit
comma still uses the configured minor pause. Numeric decimal points remain
inside the spoken number; Japanese middle dots and triangle bullets use minor
and list-item pause strengths respectively. Raw label-pause provenance remains
available in the Japanese utterance diagnostics.

## Tests

From this folder, run:

```bat
python -m unittest test_festvox_core.py
set QT_QPA_PLATFORM=offscreen
python -m unittest test_festvox_gui.py
```

The builder's disposable-bank regression is one directory up:

```bat
python -m unittest discover -s .. -p test_utau2festvox.py
```

Tests create temporary generated databases and WAVs only. They do not build,
modify, or delete the configured UTAU source or a real installed WSL voice.
Project regressions use temporary version-4 folders and cover `project.json`
opening, rejection of older project files, segment-ID round trips, duplicate
survival through Re-render and Undo/Redo, Generate confirmation, immediate
cross-tab playback, and export of the latest audio.

## Sentence switching and generated-audio caching

The Sentences tab keeps sentence audio and synthesis metadata resident without
rebuilding the hidden Speech waveform on every row click. Opening Speech
hydrates the waveform and parameter editor once for the selected sentence.
Sentence navigation also keeps background-generated metadata authoritative:
an older or blank hidden Speech editor cannot overwrite a completed render.
Consequently, **Edit** restores that sentence's timing, pitch, voicing, vocal
tract, intonation, and recording controls without requiring another Generate.
Generated voices built with the current builder also use a UniSyn group file by
default, avoiding repeated file-per-unit source WAV opens in WSL. Older voices
remain compatible and use their separate WAV/pitchmark layout until rebuilt.

If a rebuilt voice still uses separate source WAVs, verify that
`group/<voice>_diphone.group` exists and generated metadata reports
`runtime_audio_storage` as `grouped`. Builds made with `--skip-pm` or
`--runtime-audio-storage separate` intentionally use the compatibility path.

Normal crossover-aware WSL renders also keep the project-local Festival helper
warm between jobs. The backend serializes access, recycles it after 32 renders,
and closes it on native-binary changes, voice metadata invalidation, or
application shutdown. Legacy joins deliberately remains a stock one-shot
Festival process. A July 23 Kal soak produced byte-identical audio for 70 jobs
across three helper processes; warm short renders were about 31 ms, and a
657-phone/56-second render took 2.59 seconds. These timings are diagnostic
fixtures, not latency guarantees for every machine or voice.
For a WSL-only registration, the backend stats generated metadata through the
`\\wsl.localhost` view before reusing cached metadata or the warm interpreter.
It does not hash the large index or launch a WSL metadata process per render.
A blank distro name is resolved once from the registry or one cached
`wsl --list --quiet` probe. Custom native executables are fingerprinted at the
configured path.

Closing the main window releases playback and the warm Festival/WSL helper
exactly once. A close requested during synthesis cancels the remainder of a
batch, waits for the current backend call to finish safely, and then exits;
the hidden helper is not left behind after a normal close.

Generate, Re-render, and their batch variants run blocking backend calls and
source-filter DSP on a synthesis worker. Qt's event loop remains active, so the
window repaints and remains interactive while Festival/WSL is working. The
pointer stays normal, Speech/Sentences tabs remain switchable, and waveform and
sentence-list scrolling stay available. Controls that would invalidate the
in-flight request, plus duplicate Generate/Re-render actions, are temporarily
locked. During a batch, sentence navigation and text remain enabled because
the worker uses a fixed target-state snapshot rather than the visible editor.
Batch Stop
still stops after the current sentence. An individual Festival process is not
force-terminated mid-write and instead finishes or reaches the configured WSL
timeout.

With `Fall = 0`, ordinary Japanese statement punctuation leaves the generated
linguistic F0 untouched. Questions and expressive/continuation punctuation
still use the shared intonation layer, and a nonzero Fall remains an explicit
statement overlay. Automatic voicing continues to protect true pause samples,
but an explicit continuous Voicing curve is editable and final across the
whole waveform, including a `pau` region containing audible unit-edge speech.

## Troubleshooting (Festival/WSL)
- **"wsl.exe not found"** — install WSL or set the executable in the settings
  dialog.
- **"festival not found inside WSL"** — `sudo apt install festival` in WSL;
  or set the full binary path (e.g. `/usr/local/bin/festival`).
- **Timeout on first Generate** — a cold WSL boot plus a Multisyn voice load
  can exceed the default 180 s; raise it in the settings dialog, and prefer
  voices inside the WSL filesystem.
- **"SIOD ERROR: unbound variable voice_..."** — the voice's `.scm` didn't
  define the expected function; check the voice entry in config.json
  (`voice` / `scm` keys) or load a site file via *Extra scheme*.
- **Re-render fails on Festival** — a typed phone isn't in that voice's
  phoneset (Festival is strict; the diphone engine is more forgiving).
- **Integrated `[q]` disappears in English text** - reload or rebuild with the
  current code. Inline phones are validated against the selected generated
  voice's declared inventory; only built-in Kal uses the narrow `radio`
  compatibility list.
- **An automatic recording inserts a gap** - click its Recordings block and
  choose **Inspect selected recording**. The dialog identifies the exact OTO
  row, explains each adjacent OTO edge and the automatic choice, and compares
  its recorded outer context with the utterance. The displayed WAV filename is
  not used to classify context. Re-rendering applies the directional and
  voiced-sibilant guards even to an older generated voice; rebuilding also
  embeds that selector, source slice metadata, and the guarded
  following-transition tail clamp in the bank.
- **GUI freezes during synthesis** - this is not expected. Generate and
  Re-render run on a synthesis worker while Qt remains interactive. Capture
  the status text and backend error if the window stops repainting or scrolling.
- **A project opens with Re-render enabled** - its audition WAV was restored,
  but an edit still needs optimal synthesis. Press Re-render before final
  export. If the status says audio needs rendering, the cache file was absent;
  the GUI leaves the saved segment plan intact instead of silently generating.
- **Re-render changed Japanese phone lengths** - current builds treat this as a
  regression. Re-render preserves every editor segment duration exactly while
  refreshing F0, voicing, recording, and tract metadata. Generate Audio is the
  action that may create a new contextual timing plan.
- **SIOD reports `unbound variable : let*` or `unbound variable : =`** - the
  voice was generated by an older builder whose selector Scheme was not
  portable to this Festival SIOD build. Rerun `build_festival_voice.py`
  with the current source; the generated selector now uses portable nested
  bindings and equality checks. Rebuilding is required because the failing
  Scheme is stored inside the generated voice.
