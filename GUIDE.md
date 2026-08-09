# Building a FestVox diphone voice from a UTAU bank — full walkthrough

This guide takes you from **a set of recorded samples** to **a working
synthesizer** that speaks **English** (the default), **Asaxi**, and **Japanese**,
start to finish. No prior knowledge of Festival or diphone synthesis is
assumed. Japanese CV, VCV, CVVC, mixed-bank, kanji, accent, and refinement
support are documented in
[JAPANESE_UTAU_INTEGRATION_DESIGN.md](JAPANESE_UTAU_INTEGRATION_DESIGN.md).
Use `japanese_utau.py` for read-only bank analysis. The separate Phase 1 text
analysis API and its current limitations are documented in
[JAPANESE_PHASE1_IMPLEMENTATION.md](JAPANESE_PHASE1_IMPLEMENTATION.md); the
current GUI uses this frontend only for isolated Japanese voices. Phase 2
can infer/serialize Japanese bank profiles and compile deterministic, read-only
CV/VCV/CVVC candidate coverage with `japanese_profiles.py` and
`japanese_candidates.py`; see
[JAPANESE_PHASE2_IMPLEMENTATION.md](JAPANESE_PHASE2_IMPLEMENTATION.md). Phase 3
can now compile those candidates into a separate Japanese Festival/UniSyn voice
and create explicit phone timing, F0, and manual-unit plans. See
[JAPANESE_PHASE3_IMPLEMENTATION.md](JAPANESE_PHASE3_IMPLEMENTATION.md). The
corrected CV/VCV/CVVC source assembly, contribution-plan format, bounded CV
fallback, and human comparison corpus are documented in
[JAPANESE_ASSEMBLY_REMEDIATION.md](JAPANESE_ASSEMBLY_REMEDIATION.md). The
Japanese mora/accent workflow is available in the Speech tab. Select Japanese
with Festival/WSL, Generate once, then choose **Pitch accent** in the
Parameter menu. Single-click selects, double-click places a nucleus, and
dragging the existing triangle moves it. Right-click marks a phrase unaccented;
split/merge accent phrases from the mora
controls. Per-mora cent offsets re-render through UniSyn; a user-drawn Pitch
curve remains final. Question shape belongs only to **Intonation blocks**.
Fresh Japanese renders start with punctuation-derived Intonation blocks, so a
Japanese full-width `？` uses the same editable question-rise layer as `?`.
Semantic pause totals are under **Options > Phrase pauses...** and apply with
Re-render. Use
**Voicebank > Analyze Japanese UTAU bank...** for read-only coverage and
unresolved aliases. Profile changes are labeled as requiring a voice rebuild.
For mixed Japanese banks, the selected build method is authoritative: an
explicit CVVC build retains VCV rows for analysis but excludes them from the
generated unit selector. Plain CV plus VC/VV rows are the CVVC method; see
`docs/development/STRICT_CVVC_RUNTIME_SELECTION.md`.
Phase 5 adds optional structural/Open-JTalk-label/external-HTS-JSON baselines,
and generated-voice join diagnostics. Dynamic pitch-bank and voice-color
routing metadata remains available only through the experimental API; the
stable GUI builds and selects separate generated voice configurations. Manual
occurrence choices and the continuous Pitch curve remain final.
See [JAPANESE_PHASE5_IMPLEMENTATION.md](JAPANESE_PHASE5_IMPLEMENTATION.md).

### Japanese Phase 3 command line

Build generated output outside the read-only source bank:

```text
py -3.14 japanese_festival.py <source-bank-or-subbank> <generated-output> \
  --bank-type <cv|vcv|cvvc> --name <voice-name> \
  --pitch 180 --wsl-distro Ubuntu
```

`--bank-type` is required. Read-only analysis may recommend a type, but a
generated voice is always one explicit configuration. Generated metadata
records a source recording bundle separately from that configuration, plus
its primary/supported languages, scoped alias and phone namespaces, and
Festival entry point. The GUI uses those fields to select the primary language
and disable unsupported language choices. Older generated voices are labeled
as legacy metadata and should be rebuilt when practical.

Then create the human-review corpus in an ignored rendered-audio directory:

```text
py -3.14 japanese_listening_set.py <generated-output> <listening-output> \
  --frontend auto --pitch 180 --wsl-distro Ubuntu
```

The listening manifest reports every Festival fallback and warning, the
speaker-relative contour model, punctuation blocks, and every mora/phone
timing allocation. A clean manifest proves that the route is structurally
complete; it does not certify naturalness.

For the focused assembly audit, write exact source-contribution JSON beside
each comparison WAV:

```text
py -3.14 japanese_assembly_listening.py \
  <generated-output> <assembly-listening-output> --wsl-distro Ubuntu
```

Recorded VC/VCV material remains preferred according to the explicit bank
type. A pure CV bank receives deterministic audible bridges where no transition
recording exists; these are always labeled as fallbacks in the manifest.

Inspect joins or run packaging checks without touching a source bank:

```text
py -3.14 japanese_quality.py <generated-output> <plan.json> --output <report.json>
py -3.14 japanese_release.py . JAPANESE_DEPENDENCIES_AND_LICENSES.md
```

`requirements-japanese-optional.txt` pins the tested pyopenjtalk frontend. Kana
and supported romaji remain dependency-free.

---

## 0. The big picture

```
UTAU voicebank                festvox.json            FestVox_DBs\asaxi_lem\
(wav samples + oto.ini)  ──▶  (you edit paths)  ──▶   (the built database)
        record                    configure                  build
                                                                │
                                                                ▼
                                            synth_diphone.py
                                            (turn text into a .wav)
```

Three moving parts, each in its own place:

| Thing | Where it lives | What it is |
| --- | --- | --- |
| **Voicebank** | `D:\UTAU\voice\…\4_Fis3\` | your recordings + `oto.ini` |
| **Config** | `99_Tools\festvox\festvox.json` | the paths you edit |
| **Builder** | `99_Tools\festvox\utau2festvox.py` | UTAU → FestVox DB |
| **Database** | `E:\Portable_Software\FestVox_DBs\asaxi_lem\` | the built voice |
| **Renderer** | `99_Tools\festvox\synth_diphone.py` | text → speech |

The database lives **outside** the voicebank on purpose — the voicebank stays
clean, and you can keep several built voices side by side in `FestVox_DBs`.

---

## 1. Record the voicebank (in UTAU / OpenUTAU)

You need two things in the bank folder:

1. **`.wav` samples** — 16-bit mono. The Lem bank records long strings of
   phones at one pitch (e.g. `b_rr_ch_rr_d_rr_…F#3.wav`).
2. **`oto.ini`** — the timing map UTAU writes. Each line is:

   ```
   filename.wav=ALIAS,Offset,Consonant,Blank,Preutterance,Overlap
   ```

   The **alias** is what matters for diphones. Use the `phone1 phone2`
   convention — a space-separated pair naming the two sounds the clip
   transitions between:

   ```
   b_rr_…F#3.wav=b rrF#3,1025.3,153.2,-287.5,53.0,22.7
                        └── alias "b rr" (the b→rr diphone) + pitch tag F#3
   ```

   - A **single-token** alias (`rr`) is treated as a steady-state sustain and
     becomes the unnumbered `rr-rr` diphone. The GUI uses these `X-X` units for
     indefinite vowel preview stretches, preserving the original attack and
     release while repeating a stable middle. Numbered alternatives remain
     ordinary context choices and do not replace the default sustain.
   - **Silence / breaths:** `-` means silence (→ `pau`); `inh`, `exh`, `br`,
     `BR` are excluded from the database automatically.
   - A trailing dash (`b-`) marks a **word-final** allophone and is kept as a
     distinct unit (`b_`).

### OpenUtau subbanks, pitch suffixes, and voice colors

[OpenUtau's voicebank guide](https://github.com/openutau/OpenUtau/wiki/Voicebank-development)
defines pitch and vocal-mode recordings as subbanks and stores their setup in
`character.yaml`; the importer follows that model.

A bank may keep a zero-byte root `oto.ini` and put the real OTO files in
folders such as `P3_E3/` or `Power/`. The converter reads root and nested OTOs
and resolves every WAV relative to the OTO that names it. Equal WAV basenames
in different subbanks remain separate generated files.

Alias cleanup uses this order:

1. `character.yaml` subbanks (`color`, `prefix`, `suffix`, `tone_ranges`);
2. legacy `prefix.map` rows (`tone`, `prefix`, `suffix`);
3. explicit `--alias-prefix` / `--alias-suffix` values;
4. an ordinary trailing pitch tag such as `E3` or `F#4`.

The order is iterative. If `P` is a manually declared color suffix, both
`ayPE3` and `ayE3P` normalize to `ay`. Exact metadata is safer than guessing:
the builder never strips an arbitrary `P` unless the bank metadata or user
declares it.

When `character.yaml` declares colors, the default/uncolored subbanks are
built by default. Use `--voice-color Headvoice` (for example) to build a
separate color. The stable builder refuses merged `all` colors and merged pitch
subbanks; build each requested pitch/color as a separate voice configuration.
Source pitch/color/tone-range metadata is still saved in
`dic/diphone_index.json` for provenance and future experiments. Experimental
code may opt into deterministic routing explicitly, but the normal GUI never
does so and manual occurrence choices remain final.

> **What "the samples are the same" means for rebuilding:** as long as a new
> recording set uses this same alias scheme and the same `oto.ini` format,
> you rebuild with one command (Step 3) — nothing else changes.

**Aim for full coverage.** For clean synthesis the bank should contain every
`phoneA phoneB` pair you expect to speak, plus `- X` (silence→phone) and
`X -` (phone→silence) for word edges. The Lem bank has all of them (4252
diphones); gaps are covered by fallbacks (see §7) but real recordings sound
better.

---

## 2. Configure `festvox.json`

Copy `festvox.example.json` to the ignored `festvox.json`, then set the paths
for your machine:

```json
{
  "output_root": "E:/Portable_Software/FestVox_DBs",
  "synth_output_dir": "E:/Portable_Software/FestVox_DBs/_samples",
  "default_voice": "asaxi_lem",
  "default_lang": "en",

  "voices": {
    "asaxi_lem": {
      "bank": "D:/UTAU/voice/Lem_V4Bi_Civet/4_Fis3",
      "name": "asaxi",
      "copy_wavs": true
    }
  }
}
```

| Field | Meaning |
| --- | --- |
| `output_root` | folder where databases are built. Each voice builds to `output_root/<voice key>` (here `…/FestVox_DBs/asaxi_lem`). |
| `synth_output_dir` | where the standalone renderer drops `.wav` files by default. |
| `default_voice` | which voice the renderer uses when you don't say `--voice`. |
| `default_lang` | language used when you don't pass `--lang` — `en` (default), `asaxi`, or `ja`. |
| `synth_speed` | default pace when you don't pass `--speed`: `1.0` normal, `>1` faster, `<1` slower. |
| `voices` | one entry per voice. `bank` = the UTAU folder; `name` = the phoneset label baked into filenames; `copy_wavs` = copy the audio into the DB (recommended so the DB is self-contained). Optional `out` overrides the build location for that one voice. |

Each voice may also set `character_yaml`, `prefix_map`, `alias_prefixes`,
`alias_suffixes`, and `voice_color`. Paths are optional because metadata at the
bank root is detected automatically. Point `character_yaml` at a parent file
when `bank` deliberately names only one subbank folder:

```json
"my_headvoice": {
  "bank": "D:/UTAU/voice/MyBank",
  "name": "my_headvoice",
  "copy_wavs": true,
  "character_yaml": "D:/UTAU/voice/MyBank/character.yaml",
  "voice_color": "Headvoice",
  "alias_suffixes": []
}
```

Use forward slashes `/` (they work on Windows too). You can list several
voices and build them all at once.

> Tip: the builder and the renderer both auto-find `festvox.json` — they look
> at `--config`, then `$FESTVOX_CONFIG`, then the current folder, then next to
> the script. So editing this one file is enough.

---

## 3. Build the database

From `99_Tools\festvox\`:

```
python utau2festvox.py                 # builds every voice in festvox.json
python utau2festvox.py --voice asaxi_lem   # just one
```

One-off build without touching the config:

```
python utau2festvox.py --bank "D:/UTAU/voice/…/4_Fis3" --out "E:/…/FestVox_DBs/test" --name asaxi
```

Whole OpenUtau bank, explicit color, or a subbank with metadata elsewhere:

```
python utau2festvox.py --bank "D:/UTAU/voice/MyBank" --voice-color Headvoice
python utau2festvox.py --bank "D:/UTAU/voice/MyBank/P3_E3" --character-yaml "D:/UTAU/voice/MyBank/character.yaml"
python utau2festvox.py --bank "D:/UTAU/voice/LegacyBank" --alias-suffix P
```

`--character-yaml` and `--prefix-map` are file paths. `--alias-prefix` and
`--alias-suffix` are repeatable exact strings. Prefer metadata; use manual
affixes only for incomplete or unusual banks.

You'll see a report like:

```
diphones indexed : 7884
wav files used   : 811
variant takes    : 3632 alternatives preserved
unmapped tokens  : ['-aw11']
```

The result in `E:\Portable_Software\FestVox_DBs\asaxi_lem\`:

```
wav/                       the copied 16-bit samples (renamed to be safe)
dic/asaxi_diphone.scm      Festival index list (diphone wav start mid end)
dic/asaxi_diphone.est      EST index (for real Festival / make_lpc)
dic/diphone_index.json     the index the Python renderer reads
festival/asaxi_diphone_stub.scm   scaffold for a real Festival voice
conversion_report.txt      what was indexed, skipped, or unmapped
```

**Always glance at `conversion_report.txt`.**

- `unmapped tokens` — aliases the builder didn't recognize. A stray one like
  `-aw11` (a typo in the oto) is harmless; a whole phone missing means you
  should add it to `PHONEME_MAP` (see §7) and rebuild.
- `bad oto lines` — malformed lines, **or** timings that came out impossible
  (`start < mid < end` failed). Empty is good.
- `missing wavs` — oto references a file that isn't there.
- `metadata sources`, `voice color`, and `default subbank` — which declarations
  controlled alias cleanup and which color was included.
- `alias cleanup` — counts metadata/manual affixes separately from legacy pitch
  tags. A `WARNING affixes` line means the builder saw likely undeclared
  endings and prints the exact metadata/manual options to use.
- `pitch subbanks` — confirms provenance was retained but dynamic F0 routing
  was not enabled.

---

## 4. Point the tools at the database

**Standalone renderer** — nothing to do; it reads `festvox.json` and resolves
`default_voice` → `output_root/asaxi_lem`.

**Vocab Forge (optional)** — its own `config.json` may point a `festvox_db`
list at the generated output. FestVox itself does not import Vocab Forge code:

```json
"festvox_db": [
  "E:/Portable_Software/FestVox_DBs/asaxi_lem",
  "/sessions/…/FestVox_DBs/asaxi_lem"
]
```

It uses the first path that exists, so you can keep several and reorder. To
switch voices, drop a different DB path at the top of the list.

---

## 5. Synthesize

### Standalone FestVox renderer

From `99_Tools\festvox\`:

```
python synth_diphone.py "the velveteen rabbit"            # English (default)
python synth_diphone.py "Onă Gaksamipỏpỏ" --lang asaxi    # Asaxi
python synth_diphone.py "konnichiwa"       --lang ja      # Japanese
```

Handy options:

```
--lang en|asaxi|ja     language (default: festvox.json "default_lang", = en)
--speed 1.5            pace: >1 faster, <1 slower (default: "synth_speed", = 1.0)
--voice asaxi_lem      pick a voice from festvox.json
--db  "E:/…/some_db"   use a DB directly, ignoring the config
--outdir "E:/…/clips"  where to write (default: synth_output_dir)
--out  "hi.wav"        exact output filename
--config "…/festvox.json"   use a specific config
```

**Speed** is concatenative, not time-stretch: it changes how much of each
recorded phone is used, so it never alters pitch. Faster (`>1`) always works;
slowing (`<1`) is capped by how much audio was actually recorded per phone.
`--speed` overrides the config's `synth_speed` for that one call. An external
consumer such as Vocab Forge may expose the same option when it calls this
renderer.

It prints the phones used, the diphones chosen, and anything skipped (should
be empty), then writes the `.wav`.

### Optional Vocab Forge integration

```
python vocab_forge.py synth "real isn't how you are made"   # English (default)
python vocab_forge.py synth "kozèvkozè" --lang asaxi
```

Or in the web UI's **Anki assets** panel (Add / Edit tabs) press **♪ synth**
on a word or sentence row to render and store the clip on the card (that
button always renders the Asaxi lexeme).

### Languages in the legacy pure-Python renderer

- **English** (default) — CMU dictionary; run `pip install cmudict` once.
- **Asaxi** (`--lang asaxi`) — rule-based from *00_Phonemes of the Asaxi
  Language*; no extra library.
- **Japanese** (`--lang ja`) — kana **or** Hepburn romaji, via
  `en-jap-mapping.yaml` in this folder; no extra library. Kanji is not
  handled on this legacy path. The FestVox GUI's isolated Japanese Festival
  route instead uses the Phase 1 frontend and supports optional pyopenjtalk
  kanji analysis, canonical accent structure, and Phase 3-5 synthesis/editing.

Change the everyday default by editing `default_lang` in `festvox.json`.

> **Upgrading to natural (unit-selection) speech?** This tool is *diphone*
> synthesis. For the Multisyn upgrade — how much speech to record, Audacity
> vs `oto.ini` labelling, the recording/boundary protocol, and the
> `corpus_extract.py` / `labels2festvox.py` tools — see **`MULTISYN.md`**.

---

## 6. Rebuilding & adding voices

- **Re-recorded the same bank?** Just run `python utau2festvox.py` again — it
  remeasures every wav and overwrites the DB.
- **A whole new voice?** Add another entry under `voices` in `festvox.json`
  (its own `bank`, a unique key) and build. It lands in
  `output_root/<new key>`. Point `festvox_db` (vocab_forge) or `--voice` at it.

---

## 7. Adding or remapping phonemes

Open the **`PHONEME_MAP`** block at the top of `utau2festvox.py`. Keys are
normalized alias *tokens*: declared metadata/manual affixes and the ordinary
pitch tag are stripped before the alias is split on spaces:

```python
PHONEME_MAP = {
    "-": "pau",              # silence
    "inh": None, "exh": None, # excluded (breaths)
    "k": "k", "a": "a", ...   # map a token to a Festival phone name
}
```

Rules of thumb:

- Unlisted plain-ASCII tokens map to **themselves**, so you only add entries
  for special cases.
- Numbered takes keep the base phone spelling while every take is indexed with
  its recorded left/right context. Festival and the Python renderer choose the
  closest context automatically.
- A two-token alias whose second token ends in `-` (for example `eh l-`) is a
  coda triphone with embedded silence and is skipped. The ordinary `eh l` and
  explicit `l -` transitions provide the compatible two-part path.
- English `l` context follows the standard clear/dark rule: light before a
  vowel or `/j/`, dark before another consonant or a pause. The implementation
  checks the phone after `l` even when `l` is the right half of the current
  diphone. See [Wells's UCL allophony notes](https://www.phon.ucl.ac.uk/home/wells/p201-2-lecture.PDF);
  [Carter & Local (2007)](https://doi.org/10.1017/S0025100307002939) is the
  dialect-variation caveat. The GUI's per-occurrence override remains the final
  authority for a particular recording.
- Map a token to `None` to **exclude** it (breaths, noise).
- After editing, **rebuild** (Step 3) and re-check the report.

The renderer also has acoustic **fallbacks** (`synth_diphone.py`): missing
`k i` is covered by `k iy` (arpasing convention), Japanese-style CV units fill
other gaps. That's why the test set synthesizes with **0 skipped diphones**
even though not every pair was recorded — but recording the real diphone
always sounds better than a fallback.

---

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No festvox.json found` | run from `99_Tools\festvox\`, or pass `--config PATH`, or set `FESTVOX_CONFIG`. |
| `No diphone DB found` (renderer) | build it first (Step 3); check the path in `festvox_db` / `festvox.json` actually exists. |
| Lots of `unmapped tokens` | your bank uses phones not in `PHONEME_MAP` — add them (§7) and rebuild. |
| No units, or aliases still look like `ayP` | point to `character.yaml` or `prefix.map`; if neither describes the marker, pass exact repeatable `--alias-prefix` / `--alias-suffix` values. The failure message includes examples. |
| Wrong OpenUtau color was included | omit `--voice-color` for the default/uncolored bank, or name one declared color. Build colors as separate voices; `all` deliberately merges their automatic alternative pool. |
| Bank has an empty root `oto.ini` | supported: the converter reads nested OTO files automatically. Check `oto files read` in the report. |
| `bad oto lines` non-empty | malformed oto lines, or a `Blank` sign that made `end ≤ mid`. See the timing note in `utau2festvox.py`. |
| English says `not in CMU dictionary` | that word isn't in cmudict; try another, or spell it phonetically. |
| Robotic / clipped pacing | tune `HALF_MS` (phone length) and `CROSSFADE_MS` at the top of `synth_diphone.py`. |
| Want a real Festival binary voice | use `dic/asaxi_diphone.est` + `festival/…_stub.scm` and run FestVox's `make_lpc` over `wav/`. |

---

## 9. Appendix — how the timing conversion works

UTAU stores **relative milliseconds**; FestVox needs **absolute seconds** with
three points per diphone: where the first phone starts, the boundary between
the two phones, and where the second phone ends.

```
start = Offset
mid   = Offset + Preutterance          (UTAU's alignment point = the boundary)
end   = Offset + |Blank|   if Blank < 0     (length measured from the offset)
      = file_length - Blank if Blank ≥ 0    (milliseconds trimmed off the end)
```

The builder opens each `.wav` with Python's `wave` module to measure
`file_length`, which is the only way to resolve the negative-`Blank` case. It
verifies `start < mid < end` for every entry; anything that fails is reported
as a bad line rather than silently producing garbage audio.

For a VCV phrase-start alias such as `- V`, `mid` is the audible vowel onset.
The generated alias contributes only `pau-V`; it is never recycled as a
medial `V-V` recording. This keeps the pre-vowel silence outside the GUI vowel
region and outside phrase interiors.

When an entry's `end` extends beyond the midpoint of the following OTO
transition, conversion clamps only the generated unit slice to that midpoint.
This prevents the next transition's release or silence from leaking into a
diphone such as `iy-dh`. `raw_end` and `tail_clamped` in the generated index
retain the diagnosis. Rebuild an existing generated voice to receive this
index/audio fix; rebuilding never edits the source UTAU bank.

> Note: this bank uses standard UTAU `Blank` semantics (negative = length from
> offset). The math is documented at the top of `utau2festvox.py`.

---

## Source-bank safety

Treat the UTAU source bank as read-only. Build outputs belong in the configured
FestVox output folder or in WSL, never inside or over the source bank. The GUI
uninstaller rejects `oto.ini` folders and the Lem ground-truth path. Deleting an
existing generated voice requires one explicit permanent-file-deletion
confirmation; several selected generated voices share one consolidated
warning. If a folder is already missing, the GUI can remove its stale saved
entry without issuing a filesystem delete.

---

## Building a real Festival voice - `build_festival_voice.py`

The database above drives `synth_diphone.py` directly; its `festival/` stub is
only an index scaffold. The primary builder now runs from Windows with one
language-scoped command. Give it Windows source, OTO, and output paths; it
derives WSL paths only when invoking Festival or EST.

```powershell
$Builder = ".\99_Tools\festvox\build_festival_voice.py"
$Source = "X:\UTAU\voice\MyBank\F3"
$VoiceRoot = ".\99_Tools\festvox\generated_voices"

py -3.14 $Builder `
  --language en `
  --bank-type arpasing `
  --samples $Source `
  --oto "$Source\oto.ini" `
  --output "$VoiceRoot\my_bank_en" `
  --name my_bank_en `
  --test `
  --test-text "this is a test"
```

Use `--language asaxi --bank-type arpasing` for an Asaxi-primary configuration.
Add repeatable `--enable-language` flags to one ARPAsing build when its inventory
supports multiple frontends; for example, `--enable-language asaxi
--enable-language ja` on an English-primary build. Shared ARPAsing builds use
`profiles/en-jap-mapping.yaml` unless `--phoneme-map` overrides it. Use
`--language ja` with an explicit `cv`, `vcv`, or `cvvc` bank type for a
standalone Japanese bank. See `UNIFIED_VOICE_BUILDER.md` for complete commands,
profile/affix options, overwrite protection, and registration migration.
The default `--source-window-mode adaptive --source-window-ms 60` prevents a
long source half from being compressed into an ordinary short target phone.
Adaptive mode retains hidden full-side variants of the same selected recording
for genuinely stretched phones. Use `--source-window-mode bounded` for a
strictly capped A/B build, or `--source-window-mode full` to restore the legacy
whole-OTO behavior. Changing this policy requires rebuilding the voice.
Every build accepts exactly one OTO file or one explicitly selected
single-pitch folder. Multiple OTO scopes, conflicting pitch tags in aliases or
source WAV names, and automatic multi-folder discovery are refused before an
output folder is created. A single pitch folder may still contain split nested
OTO files.

Generated voices treat `cl` as a language-neutral structural timing phone.
Canonical `V-cl-C` is displayed and edited normally while its source sequence
is `V-C-C`, with a bounded `C-C` consonant hold generated from the selected
single-pitch bank. This also applies when `cl` is typed directly in Phonemes
mode. An explicit OTO alias named `cl` does not opt the bank out of this rule.
Only a creator declaration does:

```powershell
py -3.14 $Builder ... --literal-phone-map cl_literal=cl
```

The generated voice then supports both structural `cl` and authored
`cl_literal`. The mapping is accepted only when the compiled index contains
non-silence incoming `X-cl` and outgoing `cl-X` units. The declaration is
persisted in all runtime manifests. `--special-phone-mode cl=literal` is a
compatibility shorthand for the same additional mapping, not a request to
disable structural `cl`. See `SPECIAL_PHONE_REALIZATION.md`.

Japanese profiles also hold bank-specific moraic-`N` aliases and
following-phone routes; these are never guessed as a universal UTAU naming
standard. In CV banks, `* V` aliases are vowel-blend fallbacks based on their
OTO offset/preutterance/overlap, while `- V` aliases remain phrase starts.

English, Asaxi, and Japanese now share `speaker_pitch.py`. For a build, it reads
only FRQ files adjacent to WAVs referenced by the selected single-pitch OTO
scope. When fewer than three are valid, it estimates short deterministic
windows from those same selected source WAVs. It never samples unrelated
subbanks merely because they share a source root. Generated metadata records
median F0, 10th/90th percentiles, voiced sample count, source-relative files,
method, and diagnostics. With no explicit `--f0`, the voice default is that
measured median with zero automatic headroom; an E3-only Lem scope is
approximately `164.81 Hz`, not `202 Hz`. Older manifests tagged
`speaker_median_plus_headroom` resolve through their stored source median.
Japanese no longer assumes 180 Hz; its baseline is centered on the same source
result used by English. Explicit `--f0`, `--f0-min`, and `--f0-max` remain
final build overrides.

They also share one source-pitchmark policy. A usable UTAU FRQ contour wins for
its recording; otherwise WORLD Harvest (default) or DIO (`--f0-estimator dio`)
estimates speech F0 and StoneMask refines it. Harvest is the quality setting;
DIO trades some voiced coverage for speed. This is independent of the selected
language or phone inventory. Each `pm/*.pm` is accompanied by deterministic
`pm/*.f0.json` analyzed-F0 provenance, so the GUI can distinguish voiced F0
from the default-rate epochs needed to traverse unvoiced audio.

The builders retain OTO-aware phone-center geometry without conditioning the
generated waveform. ARPAsing/Asaxi can use the bounded end of positive OTO
overlap as a left center and chain the preceding unit to the same anchor when
both transitions occur in one recording. Japanese retains its explicit
CV/VCV/CVVC center geometry. Zero-overlap rows preserve their raw offset by
default. `--zero-overlap-guard-ms` is retained only for an explicit diagnostic
A/B build and must be set to a nonzero value; listening validation rejected it
as a default because moving the source onset can damage the following handoff.
Positive overlap is never replaced. Generated WAVs remain unnormalised copies.
Normal WSL synthesis loads the project-local native UniSyn helper, which retains
Festival unit selection and target relations while rendering pitch-synchronous
crossovers. **Fault Mode > Legacy joins** bypasses that helper and restores the
exact stock-Festival waveform path.

Rendered acoustic validation is read-only.
**Generate > Inspect joins and UniSyn windows...** analyzes each exact UniSyn or
pure-Python handoff for independent level, sample/derivative, F0, phase,
period-shape, and spectral-trajectory evidence. Voiced analysis uses rendered
target epochs; unvoiced analysis uses non-crossing short frames. The dialog
keeps the loudness curve as context, ranks joins without hiding component
metrics, offers detailed plots for the selected join, and exports strict JSON.
For Festival/WSL, its Selected Join tab previews the actual rendered crossover.
When the native renderer corrects a measured phase discontinuity between
adjacent mapped periods inside one selected recording, the inspector adds a
**Source trajectory** tab. Downward teal markers locate those epochs in the
overview; the table preserves the source-frame pair, sample shift, correlation
before/after, and reason in render order.
The join list follows phone render order by default, and selecting a Recordings
block focuses the handoff in the middle of that phone. The main waveform can
show the same join through **View > Rendered joins in waveform**. Its green
left/right handles edit the per-occurrence crossover request directly; the
inspector provides the same controls. The label reports requested, rendered,
and context-cap milliseconds because a long request is shortened when it would
consume unsafe consonant or phone-edge context. Timing edits carry the
crossover with its phone-relative position.

The default request is 40 ms, bounded to 0-100 ms and snapped to usable source
and target pitchmarks. The runtime applies complementary raised-cosine windows
without globally normalising naturally different phone levels. It does not
count a fixed number of periods: the requested duration remains stable as pitch
changes. Within a periodic vowel, sonorant, or voiced-fricative source unit, it
also tests adjacent mapped periods with non-wrapped correlation. A source-frame
centre moves by at most one quarter of the local period only when correlation
starts below `0.82`, reaches at least `0.90`, and improves by at least `0.15`.
Short correction runs touching a crossover are rejected; longer runs taper
toward it. Contextual and manual recording selection runs unchanged before
concatenation, and changing a crossover marks the waveform for Re-render
without regenerating phone timing, F0, or unit choices. **Legacy joins**
disables both the crossover and same-unit phase correction and reproduces the
stock-Festival waveform byte for byte. No post-render PCM repair is applied.
Setting the sentence default to `0 ms` with no positive occurrence override is
an explicit no-crossover control and uses one-shot stock Festival; a positive
occurrence override still invokes the native helper. Unlike Legacy, this does
not also force historical source-window settings or discard overrides.
The older source-window radius/method controls remain available for diagnosing
where each unit is cut, but they do not replace the per-occurrence rendered
crossover length. See
[JOIN_DISCONTINUITY_DIAGNOSTICS.md](JOIN_DISCONTINUITY_DIAGNOSTICS.md).

The Japanese structural contour uses semitone components relative to that
speaker center and bounded range: breath-group reset, phrase-initial accent
shape, nucleus fall, saturated downstep, sentence declination, and phrase-final
lowering. It does not create an interrogative-rise target. The authority order
is structural speaker baseline, pitch-accent edits and per-mora offsets,
general Intonation blocks, continuous Pitch points, then the shared 50-500 Hz
safety bound. Continuous points are never silently replaced.

Japanese timing is mora-first rather than the sum of unrelated fixed phone
durations. Ordinary CV, vowel-only, palatalized, geminate `cl`, moraic `N`, long
vowel, devoiced-vowel, and phrase-final morae have separate allocation rules;
Speed scales timing independently of F0. Generated plan schema 2 stores, for
each mora and phone, predicted duration, source reference, source-safe minimum
and maximum, requested stretch, and final duration. OTO slice geometry supplies
the source estimate when available, while OTO offset, preutterance, overlap,
consonant, and cutoff remain alignment landmarks rather than linguistic target
durations. **Generate > Render details...** exposes the same timing rows.

The source bank is opened read only. A generated voice gets pitchmarks, a
UniSyn EST index, one language entry point, `dic/diphone_index.json`,
`dic/unit_alternatives.json` where applicable, and a portable
`dic/voice_manifest.json`. The manifest separates recording-bundle identity
from configuration identity, so English and Japanese interpretations of the
same WAV inventory cannot be mistaken for one multilingual voice. Generated
Scheme resolves its root from Festival's load path instead of baking in the
destination directory.

`diphone_geometry_model` and each recording alternative retain raw OTO
landmarks plus the selected left/right center method. Generated WAV copies are
not level-normalised or join-tapered. Normal WSL renders use the native helper;
direct Festival and **Legacy joins** use stock `UniSyn`. Use **Generate >
Inspect joins and UniSyn windows...** for level, phase, period-shape,
spectral-trajectory, and broadband-impulse evidence at rendered handoffs.
Per-occurrence crossover edits are shared by the inspector and waveform view,
apply on the next Re-render, and never replace contextual/manual unit
selection.

The selector uses conservative SIOD syntax supported by Ubuntu Festival and
avoids extended `let*` bindings and numeric `=`, which are unbound in some
SIOD builds. Rebuild old voices that report either error. `--test` uses fresh
temporary artifacts and requires Festival to return a nonempty WAV. Japanese
tests render the canonical phone, duration, F0, and unit-override plan rather
than testing metadata alone.

The build also installs its final `dic/diphone_index.json` after silence and
onset repairs. The Windows editor reads this generated copy to locate unnumbered
`X-X` sustain regions for indefinite vowel previews; it never needs to write to
the UTAU source bank. For an older generated voice, it can read the same index
and WAVs from the legacy `db/dic/` and `db/wav/` layout without altering them.

The Windows GUI adds generated F0 capture, phrase-edge pitch controls,
punctuation-block previews, protected phrase pauses, and a visible Recordings
view for the actual automatic or per-occurrence take. **Options > Phrase
pauses...** edits minor, major, and sentence totals in milliseconds. Each
internal break has four parts: an outgoing guard and outgoing half-gap owned
by the preceding phrase, followed by an incoming half-gap and incoming guard
owned by the next phrase. Those internal parts are not exposed as settings;
the setting remains one semantic total. These edits are
re-rendered through Festival/UniSyn rather than pitch-shifting finished audio.
Pitch edits remain relative to the separate generated contour. Segment IDs
keep those deviations attached to surviving phones across timing changes,
insertions, and deletions; deleting one phone never total-scales later F0.
Pause targets are rebuilt canonically and an already-rendered baseline is not
recentered again on an ordinary Re-render, preventing cumulative contour drift.
Generate and Re-render reload generated-bank take and sustain indexes first,
so replacing a voice under the same registered name cannot reuse old `takeN`
meanings. Automatic selection compares each ordinary take with the explicit
base's actual context score. This lets an incoming vowel-context take beat a
base recorded in a consonant cluster when the target also follows a vowel,
while rejecting a take whose only advantage is a phrase-edge pause. When no
take improves safely, the unnumbered `base` remains selected regardless of
metadata order.

In Recordings, a compiled vowel-to-vowel edge is labeled `VV`; `VCV` remains a
valid UTAU source-bank family/profile name, not the name of that runtime
diphone. **View PSOLA source pitchmarks...** opens the generated unit actually
used by UniSyn. Orange points show the persisted analyzed voiced contour, gaps
remain visibly unvoiced, and a dashed line shows the filled epoch rate PSOLA
uses through both voiced and unvoiced regions. Source WAV and PM files are read
only from the generated voice, never from the original UTAU bank.

Its version-4 project format uses a dedicated folder with `project.json`,
`cache/`, and `exports/`. **Open Project JSON** selects the `project.json`
inside that folder. Pre-version-4 project JSON is not currently supported by
the GUI. Stable segment IDs preserve duplicated Speech regions through
Re-render, save/load, Undo, and Redo. A fresh Generate warns that it may reset manual timing, pitch,
segment, or recording edits. The returned waveform is committed immediately
as the shared Speech/Sentences/cache/playback/export preview.
Generate/Re-render All operate in project order; Export Batch writes separate
numbered WAVs or one merged file. Batch work does not select the sentence being
rendered: the active row/tab remains under user control, and results commit to
their target sentence directly. A sentence edited during its own in-flight
render keeps the edit and discards that stale result. The Sentences tab
supports editable text,
phrase reorder, cut,
merge, sentence/phrase speakers, dictionary routing, faults, direct playback,
board-wide rectangle/Ctrl/Shift/Alt multi-selection, insertion-marked group
drag ordering, selected WAV export/removal, and `Ctrl+C`/`Ctrl+X`/`Ctrl+V`/
`Ctrl+D` item editing. Speech regions use the same shortcuts and Shift-drag
move preview. Rectangle selection can begin in blank Sentences workspace.
`Ctrl+R` generates selected sentence rows, or all rows with no selection,
without playback. Enter in a row generates it; Shift+Enter inserts a line
break interpreted as another phrase inside that sentence, and the field grows
to a bounded scrollable height.
**Follow spoken sentence** scrolls the board only when playback enters a new
sentence and can be disabled independently of Speech-tab playhead following.
An initial Generate All stores every returned sentence waveform immediately.
Repeated identical failures in one batch are collected into one summary dialog
instead of forcing one acknowledgement per sentence.
Speech and Sentences gain
controls are disabled until audio exists, clamp their upper bound to measured
headroom by default, and display pending changes until Generate or Re-render
applies them. Re-render-required timing changes use a restrained blue waveform
with red boundaries and highlight the action yellow.
Generate-required changes color the Generate button only. Sentences rows use a
slightly desaturated neutral background with no Generate-pending badge, while
the Generate button remains yellow. Reverting text to the last rendered value
restores the normal row. **Allow clipping** explicitly restores the full +12 dB
range.
Fault Mode includes
single-pause comparison as well as bounded multi-phone pitch-estimation
corruption, exact heard-frequency pins, sustain-loop comparison, and
compensated 8/4/2/1-bit quantization. The pitch pin menu appears only while the
fault is active. Voicebank menu actions replace or remove installed speaker
icons, and replacement removes obsolete `speaker.*` extensions.
**Options > Application caches** shows approximate in-memory Audio, Voice, and
Model usage. Clearing Audio drops decoded recordings, reusable slices,
sustains, and waveform summaries; clearing Voice drops parsed generated-bank
metadata; clearing Model drops pronunciation/front-end and profile data. The
next operation reconstructs these values. The commands never delete a source
bank, generated voice, project, project WAV cache, export, dictionary,
configuration, or application file. Cache limits and the matched benchmark are
recorded in `docs/development/PROMPT0A_SYNTHESIS_EFFICIENCY.md`.
The previous Song implementation is intentionally absent. Its future rebuild
goal and MIDI import contract are recorded in `SONG_MODE_FUTURE.md`. See
`festvox_gui/README.md` for every active control, project-cache behavior,
dependencies, and voicebank-format compatibility.

Numbered OTO choices use conservative two-sided context scoring. Context comes
only from adjacent, time-ordered aliases in the same `oto.ini` recording; WAV
filenames are never parsed for phonemes. A missing adjacent transition is
unknown `*`, not silence. Exact symbols are preferred, then compatible
articulatory classes are scored across languages (vowel, silence, voicing,
manner, liquid/nasal/glide). Strict CV aliases are classified by the edge next
to the target: preceding `ka` contributes `/a/`, while following `ka`
contributes `/k/`. Literal aliases that cannot be decomposed remain
unclassified and do not broad-match one another.

For a diphone ending in voiced `z`, `zh`, `zi`, `dz`, or `jh`, the automatic
selector first looks for a following context verified as a vowel, nasal,
liquid, glide, or voiced continuant. If none exists it prefers unannotated OTO
context over a verified stop/affricate context; if all choices are risky it
retains the base. The ordinary context score resolves choices inside the
winning tier. This protects voiced sibilants from stop-conditioned recordings
that contain contextual devoicing while keeping every manual take available.

A match on a phrase-edge `pau` cannot by itself select a take recorded in an
unrelated spoken context. Existing voices receive these guards from the GUI at
render time; new builds embed the same SIOD-compatible rules and persist
`context_model: oto_directional_v1`. UTAU `V C-` coda triphones are ignored
because their silent tail would precede a second `C-pau` unit and cause a
post-silence blip. The Recordings inspector identifies the exact alias, WAV,
OTO line, directional edge evidence, selection reason, slice, and score; it
also states that WAV filenames were ignored. For Japanese, **Inspect selected
mora contributions...** lists every incoming/outgoing source role, alias, WAV,
source interval, target edge, and visible fallback that assembles that mora.
Manual choices follow unchanged
diphones across phone insertions or deletions.

For a generated voice, **Dictionary > Install dictionary into selected
voicebank** parses the source and writes a deterministic cleaned dictionary
under that generated voice's `dic/` directory. The GUI remembers the file by
engine, voice, and language and auto-loads it when that combination is selected.
This operation never writes into the UTAU source bank.

Keep current generated voices as immediate children of the Windows root
configured in **Options > WSL / Festival settings**. An optional WSL root can
be configured beside it. **Voicebank > Voicebank manager...** refreshes both;
folders added or removed outside the program appear or disappear on refresh.
Only auto-discovered registrations are cleaned up, and only after that root was
successfully read. Manual registrations and entries on temporarily unavailable
roots remain intact. The manager also adds individual Windows/WSL folders and
supports multi-selected generated-voice deletion after one consolidated
Delete/Cancel warning.

The standard Kal voice remains an English-only built-in at
`/usr/share/festival/voices/english/kal_diphone`; an automatically mirrored
Windows copy is optional and cannot shadow or remove that entry. Legacy combined
Asaxi/English Scheme remains readable.
Current builds instead generate one `voice_<name>` for English or Asaxi, or a
distinct `voice_<name>_ja` for Japanese. Japanese consumes the canonical
Japanese frontend model and never routes through the English phoneset.

## Contextual Japanese timing and voicing

Fresh Japanese generation uses the source-relative contextual duration model
by default. Change it under **Options > Japanese duration model**; **Legacy
mora timing** preserves the older allocator. The selected recording remains
the absolute timing baseline. Open JTalk/canonical phone, mora, accent, and
boundary context contributes only bounded residuals, and Speed uses
class-specific elasticity rather than scaling every phone identically.

Moraic `/N/` is a timing nucleus with its own bounds. Integrated aliases such
as `nn`, `nng`, `mm`, and `xn` therefore follow rhyme/vowel timing controls and
are excluded from consonant-only stretching. This role applies only to a
canonical Japanese moraic nasal; an ordinary `nn` in another language remains
a consonant.

**Options > Japanese vowel devoicing** chooses contextual realization or the
legacy duration-only behavior. **Renderer** selects automatic
natural/source-filter handling, the source-filter residual path explicitly,
or the accurately named shortened-voiced fallback. Direct waveform TD-PSOLA
still controls duration and F0; it does not remove periodic excitation from a
voiced recording. The source-filter stage performs that separate realization
after UniSyn returns explicit segment boundaries.

The Speech parameter menu's **Voicing** view is available for rendered audio
in every language. Its dashed curve is the measured automatic analysis and
remains stable after regeneration. The editable curve is the final authority,
with roughly 32 ms analysis granularity and 8 ms hops. A value near zero uses
continuous shaped stochastic excitation through the measured tract envelope;
it is not a simple gain fade or white-noise replacement.

Generate All and Re-render All show completed/total progress in the status bar.
The adjacent stop button halts after the currently running synchronous
Festival sentence and keeps already completed sentence audio. Navigation,
tab switching, and text editing remain available during the batch because the
worker renders a fixed sentence-state snapshot instead of the visible editor.

For implementation details, fitting commands, safety bounds, corpus/licensing
requirements, and objective results, see `PROMPT19_IMPLEMENTATION.md`,
`docs/japanese_duration_model.md`, and `PROMPT19_BENCHMARK_REPORT.md`.

## Prompt 20 prosody and vocal-tract controls

Current Japanese generation uses speaker-relative log F0 with separate lexical
accent, phrase contour, reset, declination, boundary, and question components.
Later repeated phrases vary in contour shape without cumulative frequency
drift. The active pitch and duration model IDs and the no-drift status are shown
in Render Details.

Japanese commas, major punctuation, and sentence endings use fitted middle-gap
defaults while retaining two protected guards. Open JTalk grammatical nodes are
kept on each mora. The current model shortens ordinary and negative auxiliaries
slightly; particles and polite forms are visible in diagnostics but are not
adjusted when the evidence is unstable. Phrase-initial vowel units receive a
bounded acoustic-edge correction because their audible vowel often begins
before Festival's logical Segment boundary. This never replaces the selected
recording. A general phrase-final correction is deliberately absent because
the rendered held-out comparison showed that it made effective mora timing too
short.

Open JTalk quotation and bracket labels are reconciled with the source before
pause planning. A non-spoken boundary such as `」は` keeps its raw `pau`
provenance and accent separation but does not become a rendered phrase gap.
An explicit comma at the same mora boundary remains audible. Decimal points
between digits are spoken as part of the number; `・` is a minor list pause and
`▽` is a list-item pause.

**Re-render Phonemes** always keeps every current segment duration exactly,
including manual timing edits. It rebuilds F0, voicing, and unit metadata on
that timeline. Use **Generate Audio** when a fresh contextual duration plan is
actually wanted.

The Speech Parameter Editor's **Vocal tract length** curve shifts the spectral
envelope independently of pitch and duration. A ratio above `1.0` lowers
resonances; a ratio below `1.0` raises them. **Chipmunk range** expands the same
bounded control and defaults off. Reset returns the exact identity waveform.
The formant diagnostic analyzes the final synthesized waveform, not only an
internal envelope target. This is resonance control, not complete biological
gender conversion. See `PROMPT20_IMPLEMENTATION_REPORT.md` for data provenance,
measurements, commands, and known limitations.
