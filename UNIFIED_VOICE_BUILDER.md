# Unified Festival Voice Builder

`build_festival_voice.py` is the primary front door for generated Festival
voices. Run it from Windows with Windows-visible paths. The builder invokes EST
and Festival locally when present, or through WSL after deriving `/mnt/...`
paths internally.

The source UTAU bank is read only. Converted WAVs, pitchmarks, indexes, Scheme,
reports, and smoke-test audio are written only beneath `--output`.

Install the language-independent FRQ-less fallback once with:

```powershell
py -3.14 -m pip install -r .\99_Tools\festvox\requirements-source-f0.txt
```

## Required choices

Every build declares these independently:

- `--language ja|en|asaxi`
- `--bank-type cv|vcv|cvvc|arpasing`
- `--samples` for the read-only recording root
- exactly one explicit `--oto` file or single-pitch folder
- `--output` for one generated voice folder
- `--name` for a Festival-safe generated name

Source-window behavior is optional and language-independent. The default is
`--source-window-mode adaptive --source-window-ms 60`; see
[Source-window modes](#source-window-modes).

Japanese-only banks accept `cv`, `vcv`, or `cvvc`. ARPAsing builds accept an
English or Asaxi primary language and may add explicit frontends with repeatable
`--enable-language en|asaxi|ja`. Enabled frontends share one unit database and
one profile-scoped alias namespace, but retain distinct Festival entry points
and GUI language compatibility.

Do not append `_ja` to a Japanese `--name`; the Japanese compiler adds that
suffix to its Scheme symbol and entry point.

## Windows examples

Set paths appropriate to the local machine:

```powershell
$Builder = ".\99_Tools\festvox\build_festival_voice.py"
$VoiceRoot = ".\99_Tools\festvox\generated_voices"
$Source = "X:\UTAU\voice\JapaneseBank\P3_E3"

py -3.14 $Builder `
  --language ja `
  --bank-type cvvc `
  --samples $Source `
  --oto "$Source\oto.ini" `
  --output "$VoiceRoot\japanese_bank_p3_e3" `
  --name japanese_bank_p3_e3 `
  --f0-estimator harvest `
  --test
```

English ARPAsing:

```powershell
$Source = "X:\UTAU\voice\EnglishBank\F3"

py -3.14 $Builder `
  --language en `
  --bank-type arpasing `
  --samples $Source `
  --oto "$Source\oto.ini" `
  --output "$VoiceRoot\english_bank_f3" `
  --name english_bank_f3 `
  --f0-estimator harvest `
  --test `
  --test-text "this is a test"
```

Asaxi uses the same ARPAsing source interpretation but receives a distinct
configuration and front end:

```powershell
py -3.14 $Builder `
  --language asaxi `
  --bank-type arpasing `
  --samples $Source `
  --oto "$Source\oto.ini" `
  --output "$VoiceRoot\asaxi_bank_f3" `
  --name asaxi_bank_f3 `
  --test `
  --test-text "taki"
```

Fully integrated English, Asaxi, and Japanese ARPAsing voice:

```powershell
$Source = "X:\UTAU\voice\IntegratedBank"

py -3.14 $Builder `
  --language en `
  --enable-language asaxi `
  --enable-language ja `
  --bank-type arpasing `
  --samples $Source `
  --output "$VoiceRoot\lem_v4bi_integrated" `
  --name lem_v4bi_integrated `
  --test `
  --test-text "this is a test"
```

The integrated build uses `profiles/en-jap-mapping.yaml` by default. Override
it with `--phoneme-map PATH`. The profile declares the shared phone inventory,
bounded relative timing weights, kana/grapheme mappings, and bank-specific
moraic-nasal targets. Existing ASCII ARPAsing aliases remain authoritative;
the profile does not globally reinterpret `PHONEME_MAP`. Mapping conflicts and
sequences that cannot form one diphone remain visible in generated metadata.

Open JTalk label-derived morae also resolve through this profile. If its
all-kana reading cannot be aligned, the planner recovers a stable hiragana key
from the canonical Japanese phone tuple and then applies the voice's mapping.
For the bundled Lem profile this keeps the Japanese tap on `dx` (`dxy` before
`i` and in palatalized contexts) instead of leaking canonical `r` into the
integrated phone sequence. Japanese-only banks remain in their canonical
namespace because they do not carry an ARPAsing `japanese_phoneme_map`.

The Japanese-only `--profile` option is different: it points to a Japanese
CV/VCV/CVVC bank-analysis profile JSON. `--phoneme-map` is for a shared
ARPAsing build.

Festival and `speech-tools` must be installed in the selected WSL
distribution. The default is `Ubuntu`; override it with `--wsl-distro`.

## OTO scope and bank metadata

Pass exactly one `--oto`: either one `oto.ini` file or one folder representing a
single pitch. A folder may contain split nested `oto.ini` files, but the builder
refuses multiple `--oto` arguments, automatic multi-folder discovery, and a
folder containing conflicting pitch tags. Every selected OTO must remain inside
`--samples`. Build each pitch as a separate generated voice; automatic
multipitch merging is deliberately disabled.

Pitch tags use standard uppercase UTAU note spelling (`E3`, `F#3`, and so on).
The validator checks the final note suffix in each source-WAV and alias field
separately. Earlier numbered phoneme takes such as `a11E3`, `b1PE3`, or
`E1PE3` therefore resolve to the final `E3` instead of masquerading as extra
pitches.

This check is enforced by the unified command, the compatibility command, and
the conversion API. Automatic speaker-pitch analysis is likewise restricted to
WAV recordings referenced by the selected OTO scope and their adjacent FRQ
files; unrelated pitch folders cannot influence the generated F0 range.

English and Asaxi accept `--character-yaml`, `--prefix-map`, repeatable
`--alias-prefix`, repeatable `--alias-suffix`, and one `--voice-color`. The
converter still prefers `character.yaml`, then `prefix.map`, when these are
auto-detected. A merged `--voice-color all` build is intentionally rejected by
the stable front door; build separate generated voices for separate colors.

Japanese profile and alias decisions belong to the Japanese profile layer.
Use `--profile` for an explicit profile. The front door still requires the
chosen `--bank-type`, so inference cannot silently turn a CVVC build into VCV
or CV.

## Structural and literal special phones

Generated voices always retain language-neutral structural `cl`. Canonical
`V-cl-C` is sourced as `V-C-C`, with a bounded generated `C-C` consonant hold;
this behavior is independent of the selected frontend language and also
applies to direct Phonemes input.

An OTO alias named `cl` is not a declaration of linguistic meaning. To expose
authored `/cl/` units without losing structural closure, choose a distinct
canonical token:

```powershell
py -3.14 $Builder ... `
  --literal-phone-map cl_literal=cl
```

The resulting voice accepts both `cl` and `cl_literal`. The builder requires
non-silence incoming `X-cl` and outgoing `cl-X` units and rejects a display
name that collides with a source phone. `--special-phone-mode cl=literal`
remains a compatibility shorthand for the same additional
`cl_literal=cl` mapping; it does not replace structural `cl`. The mapping is
written to the runtime index, alternatives metadata, and portable manifest.
See `SPECIAL_PHONE_REALIZATION.md`.

An explicit `--bank-type cvvc` is a strict runtime contract, including for a
mixed CVVC/VCV source bank. The generated voice uses ordinary CV, phrase-start
CV, optional `* V` vowel blends, recorded VC/VV transitions, and releases. It
does not expose VCV mora aliases such as `a か`, `a の`, or `i な` as Festival
alternatives. Those OTO rows remain intact and traceable in candidate metadata,
where they are marked non-selectable. ASCII two-phone `V C` and `V V` aliases
remain CVVC transitions; a profile-declared moraic-nasal alias is never guessed
into a consonant. Use `--bank-type vcv` when VCV mora recordings are intended.
See `docs/development/STRICT_CVVC_RUNTIME_SELECTION.md` for the policy and
real-bank validation.

Generate an editable profile first, then pass it back to the builder:

```powershell
py -3.14 .\99_Tools\festvox\japanese_profiles.py $Source `
  --bank-configuration cvvc `
  --output .\my-japanese-profile.json
```

Bank-specific moraic-nasal realizations belong in the generated profile's
`moraic_nasal_allophones` object. The IDs and aliases are defined by the bank,
not by this example:

```json
{
  "moraic_nasal_allophones": {
    "coronal": {
      "mora_aliases": ["ん"],
      "context_aliases": ["n"],
      "following_phones": ["t", "d", "n", "r"],
      "default": false
    },
    "labial": {
      "mora_aliases": ["んm", "ん2"],
      "context_aliases": ["m"],
      "following_phones": ["p", "b", "m", "f"],
      "default": false
    },
    "velar": {
      "mora_aliases": ["んng", "ん3"],
      "context_aliases": ["ng"],
      "following_phones": ["k", "g"],
      "default": false
    },
    "bank_final": {
      "mora_aliases": ["んn", "ん1"],
      "context_aliases": ["nn"],
      "following_phones": ["s", "sh", "z"],
      "default": true,
      "note": "Example only; confirm this bank's documentation and audio."
    }
  }
}
```

An alias may occur in only one group, and at most one group may be the default.
Configured numbered aliases remain allophones rather than numbered takes.
Unconfigured allophone-like rows stay visible in diagnostics. The selector
uses the same configured group on the incoming and outgoing side of `/N/`,
while a manual per-occurrence recording choice remains final.

For CV banks, `- V` is a phrase-start alias. `* V` is instead treated as a
vowel-blend fallback: its offset is the audible vowel onset and its OTO
preutterance/overlap geometry controls a bounded crossfade when no explicit
vowel transition exists. Exact VCV or VC recordings are preferred whenever
available.

For VCV banks, a phrase-start `- V` contributes only `pau-V`. Its phone
boundary is exactly `Offset + Preutterance`, because UTAU preutterance marks
the vowel onset. Offset and overlap are retained as source context but are not
used as the visible vowel boundary. The phrase-start alias never supplies a
medial `V-V` unit; ordinary CV/VCV rows supply medial and sustained vowels.

Uppercase `R` and `RB` are common rest/rest-plus-breath tokens in CVVC OTOs.
Context-prefixed forms such as `a RPE3` and `a RBPE3`, plus numbered forms such
as `a R1PE3` and `a RB1PE3`, remain traceable non-speech sources but are not
selectable speech candidates. Case is significant: these are never shortened
to lowercase `r`; a real tapped-r transition must use an alias such as
`a rPE3`.

Both language routes run the same source-speaker analysis before dispatch.
Valid recursive UTAU `FREQ0003` files are preferred; the median of their
header averages preserves the established English base-pitch behavior, while
their voiced frame table supplies the 10th/90th-percentile range and voiced
sample count. Fewer than three usable FRQ files triggers deterministic
autocorrelation over a spread of source WAVs, followed only then by the
documented conservative fallback. The parser follows OpenUtau's
[`FREQ0003` reader](https://github.com/stakira/OpenUtau/blob/master/OpenUtau.Core/Classic/Frq.cs).

`--f0`, `--f0-min`, and `--f0-max` explicitly override the effective build
pitch or pitchmark search range. They do not replace the measured source
statistics in metadata. `--skip-pm` is a structural/debug build and suppresses
the acoustic smoke render.

Without `--f0`, `average_pitch_hz` is the selected OTO scope's measured speaker
median. No automatic melodic-headroom transposition is added. For example, the
Lem V4Bi `3_E3` scope resolves to about 164.81 Hz, not 202 Hz. New manifests
record `default_pitch_source: speaker_median` and
`automatic_pitch_headroom_semitones: 0.0`. The GUI also recognizes the older
`speaker_median_plus_headroom` metadata and recovers its stored speaker median,
while preserving an explicit `builder_override`.

Pitchmark source analysis is identical for English, Asaxi, integrated
ARPAsing, and Japanese-only builds. Per-recording UTAU FRQ data is used first.
If that recording has no usable FRQ, `--f0-estimator harvest` (the default) or
`--f0-estimator dio` analyzes its speech waveform with WORLD and StoneMask.
Harvest favors voiced coverage and low-SNR robustness; DIO is the faster
choice for large clean banks. The option is a fallback only and never replaces
valid FRQ data.

Every generated `pm/*.pm` has a deterministic neighboring `*.f0.json` file
containing the exact sanitized analyzed contour used to place its PSOLA epochs.
Epochs use a consistent negative-going, low-pass zero-crossing convention.
Every normal file also has an exact `*.legacy.pm` timeline; the two timelines
are intentionally identical in the current build. The earlier independent
residual-epoch experiment was rejected because recordings did not share one
reliable residual phase. `pm/pitchmark_sources.json` schema 2 records
FRQ/WORLD provenance, the active epoch method, and the legacy file per
generated WAV.
The old EST `pitchmark`/`pda` route is not used on ordinary speech waveforms;
EST's pitchmark utility is intended for laryngograph/EGG input and can lock to
the wrong harmonic for an entire syllable.

## OTO-aware phone-center geometry

UniSyn joins `A-B` and `B-C` at the center of their shared phone B. The builder
retains the source landmarks needed to explain those centers, but does not
normalise or taper generated WAV copies.

The ARPAsing converter also maps positive OTO overlap into those center cuts.
Its left edge is the bounded end of the OTO overlap. When the following `B-C`
transition exists later in the same recording, `A-B` ends at that exact same
anchor; otherwise it uses a valid OTO fixed-region end and finally the OTO
region end as a visible fallback. Japanese retains its role-specific CV, VCV,
and CVVC center geometry.

An OTO overlap of zero retains its raw-offset geometry. The default is
`--zero-overlap-guard-ms 0`. A nonzero value remains available only for an
explicit diagnostic A/B build, capped at one quarter of preutterance. Listening
validation rejected inferred source-cut movement as a default because it can
damage the following handoff even when a discontinuity metric improves.
Positive OTO overlap remains authoritative, and generated metadata stores the
raw overlap, effective overlap, method, and policy separately. The builder does
not normalize gain, replace contextual candidates, or repair rendered PCM.

Generated voices remain ordinary Festival `UniSyn` databases. In the GUI's
normal WSL route, the project-local native helper consumes Festival's completed
Unit map and performs the final millisecond crossover; direct Festival use
continues to use stock `us_generate_wave`. The builder does not repair rendered
PCM. Contextual and manual unit choices are never replaced to improve a
diagnostic score. Fault Mode > Legacy joins bypasses the helper and invokes the
stock renderer with paired pre-fix bridge/database geometry where available.
Normal and Legacy pitchmark tracks currently use the same negative-going
zero-crossing epochs. Use [JOIN_SYNTHESIS.md](JOIN_SYNTHESIS.md) for the runtime
and editor contract, and use
[JOIN_DISCONTINUITY_DIAGNOSTICS.md](JOIN_DISCONTINUITY_DIAGNOSTICS.md) to
inspect rendered handoffs without modifying them.

## Source-window modes

Long OTO regions should not be compressed wholesale into ordinary short
phones. After contextual or manual recording selection, the shared
`source_window.py` policy chooses geometry for that same recording:

- `adaptive` (default): each normal source half is capped at
  `--source-window-ms` (60 ms by default). Stable hidden index variants retain
  a full left side, right side, or both. A full side is selected only when the
  corresponding target phone duration can accommodate that source half.
- `bounded`: always use the capped primary window and emit no full-window
  variants. This is useful for an A/B test of strict source bounding.
- `full`: restore the legacy whole-OTO geometry exactly.

The adaptive activation threshold is twice the complete source-half duration,
because one diphone side normally maps to approximately half of its target
phone. Selection is deterministic and recording-first: the automatic context
rule or manual per-occurrence take remains authoritative, then only that
recording's source-window variant is chosen. No gain normalization, contextual
re-ranking, waveform fade, or post-render repair is performed.

Festival indexes are static, so adaptive expansion switches from the bounded
half to its complete half at the threshold; it does not continuously reveal
arbitrary intermediate source lengths. Rebuild a voice to change the mode or
cap. Generated metadata retains both primary and complete OTO geometry for
inspection.

## Output identity

The generated `dic/diphone_index.json` and `dic/voice_manifest.json` record:

- a source-recording bundle ID derived from the recording inventory;
- a separate voice-configuration ID;
- the primary and supported language;
- the alias and canonical-phone namespaces;
- the one language entry point;
- selected OTO/profile policy;
- shared source-speaker pitch statistics, relative FRQ/WAV files used, and
  diagnostics;
- the effective average pitch and pitchmark range;
- deterministic phone-center join-level diagnostics;
- native UniSyn join-crossfade mode and bounded window settings;
- builder and schema versions.

Two configurations over identical recordings share a source bundle ID but not
a configuration ID. OTO paths in generated metadata are source-root-relative;
generated Scheme resolves its voice root from Festival's load path. Identical
build inputs therefore remain destination-independent.

## Runtime audio storage

New builds default to:

```text
--runtime-audio-storage grouped
```

After WAV slices and pitchmarks are generated, the builder creates
`group/<voice-name>_diphone.group`. Festival's UniSyn database reads this
indexed group at runtime instead of reopening the individual generated WAV and
pitchmark files for every selected unit. The packing pass is a one-time build
cost. It uses RIFF source files and 16-bit short samples and is generated
atomically, so a failed rebuild does not replace a previous valid group.

The generated Scheme probes for the group and falls back to the original
`wav/` and `pm/` layout if it is missing or unusable. Those files are retained;
grouping does not change candidate selection, context scoring, phone timing,
F0, or manual per-occurrence choices. Runtime storage mode, relative group
path, size, and SHA-256 are recorded in generated metadata where applicable.

Use `--runtime-audio-storage separate` to diagnose the historical file-per-unit
path. `--skip-pm` necessarily defers grouping and emits a voice that uses the
separate layout. Existing generated voices continue to work, but must be
rebuilt or overwritten with the current builder to gain the grouped cache.

## Build test

`--test` performs a real Festival render when pitchmarks are present.

- English and Asaxi synthesize text plus an explicit phone smoke utterance.
- Japanese analyzes the test text into the canonical Japanese utterance,
  constructs explicit phone durations, F0 targets, and unit overrides, and
  renders that exact plan through the generated Japanese entry point.

Defaults are language-specific: Japanese uses `たき`, English uses
`this is a test`, and Asaxi uses `taki`. `--test-text` overrides the default.

The test uses fresh temporary filenames so stale WAVs cannot produce a false
pass. Successful output is retained as `test_text.wav`, `test_text.seg`, and,
for the ARPAsing route, `test_phones.wav` inside the generated folder.

## Output protection

The builder rejects:

- source/output nesting in either direction;
- an output folder containing `oto.ini`;
- a nonempty output folder unless `--overwrite` is explicit;
- an OTO outside the selected source root;
- a language/bank-type mismatch.

`--overwrite` updates known generated files and does not recursively erase the
folder. Unrelated files are preserved. It never grants permission to write to
the source bank.

## GUI registration and migration

Choose one Windows generated-voice root under **Options > WSL / Festival
settings**. **Voicebank > Add Festival voice folder...** accepts generated
voices beneath that root. The saved registration includes the canonical
Windows path and a derived WSL runtime path.

Older `/mnt/<drive>/...` registrations migrate to their Windows path. Older
`/home/...` registrations remain visible as legacy WSL-only entries; the GUI
does not move, delete, or reinterpret those folders. Rebuild them into the
configured Windows root when convenient.

The older `--db` and `--utau` command forms remain available only for existing
workflows. They can generate the historical combined Asaxi/English Scheme and
should not be used for new language-scoped voices.

Legacy `--db` input and `--out` must be separate directories. Join
conditioning writes only generated WAV copies; the builder rejects an
in-place database so repeated processing cannot compound and an input database
cannot be modified accidentally.
