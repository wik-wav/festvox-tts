# Japanese Frontend Phase 1 Implementation

Status: implemented and locally verified on 2026-07-14.

This note records the implemented text-analysis boundary described in
`JAPANESE_UTAU_INTEGRATION_DESIGN.md`. Phase 1 creates a canonical Japanese
utterance from text. It does not build a voice, select UTAU recordings, change
Festival Scheme, generate durations/F0, or synthesize a waveform.

## Files added

- `japanese_models.py`: typed canonical linguistic model and provisional JSON
  serialization.
- `japanese_kana_frontend.py`: dependency-free kana/romaji analysis.
- `japanese_openjtalk.py`: optional pyopenjtalk adapter and named full-context
  label parser.
- `japanese_frontend.py`: `auto`, `openjtalk`, and `kana` dispatcher.
- `test_japanese_frontend.py`: deterministic fixtures and optional local
  integration coverage.

No English converter, builder, generated voice, GUI, Festival, duration, F0,
PSOLA, or waveform file was changed for Phase 1.

## Public API

```python
from japanese_frontend import analyze_japanese, resolve_japanese_frontend

utterance = analyze_japanese("かな？", mode="kana")
frontend = resolve_japanese_frontend("auto")
```

The frontend protocol is:

```python
class JapaneseFrontend(Protocol):
    def analyze(self, text: str) -> JapaneseUtterance:
        ...
```

Dispatcher behavior is deterministic and visible in `frontend_name` and
`provenance`:

- `kana` always selects the dependency-free frontend.
- `openjtalk` requires pyopenjtalk and raises an actionable
  `OpenJTalkUnavailableError` when it cannot be loaded.
- `auto` prefers pyopenjtalk. If unavailable or analysis fails, it returns the
  kana result with an explicit fallback diagnostic.

Importing these modules does not import `utau2festvox`, `festvox_core`, or
`synth_diphone`. pyopenjtalk is imported only when its adapter is used.

## Canonical model

`japanese_models.py` defines:

```text
JapaneseUtterance
  source_text
  normalized_reading
  phrases[]
    accent_phrases[]
      moras[]
        surface
        reading
        consonant?
        vowel?
        special_mora?
        devoiced?
        phones[]
      accent_state
      accent_nucleus?
      interrogative
      boundary_strength
  phones[]
  diagnostics[]
  frontend_name / frontend_version
  confidence / provenance
```

The concrete dataclasses are `JapanesePhone`, `JapaneseMora`,
`JapaneseAccentPhrase`, `JapanesePhrase`, `JapaneseUtterance`, and
`JapaneseFrontendDiagnostic`.

Accent nuclei are zero-based mora indexes within an accent phrase. The model
distinguishes `accented`, `unaccented`, `unknown`, and `unavailable`. Open
JTalk's one-based accent type is retained in provenance before conversion;
label value zero is represented as unaccented.

Serialization is deliberately labeled
`festvox.japanese_utterance.phase1-provisional`. This is not generated-voice
metadata schema version 2 and must not be treated as frozen.

## Canonical phones

The linguistic phone namespace is separate from UTAU aliases and English
ARPAbet/ARPAsing tokens.

- Vowels: `a`, `i`, `u`, `e`, `o`.
- Open JTalk `I` and `U`: canonical `i` and `u` with `devoiced=True`.
- Special morae: `N` for moraic nasal and `cl` for geminate closure.
- Boundaries: `pau` for pause and `sil` for utterance silence.
- Palatalized consonants include `by`, `gy`, `hy`, `ky`, `my`, `ny`, `py`,
  and `ry`.
- Affricates include `ch`, `j`, and `ts`.
- Long vowels are a separate mora containing the lengthened vowel. No
  consonant is invented.
- Unknown Open JTalk labels retain their exact symbol, raw label, and an
  `unknown_openjtalk_phone` diagnostic.

Source-bank aliases are not mapped or selected in this phase.

## Kana/romaji fallback

The fallback accepts normalized hiragana, katakana (including NFKC-normalized
half-width forms), and the supported Hepburn-style romaji table. It recognizes
gojuon morae, common foreign-sound combinations, palatalized morae, affricates,
`ん`, `っ`, and `ー`. Punctuation and `[pau]` create phrase boundaries and
canonical pause phones. Question marks set interrogative state.

Limitations are explicit:

- Kanji reading and morphology are unavailable. Each unsupported character is
  retained as an unknown mora and reported; it is never silently spelled or
  deleted.
- Lexical pitch accent is unavailable. Accent state is `unavailable`, with a
  deterministic neutral default and a diagnostic.
- The fallback does not predict vowel devoicing, words/bunsetsu, dialect, or a
  final pitch contour.
- Romaji is phonetic input, not an orthographic Japanese converter. For
  example, `konnichiwa` has normalized reading `こんにちわ`.

## Open JTalk adapter

`OpenJTalkJapaneseFrontend` checks for pyopenjtalk without making it a required
application import. When available it calls only:

- `pyopenjtalk.g2p(text, kana=True)` for the normalized reading;
- `pyopenjtalk.extract_fullcontext(text)` for linguistic labels.

It does not call `tts`, request an HTS waveform, or make Open JTalk duration/F0
authoritative. The installed package version is recorded when available.

pyopenjtalk was not installed in the verification environment, so its optional
integration test skips cleanly. Parser behavior is covered by static labels and
does not download a dictionary or model.

## Full-context fields parsed

The parser splits the quinphone and each context group in named stages rather
than one positional regular expression. `ParsedOpenJTalkLabel` retains every
raw group, including unrecognized future groups.

- Quinphone: two preceding phones, current phone, and two following phones
  from `p2^p1-current+n1=n2`.
- `A`: mora position relative to the accent nucleus, forward mora position,
  and backward mora position.
- `F`: accent-phrase mora count, one-based accent type/nucleus, interrogative
  flag, emotion placeholder, accent-phrase position in the breath group, and
  mora span.
- `I`: breath-group accent-phrase/mora counts, utterance position fields, and
  accent/mora span fields.
- `K`: utterance breath-group, accent-phrase, and mora totals.

Speech phones sharing the same `A` mora position are grouped into one mora.
`F` position changes create accent phrases. `pau` creates a phrase boundary;
initial/final `sil` records utterance boundaries. Reading morae are aligned to
the label groups when counts agree. A mismatch keeps label phones authoritative
and produces a diagnostic.

## Assumptions and unresolved semantics

- The parser follows the standard JPCommon A/F/I/K shape represented by the
  static fixtures and the Open JTalk sources cited in the design document.
- `F` accent type uses zero for unaccented and positive one-based mora indexes.
- `B`-`E`, `G`, `H`, and `J` are retained raw but are not promoted in Phase 1.
- Some detailed F/I span names vary in descriptions of Open JTalk label
  versions. Those values are retained and exposed, but do not drive duration,
  F0, unit selection, or synthesis.
- Source punctuation-to-label phrase alignment is best effort. Label `pau` and
  `sil` remain authoritative when counts differ.
- A locally installed pyopenjtalk version should be sampled before any schema
  is frozen; raw-label retention makes that comparison lossless.

## Tests

`test_japanese_frontend.py` contains 29 tests covering:

- model serialization;
- hiragana, katakana, and romaji;
- long vowels, moraic nasal, geminate, and palatalized morae;
- punctuation, explicit pauses, and questions;
- unsupported kanji and neutral fallback accent;
- required Open JTalk and automatic fallback behavior;
- named full-context parsing and raw provenance;
- multiple accent phrases, accented/unaccented state, and interrogatives;
- unknown fields/phones, malformed labels, and devoiced vowels;
- Open JTalk-shaped pause/silence boundaries;
- isolation from the English pipeline;
- optional local pyopenjtalk integration.

The current result is 29 tests run, 28 passed, and one optional integration
test skipped because pyopenjtalk is not installed.

## Phase 2 boundary

Phase 2 may consume this utterance model while implementing bank profiles,
alias overrides, coverage graphs, and CV/VCV/CVVC candidate compilation with
versioned generated metadata. It must not reinterpret English phones or make
the provisional Phase 1 JSON a frozen voice schema.

Festival phonesets, UniSyn hooks, durations, F0 generation, GUI accent tools,
waveform conversion, and multipitch selection remain later phases.
