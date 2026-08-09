# Japanese Contextual Duration Model

This document describes the deterministic Japanese phone-duration model used
by the normal Festival/UniSyn synthesis path. It separates established
phonetic findings from conservative engineering choices. Runtime synthesis
does not depend on a corpus, forced aligner, or machine-learning package.

## Phonetic basis

### Mora timing is statistical

Japanese word duration often grows with mora count, but a mora is not a fixed
acoustic interval. Port, Dalby, and O'Dell found evidence for mora-level timing
while also measuring segmental variation; Warner and Arai found that moraic
organization remains relevant in spontaneous speech without literal segment
isochrony. The implementation therefore predicts phone durations inside a
mora instead of assigning every mora one equal duration.

Sources: [Port, Dalby, and O'Dell (1987), DOI 10.1121/1.394510](https://doi.org/10.1121/1.394510)
and [Warner and Arai (2001), DOI 10.1121/1.1344156](https://doi.org/10.1121/1.1344156).

### Partial CV compensation

Longer onset consonants tend to be followed by somewhat shorter vowels, but
the compensation is incomplete. The model applies a bounded fraction of the
onset deviation to the following vowel. It never forces `C + V` to equal one
fixed mora duration.

Source: [Kawahara (2017), DOI 10.1121/1.4994674](https://doi.org/10.1121/1.4994674).

### Boundary lengthening

Spontaneous Japanese lengthens material near prosodic boundaries. The effect
is not distributed uniformly across all phones, so this model applies small,
bounded adjustments to rhyme-like material at accent-phrase, phrase, and
utterance edges rather than stretching stop releases.

Source: [Hori and Mori (2016), DOI 10.24467/onseikenkyu.20.2_38](https://doi.org/10.24467/onseikenkyu.20.2_38).

### Vowel devoicing

High-vowel devoicing is probabilistic, not a guaranteed rewrite of every `/i/`
or `/u/` between voiceless consonants. Maekawa and Kikuchi report substantial
contextual variation in spontaneous speech. The duration model marks likely
environments and shortens eligible intervals, while a separate realization
stage decides whether the rendered excitation should remain voiced, use a
naturally aperiodic source, or use source-filter residual modification. Long
vowels are excluded from automatic short-high-vowel devoicing.

Source: [Maekawa and Kikuchi (2005), DOI 10.1515/9783110197686.2.205](https://doi.org/10.1515/9783110197686.2.205).

### Special morae

The following are independent classes rather than ordinary CV morae:

- moraic nasal `/N/`;
- geminate closure `cl`;
- long-vowel continuation;
- contextually devoiced high vowel;
- pause and silence.

The moraic nasal is also distinct from an onset nasal. Sato found a clear
durational distinction and a following-consonant effect: moraic nasals are
longer before voiced than before voiceless consonants. The model has a
dedicated `/N/` class and a small pre-voiceless shortening term.

Source: [Sato (1993), DOI 10.1159/000261925](https://doi.org/10.1159/000261925).

### Accent

Pitch accent remains primarily an F0 structure. Accent-nucleus and
accent-phrase-edge duration terms are deliberately small. The implementation
does not impose a large duration multiplier merely because a mora carries a
nucleus.

## Runtime formulation

For each phone, `japanese_duration.predict_phone_duration()` evaluates:

```text
log(target duration)
    = log(project-speaker mora allocation)
    + bounded source-geometry residual
    + bounded contextual residual
    + phone-class speaking-rate adjustment

rendered edge duration
    = target duration - bounded acoustic edge compensation
```

Project-speaker mora anchors retain the absolute timing scale. Kokoro supplies
relative phone allocation and bounded residual evidence; its speaker's speaking
rate is never copied into the voice. Selected-unit geometry contributes only a
narrow within-bank ratio. The shipped model is
`japanese_contextual_source_anchor_kokoro_b453f6caf042_v7`, schema 1, in
`profiles/japanese_duration_priors_v1.json`.

### Source baseline

The baseline hierarchy is:

1. selected source-unit contribution geometry;
2. generated voice timing profile for the same phone/class;
3. conservative class reference when source evidence is absent.

OTO offset, preutterance, overlap, consonant, and cutoff remain alignment
landmarks. They are never interpreted as literal spoken target durations.
Within-bank geometry contributes a bounded ratio so an unusually long source
slice cannot dominate the linguistic target.

### Context features

`JapaneseDurationContext` retains:

- phone identity, class, and position within its mora;
- preceding and following phone;
- mora position in accent phrase, phrase, and utterance;
- special-mora type and long-vowel status;
- accent state, zero-based nucleus distance, and phrase-final flags;
- boundary strength and interrogative state;
- Open JTalk devoicing evidence and neighboring likely-devoicing state;
- named raw Open JTalk A, F, I, and K context fields and parsed positions;
- NJD surface, part of speech, conjugation type/form, lexical-node position,
  coarse grammatical role, and function-word status when Open JTalk exposes
  them.

Missing or malformed Open JTalk fields remain unavailable rather than being
invented. The dependency-free kana frontend still supplies canonical mora
structure and uses neutral accent diagnostics.

### Effects and bounds

The versioned priors contain separate coefficients for devoiced high vowels,
geminate closure, long-vowel continuation, moraic nasal, accent/phrase edges,
utterance-final rhyme, interrogative final rhyme, and consecutive-devoicing
avoidance. Speaking-rate elasticity differs by class: vowels and moraic nasals
absorb more rate change than stops and closures.

Open JTalk morphology is grouped into inspectable roles including particle,
auxiliary, negative auxiliary, polite auxiliary, polite copula, function word,
and content word. Robust training/held-out Kokoro residuals supported small
log-duration terms for ordinary auxiliaries (`-0.045`) and negative auxiliaries
(`-0.075`). Particle timing made absolute error worse, and polite categories
changed direction out of sample, so those categories remain in diagnostics but
receive no shipped timing coefficient.

All contextual effects are additive in log space and constrained by the global
context ratio. Source geometry has class-specific limits. The moraic nasal has
an additional target/reference ratio of `0.65..1.35` and source-geometry ratio
of `0.92..1.08`; this prevents long CV-style `nn` recordings from becoming
unrealistic held nasals.

Generated Japanese segments carry a `timing_role`. Canonical `/N/` and mapped
bank aliases such as `nn`, `nng`, `mm`, and `xn` carry
`timing_role="moraic_nasal"`. In the GUI they respond to rhyme/vowel timing
edits, not consonant-only edits. A plain `nn` in a non-Japanese route retains
ordinary consonant behavior.

## Corpus fitting and evaluation

`japanese_duration_corpus.py` supports:

```powershell
python japanese_duration_corpus.py fit `
  --jsut <extracted-jsut-root> `
  --csj <optional-licensed-csj-root> `
  --output <priors.json> `
  --report-json <fit-report.json> `
  --report-markdown <fit-report.md>

python japanese_duration_corpus.py evaluate `
  --jsut <extracted-jsut-root> `
  --csj <optional-licensed-csj-root> `
  --priors profiles/japanese_duration_priors_v1.json `
  --output-json <evaluation.json> `
  --output-markdown <evaluation.md>
```

HTK timestamps are converted with `(end - start) / 10000` milliseconds. JSUT
forced-alignment timings are treated as a silver reference. The loader raises
actionable errors for missing roots, missing labels, invalid timestamps, and
alignment failures. Silence is separated from ordinary-phone statistics.
Splits occur by utterance; the six `BASIC5000` fixed-validation IDs in
`profiles/japanese_duration_validation_v1.json` are excluded from fitting, and
the remaining held-out selection is deterministic.

The fitter removes corpus phone/class scale, normalizes speaking rate, applies
robust trimming, and estimates dimensionless context residuals. It records the
exact relative corpus file list, sample counts, dispersion, exclusions,
feature schema, and provenance. A fitted file does not replace project-speaker
absolute timing.

The Prompt 20 fit also uses the read-only Kokoro Speech Dataset as silver
reference evidence. Phrase punctuation boundaries are refined to sustained
low-energy intervals before pause statistics are calculated. The fitted pause
settings are 340 ms for minor punctuation, 530 ms for major punctuation, and
800 ms for sentence boundaries. Synthesis represents each setting as two
protected 80 ms edge guards plus an editable middle interval 80 ms shorter, so
the rendered totals are 420, 610, and 880 ms respectively.

NJD nodes are aligned to canonical morae so grammatical categories can be
audited independently. On held-out phones, the retained ordinary-auxiliary term
reduced MAE from 16.92 to 14.81 ms and the negative-auxiliary term from 15.11
to 7.64 ms. Rejected categories are recorded in fit provenance rather than
silently forced into the model.

`japanese_phrase_edges.py` measures sustained-energy onset and offset around
logical phrase boundaries in the real source and synthesized waveforms. The
training set showed a 51.791 ms median source-relative excess for vowel-initial
phrases, versus 7.921 ms for consonant-initial phrases. The model therefore
subtracts up to 50 ms from a phrase-initial vowel-only mora, bounded to at most
55 percent of the phone and a 30 ms minimum. A final-vowel correction was
explicitly rejected: held-out effective final-mora timing already matched the
source, and the proposed correction made endings too short.

The final deterministic benchmark reports the following contextual/legacy
results (milliseconds unless noted):

| split | duration MAE | duration p90 | pitch MAE (st) | pitch correlation |
| --- | ---: | ---: | ---: | ---: |
| training | 31.126 / 33.271 | 61.322 / 66.177 | 1.850 / 1.905 | 0.553 / 0.482 |
| held out | 32.209 / 34.149 | 58.491 / 67.396 | 1.993 / 2.105 | 0.435 / 0.359 |

Held-out median cumulative boundary drift is 237.834 ms for contextual timing
versus 220.596 ms for legacy timing, so that aggregate metric remains an open
weakness even though per-phone and rate-normalized errors improve.

[JSUT](https://arxiv.org/abs/1711.00354) is an open read-speech corpus but its
audio and labels are not vendored here. [CSJ](https://clrd.ninjal.ac.jp/csj/en/)
is optional and subject to NINJAL access and licensing terms; commercial use
requires separate consideration. Runtime and tests work without either corpus.

## Configuration and fallback

The GUI exposes **Options > Japanese duration model**:

- `contextual`: source-relative contextual prediction (default);
- `legacy`: the prior mora allocator and its established timing.

Invalid or unavailable priors produce an explicit diagnostic and fall back to
legacy timing. Re-render keeps manually edited phone boundaries; a fresh
Generate applies the selected duration provider. Speed changes timing without
changing F0.

## Limitations

- Kokoro alignments are silver references and use a different speaker and
  recording style; only bounded relative evidence is transferred.
- JSUT automatic boundaries contain alignment uncertainty and differ in
  speaker/style from a selected UTAU bank.
- CSJ gives stronger spontaneous-speech evidence but is optional and licensed.
- Open JTalk context is not ground truth for devoicing.
- Grammatical labels are Open JTalk analyses, not manually annotated truth.
- Particle and polite-copula timing remain unresolved because train and held-out
  evidence did not justify a stable coefficient.
- Lexical and punctuation F0 are modeled separately from phone duration.
- Better duration metrics do not prove perceptual naturalness. Generated A/B
  clips still require human listening.
