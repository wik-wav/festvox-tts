# Prompt 19 Implementation

Prompt 19 extends the existing Japanese Festival/UniSyn path; it does not add a
parallel synthesizer. The normal Speech-tab Generate and Re-render operations
now invoke context-sensitive duration prediction and, when eligible, a
source-filter vowel-realization stage. English and Asaxi routing remains
separate and retains its existing timing behavior.

## Production path

1. `japanese_frontend.analyze_japanese()` obtains canonical phones, morae,
   accent phrases, punctuation, and raw Open JTalk full-context labels.
2. `japanese_duration.build_duration_contexts()` derives named phone, class,
   mora, accent, boundary, neighboring-phone, and devoicing features.
3. `japanese_synthesis.create_synthesis_plan()` selects either `legacy` or
   `contextual` duration mode, applies selected-source baselines and bounded
   residuals, and emits explicit Segment durations, F0 daughters, unit
   overrides, `timing_role`, and diagnostics.
4. `FestivalWSLBackend.synth_phones()` renders that exact plan through the
   generated Japanese voice's direct-waveform UniSyn TD-PSOLA entry point.
5. `japanese_devoicing.apply_vowel_realizations()` measures the returned
   waveform and applies eligible automatic realization decisions or a final
   user-authored Voicing curve.
6. The GUI stores source/automatic/manual voicing curves, timing rows, and
   decisions in project state. Manual unit choices remain final and are never
   changed to solve duration or voicing problems.

## Runtime options

The persisted GUI keys and menus are:

| Key | Values | Menu |
| --- | --- | --- |
| `japanese_duration_model` | `contextual`, `legacy` | Options > Japanese duration model |
| `japanese_vowel_devoicing` | `contextual`, `legacy` | Options > Japanese vowel devoicing |
| `japanese_devoicing_renderer` | `auto`, `source_filter`, `shortened_voiced` | Japanese vowel devoicing > Renderer |

`auto` follows this conservative order: already-aperiodic source, source-filter
residual modification, then accurately labelled shortened-voiced fallback.
Old `mixed_excitation` project values migrate to `source_filter`.

## Contextual duration

The versioned model is
`profiles/japanese_duration_priors_v1.json`, model
`japanese_contextual_source_anchor_kokoro_b453f6caf042_v7`. Prompt 20
superseded the original Prompt 19 v4 coefficients with Kokoro-relative mora
allocation, validated grammatical residuals, punctuation-pause refinement, and
source-relative phrase-edge calibration. It predicts phone durations in log
space from the selected speaker's source geometry, bounded class-normalized
geometry, Open JTalk/canonical context residuals, and class-specific speed
elasticity. It includes partial CV compensation, long vowels, geminates,
moraic nasals, likely high-vowel devoicing, consecutive-devoicing avoidance,
accent/phrase edges, and utterance-final effects.

The new explicit `moraic_nasal` timing role fixes bank aliases such as `nn`,
`nng`, `mm`, and `xn`: they are bounded by Japanese `/N/` priors and are edited
with rhyme timing, not consonant timing. Non-Japanese `nn` is unchanged.

`japanese_duration_corpus.py` supplies deterministic JSUT fitting/evaluation
and an optional CSJ TextGrid adapter. It fits dimensionless residuals, excludes
fixed and deterministic held-out utterances, records relative file provenance,
and never imports into runtime synthesis.

## Continuous voicing

The Speech-tab parameter menu exposes **Voicing** for any rendered language.
The dashed curve is measured from the current source render and remains stable
after regeneration. The editable curve uses bounded control spacing (about
32 ms analysis frames with 8 ms hops), so it is substantially more granular
than one point per phone without creating one point per audio sample.

`source_filter_voicing.py` implements deterministic reconstructive short-time
source/filter analysis:

```text
speech = (harmonic excitation + stochastic excitation) * tract envelope
```

It divides out a smooth envelope, separates pitch-scaled harmonic and
aperiodic residual components, modifies their mixture, and resynthesizes both
through the same envelope with complementary overlap windows. Strongly voiced
source frames do not contain enough trustworthy noise for a zero endpoint, so
one continuous deterministic shaped-noise source is used instead of repeated
frame-local random noise. Low-frequency noise is constrained and the zero
endpoint follows the supplied real-devoiced level reference rather than being
normalized back to voiced RMS.

Manual Voicing points are final. The automatic dashed analysis remains visible
for comparison and is never silently replaced by a regenerated manual curve.

## Can FestVox PSOLA perform Japanese vowel devoicing?

1. These generated voices use waveform and EST pitchmark files with
   `Synth_Method 'UniSyn`, `us_sigpr 'psola`, and direct waveform TD-PSOLA.
   They do not contain an LPC coefficient/residual database.
2. UniSyn can duration-map voiced and already-unvoiced source regions using
   source and target epochs.
3. Removing target F0 was not accepted by the installed path. A deliberately
   very low 40 Hz target still produced a strongly periodic pulse train.
4. Standard TD-PSOLA therefore does not convert a periodically voiced source
   vowel into credible Japanese devoicing. It only rearranges source frames.
5. An LPC-residual Festival branch could support excitation replacement, but
   this voice does not use that representation. Rebuilding the entire voice
   around it would be invasive.
6. Naturally aperiodic selected source intervals are retained when detected.
   The bank does not yet expose a complete explicit voiced/devoiced allophone
   inventory.
7. The implemented minimum extension is post-UniSyn source-filter residual
   modification with continuous stochastic excitation and safe
   shortened-voiced fallback. Duration and voicing remain separate decisions.
8. Objective results are in `PROMPT19_BENCHMARK_REPORT.md`. The supplied real
   reference endpoint is matched closely in level and spectral centroid;
   harmonic ridges are strongly reduced while the tract envelope is retained.
9. Human listening of the final Prompt 19 clips has not yet been recorded, so
   acoustic naturalness is not claimed.
10. Remaining failures include incomplete source-allophone metadata, short
    frames that cannot pass all quality gates, and residual periodicity above
    the supplied natural devoiced reference.

This follows the source/filter separation principle described by
[Degottex (2010)](https://gillesdegottex.eu/pdf/Degottex2010_PhD_v4_Final.pdf)
without claiming a full glottal-flow inverse-filter implementation.

## Other Prompt 19 production changes

- Japanese and Asaxi render calibration now uses language-aware reference
  level handling instead of leaving them systematically quieter than English.
- Speaker pitch metadata keeps the measured median, while the automatic
  contour floor prevents routine downward pitching below that median; users
  may still set a lower value explicitly.
- Generate All and Re-render All show sentence progress in the status bar and
  provide a stop-after-current-sentence button.
- The join analyzer now adds voiced-only F1-F4, bandwidth, prominence,
  formant-balance, spectral-envelope, trajectory, and pop evidence while
  preserving all original component metrics and read-only behavior.
- Open JTalk availability checks now distinguish an importable package from an
  operational dictionary installation.

## Validation and safety

The fixed manifest is `profiles/japanese_duration_validation_v1.json`. The A/B
runner creates ignored WAV, JSON, Markdown, and spectrogram artifacts and
refuses output inside a generated voice or UTAU source bank. Builders and
runtime analysis continue to read source UTAU banks only; no source recording,
OTO row, FRQ file, prefix map, or character file is modified.

See `docs/japanese_duration_model.md` for model assumptions, corpus commands,
licensing, and limitations, and `PROMPT19_BENCHMARK_REPORT.md` for measured
results and reproduction commands.
