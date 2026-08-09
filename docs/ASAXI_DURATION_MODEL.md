# Asaxi Duration Model

## Status

`asaxi-moraic-rules-v1` is the provisional, recording-independent duration
model used by Asaxi Text generation through Festival/WSL. It replaces the
former accidental Festival `Duration_Default` result, in which every phone was
100 ms long.

This is an explicit engineering prior, not an attested or trained model.
Future Asaxi recordings should replace its constants through a versioned model
without changing the canonical mora or editor data.

## Research Basis

The model follows the Asaxi grammar's documented mora timing while avoiding
strict mechanical isochrony:

- Port, Dalby, and O'Dell found an approximately constant timing increment per
  Japanese mora while retaining intrinsic segment differences:
  <https://pubmed.ncbi.nlm.nih.gov/3584695/>.
- Kawahara found partial consonant-vowel duration compensation in more than
  200,000 spontaneous Japanese CV morae:
  <https://pubmed.ncbi.nlm.nih.gov/28764476/>.
- Beckman's measurements did not support a strict prediction that every mora
  must have exactly the same duration:
  <https://www.degruyterbrill.com/document/doi/10.1159/000261655/html>.
- Kaiki, Takeda, and Sagisaka identify phone class, neighboring phones,
  phrase position, and gemination among practical duration factors:
  <https://www.isca-archive.org/icslp_1990/kaiki90_icslp.html>.
- Articulatory measurements of Japanese long consonants support treating a
  geminate hold as a substantial interval rather than duplicating a short
  consonant:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC2827771/>.

Japanese is evidence for a timing architecture, not a claim that Asaxi has
Japanese coefficients. The Asaxi-specific phonological decisions come from
the vault's phoneme, phonotactic, and prosody chapters.

## Pipeline

1. `asaxi_frontend.py` identifies graphemes, phones, and morae.
2. `asaxi_prosody.py` resolves lexical/morphological forms and creates the
   utterance's canonical phone and mora plan.
3. Festival `SynthText` supplies the exact phone sequence supported by the
   selected generated voice.
4. `asaxi_duration.py` aligns that sequence to the canonical morae and replaces
   only the spoken phone durations.
5. Asaxi F0 targets are generated on the retimed segment timeline.
6. The normal explicit `Segments`/UniSyn pass renders those durations and F0
   targets through the same shared synthesis features as other languages.

Pause durations remain owned by punctuation and phrase assembly. Re-render
uses the editor's current segment durations and does not invoke the duration
planner again.

## Rules

The default beat is 120 ms at speed `x1.00`. It is divided using broad,
auditable phone-class priors:

| Phone role or class | Relative behavior |
| --- | --- |
| Voiceless stop | Short closure/release; longer than a voiced stop |
| Voiced stop | Shorter than its voiceless counterpart |
| Affricate | Longer onset than a stop |
| Fricative | Longer onset; partially compensated by a shorter nucleus |
| Nasal/liquid | Intermediate sonorant timing |
| Glide/tap | Short transition |
| Vowel nucleus | Receives the remaining mora body; intrinsic vowel and diphthong differences remain |
| Coda | Bounded share of the same mora, not another full beat |
| Syllabic nasal | One nucleus-like mora |
| Geminate stop | One structural hold mora |
| Geminate continuant | Extends the following onset by one hold mora without duplicating its phone |
| Glottal coda | Short closure inside its mora; no extra mora |

Complex morae are only partially compressed toward the 120 ms target. A CVC
mora can therefore be slightly longer than a simple CV mora. A doubled vowel
is two morae; an Asaxi diphthong is one. Phrase-final lengthening is bounded
and concentrated on the final nucleus and coda, with weaker lengthening at a
minor boundary.

Speed scales modeled spoken-phone durations. It does not rewrite the semantic
pause settings.

## Diagnostics

`Synthesis.asaxi_prosody` records:

- `duration_model: "moraic_rules"`;
- `duration_model_id: "asaxi-moraic-rules-v1"`;
- phrase-local `duration_plans`, including every phone's class, mora role,
  duration, modifiers, and absorbed continuant-geminate morae;
- final rendered mora and segment alignment;
- alignment diagnostics.

The **Disable phone timing** fault remains authoritative and records
`duration_fault_override: "equal_phone_timing"`.

## Tests

Run the focused model and backend tests:

```powershell
py -3.14 -m unittest -v test_asaxi_duration
py -3.14 -m unittest -v `
  test_festvox_core.PhrasePauseTests.test_asaxi_seed_injects_inferred_dotted_geminate_pronunciation
```

The tests cover CV allocation, fricative-vowel compensation, stop voicing,
closed morae, syllabic nasals, stop and continuant geminates, diphthongs,
doubled vowels, glottal codas, phrase-final lengthening, speed scaling,
determinism, and live-backend handoff.

## Limitations

- The constants are typological priors and have not been fitted to an Asaxi
  speaker.
- The model does not yet use lexical frequency, speaking style, syntax, or
  measured allophonic timing.
- Naturalness still requires human listening.
- The planned recording corpus should provide phone/mora boundary labels,
  phrase position, speech rate, and repeated tokens. A later model should
  retain this ruleset as a deterministic fallback and a fault comparison.

No duration analysis reads from or writes to a source UTAU bank.
