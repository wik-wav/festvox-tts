# English syllabification

## Purpose

`english_syllables.py` provides a deterministic, dependency-free parser for
the English ARPAbet/ARPAsing phone stream already produced by a synthesis
frontend. It is diagnostic metadata for future linguistic features. It does
not perform G2P, choose recordings, change phone timing, or alter audio.

Every English `festvox_core.Synthesis` receives an
`english_syllabification` dictionary. The Speech tab visualizes that metadata
behind every continuous **Pitch curve**, **Voicing**, and **Vocal tract
length** timeline when **Show syllables / morae** is enabled. The shared view
uses alternating bands, dotted boundaries, and zoom-dependent labels. English
displays syllables; Japanese and Asaxi use their own existing mora plans and
never receive English syllabification.

## API and model

The public entry point is:

```python
from english_syllables import syllabify_english

result = syllabify_english(
    ["r", "ae1", "b", "ih0", "t"],
    word_boundaries=None,
)
```

The typed result is `EnglishSyllabification`. It retains:

- original and normalized phones;
- ordered `EnglishSyllable` records;
- zero-based, half-open phone spans;
- onset, nucleus, and coda phones;
- CMU stress `0`, `1`, or `2` when present;
- inferred, word, pause, and utterance boundary provenance;
- pause indexes;
- confidence and explicit diagnostics; and
- deterministic versioned dictionary serialization.

Unknown phones and nucleus-free spans are retained with diagnostics. Pause
phones are retained in the utterance phone list but never placed inside a
syllable. Numbered recording suffixes such as `ae1__u12` are ignored for
linguistic matching without losing the original phone spelling.

## Boundary algorithm

The parser uses the maximal-onset principle between adjacent nuclei, limited
by an explicit English onset inventory. For example:

```text
r ae1 b ih0 t      -> r ae1 | b ih0 t
eh1 k s t r ah0   -> eh1 k | s t r ah0
ae1 t l ah0 s     -> ae1 t | l ah0 s
```

Supplying `word_boundaries` prevents a consonant at the end of one lexical
word from being reassigned to the next word. A boundary value is a position
between phones: `2` means that a new word begins at phone index 2.

The current synthesis metadata usually has only a phrase-level phone stream,
so its default result is phrase-level syllabification. A future English
frontend can pass lexical phone spans without changing this model or its
serialization. The parser deliberately does not infer orthographic word
alignment from text because doing so without frontend provenance would make
contractions, heteronyms, and pronunciation overrides unreliable.

## GUI and persistence

The continuous-curve overlay is a debugging surface, not an edit mode:

- **Show syllables / morae** is off by default and persists as a local view
  preference;
- the toggle appears in the common Parameter Editor header for Pitch curve,
  Voicing, and Vocal tract length;
- English shows parser syllables, while Japanese and Asaxi show aligned morae;
- bands and labels do not constrain the continuous voicing curve;
- labels are omitted when a syllable is too narrow to read;
- at most 24 visible labels are instantiated at once;
- edited phone sequences trigger immediate metadata recomputation;
- old projects without this field recompute it when displayed; and
- current projects persist the metadata in `project.json`.

The visualization uses three persistent plot primitives plus a bounded set of
visible text labels, so long utterances do not create one graphics object per
syllable.

## Resource lifecycle changes made with this feature

Sentence, phrase-preview, clipboard, duplicate, and undo snapshots share
immutable rendered NumPy audio buffers while deep-copying editable metadata.
Waveform widgets also reuse finite float32 buffers. New audio replaces the
old array rather than mutating it in place.

Closing the main window now shuts down playback and the warm Festival/WSL
runtime exactly once. If synthesis is active, closing requests cancellation
of the remaining batch and completes after the in-flight backend call exits.

## Tests

From `99_Tools/festvox`:

```powershell
py -3.14 -m unittest test_english_syllables -v
py -3.14 -m unittest festvox_gui.test_festvox_core -v
$env:QT_QPA_PLATFORM = "offscreen"
py -3.14 -m unittest festvox_gui.test_festvox_gui -v
```

The parser tests cover ordinary syllables, legal and illegal onset clusters,
stress, pauses, syllabic consonants, explicit word boundaries, numbered
alternatives, unknown phones, nucleus-free spans, and deterministic
serialization. Core and GUI tests cover automatic metadata attachment,
language isolation, project persistence, overlay geometry, shared audio
storage, and idempotent shutdown.
