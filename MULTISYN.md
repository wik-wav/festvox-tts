# Asaxi Multisyn upgrade — engineering assessment, data budget & protocol

Moving the Asaxi voice from **diphone** synthesis (one unit per type, rigid
join) to **Multisyn unit selection** (many units per type, natural runs
chosen at synthesis time). Your existing UTAU reclist already covers 100 % of
Asaxi diphone bases — it stays as the **coverage floor**. This document
decides the annotation format, quantifies how much *new* continuous speech to
record, and specifies exactly how to record and label it so the new units
join cleanly with the reclist.

---

## Part 1 — Engineering assessment

### 1.1 Audacity labels vs `oto.ini` — Audacity wins, definitively

**Record and export Audacity label tracks (`.txt`).** Not `oto.ini`.

Multisyn does not concatenate at hand-authored crossfade regions. It:

1. treats the database as a stream of phones, each with **one boundary time**
   (start = previous phone's end) — an HTK/EST *segmentation*;
2. extracts acoustic features (MFCC + F0) *itself* at those boundaries;
3. computes **join cost** = spectral + F0 + energy discontinuity between two
   candidate units, and **target cost** = linguistic fit, at run time;
4. joins pitch-synchronously at the chosen point.

Line those requirements up against the two formats:

| What Multisyn consumes | Audacity `.txt` | `oto.ini` |
| --- | --- | --- |
| one **phone boundary** per segment | ✅ a flat `start end label` track *is* a segmentation — 1 label per phone, maps 1:1 to `.lab`/`.mlf` | ⚠️ one entry per *unit/alias*, five landmarks (offset, consonant, preutterance, overlap, blank) describing a **crossfade region**, not a phone cut |
| contiguous segmentation of a **whole sentence** | ✅ native — a label track is the whole utterance | ❌ oto's mental model is one file = one unit; continuous sentences need many overlapping entries fighting the format |
| the boundary at a **stable acoustic landmark** | ✅ you place it exactly where you want the cut | ⚠️ `preutterance` is an alignment point for concatenation, `overlap` a fade width — neither is the phone boundary Multisyn needs |
| features for join cost | ✅ computed from the wav at the boundary | the extra oto landmarks are **ignored** — they describe manual crossfades Multisyn replaces |

The tempting argument for oto is "it carries *more* data per unit (5 numbers
vs 2)." That is true and irrelevant: those five numbers encode **diphone
crossfade geometry**, which unit selection throws away in favour of automatic
pitch-synchronous joins and a learned acoustic join cost. More data of the
wrong *shape* loses to two numbers of the right shape. `oto.ini` is the ideal
format for the *diphone* database you already have; it is the wrong tool for
*Multisyn*.

**Consequence for the scripts:** Script 2 parses Audacity `.txt` → EST `.lab`
+ HTK `.mlf` (it also has a clearly-marked best-effort oto path, because you
offered it, but Audacity is the supported route).

### 1.2 How much *new* speech to record — the redundancy budget

**Target: ≈ 1 hour (~1,000 sentences) of continuous Asaxi. Practical minimum
for "natural": ~30 min (~450 sentences).**

Here is the reasoning, grounded in your own corpus (numbers measured by
running the Asaxi g2p over the *Velveteen Rabbit* translation):

- avg **phones/sentence ≈ 46**, avg **diphones/sentence ≈ 46**
- one text (285 sentences) already exercises **785 distinct diphones**

Unit-selection quality is a function of **candidate density** — how many
instances of each unit the selector can choose among. The floor (1 per
diphone) you already have; naturalness comes from redundancy **R** (instances
per unit in varied contexts). Empirically:

| Voice class | Speech | Instances/unit (R) |
| --- | --- | --- |
| Diphone (what you have) | 3–5 min | 1 |
| Natural limited unit-selection | 30–40 min | ~5–15 (head) |
| **General-purpose (target)** — CMU ARCTIC parity | **~60 min, ~1,000–1,150 sents** | **head 50–300, mid ~10–20, tail ≥1** |
| Commercial | 5–10 h | hundreds |

Quantify it as a token budget. Continuous Asaxi runs ≈ **12 phones/s**, so:

```
tokens(minutes)  ≈ minutes · 60 · 12          (diphone tokens ≈ phone tokens)
   30 min  →  ~21,600 tokens
   60 min  →  ~43,200 tokens
```

Spread Zipfian over the ~800 legal Asaxi diphones:

```
mean instances/diphone  ≈ tokens / 800
   30 min  →  ~27 avg   (head in the hundreds, tail 1–5)
   60 min  →  ~54 avg
```

Because the distribution is Zipfian, you never get uniform R; you target R for
the frequent units and rely on the reclist floor for the rare tail. Solving
for the sentences needed to bring the **head** constructions to a target R:

```
N_sentences ≈ (C_target · R) / (D̄ − 1)
   C_target = # high-frequency constructions to make redundant (~300–400)
   R        = target instances                (12–20)
   D̄        = mean diphones/sentence          (≈46 measured)

   (350 · 15) / 45  ≈ 117 sentences  → covers the HEAD only
```

…but head-only speech sounds uneven, so you pad to ARCTIC scale (~1,000
sentences) to also lift the mid-frequency band and add prosodic variety
(declaratives, questions, imperatives, lists). Hence:

- **Minimum viable:** ~30 min / ~450 sentences → the top ~300 grammatical
  constructions reach R≈8–12; noticeably more natural than diphone.
- **Recommended:** ~60 min / ~1,000 sentences → ARCTIC-class general voice.

**This is what Script 1 operationalizes:** it targets grammatical
constructions (not diphone minimums, which are already met) up to a
redundancy R, stopping at your chosen minute budget.

---

## Part 2 — the scripts (summary; details in each file's header)

### Script 1 — `corpus_extract.py` (grammar-aware recording script)

Because diphone coverage is done, it **skips greedy diphone minimization** and
instead mines **high-frequency grammatical constructions** — particle chains
(`to wo`, `dåni … zè-`), verb affix stacks (tense/neg/aspect + `-ů/-nů`), and
function-word-bearing word n-grams — then greedily selects the fewest
sentences that raise each to a target redundancy R, capped at your minute
budget. Outputs a numbered recording script + a coverage report.

```
python corpus_extract.py --corpus "…/Velveteen Rabbit (Reader's Text).md" \
       "…/Asaxi - 218 Syntax Test Cases.md" --minutes 60 --redundancy 12
```

Measured run (20-min slice of the current corpus): 547 candidate sentences →
354 selected, 767 distinct diphones exercised, grammatical targets tracked to
R. Point it at a larger corpus and raise `--minutes` for the full hour.

### Script 2 — `labels2festvox.py` (annotations → FestVox labels)

Reads your **Audacity `.txt`** label tracks and writes the two label formats
Multisyn compiles from:

```
python labels2festvox.py --labels path/to/labels/ --out festvox_labels --name asaxi_ms
```

- `festvox_labels/lab/<utt>.lab` — EST label files (`end_time  color  phone`)
- `festvox_labels/asaxi_ms.mlf` — one HTK master label file (100 ns units)

It fills inter-label gaps with `pau`, collapses adjacent identical labels, and
normalizes silence/breath symbols (`-`, `sil`, `sp`, `inh`, `exh`, `br`) to
`pau`. A best-effort `--from oto` path exists but Audacity is recommended.

---

## Part 3 — recording & boundary protocol

### 3.1 Recording the continuous speech

Consistency with the existing reclist matters more than absolute quality —
mismatched units are what make unit selection audibly "seam."

- **Same voice, same room, same mic, same distance, same gain** as the
  reclist session. If those are gone, re-record enough of the reclist to
  re-anchor timbre.
- **Pitch.** The reclist is monotone at **F#3 (~185 Hz)**. Natural sentences
  *must* have real intonation, but keep the **median** near F#3 and the range
  moderate (±4–5 semitones). Wild F0 excursions raise F0 join cost against the
  monotone reclist units and cause audible pitch jumps. Do **not** whisper or
  creak — modal voice only (creak destroys the join-cost F0 track).
- **Cadence.** One steady speaking rate, ~**11–13 phones/s**. Don't
  accelerate through function words — those are exactly the units you're
  banking. Leave a clear ~**300 ms silence** at each sentence start and end.
- **Pitch drift.** Re-anchor pitch every few sentences (play an F#3 reference
  tone, hum to it). Track fatigue: energy and F0 sag over a session — record
  in **10–15 min blocks**, re-check level and pitch each block.
- **Breath.** Breathe **only in the leading/trailing silence**, never
  mid-sentence in a unit you'll cut. Label audible breaths as their own
  segment (→ normalized to `pau`) so they never get selected as speech.
- **Levels.** −18 dBFS average, peaks < −6 dBFS, no limiting/compression, no
  noise reduction (it smears spectra and wrecks join cost). 44.1 kHz/16-bit
  mono to match the bank.
- **One sentence per take/file**, named to match Script 1's ids
  (`asx_0001.wav`). Re-record fluffs; don't splice.

### 3.2 Placing boundaries for clean Multisyn joins

**Important correction (thanks to your oto screenshots).** Multisyn's units are
diphones the engine extracts *from your phone labels* — it splices in the stable
middle of each phone and computes those mid-phone points **itself** from the
boundaries you give it. So **you never hand-place a steady-state splice**; your
only job is to mark the **phone TRANSITIONS** accurately. (An earlier draft of
this file said "cut at the formant-steady midpoint" — that was diphone-splicing
advice and is wrong for *labeling*; the engine does the mid-phone cut.)

Two consequences that your CV instinct half-covers:

- **Every phone gets TWO boundaries** — onset and offset — and each is a
  transition to a neighbour. A CV oto gives only **one** anchor (the C→V point);
  a continuous label track must also mark the consonant's **own onset**.
- Match the reclist's segmentation convention so units are interchangeable.

Per manner (the **C→V** boundary is the one your oto Pre/Con marks):

- **Stops (`p t k b d g`, held `cl`):** C→V at the **burst release**; the silent
  closure belongs to the stop. prev→stop at the closure onset.
- **Voiceless fricatives (`f s sh th h`):** C→V at the **vowel onset** (= frication
  offset, where periodic voicing starts). prev→f at frication onset. **For these
  your CV preutterance ≈ vowel onset — you are exactly right** (image 1, `f ae`).
- **Affricates (`ch ts dz jh`):** C→V at the **end of frication** / vowel voicing
  onset.
- **Voiced fricatives (`v z zh dh`):** C→V at the **vowel onset**; prev→C at the
  frication/voicing onset.
- **Nasals (`m n ng nn mm nng`):** C→V at the **vowel onset** — where full oral
  formants appear. ⚠ In your `n ae` shot (image 2) that point is the **`Con`
  marker, not `Pre`**: for a voiced nasal, Pre sits *inside* the murmur (its
  onset = the prev→n boundary). Label n→V at the formant brightening, and n's
  onset at the murmur start — two boundaries, and neither is the Pre line.
- **Glides (`w y`):** C→V at the **constriction release** — where F2 stops moving
  toward the vowel target (≈ vowel onset). Your `y ae` shot (image 3): Pre is at
  the glide onset (prev→y); the y→ae boundary is later, near `Con`.
- **Diphthongs (`ay aw ow oy ey`):** these are **single phones** — label
  `[onset, offset]` and **do not cut inside them**; the internal glide is part
  of the phone. (This replaces the old "diphthong parts / midpoint" line.)
- **Taps/liquids (`dx r l ry ly`):** at their acoustic **edges** (constriction
  on/offset), like any phone.
- **Silences/pauses:** snap boundaries to a **zero crossing** and label `pau`.
  Trim leading/trailing silence to ~50 ms, consistent across files.
- **Consistency beats correctness.** If the reclist put stop boundaries at the
  burst, put *these* at the burst too. A systematic offset that matches the
  reclist joins cleanly; a "more accurate" but *different* convention does not.
- **Zoom to sample level** for stops and silences; use spectrogram view for
  fricatives/nasals/formants. In Audacity: Analyze → label track, one label
  per phone, drag boundaries against the spectrogram, then **Export Labels**.
- **Never leave gaps or overlaps** between adjacent phones within a word —
  Script 2 fills gaps with `pau`, which you do *not* want inside a word.

Feed the exported `.txt` to `labels2festvox.py`, then compile the Multisyn
voice from `festvox_labels/` + the wavs alongside your existing reclist units.

### 3.3 The same boundary rules, explained in `oto.ini` terms

You know oto geometry, so here is the precise translation — with the correction
you rightly pushed on. A Multisyn label is a **phone boundary** (a point). The
gotcha a CV oto hides: an oto has **one** preutterance per unit, but a continuous
label track needs a boundary at **both edges of every phone**. So the oto→Multisyn
mapping of the **C→V boundary depends on voicing**:

- **Voiceless obstruents (`p t k`, `f s sh th h`, `ch ts`):** the CV
  **preutterance ≈ the burst/vowel-onset boundary** — copying Pre works. This is
  the `f ae` case you cited, and you're right.
- **Voiced consonants (nasals, glides, voiced fricatives, liquids):** the C→V
  boundary Multisyn wants is the **vowel onset ≈ your oto `Consonant` marker**
  (fixed-region end), **not** the preutterance. For these, `Pre` sits at the
  consonant's *own onset* — which is the *other* boundary (prev→C). You label
  both. (This is the `n ae` case: label at `Con`, not `Pre`.)

| Phone type | prev→C boundary (onset) | **C→V boundary** (the join point) | oto landmark at C→V |
| --- | --- | --- | --- |
| **Stops / `cl`** (`p t k b d g`) | closure onset | **burst release** | preutterance |
| **Voiceless fric.** (`f s sh th h`) | frication onset | **vowel onset** (fric. offset) | preutterance |
| **Affricates** (`ch ts dz jh`) | closure/fric. onset | **vowel onset** (fric. offset) | preutterance |
| **Voiced fric.** (`v z zh dh`) | frication onset | **vowel onset** | `Consonant` marker |
| **Nasals** (`m n ng nn mm nng`) | murmur onset (`Pre`) | **vowel onset** (formants appear) | **`Consonant` marker, not `Pre`** |
| **Glides** (`w y`) | glide onset (`Pre`) | **constriction release** ≈ vowel onset | `Consonant` marker |
| **Taps / liquids** (`dx r l ry ly`) | constriction onset | constriction offset | `Consonant` marker |
| **Diphthongs** (`ay aw ow oy ey`) | — single phone — | label `[onset, offset]`, **no internal cut** | — |

Two more mappings from oto habits:

- **`overlap` → forget it.** In diphones you widen `overlap` to smooth a seam.
  In Multisyn there is no crossfade to smooth; the join is butt-spliced at a
  pitch mark and scored. Set your mental `overlap` to **0** — a wide overlap
  would only blur where the true boundary is. (The `overlap` *number* is not
  written to the `.lab` at all; only the boundary time is.)
- **left/right blank (cutoff) → the neighbouring phone's boundary.** In oto,
  blank marks where a unit stops being usable. In a continuous label track the
  "cutoff" of phone *N* is simply the `preutterance` of phone *N+1* — there is
  no dead zone between them. That is why **§3.2 forbids gaps inside a word**:
  an oto with a right-blank that doesn't meet the next unit's left-offset would
  leave silence Multisyn fills with `pau`, which you never want mid-word.
- **`consonant` (the fixed-region end) → its POSITION is useful, its meaning
  isn't.** As a "don't-stretch length" it's irrelevant to Multisyn. But its
  *position* ≈ **vowel onset**, which for voiced consonants **is** the C→V
  boundary — so for nasals/glides/voiced fricatives, read the boundary off the
  `Consonant` line, not `Pre`. (For voiceless obstruents `Pre` already sits
  there.)

So if you would rather annotate in UTAU than Audacity, the honest summary is:
**you'd be maintaining a whole oto (five numbers per unit) just to read one or
two *positions* off it** — `Pre` for voiceless obstruents, the `Consonant` line
for voiced consonants — **and discard the rest**, all while fighting oto's
one-unit-per-file model on continuous sentences. That's the concrete, oto-level
reason Part 1 lands on Audacity: a label track lets you drop the phone-boundary
points directly, with none of the crossfade machinery to reverse-engineer.

> **Heads-up on the `--from oto` fallback:** `labels2festvox.py` currently reads
> each entry's **`preutterance`** as the boundary. Per the correction above that
> is right for *voiceless* units but lands *inside the murmur/glide* for voiced
> ones — so the oto path is doubly "best-effort." If you must go through oto,
> set `Pre` at the vowel onset for voiced consonants too (non-standard for
> singing, but it makes the reduction correct), or just use Audacity.

---

## Appendix — Adding Japanese (assessment)

**Verdict: easy for kana/romaji input; hard only for kanji.** You already have
every piece except a kanji reader.

**Why it's easy.** The uploaded `en-jap-mapping.yaml` is a complete OpenUTAU
phonemizer: **1,247 hiragana + 289 katakana** graphemes already map to *this
bank's phone set* (`ガ → [ng,a]`, `し → [sh,i]`, …). And the 4_Fis3 bank was
recorded with the Japanese-support phones (`a i u e o`, `ts dz h`, the
palatalized `Cy` series, syllabic `nn mm nng xn`) — verified present in the
built DB as diphone units. So the acoustic coverage and the g2p table both
already exist.

**What was done.** `synth_diphone.py` now has `g2p_japanese()` that (a) loads
the kana→phone table straight from that YAML, and (b) accepts **romaji** via a
built-in Hepburn romaji→kana front (sokuon `っ`→`cl`, long vowels→lengthen).
Working today, 0 missing diphones:

```
python synth_diphone.py "konnichiwa" --lang ja     # → k o nn ny i ch i w a
python synth_diphone.py "きゃっと"   --lang ja     # → ky a cl t o
```

**The one hard part — kanji.** Real Japanese text is kanji+kana. Kanji→reading
needs a morphological analyzer (MeCab/UniDic, or `pykakasi`/`fugashi`) — an
external dependency and the only non-trivial addition. Difficulty ladder:

| Input | Difficulty | Status |
| --- | --- | --- |
| **kana** (ひらがな/カタカナ) | trivial | ✅ done (YAML lookup) |
| **romaji** | easy | ✅ done (built-in Hepburn) |
| **kanji / mixed** | moderate | needs `pip install fugashi unidic-lite` → readings → existing pipeline |

Prosody caveat: pitch-accent isn't modeled (the bank is monotone), so Japanese
will sound flat — same limitation as Asaxi/English here, and exactly what the
Multisyn upgrade above fixes once you record intonation.

*Default synthesis language is now **English** (`festvox.json → "default_lang"`);
`--lang asaxi|en|ja` overrides per call.*
