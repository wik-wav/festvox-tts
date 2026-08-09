# -*- coding: utf-8 -*-
"""
corpus_extract.py — grammar-aware recording-script builder for a Multisyn
(unit-selection) upgrade of an Asaxi voice.

WHY THIS IS NOT A GREEDY DIPHONE SELECTOR
-----------------------------------------
The existing UTAU reclist already covers 100 % of Asaxi's diphone bases
(>=1 instance of every legal diphone). Multisyn does not need MORE diphone
*types* — it needs REDUNDANCY: many instances of each unit in varied
prosodic and coarticulatory contexts, and — crucially for a fusional/
agglutinative language — whole grammatical chunks (particle chains, affixed
verb stacks, frequent collocations) recorded *intact* so the unit selector
can lift a natural run rather than stitch one.

So this script:
  1. reads the Asaxi corpus, splits it into sentences, g2p's each one;
  2. mines high-frequency GRAMMATICAL constructions —
        * particle / function-word n-grams (e.g. "to wo", "dåni ... zè-"),
        * verb affix stacks (tense/neg/aspect prefixes + -ů/-nů),
        * plain word bigrams/trigrams (collocations);
  3. greedily selects the fewest sentences that raise every high-frequency
     construction to a target redundancy R (and tops up diphone redundancy
     as a secondary term), until the time budget from MULTISYN.md is met;
  4. writes a finalized recording script + a coverage report.

Stdlib only. Reuses the Asaxi g2p from the adjacent synth_diphone.py.

Usage
    python corpus_extract.py --corpus FILE [FILE ...] \
        [--minutes 60] [--redundancy 12] [--ngram-top 300] \
        [--out recording_script.txt]
Config file (festvox.json) may supply a "corpus_extract" block with the same
keys; CLI overrides it.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# --- reuse the Asaxi grapheme->phone rules from the synthesizer ------------
try:
    from synth_diphone import g2p_asaxi
except Exception as e:                                    # pragma: no cover
    sys.exit(f"cannot import g2p_asaxi from festvox/synth_diphone.py: {e}")

# --------------------------------------------------------------------------
# Asaxi grammatical knowledge (from 01_Function words in Asaxi + the grammar).
# These drive "grammatical chunk" detection — the thing Multisyn most wants
# recorded intact. Edit freely; everything downstream is data-driven.
# --------------------------------------------------------------------------
PARTICLES = set("""
to ă dhè sè bă zá då ni izo måmå ăni ga ja se si ŕa dzè sèni sèwo vå zå nivå
måniåkam chě chěxa chěná kè kkè tte wă wë nỏwă nỏwë naŕè nánaŕè ken ken.ná
xăxă dăxă pùxă xădăchỏxă hè kă xăkă náxăhè náxăkă nå panå hùnå vanå nåsi onå
opùnå nanå izånixå gămă ximă gănå ně ë e me ő wő aŕa iŕè jỏ wå ox vi pxů xă ná
nèŕa xiŕa dåni nani pùni bi ỏ gőnigő säsä săsă xa dăgo onă onýj anő ponă ponýj
wo no xő ko gő jo wa na xa ka gja hja nå
""".split())

# verb prefixes (tense / polarity / aspect / voice / mood) and suffixes
VPREFIX = ["pazè", "hùzè", "hùpa", "izozè", "kozè", "kopa", "opa", "ozè",
           "panå", "nåhè", "náxăbăhè", "náxăhè", "náxăkă", "băhè", "xăhè",
           "zè", "pa", "hù", "ko", "sỏ", "mi", "na", "ni", "chå", "jå",
           "xè", "ná", "xă", "fů", "bă", "kă", "dă"]
VSUFFIX = ["nů", "xů", "wů", "ků", "ŕů", "shů", "chů", "jů", "sů", "ŋů",
           "pů", "zhů", "ů", "ŕa", "nă", "nýj", "kam", "shá", "ŕo", "no",
           "ma", "wa", "hè", "wë", "kă", "ná"]


# --------------------------------------------------------------- corpus load
def strip_markdown(text):
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith(("#", "|", ">", "`", "---")):
            continue
        if s.startswith(("- ", "* ", "1.")) and "  " in s:
            continue                                       # list scaffolding
        if re.match(r"^[A-Za-z _]+:", s) and len(s) < 40:  # frontmatter-ish
            continue
        # drop wiki-link markup and bold/italic
        s = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", s)
        s = s.replace("**", "").replace("*", "").replace("__", "")
        out.append(s)
    return " ".join(out)


def split_sentences(text):
    # Asaxi ends clauses with . ! ? and the story uses « » and — dashes
    parts = re.split(r"(?<=[.!?])\s+|[—«»“”]", text)
    sents = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip(" -–—\t")
        # keep things that look like real Asaxi sentences (>=3 words)
        if len(p.split()) >= 3 and re.search(r"[a-zà-ÿŕńśŋůåăëèěýùáőỏ]", p, re.I):
            sents.append(p)
    return sents


def words_of(sentence):
    return re.findall(r"[^\s.,;:!?«»\"()\[\]0-9]+", sentence.lower())


# ------------------------------------------------------ grammatical features
def affix_tags(word):
    """Return grammatical tags for a word: matched prefixes/suffixes and
    whether it is a particle. These are the 'intact chunk' atoms."""
    tags = []
    if word in PARTICLES:
        tags.append(f"PTL:{word}")
    for pre in VPREFIX:
        if word.startswith(pre) and len(word) > len(pre) + 1:
            tags.append(f"PRE:{pre}-")
            break
    for suf in VSUFFIX:
        if word.endswith(suf) and len(word) > len(suf) + 1:
            tags.append(f"SUF:-{suf}")
            break
    return tags


def constructions(words):
    """All grammatical constructions in a sentence (the things we want
    recorded intact and redundantly)."""
    cons = Counter()
    # particle chains: runs of >=2 adjacent function words
    run = []
    for w in words + [None]:
        if w in PARTICLES:
            run.append(w)
        else:
            if len(run) >= 2:
                cons[f"CHAIN:{' '.join(run)}"] += 1
            run = []
    # affix tags (per word)
    for w in words:
        for tag in affix_tags(w):
            cons[tag] += 1
    # word bigrams / trigrams involving at least one function word or affix
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            if any(g in PARTICLES or affix_tags(g) for g in gram):
                cons[f"{n}G:{' '.join(gram)}"] += 1
    return cons


# --------------------------------------------------------------- selection
def diphones(phones):
    seq = ["pau"] + phones + ["pau"]
    return [f"{seq[i]}-{seq[i+1]}" for i in range(len(seq) - 1)]


def build(corpus_files, minutes, redundancy, ngram_top, phones_per_sec,
          diph_weight):
    # 1) load + featurize every candidate sentence
    raw = []
    for fp in corpus_files:
        raw.append(strip_markdown(Path(fp).read_text(encoding="utf-8")))
    seen, cands = set(), []
    global_con = Counter()
    for text in raw:
        for s in split_sentences(text):
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            w = words_of(s)
            ph = g2p_asaxi(s)
            if len(ph) < 4:
                continue
            con = constructions(w)
            global_con.update(con)
            cands.append({"text": s, "words": w, "phones": ph,
                          "diph": diphones(ph), "con": con,
                          "sec": len(ph) / phones_per_sec})
    if not cands:
        sys.exit("no usable sentences found in the corpus.")

    # 2) which constructions are 'high-frequency' enough to target?
    #    the top-N by corpus frequency (excluding hapax — those are covered
    #    once already by the diphone reclist).
    ranked = [c for c, n in global_con.most_common() if n >= 2]
    targets = set(ranked[:ngram_top])
    diph_all = Counter()
    for c in cands:
        diph_all.update(c["diph"])
    # 'frequent' diphones worth extra redundancy: top half by corpus freq
    diph_freq = [d for d, n in diph_all.most_common() if n >= 3]
    diph_targets = set(diph_freq)

    # 3) greedy set-cover with redundancy toward R, time-budgeted
    budget = minutes * 60.0
    con_have, diph_have = Counter(), Counter()
    chosen, used_sec = [], 0.0
    pool = cands[:]

    def gain(c):
        g = 0.0
        for con in c["con"]:
            if con in targets and con_have[con] < redundancy:
                # marginal value shrinks as we approach R (log-ish)
                g += 1.0 / (1 + con_have[con])
        for d in c["diph"]:
            if d in diph_targets and diph_have[d] < redundancy:
                g += diph_weight / (1 + diph_have[d])
        return g / max(1.0, c["sec"] ** 0.5)          # favor efficient length

    while pool and used_sec < budget:
        best = max(pool, key=gain)
        if gain(best) <= 0:
            break                                     # everything at target
        chosen.append(best)
        pool.remove(best)
        used_sec += best["sec"]
        con_have.update({k: 1 for k in best["con"] if k in targets})
        diph_have.update({k: 1 for k in best["diph"] if k in diph_targets})

    # 4) coverage stats
    cov_targets = sum(1 for c in targets if con_have[c] >= redundancy)
    report = {
        "candidates": len(cands),
        "selected": len(chosen),
        "minutes_selected": round(used_sec / 60.0, 1),
        "minutes_budget": minutes,
        "targets_total": len(targets),
        "targets_at_R": cov_targets,
        "R": redundancy,
        "mean_target_instances": round(
            sum(min(con_have[c], redundancy) for c in targets) /
            max(1, len(targets)), 2),
        "distinct_diphones_covered": len(
            [d for d in diph_targets if diph_have[d] > 0]),
    }
    return chosen, report, con_have, targets


def write_script(chosen, report, out_path):
    lines = ["# Asaxi Multisyn recording script",
             f"# selected {report['selected']} sentences "
             f"(~{report['minutes_selected']} min of speech)",
             f"# grammatical targets at redundancy R>={report['R']}: "
             f"{report['targets_at_R']}/{report['targets_total']}",
             "# columns: <id> <TAB> <sentence>   (phones on the next # line)",
             ""]
    for i, c in enumerate(chosen, 1):
        lines.append(f"asx_{i:04d}\t{c['text']}")
        lines.append(f"#   phones: {' '.join(c['phones'])}")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    rep = Path(out_path).with_suffix(".report.txt")
    rep.write_text(
        "corpus_extract coverage report\n" +
        "\n".join(f"{k:26}: {v}" for k, v in report.items()) + "\n",
        encoding="utf-8")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="*", default=None,
                    help="corpus text/markdown files")
    ap.add_argument("--config", default=None, help="festvox.json (optional)")
    ap.add_argument("--minutes", type=float, default=None,
                    help="target minutes of NEW speech (default 60)")
    ap.add_argument("--redundancy", type=int, default=None,
                    help="target instances per construction (default 12)")
    ap.add_argument("--ngram-top", type=int, default=None,
                    help="how many top constructions to target (default 300)")
    ap.add_argument("--phones-per-sec", type=float, default=12.0)
    ap.add_argument("--diph-weight", type=float, default=0.35,
                    help="weight of secondary diphone-redundancy term")
    ap.add_argument("--out", default="recording_script.txt")
    a = ap.parse_args()

    cfg = {}
    cf = Path(a.config) if a.config else \
        Path(__file__).resolve().parent / "festvox.json"
    if cf.is_file():
        cfg = json.loads(cf.read_text(encoding="utf-8")).get("corpus_extract", {})

    corpus = a.corpus or cfg.get("corpus")
    if not corpus:
        sys.exit("no --corpus files given (and none in festvox.json "
                 "'corpus_extract.corpus').")
    minutes = a.minutes if a.minutes is not None else cfg.get("minutes", 60)
    R = a.redundancy if a.redundancy is not None else cfg.get("redundancy", 12)
    top = a.ngram_top if a.ngram_top is not None else cfg.get("ngram_top", 300)

    chosen, report, con_have, targets = build(
        corpus, minutes, R, top, a.phones_per_sec, a.diph_weight)
    rep = write_script(chosen, report, a.out)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}\n      {rep}")


if __name__ == "__main__":
    main()
