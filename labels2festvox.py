# -*- coding: utf-8 -*-
"""
labels2festvox.py — turn manual phone annotations of the new continuous Asaxi
recordings into the label files FestVox Multisyn compiles a voice from.

RECOMMENDED INPUT: Audacity label tracks exported as .txt
    (see MULTISYN.md, Part 1 — Audacity's flat start/end/label track maps
     1:1 onto a phone segmentation, which is exactly what Multisyn's
     labeller/aligner consumes; oto.ini's per-unit preutterance/overlap/
     consonant landmarks are diphone-crossfade geometry, the wrong shape.)

    Audacity .txt line:   <start_sec>\\t<end_sec>\\t<label>

OUTPUTS (both — Multisyn/festival accept either):
    OUT/lab/<utt>.lab     EST label file  (end-time  color  phone)
    OUT/<name>.mlf        HTK master label file (100 ns units), all utts

A secondary --from oto  path is provided because you offered it, but it is
best-effort only (see note in oto_to_segments); prefer Audacity.

Stdlib only.

Usage
    python labels2festvox.py --labels DIR_OR_FILES [...] \
        [--out festvox_labels] [--name asaxi_ms] [--from audacity|oto]
"""
import argparse
import re
import sys
from pathlib import Path

# Normalize annotation symbols to the phone set. Silence/breath variants all
# collapse to 'pau' (Multisyn's silence model). Extend as needed.
SILENCE = {"", "-", "sil", "sp", "pau", "silence", "br", "br1", "br2", "BR",
           "inh", "exh", "rb", "ap", "cl_sil"}
NORMALIZE = {}                        # e.g. {"q": "cl"}  add your own overrides


def norm_label(lab):
    lab = lab.strip()
    if lab in SILENCE:
        return "pau"
    return NORMALIZE.get(lab, lab)


# ------------------------------------------------------- Audacity -> segments
def audacity_to_segments(path: Path):
    """Read an Audacity label .txt -> contiguous [(end_sec, phone)] with pau
    filling gaps. Returns segments in time order (Multisyn needs END times;
    each segment's start is the previous segment's end)."""
    regions = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not ln.strip() or ln.startswith("\\"):     # skip freq-range lines
            continue
        cols = ln.split("\t")
        if len(cols) < 3:
            cols = ln.split()
        if len(cols) < 3:
            continue
        try:
            start, end = float(cols[0]), float(cols[1])
        except ValueError:
            continue
        label = norm_label("\t".join(cols[2:]))
        if end < start:
            start, end = end, start
        regions.append([start, end, label])
    if not regions:
        return []
    regions.sort(key=lambda r: r[0])

    segs, cursor = [], 0.0
    for start, end, label in regions:
        if start > cursor + 1e-4:                     # gap -> pause
            segs.append((round(start, 6), "pau"))
        segs.append((round(max(end, cursor + 1e-4), 6), label))
        cursor = max(end, cursor)
    return segs


# ------------------------------------------------------------ oto -> segments
def oto_to_segments(path: Path, report):
    """BEST-EFFORT oto.ini -> segments, grouped by wav. UTAU landmarks are a
    per-unit crossfade model, not a phone segmentation, so this only yields
    coarse [boundary, end] pairs per alias. Prefer Audacity. Returns a dict
    {utt_stem: segments}."""
    from collections import defaultdict
    import wave
    by_wav = defaultdict(list)
    for enc in ("utf-8", "cp932", "latin-1"):
        try:
            lines = path.read_text(encoding=enc).splitlines()
            break
        except UnicodeDecodeError:
            continue
    base = path.parent
    for ln in lines:
        if "=" not in ln:
            continue
        wav, rest = ln.split("=", 1)
        parts = rest.split(",")
        if len(parts) < 6:
            continue
        alias = re.sub(r"[A-G]#?\d+$", "", parts[0]).strip()
        try:
            offset, _cons, blank, preutt, _ov = map(float, parts[1:6])
        except ValueError:
            continue
        toks = alias.split() or [alias]
        wpath = base / wav.strip()
        if not wpath.exists():
            report["missing_wav"].add(wav.strip())
            continue
        with wave.open(str(wpath), "rb") as w:
            total = w.getnframes() / w.getframerate() * 1000.0
        start = offset
        mid = offset + preutt
        end = offset + abs(blank) if blank < 0 else total - blank
        if not (0 <= start < mid < end):
            continue
        stem = Path(wav.strip()).stem
        # emit boundary + end as two segment ends (phone1|phone2)
        p1 = norm_label(toks[0])
        p2 = norm_label(toks[1]) if len(toks) > 1 else toks[0]
        by_wav[stem].append((round(mid / 1000, 6), p1))
        by_wav[stem].append((round(end / 1000, 6), p2))
    for stem in by_wav:
        by_wav[stem].sort()
    report["oto_note"] = ("oto.ini is best-effort — boundaries are per-unit "
                          "crossfade landmarks, not a true phone segmentation")
    return by_wav


def collapse(segs):
    """Merge consecutive identical labels (e.g. a leading pau + labelled pau)
    into one segment ending at the later boundary."""
    out = []
    for end, phone in segs:
        if out and out[-1][1] == phone:
            out[-1] = (end, phone)
        else:
            out.append((end, phone))
    return out


# --------------------------------------------------------------- EST / MLF io
def write_lab(segs, out_path: Path):
    """EST label file: header '#', then rows  <end_time> <color> <phone>."""
    lines = ["#"]
    for end, phone in segs:
        lines.append(f"    {end:.6f} 26 {phone}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mlf_block(stem, segs):
    """HTK MLF block: integer 100 ns start/end per phone, '.'-terminated."""
    out = [f'"*/{stem}.lab"']
    prev = 0
    for end, phone in segs:
        e = int(round(end * 1e7))
        out.append(f"{prev} {e} {phone}")
        prev = e
    out.append(".")
    return "\n".join(out)


# --------------------------------------------------------------------- driver
def gather_inputs(paths, mode):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += sorted(p.glob("*.ini" if mode == "oto" else "*.txt"))
        elif p.is_file():
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", required=True,
                    help="Audacity .txt files/dirs (or oto.ini with --from oto)")
    ap.add_argument("--from", dest="src", default="audacity",
                    choices=["audacity", "oto"],
                    help="annotation source (default: audacity — recommended)")
    ap.add_argument("--out", default="festvox_labels")
    ap.add_argument("--name", default="asaxi_ms",
                    help="basename for the combined .mlf")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "lab").mkdir(parents=True, exist_ok=True)
    report = {"missing_wav": set()}
    mlf = ["#!MLF!#"]
    n = 0

    if a.src == "audacity":
        for fp in gather_inputs(a.labels, "audacity"):
            segs = collapse(audacity_to_segments(fp))
            if not segs:
                print(f"! no segments in {fp.name}", file=sys.stderr)
                continue
            stem = fp.stem
            write_lab(segs, out / "lab" / f"{stem}.lab")
            mlf.append(mlf_block(stem, segs))
            n += 1
    else:
        for fp in gather_inputs(a.labels, "oto"):
            for stem, segs in oto_to_segments(fp, report).items():
                segs = collapse(segs)
                write_lab(segs, out / "lab" / f"{stem}.lab")
                mlf.append(mlf_block(stem, segs))
                n += 1

    (out / f"{a.name}.mlf").write_text("\n".join(mlf) + "\n", encoding="utf-8")
    print(f"wrote {n} .lab files -> {out/'lab'}")
    print(f"wrote combined MLF   -> {out/(a.name + '.mlf')}")
    if report.get("oto_note"):
        print("note:", report["oto_note"])
    if report["missing_wav"]:
        print("missing wavs:", sorted(report["missing_wav"])[:10])


if __name__ == "__main__":
    main()
