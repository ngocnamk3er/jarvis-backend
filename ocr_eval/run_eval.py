"""Measure where the OCR tools put their boxes on the synthetic documents.

    python -m ocr_eval.run_eval --split tune --lang-mode current
    python -m ocr_eval.run_eval --split heldout --lang-mode matched

For each target the way an agent uses the tools is imitated: look for the text in every line and every
table cell, take the best match. Then the box is compared with the true one:
  correct  found, and overlapping the true box by IoU >= 0.5
  wrong    found, but somewhere else (or a decoy that was boxed: a false positive)
  miss     the text is in the image and was not found
  decoy ok a decoy (text that is NOT there) correctly not found
"""

import argparse
import io
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from app.agents import ocr
from ocr_eval.generate import cases


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def find(lines, tables, text):
    best = None
    spans = list(lines) + [ocr.cell_line([c]) for t in tables for c in t.cells if c.words]
    for line in spans:
        hit = ocr.locate(line, text)
        if hit and (best is None or hit[2] > best[2]):
            best = hit
    return best


def run_case(case, lang_mode, psm=None, scale=None, engine=None):
    buf = io.BytesIO()
    case.image.save(buf, "PNG")
    lang = "vie+eng" if lang_mode == "current" else case.lang
    t0 = time.time()
    try:
        _, lines, tables = ocr.run_ocr(buf.getvalue(), lang, psm=psm, scale=scale, engine=engine)
    except Exception as e:  # noqa: BLE001
        return case, [("error", str(e)[:60], None)] * len(case.targets), time.time() - t0
    out = []
    for t in case.targets:
        hit = find(lines, tables, t.text)
        if t.box is None:
            out.append(("wrong" if hit else "decoy ok", t.label, None))
        elif hit is None:
            out.append(("miss", t.label, None))
        else:
            v = iou(hit[0], t.box)
            out.append(("correct" if v >= 0.5 else "wrong", t.label, round(v, 2)))
    return case, out, time.time() - t0


def dump_cases(folder, chosen):
    """Save the cases so they can be read on a machine that has no fonts or no tesseract."""
    import os

    os.makedirs(folder, exist_ok=True)
    index = []
    for c in chosen:
        c.image.save(os.path.join(folder, f"{c.name}.png"))
        index.append({"name": c.name, "category": c.category, "lang": c.lang,
                      "targets": [{"label": t.label, "text": t.text, "box": list(t.box) if t.box else None} for t in c.targets]})
    json.dump(index, open(os.path.join(folder, "index.json"), "w"), ensure_ascii=False)
    print(f"wrote {len(index)} cases to {folder}")


def load_cases(folder, split):
    import os

    from PIL import Image

    from ocr_eval.generate import Case, Target

    out = []
    for c in json.load(open(os.path.join(folder, "index.json"))):
        out.append(Case(c["name"], c["category"], c["lang"], Image.open(os.path.join(folder, c["name"] + ".png")).convert("RGB"),
                        [Target(t["label"], t["text"], tuple(t["box"]) if t["box"] else None) for t in c["targets"]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="both", choices=["tune", "heldout", "both"])
    ap.add_argument("--lang-mode", default="matched", choices=["current", "matched"])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--psm", default=None)
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--json", default="")
    ap.add_argument("--engine", default=None, choices=["tesseract", "paddle"])
    ap.add_argument("--dump", default="", help="write the cases (images and true boxes) to this directory and stop")
    ap.add_argument("--load", default="", help="read the cases from a directory made with --dump instead of generating them")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.dump:
        dump_cases(a.dump, cases(a.split))
        return
    chosen = load_cases(a.load, a.split) if a.load else cases(a.split)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(lambda c: run_case(c, a.lang_mode, a.psm, a.scale, a.engine), chosen))
    by_cat = defaultdict(lambda: defaultdict(int))
    ious = defaultdict(list)
    for case, rows, _ in results:
        for kind, label, v in rows:
            by_cat[case.category][kind] += 1
            if kind == "correct" and v is not None:
                ious[case.category].append(v)
            if a.verbose and kind in ("wrong", "miss", "error"):
                print(f"    {case.name:16} {kind:7} {label} {v if v is not None else ''}")
    kinds = ["correct", "wrong", "miss", "decoy ok"]
    print(f"\nsplit={a.split} lang={a.lang_mode}\n{'category':12} " + " ".join(f"{k:>8}" for k in kinds) + "   mean IoU of the correct")
    total = defaultdict(int)
    for cat, c in by_cat.items():
        print(f"{cat:12} " + " ".join(f"{c[k]:8d}" for k in kinds) + f"   {sum(ious[cat]) / len(ious[cat]):.2f}" if ious[cat] else f"{cat:12} " + " ".join(f"{c[k]:8d}" for k in kinds))
        for k in kinds:
            total[k] += c[k]
    real = total["correct"] + total["wrong"] + total["miss"] - sum(1 for _, rows, _ in results for kind, lab, _ in rows if kind == "wrong" and lab.startswith("decoy"))
    n = sum(total[k] for k in kinds)
    decoys = total["decoy ok"] + sum(1 for _, rows, _ in results for kind, lab, _ in rows if kind == "wrong" and lab.startswith("decoy"))
    false_pos = sum(1 for _, rows, _ in results for kind, lab, _ in rows if kind == "wrong" and lab.startswith("decoy"))
    print(f"{'ALL':12} " + " ".join(f"{total[k]:8d}" for k in kinds))
    print(f"\ntargets in the image: {real}; found correctly: {total['correct']} ({100 * total['correct'] / max(real, 1):.0f}%); wrong box (incl. decoys): {total['wrong']}; "
          f"decoys boxed by mistake: {false_pos} of {decoys}")
    if a.json:
        json.dump({"split": a.split, "lang": a.lang_mode, "totals": total}, open(a.json, "w"))


if __name__ == "__main__":
    main()
