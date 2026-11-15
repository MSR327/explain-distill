"""Score explanation outputs of any system (base / student / teacher / Groq) on the held-out eval pairs.

    python -m distill.score data/outputs/*.jsonl --out results/quality.json

Every system is scored on ALL eval pairs (not only those the teacher passed), so a student is never graded
on a subset chosen by its teacher. Per item: valid JSON, strict pass (all of the filter's rules), citation
precision, lexical missing-skill precision, and skill-list F1 against the teacher's filtered answer when one
exists. System comparisons use a paired bootstrap over pairs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from jobmatch.explain import check_citations, parse_json

from . import filter as quality

STRUCTURAL = {"no_json", "missing_keys", "bad_types"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", s.lower()).strip()


def _f1(pred: set, gold: set) -> float | None:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    return 2 * tp / (len(pred) + len(gold))


def item_metrics(text: str, resume: str, posting: str, reference: dict | None = None) -> dict:
    reasons, _ = quality.check(text, resume, posting)
    m = {"valid": not (set(reasons) & STRUCTURAL), "strict": not reasons, "reasons": reasons}
    if not m["valid"]:
        return m
    obj = parse_json(text)
    ev = obj["evidence"]
    m["n_quotes"], m["n_quotes_ok"] = len(ev), len(check_citations(ev, posting))
    res = f" {_norm(resume)} "
    miss = [s for s in obj["missing_skills"] if _norm(s)]
    m["n_missing"], m["n_missing_ok"] = len(miss), sum(f" {_norm(s)} " not in res for s in miss)
    if reference:
        for k in ("matched_skills", "missing_skills"):
            m[f"{k}_f1"] = _f1({_norm(s) for s in obj[k]}, {_norm(s) for s in reference[k]})
    return m


def summarize(items: list[dict]) -> dict:
    n = len(items)
    valid = [x for x in items if x["valid"]]

    def ratio(a, b):
        return sum(x.get(a, 0) for x in valid) / max(1, sum(x.get(b, 0) for x in valid))

    out = {"n": n, "valid_json": len(valid) / n, "strict_pass": sum(x["strict"] for x in items) / n,
           "citation_precision": ratio("n_quotes_ok", "n_quotes"),
           "missing_skill_precision": ratio("n_missing_ok", "n_missing")}
    for k in ("matched_skills_f1", "missing_skills_f1"):
        vals = [x[k] for x in valid if k in x]
        out[k] = float(np.mean(vals)) if vals else None
    reasons = {}
    for x in items:
        for r in x["reasons"]:
            reasons[r] = reasons.get(r, 0) + 1
    out["fail_reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    return out


def paired_ci(a: list[float], b: list[float], n_boot: int = 5000, seed: int = 0) -> list[float]:
    """Mean of (b - a) with a 95% paired-bootstrap interval."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boot = (y - x)[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return [float((y - x).mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def score_files(paths: list[str], pairs: list[dict], resumes: dict, postings: dict, references: dict) -> dict:
    keys = [p["resume_job_id"] + "|" + p["job_id"] for p in pairs if p["split"] == "eval"]
    by_key = {p["resume_job_id"] + "|" + p["job_id"]: p for p in pairs}
    systems, per_item = {}, {}
    for path in paths:
        rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        name = rows[0].get("system") or Path(path).stem
        texts = {r["resume_job_id"] + "|" + r["job_id"]: r["text"] for r in rows}
        items = []
        for k in keys:
            p = by_key[k]
            if k not in texts:
                continue
            items.append(item_metrics(texts[k], resumes[p["resume_job_id"]], postings[p["job_id"]], references.get(k)))
        systems[name] = summarize(items)
        systems[name]["coverage"] = len(items) / len(keys)
        per_item[name] = {k: x for k, x in zip([k for k in keys if k in texts], items, strict=True)}
    return {"systems": systems, "per_item": per_item}


def main() -> None:
    from .teacher import _read, load_index

    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", nargs="+")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/quality.json")
    a = ap.parse_args()
    data = Path(a.data)
    index = load_index()
    postings = dict(zip(index.jobs["job_id"].astype(str), index.jobs["description"], strict=True))
    resumes = {r["job_id"]: r["resume"] for r in _read(data / "resumes.jsonl")}
    references = {r["resume_job_id"] + "|" + r["job_id"]: json.loads(r["messages"][-1]["content"])
                  for r in _read(data / "sft_eval.jsonl")}
    res = score_files(a.outputs, _read(data / "pairs.jsonl"), resumes, postings, references)
    names = list(res["systems"])
    common = set.intersection(*(set(res["per_item"][n]) for n in names)) if names else set()
    res["paired_strict_pass_ci95"] = {
        f"{b} - {names[0]}": paired_ci([res["per_item"][names[0]][k]["strict"] for k in sorted(common)],
                                       [res["per_item"][b][k]["strict"] for k in sorted(common)])
        for b in names[1:]} if len(names) > 1 and common else {}
    res["n_common"] = len(common)
    out = {k: v for k, v in res.items() if k != "per_item"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
