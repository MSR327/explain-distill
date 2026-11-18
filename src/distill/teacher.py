"""Teacher data: synthetic resumes -> (resume, job) pairs with the app's skill hints -> teacher explanations.

    python -m distill.teacher resumes --n-train 1000 --n-eval 120 --model Qwen/Qwen3-14B-AWQ --tp 2
    python -m distill.teacher pairs                     # CPU: project 2's retrieval + skill extraction
    python -m distill.teacher explain --model Qwen/Qwen3-14B-AWQ --tp 2
    python -m distill.teacher filter                    # local: rejection sampling -> sft_train/sft_eval.jsonl

Every stage reads the previous stage's file and writes its own, so any stage can be re-run alone. `--limit N`
runs a stage on the first N items (the Kaggle notebook does a smoke pass with it before the full run).
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
from pathlib import Path

from jobmatch.explain import build_messages
from jobmatch.metrics import ngram_overlap
from jobmatch.synth import SENIORITY, synth_messages

from . import filter as quality
from .data import pick_pair_jobs, pick_sources, skill_hints, split_of

log = logging.getLogger("distill.teacher")
DATA = Path("data")
INDEX_REPO, INDEX_VERSION = "srish327/tech-jobs-index", "v1"
P2_SYNTH = ("https://raw.githubusercontent.com/MSR327/resume-job-matcher/81320b5/data/eval/synth.jsonl")
MAX_OVERLAP = 0.10  # same rule as project 2: reject resumes sharing >10% of 4-grams with the posting


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def load_index():
    from huggingface_hub import snapshot_download
    from jobmatch.index import JobIndex

    root = snapshot_download(INDEX_REPO, repo_type="dataset", allow_patterns=[f"{INDEX_VERSION}/*"])
    return JobIndex.load(f"{root}/{INDEX_VERSION}", mmap=True)


def p2_eval_job_ids() -> set[str]:
    import urllib.request

    return {json.loads(x)["job_id"] for x in urllib.request.urlopen(P2_SYNTH).read().decode().splitlines() if x}


def make_resumes(gen, jobs: list[dict], seed: int = 0, tries: int = 2) -> tuple[list[dict], collections.Counter]:
    """One resume per job, in batches across all jobs; a job that fails is retried once with a new sample."""
    from jobmatch.explain import parse_json

    rng = random.Random(seed)
    seniority = {j["job_id"]: rng.choice(SENIORITY) for j in jobs}
    done, stats, todo = {}, collections.Counter(), list(jobs)
    for attempt in range(tries):
        outs = gen.generate([synth_messages(j, seniority[j["job_id"]]) for j in todo], temperature=0.8,
                            max_tokens=600, seed=seed + attempt)
        retry = []
        for j, text in zip(todo, outs, strict=True):
            try:
                resume = str(parse_json(text).get("resume", "")).strip()
            except ValueError:
                stats["no_json"] += 1
                retry.append(j)
                continue
            ov = ngram_overlap(resume, j["description"], n=4)
            if len(resume.split()) < 60:
                stats["too_short"] += 1
                retry.append(j)
            elif ov > MAX_OVERLAP:
                stats["copied_posting"] += 1
                retry.append(j)
            else:
                done[j["job_id"]] = {"job_id": j["job_id"], "split": split_of(j["job_id"]),
                                     "seniority": seniority[j["job_id"]], "resume": resume, "overlap4": ov}
        todo = retry
        if not todo:
            break
    stats["kept"] = len(done)
    return [done[j["job_id"]] for j in jobs if j["job_id"] in done], stats


def make_pairs(index, resumes: list[dict], embed, extract, tau: float, alpha: dict, seed: int = 0) -> list[dict]:
    """Three (resume, job) pairs per resume with the exact messages the app would send the explainer."""
    import numpy as np
    from jobmatch.pipeline import Matcher

    ids = index.jobs["job_id"].astype(str).tolist()
    pos = {j: i for i, j in enumerate(ids)}
    descriptions = index.jobs["description"].tolist()
    splits = [split_of(j) for j in ids]
    by_split = {s: [i for i, x in enumerate(splits) if x == s] for s in ("train", "eval")}
    matcher = Matcher(index, embed.query, embed.encode, extract, alpha=alpha, tau=tau)
    rng, out = random.Random(seed), []
    for r in resumes:
        src = pos[r["job_id"]]
        split = splits[src]
        ranked = [m.job_idx for m in matcher.match(r["resume"], k=50, mode="hybrid", explain_top=False, pool=50)]
        ranked = [d for d in ranked if splits[d] == split]
        skills = sorted({s.strip() for s in extract(r["resume"]) if s.strip()}, key=str.lower)
        rvec = np.asarray(embed.encode(skills)) if skills else np.zeros((0, index.vectors.shape[1]), np.float32)
        for d, kind in pick_pair_jobs(src, ranked, by_split[split], descriptions, rng):
            matched, missing = skill_hints(index, rvec, d, tau)
            job = index.jobs.iloc[d].to_dict()
            out.append({"resume_job_id": r["job_id"], "job_id": ids[d], "kind": kind, "split": split,
                        "rank": ranked.index(d) + 1 if d in ranked else None,
                        "messages": build_messages(r["resume"], job, matched, missing)})
    return out


def run_filter(pairs: list[dict], teacher: list[dict], resumes: dict, postings: dict,
               do_repair: bool = True) -> tuple[dict, list, list]:
    """Rejection sampling with project 2's checks (plus `repair`); returns stats and SFT rows for train/eval.

    `kept_strict` is the share of teacher answers that pass untouched (a measure of the teacher); `kept` also
    counts answers salvaged by dropping their unverifiable evidence items.
    """
    out = {p["resume_job_id"] + "|" + p["job_id"]: p for p in pairs}
    stats = {"total": 0, "kept": 0, "kept_strict": 0, "kept_repaired": 0, "reasons": collections.Counter(),
             "by_kind": collections.defaultdict(lambda: [0, 0, 0])}
    rows = {"train": [], "eval": []}
    for t in teacher:
        p = out[t["resume_job_id"] + "|" + t["job_id"]]
        resume, posting = resumes[p["resume_job_id"]], postings[p["job_id"]]
        reasons, target = quality.check(t["text"], resume, posting)
        stats["total"] += 1
        stats["by_kind"][p["kind"]][2] += 1
        stats["reasons"].update(reasons)
        repaired = False
        if not target and do_repair and (fixed := quality.repair(t["text"], resume, posting)):
            target, repaired = quality.check(fixed, resume, posting)[1], True
        if target:
            stats["kept"] += 1
            stats["kept_repaired" if repaired else "kept_strict"] += 1
            stats["by_kind"][p["kind"]][0] += 1
            stats["by_kind"][p["kind"]][1] += not repaired
            rows[p["split"]].append({"messages": [*p["messages"], {"role": "assistant", "content": target}],
                                     "resume_job_id": p["resume_job_id"], "job_id": p["job_id"], "kind": p["kind"],
                                     "repaired": repaired})
    stats["reasons"] = dict(stats["reasons"].most_common())
    stats["by_kind"] = {k: {"kept": a, "kept_strict": b, "total": c} for k, (a, b, c) in stats["by_kind"].items()}
    return stats, rows["train"], rows["eval"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["resumes", "pairs", "explain", "filter"])
    ap.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--engine", choices=["vllm", "hf", "llamacpp"], default="vllm")
    ap.add_argument("--tokenizer", default=None, help="llamacpp engine: HF repo with the chat template")
    ap.add_argument("--llama-cpp", default="llama.cpp")
    ap.add_argument("--batch-size", type=int, default=16, help="hf engine only")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=120)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--no-repair", action="store_true", help="filter stage: reject instead of salvaging answers")
    a = ap.parse_args()
    data = Path(a.data)

    if a.stage == "resumes":
        from .gen import make_generator

        index = load_index()
        jobs_by_id = {str(j["job_id"]): j for j in index.jobs.to_dict("records")}
        src = pick_sources(list(jobs_by_id), p2_eval_job_ids(), a.n_train, a.n_eval)[: a.limit]
        gen = make_generator(a.engine, a.model, a.tp, a.batch_size, a.tokenizer, a.llama_cpp)
        rows, stats = make_resumes(gen, [jobs_by_id[j] for j in src])
        _write(data / "resumes.jsonl", rows)
        print("RESUMES", json.dumps({"requested": len(src), **stats}), flush=True)
    elif a.stage == "pairs":
        from jobmatch.config import ALPHA, SKILL_MODEL, TAU
        from jobmatch.embed import OnnxEmbedder
        from jobmatch.index import clean_skills
        from skillex.onnx_predict import OnnxSkillExtractor

        index = load_index()
        emb, ext = OnnxEmbedder(), OnnxSkillExtractor(SKILL_MODEL, low_memory=True)
        resumes = _read(data / "resumes.jsonl")[: a.limit]
        pairs = make_pairs(index, resumes, emb, lambda t: [s["text"] for s in clean_skills(ext.extract(t)["spans"])],
                           TAU, ALPHA)
        _write(data / "pairs.jsonl", pairs)
        print("PAIRS", json.dumps(collections.Counter(f'{p["split"]}/{p["kind"]}' for p in pairs)), flush=True)
    elif a.stage == "explain":
        from .gen import make_generator

        pairs = _read(data / "pairs.jsonl")[: a.limit]
        gen = make_generator(a.engine, a.model, a.tp, a.batch_size, a.tokenizer, a.llama_cpp)
        texts = gen.generate([p["messages"] for p in pairs], temperature=0.3, max_tokens=700)
        _write(data / "teacher.jsonl", [{"resume_job_id": p["resume_job_id"], "job_id": p["job_id"],
                                         "model": a.model, "text": t} for p, t in zip(pairs, texts, strict=True)])
        print("TEACHER", len(texts), "outputs", flush=True)
    else:
        index = load_index()
        postings = dict(zip(index.jobs["job_id"].astype(str), index.jobs["description"], strict=True))
        resumes = {r["job_id"]: r["resume"] for r in _read(data / "resumes.jsonl")}
        stats, train, ev = run_filter(_read(data / "pairs.jsonl"), _read(data / "teacher.jsonl"), resumes, postings,
                                    do_repair=not a.no_repair)
        _write(data / "sft_train.jsonl", train)
        _write(data / "sft_eval.jsonl", ev)
        (data / "filter_stats.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
