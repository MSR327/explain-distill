"""Splits and pair selection for teacher data.

Leakage rules:
- Jobs are split by a stable hash of the job id (85% train / 15% eval). Train pairs use only train-split jobs and
  eval pairs only eval-split jobs, so the student is scored on postings it never saw.
- Resumes written from project 2's evaluation jobs (data/eval/synth.jsonl there) are excluded as sources, so
  nothing here overlaps project 2's L1/L2 queries.
- Each resume gets three jobs of different fit (source, a near retrieval neighbour, a far random job), so the
  student also learns to say "poor fit" and list real gaps.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

EVAL_FRAC = 0.15


def split_of(job_id: str, eval_frac: float = EVAL_FRAC) -> str:
    h = int(hashlib.sha256(str(job_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if h < eval_frac else "train"


def pick_sources(job_ids: list[str], exclude: set[str], n_train: int, n_eval: int, seed: int = 0) -> list[str]:
    """Source jobs for synthetic resumes: n_train from the train split, n_eval from the eval split."""
    rng = random.Random(seed)
    pool = sorted(set(job_ids) - exclude)
    rng.shuffle(pool)
    train = [j for j in pool if split_of(j) == "train"][:n_train]
    ev = [j for j in pool if split_of(j) == "eval"][:n_eval]
    return train + ev


def pick_pair_jobs(source: int, ranked: list[int], same_split: list[int], descriptions, rng: random.Random,
                   near_ranks: tuple[int, int] = (1, 5)) -> list[tuple[int, str]]:
    """(job index, kind) for one resume: its source job, a near neighbour, a far job.

    `ranked`: the app's ranking (best first) restricted to the resume's split; `same_split`: all job indices of
    that split. Reposts (identical description to the source) are never picked as near/far.
    """
    src_desc = descriptions[source]
    ranked = [d for d in ranked if d != source and descriptions[d] != src_desc]
    out = [(source, "source")]
    lo, hi = near_ranks
    if ranked[lo - 1:hi]:
        out.append((rng.choice(ranked[lo - 1:hi]), "near"))
    top = set(ranked[:50]) | {source}
    far = [d for d in same_split if d not in top and descriptions[d] != src_desc]
    if far:
        out.append((rng.choice(far), "far"))
    return out


def skill_hints(index, rvec: np.ndarray, d: int, tau: float) -> tuple[list[str], list[str]]:
    """matched / missing job skills for job d, exactly as the app's pipeline computes them."""
    from jobmatch.rerank import skill_coverage

    names = [s["text"] for s in index.skills[d]]
    miss = set(skill_coverage(rvec, index.job_skill_vecs(d), tau)[1])
    return [n for i, n in enumerate(names) if i not in miss], [names[i] for i in sorted(miss)]
