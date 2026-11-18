"""Large-model reference: Project 2's production explainer (Groq gpt-oss-120b, JSON mode) on held-out eval pairs.

    python scripts/groq_baseline.py --n-per-kind 20 --out data/outputs/groq-gpt-oss-120b.jsonl

Groq's free tier allows ~200k tokens/day, shared with the live Project 2 app, and one pair costs ~2k tokens, so
this scores a stratified sample (same number of near / far / source pairs, fixed seed) instead of all 360.
distill.score compares systems on the pairs they have in common, so the paired CIs stay valid on the subset.
Resumable: pairs already in --out are skipped, and the run stops before --token-budget is exceeded.
Reads GROQ_API_KEY from the environment or ../resume-job-matcher/.env (never printed).
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from jobmatch.llm import LLMError, groq


def api_key() -> str:
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    env = Path(__file__).resolve().parents[2] / "resume-job-matcher" / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("GROQ_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("GROQ_API_KEY not set")


def sample(pairs: list[dict], n_per_kind: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for kind in sorted({p["kind"] for p in pairs}):
        group = [p for p in pairs if p["kind"] == kind]
        out += rng.sample(group, min(n_per_kind, len(group)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs.jsonl")
    ap.add_argument("--n-per-kind", type=int, default=20)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--system", default="groq-gpt-oss-120b")
    ap.add_argument("--token-budget", type=int, default=130_000)
    ap.add_argument("--pace", type=float, default=15.0, help="seconds between calls: free tier is 8k tokens/min")
    ap.add_argument("--out", default="data/outputs/groq-gpt-oss-120b.jsonl")
    a = ap.parse_args()

    pairs = [p for p in map(json.loads, Path(a.pairs).read_text().splitlines()) if p["split"] == "eval"]
    todo = sample(pairs, a.n_per_kind)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    key = lambda r: (r["resume_job_id"], r["job_id"])  # noqa: E731  (a resume appears in several pairs)
    done = {key(json.loads(x)) for x in out.read_text().splitlines()} if out.exists() else set()
    llm = groq(api_key(), model=a.model, rate_limit_retries=6)  # the app fails fast; a batch job should wait
    used = failed = 0
    with out.open("a") as f:
        for i, p in enumerate(todo):
            if key(p) in done:
                continue
            if used >= a.token_budget:
                print(f"token budget {a.token_budget} reached; rerun later to resume", flush=True)
                break
            try:
                r = llm.complete(p["messages"], temperature=0.0, max_tokens=700, json_mode=True)
                text, used = r.text, used + r.prompt_tokens + r.completion_tokens
            except LLMError as e:  # not written: a rerun retries it (a rate-limit failure is not a model answer)
                failed += 1
                print("error:", e, flush=True)
                continue
            finally:
                time.sleep(a.pace)
            f.write(json.dumps({"system": a.system, "resume_job_id": p["resume_job_id"], "job_id": p["job_id"],
                                "text": text}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(todo)} tokens {used}", flush=True)
    n = len(out.read_text().splitlines())
    print("GENERATED", json.dumps({"system": a.system, "n": n, "tokens": used, "failed": failed}), flush=True)


if __name__ == "__main__":
    main()
