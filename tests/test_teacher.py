import json
import random

from distill.data import pick_pair_jobs, pick_sources, split_of
from distill.gen import FakeGenerator
from distill.teacher import make_resumes, run_filter

POSTING = "Build streaming data pipelines with Apache Kafka and Spark. Deploy services on Kubernetes in production."


def test_split_is_stable_and_roughly_15_percent():
    ids = [str(i) for i in range(4000)]
    assert [split_of(i) for i in ids] == [split_of(i) for i in ids]
    assert 0.12 < sum(split_of(i) == "eval" for i in ids) / len(ids) < 0.18


def test_sources_respect_split_and_exclusions():
    ids = [str(i) for i in range(2000)]
    src = pick_sources(ids, exclude={"5", "6"}, n_train=50, n_eval=10)
    assert len(src) == 60 and not {"5", "6"} & set(src)
    assert [split_of(j) for j in src] == ["train"] * 50 + ["eval"] * 10


def test_pair_jobs_skip_reposts_and_cover_three_kinds():
    desc = {i: f"job {i}" for i in range(200)}
    desc[7] = desc[0]  # repost of the source
    ranked = [7, 3, 4, 5, 6, *range(8, 60)]
    out = pick_pair_jobs(0, ranked, list(range(200)), desc, random.Random(0))
    kinds = dict((k, d) for d, k in out)
    assert kinds["source"] == 0 and kinds["near"] in {3, 4, 5, 6, 8} and kinds["far"] >= 60
    assert 7 not in [d for d, _ in out]


def test_make_resumes_retries_and_rejects_copies():
    jobs = [{"job_id": j, "title": "DE", "description": POSTING} for j in ("a", "b")]
    own = " ".join(f"word{i}" for i in range(80))
    calls = {"n": 0}

    def reply(conv):
        calls["n"] += 1
        if "Kafka" in conv[1]["content"] and calls["n"] == 1:
            return json.dumps({"resume": (POSTING + " ") * 8})  # copies the posting -> rejected, retried
        return json.dumps({"resume": own})

    rows, stats = make_resumes(FakeGenerator(reply), jobs)
    assert [r["job_id"] for r in rows] == ["a", "b"] and stats["copied_posting"] == 1 and stats["kept"] == 2


def test_run_filter_routes_by_split_and_counts_reasons():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    pairs = [{"resume_job_id": "r1", "job_id": "j1", "kind": "source", "split": "train", "messages": msgs},
             {"resume_job_id": "r1", "job_id": "j2", "kind": "far", "split": "eval", "messages": msgs}]
    ok = json.dumps({"fit_summary": "fit", "matched_skills": [], "missing_skills": ["Kafka"],
                     "evidence": [{"claim": "a", "quote": "Build streaming data pipelines with Apache Kafka"},
                                  {"claim": "b", "quote": "Deploy services on Kubernetes in production"}]})
    teacher = [{"resume_job_id": "r1", "job_id": "j1", "text": ok},
               {"resume_job_id": "r1", "job_id": "j2", "text": "oops"}]
    stats, train, ev = run_filter(pairs, teacher, {"r1": "python dev"}, {"j1": POSTING, "j2": POSTING})
    assert stats["kept"] == 1 and stats["reasons"] == {"no_json": 1} and ev == []
    assert train[0]["messages"][-1]["role"] == "assistant"
    assert stats["by_kind"]["far"] == {"kept": 0, "kept_strict": 0, "total": 1}
    assert stats["kept_strict"] == 1 and stats["kept_repaired"] == 0 and train[0]["repaired"] is False


def test_run_filter_repairs_partially_bad_answers_unless_disabled():
    msgs = [{"role": "user", "content": "u"}]
    pairs = [{"resume_job_id": "r1", "job_id": "j1", "kind": "source", "split": "train", "messages": msgs}]
    partly_bad = json.dumps({"fit_summary": "fit", "matched_skills": [], "missing_skills": ["Kafka"], "evidence": [
        {"claim": "a", "quote": "Build streaming data pipelines with Apache Kafka"},
        {"claim": "b", "quote": "Deploy services on Kubernetes in production"},
        {"claim": "c", "quote": "requires a PhD in astrophysics and five years of telescope work"}]})
    teacher = [{"resume_job_id": "r1", "job_id": "j1", "text": partly_bad}]
    args = (pairs, teacher, {"r1": "python dev"}, {"j1": POSTING})
    stats, train, _ = run_filter(*args)
    assert (stats["kept"], stats["kept_strict"], stats["kept_repaired"]) == (1, 0, 1) and train[0]["repaired"]
    assert len(json.loads(train[0]["messages"][-1]["content"])["evidence"]) == 2
    assert run_filter(*args, do_repair=False)[0]["kept"] == 0 and stats["reasons"] == {"quote_not_in_posting": 1}


def test_importing_distill_keeps_jax_off_the_gpu():
    # bm25s runs a JAX op at import; on a GPU box JAX would grab 75% of each GPU before llama-server starts.
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    out = subprocess.run([sys.executable, "-c", "import distill, os; print(os.environ['JAX_PLATFORMS'])"],
                         capture_output=True, text=True, env=env, check=True).stdout
    assert out.strip() == "cpu"
