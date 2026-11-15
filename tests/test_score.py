import json

from distill.fmt import EOS, to_prompt_completion
from distill.score import item_metrics, paired_ci, score_files, summarize

POSTING = "Build streaming data pipelines with Apache Kafka and Spark. Deploy services on Kubernetes in production."
RESUME = "Python developer who deployed services on Kubernetes."
GOOD = json.dumps({"fit_summary": "ok", "matched_skills": ["Kubernetes"], "missing_skills": ["Kafka", "Spark"],
                   "evidence": [{"claim": "a", "quote": "Build streaming data pipelines with Apache Kafka"},
                                {"claim": "b", "quote": "Deploy services on Kubernetes in production"}]})
HALF = json.dumps({"fit_summary": "ok", "matched_skills": ["Kubernetes", "Python"], "missing_skills": ["Kafka"],
                   "evidence": [{"claim": "a", "quote": "Build streaming data pipelines with Apache Kafka"},
                                {"claim": "b", "quote": "Ten years of COBOL on mainframes required here"}]})


def test_item_metrics_and_summary():
    ref = json.loads(GOOD)
    g, h, bad = (item_metrics(t, RESUME, POSTING, ref) for t in (GOOD, HALF, "not json"))
    assert g["strict"] and g["matched_skills_f1"] == 1.0 and g["missing_skills_f1"] == 1.0
    assert h["valid"] and not h["strict"] and (h["n_quotes"], h["n_quotes_ok"]) == (2, 1)
    assert abs(h["matched_skills_f1"] - 2 / 3) < 1e-9 and not bad["valid"]
    s = summarize([g, h, bad])
    assert abs(s["valid_json"] - 2 / 3) < 1e-9 and abs(s["strict_pass"] - 1 / 3) < 1e-9
    assert s["citation_precision"] == 3 / 4 and s["fail_reasons"] == {"quote_not_in_posting": 1, "no_json": 1}


def test_paired_ci_detects_consistent_gain():
    d, lo, hi = paired_ci([0, 0, 1, 0] * 25, [1, 1, 1, 0] * 25)
    assert d == 0.5 and lo > 0.3 and hi <= 0.7


def test_score_files_uses_only_eval_pairs(tmp_path):
    pairs = [{"resume_job_id": "r", "job_id": "j1", "split": "eval"},
             {"resume_job_id": "r", "job_id": "j2", "split": "train"}]
    f = tmp_path / "student.jsonl"
    f.write_text("".join(json.dumps({"resume_job_id": "r", "job_id": j, "text": GOOD, "system": "s"}) + "\n"
                         for j in ("j1", "j2")))
    res = score_files([str(f)], pairs, {"r": RESUME}, {"j1": POSTING, "j2": POSTING}, {})
    assert res["systems"]["s"]["n"] == 1 and res["systems"]["s"]["coverage"] == 1.0


class _Tok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert add_generation_prompt and not enable_thinking and not tokenize
        return "".join(m["content"] for m in messages) + "<A>"


def test_prompt_completion_appends_eos():
    row = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                        {"role": "assistant", "content": "{}"}]}
    assert to_prompt_completion(_Tok(), row) == {"prompt": "SU<A>", "completion": "{}" + EOS}
