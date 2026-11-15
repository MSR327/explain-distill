import json

from distill.filter import check

POSTING = ("We need an engineer who can build streaming data pipelines with Apache Kafka and Spark. "
           "Experience deploying services on Kubernetes in production is required. Python is a must.")
RESUME = "Data engineer. Built batch ETL in Python and Airflow; deployed services on Kubernetes."


def good(**over):
    obj = {"fit_summary": "Solid fit on Python and Kubernetes; no streaming experience.",
           "matched_skills": ["Python", "Kubernetes"], "missing_skills": ["Kafka", "Spark"],
           "evidence": [{"claim": "needs streaming",
                         "quote": "build streaming data pipelines with Apache Kafka and Spark"},
                        {"claim": "needs k8s", "quote": "Experience deploying services on Kubernetes in production"}]}
    obj.update(over)
    return json.dumps(obj)


def test_keeps_grounded_output_and_canonicalizes():
    reasons, target = check("Sure! ```json\n" + good() + "\n```", RESUME, POSTING)
    assert reasons == [] and json.loads(target)["missing_skills"] == ["Kafka", "Spark"]
    assert target.startswith('{"fit_summary"')  # canonical key order, chatter stripped


def test_rejects_invented_quote():
    ev = [{"claim": "x", "quote": "must have ten years of Rust and Haskell experience"},
          {"claim": "y", "quote": "Experience deploying services on Kubernetes in production"}]
    assert check(good(evidence=ev), RESUME, POSTING) == (["quote_not_in_posting"], None)


def test_rejects_missing_skill_that_resume_has():
    assert check(good(missing_skills=["Kafka", "Airflow"]), RESUME, POSTING)[0] == ["missing_skill_in_resume"]


def test_rejects_bad_shapes():
    assert check("no json here", RESUME, POSTING)[0] == ["no_json"]
    assert check(json.dumps({"fit_summary": "x"}), RESUME, POSTING)[0] == ["missing_keys"]
    assert check(good(matched_skills="Python"), RESUME, POSTING)[0] == ["bad_types"]
    one = [{"claim": "k8s", "quote": "Experience deploying services on Kubernetes in production"}]
    assert check(good(evidence=one), RESUME, POSTING)[0] == ["n_evidence"]
    short = [{"claim": "a", "quote": "Python is a must"}, *one]
    assert "quote_length" in check(good(evidence=short), RESUME, POSTING)[0]
