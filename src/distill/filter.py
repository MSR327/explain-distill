"""Quality filter for teacher outputs: the student only ever sees grounded, well-formed examples.

Each rule mirrors something the app or project 2's evaluation checks, so filtering is "rejection sampling
against the app's own metrics". `check` returns the reasons an output fails (empty = keep) and, for kept
outputs, the canonical JSON target the student is trained to produce.
"""

from __future__ import annotations

import json
import re

from jobmatch.explain import check_citations, parse_json

KEYS = ("fit_summary", "matched_skills", "missing_skills", "evidence")


def _norm_words(s: str) -> str:
    return " " + re.sub(r"[^a-z0-9+#]+", " ", s.lower()).strip() + " "


def check(text: str, resume: str, posting: str) -> tuple[list[str], str | None]:
    """(failure reasons, canonical target JSON or None)."""
    try:
        obj = parse_json(text)
    except ValueError:
        return ["no_json"], None
    reasons = []
    if any(k not in obj for k in KEYS):
        return ["missing_keys"], None
    summary = obj["fit_summary"]
    matched, missing, evidence = obj["matched_skills"], obj["missing_skills"], obj["evidence"]
    if not (isinstance(summary, str) and summary.strip()):
        reasons.append("empty_summary")
    lists_ok = all(isinstance(x, list) and all(isinstance(s, str) for s in x) for x in (matched, missing))
    ev_ok = isinstance(evidence, list) and all(
        isinstance(e, dict) and isinstance(e.get("quote"), str) and isinstance(e.get("claim"), str) for e in evidence)
    if not (lists_ok and ev_ok):
        return [*reasons, "bad_types"], None
    if not 2 <= len(evidence) <= 4:
        reasons.append("n_evidence")
    if any(not 5 <= len(e["quote"].split()) <= 25 for e in evidence):
        reasons.append("quote_length")
    if len(check_citations(evidence, posting)) < len(evidence):
        reasons.append("quote_not_in_posting")
    resume_words = _norm_words(resume)
    if any(_norm_words(s) in resume_words for s in missing if s.strip()):
        reasons.append("missing_skill_in_resume")  # says "missing" but the resume literally has it
    if reasons:
        return reasons, None
    target = {"fit_summary": summary.strip(), "matched_skills": [s.strip() for s in matched],
              "missing_skills": [s.strip() for s in missing],
              "evidence": [{"claim": e["claim"].strip(), "quote": e["quote"].strip()} for e in evidence]}
    return [], json.dumps(target, ensure_ascii=False)


def repair(text: str, resume: str, posting: str) -> str | None:
    """Salvage an output that failed `check` only on some evidence items or skills (else None).

    Drops evidence items whose quote is the wrong length or not found in the posting, keeps the first 4 verified
    ones, and removes "missing" skills the resume has. Returns the repaired JSON text, which the caller must
    still pass through `check`. The 14B teacher paraphrases a quote or two in most answers; rejecting the whole
    answer threw away 74% of them and skewed the data toward unrelated job pairs (strict pass rate by pair kind:
    source 12%, near 23%, far 44%).
    """
    try:
        obj = parse_json(text)
    except ValueError:
        return None
    if any(k not in obj for k in KEYS) or not (isinstance(obj["fit_summary"], str) and obj["fit_summary"].strip()):
        return None
    matched, missing, evidence = obj["matched_skills"], obj["missing_skills"], obj["evidence"]
    if not all(isinstance(x, list) and all(isinstance(s, str) for s in x) for x in (matched, missing)):
        return None
    if not isinstance(evidence, list):
        return None
    good = [e for e in evidence
            if isinstance(e, dict) and isinstance(e.get("quote"), str) and isinstance(e.get("claim"), str)
            and 5 <= len(e["quote"].split()) <= 25 and check_citations([e], posting)][:4]
    if len(good) < 2:
        return None
    resume_words = _norm_words(resume)
    missing = [s for s in missing if not (s.strip() and _norm_words(s) in resume_words)]
    return json.dumps({**obj, "missing_skills": missing, "evidence": good}, ensure_ascii=False)
