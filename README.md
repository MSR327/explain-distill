# explain-distill

Distill the grounded job-match explanations of [resume-job-matcher](https://github.com/MSR327/resume-job-matcher)
into small open LLMs (Qwen3 0.6B / 1.7B / 4B), then benchmark how to serve them on free hardware.

**Status:** work in progress. Design: [docs/design.md](docs/design.md).

Part of a three-project series:
[skillspan-ner](https://github.com/MSR327/skillspan-ner) (skill extraction) →
[resume-job-matcher](https://github.com/MSR327/resume-job-matcher) (retrieval + LLM explanations) →
explain-distill (make the LLM small, cheap and unlimited).

## Pipeline

1. **Resumes** (Kaggle T4, vLLM): synthetic resumes written by Qwen3-14B-AWQ from 1,120 job postings.
2. **Pairs** (CPU): each resume gets its source job, a near neighbour from the app's own ranking, and a far job,
   with the exact prompt and skill hints the app sends its explainer.
3. **Teacher** (Kaggle T4, vLLM): Qwen3-14B-AWQ writes JSON explanations.
4. **Filter**: keep only outputs that pass the app's own checks (verbatim quotes, correct missing skills, valid JSON).
5. **Fine-tune** three students with QLoRA, **evaluate** on held-out jobs, **benchmark** formats × engines.

## Run

```bash
pip install -e ".[pairs,dev]"
python scripts/make_probe_notebook.py                 # 0. Kaggle environment probe
python scripts/make_teacher_notebook.py <commit> resumes
python -m distill.teacher pairs                       # local, after downloading resumes.jsonl
python scripts/make_teacher_notebook.py <commit> explain
python -m distill.teacher filter
pytest -q
```
