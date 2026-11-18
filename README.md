# explain-distill

Distill the grounded job-match explanations of [resume-job-matcher](https://github.com/MSR327/resume-job-matcher)
into small open LLMs (Qwen3 0.6B / 1.7B / 4B), then benchmark how to serve them on free hardware.

**Headline (360 held-out resume–job pairs):** QLoRA distillation lifts Qwen3-0.6B from **19.7% → 50.3%** strict
pass (paired +30.6 pts, 95% CI 24.2–36.9) and quote precision from 75.6% → 93.8%. On a 60-pair subset the 1.7B
student reaches **92% of gpt-oss-120b's** strict-pass rate at ~1/70 of the total parameters (gpt-oss is a MoE
with ~5B active), and both students beat it on missing-skills F1. Everything ran on free tiers (Kaggle 2×T4, Groq free API).

**Status:** 0.6B and 1.7B students done; 4B training; serving benchmark and CPU deployment next.
Design: [docs/design.md](docs/design.md).

Part of a three-project series:
[skillspan-ner](https://github.com/MSR327/skillspan-ner) (skill extraction) →
[resume-job-matcher](https://github.com/MSR327/resume-job-matcher) (retrieval + LLM explanations) →
explain-distill (make the LLM small, cheap and unlimited).

## Results

Every system is scored with resume-job-matcher's own checks. **Strict pass** = valid JSON with all keys, 2–4
evidence quotes of 5–25 words that appear verbatim in the posting, and no "missing" skill that the resume
actually contains. Greedy decoding, same prompts for all systems.

**All 360 held-out pairs** (120 unseen resumes × source / near / far job):

| System | Strict pass | Quote precision | Valid JSON | Matched-skills F1 | Missing-skills F1 |
|---|---|---|---|---|---|
| Qwen3-0.6B (base) | 19.7% | 75.6% | 95.8% | 0.38 | 0.33 |
| Qwen3-1.7B (base) | 7.2% | 36.2% | 77.8% | 0.48 | 0.43 |
| **0.6B student** | **50.3%** | 93.8% | 98.9% | 0.58 | 0.44 |
| **1.7B student** | **53.1%** | 94.9% | 99.2% | 0.62 | 0.46 |
| 4B student | *training* | | | | |

**60-pair subset with a large-model reference** (20 per kind, fixed seed; Groq's free tier allows ~200k
tokens/day, so the 120B model is scored on a stratified subset and all systems are compared on the same pairs):

| System | Params | Strict pass | Quote precision | Missing-skills F1 |
|---|---|---|---|---|
| Qwen3-0.6B (base) | 0.6B | 28.3% | 84.0% | 0.40 |
| 0.6B student | 0.6B | 48.3% | 93.5% | **0.51** |
| 1.7B student | 1.7B | 56.7% | **95.3%** | 0.50 |
| gpt-oss-120b (Groq, production explainer) | 120B | **61.7%** | 94.2% | 0.38 |

Findings:
- **Distillation, not size, buys rule-following.** The base 1.7B is *worse* than the base 0.6B: it paraphrases
  quotes (36% verbatim) and writes `// comments` inside its JSON. After fine-tuning both follow the format.
- **0.6B is enough.** 1.7B − 0.6B student = +2.8 pts strict pass, 95% CI −3.6 to +8.9 (paired bootstrap): not
  distinguishable on 360 pairs, so the 3× smaller model is the deployment pick.
- **The remaining errors are quote length** (quotes just outside 5–25 words: 88/360 for 0.6B) — a data issue, not
  a capacity one: the 14B teacher paraphrased often, and only 26% of its outputs passed strictly (see below).

## Pipeline

| Step | Where | What |
|---|---|---|
| 1. Resumes | Kaggle 2×T4 | Qwen3-14B (Q4_K_M GGUF, llama.cpp CUDA) writes 1,115 synthetic resumes from real postings |
| 2. Pairs | CPU | each resume × its source job, a BM25 near neighbour and a far job (3,345 pairs; 120 resumes held out = 360 eval pairs), with the exact prompt the app sends |
| 3. Teacher | Kaggle 2×T4 | Qwen3-14B writes JSON explanations, one llama-server per GPU, 8 slots each |
| 4. Filter + repair | CPU | the app's own checks; 880/3,345 pass strictly, a repair step drops unverifiable quotes and wrong "missing" skills and keeps ≥2 verified quotes → 1,944 usable (1,734 train / 210 eval-loss) |
| 5. Students | Kaggle 2×T4 | QLoRA (nf4, r16/α32, all linear layers), 2 epochs, fp16, DDP via torchrun; merged to fp16 |
| 6. Score | CPU | `distill.score`, paired bootstrap CIs; Groq reference via `scripts/groq_baseline.py` |

Engineering notes (each cost a failed run, so each is now guarded by a smoke stage that runs first):
- vLLM 0.33 does not start on T4 (sm_75), so generation uses llama.cpp built with CUDA for sm_75
  (`GGML_CUDA_NO_VMM=ON`: Kaggle has no `libcuda.so` to link); the 34-min build is cached in the data repo.
- Qwen3 ships bf16 weights, T4 has no bf16, and fp16 AMP's grad scaler cannot unscale bf16 grads →
  `fix_dtypes` casts trainable params to fp32 and the rest to fp16 before training.
- Kaggle's torchao 0.10 makes PEFT 0.21 refuse to merge an fp16 LoRA → uninstalled in the notebook.
- `bm25s` imports JAX, which preallocates 75% of every GPU → `JAX_PLATFORMS=cpu`.
- Every student notebook first runs train / generate / merge / generate on a tiny slice with the *same* launcher
  as the full run (torchrun ×2) and reports all failures at once; a CPU dry run (`scripts/dry_run_student.py`)
  checks the non-GPU parts before anything reaches Kaggle.

## Run

```bash
pip install -e ".[pairs,dev]"
python scripts/make_teacher_notebook.py <commit> resumes   # Kaggle
python -m distill.teacher pairs                            # local
python scripts/make_teacher_notebook.py <commit> explain   # Kaggle
python -m distill.teacher filter                           # local
python scripts/make_student_notebook.py <commit> 0.6B      # Kaggle (also 1.7B, 4B)
python -m distill.score data/outputs/*.jsonl --out results/quality.json
python scripts/groq_baseline.py --n-per-kind 20            # large-model reference (GROQ_API_KEY)
pytest -q
```
