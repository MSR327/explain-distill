# explain-distill

Distill the grounded job-match explanations of [resume-job-matcher](https://github.com/MSR327/resume-job-matcher)
into small open LLMs (Qwen3 0.6B / 1.7B / 4B), then benchmark how to serve them on free hardware.

**Headline (360 held-out resume–job pairs):** QLoRA distillation lifts Qwen3-0.6B from **19.7% → 50.3%** strict
pass (paired +30.6 pts, 95% CI 24.2–36.9) and quote precision from 75.6% → 93.8%. On a 60-pair subset the 4B
student **matches gpt-oss-120b** (65.0% vs 61.7% strict pass; paired diff +3.3 pts, CI −13 to +20) at 1/30 of the
total parameters (gpt-oss is a MoE with ~5B active), and every student beats it on missing-skills F1. Everything ran on free tiers (Kaggle 2×T4, Groq free API).

**Status:** three students trained and scored, serving benchmark done; CPU Space deployment next.
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
| Qwen3-4B (base) | 19.2% | 51.0% | 100% | 0.58 | 0.47 |
| **0.6B student** | **50.3%** | 93.8% | 98.9% | 0.58 | 0.44 |
| **1.7B student** | **53.1%** | 94.9% | 99.2% | 0.62 | 0.46 |
| **4B student** | **55.8%** | 95.1% | 99.4% | 0.70 | 0.51 |

**60-pair subset with a large-model reference** (20 per kind, fixed seed; Groq's free tier allows ~200k
tokens/day, so the 120B model is scored on a stratified subset and all systems are compared on the same pairs):

| System | Params | Strict pass | Quote precision | Missing-skills F1 |
|---|---|---|---|---|
| Qwen3-0.6B (base) | 0.6B | 28.3% | 84.0% | 0.40 |
| 0.6B student | 0.6B | 48.3% | 93.5% | **0.51** |
| 1.7B student | 1.7B | 56.7% | 95.3% | 0.50 |
| 4B student | 4B | **65.0%** | **97.0%** | **0.59** |
| gpt-oss-120b (Groq, production explainer) | 120B | 61.7% | 94.2% | 0.38 |

Findings:
- **Distillation, not size, buys rule-following.** Base models do not get better with size (19.7% / 7.2% /
  19.2% strict pass for 0.6B / 1.7B / 4B): the 1.7B paraphrases quotes (36% verbatim) and writes `// comments`
  inside its JSON. After fine-tuning all three follow the format (50–56%).
- **Returns flatten with size.** Strict pass 50.3% → 53.1% → 55.8%. Paired differences: 1.7B − 0.6B = +2.8 pts
  (CI −3.6 to +8.9), 4B − 0.6B = +5.6 pts (CI −0.8 to +11.7). The 4B is better on skill F1 (0.70 vs 0.58), but
  the 0.6B is ~7× smaller and the CPU-deployment pick; the 4B is the pick when a GPU is available.
- **A 4B student matches the 120B production model** on the 60-pair subset; the small subset makes the interval
  wide, so the honest claim is "no detectable gap", not "better".
- **The remaining errors are quote length** (quotes just outside 5–25 words: 88/360 for 0.6B, 93/360 for 4B) — a data issue, not
  a capacity one: the 14B teacher paraphrased often, and only 26% of its outputs passed strictly (see below).

## Serving: formats × engines

One Kaggle T4 per GPU row; CPU rows hide the GPUs and use 4 threads. Latency = one request at a time
(~1.5k-token prompt, ~300 generated tokens). Throughput = every eval prompt over 8 llama.cpp slots; cost at
$0.35 per T4-hour. Notebooks: `scripts/make_bench_notebook.py`; raw rows in the private data repo `bench/`.

| Model | Engine / format | Size | p50 latency | Gen tok/s (1 stream) | 8-slot throughput | $ / 1k requests |
|---|---|---|---|---|---|---|
| 0.6B | Transformers fp16 (T4) | 1.19 GB | 13.0 s | 23 | — | — |
| 0.6B | Transformers NF4 (T4) | 0.53 GB | 19.2 s | 18 | — | — |
| 0.6B | llama.cpp f16 (T4) | 1.20 GB | 3.2 s | 101 | 1.10 req/s | $0.09 |
| 0.6B | llama.cpp Q8_0 (T4) | 0.64 GB | 2.2 s | 159 | 1.05 req/s | $0.09 |
| 0.6B | **llama.cpp Q4_K_M (T4)** | **0.40 GB** | **1.7 s** | **190** | 1.10 req/s | $0.09 |
| 0.6B | llama.cpp Q8_0 (CPU ×4) | 0.64 GB | 32.1 s | 14 | — | — |
| 0.6B | **llama.cpp Q4_K_M (CPU ×4)** | 0.40 GB | **24.8 s** | 17 | — | free |
| 4B | Transformers fp16 (T4) | 8.05 GB | 23.5 s | 14 | — | — |
| 4B | Transformers NF4 (T4) | 2.60 GB | 23.7 s | 14 | — | — |
| 4B | llama.cpp f16 (T4) | 8.05 GB | 13.8 s | 26 | 0.33 req/s | $0.29 |
| 4B | llama.cpp Q4_K_M (T4) | 2.50 GB | 7.0 s | 53 | 0.32 req/s | $0.31 |
| 4B | llama.cpp Q4_K_M (CPU ×4) | 2.50 GB | 103.6 s | 5 | — | — |

**Does quantization cost quality?** Same eval prompts through each GGUF (llama.cpp, greedy):

| Model | f16 | Q8_0 | Q4_K_M | Q4_K_M − f16 (paired 95% CI) |
|---|---|---|---|---|
| 0.6B (360 pairs) | 48.9% | 48.9% | 48.3% | −0.6 pts (−6.4 to +5.3) |
| 4B (120 pairs) | 56.7% | 55.0% | 53.3% | −3.3 pts (−11.7 to +4.2) |

Findings:
- **Q4_K_M is free quality-wise for the 0.6B** (−0.6 pts, quote precision 93.2% → 90.6%) at a third of the size,
  so it is the deployment format. For the 4B the point estimate drops 3 pts; Q8_0 is the safer 4B choice.
- **A serving engine matters more than the format:** the same fp16 weights generate 4.3× faster in llama.cpp than
  in Transformers `generate` (101 vs 23 tok/s), and Q4_K_M adds another 1.9×. bitsandbytes NF4 saves memory but is
  *slower* than fp16 on a T4 (dequantization on every matmul).
- **8 slots only ~2× a single stream** and throughput is flat across formats (1.1 req/s for all three 0.6B
  GGUFs), so the batch bottleneck is not the GPU matmuls (likely per-token CPU work on Kaggle's 4 vCPUs; not
  investigated). Even so, a T4 serves 1k explanations for ~$0.09.
- **CPU-only is usable but slow:** ~25 s per explanation for the 0.6B Q4_K_M on 4 cores (prompt processing at
  ~150 tok/s is ~10 s of it). A free 2-vCPU Hugging Face Space will be slower still, so in the app the student is
  the *unlimited fallback* behind Groq (~2 s), not the primary. The 4B is not viable on CPU (~100 s).

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
