# Design: distilling grounded job-match explanations, then serving them cheaply

## Problem

[resume-job-matcher](https://github.com/MSR327/resume-job-matcher) (project 2) explains each match with
gpt-oss-120b on Groq's free tier: 8k tokens/min and 200k tokens/day. One request needs ~4k tokens, so the app
explains only the top 3 and falls back to plain skill overlap when Groq is busy. Goal: a small open model,
fine-tuned for exactly this task, that keeps most of the quality and can be served for free with no rate limit.

## Questions the project answers

1. How much of a strong model's **grounding** (verbatim citations, correct missing skills) survives distillation
   into 0.6B / 1.7B / 4B students?
2. Where does quality stop improving with size?
3. Which **format x engine** (fp16, bitsandbytes NF4, GGUF f16/Q8_0/Q4_K_M x Transformers, llama.cpp CUDA,
   llama.cpp CPU) gives the best quality-per-latency on a free T4 and on CPU, and what does a request cost?

## Pipeline

| Stage | Where | Input -> output |
|---|---|---|
| 0. Environment probe | Kaggle T4 x2 | which engine starts on a T4 (vLLM did not; Transformers did), tokens/s |
| 1. Resumes | Kaggle (llama.cpp CUDA) | 1,120 source jobs -> synthetic resumes (project 2's prompt and 4-gram overlap rule) |
| 2. Pairs | local CPU | each resume -> source / near (app rank 1-5) / far job + the app's skill hints and exact messages |
| 3. Teacher explanations | Kaggle (llama.cpp CUDA) | pairs -> Qwen3-14B (Q4_K_M GGUF) JSON explanations |
| 4. Filter | local | rejection sampling with project 2's checks -> SFT train / eval |
| 5. SFT | Kaggle | QLoRA on Qwen3-0.6B / 1.7B / 4B |
| 6. Quality eval | Kaggle + local | held-out pairs: citation pass rate, missing-skill precision, valid JSON, judge grades |
| 7. Serving benchmark | Kaggle T4 + CPU | format x engine grid: quality kept, tokens/s, p50/p95, memory, $/1k requests |
| 8. Ship | HF Hub / Space | adapters + quantized weights; project 2 calls the student first, Groq as fallback |

## Decisions

- **Students: Qwen3 (0.6B, 1.7B, 4B).** The newer Qwen3.5 small models are vision-language models with hybrid
  linear attention (`layer_types: full_attention + linear_attention`), which AWQ/GPTQ/GGUF tools and serving
  engines support unevenly; that would confound the benchmark. Qwen3 is a standard dense transformer, Apache-2.0,
  with official AWQ/GPTQ/GGUF builds.
- **Teacher: Qwen3-14B, Q4_K_M GGUF on llama.cpp CUDA**, not Groq: ~3k explanations at ~4k tokens each is ~12M tokens, 60 days of
  Groq's free daily cap. The teacher is from the students' family (common for distillation); the judge stays a
  different family (Granite), as in project 2.
- **No vLLM on the T4s.** The plan was vLLM (AWQ teacher, GPTQ/SmoothQuant students). Probe v1/v2 on Kaggle
  (driver 580, `pip install vllm` -> 0.33.0 + torch 2.14) failed at engine start for every configuration tried,
  and a pinned older vLLM could not be installed in a side venv. Transformers works but is too slow for ~4k
  teacher generations, so the GPU engine is llama.cpp built for sm_75: one `llama-server` per T4 with 8 parallel
  slots, prompts rendered with the original Qwen3 chat template and sent to the raw `/completion` endpoint. The
  build needs `GGML_CUDA_NO_VMM=ON` because the image has no unversioned `libcuda.so` (CMake target
  `CUDA::cuda_driver`). `distill.quantize` still produces GPTQ W4A16 / SmoothQuant W8A8 checkpoints, but they are
  vLLM formats with no engine on this hardware, so the benchmark covers GGUF and bitsandbytes instead.
- **Same prompt as the app.** Pairs carry `jobmatch.explain.build_messages(...)` output verbatim, including the
  skill hints the pipeline computes, so a student is a drop-in replacement.
- **Rejection sampling.** A teacher output is kept only if it is valid JSON with the expected types, has 2-4
  evidence items of 5-25 words, every quote is found in the posting (project 2's normalized check), and no
  "missing" skill appears in the resume. Retention per reason is reported: it measures the teacher.
- **No leakage.** Jobs are split 85/15 by a stable hash of the job id; train pairs use only train jobs, eval
  pairs only eval jobs. Project 2's evaluation jobs are never used as sources.
- **Fail fast.** Every Kaggle notebook checks HF write access first and runs a smoke pass on a few items before
  the full run (lessons from project 2's 32-minute failed build).

## Risks

- Small students may not copy quotes exactly. That is measurable, and either outcome is a finding.
- Teacher quality bounds student quality; the retention rate and project 2's L3 judge calibration
  (kappa 0.56, lenient at the top grade) say how far to trust judge-based numbers.
- Kaggle's image changes under you: stage 0 found vLLM broken on T4, and the first llama.cpp CUDA build failed
  at CMake. Builds are pinned to a llama.cpp commit and every build step fails loudly.
