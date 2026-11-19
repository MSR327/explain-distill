---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
language: en
tags: [distillation, qlora, gguf, llama.cpp, json, recruiting]
pipeline_tag: text-generation
---

# qwen3-0.6b-explain

Qwen3-0.6B distilled to write **grounded resume ↔ job-fit explanations** as JSON: a fit summary, matched and
missing skills, and 2–4 evidence quotes copied verbatim from the job posting. Teacher: Qwen3-14B, filtered with
the checks of [resume-job-matcher](https://github.com/MSR327/resume-job-matcher). Code, data pipeline and
evaluation: [explain-distill](https://github.com/MSR327/explain-distill).

## Results (360 held-out resume–job pairs)

| | Base Qwen3-0.6B | This model |
|---|---|---|
| Strict pass (valid JSON, verbatim 5–25-word quotes, correct missing skills) | 19.7% | **50.3%** |
| Quote precision | 75.6% | **93.8%** |
| Valid JSON | 95.8% | **98.9%** |

Paired difference +30.6 pts (95% CI 24.2–36.9). The Q4_K_M GGUF scores within 0.6 pts of f16.

## Files

- `gguf/qwen3-0.6b-explain-Q4_K_M.gguf` (0.40 GB, recommended), `gguf/qwen3-0.6b-explain-Q8_0.gguf`
- `merged/`: fp16 Transformers weights; `adapter/`: the LoRA adapter (r16/α32, all linear layers)

## Run (CPU, llama.cpp)

```bash
hf download srish327/qwen3-0.6b-explain gguf/qwen3-0.6b-explain-Q4_K_M.gguf --local-dir .
llama-server -m gguf/qwen3-0.6b-explain-Q4_K_M.gguf --jinja -c 4096 --port 8080
```

Send the same chat messages resume-job-matcher builds (`jobmatch.explain.build_messages`) to
`/v1/chat/completions` with `"chat_template_kwargs": {"enable_thinking": false}`: the model was trained on that
rendering. Speed: ~1.7 s per explanation on a T4, ~25 s on 4 CPU cores.

## Limits

Trained on synthetic resumes (written by the teacher from real postings) and English tech job postings. About
half of the outputs still fail the strict checks, mostly quotes just outside 5–25 words; validate quotes against
the posting before showing them, as resume-job-matcher does.
