"""vLLM serving benchmark for one model/format on the GPU it runs on.

    python -m distill.bench_vllm --model out/w4a16 --system student-1.7b-w4a16 \
        --out data/outputs/student-1.7b-w4a16.jsonl

Two numbers per format, because they answer different questions:
- throughput: all eval prompts submitted at once (continuous batching), output tokens/s and requests/s:
  "how many explanations per GPU-hour", which sets $/1k requests;
- latency: the same prompts one at a time (a single user), p50/p95 seconds per explanation.
The throughput run's outputs are saved and scored for quality like every other system.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# GCP on-demand price of one T4, GPU only (cloud.google.com/products/compute/gpus-pricing, checked 2026-11-15).
# The VM is extra, so $/1k requests computed from it is a lower bound.
T4_USD_PER_GPU_HOUR = 0.35


def main() -> None:
    from vllm import LLM, SamplingParams

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--pairs", default="data/pairs.jsonl")
    ap.add_argument("--quantization", default=None, help="e.g. bitsandbytes; compressed-tensors is auto-detected")
    ap.add_argument("--n-latency", type=int, default=30)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from .fmt import prompt_text

    pairs = [p for p in map(json.loads, Path(a.pairs).read_text().splitlines()) if p["split"] == "eval"]
    t = time.time()
    llm = LLM(model=a.model, dtype="half", max_model_len=a.max_model_len, gpu_memory_utilization=0.90,
              quantization=a.quantization)
    load_s = time.time() - t
    tok = llm.get_tokenizer()
    prompts = [prompt_text(tok, p["messages"]) for p in pairs]
    sp = SamplingParams(temperature=0.0, max_tokens=700)

    llm.generate(prompts[:2], sp)  # warm-up (CUDA graphs, caches)
    t = time.time()
    outs = llm.generate(prompts, sp)
    thr_s = time.time() - t
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)

    lat = []
    for pr in prompts[: a.n_latency]:
        t = time.time()
        llm.generate([pr], sp, use_tqdm=False)
        lat.append(time.time() - t)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps({"system": a.system, "resume_job_id": p["resume_job_id"],
                                               "job_id": p["job_id"], "text": o.outputs[0].text},
                                              ensure_ascii=False) + "\n" for p, o in zip(pairs, outs, strict=True)))
    req_per_s = len(outs) / thr_s
    size = sum(f.stat().st_size for f in Path(a.model).glob("*.safetensors")) / 1e9 if Path(a.model).exists() else None
    summary = {"system": a.system, "engine": "vllm", "model": a.model, "quantization": a.quantization,
               "size_gb": round(size, 3) if size else None, "load_s": round(load_s, 1), "n": len(outs),
               "throughput_gen_tok_s": gen_tokens / thr_s, "throughput_req_s": req_per_s,
               "usd_per_1k_requests": T4_USD_PER_GPU_HOUR / 3600 / req_per_s * 1000,
               "latency_p50_s": float(np.percentile(lat, 50)), "latency_p95_s": float(np.percentile(lat, 95)),
               "gen_tokens_mean": gen_tokens / len(outs)}
    Path(a.out).with_suffix(".bench.json").write_text(json.dumps(summary, indent=1))
    print("BENCH", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
