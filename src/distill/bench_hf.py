"""Naive baseline: Hugging Face Transformers `generate`, one request at a time, fp16 on one GPU.

    python -m distill.bench_hf --model out/merged --system student-1.7b-hf --n 30 --out data/outputs/hf.jsonl

The "what you get without a serving engine" row of the benchmark. `--load-4bit` loads bitsandbytes NF4 weights
(the QLoRA training format), the one 4-bit GPU format that needs no serving engine.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .fmt import prompt_text

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--pairs", default="data/pairs.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs = [p for p in map(json.loads, Path(a.pairs).read_text().splitlines()) if p["split"] == "eval"][: a.n]
    tok = AutoTokenizer.from_pretrained(a.model)
    t = time.time()
    quant = None
    if a.load_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map={"": 0},
                                                 quantization_config=quant).eval()
    load_s = time.time() - t
    rows, lat, gen = [], [], []
    for p in pairs:
        enc = tok(prompt_text(tok, p["messages"]), return_tensors="pt").to(model.device)
        t = time.time()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=700, do_sample=False, pad_token_id=tok.pad_token_id)
        lat.append(time.time() - t)
        new = out[0, enc["input_ids"].shape[1]:]
        gen.append(len(new))
        rows.append({"system": a.system, "resume_job_id": p["resume_job_id"], "job_id": p["job_id"],
                     "text": tok.decode(new, skip_special_tokens=True)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    summary = {"system": a.system, "engine": "transformers", "format": "bnb-nf4" if a.load_4bit else "fp16",
               "size_gb": round(model.get_memory_footprint() / 1e9, 3), "n": len(rows), "load_s": round(load_s, 1),
               "latency_p50_s": float(np.percentile(lat, 50)), "latency_p95_s": float(np.percentile(lat, 95)),
               "gen_tok_s_mean": float(np.sum(gen) / np.sum(lat)), "gen_tokens_mean": float(np.mean(gen))}
    Path(a.out).with_suffix(".bench.json").write_text(json.dumps(summary, indent=1))
    print("BENCH", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
