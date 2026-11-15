"""Generate explanations for the held-out eval pairs with any model (base, merged student, teacher).

    python -m distill.generate --model out/qwen3-1.7b/merged --system student-1.7b --out data/outputs/student-1.7b.jsonl

Default engine is Transformers (`hf`): vLLM 0.33 does not start on Kaggle's T4s. Greedy decoding
(temperature 0) so systems are compared on their best guess, not on sampling luck.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    from .gen import make_generator

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--pairs", default="data/pairs.jsonl")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--engine", choices=["vllm", "hf", "llamacpp"], default="hf")
    ap.add_argument("--tokenizer", default=None, help="llamacpp engine: HF repo with the chat template")
    ap.add_argument("--llama-cpp", default="llama.cpp")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs = [p for p in map(json.loads, Path(a.pairs).read_text().splitlines()) if p["split"] == a.split][: a.limit]
    t = time.time()
    gen = make_generator(a.engine, a.model, a.tp, batch_size=a.batch_size, tokenizer=a.tokenizer,
                         llama_cpp=a.llama_cpp)
    load_s = time.time() - t
    t = time.time()
    texts = gen.generate([p["messages"] for p in pairs], temperature=0.0, max_tokens=700)
    gen_s = time.time() - t
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps({"system": a.system, "resume_job_id": p["resume_job_id"],
                                               "job_id": p["job_id"], "text": x}, ensure_ascii=False) + "\n"
                                   for p, x in zip(pairs, texts, strict=True)))
    print("GENERATED", json.dumps({"system": a.system, "n": len(texts), "load_s": round(load_s, 1),
                                   "gen_s": round(gen_s, 1)}), flush=True)


if __name__ == "__main__":
    main()
