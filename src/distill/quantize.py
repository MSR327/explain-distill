"""Post-training quantization of a merged student with llm-compressor (compressed-tensors format, served by vLLM).

    python -m distill.quantize --merged out/merged --scheme W4A16 --out out/w4a16   # GPTQ int4 weights
    python -m distill.quantize --merged out/merged --scheme W8A8 --out out/w8a8     # SmoothQuant + GPTQ int8

Calibration uses the student's own task prompts (train split, with the teacher's answers), which is what the
quantized model will see in production; lm_head stays in full precision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fmt import to_prompt_completion


def calibration(tok, path: str, n: int):
    from datasets import Dataset

    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()][:n]
    texts = []
    for r in rows:
        pc = to_prompt_completion(tok, r)
        texts.append(pc["prompt"] + pc["completion"])
    return Dataset.from_dict({"text": texts})


def main() -> None:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--scheme", choices=["W4A16", "W8A8"], required=True)
    ap.add_argument("--calib", default="data/sft_train.jsonl")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.merged)
    model = AutoModelForCausalLM.from_pretrained(a.merged, dtype="auto")
    recipe = [GPTQModifier(targets="Linear", scheme=a.scheme, ignore=["lm_head"])]
    if a.scheme == "W8A8":
        from llmcompressor.modifiers.transform import SmoothQuantModifier

        recipe.insert(0, SmoothQuantModifier(smoothing_strength=0.8))
    ds = calibration(tok, a.calib, a.n)
    oneshot(model=model, tokenizer=tok, dataset=ds, recipe=recipe, max_seq_length=2048,
            num_calibration_samples=len(ds), text_column="text")
    model.save_pretrained(a.out, save_compressed=True)
    tok.save_pretrained(a.out)
    size = sum(f.stat().st_size for f in Path(a.out).glob("*.safetensors")) / 1e9
    print("QUANTIZED", json.dumps({"scheme": a.scheme, "out": a.out, "size_gb": round(size, 3),
                                   "n_calibration": len(ds)}), flush=True)


if __name__ == "__main__":
    main()
