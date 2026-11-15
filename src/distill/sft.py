"""QLoRA SFT of a Qwen3 student on filtered teacher explanations, then merge for serving.

    torchrun --nproc_per_node 2 -m distill.sft --model Qwen/Qwen3-1.7B --out out/qwen3-1.7b   # Kaggle T4 x2
    python -m distill.sft --model Qwen/Qwen3-0.6B --out /tmp/s --no-4bit --max-steps 2 --limit 8 # local smoke

Loss is on the completion only (TRL prompt-completion format). T4s have no bf16, so compute is fp16. Examples
longer than --max-length are dropped, never truncated: truncation would cut the JSON answer the model learns.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .fmt import to_prompt_completion

LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def load_rows(path: str, limit: int | None = None) -> list[dict]:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    return rows[:limit] if limit else rows


def build_dataset(tok, rows: list[dict], max_length: int):
    from datasets import Dataset

    pc = [to_prompt_completion(tok, r) for r in rows]
    lens = [len(tok(p["prompt"] + p["completion"])["input_ids"]) for p in pc]
    kept = [p for p, n in zip(pc, lens, strict=True) if n <= max_length]
    return Dataset.from_list(kept), {"n_rows": len(rows), "n_kept": len(kept), "max_tokens": max(lens, default=0),
                                     "mean_tokens": sum(lens) / max(1, len(lens))}


def main() -> None:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", default="data/sft_train.jsonl")
    ap.add_argument("--eval", default="data/sft_eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-4bit", action="store_true")
    a = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    tok = AutoTokenizer.from_pretrained(a.model)
    train, tstats = build_dataset(tok, load_rows(a.train, a.limit), a.max_length)
    ev, _ = build_dataset(tok, load_rows(a.eval, 64 if a.limit is None else a.limit), a.max_length)
    if local_rank == 0:
        print("DATA", json.dumps(tstats), flush=True)

    gpu = torch.cuda.is_available()
    quant = None if a.no_4bit or not gpu else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if gpu else torch.float32, quantization_config=quant,
        device_map={"": local_rank} if gpu else None)
    args = SFTConfig(
        output_dir=a.out, num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_steps=10, per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum, per_device_eval_batch_size=a.batch, eval_strategy="epoch",
        max_length=a.max_length, fp16=gpu, bf16=False, gradient_checkpointing=True, logging_steps=10,
        save_strategy="no", report_to="none", optim="paged_adamw_8bit" if quant else "adamw_torch",
        ddp_find_unused_parameters=False if gpu else None, use_cpu=not gpu, seed=0)  # no silent MPS on a Mac
    trainer = SFTTrainer(model=model, args=args, train_dataset=train, eval_dataset=ev, processing_class=tok,
                         peft_config=LoraConfig(task_type="CAUSAL_LM", **LORA))
    result = trainer.train()
    if local_rank == 0:
        trainer.model.save_pretrained(f"{a.out}/adapter")
        tok.save_pretrained(f"{a.out}/adapter")
        log = [x for x in trainer.state.log_history if "loss" in x or "eval_loss" in x]
        Path(f"{a.out}/train_log.json").write_text(json.dumps(
            {"model": a.model, "data": tstats, "train_runtime_s": result.metrics.get("train_runtime"),
             "train_loss": result.metrics.get("train_loss"), "log": log, "lora": LORA,
             "args": {k: v for k, v in vars(a).items()}}, indent=1))
        print("TRAINED", json.dumps({"loss": result.metrics.get("train_loss"),
                                     "runtime_s": result.metrics.get("train_runtime"),
                                     "eval_loss": next((x["eval_loss"] for x in reversed(log) if "eval_loss" in x),
                                                       None)}), flush=True)


def merge(base: str, adapter: str, out: str) -> None:
    """Merge LoRA into fp16 base weights: what vLLM / quantizers / llama.cpp consume."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
    PeftModel.from_pretrained(model, adapter).merge_and_unload().save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


if __name__ == "__main__":
    main()
