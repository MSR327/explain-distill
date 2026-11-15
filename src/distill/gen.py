"""Batch text generation behind one interface: vLLM on a GPU, or a fake for tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class Generator(Protocol):
    def generate(self, conversations: list[list[dict]], temperature: float, max_tokens: int,
                 seed: int = 0) -> list[str]: ...


class FakeGenerator:
    def __init__(self, reply: Callable[[list[dict]], str]):
        self.reply, self.calls = reply, 0

    def generate(self, conversations, temperature, max_tokens, seed=0):
        self.calls += 1
        return [self.reply(c) for c in conversations]


class HFGenerator:
    """Fallback without vLLM: Transformers `generate` in length-sorted batches (less padding), fp16, layers split
    over the visible GPUs. Slower than vLLM's continuous batching, but it ran reliably on Kaggle T4s in project 2.
    """

    def __init__(self, model: str, batch_size: int = 16, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        gpu = torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float16 if gpu else torch.float32, device_map="auto" if gpu else None).eval()
        self.bs = batch_size

    def generate(self, conversations, temperature, max_tokens, seed=0):
        import time

        self.torch.manual_seed(seed)
        prompts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                   for c in conversations]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out, t0 = [""] * len(prompts), time.time()
        for b in range(0, len(order), self.bs):
            idx = order[b:b + self.bs]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.model.device)
            kw = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=max_tokens, pad_token_id=self.tok.pad_token_id, **kw)
            for i, seq in zip(idx, gen[:, enc["input_ids"].shape[1]:], strict=True):
                out[i] = self.tok.decode(seq, skip_special_tokens=True)
            if (b // self.bs) % 10 == 0:
                print(f"generated {min(b + self.bs, len(order))}/{len(order)} {(time.time() - t0) / 60:.1f} min",
                      flush=True)
        return out


def make_generator(engine: str, model: str, tp: int = 1, batch_size: int = 16):
    if engine == "hf":
        return HFGenerator(model, batch_size=batch_size)
    return VLLMGenerator(model, tp)


class VLLMGenerator:
    """Offline vLLM engine. T4s have no bf16, so dtype is fp16; Qwen3's thinking mode is turned off."""

    def __init__(self, model: str, tp: int = 1, max_model_len: int = 6144, gpu_memory_utilization: float = 0.90):
        from vllm import LLM

        self.llm = LLM(model=model, dtype="half", tensor_parallel_size=tp, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization)
        self.tok = self.llm.get_tokenizer()
        self.max_prompt = max_model_len

    def generate(self, conversations, temperature, max_tokens, seed=0):
        from vllm import SamplingParams

        prompts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                   for c in conversations]
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed)
        return [o.outputs[0].text for o in self.llm.generate(prompts, params)]
