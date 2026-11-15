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


class LlamaCppGenerator:
    """One llama-server per GPU (CUDA build), `parallel` slots each, requests spread over all slots.

    vLLM 0.33 does not start on Kaggle's T4s (compute capability 7.5; probe v1/v2), so this is the GPU engine
    there. `model` is "<hf repo>:<gguf file>" (e.g. "Qwen/Qwen3-14B-GGUF:Qwen3-14B-Q4_K_M.gguf") or a local
    .gguf path; prompts are rendered with the original model's tokenizer (`tokenizer`) so they match training.
    """

    def __init__(self, model: str, tokenizer: str, llama_cpp: str = "llama.cpp", n_gpus: int | None = None,
                 parallel: int = 8, ctx: int = 3072, ngl: int = 99):
        import atexit
        import os

        from transformers import AutoTokenizer

        from .llamacpp import Server

        if not os.path.exists(model):  # "<repo>:<file>" on the Hub
            from huggingface_hub import hf_hub_download

            repo, fname = model.split(":", 1)
            model = hf_hub_download(repo, fname)
        if n_gpus is None:
            import torch

            n_gpus = max(1, torch.cuda.device_count())
        self.tok = AutoTokenizer.from_pretrained(tokenizer)
        self.servers = [Server(model, llama_cpp, ctx=ctx, ngl=ngl, parallel=parallel, gpu=g if ngl else None,
                               log=f"llama-server-{g}.log").__enter__() for g in range(n_gpus)]
        self.slots = parallel * n_gpus
        atexit.register(self.close)  # never leave servers holding the GPUs

    def close(self):
        for s in self.servers:
            if s.proc.poll() is None:
                s.__exit__(None, None, None)

    def generate(self, conversations, temperature, max_tokens, seed=0):
        import time
        from concurrent.futures import ThreadPoolExecutor

        from .fmt import prompt_text

        prompts = [prompt_text(self.tok, c) for c in conversations]
        out, done, t0 = [""] * len(prompts), [0], time.time()

        def one(i):
            r = self.servers[i % len(self.servers)].complete(prompts[i], max_tokens, temperature, seed + i)
            out[i] = r["text"]
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"generated {done[0]}/{len(prompts)} {(time.time() - t0) / 60:.1f} min", flush=True)

        with ThreadPoolExecutor(self.slots) as pool:
            list(pool.map(one, range(len(prompts))))
        return out


def make_generator(engine: str, model: str, tp: int = 1, batch_size: int = 16, tokenizer: str | None = None,
                   llama_cpp: str = "llama.cpp"):
    if engine == "hf":
        return HFGenerator(model, batch_size=batch_size)
    if engine == "llamacpp":
        return LlamaCppGenerator(model, tokenizer or model.split(":", maxsplit=1)[0].removesuffix("-GGUF"), llama_cpp,
                                 n_gpus=tp)
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
