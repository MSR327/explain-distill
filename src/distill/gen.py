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
