"""One prompt format for training, evaluation and serving (train/serve parity).

Qwen3's chat template renders `prompt + answer + "<|im_end|>\\n"` for a finished turn, where `prompt` is the
generation prompt with thinking disabled (an empty <think></think> block). Training on exactly
(prompt_text, answer + EOS) means the fine-tuned model sees the same tokens vLLM / llama.cpp send it later.
"""

from __future__ import annotations

EOS = "<|im_end|>"


def prompt_text(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def to_prompt_completion(tok, row: dict) -> dict:
    """SFT row {"messages": [system, user, assistant]} -> {"prompt": str, "completion": str} for TRL."""
    *prompt, answer = row["messages"]
    assert answer["role"] == "assistant"
    return {"prompt": prompt_text(tok, prompt), "completion": answer["content"] + EOS}
