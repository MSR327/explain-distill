"""Generate notebooks/00_env_probe.ipynb: a 15-minute Kaggle check before any real GPU run.

    python scripts/make_probe_notebook.py

Answers, before 10+ GPU-hours are spent: does the current vLLM install and run on Kaggle's T4s (driver/CUDA,
no bf16), does the AWQ teacher load with tensor parallelism, and how many tokens/s does it generate on
prompts shaped like the real ones (~1.5k tokens in, ~300 out)? That decides the teacher-data budget.
"""

import json

PROBE = '''import json, os, sys, time
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import torch
import vllm
from vllm import LLM, SamplingParams

model, tp = sys.argv[1], int(sys.argv[2])
info = {"torch": torch.__version__, "cuda": torch.version.cuda, "vllm": vllm.__version__,
        "gpus": torch.cuda.device_count(), "capability": torch.cuda.get_device_capability(0)}
print(info, flush=True)
t = time.time()
llm = LLM(model=model, dtype="half", tensor_parallel_size=tp, max_model_len=4096, gpu_memory_utilization=0.90)
load_s = time.time() - t
tok = llm.get_tokenizer()
filler = "The candidate built data pipelines in Python and SQL and deployed models with Docker. " * 300
msgs = [[{"role": "user", "content": filler[:6000] + f"\\nItem {i}: answer with one JSON object with keys a, b, c."}]
        for i in range(64)]
prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
t = time.time()
outs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=300))
dt = time.time() - t
gen = sum(len(o.outputs[0].token_ids) for o in outs)
res = {**info, "model": model, "tp": tp, "load_s": round(load_s, 1), "n": len(outs),
       "prompt_tokens": sum(len(o.prompt_token_ids) for o in outs), "gen_tokens": gen, "gen_s": round(dt, 1),
       "gen_tok_per_s": round(gen / dt, 1), "sample": outs[0].outputs[0].text[:160]}
print("PROBE_RESULT", json.dumps(res), flush=True)
with open("probe.jsonl", "a") as f:
    f.write(json.dumps(res) + "\\n")
'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def build() -> dict:
    cells = [
        md("# Environment probe: vLLM on Kaggle T4s\n\nNeeds GPU T4 x2 and Internet. No secrets. "
           "Prints `PROBE_RESULT {...}` lines: versions, load time, generation tokens/s."),
        code("!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv\n"
             "import torch; print('preinstalled torch', torch.__version__, 'cuda', torch.version.cuda)"),
        code('%pip install -q "vllm"\n!pip show vllm torch 2>/dev/null | grep -E "^(Name|Version)"'),
        code("%%writefile probe.py\n" + PROBE),
        code("# Small model first (fast failure if the stack is broken), then the real teacher on both GPUs.\n"
             "!python probe.py Qwen/Qwen3-0.6B 1 2>&1 | grep -E 'PROBE_RESULT|Error|error|Traceback' | tail -20\n"
             "!python probe.py Qwen/Qwen3-14B-AWQ 2 2>&1 | grep -E 'PROBE_RESULT|Error|error|Traceback' | tail -20"),
        code("print(open('probe.jsonl').read() if __import__('os').path.exists('probe.jsonl') else 'no results')"),
    ]
    kernel = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return {"cells": cells, "metadata": {"kernelspec": kernel,
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    with open("notebooks/00_env_probe.ipynb", "w") as f:
        json.dump(build(), f, indent=1)
    print("written notebooks/00_env_probe.ipynb")
