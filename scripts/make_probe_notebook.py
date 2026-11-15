"""Generate notebooks/00_env_probe.ipynb: a short Kaggle check before any real GPU run.

    python scripts/make_probe_notebook.py

v1 (Kaggle T4 x2, driver 580.178.04, preinstalled torch 2.11.0+cu130) showed `pip install vllm` -> vLLM 0.33.0 +
torch 2.14.0 failing at engine start ("Engine core initialization failed") for both Qwen3-0.6B and 14B-AWQ; a
grep in v1 hid the root cause. v2 is a diagnosis matrix: every config runs in its own process, its FULL log is
kept, and the probe prints the lines that matter (errors, backend/capability messages) plus a PROBE_RESULT
line per config, then falls back to plain Transformers so the teacher run has a working path either way.
"""

import json

PROBE = r'''import json, os, sys, time
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
model, tp, kwargs = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
import torch
import vllm
from vllm import LLM, SamplingParams

info = {"vllm": vllm.__version__, "torch": torch.__version__, "capability": torch.cuda.get_device_capability(0),
        "env_backend": os.environ.get("VLLM_ATTENTION_BACKEND"), "kwargs": kwargs}
print("INFO", json.dumps(info), flush=True)
t = time.time()
llm = LLM(model=model, dtype="half", tensor_parallel_size=tp, max_model_len=4096, gpu_memory_utilization=0.90,
          **kwargs)
load_s = time.time() - t
tok = llm.get_tokenizer()
filler = "The candidate built data pipelines in Python and SQL and deployed models with Docker. " * 300
msgs = [[{"role": "user", "content": filler[:6000] + f"\nItem {i}: answer with one JSON object with keys a, b, c."}]
        for i in range(64)]
prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
t = time.time()
outs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=300))
dt = time.time() - t
gen = sum(len(o.outputs[0].token_ids) for o in outs)
print("PROBE_RESULT", json.dumps({**info, "model": model, "tp": tp, "load_s": round(load_s, 1), "n": len(outs),
      "gen_tokens": gen, "gen_s": round(dt, 1), "gen_tok_per_s": round(gen / dt, 1),
      "sample": outs[0].outputs[0].text[:120]}), flush=True)
'''

HF_PROBE = r'''import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id, bs = sys.argv[1], int(sys.argv[2])
tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
t = time.time()
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto").eval()
load_s = time.time() - t
filler = "The candidate built data pipelines in Python and SQL and deployed models with Docker. " * 300
msgs = [[{"role": "user", "content": filler[:6000] + f"\nItem {i}: answer with JSON."}] for i in range(bs)]
prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
t = time.time()
with torch.inference_mode():
    out = model.generate(**enc, max_new_tokens=300, do_sample=False, pad_token_id=tok.pad_token_id)
dt = time.time() - t
gen = int((out[:, enc["input_ids"].shape[1]:] != tok.pad_token_id).sum())
print("PROBE_RESULT", json.dumps({"engine": "transformers", "torch": torch.__version__, "model": model_id,
      "batch": bs, "load_s": round(load_s, 1), "gen_tokens": gen, "gen_s": round(dt, 1),
      "gen_tok_per_s": round(gen / dt, 1)}), flush=True)
'''

RUNNER = r'''import json, os, re, subprocess, sys
KEEP = re.compile(r"(error|exception|not supported|capability|sm_?\d|flash|bfloat|triton|backend|cuda|"
                  r"PROBE_RESULT|INFO \{)", re.I)

def run(tag, cmd, env=None):
    log = f"logs/{tag}.log"
    os.makedirs("logs", exist_ok=True)
    with open(log, "w") as f:
        p = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})},
                           timeout=1500)
    lines = open(log).read().splitlines()
    ok = any(x.startswith("PROBE_RESULT") for x in lines)
    print(f"===== {tag}: {'OK' if ok else 'FAILED'} (exit {p.returncode}) =====")
    if ok:
        print(*[x for x in lines if x.startswith(("PROBE_RESULT", "INFO"))], sep="\n")
    else:
        keep = [x[:300] for x in lines if KEEP.search(x)]
        print(*keep[-40:], sep="\n")
        print("--- last 15 lines ---", *[x[:300] for x in lines[-15:]], sep="\n")
    return ok
'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def build() -> dict:
    small, teacher = "Qwen/Qwen3-0.6B", "Qwen/Qwen3-14B-AWQ"
    cells = [
        md("# Environment probe v2: why vLLM fails on Kaggle T4s, and what works\n\nNeeds GPU T4 x2 and Internet. "
           "No secrets. Prints one `===== tag: OK/FAILED =====` block per configuration."),
        code("!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv\n"
             "import torch; print('preinstalled torch', torch.__version__, 'cuda', torch.version.cuda)"),
        code("%%writefile probe.py\n" + PROBE),
        code("%%writefile hf_probe.py\n" + HF_PROBE),
        code("%%writefile runner.py\n" + RUNNER),
        code(f"""# A. plain Transformers first (no vLLM installed yet): the guaranteed fallback for the teacher.
from runner import run
import sys
run("hf-0.6b-bs16", f"{{sys.executable}} hf_probe.py {small} 16")
run("hf-qwen3-8b-bs16", f"{{sys.executable}} hf_probe.py Qwen/Qwen3-8B 16")"""),
        code('%pip install -q "vllm"\n!pip show vllm torch 2>/dev/null | grep -E "^(Name|Version)"'),
        code(f"""# B. latest vLLM: default (to see the real error), then without FlashInfer / CUDA graphs.
from runner import run
import json, sys
run("vllm-latest-default", f"{{sys.executable}} probe.py {small} 1")
eager = json.dumps({{"enforce_eager": True}})
ok = False
for backend in ("TRITON_ATTN", "FLEX_ATTENTION"):
    ok = run(f"vllm-latest-{{backend}}", f"{{sys.executable}} probe.py {small} 1 '{{eager}}'",
             env={{"VLLM_ATTENTION_BACKEND": backend}})
    if ok:
        run(f"vllm-latest-{{backend}}-teacher", f"{{sys.executable}} probe.py {teacher} 2 '{{eager}}'",
            env={{"VLLM_ATTENTION_BACKEND": backend}})
        break"""),
        code(f"""# C. an older vLLM (when T4 support was widely used), in its own venv so it can't break the kernel.
!python -m venv /kaggle/working/v010 && /kaggle/working/v010/bin/pip install -q "vllm==0.10.2" 2>&1 | tail -2
from runner import run
ok = run("vllm-0.10.2", f"/kaggle/working/v010/bin/python probe.py {small} 1")
if ok:
    run("vllm-0.10.2-teacher", f"/kaggle/working/v010/bin/python probe.py {teacher} 2")"""),
    ]
    kernel = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return {"cells": cells, "metadata": {"kernelspec": kernel, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    with open("notebooks/00_env_probe.ipynb", "w") as f:
        json.dump(build(), f, indent=1)
    print("written notebooks/00_env_probe.ipynb")
