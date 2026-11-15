"""Generate notebooks/03_bench_<size>.ipynb: the format x engine serving benchmark for one student.

    python scripts/make_bench_notebook.py <commit> 1.7B [--out PATH]

Rows (all on Kaggle T4 x2, one GPU used per row):
- Transformers fp16 and bitsandbytes NF4, one request at a time: "no serving engine" baselines;
- llama.cpp CUDA (sm_75) on GGUF f16 / Q8_0 / Q4_K_M: single-user latency, then every eval prompt over 8 slots
  (throughput, $/1k requests);
- llama.cpp CPU-only (GPUs hidden, 4 threads) on Q8_0 / Q4_K_M: the free-deployment target, a CPU Space.
vLLM is not a row: 0.33 does not start on T4s (probe v2), so the GPTQ W4A16 / SmoothQuant W8A8 checkpoints that
`distill.quantize` makes (vLLM formats) have no engine here. Each heavy step is a subprocess; every output file
and benchmark summary is uploaded to the private data repo so results survive the session.
"""

import argparse
import json

from nbcommon import build_llama_cpp_cuda

DATA_REPO = "explain-distill-data"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def build(commit: str, size: str) -> dict:
    name = f"qwen3-{size.lower()}-explain"
    cells = [
        md(f"# Serving benchmark: {name}\n\nTransformers fp16 / NF4 baselines; GGUF f16 / Q8_0 / Q4_K_M on "
           "llama.cpp CUDA (T4) and Q8_0 / Q4_K_M on llama.cpp CPU (4 threads). Needs GPU T4 x2, Internet, "
           "secret `hg`."),
        code(f'%pip install -q "bitsandbytes" "datasets" '
             f'"distill @ git+https://github.com/MSR327/explain-distill.git@{commit}"\n'
             f'COMMIT, NAME, SIZE = "{commit}", "{name}", "{size}"'),
        code(build_llama_cpp_cuda("llama-server llama-quantize")),
        code(f"""import os, shutil, subprocess, sys, json, glob

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("hg")
api = HfApi()
USER = api.whoami()["name"]
DATA, MODEL_REPO = f"{{USER}}/{DATA_REPO}", f"{{USER}}/{{NAME}}"
os.makedirs("data", exist_ok=True)
shutil.copy(hf_hub_download(DATA, "pairs.jsonl", repo_type="dataset"), "data/pairs.jsonl")
MERGED = snapshot_download(MODEL_REPO, allow_patterns=["merged/*"]) + "/merged"
print("ready", MERGED, os.listdir(MERGED))


def sh(cmd, marker):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    lines = [x for x in p.stdout.splitlines() if x.startswith(marker)]
    print(*lines, sep="\\n")
    if p.returncode or not lines:
        print("FAILED:", cmd, p.stderr[-3000:])
        return None
    return lines


def upload(system):
    for f in glob.glob(f"outputs/{{system}}*"):
        api.upload_file(path_or_fileobj=f, path_in_repo=f"bench/{{os.path.basename(f)}}", repo_id=DATA,
                        repo_type="dataset", commit_message=f"bench {{system}} (code {{COMMIT}})")"""),
        code("""# Transformers baselines, one request at a time on GPU 0.
for tag, extra in (("hf-fp16", ""), ("hf-nf4", "--load-4bit")):
    sh(f"CUDA_VISIBLE_DEVICES=0 {sys.executable} -m distill.bench_hf --model {MERGED} --n 30 {extra} "
       f"--system {NAME}-{tag} --out outputs/{NAME}-{tag}.jsonl", "BENCH")
    upload(f"{NAME}-{tag}")"""),
        code("""%pip install -q ./llama.cpp/gguf-py sentencepiece
sh(f"{sys.executable} -m distill.llamacpp convert --merged {MERGED} --llama-cpp llama.cpp --out gguf", "GGUF")
print(*[f"{f} {os.path.getsize(f) / 1e9:.2f} GB" for f in sorted(glob.glob("gguf/*.gguf"))], sep="\\n")"""),
        code("""# llama.cpp CUDA, one T4: 30 sequential requests (latency), then all eval prompts on 8 slots.
for q in ("f16", "Q8_0", "Q4_K_M"):
    system = f"{NAME}-llamacpp-cuda-{q}"
    sh(f"{sys.executable} -m distill.llamacpp bench --gguf gguf/model-{q}.gguf --llama-cpp llama.cpp "
       f"--tokenizer {MERGED} --gpu 0 --ngl 99 --parallel 8 --n-latency 30 --system {system} "
       f"--out outputs/{system}.jsonl", "BENCH")
    upload(system)"""),
        code("""# llama.cpp CPU only (GPUs hidden: a CUDA build offloads prompt matmuls), 4 threads, 20 requests.
for q in ("Q8_0", "Q4_K_M"):
    system = f"{NAME}-llamacpp-cpu-{q}"
    sh(f"{sys.executable} -m distill.llamacpp bench --gguf gguf/model-{q}.gguf --llama-cpp llama.cpp "
       f"--tokenizer {MERGED} --gpu none --ngl 0 --threads 4 --n 20 --system {system} "
       f"--out outputs/{system}.jsonl", "BENCH")
    upload(system)
for q in ("Q8_0", "Q4_K_M"):
    api.upload_file(path_or_fileobj=f"gguf/model-{q}.gguf", path_in_repo=f"gguf/{NAME}-{q}.gguf",
                    repo_id=MODEL_REPO, commit_message=f"GGUF {q} (code {COMMIT})")
print("done")"""),
    ]
    kernel = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return {"cells": cells, "metadata": {"kernelspec": kernel, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("commit")
    ap.add_argument("size", choices=["0.6B", "1.7B", "4B"])
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or f"notebooks/03_bench_{a.size}.ipynb"
    with open(out, "w") as f:
        json.dump(build(a.commit, a.size), f, indent=1)
    print("written", out)
