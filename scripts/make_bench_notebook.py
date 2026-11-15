"""Generate notebooks/03_bench_<size>.ipynb: the format x engine serving benchmark for one student.

    python scripts/make_bench_notebook.py <commit> 1.7B [--out PATH]

Order matters for dependencies: llm-compressor (quantize) -> Transformers baseline -> vLLM (installed last,
it may pin its own torch) -> llama.cpp CPU build (GGUF). Each heavy step is a subprocess. Every output file
and benchmark summary is uploaded to the private data repo, so results survive the session.
"""

import argparse
import json

DATA_REPO = "explain-distill-data"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def build(commit: str, size: str) -> dict:
    name = f"qwen3-{size.lower()}-explain"
    cells = [
        md(f"# Serving benchmark: {name}\n\nfp16 / W4A16 (GPTQ) / W8A8 (SmoothQuant+GPTQ) / bitsandbytes on vLLM "
           "(T4), Transformers fp16 baseline, GGUF Q8_0 / Q4_K_M on llama.cpp (CPU, 4 threads). Needs GPU T4 x2, "
           "Internet, secret `hg`."),
        code(f'%pip install -q "llmcompressor" "datasets" '
             f'"distill @ git+https://github.com/MSR327/explain-distill.git@{commit}"\n'
             f'COMMIT, NAME, SIZE = "{commit}", "{name}", "{size}"'),
        code(f"""import os, shutil, subprocess, sys, json, glob

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("hg")
api = HfApi()
USER = api.whoami()["name"]
DATA, MODEL_REPO = f"{{USER}}/{DATA_REPO}", f"{{USER}}/{{NAME}}"
os.makedirs("data", exist_ok=True)
for f in ("pairs.jsonl", "sft_train.jsonl"):
    shutil.copy(hf_hub_download(DATA, f, repo_type="dataset"), f"data/{{f}}")
MERGED = snapshot_download(MODEL_REPO, allow_patterns=["merged/*"]) + "/merged"
print("ready", MERGED, os.listdir(MERGED))
RESULTS = []


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
        code("""for scheme in ("W4A16", "W8A8"):
    sh(f"{sys.executable} -m distill.quantize --merged {MERGED} --scheme {scheme} --out q/{scheme}", "QUANTIZED")"""),
        code("""sh(f"{sys.executable} -m distill.bench_hf --model {MERGED} --system {NAME}-hf-fp16 "
   f"--out outputs/{NAME}-hf-fp16.jsonl", "BENCH")
upload(f"{NAME}-hf-fp16")"""),
        code("""%pip install -q "vllm"
runs = [(MERGED, "fp16", ""), ("q/W4A16", "w4a16", ""), ("q/W8A8", "w8a8", ""),
        (MERGED, "bnb4", "--quantization bitsandbytes")]
for model, tag, extra in runs:
    if os.path.exists(model):
        sh(f"{sys.executable} -m distill.bench_vllm --model {model} --system {NAME}-vllm-{tag} {extra} "
           f"--out outputs/{NAME}-vllm-{tag}.jsonl", "BENCH")
        upload(f"{NAME}-vllm-{tag}")"""),
        code("""# llama.cpp, CPU build (the free-deployment target: a CPU-only Space).
!git clone -q --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp && git log -1 --format=%h \\
  && cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA=OFF > /dev/null \\
  && cmake --build build -j 4 --target llama-quantize llama-server > /dev/null && ls build/bin | head
%pip install -q ./llama.cpp/gguf-py sentencepiece
sh(f"{sys.executable} -m distill.llamacpp convert --merged {MERGED} --llama-cpp llama.cpp --out gguf", "GGUF")
for q in ("Q8_0", "Q4_K_M"):
    sh(f"{sys.executable} -m distill.llamacpp bench --gguf gguf/model-{q}.gguf --llama-cpp llama.cpp "
       f"--tokenizer {MERGED} --n 20 --threads 4 --system {NAME}-llamacpp-{q} "
       f"--out outputs/{NAME}-llamacpp-{q}.jsonl", "BENCH")
    upload(f"{NAME}-llamacpp-{q}")
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
