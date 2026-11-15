"""Generate notebooks/02_student_<size>.ipynb: QLoRA SFT -> merge -> upload -> eval generations (base + student).

    python scripts/make_student_notebook.py <commit> 0.6B|1.7B|4B [--epochs 2] [--out PATH]

Training runs before vLLM is installed (vLLM may pin a different torch); every heavy step is a subprocess, so
an install never changes the torch already loaded in the kernel. A 3-step smoke run on one GPU must succeed
before the full two-GPU run starts.
"""

import argparse
import json

DATA_REPO = "explain-distill-data"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def build(commit: str, size: str, epochs: float = 2.0) -> dict:
    base = f"Qwen/Qwen3-{size}"
    name = f"qwen3-{size.lower()}-explain"
    cells = [
        md(f"# Student {base}: QLoRA SFT, merge, eval generations\n\nNeeds GPU T4 x2, Internet, secret `hg`. "
           f"Uploads the adapter and merged fp16 weights to a private HF model repo `{name}`, and eval outputs "
           f"to the private dataset repo `{DATA_REPO}`."),
        code(f'%pip install -q "trl==1.18.0" "peft==0.21.3" "bitsandbytes" "datasets" '
             f'"distill @ git+https://github.com/MSR327/explain-distill.git@{commit}"\n'
             f'COMMIT, BASE, NAME, EPOCHS = "{commit}", "{base}", "{name}", {epochs}'),
        code(f"""import os, shutil, subprocess, sys, json, time

from huggingface_hub import HfApi, hf_hub_download
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("hg")
api = HfApi()
USER = api.whoami()["name"]
DATA, MODEL_REPO = f"{{USER}}/{DATA_REPO}", f"{{USER}}/{{NAME}}"
api.create_repo(MODEL_REPO, private=True, exist_ok=True)
os.makedirs("data", exist_ok=True)
for f in ("sft_train.jsonl", "sft_eval.jsonl", "pairs.jsonl"):
    shutil.copy(hf_hub_download(DATA, f, repo_type="dataset"), f"data/{{f}}")
print("HF ok; data ready:", os.listdir("data"))


def sh(cmd, marker):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    lines = [x for x in p.stdout.splitlines() if x.startswith(marker)]
    print(*lines, sep="\\n")
    if p.returncode or not lines:
        print(p.stdout[-2000:], p.stderr[-6000:])
        raise RuntimeError(cmd)
    return lines"""),
        code("""# Smoke: 3 steps on one GPU (catches fp16/NaN/OOM/format problems in minutes).
sh(f"CUDA_VISIBLE_DEVICES=0 {sys.executable} -m distill.sft --model {BASE} --out smoke --max-steps 3 --limit 16",
   "TRAINED")"""),
        code("""t = time.time()
sh(f"torchrun --nproc_per_node 2 -m distill.sft --model {BASE} --out out --epochs {EPOCHS}", "TRAINED")
print(f"train {(time.time() - t) / 60:.1f} min")
sh(f"{sys.executable} -c \\"from distill.sft import merge; merge('{BASE}', 'out/adapter', 'out/merged'); "
   f"print('MERGED')\\"", "MERGED")"""),
        code("""api.upload_folder(folder_path="out/adapter", path_in_repo="adapter", repo_id=MODEL_REPO,
                  commit_message=f"LoRA adapter (code {COMMIT})")
api.upload_folder(folder_path="out/merged", path_in_repo="merged", repo_id=MODEL_REPO,
                  commit_message=f"merged fp16 (code {COMMIT})")
api.upload_file(path_or_fileobj="out/train_log.json", path_in_repo="train_log.json", repo_id=MODEL_REPO)
print("uploaded", MODEL_REPO)"""),
        code("""%pip install -q "vllm"
os.makedirs("outputs", exist_ok=True)
for model, system in ((BASE, f"base-{BASE.split('-')[-1]}"), ("out/merged", NAME)):
    sh(f"{sys.executable} -m distill.generate --model {model} --system {system} --out outputs/{system}.jsonl",
       "GENERATED")
    api.upload_file(path_or_fileobj=f"outputs/{system}.jsonl", path_in_repo=f"outputs/{system}.jsonl",
                    repo_id=DATA, repo_type="dataset", commit_message=f"eval outputs {system} (code {COMMIT})")
print("done")"""),
    ]
    kernel = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return {"cells": cells, "metadata": {"kernelspec": kernel, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("commit")
    ap.add_argument("size", choices=["0.6B", "1.7B", "4B"])
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or f"notebooks/02_student_{a.size}.ipynb"
    with open(out, "w") as f:
        json.dump(build(a.commit, a.size, a.epochs), f, indent=1)
    print("written", out)
