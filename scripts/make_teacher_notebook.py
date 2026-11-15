"""Generate notebooks/01_teacher_<stage>.ipynb pinned to a commit of this repo.

    python scripts/make_teacher_notebook.py <commit> resumes|explain [--out PATH]

Each run: install this package (+ build llama.cpp CUDA, or vLLM) -> fail-fast HF auth -> smoke pass on a
few items (raises if nothing is kept) -> full run -> upload the stage's output to the private HF dataset repo.
The pairs stage runs locally between the two GPU stages (CPU work; no reason to hold a GPU for it).
"""

import argparse
import json

from nbcommon import build_llama_cpp_cuda

DATA_REPO = "explain-distill-data"  # private dataset repo under the token's account


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


STAGES = {
    "resumes": {"needs": [], "out": "resumes.jsonl", "args": "--n-train 1000 --n-eval 120", "smoke": 6,
                "marker": "RESUMES"},
    "explain": {"needs": ["pairs.jsonl"], "out": "teacher.jsonl", "args": "", "smoke": 8, "marker": "TEACHER"},
}


TEACHER = "Qwen/Qwen3-14B-GGUF:Qwen3-14B-Q4_K_M.gguf"



def build(commit: str, stage: str, model: str = TEACHER, tp: int = 2, engine: str = "llamacpp",
          vllm_spec: str = "vllm") -> dict:
    st = STAGES[stage]
    install = f'"{vllm_spec}" ' if engine == "vllm" else ""
    cells = [
        md(f"# Teacher data, stage `{stage}`\n\nTeacher {model} on {engine} (tp={tp}). Needs GPU T4 x2, Internet, "
           "secret `hg` (HF write token). Output goes to a private HF dataset repo. On T4s the engine is llama.cpp "
           "built with CUDA for sm_75: vLLM 0.33 does not start on these GPUs (probe v1/v2)."),
        code(f'%pip install -q {install}"distill @ git+https://github.com/MSR327/explain-distill.git@{commit}"\n'
             f'COMMIT, STAGE, MODEL, TP, ENGINE = "{commit}", "{stage}", "{model}", {tp}, "{engine}"'),
        code(f"""# Fail fast: HF token + write access before loading a 14B model.
import os

from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("hg")
api = HfApi()
REPO = f"{{api.whoami()['name']}}/{DATA_REPO}"
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
print("HF auth ok ->", REPO)"""),
        *([code(build_llama_cpp_cuda())] if engine == "llamacpp" else []),
        code(f"""from huggingface_hub import hf_hub_download
import shutil
os.makedirs("data", exist_ok=True)
for name in {st["needs"]!r}:
    shutil.copy(hf_hub_download(REPO, name, repo_type="dataset"), f"data/{{name}}")
    print("downloaded", name)"""),
        code(f"""# Smoke pass on a few items: catches a broken stack in minutes, not hours.
import subprocess, sys
def run(extra, data):
    for d in ("data", data):
        os.makedirs(d, exist_ok=True)
    for name in {st["needs"]!r}:
        shutil.copy(f"data/{{name}}", f"{{data}}/{{name}}")
    cmd = (f"{{sys.executable}} -m distill.teacher {{STAGE}} --model {{MODEL}} --tp {{TP}} --engine {{ENGINE}} "
           f"--data {{data}} {{extra}}")
    # Stream progress ("generated N/M X min") live so a long run is visibly alive; stderr goes to a file.
    lines = []
    with open(f"{{data}}/stderr.log", "w") as err:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
        for x in p.stdout:
            if x.startswith(("{st["marker"]}", "generated")):
                print(x.rstrip(), flush=True)
            if x.startswith("{st["marker"]}"):
                lines.append(x.rstrip())
        p.wait()
    if p.returncode or not lines:
        print(open(f"{{data}}/stderr.log").read()[-4000:])
        import glob
        for log in sorted(glob.glob("llama-server-*.log")):
            print(f"--- {{log}} (tail) ---", open(log).read()[-2500:], sep="\\n")
        raise RuntimeError(f"stage failed: {{cmd}}")
    return lines

run("--limit {st["smoke"]}", "smoke")
print(open("smoke/{st["out"]}").read()[:1500])"""),
        code(f"""import time
t = time.time()
run("{st["args"]}", "data")
print(f"full run {{(time.time() - t) / 60:.1f}} min")"""),
        code(f"""api.upload_file(path_or_fileobj="data/{st["out"]}", path_in_repo="{st["out"]}", repo_id=REPO,
                repo_type="dataset", commit_message=f"{stage} ({{MODEL}}, code commit {{COMMIT}})")
print("uploaded {st["out"]}")"""),
    ]
    kernel = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return {"cells": cells, "metadata": {"kernelspec": kernel, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("commit")
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--out")
    ap.add_argument("--model", default=TEACHER)
    ap.add_argument("--engine", choices=["vllm", "hf", "llamacpp"], default="llamacpp")
    ap.add_argument("--vllm-spec", default="vllm", help='pip spec, e.g. "vllm==0.10.2"')
    a = ap.parse_args()
    out = a.out or f"notebooks/01_teacher_{a.stage}.ipynb"
    with open(out, "w") as f:
        json.dump(build(a.commit, a.stage, a.model, engine=a.engine, vllm_spec=a.vllm_spec), f, indent=1)
    print("written", out)
