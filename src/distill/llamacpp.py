"""GGUF conversion and llama.cpp serving benchmark (CPU, or GPU with a CUDA build).

    python -m distill.llamacpp convert --merged out/merged --llama-cpp llama.cpp --out gguf/  # f16 -> Q8_0, Q4_K_M
    python -m distill.llamacpp bench --gguf gguf/model-Q4_K_M.gguf --llama-cpp llama.cpp --tokenizer out/merged \
        --pairs data/pairs.jsonl --n 50 --system student-1.7b-Q4_K_M --out data/outputs/student-1.7b-Q4_K_M.jsonl

The benchmark talks to `llama-server` over HTTP (the deployment shape) using the raw /completion endpoint with
the exact training-time prompt string (`fmt.prompt_text`), so serving cannot drift from training through a
different chat template. Requests are sequential (one user): latency per request, prompt and generation
tokens/s from the server's own timings.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

QUANTS = ("Q8_0", "Q4_K_M")


def convert(merged: str, llama_cpp: str, out: str, quants=QUANTS) -> list[str]:
    Path(out).mkdir(parents=True, exist_ok=True)
    f16 = f"{out}/model-f16.gguf"
    subprocess.run([sys.executable, f"{llama_cpp}/convert_hf_to_gguf.py", merged, "--outtype", "f16",
                    "--outfile", f16], check=True)
    files = [f16]
    for q in quants:
        dst = f"{out}/model-{q}.gguf"
        subprocess.run([f"{llama_cpp}/build/bin/llama-quantize", f16, dst, q], check=True,
                       stdout=subprocess.DEVNULL)
        files.append(dst)
    return files


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """llama-server subprocess; context manager so it is always stopped."""

    def __init__(self, gguf: str, llama_cpp: str, threads: int | None = None, ctx: int = 4096, ngl: int = 0):
        self.port = _free_port()
        cmd = [f"{llama_cpp}/build/bin/llama-server", "-m", gguf, "--port", str(self.port), "-c", str(ctx),
               "-ngl", str(ngl), "--parallel", "1"]
        if threads:
            cmd += ["-t", str(threads)]
        self.cmd = cmd

    def __enter__(self):
        import httpx

        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.url = f"http://127.0.0.1:{self.port}"
        t = time.time()
        while time.time() - t < 300:
            try:
                if httpx.get(f"{self.url}/health", timeout=2).status_code == 200:
                    self.load_s = time.time() - t
                    return self
            except httpx.HTTPError:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited: {' '.join(self.cmd)}")
            time.sleep(0.5)
        raise TimeoutError("llama-server did not become healthy")

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait(timeout=30)

    def complete(self, prompt: str, max_tokens: int = 700) -> dict:
        import httpx

        t = time.time()
        r = httpx.post(f"{self.url}/completion", timeout=600, json={
            "prompt": prompt, "n_predict": max_tokens, "temperature": 0, "stop": ["<|im_end|>"], "cache_prompt": False})
        r.raise_for_status()
        j = r.json()
        tm = j.get("timings", {})
        return {"text": j["content"], "latency_s": time.time() - t, "prompt_tokens": tm.get("prompt_n"),
                "gen_tokens": tm.get("predicted_n"), "prompt_tok_s": tm.get("prompt_per_second"),
                "gen_tok_s": tm.get("predicted_per_second")}


def bench(gguf: str, llama_cpp: str, tokenizer: str, pairs: list[dict], system: str, threads: int | None = None,
          ngl: int = 0) -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer

    from .fmt import prompt_text

    tok = AutoTokenizer.from_pretrained(tokenizer)
    rows, stats = [], []
    with Server(gguf, llama_cpp, threads=threads, ngl=ngl) as srv:
        srv.complete(prompt_text(tok, pairs[0]["messages"]), max_tokens=8)  # warm-up
        for p in pairs:
            r = srv.complete(prompt_text(tok, p["messages"]))
            rows.append({"system": system, "resume_job_id": p["resume_job_id"], "job_id": p["job_id"],
                         "text": r.pop("text")})
            stats.append(r)
        load_s = srv.load_s
    lat = [s["latency_s"] for s in stats]
    summary = {"system": system, "gguf": os.path.basename(gguf), "size_gb": round(os.path.getsize(gguf) / 1e9, 3),
               "n": len(stats), "threads": threads, "ngl": ngl, "load_s": round(load_s, 1),
               "latency_p50_s": float(np.percentile(lat, 50)), "latency_p95_s": float(np.percentile(lat, 95)),
               "gen_tok_s_mean": float(np.mean([s["gen_tok_s"] for s in stats if s["gen_tok_s"]])),
               "prompt_tok_s_mean": float(np.mean([s["prompt_tok_s"] for s in stats if s["prompt_tok_s"]])),
               "gen_tokens_mean": float(np.mean([s["gen_tokens"] or 0 for s in stats]))}
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["convert", "bench"])
    ap.add_argument("--llama-cpp", default="llama.cpp")
    ap.add_argument("--merged")
    ap.add_argument("--out")
    ap.add_argument("--gguf")
    ap.add_argument("--tokenizer")
    ap.add_argument("--pairs", default="data/pairs.jsonl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--system")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--ngl", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "convert":
        print("GGUF", json.dumps(convert(a.merged, a.llama_cpp, a.out)), flush=True)
        return
    pairs = [p for p in map(json.loads, Path(a.pairs).read_text().splitlines()) if p["split"] == "eval"][: a.n]
    rows, summary = bench(a.gguf, a.llama_cpp, a.tokenizer, pairs, a.system, a.threads, a.ngl)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    Path(a.out).with_suffix(".bench.json").write_text(json.dumps(summary, indent=1))
    print("BENCH", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
