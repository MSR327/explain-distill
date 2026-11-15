"""GGUF conversion and llama.cpp serving benchmark (CPU, or GPU with a CUDA build).

    python -m distill.llamacpp convert --merged out/merged --llama-cpp llama.cpp --out gguf/  # f16 -> Q8_0, Q4_K_M
    python -m distill.llamacpp bench --gguf gguf/model-Q4_K_M.gguf --llama-cpp llama.cpp --tokenizer out/merged \
        --pairs data/pairs.jsonl --n 50 --system student-1.7b-Q4_K_M --out data/outputs/student-1.7b-Q4_K_M.jsonl

The benchmark talks to `llama-server` over HTTP (the deployment shape) using the raw /completion endpoint with
the exact training-time prompt string (`fmt.prompt_text`), so serving cannot drift from training through a
different chat template. Two passes, as for any serving engine:
- latency: requests one at a time (a single user), p50/p95 seconds and the server's own tokens/s;
- throughput (`--parallel N > 1`): every prompt submitted at once over N slots, requests/s and $/1k requests
  (GPU runs only; a CPU Space serves one user at a time).
`--gpu none` hides every GPU from the server: a CUDA build with -ngl 0 still offloads large prompt matmuls to
the GPU, which would flatter a "CPU" number.
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

# GCP on-demand price of one T4, GPU only (cloud.google.com/products/compute/gpus-pricing, checked 2026-11-15).
# The VM is extra, so $/1k requests computed from it is a lower bound.
T4_USD_PER_GPU_HOUR = 0.35
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

    def __init__(self, gguf: str, llama_cpp: str, threads: int | None = None, ctx: int = 4096, ngl: int = 0,
                 parallel: int = 1, gpu: int | str | None = None, log: str | None = None):
        """`ctx` is per slot; llama-server splits -c across --parallel slots, so it gets ctx * parallel.
        `gpu`: index of the one GPU to use, "" for none (true CPU), None to leave CUDA_VISIBLE_DEVICES alone."""
        self.port = _free_port()
        cmd = [f"{llama_cpp}/build/bin/llama-server", "-m", gguf, "--port", str(self.port),
               "-c", str(ctx * parallel), "-ngl", str(ngl), "--parallel", str(parallel)]
        if threads:
            cmd += ["-t", str(threads)]
        self.cmd, self.log = cmd, log
        self.env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else None

    def __enter__(self):
        import httpx

        out = open(self.log, "w") if self.log else subprocess.DEVNULL  # closed with the process
        self.proc = subprocess.Popen(self.cmd, stdout=out, stderr=subprocess.STDOUT, env=self.env)
        self.url = f"http://127.0.0.1:{self.port}"
        t = time.time()
        while time.time() - t < 900:
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

    def complete(self, prompt: str, max_tokens: int = 700, temperature: float = 0.0, seed: int = 0) -> dict:
        import httpx

        t = time.time()
        r = httpx.post(f"{self.url}/completion", timeout=1800, json={
            "prompt": prompt, "n_predict": max_tokens, "temperature": temperature, "seed": seed,
            "stop": ["<|im_end|>"], "cache_prompt": False})
        r.raise_for_status()
        j = r.json()
        tm = j.get("timings", {})
        return {"text": j["content"], "latency_s": time.time() - t, "prompt_tokens": tm.get("prompt_n"),
                "gen_tokens": tm.get("predicted_n"), "prompt_tok_s": tm.get("prompt_per_second"),
                "gen_tok_s": tm.get("predicted_per_second")}


def bench(gguf: str, llama_cpp: str, tokenizer: str, pairs: list[dict], system: str, threads: int | None = None,
          ngl: int = 0, parallel: int = 1, n_latency: int | None = None, gpu: int | str | None = None
          ) -> tuple[list[dict], dict]:
    from concurrent.futures import ThreadPoolExecutor

    from transformers import AutoTokenizer

    from .fmt import prompt_text

    tok = AutoTokenizer.from_pretrained(tokenizer)
    prompts = [prompt_text(tok, p["messages"]) for p in pairs]
    n_latency = len(prompts) if n_latency is None else min(n_latency, len(prompts))
    texts, stats, thr = [None] * len(prompts), [], {}
    with Server(gguf, llama_cpp, threads=threads, ngl=ngl, parallel=parallel, gpu=gpu) as srv:
        srv.complete(prompts[0], max_tokens=8)  # warm-up
        for i in range(n_latency):
            r = srv.complete(prompts[i])
            texts[i] = r.pop("text")
            stats.append(r)
        if parallel > 1:
            t = time.time()
            with ThreadPoolExecutor(parallel) as pool:
                res = list(pool.map(srv.complete, prompts))
            wall = time.time() - t
            texts = [r["text"] for r in res]  # every prompt has an output to score
            req_s = len(res) / wall
            thr = {"parallel": parallel, "throughput_req_s": req_s,
                   "throughput_gen_tok_s": sum(r["gen_tokens"] or 0 for r in res) / wall,
                   "usd_per_1k_requests": T4_USD_PER_GPU_HOUR / 3600 / req_s * 1000}
        load_s = srv.load_s
    rows = [{"system": system, "resume_job_id": p["resume_job_id"], "job_id": p["job_id"], "text": x}
            for p, x in zip(pairs, texts, strict=True) if x is not None]
    lat = [s["latency_s"] for s in stats]
    summary = {"system": system, "engine": "llama.cpp", "gguf": os.path.basename(gguf),
               "size_gb": round(os.path.getsize(gguf) / 1e9, 3), "n": len(rows), "n_latency": len(stats),
               "threads": threads, "ngl": ngl, "gpu": gpu, "load_s": round(load_s, 1),
               "latency_p50_s": float(np.percentile(lat, 50)), "latency_p95_s": float(np.percentile(lat, 95)),
               "gen_tok_s_mean": float(np.mean([s["gen_tok_s"] for s in stats if s["gen_tok_s"]])),
               "prompt_tok_s_mean": float(np.mean([s["prompt_tok_s"] for s in stats if s["prompt_tok_s"]])),
               "gen_tokens_mean": float(np.mean([s["gen_tokens"] or 0 for s in stats])), **thr}
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
    ap.add_argument("--n", type=int, default=None, help="eval pairs (default: all)")
    ap.add_argument("--system")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=1, help=">1 adds a throughput pass over that many slots")
    ap.add_argument("--n-latency", type=int, default=None, help="sequential requests (default: all --n)")
    ap.add_argument("--gpu", default=None, help='GPU index, or "none" to hide all GPUs (true CPU run)')
    a = ap.parse_args()
    if a.stage == "convert":
        print("GGUF", json.dumps(convert(a.merged, a.llama_cpp, a.out)), flush=True)
        return
    pairs = [p for p in map(json.loads, Path(a.pairs).read_text().splitlines()) if p["split"] == "eval"][: a.n]
    gpu = "" if a.gpu == "none" else (int(a.gpu) if a.gpu is not None else None)
    rows, summary = bench(a.gguf, a.llama_cpp, a.tokenizer, pairs, a.system, a.threads, a.ngl, a.parallel,
                          a.n_latency, gpu)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    Path(a.out).with_suffix(".bench.json").write_text(json.dumps(summary, indent=1))
    print("BENCH", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
