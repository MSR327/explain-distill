"""bench() bookkeeping with a fake llama-server: latency rows, the 8-slot throughput pass, cost and output rows.

The llama-server binary itself is exercised on Kaggle (teacher stage); this covers the Python around it.
"""

import pytest

pytest.importorskip("transformers")

from distill import llamacpp


class FakeTok:
    def apply_chat_template(self, messages, **_):
        return "|".join(m["content"] for m in messages)


class FakeServer:
    def __init__(self, gguf, llama_cpp, threads=None, ngl=0, parallel=1, gpu=None, **_):
        self.parallel, self.load_s = parallel, 1.5

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def complete(self, prompt, max_tokens=700, **_):
        return {"text": f"out:{prompt}", "latency_s": 0.5, "prompt_tokens": 100, "gen_tokens": 50,
                "prompt_tok_s": 1000.0, "gen_tok_s": 100.0}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: FakeTok()))
    monkeypatch.setattr(llamacpp, "Server", FakeServer)
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"0" * 1000)
    pairs = [{"resume_job_id": f"r{i}", "job_id": f"j{i}", "messages": [{"role": "user", "content": f"p{i}"}]}
             for i in range(5)]
    return str(gguf), pairs


def test_latency_only(fake):
    gguf, pairs = fake
    rows, s = llamacpp.bench(gguf, "llama.cpp", "tok", pairs, "sys", threads=4, n_latency=3)
    assert [r["resume_job_id"] for r in rows] == ["r0", "r1", "r2"]  # only requests that ran are scored
    assert s["n"] == 3 and s["n_latency"] == 3 and s["latency_p50_s"] == 0.5
    assert s["gen_tok_s_mean"] == 100.0 and "throughput_req_s" not in s


def test_throughput_pass_outputs_every_prompt(fake):
    gguf, pairs = fake
    rows, s = llamacpp.bench(gguf, "llama.cpp", "tok", pairs, "sys", ngl=99, parallel=8, n_latency=2, gpu=0)
    assert len(rows) == 5 and rows[4]["text"] == "out:p4"
    assert s["parallel"] == 8 and s["throughput_req_s"] > 0
    assert s["usd_per_1k_requests"] == pytest.approx(llamacpp.T4_USD_PER_GPU_HOUR / 3600 / s["throughput_req_s"]
                                                     * 1000)
    assert s["size_gb"] == 0.0 and s["gguf"] == "model-Q4_K_M.gguf"
