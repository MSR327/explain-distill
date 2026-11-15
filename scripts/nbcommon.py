"""Cells shared by the Kaggle notebook generators."""

# Pinned so runs are reproducible; the same commit was built and tested on CPU locally.
LLAMA_CPP_COMMIT = "b54752df40e4d8e53d34ef5a598336b4e4871d39"


def build_llama_cpp_cuda(targets: str = "llama-server", data_repo: str = "explain-distill-data") -> str:
    """Notebook cell: llama.cpp with CUDA for the T4 (compute capability 7.5), failing loudly.

    Needs HF_TOKEN in the environment (run it after the HF auth cell). The compiled build/bin is cached in the
    private data repo, keyed by llama.cpp commit + CUDA version: the CUDA build compiles hundreds of kernel
    variants on Kaggle's 4 cores (20-30 min), so only the first notebook pays for it. The source checkout is
    always fetched (convert_hf_to_gguf.py and gguf-py come from it).

    GGML_CUDA_NO_VMM=ON: by default ggml-cuda links libcuda.so (CMake target CUDA::cuda_driver) for its virtual
    memory pool. Kaggle's image has the toolkit but no unversioned libcuda.so, so configure failed there
    (teacher-resumes v2). Without VMM the pool is a plain cudaMalloc pool, which is fine for fixed-size serving.
    """
    return f"""# llama.cpp CUDA for compute capability 7.5: cached binaries if present, else build (~25 min) + cache.
import os, re, subprocess, tarfile
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

def _build_sh(cmd, log):
    with open(log, "w") as f:
        p = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        print(open(log).read()[-6000:])
        raise RuntimeError(f"failed: {{cmd}}")

_build_sh("(nvcc --version || /usr/local/cuda/bin/nvcc --version) 2>&1 || true", "nvcc.log")
cuda = (re.findall(r"release ([0-9.]+)", open("nvcc.log").read()) or ["unknown"])[0]
TARGETS = "{targets}".split()
_api = HfApi()
_repo = f"{{_api.whoami()['name']}}/{data_repo}"
_tar = f"llama-cpp-bin/{LLAMA_CPP_COMMIT[:10]}-cuda{{cuda}}-sm75.tar.gz"
print("CUDA", cuda, "cache", _tar)
if not os.path.exists("llama.cpp/.git"):
    _build_sh("git init -q llama.cpp && cd llama.cpp && git remote add origin https://github.com/ggml-org/llama.cpp "
       "&& git fetch -q --depth 1 origin {LLAMA_CPP_COMMIT} && git checkout -q FETCH_HEAD", "clone.log")
try:
    with tarfile.open(hf_hub_download(_repo, _tar, repo_type="dataset")) as t:
        t.extractall("llama.cpp", filter="data")
    print("using cached llama.cpp binaries")
except EntryNotFoundError:
    _build_sh("cd llama.cpp && cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DCMAKE_CUDA_ARCHITECTURES=75 "
       "-DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release", "cmake.log")
    _build_sh("cd llama.cpp && cmake --build build -j 4 --target llama-server llama-quantize", "build.log")
    with tarfile.open("llama-bin.tar.gz", "w:gz") as t:
        t.add("llama.cpp/build/bin", arcname="build/bin")
    _api.upload_file(path_or_fileobj="llama-bin.tar.gz", path_in_repo=_tar, repo_id=_repo, repo_type="dataset",
                     commit_message=f"llama.cpp {LLAMA_CPP_COMMIT[:10]} CUDA {{cuda}} sm_75 binaries")
    print("built and cached llama.cpp binaries")
for t in TARGETS:
    assert os.access(f"llama.cpp/build/bin/{{t}}", os.X_OK), t
print("llama.cpp ready:", *TARGETS)"""
