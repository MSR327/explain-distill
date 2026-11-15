"""Cells shared by the Kaggle notebook generators."""

# Pinned so runs are reproducible; the same commit was built and tested on CPU locally.
LLAMA_CPP_COMMIT = "b54752df40e4d8e53d34ef5a598336b4e4871d39"


def build_llama_cpp_cuda(targets: str = "llama-server") -> str:
    """Notebook cell: llama.cpp with CUDA for the T4 (compute capability 7.5), failing loudly.

    GGML_CUDA_NO_VMM=ON: by default ggml-cuda links libcuda.so (CMake target CUDA::cuda_driver) for its virtual
    memory pool. Kaggle's image has the toolkit but no unversioned libcuda.so, so configure failed there
    (teacher-resumes v2). Without VMM the pool is a plain cudaMalloc pool, which is fine for fixed-size serving.
    """
    return f"""# llama.cpp with CUDA for compute capability 7.5 (~10 min build). Raises if the build fails.
import os, subprocess
def sh(cmd, log):
    with open(log, "w") as f:
        p = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        print(open(log).read()[-6000:])
        raise RuntimeError(f"failed: {{cmd}}")
sh("(nvcc --version || /usr/local/cuda/bin/nvcc --version) 2>&1 || true", "nvcc.log")
print(open("nvcc.log").read().strip()[-80:])
sh("git init -q llama.cpp && cd llama.cpp && git remote add origin https://github.com/ggml-org/llama.cpp "
   "&& git fetch -q --depth 1 origin {LLAMA_CPP_COMMIT} && git checkout -q FETCH_HEAD", "clone.log")
sh("cd llama.cpp && cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DCMAKE_CUDA_ARCHITECTURES=75 "
   "-DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release", "cmake.log")
sh("cd llama.cpp && cmake --build build -j 4 --target {targets}", "build.log")
for t in "{targets}".split():
    assert os.path.exists(f"llama.cpp/build/bin/{{t}}"), t
print("llama.cpp built:", "{targets}")"""
