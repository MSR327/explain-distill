# CPU-only Hugging Face Docker Space serving a distilled explain-distill student with llama.cpp's llama-server
# (OpenAI-compatible /v1/chat/completions on port 7860). Free tier: 2 vCPU, 16 GB RAM.
FROM debian:bookworm-slim AS build
ARG LLAMA_CPP_COMMIT=b54752df40e4d8e53d34ef5a598336b4e4871d39
RUN apt-get update && apt-get install -y --no-install-recommends git cmake g++ make ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git init -q /src && cd /src && git remote add origin https://github.com/ggml-org/llama.cpp \
    && git fetch -q --depth 1 origin ${LLAMA_CPP_COMMIT} && git checkout -q FETCH_HEAD \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build -j 4 --target llama-server

FROM python:3.12-slim
RUN pip install --no-cache-dir "huggingface_hub>=0.34" && useradd -m -u 1000 user
COPY --from=build /src/build/bin/llama-server /usr/local/bin/llama-server
COPY start.sh /start.sh
USER user
# 0.6B Q4_K_M: same strict pass as f16 (-0.6 pts) at 0.40 GB; ~25 s per explanation on 4 CPU cores.
ENV HOME=/home/user MODEL_REPO=srish327/qwen3-0.6b-explain MODEL_FILE=gguf/qwen3-0.6b-explain-Q4_K_M.gguf
EXPOSE 7860
CMD ["sh", "/start.sh"]
