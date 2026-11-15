#!/bin/sh
# Download the GGUF (cached in the image's home dir after the first start), then serve it.
# LLAMA_API_KEY (a Space secret) makes llama-server reject requests without that bearer key.
set -e
GGUF=$(python -c "from huggingface_hub import hf_hub_download as d; import os; print(d(os.environ['MODEL_REPO'], os.environ['MODEL_FILE']))")
exec llama-server -m "$GGUF" --host 0.0.0.0 --port 7860 -c 4096 --parallel 1 -t 2 --no-webui --jinja
