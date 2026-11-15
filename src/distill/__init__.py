"""Distill project 2's grounded explanations into small open LLMs."""

import os

# bm25s (project 2's lexical index) runs a JAX op at import time when JAX is installed, and JAX then preallocates
# 75% of every visible GPU's memory for the life of the process. Kaggle's image ships jax[cuda], so importing the
# index left ~4 GB on each T4 and llama-server could not load the teacher (teacher-resumes v4: cudaMalloc of the
# 8.2 GB weights failed). JAX is only used for bm25s's small top-k, so keep it on the CPU everywhere.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
