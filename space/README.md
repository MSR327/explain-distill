---
title: explain-distill student
emoji: 🧭
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

Serves a Qwen3 student distilled to write grounded resume/job-fit explanations (see
[explain-distill](https://github.com/MSR327/explain-distill)) with llama.cpp's `llama-server` on CPU.
OpenAI-compatible: `POST /v1/chat/completions` with `chat_template_kwargs: {"enable_thinking": false}` (the
student was trained on that prompt rendering). Requests need the bearer key set as the `LLAMA_API_KEY` secret.
Used by [resume-job-matcher](https://resume-job-matcher-msr.streamlit.app) as its first explainer, with Groq as the
fallback when this Space is asleep or busy.
