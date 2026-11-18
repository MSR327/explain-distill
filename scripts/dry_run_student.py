"""Run the student notebook's setup + smoke cells on this machine (CPU), with Kaggle and Hugging Face stubbed.

    python scripts/dry_run_student.py [0.6B]        # needs torch, transformers, trl, peft, datasets

Catches every bug that does not depend on the GPU (wrong flags, paths, markers, merge, generation glue) before a
notebook is sent to Kaggle. What it cannot catch: T4-only behaviour (fp16/bf16, 4-bit kernels, CUDA memory). The
notebook's own smoke cell covers those in the first minutes of the Kaggle run and reports all failures together.
"""

import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_student_notebook import build  # noqa: E402


class FakeApi:
    def __init__(self, *a, **k):
        pass

    def whoami(self):
        return {"name": "dryrun"}

    def create_repo(self, *a, **k):
        pass

    def upload_file(self, *a, **k):
        pass

    def upload_folder(self, *a, **k):
        pass


def local_download(repo, filename, repo_type=None, **k):
    return str(ROOT / "data" / filename)


def main(size: str = "0.6B") -> None:
    import huggingface_hub

    huggingface_hub.HfApi, huggingface_hub.hf_hub_download = FakeApi, local_download
    secrets = types.ModuleType("kaggle_secrets")
    secrets.UserSecretsClient = type("UserSecretsClient", (), {"get_secret": lambda self, name: ""})
    sys.modules["kaggle_secrets"] = secrets
    os.environ["PYTHONPATH"] = str(ROOT / "src")

    cells = ["".join(c["source"]) for c in build("local", size)["cells"] if c["cell_type"] == "code"]
    pip, auth, smoke = cells[0], cells[1], cells[2]
    work = tempfile.mkdtemp(prefix="dry_run_")
    os.chdir(work)
    env: dict = {}
    for name, src in (("vars", "\n".join(x for x in pip.splitlines() if not x.startswith("%"))),
                      ("auth+data", auth), ("smoke", smoke)):
        print(f"--- running cell: {name}", flush=True)
        exec(compile(src, name, "exec"), env)
    for f in ("smoke/gen_base.jsonl", "smoke/gen_merged.jsonl", "smoke/adapter/adapter_config.json"):
        assert Path(f).exists() and Path(f).stat().st_size > 0, f"missing {f}"
    print(f"DRY RUN PASSED ({work})")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
