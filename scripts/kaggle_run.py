"""Build, push and fetch a Kaggle GPU job that runs remote/jobs.py with this repo's kx/ code.

  uv run python scripts/kaggle_run.py push <job> '<cfg json>'   # bundle + push (private kernel, GPU, internet)
  uv run python scripts/kaggle_run.py status <job>
  uv run python scripts/kaggle_run.py fetch <job>               # download outputs to runs/<job>/
  uv run python scripts/kaggle_run.py local <job> '<cfg json>'  # same job on this CPU, into runs/<job>-cpu/
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "<kaggle-user>"
KAGGLE = ["uvx", "--from", "kaggle", "--with", "six", "kaggle"]
BUNDLE = ["kx/__init__.py", "kx/kronos_env.py", "kx/forecast.py", "kx/synthetic.py", "remote/__init__.py",
          "remote/jobs.py", "scripts/fetch_kronos.sh"]

BOOT = '''import json, subprocess, sys
from pathlib import Path
FILES = {files}
JOB, CFG, COMMIT = {job!r}, {cfg}, {commit!r}
root = Path("/tmp/k")  # outside /kaggle/working so the Kronos clone is not part of the outputs
for rel, src in FILES.items():
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src)
subprocess.run(["bash", str(root / "scripts/fetch_kronos.sh")], check=True)
try:
    import einops  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops==0.8.1"], check=True)
sys.path.insert(0, str(root))
import torch
from remote.jobs import run
m = run(JOB, CFG, Path("/kaggle/working/out"), "cuda" if torch.cuda.is_available() else "cpu")
m["git_commit"] = COMMIT
Path("/kaggle/working/out/manifest.json").write_text(json.dumps(m, indent=1))
print(json.dumps(m))
'''


def slug(job: str) -> str:
    return f"kronos-eval-{job}"


def push(job: str, cfg: dict) -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    files = {rel: (ROOT / rel).read_text() for rel in BUNDLE}
    d = ROOT / "build" / "kaggle" / job
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(BOOT.format(files=repr(files), job=job, cfg=repr(cfg), commit=commit))
    meta = {"id": f"{USER}/{slug(job)}", "title": slug(job), "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": cfg.get("datasets", []), "competition_sources": [], "kernel_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    subprocess.run(KAGGLE + ["kernels", "push", "-p", str(d)], check=True)


def main() -> None:
    cmd, job = sys.argv[1], sys.argv[2]
    if cmd == "push":
        push(job, json.loads(sys.argv[3]))
    elif cmd == "status":
        subprocess.run(KAGGLE + ["kernels", "status", f"{USER}/{slug(job)}"], check=True)
    elif cmd == "fetch":
        out = ROOT / "runs" / job
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(KAGGLE + ["kernels", "output", f"{USER}/{slug(job)}", "-p", str(out), "-o"], check=True)
    elif cmd == "local":
        sys.path.insert(0, str(ROOT))
        from remote.jobs import run
        print(json.dumps(run(job, json.loads(sys.argv[3]), ROOT / "runs" / f"{job}-cpu", "cpu"), indent=1))


if __name__ == "__main__":
    main()
