"""Private Kaggle dataset holding H1 windows (prices + date indices, no issue codes). Deleted after the runs.

  uv run python scripts/kaggle_dataset.py create | version | delete
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from kaggle_run import KAGGLE, USER  # noqa: E402

SLUG = "kronos-h1-dev"
d = ROOT / "build" / "kaggle_ds" / SLUG
cmd = sys.argv[1]
if cmd == "delete":
    subprocess.run(KAGGLE + ["datasets", "delete", f"{USER}/{SLUG}", "-y"], check=True)
else:
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "data" / "h1" / "windows.npz", d / "windows.npz")
    (d / "dataset-metadata.json").write_text(json.dumps({"title": SLUG, "id": f"{USER}/{SLUG}",
                                                         "licenses": [{"name": "other"}]}))
    extra = ["-m", "update"] if cmd == "version" else []  # create is private unless --public is passed
    subprocess.run(KAGGLE + ["datasets", cmd, "-p", str(d)] + extra, check=True)
