"""Download the exact public checkpoint needed by 05a; never resolve main.

Run from the repository: .venv/bin/python experiments/prepare_donut_foundations.py
The notebook itself loads local files offline. All caches stay in the checkout.
"""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_XET_CACHE", str(ROOT / ".cache/huggingface/xet"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from huggingface_hub import snapshot_download


def main():
    policy = json.loads((ROOT / "configs/donut_foundations_v1.json").read_text())
    destination = ROOT / policy["checkpoint_relative_path"]
    snapshot_download(
        repo_id=policy["model_id"],
        revision=policy["revision"],
        allow_patterns=policy["download_files"],
        local_dir=destination,
        token=False,
        max_workers=2,
    )
    hashes = {}
    for name in policy["download_files"]:
        with (destination / name).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {
        "model_id": policy["model_id"],
        "revision": policy["revision"],
        "files_sha256": hashes,
    }
    (destination / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
