# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MMLongBench-Doc from the exact VLMEvalKitMcore revision used by Gym."""

import hashlib
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from benchmarks.mcore_source import locked_mcore_source


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
SERVER_DIR = REPO_ROOT / "resources_servers" / "vlm_eval_kit"
MCORE_COMMIT = "6962c8d06b2b7b26a74a73d6212c06562b63e1b7"
MCORE_DIR = REPO_ROOT / "results" / "oracle-upstream" / f"VLMEvalKitMcore-{MCORE_COMMIT}"
MCORE_URL = (
    "https://gitlab-master.nvidia.com/api/v4/projects/"
    "matthieul%2FVLMEvalKitMcore/repository/archive.tar.gz?sha=" + MCORE_COMMIT
)
MMLONGBENCH_SHA256 = "bb9a228ec3d0c6e116e4401d7dfe4dc1b07259410b23ae21b47f128813d957e9"
OUTPUT_PATH = BENCHMARK_DIR / "data" / "mmlongbench_doc_benchmark.jsonl"


def ensure_mcore_checkout() -> Path:
    with locked_mcore_source(MCORE_DIR):
        return _ensure_mcore_checkout_unlocked()


def _ensure_mcore_checkout_unlocked() -> Path:
    if not MCORE_DIR.exists():
        token = os.environ.get("GITLAB_TOKEN")
        if not token:
            raise RuntimeError("GITLAB_TOKEN is not set in this shell; restart the session.")
        MCORE_DIR.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".mmlongbench-source-", dir=MCORE_DIR.parent) as temp_dir:
            archive_path = Path(temp_dir) / "source.tar.gz"
            request = Request(MCORE_URL, headers={"PRIVATE-TOKEN": token})
            with urlopen(request, timeout=120) as response, archive_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(temp_dir, filter="data")
            checkout = next(path for path in Path(temp_dir).iterdir() if path.is_dir())
            source_file = checkout / "vlmeval" / "dataset" / "mmlongbench.py"
            if hashlib.sha256(source_file.read_bytes()).hexdigest() != MMLONGBENCH_SHA256:
                raise RuntimeError("MMLongBench-Doc scorer source hash mismatch")
            (checkout / ".source-commit").write_text(MCORE_COMMIT + "\n")
            checkout.rename(MCORE_DIR)
    if (MCORE_DIR / ".source-commit").read_text().strip() != MCORE_COMMIT:
        raise RuntimeError("MMLongBench-Doc source commit mismatch")
    source_file = MCORE_DIR / "vlmeval" / "dataset" / "mmlongbench.py"
    if hashlib.sha256(source_file.read_bytes()).hexdigest() != MMLONGBENCH_SHA256:
        raise RuntimeError("MMLongBench-Doc scorer source hash mismatch")
    return MCORE_DIR


def prepare() -> Path:
    source = ensure_mcore_checkout()
    python = SERVER_DIR / ".venv" / "bin" / "python"
    subprocess.run(["uv", "sync", "--project", str(SERVER_DIR), "--locked", "--python", "3.13.14"], check=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(source), str(REPO_ROOT), env.get("PYTHONPATH", "")])
    code = (
        "from benchmarks.mmlongbench_doc.prepare_data import prepare_mmlongbench_doc; "
        f"prepare_mmlongbench_doc({str(OUTPUT_PATH)!r}, max_pages=100)"
    )
    subprocess.run([str(python), "-c", code], cwd=REPO_ROOT, env=env, check=True)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(prepare())
