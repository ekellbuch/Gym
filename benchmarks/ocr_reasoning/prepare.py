# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare published OCR Reasoning answers from the pinned VLMEvalKitMcore fork."""

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
MCORE_HASHES = {
    "vlmeval/dataset/image_vqa.py": "47a46ef8b879d6acf15ac0d4ef8bb046b47fedba9b0532a80843101525b7330d",
    "vlmeval/dataset/utils/ocr_reasoning.py": "e4fa835a637fca884b3948d5b0706413b58e03d7c8e0e0cd7935bd5011f40fa3",
}
OUTPUT_PATH = BENCHMARK_DIR / "data" / "ocr_reasoning_benchmark.jsonl"


def _verify_source(source: Path) -> None:
    if (source / ".source-commit").read_text().strip() != MCORE_COMMIT:
        raise RuntimeError("OCR Reasoning source revision mismatch")
    for relative_path, expected_hash in MCORE_HASHES.items():
        actual = hashlib.sha256((source / relative_path).read_bytes()).hexdigest()
        if actual != expected_hash:
            raise RuntimeError(f"OCR Reasoning source hash mismatch: {relative_path}")


def ensure_mcore_checkout() -> Path:
    with locked_mcore_source(MCORE_DIR):
        return _ensure_mcore_checkout_unlocked()


def _ensure_mcore_checkout_unlocked() -> Path:
    if not MCORE_DIR.exists():
        token = os.environ.get("GITLAB_TOKEN")
        if not token:
            raise RuntimeError("GITLAB_TOKEN is not set in this shell; restart the session.")
        MCORE_DIR.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".ocr_reasoning-source-", dir=MCORE_DIR.parent) as temp_dir:
            archive_path = Path(temp_dir) / "source.tar.gz"
            request = Request(MCORE_URL, headers={"PRIVATE-TOKEN": token})
            with urlopen(request, timeout=120) as response, archive_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(temp_dir, filter="data")
            checkout = next(path for path in Path(temp_dir).iterdir() if path.is_dir())
            (checkout / ".source-commit").write_text(MCORE_COMMIT + "\n")
            _verify_source(checkout)
            checkout.rename(MCORE_DIR)
    _verify_source(MCORE_DIR)
    return MCORE_DIR


def prepare() -> Path:
    source = ensure_mcore_checkout()
    python = SERVER_DIR / ".venv" / "bin" / "python"
    subprocess.run(["uv", "sync", "--project", str(SERVER_DIR), "--locked", "--python", "3.13.14"], check=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(source), str(REPO_ROOT), env.get("PYTHONPATH", "")])
    code = (
        "from benchmarks.ocr_reasoning.prepare_data import prepare_ocr_reasoning; "
        f"prepare_ocr_reasoning({str(OUTPUT_PATH)!r})"
    )
    subprocess.run([str(python), "-c", code], cwd=REPO_ROOT, env=env, check=True)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(prepare())
