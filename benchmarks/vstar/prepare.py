# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare VStarBench (without CI) from pinned official VLMEvalKitMcore.

The source commit and file hashes are shared with the native scorer; a fresh
machine needs only exported GITLAB_TOKEN and the public dataset download.
"""

import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from benchmarks.mcore_source import locked_mcore_source
from resources_servers.vlm_eval_kit.official_vstar import MCORE_COMMIT, verify_vstar_source


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = REPO_ROOT / "resources_servers" / "vlm_eval_kit"
SOURCE_DIR = REPO_ROOT / "results" / "oracle-upstream" / f"VLMEvalKitMcore-{MCORE_COMMIT}"
SOURCE_URL = (
    "https://gitlab-master.nvidia.com/api/v4/projects/"
    "matthieul%2FVLMEvalKitMcore/repository/archive.tar.gz?sha=" + MCORE_COMMIT
)
OUTPUT_FPATH = Path(__file__).resolve().parent / "data" / "vstar_benchmark.jsonl"


def _ensure_source() -> None:
    """Fetch the reviewed commit once and reject a mismatched checkout."""
    with locked_mcore_source(SOURCE_DIR):
        _ensure_source_unlocked()


def _ensure_source_unlocked() -> None:
    if not SOURCE_DIR.exists():
        token = os.environ.get("GITLAB_TOKEN")
        if not token:
            raise RuntimeError("GITLAB_TOKEN is not set in this shell; restart the session.")
        SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a complete checkout. A failed fetch leaves no misleading
        # source directory that could be mistaken for prepared data next run.
        with tempfile.TemporaryDirectory(prefix=".vstar-source-", dir=SOURCE_DIR.parent) as temp_dir:
            archive_path = Path(temp_dir) / "source.tar.gz"
            request = Request(SOURCE_URL, headers={"PRIVATE-TOKEN": token})
            with urlopen(request, timeout=120) as response, archive_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(temp_dir, filter="data")
            checkout = next(path for path in Path(temp_dir).iterdir() if path.is_dir())
            verify_vstar_source(checkout)
            (checkout / ".source-commit").write_text(MCORE_COMMIT + "\n")
            checkout.rename(SOURCE_DIR)

    actual_commit = (SOURCE_DIR / ".source-commit").read_text().strip()
    if actual_commit != MCORE_COMMIT:
        raise RuntimeError(f"VStarBench source commit mismatch: {actual_commit}")
    verify_vstar_source(SOURCE_DIR)


def prepare() -> Path:
    """Write the exact VStarBench static-MCQ rows consumed by NeMo Gym."""
    _ensure_source()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(SOURCE_DIR), str(REPO_ROOT)])
    helper = f"from benchmarks.vstar.prepare_data import prepare_vstar; prepare_vstar({str(OUTPUT_FPATH)!r})"
    subprocess.run(
        ["uv", "run", "--locked", "--project", str(SERVER_DIR), "python", "-c", helper],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    return OUTPUT_FPATH


if __name__ == "__main__":
    print(prepare())
