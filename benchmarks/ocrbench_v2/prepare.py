# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare OCRBench_v2 language splits from the exact pinned mcore fork."""

import argparse
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from benchmarks.mcore_source import locked_mcore_source
from resources_servers.vlm_eval_kit.official_ocrbench_v2 import MCORE_COMMIT, verify_ocrbench_v2_source


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = REPO_ROOT / "resources_servers" / "vlm_eval_kit"
SOURCE_DIR = REPO_ROOT / "results" / "oracle-upstream" / f"VLMEvalKitMcore-{MCORE_COMMIT}"
SOURCE_URL = (
    "https://gitlab-master.nvidia.com/api/v4/projects/"
    "matthieul%2FVLMEvalKitMcore/repository/archive.tar.gz?sha=" + MCORE_COMMIT
)
DATA_DIR = Path(__file__).resolve().parent / "data"


def _ensure_source() -> None:
    with locked_mcore_source(SOURCE_DIR):
        _ensure_source_unlocked()


def _ensure_source_unlocked() -> None:
    if not SOURCE_DIR.exists():
        token = os.environ.get("GITLAB_TOKEN")
        if not token:
            raise RuntimeError("GITLAB_TOKEN is not set in this shell; restart the session.")
        SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".ocrbench-v2-source-", dir=SOURCE_DIR.parent) as temp_dir:
            archive_path = Path(temp_dir) / "source.tar.gz"
            request = Request(SOURCE_URL, headers={"PRIVATE-TOKEN": token})
            with urlopen(request, timeout=120) as response, archive_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(temp_dir, filter="data")
            checkout = next(path for path in Path(temp_dir).iterdir() if path.is_dir())
            verify_ocrbench_v2_source(checkout)
            (checkout / ".source-commit").write_text(MCORE_COMMIT + "\n")
            checkout.rename(SOURCE_DIR)

    actual_commit = (SOURCE_DIR / ".source-commit").read_text().strip()
    if actual_commit != MCORE_COMMIT:
        raise RuntimeError(f"OCRBench_v2 source commit mismatch: {actual_commit}")
    verify_ocrbench_v2_source(SOURCE_DIR)


def prepare(language: str = "en") -> Path:
    """Write one language's JSONL with full gold annotation and submission text."""
    if language not in {"en", "cn"}:
        raise ValueError(f"Unsupported OCRBench_v2 language: {language!r}")
    _ensure_source()
    output_path = DATA_DIR / f"ocrbench_v2_{language}_benchmark.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(SOURCE_DIR), str(REPO_ROOT)])
    helper = (
        "from benchmarks.ocrbench_v2.prepare_data import prepare_ocrbench_v2_language; "
        f"prepare_ocrbench_v2_language({str(output_path)!r}, {language!r})"
    )
    subprocess.run(
        ["uv", "run", "--locked", "--project", str(SERVER_DIR), "python", "-c", helper],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["en", "cn"], default="en")
    print(prepare(parser.parse_args().language))
