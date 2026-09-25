# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare official BabyVision references through the pinned Mcore loader.

The authors publish ``babyvision_data.zip`` at the pinned BabyVision revision.
Mcore's ``build_babyvision_tsv`` turns its ``blankAns``/``choiceAns`` fields into
the answers submitted to NeMo Gym; no answers are synthesized here.
"""

import hashlib
import os
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from benchmarks.mcore_source import locked_mcore_source
from resources_servers.vlm_eval_kit.official_babyvision import MCORE_COMMIT, verify_babyvision_source


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = REPO_ROOT / "resources_servers" / "vlm_eval_kit"
UPSTREAM_DIR = REPO_ROOT / "results" / "oracle-upstream"
MCORE_DIR = UPSTREAM_DIR / f"VLMEvalKitMcore-{MCORE_COMMIT}"
DATA_COMMIT = "7f92fd4b1dc1c68b7b936a9bc09c68b4a944a55a"
DATA_ZIP = UPSTREAM_DIR / f"babyvision_data-{DATA_COMMIT[:8]}.zip"
DATA_DIR = UPSTREAM_DIR / f"BabyVision-data-{DATA_COMMIT}"
DATA_URL = f"https://raw.githubusercontent.com/UniPat-AI/BabyVision/{DATA_COMMIT}/data/babyvision_data.zip"
DATA_SHA256 = "e5e16e7104b080bac7e67bb98d74bd5057f7151f487cd2f61991106253751e54"
MCORE_URL = (
    "https://gitlab-master.nvidia.com/api/v4/projects/"
    "matthieul%2FVLMEvalKitMcore/repository/archive.tar.gz?sha=" + MCORE_COMMIT
)
OUTPUT_FPATH = Path(__file__).resolve().parent / "data" / "babyvision_benchmark.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_mcore() -> None:
    with locked_mcore_source(MCORE_DIR):
        _ensure_mcore_unlocked()


def _ensure_mcore_unlocked() -> None:
    if not MCORE_DIR.exists():
        token = os.environ.get("GITLAB_TOKEN")
        if not token:
            raise RuntimeError("GITLAB_TOKEN is not set in this shell; restart the session.")
        UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".babyvision-mcore-", dir=UPSTREAM_DIR) as temp_dir:
            archive_path = Path(temp_dir) / "source.tar.gz"
            request = Request(MCORE_URL, headers={"PRIVATE-TOKEN": token})
            with urlopen(request, timeout=120) as response, archive_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(temp_dir, filter="data")
            checkout = next(path for path in Path(temp_dir).iterdir() if path.is_dir())
            verify_babyvision_source(checkout)
            (checkout / ".source-commit").write_text(MCORE_COMMIT + "\n")
            checkout.rename(MCORE_DIR)
    if (MCORE_DIR / ".source-commit").read_text().strip() != MCORE_COMMIT:
        raise RuntimeError("BabyVision source commit mismatch")
    verify_babyvision_source(MCORE_DIR)


def _ensure_data() -> Path:
    UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_ZIP.exists():
        with tempfile.NamedTemporaryFile(prefix=".babyvision-data-", dir=UPSTREAM_DIR, delete=False) as output:
            partial = Path(output.name)
            try:
                with urlopen(DATA_URL, timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if _sha256(partial) != DATA_SHA256:
                    raise RuntimeError("Official BabyVision dataset ZIP hash mismatch")
                partial.rename(DATA_ZIP)
            finally:
                partial.unlink(missing_ok=True)
    if _sha256(DATA_ZIP) != DATA_SHA256:
        raise RuntimeError("Official BabyVision dataset ZIP hash mismatch")

    source_root = DATA_DIR / "babyvision_data"
    if not source_root.exists():
        with tempfile.TemporaryDirectory(prefix=".babyvision-extract-", dir=UPSTREAM_DIR) as temp_dir:
            with zipfile.ZipFile(DATA_ZIP) as archive:
                archive.extractall(temp_dir)
            extracted = Path(temp_dir) / "babyvision_data"
            if not extracted.exists():
                raise RuntimeError("Official BabyVision archive lacks babyvision_data/")
            Path(temp_dir).rename(DATA_DIR)
    meta = source_root / "meta_data.jsonl"
    rows = [line for line in meta.read_text().splitlines() if line.strip()]
    if len(rows) != 388:
        raise RuntimeError(f"Expected 388 official BabyVision rows, got {len(rows)}")
    return source_root


def prepare() -> Path:
    _ensure_mcore()
    source_root = _ensure_data()
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    lmu_data = OUTPUT_FPATH.parent / "LMUData"
    lmu_data.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["BABYVISION_DATA_ROOT"] = str(source_root)
    env["LMUData"] = str(lmu_data)
    env["PYTHONPATH"] = os.pathsep.join([str(MCORE_DIR), str(REPO_ROOT)])
    helper = (
        f"from benchmarks.babyvision.prepare_data import prepare_BabyVision; prepare_BabyVision({str(OUTPUT_FPATH)!r})"
    )
    subprocess.run(
        ["uv", "run", "--locked", "--project", str(SERVER_DIR), "python", "-c", helper],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    return OUTPUT_FPATH


if __name__ == "__main__":
    print(prepare())
