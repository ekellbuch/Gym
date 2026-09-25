# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared Mcore cache must survive simultaneous fresh-machine prepares."""

import hashlib
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_concurrent_mcore_acquisition_downloads_once(tmp_path: Path) -> None:
    source_bytes = b"# pinned scorer fixture\n"
    second_source_bytes = b"# another pinned scorer fixture\n"
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("mcore/vlmeval/dataset/utils/mathv.py")
        info.size = len(source_bytes)
        archive.addfile(info, io.BytesIO(source_bytes))
        info = tarfile.TarInfo("mcore/vlmeval/dataset/mmlongbench.py")
        info.size = len(second_source_bytes)
        archive.addfile(info, io.BytesIO(second_source_bytes))
    archive_path = tmp_path / "source.tar.gz"
    archive_path.write_bytes(archive_buffer.getvalue())
    fetch_log = tmp_path / "fetches.log"
    checkout = tmp_path / "VLMEvalKitMcore-testcommit"
    code = (
        "import importlib, os, time\n"
        "from pathlib import Path\n"
        "p = importlib.import_module('benchmarks.' + os.environ['MCORE_TEST_BENCHMARK'] + '.prepare')\n"
        "p.MCORE_DIR = Path(os.environ['MCORE_TEST_CHECKOUT'])\n"
        "p.MCORE_URL = os.environ['MCORE_TEST_URL']\n"
        "p.MCORE_COMMIT = 'testcommit'\n"
        "if hasattr(p, 'MATHV_SHA256'):\n"
        "    p.MATHV_SHA256 = os.environ['MCORE_TEST_SHA256']\n"
        "else:\n"
        "    p.MMLONGBENCH_SHA256 = os.environ['MCORE_TEST_SECOND_SHA256']\n"
        "original_urlopen = p.urlopen\n"
        "def counted_urlopen(request, timeout):\n"
        "    with open(os.environ['MCORE_TEST_FETCH_LOG'], 'a') as log:\n"
        "        log.write('fetch\\n')\n"
        "    time.sleep(0.15)\n"
        "    return original_urlopen(request, timeout=timeout)\n"
        "p.urlopen = counted_urlopen\n"
        "p.ensure_mcore_checkout()\n"
    )
    env = os.environ.copy()
    env.update(
        GITLAB_TOKEN="test-token",
        MCORE_TEST_CHECKOUT=str(checkout),
        MCORE_TEST_URL=archive_path.as_uri(),
        MCORE_TEST_FETCH_LOG=str(fetch_log),
        MCORE_TEST_SHA256=hashlib.sha256(source_bytes).hexdigest(),
        MCORE_TEST_SECOND_SHA256=hashlib.sha256(second_source_bytes).hexdigest(),
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]),
    )
    workers = []
    for benchmark in ("mathvision", "mmlongbench_doc") * 3:
        worker_env = {**env, "MCORE_TEST_BENCHMARK": benchmark}
        workers.append(subprocess.Popen([sys.executable, "-c", code], cwd=REPO_ROOT, env=worker_env))
    for worker in workers:
        assert worker.wait(timeout=30) == 0
    assert fetch_log.read_text().splitlines() == ["fetch"]
    assert (checkout / ".source-commit").read_text().strip() == "testcommit"
    assert (checkout / "vlmeval/dataset/utils/mathv.py").read_bytes() == source_bytes
    assert (checkout / "vlmeval/dataset/mmlongbench.py").read_bytes() == second_source_bytes

    # A warm but corrupted cache is rejected rather than silently trusted.
    (checkout / "vlmeval/dataset/utils/mathv.py").write_bytes(b"corrupt")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**env, "MCORE_TEST_BENCHMARK": "mathvision"},
        capture_output=True,
    )
    assert result.returncode != 0
    assert b"source hash mismatch" in result.stderr
    assert fetch_log.read_text().splitlines() == ["fetch"]

    (checkout / "vlmeval/dataset/utils/mathv.py").write_bytes(source_bytes)
    (checkout / ".source-commit").write_text("wrong-commit\n")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**env, "MCORE_TEST_BENCHMARK": "mmlongbench_doc"},
        capture_output=True,
    )
    assert result.returncode != 0
    assert b"source commit mismatch" in result.stderr
    assert fetch_log.read_text().splitlines() == ["fetch"]
