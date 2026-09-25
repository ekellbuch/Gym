# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare CharXiv RQ from the pinned official VLMEvalKitMcore source.

The shell entrypoint fetches and verifies the source archive. This module uses
the current Gym resource-server environment and the official preparation
functions ported verbatim from internal Gym 4718d1c5bbbba3a143be60849810da5fc6ab631b.
"""

import hashlib
import os
import subprocess
from pathlib import Path

import yaml


BENCHMARK_DIR = Path(__file__).parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
SERVER_DIR = REPO_ROOT / "resources_servers" / "vlm_eval_kit"
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "charxiv_rq_benchmark.jsonl"

# The source hash must match the scorer verified by the current Gym server.
MCORE_VLMEVALKIT_COMMIT = "6e98cbf20bd88a6469a1a298f67ec266196a4bbf"
MCORE_DIR = REPO_ROOT / "results" / "oracle-upstream" / f"VLMEvalKitMcore-{MCORE_VLMEVALKIT_COMMIT}"
MCORE_CHARXIV_SHA256 = "6348576f7f21c799ccc3954be44e74bc702c1c4afe9543bc0d1ad4cfc12e75da"


def _charxiv_opt_kwargs() -> dict:
    """Read the opt-in CharXiv !108 eval-optimization knobs from this benchmark's config.

    Defaults reproduce current behavior (no extended prompt, no image resize)."""
    cfg = yaml.safe_load((BENCHMARK_DIR / "config.yaml").read_text())
    rs = cfg["charxiv_rq_benchmark_resources_server"]["resources_servers"]["vlm_eval_kit"]
    return {
        "extended_user_prompt": rs.get("extended_user_prompt", "none"),
        "image_resize_scale": rs.get("image_resize_scale", 1.0),
        "image_resize_resample": rs.get("image_resize_resample", "lanczos"),
        "image_resize_jpeg_quality": rs.get("image_resize_jpeg_quality", 95),
    }


def _ensure_server_venv() -> Path:
    """Create the current Gym server venv and verify the pinned official scorer."""
    venv_python = SERVER_DIR / ".venv" / "bin" / "python"
    source_hash = hashlib.sha256((MCORE_DIR / "vlmeval/dataset/charxiv.py").read_bytes()).hexdigest()
    if source_hash != MCORE_CHARXIV_SHA256:
        raise RuntimeError("Pinned VLMEvalKitMcore CharXiv source hash mismatch")
    subprocess.run(["uv", "sync", "--project", str(SERVER_DIR), "--locked", "--python", "3.13.14"], check=True)
    return venv_python


def prepare() -> Path:
    """Build the CharXiv_reasoning_val benchmark JSONL and return its path."""
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    venv_python = _ensure_server_venv()

    opt_kwargs = _charxiv_opt_kwargs()
    kwargs_src = ", ".join(f"{k}={v!r}" for k, v in opt_kwargs.items())
    helper = (
        "from benchmarks.charxiv_rq.prepare_data import prepare_CharXiv_reasoning_val; "
        f"prepare_CharXiv_reasoning_val({str(OUTPUT_FPATH)!r}, {kwargs_src})"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(MCORE_DIR), str(REPO_ROOT)])
    subprocess.run([str(venv_python), "-c", helper], check=True, cwd=REPO_ROOT, env=env)

    print(f"Wrote CharXiv_reasoning_val benchmark data to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
