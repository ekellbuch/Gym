# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MMLongBench-Doc scoring through pinned VLMEvalKitMcore, including headline F1."""

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any


MCORE_COMMIT = "6962c8d06b2b7b26a74a73d6212c06562b63e1b7"
MMLONGBENCH_SHA256 = "bb9a228ec3d0c6e116e4401d7dfe4dc1b07259410b23ae21b47f128813d957e9"


def load_official_scorer(source_dir: str | Path):
    """Verify the exact published scorer checkout before importing its functions."""
    source_dir = Path(source_dir).resolve()
    if (source_dir / ".source-commit").read_text().strip() != MCORE_COMMIT:
        raise RuntimeError("MMLongBench-Doc scorer revision mismatch")
    source_file = source_dir / "vlmeval" / "dataset" / "mmlongbench.py"
    if hashlib.sha256(source_file.read_bytes()).hexdigest() != MMLONGBENCH_SHA256:
        raise RuntimeError("MMLongBench-Doc scorer source hash mismatch")
    sys.path.insert(0, str(source_dir))
    module = importlib.import_module("vlmeval.dataset.mmlongbench")
    if Path(module.__file__).resolve() != source_file:
        raise RuntimeError(f"MMLongBench-Doc imported wrong scorer source: {module.__file__}")
    return module


def score_mmlongbench_reference(line: dict[str, Any], judge, official_module) -> dict[str, Any]:
    """Run the official typed-answer extraction and per-row scoring functions."""
    if judge is None:
        raise RuntimeError("MMLongBench-Doc requires its configured gpt-4o extraction judge")
    aux = official_module.MMLongBench_auxeval(judge, {"question": line["question"], "prediction": line["prediction"]})
    pred = aux["pred"]
    try:
        reward = float(official_module.eval_score(line["answer"], pred, line["answer_format"]))
    except Exception:
        # MMLongBench_acc uses the same per-row exception envelope.
        reward = 0.0
    # The official extractor can return a nonempty judge reply with no
    # "Extracted answer:" marker. That is an invalid extraction, not a scored
    # wrong response from a published oracle answer.
    return {"pred": pred, "reward": reward, "invalid_judge_response": not bool(aux.get("res")) or not bool(pred)}


def mmlongbench_f1(rows: list[dict[str, Any]]) -> float:
    """Exact dict-based form of pinned mmlongbench.py::get_f1."""
    gt_pos = [row for row in rows if row["answer"] != "Not answerable"]
    pred_pos = [row for row in rows if row["pred"] != "Not answerable"]
    recall = sum(row["reward"] for row in gt_pos) / len(gt_pos) if gt_pos else 0.0
    precision = sum(row["reward"] for row in pred_pos) / len(pred_pos) if pred_pos else 0.0
    return 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
