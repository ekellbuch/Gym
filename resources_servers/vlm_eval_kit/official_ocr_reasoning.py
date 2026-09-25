# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OCR Reasoning verifier using the pinned mcore two-stage judge and post_check."""

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any


MCORE_COMMIT = "6962c8d06b2b7b26a74a73d6212c06562b63e1b7"
MCORE_HASHES = {
    "vlmeval/dataset/image_vqa.py": "47a46ef8b879d6acf15ac0d4ef8bb046b47fedba9b0532a80843101525b7330d",
    "vlmeval/dataset/utils/ocr_reasoning.py": "e4fa835a637fca884b3948d5b0706413b58e03d7c8e0e0cd7935bd5011f40fa3",
}


def load_official_scorer(source_dir: str | Path):
    source_dir = Path(source_dir).resolve()
    if (source_dir / ".source-commit").read_text().strip() != MCORE_COMMIT:
        raise RuntimeError("OCR Reasoning scorer revision mismatch")
    for relative_path, expected_hash in MCORE_HASHES.items():
        if hashlib.sha256((source_dir / relative_path).read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"OCR Reasoning scorer hash mismatch: {relative_path}")
    sys.path.insert(0, str(source_dir))
    scorer = importlib.import_module("vlmeval.dataset.utils.ocr_reasoning")
    expected_path = source_dir / "vlmeval/dataset/utils/ocr_reasoning.py"
    if Path(scorer.__file__).resolve() != expected_path:
        raise RuntimeError(f"OCR Reasoning imported wrong scorer source: {scorer.__file__}")
    return scorer


def score_ocr_reasoning_reference(line: dict[str, Any], judge, scorer) -> dict[str, Any]:
    """Run OcrR_auxeval (reason rating, then answer extraction) and post_check."""
    if judge is None:
        raise RuntimeError("OCR Reasoning requires the configured gpt-4o-mini judge")
    candidate = dict(line)
    try:
        aux = scorer.OcrR_auxeval(judge, candidate)
        candidate["res"] = aux["res"]
        reward = float(bool(scorer.post_check(candidate, prefetch=False)))
        return {
            "reward": reward,
            "reason_score": float(aux["reason_score"]),
            "invalid_judge_response": aux["log"].endswith("All 5 retries failed.\n"),
        }
    except Exception as error:
        # The pinned auxeval raises when none of six judge replies contains [[n]].
        # Preserve that failure separately from an incorrect benchmark answer.
        return {"reward": 0.0, "reason_score": 0.0, "invalid_judge_response": True, "judge_error": str(error)}
