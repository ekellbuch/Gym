# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin and call the official VLMEvalKitMcore BabyVision judge scorer."""

import hashlib
from asyncio import to_thread
from pathlib import Path
from typing import Any


MCORE_COMMIT = "6962c8d06b2b7b26a74a73d6212c06562b63e1b7"
BABYVISION_CLASS_SHA256 = "1ffc08289c701d9c8735bf79b44d2689fe3fd69fc1feff820126e22760082c58"
BABYVISION_SCORER_SHA256 = "128696d6dd3fd04bb1337e530f2c3103c8a1fc20077f8bc49478203ff2e56870"


def _check_hash(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"BabyVision source hash mismatch: {path}: {actual}")


def verify_babyvision_source(source_dir: Path) -> None:
    _check_hash(source_dir / "vlmeval/dataset/babyvision.py", BABYVISION_CLASS_SHA256)
    _check_hash(source_dir / "vlmeval/dataset/utils/babyvision.py", BABYVISION_SCORER_SHA256)


async def score_babyvision(body: Any, judge: Any, judge_semaphore: Any) -> dict[str, Any]:
    """Delegate the verdict to pinned ``BabyVision_auxeval`` without a scoring fallback.

    The official function returns ``res=False`` after five failed judge attempts.
    Flag that transport/parsing failure for the oracle runner so it cannot be
    mistaken for a genuinely incorrect reference answer.
    """
    from vlmeval.dataset.utils import babyvision as official

    _check_hash(Path(official.__file__), BABYVISION_SCORER_SHA256)
    category = body.category
    ans_type = getattr(body, "ans_type", None)
    metrics: dict[str, Any] = {}
    if judge is None:
        invalid = True
        reward = 0.0
    else:
        line = {
            "question": body.question,
            "answer": body.answer,
            "prediction": body.response.output_text or "",
        }
        try:
            async with judge_semaphore:
                verdict = await to_thread(official.BabyVision_auxeval, judge, line)
            invalid = "All retries failed." in verdict["log"]
            reward = float(bool(verdict["res"]))
        except Exception:
            invalid = True
            reward = 0.0
    metrics[f"BabyVision/{category}"] = reward
    metrics["BabyVision"] = reward
    metrics["reward"] = reward
    if ans_type:
        metrics[f"BabyVision/ans_type/{ans_type}"] = reward
    if invalid:
        metrics["invalid_judge_response"] = True
    return metrics
