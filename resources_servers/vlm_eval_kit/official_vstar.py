# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned VLMEvalKitMcore VStarBench (static MCQ) scoring adapter.

This calls the official choice matcher and judge fallback. Source files were
fetched from VLMEvalKitMcore@6962c8d06b2b7b26a74a73d6212c06562b63e1b7;
their hashes also match the local 6e98cbf snapshot byte for byte.
"""

import hashlib
import re
from asyncio import Semaphore, to_thread
from pathlib import Path
from typing import Any


MCORE_COMMIT = "6962c8d06b2b7b26a74a73d6212c06562b63e1b7"
MCORE_FILE_SHA256 = {
    "vlmeval/dataset/image_mcq.py": "10c619337426a976d57e610b65574944bd9da474c5c51fd167def4474ad8a230",
    "vlmeval/dataset/utils/multiple_choice.py": "2e7994ee7def7a71b3b1892dd7c70f83fc386195471b45f449f84efca7a1f24e",
    "vlmeval/utils/matching_util.py": "03435cedaeaf6282599816aed4ba57f26dd4858f2a37399490acca98d9f8f5f2",
}
_THINK_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def verify_vstar_source(source_dir: Path) -> None:
    """Fail if the dataset mapping or official MCQ scorer differs from the pin."""
    for name, expected in MCORE_FILE_SHA256.items():
        actual = hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Pinned VStarBench source hash mismatch: {source_dir / name}")


async def score_vstar(
    body: Any,
    *,
    judge: Any = None,
    judge_semaphore: Semaphore | None = None,
) -> dict[str, float | bool]:
    """Score one Gym response with mcore's exact MCQ matcher and fallback."""
    prediction = _THINK_BLOCK_RE.sub("", body.response.output_text or "").strip()
    invalid_judge_response = False
    if not prediction:
        reward = 0.0
    elif judge is None:
        from vlmeval.utils.matching_util import can_infer

        reward = float(can_infer(prediction, body.choices) == body.answer)
    else:
        from vlmeval.dataset.utils.multiple_choice import extract_answer_from_item

        item = {
            "question": body.question,
            "prediction": prediction,
            "answer": body.answer,
            **body.choices,
        }
        if judge_semaphore is None:
            result = await to_thread(extract_answer_from_item, judge, item)
        else:
            async with judge_semaphore:
                result = await to_thread(extract_answer_from_item, judge, item)
        # The pinned matcher randomly selects an option after three failed judge
        # calls. Preserve its result, but never count that random fallback as an
        # oracle score, even when it happens to match the published answer.
        invalid_judge_response = result.get("log") == "Failed to predict, thus randomly generate one. "
        reward = float(result["opt"] == body.answer)
    return {
        f"VStarBench/{body.category}": reward,
        "VStarBench": reward,
        "reward": reward,
        "invalid_judge_response": invalid_judge_response,
    }
