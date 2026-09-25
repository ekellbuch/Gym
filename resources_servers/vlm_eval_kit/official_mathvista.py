# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MathVista_MINI scoring with the pinned VLMEvalKitMcore reference functions."""

import hashlib
import re
from asyncio import Semaphore, to_thread
from pathlib import Path
from typing import Any


MCORE_COMMIT = "6962c8d06b2b7b26a74a73d6212c06562b63e1b7"
MCORE_FILE_SHA256 = {
    "vlmeval/dataset/image_vqa.py": "47a46ef8b879d6acf15ac0d4ef8bb046b47fedba9b0532a80843101525b7330d",
    "vlmeval/dataset/utils/mathvista.py": "5e4f05fcd1e41477ea3a095c75b327bd575a963154480dcca34536047e71151a",
    "vlmeval/utils/matching_util.py": "03435cedaeaf6282599816aed4ba57f26dd4858f2a37399490acca98d9f8f5f2",
}
_THINK_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def verify_mathvista_source(source_dir: Path) -> None:
    """Reject a different dataset mapping or scorer than the reviewed pin."""
    for name, expected in MCORE_FILE_SHA256.items():
        actual = hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Pinned MathVista source hash mismatch: {source_dir / name}")


async def score_mathvista(
    body: Any,
    *,
    judge: Any = None,
    judge_semaphore: Semaphore | None = None,
) -> dict[str, float]:
    """Score one Gym response by the official MathVista prefetch/judge path."""
    prediction = _THINK_BLOCK_RE.sub("", body.response.output_text or "").strip()
    reward = 0.0
    if prediction:
        from vlmeval.dataset.utils.mathvista import MathVista_auxeval, post_check

        line = {
            "index": body.index,
            "question": body.question,
            "answer": body.answer,
            "question_type": body.question_type,
            "answer_type": body.answer_type,
            "answer_option": getattr(body, "answer_option", None),
            "choices": body.choices,
            "prediction": prediction,
        }
        if judge is None:
            reward = float(bool(post_check(line, prefetch=True)))
        else:
            if judge_semaphore is None:
                auxiliary = await to_thread(MathVista_auxeval, judge, line)
            else:
                async with judge_semaphore:
                    auxiliary = await to_thread(MathVista_auxeval, judge, line)
            line["res"] = auxiliary["res"]
            reward = float(bool(post_check(line, prefetch=False)))

    return {
        f"MathVista_MINI/{body.category}": reward,
        "MathVista_MINI": reward,
        "reward": reward,
    }
