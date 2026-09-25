# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native OCRBench_v2 per-item scoring from pinned VLMEvalKitMcore."""

import hashlib
import re
from asyncio import Lock, to_thread
from pathlib import Path
from typing import Any


MCORE_COMMIT = "6962c8d06b2b7b26a74a73d6212c06562b63e1b7"
MCORE_FILE_SHA256 = {
    "vlmeval/dataset/image_vqa.py": "47a46ef8b879d6acf15ac0d4ef8bb046b47fedba9b0532a80843101525b7330d",
    "vlmeval/dataset/utils/ocrbrnch_v2_eval.py": "86b9687594646d8efdee048ee837c7415b6e61280dc18758e62ba6effe449ffa",
}
_THINK_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_SPOTTING_LOCK = Lock()


def verify_ocrbench_v2_source(source_dir: Path) -> None:
    """Reject a different dataset mapping or scoring dispatcher than the pin."""
    for name, expected in MCORE_FILE_SHA256.items():
        actual = hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Pinned OCRBench_v2 source hash mismatch: {source_dir / name}")


async def score_ocrbench_v2(body: Any) -> dict[str, Any]:
    """Call the official dispatcher with the same item shape as evaluate()."""
    prediction = _THINK_BLOCK_RE.sub("", body.response.output_text or "").strip()
    category = body.category
    if not category.endswith((" en", " cn")):
        raise ValueError(f"Expected OCRBench_v2 language category, got {category!r}")
    if not prediction:
        return {f"OCRBench_v2/{category}": 0.0, "OCRBench_v2": 0.0, "reward": 0.0}

    from vlmeval.dataset.utils.ocrbrnch_v2_eval import process_predictions

    item = {
        "type": category,
        "question": body.question,
        "predict": prediction,
        "answers": body.answer,
        "bbox": getattr(body, "bbox", "without bbox"),
        "content": getattr(body, "content", "without content"),
    }
    eval_method = getattr(body, "eval", None)
    if eval_method is not None and eval_method != "without eval":
        item["eval"] = eval_method

    # The pinned spotting scorer uses fixed scratch paths and is not thread-safe.
    if category == "text spotting en":
        async with _SPOTTING_LOCK:
            scored = await to_thread(process_predictions, [item])
    else:
        scored = await to_thread(process_predictions, [item])
    result = scored[0]
    reward = float(result["score"])
    metrics = {f"OCRBench_v2/{category}": reward, "OCRBench_v2": reward, "reward": reward}
    if result.get("ignore") == "True":
        metrics["ignore"] = "True"
    return metrics
