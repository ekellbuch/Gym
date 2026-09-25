# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Official pinned Video-MME-v2 option scoring and four-question group rating."""

import hashlib
import importlib
import sys
import tempfile
from pathlib import Path
from typing import Any


MCORE_COMMIT = "6962c8d06b2b7b26a74a73d6212c06562b63e1b7"
MCORE_HASHES = {
    "vlmeval/dataset/videomme_v2.py": "e73d23be6e6b4f661b741c103416a0d3c75faee9ecd349482e515a731507fb61",
    "vlmeval/dataset/utils/videomme_v2.py": "0bc88b0c48ea80be768ab8ab5c6d72d23dd6e863a97141de54268c01be6b7c70",
}


def load_official_scorer(source_dir: str | Path):
    source_dir = Path(source_dir).resolve()
    if (source_dir / ".source-commit").read_text().strip() != MCORE_COMMIT:
        raise RuntimeError("Video-MME-v2 scorer revision mismatch")
    for relative_path, expected_hash in MCORE_HASHES.items():
        if hashlib.sha256((source_dir / relative_path).read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Video-MME-v2 scorer hash mismatch: {relative_path}")
    sys.path.insert(0, str(source_dir))
    scorer = importlib.import_module("vlmeval.dataset.utils.videomme_v2")
    expected_path = source_dir / "vlmeval/dataset/utils/videomme_v2.py"
    if Path(scorer.__file__).resolve() != expected_path:
        raise RuntimeError(f"Video-MME-v2 imported wrong scorer source: {scorer.__file__}")
    return scorer


def score_videomme_v2_reference(answer: str, prediction: str, scorer) -> dict[str, float | int]:
    """Match a published answer letter with the official option extractor."""
    extracted = scorer.extract_characters_regex_v2(prediction) if prediction else ""
    score = int(extracted == str(answer)) if extracted else -1
    return {"score": score, "reward": float(max(score, 0))}


def official_group_rating(rows: list[dict[str, Any]], scorer) -> dict[str, Any]:
    """Pass ordered scored rows to the pinned file-based get_final_rating_v2."""
    if not rows or len(rows) % 4:
        raise ValueError("Video-MME-v2 rating requires complete four-question groups")
    import pandas as pd

    ordered = sorted(rows, key=lambda row: int(row["index"]))
    indices = [int(row["index"]) for row in ordered]
    if indices != list(range(len(rows))):
        raise ValueError("Video-MME-v2 rating requires all contiguous official rows")
    columns = ["answer", "prediction", "score", "level", "group_type", "group_structure", "second_head", "third_head"]
    with tempfile.TemporaryDirectory(prefix="videomme-v2-rating-") as temp_dir:
        path = Path(temp_dir) / "scored.tsv"
        pd.DataFrame([{key: row.get(key) for key in columns} for row in ordered]).to_csv(path, sep="\t", index=False)
        return scorer.get_final_rating_v2(str(path))
