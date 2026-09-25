# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BabyVision preparation ported from Gym cfafe99c83c712033dc6b8105bbe7a409c592574.

The source dataset and scorer are the pinned BabyVision and VLMEvalKitMcore
revisions fetched by ``prepare.py``. Answer selection remains in Mcore's
``build_babyvision_tsv``; this module does not generate reference answers.
"""

import base64
import hashlib
from pathlib import Path

import orjson
from pandas import DataFrame


_IMAGE_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def image_file_to_data_url(fpath: str) -> str:
    mime = _IMAGE_MIME_BY_SUFFIX.get(Path(fpath).suffix.lower(), "image/jpeg")
    with open(fpath, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode('ascii')}"


def segments_to_content_items(segments, detail: str = "high") -> list:
    items = []
    for seg in segments:
        if seg["type"] == "image":
            items.append(
                {
                    "type": "input_image",
                    "image_url": image_file_to_data_url(seg["value"]),
                    "detail": detail,
                }
            )
        elif seg["type"] == "text":
            if seg["value"] != "":
                items.append({"type": "input_text", "text": seg["value"]})
        else:
            raise ValueError(f"Unsupported segment type: {seg['type']!r}")
    return items


def prepare_BabyVision(output_fpath: str) -> Path:
    """Build the Gym JSONL through the pinned Mcore BabyVision class."""
    from vlmeval.dataset import babyvision as official

    from resources_servers.vlm_eval_kit.official_babyvision import BABYVISION_CLASS_SHA256

    imported_hash = hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest()
    if imported_hash != BABYVISION_CLASS_SHA256:
        raise RuntimeError(f"Imported BabyVision loader differs from pinned source: {official.__file__}")
    BabyVision = official.BabyVision

    dataset_name = "BabyVision"
    dataset = BabyVision(dataset=dataset_name)
    data: DataFrame = dataset.data
    if len(data) != 388:
        raise ValueError(f"Expected 388 official BabyVision rows, got {len(data)}")

    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    def _index(row):
        value = row["index"]
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)

    with open(output_fpath, "wb") as f:
        for _, vlmevalkit_row in data.iterrows():
            messages = dataset.build_prompt(vlmevalkit_row)
            content = segments_to_content_items(messages)
            gym_row = {
                "responses_create_params": {"input": [{"role": "user", "content": content}]},
                "answer": str(vlmevalkit_row["answer"]),
                "category": str(vlmevalkit_row["category"]),
                "benchmark_name": dataset_name,
                "index": _index(vlmevalkit_row),
                "question": vlmevalkit_row["question"],
                "l2_category": str(vlmevalkit_row["l2_category"]),
                "ans_type": str(vlmevalkit_row["ans_type"]),
            }
            f.write(orjson.dumps(gym_row) + b"\n")

    return output_fpath
