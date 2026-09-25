# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare language splits of OCRBench_v2 using published annotations."""

import ast
import base64
from pathlib import Path

import orjson


IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def _content_items(segments: list[dict]) -> list[dict]:
    """Preserve the pinned Gym converter's image/text segment order."""
    items = []
    for segment in segments:
        if segment["type"] == "image":
            path = Path(segment["value"])
            image = base64.b64encode(path.read_bytes()).decode("ascii")
            mime = IMAGE_MIME.get(path.suffix.lower(), "image/jpeg")
            items.append({"type": "input_image", "image_url": f"data:{mime};base64,{image}", "detail": "high"})
        elif segment["type"] == "text":
            if segment["value"] != "":
                items.append({"type": "input_text", "text": segment["value"]})
        else:
            raise ValueError(f"Unsupported OCRBench_v2 prompt segment: {segment['type']!r}")
    return items


def _published_reference(answers: list | str, *, index: int) -> str:
    """Transport one literal published answer; preserve the full list for grading."""
    if isinstance(answers, str):
        reference = answers
    elif isinstance(answers, list) and answers:
        reference = answers[0] if isinstance(answers[0], str) else repr(answers[0])
    else:
        raise ValueError(f"OCRBench_v2 row {index}: missing published answer list")
    if not reference.strip():
        raise ValueError(f"OCRBench_v2 row {index}: first published answer is empty")
    return reference


def prepare_ocrbench_v2_language(output_fpath: str, language: str) -> Path:
    if language not in {"en", "cn"}:
        raise ValueError(f"Unsupported OCRBench_v2 language: {language!r}")
    from vlmeval.dataset.image_vqa import OCRBench_v2

    dataset = OCRBench_v2(dataset="OCRBench_v2")
    output_path = Path(output_fpath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = 0
    with output_path.open("wb") as output:
        for _, row in dataset.data.iterrows():
            category = row["category"]
            if not category.endswith(f" {language}"):
                continue
            index = int(row["index"])
            answers = ast.literal_eval(row["answer"])
            bbox_raw = row["bbox"]
            content_raw = row["content"]
            # The official scorer ignores Chinese translation rows without gold text.
            if category == "text translation cn" and answers == [""]:
                continue
            gym_row = {
                "responses_create_params": {
                    "input": [{"role": "user", "content": _content_items(dataset.build_prompt(row))}]
                },
                "answer": answers,
                "oracle_reference": _published_reference(answers, index=index),
                "category": category,
                "benchmark_name": "OCRBench_v2",
                "index": index,
                "question": row["question"],
                "bbox": ast.literal_eval(bbox_raw) if bbox_raw != "without bbox" else bbox_raw,
                "content": ast.literal_eval(content_raw) if content_raw != "without content" else content_raw,
            }
            if row["eval"] != "without eval":
                gym_row["eval"] = row["eval"]
            output.write(orjson.dumps(gym_row) + b"\n")
            selected += 1
    if not selected:
        raise ValueError(f"Official OCRBench_v2 dataset contains no {language} rows")
    return output_path


def prepare_ocrbench_v2_en(output_fpath: str) -> Path:
    return prepare_ocrbench_v2_language(output_fpath, "en")


def prepare_ocrbench_v2_cn(output_fpath: str) -> Path:
    return prepare_ocrbench_v2_language(output_fpath, "cn")
