# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize published MMLongBench-Doc references and PDF page images.

Adapted from internal Gym cfafe99c83c7::prepare_MMLongBench_DOC. Its reference
run used 100 rendered pages, not the upstream class's 40-page default.
"""

import base64
from pathlib import Path

import orjson


PROMPT_PREFIX = (
    "Extract information from all the given images, then answer the question using a single word "
    "or phrase. Return 'Not answerable' if the answer cannot be derived from the the images.\n"
)


def _data_url(path: str) -> str:
    source = Path(path)
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
        source.suffix.lower(), "image/jpeg"
    )
    return f"data:{mime};base64,{base64.b64encode(source.read_bytes()).decode('ascii')}"


def _none_if_nan(row, field: str):
    value = row.get(field)
    return None if value is None or (isinstance(value, float) and value != value) else value


def prepare_mmlongbench_doc(output_path: str | Path, max_pages: int = 100) -> Path:
    from vlmeval.dataset.mmlongbench import MMLongBench

    dataset = MMLongBench(dataset="MMLongBench_DOC", model="mcore", max_pages=max_pages)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        for _, row in dataset.data.iterrows():
            page_images = [message for message in dataset.build_prompt(row) if message["type"] == "image"]
            content = [{"type": "input_text", "text": PROMPT_PREFIX + row["question"]}]
            content.extend(
                {"type": "input_image", "image_url": _data_url(page["value"]), "detail": "high"}
                for page in page_images
            )
            gym_row = {
                "responses_create_params": {"input": [{"role": "user", "content": content}]},
                "answer": row["answer"],
                "answer_format": row["answer_format"],
                "category": _none_if_nan(row, "doc_type") or "overall",
                "benchmark_name": "MMLongBench_DOC",
                "index": int(row["index"]),
                "question": row["question"],
                "doc_id": _none_if_nan(row, "doc_id"),
                "evidence_pages": _none_if_nan(row, "evidence_pages"),
                "evidence_sources": _none_if_nan(row, "evidence_sources"),
            }
            output.write(orjson.dumps(gym_row) + b"\n")
    return output_path
