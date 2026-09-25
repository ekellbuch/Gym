# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize OCR Reasoning's published answer and reference reasoning chain.

Adapted from internal Gym cfafe99c83c7::prepare_OCR_Reasoning. The oracle runner
submits the dataset's `reasoning` field verbatim, without synthesizing an answer.
"""

from pathlib import Path

import orjson


def _none_if_nan(row, field: str):
    value = row.get(field)
    return None if value is None or (isinstance(value, float) and value != value) else value


def prepare_ocr_reasoning(output_path: str | Path) -> Path:
    from vlmeval.dataset.image_vqa import OCR_Reasoning

    dataset = OCR_Reasoning(dataset="OCR_Reasoning")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        for _, row in dataset.data.iterrows():
            messages = dataset.build_prompt(row)
            prompt = "\n".join(message["value"] for message in messages if message["type"] == "text")
            gym_row = {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{row['image']}",
                                    "detail": "high",
                                },
                                {"type": "input_text", "text": prompt},
                            ],
                        }
                    ]
                },
                "answer": row["answer"],
                "reasoning": row["reasoning"],
                "category": row["task"],
                "benchmark_name": "OCR_Reasoning",
                "index": int(row["index"]),
                "question": row["question"],
                "question_type": _none_if_nan(row, "question_type"),
                "answer_type": _none_if_nan(row, "answer_type"),
                "choices": _none_if_nan(row, "choices"),
                "answer_option": _none_if_nan(row, "answer_option"),
                "language": _none_if_nan(row, "language"),
                "format": _none_if_nan(row, "format"),
            }
            output.write(orjson.dumps(gym_row) + b"\n")
    return output_path
