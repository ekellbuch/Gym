# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize MathVision's published answers and images as NeMo Gym rows.

Adapted from Gym cfafe99c83c7::prepare_MathVision. The dataset and prompt builder
come from the exact VLMEvalKitMcore revision pinned by prepare.py.
"""

from pathlib import Path

import orjson


def prepare_mathvision(output_path: str | Path) -> Path:
    from vlmeval.dataset.image_vqa import MathVision

    dataset = MathVision(dataset="MathVision")
    data = dataset.load_data("MathVision")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as output:
        for _, row in data.iterrows():
            messages = dataset.build_prompt(row)
            text = "\n".join(message["value"] for message in messages if message["type"] == "text")
            choices = row.get("choices")
            if choices is None or (isinstance(choices, float) and choices != choices):
                choices = "[]"
            index = row["index"]
            try:
                index = int(index)
            except (TypeError, ValueError):
                index = str(index)
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
                                {"type": "input_text", "text": text},
                            ],
                        }
                    ]
                },
                "answer": row["answer"],
                "category": row["category"],
                "benchmark_name": "MathVision",
                "index": index,
                "question": row["question"],
                "choices": str(choices),
            }
            output.write(orjson.dumps(gym_row) + b"\n")
    return output_path
