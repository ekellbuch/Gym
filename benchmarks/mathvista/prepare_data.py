# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write MathVista_MINI rows in the historical Gym cfafe99c format."""

from pathlib import Path

import orjson


def prepare_mathvista(output_fpath: str) -> Path:
    from vlmeval.dataset.image_vqa import MathVista

    dataset_name = "MathVista_MINI"
    dataset = MathVista(dataset=dataset_name)
    data = dataset.load_data(dataset_name)

    def cell(row, key):
        value = row.get(key)
        return None if value is None or (isinstance(value, float) and value != value) else value

    output_path = Path(output_fpath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        for _, row in data.iterrows():
            messages = dataset.build_prompt(row)
            text = "\n".join(message["value"] for message in messages if message["type"] == "text")
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
                "category": row["task"],
                "benchmark_name": dataset_name,
                "index": int(row["index"]),
                "question": row["question"],
                "question_type": row["question_type"],
                "answer_type": row["answer_type"],
                "answer_option": cell(row, "answer_option"),
                "choices": cell(row, "choices"),
            }
            output.write(orjson.dumps(gym_row) + b"\n")
    return output_path
