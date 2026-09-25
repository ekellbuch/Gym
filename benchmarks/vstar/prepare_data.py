# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VStarBench row conversion from internal Gym cfafe99c, prepare_VStarBench.

The dataset, prompts, answer labels, and choice extraction come from pinned
VLMEvalKitMcore. This module only writes that official data in Gym's JSONL shape.
"""

from pathlib import Path

import orjson


def prepare_vstar(output_fpath: str) -> Path:
    from vlmeval.dataset.image_mcq import ImageMCQDataset
    from vlmeval.dataset.utils.multiple_choice import build_choices

    dataset_name = "VStarBench"
    dataset = ImageMCQDataset(dataset=dataset_name)
    data = dataset.load_data(dataset_name)

    output_path = Path(output_fpath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        for _, row in data.iterrows():
            messages = dataset.build_prompt(row)
            text = "\n".join(message["value"] for message in messages if message["type"] == "text")
            category = row.get("category")
            if category is None or (isinstance(category, float) and category != category):
                category = "overall"
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
                "category": category,
                "benchmark_name": dataset_name,
                "choices": build_choices(row),
                "index": int(row["index"]),
                "question": row["question"],
            }
            output.write(orjson.dumps(gym_row) + b"\n")
    return output_path
