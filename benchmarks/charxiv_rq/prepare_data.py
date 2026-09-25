# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verbatim CharXiv preparation functions from internal Gym 4718d1c5bbbba3a143be60849810da5fc6ab631b.
import base64
import os
from pathlib import Path

import orjson
from pandas import DataFrame


def _nan_to_none(row, key):
    value = row.get(key)
    return None if value is None or (isinstance(value, float) and value != value) else value


def _prepare_CharXiv(
    dataset_name: str,
    mode: str,
    output_fpath: str,
    *,
    extended_user_prompt: str = "none",
    image_resize_scale: float = 1.0,
    image_resize_resample: str = "lanczos",
    image_resize_jpeg_quality: int = 95,
) -> Path:
    """Build the Gym JSONL for one CharXiv split (both share the mcore ``CharXiv`` class).

    Carries everything the reference scorer (vlmeval/dataset/charxiv.py) needs:
    ``grading_query`` — the per-row judge prompt with a ``{PREDICTION}`` placeholder that
    ``auxeval`` formats (the ground truth is embedded in it) — plus index/question. The
    RESOLVED category string is written per row via ``qid2category(mode)`` (charxiv.py:61):
    descriptive maps the ``qid`` column, reasoning maps ``inst_category``; the raw column
    value is also kept for official-parity reconstruction.
    Nano 3 Omni paper targets (arXiv 2604.24954): DQ 88.9 / RQ 63.6.

    Opt-in CharXiv eval optimizations from VLMEvalKitMcore !108 are wired here because
    Gym imports the library and never runs its ``run.py``, so !108's CLI/env plumbing
    never fires. Both knobs default to a no-op (current behaviour):

    - ``extended_user_prompt`` selects the ``VLMEVAL_EXTENDED_USER_PROMPT`` preset the
      bumped ``CharXiv.build_prompt`` prepends to the question (``"none"`` = unchanged,
      ``"interpolation"`` = the anti-interpolation instruction). The env var is set BEFORE
      ``build_prompt`` so the text lands in the prepared JSONL.
    - ``image_resize_scale`` (with ``image_resize_resample`` / ``image_resize_jpeg_quality``)
      drives ``VLMEVAL_IMAGE_RESIZE_*``, which the bumped ``CharXiv.dump_image`` uses to
      resize the on-disk image. When scale != 1.0 the RESIZED file is base64-embedded; at
      scale 1.0 ``resize_image`` is a no-op and we embed the raw ``image`` column
      byte-for-byte, so the default output is identical to before.
    """
    from vlmeval.dataset.charxiv import CharXiv, qid2category

    # Set BEFORE build_prompt / dump_image so the bumped CharXiv reads them per row.
    os.environ["VLMEVAL_EXTENDED_USER_PROMPT"] = str(extended_user_prompt)
    os.environ["VLMEVAL_IMAGE_RESIZE_SCALE"] = str(image_resize_scale)
    os.environ["VLMEVAL_IMAGE_RESIZE_RESAMPLE"] = str(image_resize_resample)
    os.environ["VLMEVAL_IMAGE_RESIZE_JPEG_QUALITY"] = str(image_resize_jpeg_quality)
    resize_enabled = abs(float(image_resize_scale) - 1.0) > 1e-9

    dataset = CharXiv(dataset=dataset_name)
    data: DataFrame = dataset.data

    print(f"Columns: {data.columns}")
    print(data.head())

    category_map, index_col = qid2category(mode)

    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_fpath, "wb") as f:
        for _, vlmevalkit_row in data.iterrows():
            messages = dataset.build_prompt(vlmevalkit_row)
            # CharXiv is single-image; the only text segment is the question.
            text = "\n".join(m["value"] for m in messages if m["type"] == "text")

            if resize_enabled:
                # build_prompt's image segment now points at the RESIZED on-disk file
                # (CharXiv.dump_image -> resize_image); embed that file's bytes.
                image_path = next(m["value"] for m in messages if m["type"] == "image")
                with open(image_path, "rb") as image_fh:
                    image_b64 = base64.b64encode(image_fh.read()).decode("ascii")
            else:
                # Default: embed the raw base64 image column unchanged (byte-identical).
                image_b64 = vlmevalkit_row["image"]

            qid = int(vlmevalkit_row[index_col])
            gym_row = {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high",
                                },
                                {
                                    "type": "input_text",
                                    "text": text,
                                },
                            ],
                        },
                    ]
                },
                # The scoring ground truth lives inside grading_query; the raw answer
                # column (when present) is only carried for the shared verify schema.
                "answer": _nan_to_none(vlmevalkit_row, "answer"),
                "category": category_map[qid],
                "benchmark_name": dataset_name,
                "grading_query": vlmevalkit_row["grading_query"],
                "index": int(vlmevalkit_row["index"]),
                "question": vlmevalkit_row["question"],
                index_col: qid,
            }
            f.write(orjson.dumps(gym_row) + b"\n")

    return output_fpath


def prepare_CharXiv_reasoning_val(
    output_fpath: str = "data/CharXiv_reasoning_val_validation.jsonl", **charxiv_opt_kwargs
) -> Path:
    return _prepare_CharXiv("CharXiv_reasoning_val", "reasoning", output_fpath, **charxiv_opt_kwargs)
