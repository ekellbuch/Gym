# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize Video-MME-v2's official option letters and group metadata.

Adapted from internal Gym cfafe99c83c7::prepare_VideoMME_v2 using its default
video_url transport and the without-subtitles track.
"""

import ast
from pathlib import Path
from unittest.mock import patch

import orjson


HF_REPO_ID = "MME-Benchmarks/Video-MME-v2"
HF_REVISION = "6e4bebb03202e1ddbf3d37703e560e51c5aa2d64"


def _none_if_nan(row, field: str):
    value = row.get(field)
    return None if value is None or (isinstance(value, float) and value != value) else value


def prompt_text(question: str, candidates: str) -> str:
    options = ast.literal_eval(candidates)
    letters = [chr(ord("A") + index) for index in range(len(options))]
    if len(letters) > 1:
        letter_list = ", ".join(letters[:-1]) + ", or " + letters[-1]
    else:
        letter_list = letters[0]
    return (
        f"{question}\n"
        + "\n".join(options)
        + f"\nRespond with only the letter ({letter_list}) of the correct option.\nAnswer: "
    )


def prepare_videomme_v2(output_path: str | Path) -> Path:
    from vlmeval.dataset import videomme_v2 as official

    if official.modelscope_flag_set():
        raise RuntimeError("Video-MME-v2 requires Hugging Face for its pinned dataset revision")
    original_download = official.snapshot_download
    snapshot_path = original_download(repo_id=HF_REPO_ID, repo_type="dataset", revision=HF_REVISION)
    if Path(snapshot_path).name != HF_REVISION:
        raise RuntimeError(f"Video-MME-v2 downloaded an unexpected snapshot: {snapshot_path}")

    def pinned_download(*, repo_id: str, repo_type: str) -> str:
        if repo_id != HF_REPO_ID or repo_type != "dataset":
            raise ValueError(f"Unexpected Video-MME-v2 dataset source: {repo_id}")
        return snapshot_path

    # The pinned upstream loader otherwise prefers the mutable cached main ref.
    with (
        patch.object(official, "get_cache_path", return_value=snapshot_path),
        patch.object(official, "snapshot_download", side_effect=pinned_download),
    ):
        dataset = official.VideoMMEv2(dataset="Video-MME-v2", use_subtitle=False, nframe=0, fps=2.0, nframe_max=256)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        for _, row in dataset.data.iterrows():
            level = _none_if_nan(row, "level")
            video_path = str(row["video_path"])
            if not (Path(dataset.data_root) / video_path).is_file():
                raise FileNotFoundError(f"Official Video-MME-v2 media is missing: {video_path}")
            gym_row = {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt_text(row["question"], str(row["candidates"]))}
                            ],
                        }
                    ],
                    "metadata": {"video_path": video_path},
                },
                "answer": str(row["answer"]),
                "category": "None" if level is None else f"level_{int(level)}",
                "benchmark_name": "Video_MME_v2",
                "index": int(row["index"]),
                "question": row["question"],
                "level": level,
                "group_type": row["group_type"],
                "group_structure": str(row["group_structure"]),
                "second_head": _none_if_nan(row, "second_head"),
                "third_head": _none_if_nan(row, "third_head"),
                "question_id": _none_if_nan(row, "question_id"),
                "video": row["video"],
                "candidates": str(row["candidates"]),
            }
            output.write(orjson.dumps(gym_row) + b"\n")
    return output_path
