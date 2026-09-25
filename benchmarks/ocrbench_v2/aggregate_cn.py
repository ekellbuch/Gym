# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply pinned OCRBench_v2's five-group Chinese aggregate to verified rows."""

import json
import sys
from pathlib import Path

from benchmarks.ocrbench_v2.prepare import SOURCE_DIR
from resources_servers.vlm_eval_kit.official_ocrbench_v2 import verify_ocrbench_v2_source


EXPECTED_CN_ROWS = 2600


def aggregate_cn(result_path: Path) -> dict:
    verify_ocrbench_v2_source(SOURCE_DIR)
    from vlmeval.dataset.utils.ocrbrnch_v2_eval import ocrbench_v2_aggregate_accuracy

    summary = json.loads(result_path.with_suffix(".summary.json").read_text())
    if not summary["complete"]:
        raise RuntimeError("OCRBench_v2 Chinese verification is incomplete; no aggregate is valid")
    items = []
    with result_path.open() as results:
        for line in results:
            if not line.strip():
                continue
            row = json.loads(line)
            category = row["category"]
            if not category.endswith(" cn"):
                raise ValueError(f"Non-Chinese OCRBench_v2 row: {category!r}")
            item = {"type": category, "score": row["reward"]}
            if row.get("ignore") == "True":
                item["ignore"] = "True"
            items.append(item)
    if len(items) != summary["selected"]:
        raise RuntimeError("OCRBench_v2 Chinese result count differs from the verification summary")
    _, group_means = ocrbench_v2_aggregate_accuracy(items)
    expected_groups = {
        "cn_text_recognition",
        "cn_relationship_extraction",
        "cn_element_parsing",
        "cn_visual_text_understanding",
        "cn_knowledge_reasoning",
    }
    with Path(summary["input_path"]).open() as source:
        split_size = sum(1 for line in source if line.strip())
    full_split = summary["selected"] == split_size == EXPECTED_CN_ROWS
    if full_split and set(group_means) != expected_groups:
        raise RuntimeError(f"Full Chinese split lacks official aggregate groups: {expected_groups - set(group_means)}")
    output = {
        "selected": summary["selected"],
        "split_size": split_size,
        "full_split": full_split,
        "group_means": group_means,
    }
    aggregate = sum(group_means.values()) / len(group_means) if group_means else None
    if full_split:
        output["Chinese Overall Score"] = aggregate
    else:
        output["partial_group_mean"] = aggregate
    output_path = result_path.with_suffix(".aggregate.json")
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    print(json.dumps(aggregate_cn(Path(sys.argv[1])), indent=2))
