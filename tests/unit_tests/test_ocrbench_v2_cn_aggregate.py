# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check when the official OCRBench_v2 Chinese aggregate may be named."""

import json
import sys
import types

import pytest

from benchmarks.ocrbench_v2 import aggregate_cn


GROUPS = {
    "cn_text_recognition": "full-page OCR cn",
    "cn_relationship_extraction": "key information extraction cn",
    "cn_element_parsing": "document parsing cn",
    "cn_visual_text_understanding": "cognition VQA cn",
    "cn_knowledge_reasoning": "reasoning VQA cn",
}


@pytest.mark.parametrize("selected", [2600, 3])
def test_chinese_overall_score_requires_complete_full_split(tmp_path, monkeypatch, selected):
    monkeypatch.setattr(aggregate_cn, "verify_ocrbench_v2_source", lambda _: None)
    official = types.ModuleType("vlmeval.dataset.utils.ocrbrnch_v2_eval")

    def official_aggregate(items):
        means = {}
        for group, category in GROUPS.items():
            scores = [item["score"] for item in items if item["type"] == category]
            if scores:
                means[group] = sum(scores) / len(scores)
        return {}, means

    official.ocrbench_v2_aggregate_accuracy = official_aggregate
    for package in ("vlmeval", "vlmeval.dataset", "vlmeval.dataset.utils"):
        monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
    monkeypatch.setitem(sys.modules, official.__name__, official)

    source = tmp_path / "benchmark.jsonl"
    with source.open("w") as output:
        for _ in range(2600):
            output.write("{}\n")
    results = tmp_path / "results.jsonl"
    categories = list(GROUPS.values())
    with results.open("w") as output:
        for index in range(selected):
            output.write(json.dumps({"category": categories[index % len(categories)], "reward": 1.0}) + "\n")
    results.with_suffix(".summary.json").write_text(
        json.dumps({"complete": True, "selected": selected, "input_path": str(source)})
    )

    result = aggregate_cn.aggregate_cn(results)

    assert result["full_split"] is (selected == 2600)
    assert result["selected"] == selected
    assert result["split_size"] == 2600
    if selected == 2600:
        assert result["Chinese Overall Score"] == 1.0
        assert "partial_group_mean" not in result
        assert set(result["group_means"]) == set(GROUPS)
    else:
        assert "Chinese Overall Score" not in result
        assert result["partial_group_mean"] == 1.0
