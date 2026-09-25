# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.graphwalks import prepare as graphwalks_prepare


def test_prepare_uses_pinned_source_and_published_nodes(monkeypatch, tmp_path):
    entries = [
        {
            "prompt": "Find the parents of node a.",
            "answer_nodes": ["b", "a"],
            "problem_type": "parents",
            "prompt_chars": 27,
        },
        {
            "prompt": "Perform a BFS from node a with depth 2",
            "answer_nodes": ["c", "b"],
            "problem_type": "bfs",
            "prompt_chars": 38,
        },
    ]

    def load_dataset(name, *, split, revision):
        assert (name, split, revision) == ("openai/graphwalks", "train", graphwalks_prepare.DATASET_REVISION)
        return entries

    monkeypatch.setattr(graphwalks_prepare, "load_dataset", load_dataset)
    monkeypatch.setattr(graphwalks_prepare, "_build_token_counter", lambda _: lambda text: len(text))
    monkeypatch.setattr(graphwalks_prepare, "EXPECTED_SOURCE_ROWS", 2)
    output = tmp_path / "graphwalks.jsonl"

    graphwalks_prepare.prepare(output_fpath=output)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [json.loads(row["expected_answer"]) for row in rows] == [["b"], ["b", "c"]]
    assert "only the nodes at exactly depth 2" in rows[1]["responses_create_params"]["input"][0]["content"]


def test_prepare_rejects_unexpected_source_count_before_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(graphwalks_prepare, "load_dataset", lambda *args, **kwargs: [])
    output = tmp_path / "graphwalks.jsonl"

    with pytest.raises(ValueError, match="expected 1150"):
        graphwalks_prepare.prepare(output_fpath=output)

    assert not output.exists()
