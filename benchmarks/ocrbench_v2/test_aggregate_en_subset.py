# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from benchmarks.ocrbench_v2.aggregate_en_subset import EXPECTED_GROUPS, aggregate_en_subset


@pytest.fixture
def verified_subset(tmp_path):
    source = tmp_path / "subset.jsonl"
    rows = [{"category": "APP agent en"}, {"category": "chart parsing en"}]
    source.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    result = tmp_path / "results.jsonl"
    result.write_text(
        "".join(
            json.dumps({"category": row["category"], "reward": 1.0, "_oracle_reference": {"row_index": i}}) + "\n"
            for i, row in enumerate(rows)
        )
    )
    summary = {
        "complete": True,
        "selected": 2,
        "verified": 2,
        "invalid": 0,
        "errors": 0,
        "input_path": str(source),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mean_reward": 1.0,
    }
    result.with_suffix(".summary.json").write_text(json.dumps(summary))
    return result, summary


def test_aggregate_labels_subset_and_preserves_official_full_score_as_unknown(verified_subset):
    result, _ = verified_subset
    seen = []

    def pinned_stub(rows):
        seen.extend(rows)
        return {group: 1.0 for group in EXPECTED_GROUPS}, {}

    report = aggregate_en_subset(
        result, expected_counts={"APP agent en": 1, "chart parsing en": 1}, official_aggregate=pinned_stub
    )

    assert len(seen) == 2
    assert report["selected_five_group_mean"] == 1.0
    assert report["row_mean"] == 1.0
    assert report["official_full_english_overall_score"] is None
    assert set(report["excluded_full_benchmark_groups"]) == {
        "en_relationship_extraction",
        "en_text_detection",
        "en_text_spotting",
    }
    assert json.loads(result.with_suffix(".aggregate.json").read_text()) == report


def test_aggregate_rejects_duplicate_result_rows(verified_subset):
    result, _ = verified_subset
    rows = [json.loads(line) for line in result.read_text().splitlines()]
    rows[1]["_oracle_reference"]["row_index"] = 0
    result.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="Duplicate or invalid"):
        aggregate_en_subset(result, expected_counts={"APP agent en": 1, "chart parsing en": 1})

    assert not result.with_suffix(".aggregate.json").exists()


def test_aggregate_rejects_incomplete_summary(verified_subset):
    result, summary = verified_subset
    summary["errors"] = 1
    result.with_suffix(".summary.json").write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="incomplete or invalid"):
        aggregate_en_subset(result, expected_counts={"APP agent en": 1, "chart parsing en": 1})

    assert not result.with_suffix(".aggregate.json").exists()
