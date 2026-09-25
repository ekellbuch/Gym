# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.ocrbench_v2.prepare_reference_subset import (
    EXCLUDED_CATEGORIES,
    EXPECTED_COUNTS,
    filter_reference_subset,
)


def test_reference_subset_preserves_complete_rows_and_excludes_five_categories(tmp_path):
    source = tmp_path / "full.jsonl"
    subset = tmp_path / "subset.jsonl"
    rows = [
        {"category": category, "oracle_reference": f"published-{category}", "answer": [f"published-{category}"]}
        for category in EXPECTED_COUNTS
    ]
    original = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    source.write_bytes(original)

    counts = filter_reference_subset(source, subset, expected_counts=dict.fromkeys(EXPECTED_COUNTS, 1))

    assert len(EXCLUDED_CATEGORIES) == 5
    assert sum(EXPECTED_COUNTS.values()) == 7400
    assert sum(count for category, count in EXPECTED_COUNTS.items() if category not in EXCLUDED_CATEGORIES) == 6000
    assert source.read_bytes() == original
    assert counts == {category: 1 for category in EXPECTED_COUNTS if category not in EXCLUDED_CATEGORIES}
    assert subset.read_bytes() == b"".join(
        json.dumps(row).encode() + b"\n" for row in rows if row["category"] not in EXCLUDED_CATEGORIES
    )


def test_reference_subset_rejects_changed_source_without_replacing_output(tmp_path):
    source = tmp_path / "full.jsonl"
    subset = tmp_path / "subset.jsonl"
    source.write_text('{"category":"APP agent en","oracle_reference":"gold"}\n')
    subset.write_text("previous valid subset\n")

    with pytest.raises(ValueError, match="source category counts changed"):
        filter_reference_subset(source, subset)

    assert subset.read_text() == "previous valid subset\n"


def test_reference_subset_rejects_missing_published_response(tmp_path):
    source = tmp_path / "full.jsonl"
    subset = tmp_path / "subset.jsonl"
    source.write_text('{"category":"APP agent en","oracle_reference":""}\n')

    with pytest.raises(ValueError, match="missing published reference"):
        filter_reference_subset(source, subset, expected_counts={"APP agent en": 1})

    assert not subset.exists()
