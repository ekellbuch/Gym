# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report pinned OCRBench_v2 group means for the verified English reference subset."""

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

from benchmarks.ocrbench_v2.prepare import SOURCE_DIR
from benchmarks.ocrbench_v2.prepare_reference_subset import EXCLUDED_CATEGORIES, EXPECTED_COUNTS
from resources_servers.vlm_eval_kit.official_ocrbench_v2 import verify_ocrbench_v2_source


EXPECTED_SUBSET_COUNTS = {
    category: count for category, count in EXPECTED_COUNTS.items() if category not in EXCLUDED_CATEGORIES
}
EXPECTED_GROUPS = frozenset(
    {
        "en_text_recognition",
        "en_element_parsing",
        "en_mathematical_calculation",
        "en_visual_text_understanding",
        "en_knowledge_reasoning",
    }
)
MISSING_FULL_GROUPS = (
    "en_relationship_extraction",
    "en_text_detection",
    "en_text_spotting",
)


def aggregate_en_subset(
    result_path: Path,
    *,
    expected_counts: dict[str, int] = EXPECTED_SUBSET_COUNTS,
    official_aggregate: Callable | None = None,
) -> dict:
    """Reject partial/mismatched results before applying the pinned group scorer."""
    summary = json.loads(result_path.with_suffix(".summary.json").read_text())
    expected_total = sum(expected_counts.values())
    if not summary.get("complete") or any(
        summary.get(key) != value
        for key, value in {"selected": expected_total, "verified": expected_total, "invalid": 0, "errors": 0}.items()
    ):
        raise ValueError("English reference subset verification is incomplete or invalid")

    source_path = Path(summary["input_path"])
    with source_path.open("rb") as source:
        source_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if source_hash != summary["input_sha256"]:
        raise ValueError("Prepared subset differs from verified input hash")
    source_categories = []
    source_counts: Counter[str] = Counter()
    with source_path.open("rb") as source:
        for line in source:
            category = json.loads(line)["category"]
            source_categories.append(category)
            source_counts[category] += 1
    if source_counts != expected_counts:
        raise ValueError(f"English reference subset category counts changed: {source_counts!r}")

    scored = []
    seen = set()
    reward_sum = 0.0
    with result_path.open("rb") as results:
        for line in results:
            row = json.loads(line)
            row_index = row["_oracle_reference"]["row_index"]
            if not isinstance(row_index, int) or isinstance(row_index, bool) or row_index in seen:
                raise ValueError(f"Duplicate or invalid result row index: {row_index!r}")
            if not 0 <= row_index < expected_total or row.get("category") != source_categories[row_index]:
                raise ValueError(f"Result row {row_index} does not match the prepared subset")
            seen.add(row_index)
            reward = row.get("reward")
            if (
                row.get("error")
                or row.get("ignore")
                or isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not math.isfinite(reward)
            ):
                raise ValueError(f"Result row {row_index} is not a valid scored reference")
            scored.append({"type": row["category"], "score": reward})
            reward_sum += reward
    if len(seen) != expected_total:
        raise ValueError(f"Missing verified reference rows: {expected_total - len(seen)}")
    row_mean = reward_sum / expected_total
    if not math.isclose(row_mean, summary["mean_reward"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Result rewards do not match the verification summary")

    if official_aggregate is None:
        verify_ocrbench_v2_source(SOURCE_DIR)
        sys.path.insert(0, str(SOURCE_DIR))
        from vlmeval.dataset.utils.ocrbrnch_v2_eval import ocrbench_v2_aggregate_accuracy

        official_aggregate = ocrbench_v2_aggregate_accuracy
    en_groups, cn_groups = official_aggregate(scored)
    if cn_groups or set(en_groups) != EXPECTED_GROUPS:
        raise ValueError("Pinned OCRBench_v2 scorer did not return the five selected English groups")
    output = {
        "benchmark": "OCRBench_v2 English published-reference subset",
        "selected": expected_total,
        "group_means": en_groups,
        "selected_five_group_mean": sum(en_groups.values()) / len(en_groups),
        "row_mean": row_mean,
        "excluded_full_benchmark_groups": list(MISSING_FULL_GROUPS),
        "official_full_english_overall_score": None,
    }
    result_path.with_suffix(".aggregate.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    print(json.dumps(aggregate_en_subset(Path(sys.argv[1])), indent=2))
