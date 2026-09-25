# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select OCRBench_v2 English rows with complete published response strings."""

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from benchmarks.ocrbench_v2.prepare import DATA_DIR, prepare


SOURCE_PATH = DATA_DIR / "ocrbench_v2_en_benchmark.jsonl"
SUBSET_PATH = DATA_DIR / "ocrbench_v2_en_reference_subset_benchmark.jsonl"

# The five omitted categories require a structured prediction absent from answer[0]
# or cannot score that literal prediction under the pinned official evaluator.
EXPECTED_COUNTS = {
    "APP agent en": 300,
    "ASCII art classification en": 200,
    "VQA with position en": 300,
    "chart parsing en": 400,
    "cognition VQA en": 800,
    "diagram QA en": 300,
    "document classification en": 200,
    "document parsing en": 400,
    "fine-grained text recognition en": 200,
    "formula recognition en": 400,
    "full-page OCR en": 200,
    "key information extraction en": 400,
    "key information mapping en": 300,
    "math QA en": 300,
    "reasoning VQA en": 600,
    "science QA en": 300,
    "table parsing en": 400,
    "text counting en": 200,
    "text grounding en": 200,
    "text recognition en": 800,
    "text spotting en": 200,
}
EXCLUDED_CATEGORIES = frozenset(
    {
        "VQA with position en",
        "key information extraction en",
        "key information mapping en",
        "text grounding en",
        "text spotting en",
    }
)


def filter_reference_subset(
    source_path: Path, output_path: Path, *, expected_counts: dict[str, int] = EXPECTED_COUNTS
) -> Counter[str]:
    """Copy exact source rows, checking the source variant before publishing output."""
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Reference subset must not replace the complete EN dataset")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_path.parent, prefix=".ocrbench-en-subset-", delete=False
    ) as tmp:
        temp_path = Path(tmp.name)
        try:
            with source_path.open("rb") as source:
                for line_number, line in enumerate(source, start=1):
                    row = json.loads(line)
                    category = row["category"]
                    counts[category] += 1
                    if category not in EXCLUDED_CATEGORIES:
                        if not isinstance(row.get("oracle_reference"), str) or not row["oracle_reference"].strip():
                            raise ValueError(f"Line {line_number}: missing published reference")
                        tmp.write(line)
                        selected[category] += 1
            if counts != expected_counts:
                raise ValueError(f"OCRBench_v2 EN source category counts changed: {counts!r}")
            os.replace(temp_path, output_path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    return selected


def prepare_subset() -> Path:
    prepare("en")
    counts = filter_reference_subset(SOURCE_PATH, SUBSET_PATH)
    print(f"OCRBench_v2 EN published-reference subset: {sum(counts.values())} of {sum(EXPECTED_COUNTS.values())} rows")
    return SUBSET_PATH


if __name__ == "__main__":
    print(prepare_subset())
