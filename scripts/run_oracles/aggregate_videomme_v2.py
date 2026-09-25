# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rate scored Video-MME-v2 oracle rows with the pinned official group scorer."""

import argparse
import hashlib
import json
from pathlib import Path

from resources_servers.vlm_eval_kit.official_videomme_v2 import (
    MCORE_COMMIT,
    load_official_scorer,
    official_group_rating,
)


def aggregate(input_path: Path, output_path: Path) -> dict:
    payload = input_path.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line]
    if not rows or len(rows) % 4:
        raise ValueError("Video-MME-v2 requires complete four-question groups")
    ordered = sorted(rows, key=lambda row: int(row["index"]))
    if [int(row["index"]) for row in ordered] != list(range(len(ordered))):
        raise ValueError("Video-MME-v2 scored rows must have contiguous official indices")
    for offset in range(0, len(ordered), 4):
        group = ordered[offset : offset + 4]
        if any(len({row[key] for row in group}) != 1 for key in ("video", "group_type", "group_structure")):
            raise ValueError(f"Video-MME-v2 group at index {offset} mixes official video metadata")
    if any(row.get("failure_kind") or row.get("failure_reason") for row in ordered):
        raise ValueError("Video-MME-v2 scored rows contain verification failures")
    if any(row.get("score") not in (-1, 0, 1) or "prediction" not in row for row in ordered):
        raise ValueError("Video-MME-v2 scored rows lack official predictions or scores")

    root = Path(__file__).resolve().parents[2]
    source = root / "results" / "oracle-upstream" / f"VLMEvalKitMcore-{MCORE_COMMIT}"
    scorer = load_official_scorer(source)
    result = {
        "input": str(input_path),
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "row_count": len(ordered),
        "group_count": len(ordered) // 4,
        "official_rating": official_group_rating(ordered, scorer),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.input, args.output)
    print(json.dumps({key: result[key] for key in ("input_sha256", "row_count", "group_count")}, indent=2))
    print(json.dumps(result["official_rating"]["final_rating"], indent=2))


if __name__ == "__main__":
    main()
