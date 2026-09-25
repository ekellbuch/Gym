# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submit official SWE-bench gold patches in bounded batches to Gym's upstream runner."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


RUNNER = Path("resources_servers/swebench/apply_golden_patch.py")
UPSTREAM_OUTPUT = Path("temp2.jsonl")
BATCH_SIZE = 8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids(path: Path) -> list[str]:
    return [json.loads(line)["instance_id"] for line in path.read_text().splitlines() if line]


def _all_verifiers_completed(path: Path) -> bool:
    return all(json.loads(line).get("evaluation_completed") is True for line in path.read_text().splitlines() if line)


def _preserve(path: Path, label: str) -> None:
    attempt = 1
    while (destination := path.with_name(f"evidence_{path.stem}_{label}_{attempt}{path.suffix}")).exists():
        attempt += 1
    os.replace(path, destination)


def run(*, input_path: Path, count: int, output: Path, head_port: int) -> None:
    lines = [line for line in input_path.read_text().splitlines(keepends=True) if line.strip()][:count]
    selected_rows = [json.loads(line) for line in lines]
    selected_ids = [row["instance_id"] for row in selected_rows]
    if len(lines) != count or len(set(selected_ids)) != count:
        raise ValueError(f"Expected {count} distinct official SWE-bench tasks")
    if any(not row.get("patch") for row in selected_rows):
        raise ValueError("Selected SWE-bench tasks lack official gold patches")
    batch_dir = output.with_suffix(".batches")
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = batch_dir / "manifest.json"
    expected_manifest = {
        "input_sha256": _sha256(input_path),
        "runner_sha256": _sha256(RUNNER),
        "count": count,
        "batch_size": BATCH_SIZE,
    }
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected_manifest:
            raise ValueError(f"Existing batch manifest disagrees with current inputs: {manifest}")
    else:
        if any(batch_dir.iterdir()):
            raise ValueError(f"Batch directory lacks its manifest: {batch_dir}")
        manifest_temp = batch_dir / "manifest.tmp"
        manifest_temp.write_text(json.dumps(expected_manifest, sort_keys=True) + "\n")
        os.replace(manifest_temp, manifest)

    upstream_output = batch_dir / UPSTREAM_OUTPUT
    if upstream_output.exists():
        raise FileExistsError(f"Refusing to overwrite the upstream runner's {upstream_output}")

    batch_outputs = []
    for offset in range(0, count, BATCH_SIZE):
        batch_number = offset // BATCH_SIZE
        batch_input = batch_dir / f"batch_{batch_number:03d}.input.jsonl"
        batch_output = batch_dir / f"batch_{batch_number:03d}.jsonl"
        batch_ids = selected_ids[offset : offset + BATCH_SIZE]
        batch_input.write_text("".join(lines[offset : offset + BATCH_SIZE]))
        if batch_output.exists():
            if sorted(_ids(batch_output)) != sorted(batch_ids):
                raise ValueError(f"Existing batch has wrong task IDs: {batch_output}")
            if _all_verifiers_completed(batch_output):
                print(f"==> Reusing verified batch {batch_number + 1}", flush=True)
            else:
                # A transport or sandbox failure is not an oracle score. Keep its
                # evidence and rerun the identical official rows through Gym.
                _preserve(batch_output, "incomplete")
                print(f"==> Retrying incomplete official batch {batch_number + 1}", flush=True)
        if not batch_output.exists():
            print(
                f"==> Verifying official patches, batch {batch_number + 1}/{(count + BATCH_SIZE - 1) // BATCH_SIZE}",
                flush=True,
            )
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(RUNNER.resolve()),
                        f"+benchmark_jsonl={batch_input.resolve()}",
                        f"+head_server.port={head_port}",
                    ],
                    cwd=batch_dir.resolve(),
                    check=True,
                )
                if sorted(_ids(upstream_output)) != sorted(batch_ids):
                    raise ValueError(f"Upstream runner returned wrong task IDs for batch {batch_number + 1}")
                if not _all_verifiers_completed(upstream_output):
                    raise ValueError(f"Upstream verifier did not complete every task in batch {batch_number + 1}")
                os.replace(upstream_output, batch_output)
            except BaseException:
                # Preserve an interrupted upstream output without letting the next batch reuse it.
                if upstream_output.exists():
                    _preserve(upstream_output, f"batch_{batch_number:03d}.partial")
                raise
        batch_outputs.append(batch_output)

    combined = output.with_suffix(".tmp.jsonl")
    with combined.open("w") as destination:
        for batch_output in batch_outputs:
            destination.write(batch_output.read_text())
    if sorted(_ids(combined)) != sorted(selected_ids):
        raise ValueError("Combined results do not match the selected official tasks")
    os.replace(combined, output)
    print(f"==> {count} official verifier results: {output}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("Usage: swebench_gold_batches.py INPUT_JSONL TASK_COUNT OUTPUT_JSONL HEAD_PORT")
    run(input_path=Path(sys.argv[1]), count=int(sys.argv[2]), output=Path(sys.argv[3]), head_port=int(sys.argv[4]))
