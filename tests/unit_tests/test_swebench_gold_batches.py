# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for bounded submission of official SWE-bench patches."""

import json
from pathlib import Path

import pytest
from scripts.run_oracles import swebench_gold_batches


def test_completed_batches_resume_without_resubmitting_official_patches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed run reuses validated batches and submits each official task once."""
    input_path = tmp_path / "official.jsonl"
    input_path.write_text(
        "".join(json.dumps({"instance_id": str(i), "patch": "official diff"}) + "\n" for i in range(9))
    )
    runner = tmp_path / "upstream.py"
    runner.write_text("# upstream runner revision\n")
    other_run_output = tmp_path / "temp2.jsonl"
    other_run_output.write_text("another Gym run owns this file\n")
    monkeypatch.setattr(swebench_gold_batches, "RUNNER", runner)
    submitted = []

    def upstream(command: list[str], *, cwd: Path, check: bool) -> None:
        assert check is True
        assert command[3] == "+head_server.port=11132"
        batch_path = Path(command[2].split("=", 1)[1])
        rows = [json.loads(line) for line in batch_path.read_text().splitlines()]
        submitted.extend(row["instance_id"] for row in rows)
        (cwd / "temp2.jsonl").write_text(
            "".join(json.dumps(row | {"resolved": True, "evaluation_completed": True}) + "\n" for row in rows)
        )

    monkeypatch.setattr(swebench_gold_batches.subprocess, "run", upstream)
    output = tmp_path / "oracle.jsonl"
    swebench_gold_batches.run(input_path=input_path, count=9, output=output, head_port=11132)
    assert submitted == [str(i) for i in range(9)]
    assert len([json.loads(line) for line in output.read_text().splitlines()]) == 9
    assert other_run_output.read_text() == "another Gym run owns this file\n"

    submitted.clear()
    swebench_gold_batches.run(input_path=input_path, count=9, output=output, head_port=11132)
    assert submitted == []

    first_batch = output.with_suffix(".batches") / "batch_000.jsonl"
    saved_rows = [json.loads(line) for line in first_batch.read_text().splitlines()]
    saved_rows[0]["evaluation_completed"] = False
    first_batch.write_text("".join(json.dumps(row) + "\n" for row in saved_rows))
    swebench_gold_batches.run(input_path=input_path, count=9, output=output, head_port=11132)
    assert submitted == [str(i) for i in range(8)]
    assert len(list(first_batch.parent.glob("evidence_batch_000_incomplete_*.jsonl"))) == 1


def test_mismatched_batch_is_rejected_before_it_can_be_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupt saved batch cannot be included in a resumed oracle result."""
    input_path = tmp_path / "official.jsonl"
    input_path.write_text(json.dumps({"instance_id": "official", "patch": "official diff"}) + "\n")
    runner = tmp_path / "upstream.py"
    runner.write_text("# upstream runner revision\n")
    monkeypatch.setattr(swebench_gold_batches, "RUNNER", runner)

    def upstream(command: list[str], *, cwd: Path, check: bool) -> None:
        (cwd / "temp2.jsonl").write_text(
            json.dumps({"instance_id": "official", "resolved": True, "evaluation_completed": True}) + "\n"
        )

    monkeypatch.setattr(swebench_gold_batches.subprocess, "run", upstream)
    output = tmp_path / "oracle.jsonl"
    swebench_gold_batches.run(input_path=input_path, count=1, output=output, head_port=11133)
    output.with_suffix(".batches").joinpath("batch_000.jsonl").write_text(
        json.dumps({"instance_id": "other", "resolved": True}) + "\n"
    )
    with pytest.raises(ValueError, match="wrong task IDs"):
        swebench_gold_batches.run(input_path=input_path, count=1, output=output, head_port=11133)


def test_incomplete_verifier_result_is_preserved_and_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sandbox failure cannot be published as a completed official oracle result."""
    input_path = tmp_path / "official.jsonl"
    input_path.write_text(json.dumps({"instance_id": "official", "patch": "official diff"}) + "\n")
    runner = tmp_path / "upstream.py"
    runner.write_text("# upstream runner revision\n")
    monkeypatch.setattr(swebench_gold_batches, "RUNNER", runner)
    attempts = 0

    def upstream(command: list[str], *, cwd: Path, check: bool) -> None:
        nonlocal attempts
        attempts += 1
        (cwd / "temp2.jsonl").write_text(
            json.dumps({"instance_id": "official", "resolved": False, "evaluation_completed": attempts > 1}) + "\n"
        )

    monkeypatch.setattr(swebench_gold_batches.subprocess, "run", upstream)
    output = tmp_path / "oracle.jsonl"
    with pytest.raises(ValueError, match="did not complete"):
        swebench_gold_batches.run(input_path=input_path, count=1, output=output, head_port=11133)
    assert not output.exists()
    assert len(list(output.with_suffix(".batches").glob("evidence_temp2_*.jsonl"))) == 1

    swebench_gold_batches.run(input_path=input_path, count=1, output=output, head_port=11133)
    assert attempts == 2
    assert json.loads(output.read_text())["evaluation_completed"] is True
