# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest

from resources_servers.terminal_bench_4 import apply_golden_patch


@pytest.mark.asyncio
async def test_parallel_trials_preserve_failures_without_counting_them_as_zero_scores(tmp_path, monkeypatch):
    """Concurrent task failures stay visible and cannot become a complete benchmark score."""
    entered = 0
    both_running = asyncio.Event()

    async def post(**kwargs):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_running.set()
        await asyncio.wait_for(both_running.wait(), timeout=1)
        if kwargs["json"]["task_name"] == "broken":
            raise RuntimeError("sandbox unavailable")

        async def data():
            return {"evaluation_completed": True, "reward": 1, "task_id": "working"}

        return SimpleNamespace(raise_for_status=lambda: None, json=data)

    monkeypatch.setattr(apply_golden_patch.ServerClient, "load_from_global_config", lambda: SimpleNamespace(post=post))
    output = tmp_path / "results.jsonl"
    summary = await apply_golden_patch.main(
        [{"task_name": "working"}, {"task_name": "broken"}], concurrency=2, output_fpath=output
    )
    assert summary == {
        "requested": 2,
        "completed": 1,
        "passed": 1,
        "incomplete": 1,
        "score": None,
        "complete_run": False,
    }
    rows = {row["task_id"]: row for row in map(json.loads, output.read_text().splitlines())}
    assert rows["broken"] == {
        "task_id": "broken",
        "evaluation_completed": False,
        "infrastructure_error": "RuntimeError: sandbox unavailable",
    }
    assert json.loads(output.with_suffix(".summary.json").read_text()) == summary


@pytest.mark.asyncio
async def test_interrupted_run_removes_previous_complete_summary(tmp_path, monkeypatch):
    """Cancellation cannot leave a previous run's success summary beside partial results."""
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def post(**kwargs):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(apply_golden_patch.ServerClient, "load_from_global_config", lambda: SimpleNamespace(post=post))
    output = tmp_path / "results.jsonl"
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text('{"complete_run": true}')
    task = asyncio.create_task(apply_golden_patch.main([{"task_name": "task"}], concurrency=1, output_fpath=output))
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert not summary_path.exists()
    assert output.read_text() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", [float("nan"), float("inf"), "1", True, -1])
async def test_malformed_reward_is_an_incomplete_trial(tmp_path, monkeypatch, reward):
    """Invalid reward values cannot become a completed oracle result."""

    async def post(**kwargs):
        async def data():
            return {"evaluation_completed": True, "reward": reward}

        return SimpleNamespace(raise_for_status=lambda: None, json=data)

    monkeypatch.setattr(apply_golden_patch.ServerClient, "load_from_global_config", lambda: SimpleNamespace(post=post))
    summary = await apply_golden_patch.main(
        [{"task_name": "task"}], concurrency=1, output_fpath=tmp_path / "results.jsonl"
    )
    assert summary == {
        "requested": 1,
        "completed": 0,
        "passed": 0,
        "incomplete": 1,
        "score": None,
        "complete_run": False,
    }
