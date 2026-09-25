# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from resources_servers.terminal_bench_4.golden_patch import run_solution
from resources_servers.terminal_bench_4.lifecycle import Session


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code,reason", [(0, "completed"), (7, "nonzero_exit"), (None, "timeout")])
async def test_official_solution_is_uploaded_unchanged_and_uses_task_execution_settings(tmp_path, exit_code, reason):
    """Execute the published script using its task budget, user, and solution environment."""
    solution = tmp_path / "package/solution"
    solution.mkdir(parents=True)
    script = b"#!/bin/bash\nprintf 'official reference\\n'\n"
    (solution / "solve.sh").write_bytes(script)
    (solution / "helper.txt").write_bytes(b"supporting file")
    sandbox = AsyncMock()
    sandbox.exec.return_value = SimpleNamespace(return_code=0)
    uploaded = {}

    async def upload(source, target):
        import tarfile

        with tarfile.open(source) as archive:
            uploaded.update({name: archive.extractfile(name).read() for name in archive.getnames() if name != "."})

    sandbox.upload.side_effect = upload
    environment = SimpleNamespace(main=sandbox, exec=AsyncMock(), shared_logs=None)
    environment.exec.side_effect = [
        SimpleNamespace(return_code=0),
        SimpleNamespace(return_code=0),
        SimpleNamespace(return_code=exit_code) if exit_code is not None else TimeoutError("task budget exceeded"),
    ]
    task = SimpleNamespace(
        path=solution.parent,
        config=SimpleNamespace(
            agent=SimpleNamespace(user="task-user", timeout_sec=123), solution={"env": {"OFFICIAL": "value"}}
        ),
    )
    session = Session("identity", "owner", None, "session", tmp_path, task=task, environment=environment)
    termination = await run_solution(session)
    assert termination.reason == reason
    assert termination.exit_code == exit_code
    assert uploaded == {"./solve.sh": script, "./helper.txt": b"supporting file"}
    (command,) = environment.exec.call_args.args
    assert command == "bash /solution/solve.sh > /logs/agent/oracle.log 2>&1"
    assert environment.exec.call_args.kwargs == {
        "user": "task-user",
        "timeout_sec": 123,
        "env": {"DEBIAN_FRONTEND": "noninteractive", "OFFICIAL": "value"},
    }


@pytest.mark.asyncio
async def test_missing_official_solution_is_an_error_before_sandbox_execution(tmp_path):
    """Missing references fail rather than generating a substitute or scoring zero."""
    session = Session(
        "identity", "owner", None, "session", tmp_path, task=SimpleNamespace(path=tmp_path), environment=None
    )
    with pytest.raises(FileNotFoundError, match="Official solution script missing"):
        await run_solution(session)


@pytest.mark.asyncio
async def test_pre_staged_solution_runs_without_root_switching(tmp_path):
    """Mounted official solutions execute as the task user without setuid or staging commands."""
    solution = tmp_path / "solution"
    solution.mkdir()
    (solution / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    commands = []

    async def execute(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(return_code=0)

    environment = SimpleNamespace(main=AsyncMock(), exec=execute, shared_logs=SimpleNamespace(solution_staged=True))
    task = SimpleNamespace(
        path=tmp_path, config=SimpleNamespace(agent=SimpleNamespace(user=None, timeout_sec=60), solution={})
    )
    session = Session("identity", "owner", None, "session", tmp_path, task=task, environment=environment)
    result = await run_solution(session)
    assert result.reason == "completed"
    assert commands == [
        (
            "bash /solution/solve.sh > /logs/agent/oracle.log 2>&1",
            {"user": None, "timeout_sec": 60, "env": {"DEBIAN_FRONTEND": "noninteractive"}},
        )
    ]
