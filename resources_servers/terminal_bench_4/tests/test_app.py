# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.terminal_bench_4.app import TerminalBench4ResourcesServer
from resources_servers.terminal_bench_4.lifecycle import Session
from resources_servers.terminal_bench_4.models import SeedSessionResponse


@pytest.mark.asyncio
async def test_retried_oracle_requests_execute_solution_once_then_verify(tmp_path):
    """HTTP retries share solution execution and pass its termination to the ordinary verifier."""
    solution = tmp_path / "solution"
    solution.mkdir()
    (solution / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    entered = asyncio.Event()
    release = asyncio.Event()
    executions = []
    verified = []

    async def execute(command, **kwargs):
        if command == "bash /solution/solve.sh > /logs/agent/oracle.log 2>&1":
            executions.append(command)
            entered.set()
            await release.wait()
        return SimpleNamespace(return_code=0)

    sandbox = AsyncMock()
    sandbox.exec.return_value = SimpleNamespace(return_code=0)
    environment = SimpleNamespace(main=sandbox, exec=execute, shared_logs=None)
    session = Session(
        "identity",
        "owner",
        SimpleNamespace(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[])),
        "session",
        tmp_path,
        environment=environment,
        task=SimpleNamespace(
            path=tmp_path, config=SimpleNamespace(agent=SimpleNamespace(user=None, timeout_sec=60), solution={})
        ),
    )
    session.seed_response = SeedSessionResponse(session_id="session")

    class Endpoint:
        config = SimpleNamespace(is_verifying_golden_patch=True)
        _oracle_tasks = {}
        _execute_oracle = TerminalBench4ResourcesServer._execute_oracle

        async def seed_session(self, request, body):
            return session.seed_response

        def _session(self, request, session_id):
            return session

        async def verify(self, request, body):
            verified.append(body)
            return {"reward": 1, "evaluation_completed": True}

    server = Endpoint()
    first = asyncio.create_task(TerminalBench4ResourcesServer.oracle(server, None, None))
    await asyncio.wait_for(entered.wait(), timeout=1)
    second = asyncio.create_task(TerminalBench4ResourcesServer.oracle(server, None, None))
    release.set()
    assert await asyncio.gather(first, second) == [
        {"reward": 1, "evaluation_completed": True},
        {"reward": 1, "evaluation_completed": True},
    ]
    assert executions == ["bash /solution/solve.sh > /logs/agent/oracle.log 2>&1"]
    assert [body.termination.reason for body in verified] == ["completed"]
    assert [body.agent_started for body in verified] == [True]
