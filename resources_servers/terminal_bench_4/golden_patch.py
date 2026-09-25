# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute a downloaded task's supplied solution without modifying its contents."""

import asyncio

from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.lifecycle import Session
from resources_servers.terminal_bench_4.models import AgentTermination
from resources_servers.terminal_bench_4.task import resolve_env
from resources_servers.terminal_bench_4.transfers import upload_dir


async def run_solution(session: Session) -> AgentTermination:
    """Stage the entire official solution and run it within the task's agent budget."""
    solution = session.task.path / "solution"
    if not (solution / "solve.sh").is_file():
        raise FileNotFoundError(f"Official solution script missing: {solution / 'solve.sh'}")
    environment = session.environment
    # The root helper stages /solution before mounting it into non-root images.
    # Providers without shared storage retain the ordinary upload path.
    if environment.shared_logs is None or not environment.shared_logs.solution_staged:
        prepared = await environment.exec("mkdir -p /solution && chmod 777 /solution", user="root", timeout_sec=60)
        if prepared.return_code:
            raise RuntimeError(f"Cannot stage official solution: {prepared.stderr}")
        await upload_dir(environment.main, solution, "/solution")
        readable = await environment.exec("chmod -R a+rX /solution", user="root", timeout_sec=60)
        if readable.return_code:
            raise RuntimeError(f"Cannot read official solution: {readable.stderr}")
    session.result["agent_execution"] = {"started_at": lifecycle.now()}
    session.persist()
    try:
        result = await environment.exec(
            "bash /solution/solve.sh > /logs/agent/oracle.log 2>&1",
            user=session.task.config.agent.user,
            timeout_sec=session.task.config.agent.timeout_sec,
            env={"DEBIAN_FRONTEND": "noninteractive"} | resolve_env(session.task.config.solution.get("env", {})),
        )
        return AgentTermination(
            reason="completed" if result.return_code == 0 else "nonzero_exit",
            exit_code=result.return_code,
        )
    except TimeoutError as exc:
        return AgentTermination(reason="timeout", detail=str(exc))
    finally:
        session.result["agent_execution"]["finished_at"] = lifecycle.now()
        # The native finalizer also collects /logs/agent. Download now so the
        # supplied script's output survives subsequent verifier/setup failures.
        try:
            (session.directory / "agent").mkdir(parents=True, exist_ok=True)
            await environment.main.download("/logs/agent/oracle.log", session.directory / "agent" / "oracle.log")
        except (Exception, asyncio.CancelledError) as exc:
            lifecycle.exception(session, exc, "OracleLogDownloadError")
        session.persist()
