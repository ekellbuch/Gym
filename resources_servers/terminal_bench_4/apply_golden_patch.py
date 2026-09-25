# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit official TB4 solution executions concurrently to the native Gym server."""

import asyncio
import json
import math
from pathlib import Path
from uuid import uuid4

from aiohttp import ClientTimeout

from nemo_gym.global_config import get_global_config_dict
from nemo_gym.server_utils import ServerClient


async def main(
    examples: list[dict], *, concurrency: int, output_fpath: Path, resources_server: str = "terminal_bench_4"
) -> dict:
    """Persist every outcome; distinguish incomplete infrastructure trials from scored tasks."""
    if concurrency < 1 or not examples:
        raise ValueError("Oracle execution requires tasks and positive concurrency")
    client = ServerClient.load_from_global_config()
    output_fpath.with_suffix(".summary.json").unlink(missing_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    run_id = uuid4().hex

    async def execute(index: int, example: dict) -> dict:
        body = example | {"rollout_id": f"oracle-{run_id}-{index}", "client_session_id": f"oracle-{run_id}-{index}"}
        async with semaphore:
            try:
                response = await client.post(
                    server_name=resources_server,
                    url_path="/oracle",
                    json=body,
                    timeout=ClientTimeout(total=36000),
                )
                response.raise_for_status()
                result = await response.json()
                if result.get("evaluation_completed"):
                    reward = result.get("reward")
                    if (
                        isinstance(reward, bool)
                        or not isinstance(reward, (int, float))
                        or not math.isfinite(reward)
                        or not 0 <= reward <= 1
                    ):
                        raise ValueError("Completed verifier response has an invalid reward")
                return result
            except Exception as exc:
                return {
                    "task_id": example["task_name"],
                    "evaluation_completed": False,
                    "infrastructure_error": f"{type(exc).__name__}: {exc}",
                }

    tasks = [asyncio.create_task(execute(index, example)) for index, example in enumerate(examples)]
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    completed = passed = failures = 0
    try:
        with output_fpath.open("w", encoding="utf-8") as output:
            for future in asyncio.as_completed(tasks):
                result = await future
                output.write(json.dumps(result) + "\n")
                output.flush()
                if result.get("evaluation_completed") and not result.get("infrastructure_error"):
                    completed += 1
                    passed += result["reward"] == 1
                else:
                    failures += 1
                print(f"Completed {completed}/{len(examples)}; passed {passed}; incomplete {failures}", flush=True)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    summary = {
        "requested": len(examples),
        "completed": completed,
        "passed": passed,
        "incomplete": failures,
        "score": passed / completed if completed == len(examples) else None,
        "complete_run": completed == len(examples),
    }
    output_fpath.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    config = get_global_config_dict()
    with open(config["benchmark_jsonl"], encoding="utf-8") as benchmark:
        rows = [json.loads(line) for line in benchmark if line.strip()]
    if config.get("limit") is not None:
        limit = int(config["limit"])
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    summary = asyncio.run(
        main(
            rows,
            concurrency=int(config.get("concurrency", len(rows))),
            output_fpath=Path(config["output_fpath"]),
            resources_server=config.get("resources_server", "terminal_bench_4"),
        )
    )
    raise SystemExit(0 if summary["complete_run"] else 2)
