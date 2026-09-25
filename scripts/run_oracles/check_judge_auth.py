#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the configured Gym judge endpoint before an oracle run starts."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_CONFIG = Path("resources_servers/vlm_eval_kit/configs/vlm_eval_kit.yaml")


def configured_scalar(config: Path, key: str) -> str:
    """Read one literal judge setting from the small Gym benchmark YAML overlay."""
    matches = []
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*(.*?)\s*$")
    for line in config.read_text().splitlines():
        if match := pattern.match(line):
            value = match.group(1).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            matches.append(value)
    if len(matches) > 1:
        raise ValueError(f"Multiple {key} settings in {config}")
    return matches[0] if matches else ""


def check_judge_auth(config: Path, base_config: Path = BASE_CONFIG) -> None:
    model = configured_scalar(config, "judge_model")
    endpoint = configured_scalar(config, "judge_base_url") or configured_scalar(base_config, "judge_base_url")
    if not model or not endpoint or "${" in model + endpoint:
        raise ValueError(f"Literal judge_model and judge_base_url required for {config}")
    key = os.environ.get("INFERENCE_API_KEY")
    if not key:
        raise RuntimeError("INFERENCE_API_KEY is not set in this shell; restart the session.")

    payload = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1}
    ).encode()
    request = Request(
        endpoint,
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            status = response.status
    except HTTPError as exc:
        status = exc.code
    except URLError:
        raise RuntimeError("Judge endpoint is unreachable; oracle results were not scored.") from None
    if status != 200:
        raise RuntimeError(f"Judge endpoint returned HTTP {status}; oracle results were not scored.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Benchmark Gym config.yaml")
    args = parser.parse_args()
    try:
        check_judge_auth(args.config)
    except (OSError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print("Judge authentication preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
