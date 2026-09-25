# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit published reference answers to an existing Gym verifier, without model generation."""

import asyncio
import hashlib
import json
import math
from pathlib import Path

from nemo_gym.base_resources_server import BaseVerifyRequest
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    get_global_config_dict,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.prompt import PromptConfig, apply_prompt_to_row, load_prompt_config, validate_prompt_compatibility
from nemo_gym.server_utils import ServerClient


def reference_payload(
    row: dict,
    *,
    field: str,
    index: int,
    template: str = "{answer}",
    prompt: PromptConfig | None = None,
    encoding: str = "text",
) -> dict:
    """Wrap the exact reference value in the benchmark's required response format."""
    answer = row.get(field)
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Row {index}: {field!r} must contain a nonempty published string answer")
    if encoding == "json_string_list":
        # Some native verifiers request bracketed, unquoted identifiers instead of JSON.
        # Decode and serialize the published list; never compute or change its members.
        values = json.loads(answer)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"Row {index}: {field!r} must contain a JSON list of strings")
        if any(any(char in value for char in ",[]\n\r") for value in values):
            raise ValueError(f"Row {index}: reference identifiers cannot be represented without ambiguity")
        answer = "[" + ", ".join(values) + "]"
    elif encoding != "text":
        raise ValueError(f"Unknown reference_encoding: {encoding}")
    if template.count("{answer}") != 1:
        raise ValueError("response_template must contain exactly one literal {answer} placeholder")
    # Literal replacement preserves Unicode, whitespace, braces, and backslashes in the answer.
    response = NeMoGymResponse(
        id=f"reference-{index}",
        created_at=0,
        model="official-reference",
        object="response",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        output=[
            {
                "id": f"reference-message-{index}",
                "role": "assistant",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": template.replace("{answer}", answer), "annotations": []}],
            }
        ],
    )
    if prompt is not None:
        validate_prompt_compatibility([row], prompt)
        row = apply_prompt_to_row(row, prompt)
    payload = row | {"response": response.model_dump(mode="json", exclude_unset=True)}
    # Require the native question context; silently supplying an empty prompt changes judging.
    BaseVerifyRequest.model_validate(payload)
    return payload


def reference_rows(path: Path, *, limit: int | None = None):
    """Stream JSONL by physical lines; Unicode separators inside answers are not row boundaries."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    with path.open(encoding="utf-8") as source:
        count = 0
        for index, line in enumerate(source):
            if not line.strip():
                continue
            yield index, json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                break


async def submit_references(
    client: ServerClient,
    *,
    input_path: Path,
    output_path: Path,
    resources_server: str,
    reference_field: str,
    limit: int | None = None,
    response_template: str = "{answer}",
    prompt_config: str | None = None,
    reference_encoding: str = "text",
) -> dict:
    """Save verifier results and distinguish invalid grading or request failures from scores."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The output path must not overwrite the reference dataset")
    # Validate the entire selected input before any verifier request or output truncation.
    prompt = load_prompt_config(prompt_config) if prompt_config else None
    total = 0
    for index, row in reference_rows(input_path, limit=limit):
        reference_payload(
            row,
            field=reference_field,
            index=index,
            template=response_template,
            prompt=prompt,
            encoding=reference_encoding,
        )
        total += 1
    if not total:
        raise ValueError("No reference rows selected")

    with input_path.open("rb") as source:
        source_hash = hashlib.file_digest(source, "sha256").hexdigest()
    summary = {
        "input_path": str(input_path),
        "input_sha256": source_hash,
        "resources_server": resources_server,
        "reference_field": reference_field,
        "response_template": response_template,
        "prompt_config": prompt_config,
        "reference_encoding": reference_encoding,
        "selected": total,
        "verified": 0,
        "invalid": 0,
        "errors": 0,
        "mean_reward": None,
        "complete": False,
    }
    reward_sum = 0.0
    rows = iter(reference_rows(input_path, limit=limit))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_suffix(".summary.json")
    # A cancelled retry must not leave a previous run's successful summary beside partial results.
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output_path.open("w", encoding="utf-8") as output:

        async def worker() -> None:
            nonlocal reward_sum
            # A shared synchronous iterator bounds memory without queuing the whole dataset.
            for index, row in rows:
                payload = reference_payload(
                    row,
                    field=reference_field,
                    index=index,
                    template=response_template,
                    prompt=prompt,
                    encoding=reference_encoding,
                )
                provenance = {"row_index": index, "field": reference_field}
                try:
                    response = await client.post(server_name=resources_server, url_path="/verify", json=payload)
                    response.raise_for_status()
                    result = await response.json()
                    if not isinstance(result, dict):
                        raise ValueError("Verifier response must be a JSON object")
                    reward = result.get("reward")
                    invalid = (
                        result.get("mask_sample", False)
                        or result.get("invalid_judge_response", False)
                        or result.get("evaluation_completed") is False
                        or result.get("_ng_failure_class") == "judge_failed"
                        or result.get("failure_kind") == "judge_failed"
                        # Official CharXiv auxeval returns this sentinel after all
                        # judge attempts fail; its reward=0 is not a scored wrong answer.
                        or (
                            resources_server == "charxiv_rq_benchmark_resources_server"
                            and result.get("extract_answer") == "Failed to parse response"
                        )
                        or isinstance(reward, bool)
                        or not isinstance(reward, (int, float))
                        or not math.isfinite(reward)
                    )
                    if invalid:
                        summary["invalid"] += 1
                    else:
                        summary["verified"] += 1
                        reward_sum += reward
                    record = result | {"_oracle_reference": provenance}
                except Exception as error:
                    # Keep every failed row visible; an HTTP/judge failure is never a zero score.
                    summary["errors"] += 1
                    record = {"_oracle_reference": provenance, "error": f"{type(error).__name__}: {error}"}
                # Native `gym eval aggregate` can route these rows straight back to
                # the resource server for benchmark-specific metrics (e.g. COMET).
                record |= {
                    AGENT_REF_KEY_NAME: {"name": resources_server},
                    TASK_INDEX_KEY_NAME: index,
                    ROLLOUT_INDEX_KEY_NAME: 0,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()

        # Bound outstanding HTTP calls at the native judge's usual 64-request capacity.
        await asyncio.gather(*(worker() for _ in range(min(total, 64))))

    summary["mean_reward"] = reward_sum / summary["verified"] if summary["verified"] else None
    summary["complete"] = summary["verified"] == total
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    config = get_global_config_dict()
    result = asyncio.run(
        submit_references(
            ServerClient.load_from_global_config(),
            input_path=Path(config["benchmark_jsonl"]),
            output_path=Path(config["output_jsonl"]),
            resources_server=config["resources_server"],
            reference_field=config["reference_field"],
            limit=int(config["limit"]) if config.get("limit") is not None else None,
            response_template=config.get("response_template", "{answer}"),
            prompt_config=config.get("prompt_config"),
            reference_encoding=config.get("reference_encoding", "text"),
        )
    )
    raise SystemExit(0 if result["complete"] else 1)
