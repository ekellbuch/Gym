# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from scripts.run_oracles.submit_references import reference_payload, reference_rows, submit_references

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.prompt import PromptConfig


def test_reference_text_and_native_question_are_preserved():
    """Response serialization preserves the published answer and native prompt without editing the source row."""
    row = {"question": "What is the answer?", "expected_answer": "  α\\beta {x}\nline\u2028end  "}
    original = deepcopy(row)
    result = reference_payload(row, field="expected_answer", index=2, prompt=PromptConfig(user="{question}"))
    assert NeMoGymResponse.model_validate(result["response"]).output_text == row["expected_answer"]
    assert result["responses_create_params"]["input"] == [{"role": "user", "content": row["question"]}]
    assert row == original


def test_boxed_format_preserves_reference_value():
    """Formatting an MCQA answer changes only its required output envelope."""
    result = reference_payload(
        {"expected_answer": "B", "responses_create_params": {"input": []}},
        field="expected_answer",
        index=0,
        template=r"\boxed{{answer}}",
    )
    assert NeMoGymResponse.model_validate(result["response"]).output_text == r"\boxed{B}"


def test_jsonl_does_not_split_unicode_inside_answers(tmp_path):
    """A Unicode line separator in a reference is data, not another JSONL row."""
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"expected_answer": "a\u2028b"}, ensure_ascii=False) + "\n{}\n")
    assert list(reference_rows(path, limit=1)) == [(0, {"expected_answer": "a\u2028b"})]


@pytest.mark.parametrize("values", [["node_a", "node_b"], []])
def test_identifier_list_serialization_preserves_members(values):
    """Native unquoted-list formatting preserves the published identifiers, including an empty answer."""
    row = {"expected_answer": json.dumps(values), "responses_create_params": {"input": []}}
    result = reference_payload(
        row, field="expected_answer", index=0, template="Final Answer: {answer}", encoding="json_string_list"
    )
    assert (
        NeMoGymResponse.model_validate(result["response"]).output_text == "Final Answer: [" + ", ".join(values) + "]"
    )
    assert json.loads(row["expected_answer"]) == values


async def test_missing_reference_fails_before_requests_or_output_changes(tmp_path):
    """A missing published answer aborts preflight before network work or truncating earlier results."""
    source = tmp_path / "data.jsonl"
    source.write_text(json.dumps({"expected_answer": "A", "responses_create_params": {"input": []}}) + "\n{}\n")
    output = tmp_path / "results.jsonl"
    output.write_text("previous results\n")
    client = SimpleNamespace(post=AsyncMock(side_effect=AssertionError("Network work before preflight")))
    with pytest.raises(ValueError, match="Row 1"):
        await submit_references(
            client,
            input_path=source,
            output_path=output,
            resources_server="verifier",
            reference_field="expected_answer",
        )
    assert output.read_text() == "previous results\n"


async def test_cancelled_retry_does_not_retain_old_success_summary(tmp_path):
    """Cancellation leaves the current run explicitly incomplete rather than displaying an earlier score."""
    source = tmp_path / "data.jsonl"
    source.write_text(json.dumps({"expected_answer": "A", "responses_create_params": {"input": []}}) + "\n")
    output = tmp_path / "results.jsonl"
    output.with_suffix(".summary.json").write_text('{"complete": true, "mean_reward": 1}\n')
    client = SimpleNamespace(post=AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await submit_references(
            client,
            input_path=source,
            output_path=output,
            resources_server="verifier",
            reference_field="expected_answer",
        )
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["complete"] is False
    assert summary["verified"] == 0
    assert summary["mean_reward"] is None


async def test_invalid_grading_and_transport_errors_do_not_become_scores(tmp_path):
    """Only valid verifier rewards contribute to the mean; failed rows stay visible and make the run incomplete."""
    source = tmp_path / "data.jsonl"
    source.write_text(
        "".join(
            json.dumps({"expected_answer": str(i), "responses_create_params": {"input": []}}) + "\n" for i in range(4)
        )
    )
    output = tmp_path / "results.jsonl"

    async def post(*, server_name, url_path, json):
        index = int(json["response"]["output"][0]["content"][0]["text"])
        if index == 3:
            raise ConnectionError("Verifier unavailable")
        result = [{"reward": 1.0}, {"reward": 0.0}, {"reward": 0.0, "mask_sample": True}][index]
        return SimpleNamespace(raise_for_status=lambda: None, json=AsyncMock(return_value=result))

    summary = await submit_references(
        SimpleNamespace(post=post),
        input_path=source,
        output_path=output,
        resources_server="verifier",
        reference_field="expected_answer",
    )
    # Valid scores are 1 and 0; neither masked zero nor network failure belongs in the denominator.
    assert {key: summary[key] for key in ("selected", "verified", "invalid", "errors", "mean_reward", "complete")} == {
        "selected": 4,
        "verified": 2,
        "invalid": 1,
        "errors": 1,
        "mean_reward": 0.5,
        "complete": False,
    }
    records = [json.loads(line) for line in output.open()]
    assert len(records) == 4
    failed = next(record for record in records if record["_oracle_reference"]["row_index"] == 3)
    assert "reward" not in failed
    assert failed["error"] == "ConnectionError: Verifier unavailable"
    assert json.loads(output.with_suffix(".summary.json").read_text()) == summary


async def test_charxiv_judge_failure_is_unscored(tmp_path):
    """CharXiv's published judge-failure sentinel and HTTP 401 must not become oracle zeros."""
    source = tmp_path / "charxiv.jsonl"
    source.write_text(
        "".join(json.dumps({"answer": str(i), "responses_create_params": {"input": []}}) + "\n" for i in range(3))
    )

    async def post(*, server_name, url_path, json):
        answer = json["response"]["output"][0]["content"][0]["text"]
        if answer == "2":
            raise RuntimeError("HTTP 401")
        result = (
            {"reward": 0.0, "extract_answer": "Failed to parse response"}
            if answer == "1"
            else {"reward": 0.0, "extract_answer": "different answer"}
        )
        return SimpleNamespace(raise_for_status=lambda: None, json=AsyncMock(return_value=result))

    summary = await submit_references(
        SimpleNamespace(post=post),
        input_path=source,
        output_path=tmp_path / "charxiv_results.jsonl",
        resources_server="charxiv_rq_benchmark_resources_server",
        reference_field="answer",
    )
    assert {key: summary[key] for key in ("verified", "invalid", "errors", "mean_reward", "complete")} == {
        "verified": 1,
        "invalid": 1,
        "errors": 1,
        "mean_reward": 0.0,
        "complete": False,
    }
