# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vanillux's model request and Bash timeout must survive the Gym bridge."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM
from responses_api_agents.harbor_agent.custom_agents.vanillux_nemo_gym import (
    VanilluxNemoGym,
    VanilluxNemoGymLLM,
)


CONFIG = Path(__file__).parents[1] / "configs" / "harbor_agent_tmax_vanillux.yaml"
TRAIN_CONFIG = Path(__file__).parents[1] / "configs" / "harbor_agent_tmax_vanillux_train.yaml"


def _llm(*, api_base="http://localhost:8000/v1", **kwargs):
    return VanilluxNemoGymLLM(
        model_name="test-model",
        api_base=api_base,
        collect_rollout_details=True,
        **kwargs,
    )


def test_model_request_has_reference_tool_grammar_and_sampling() -> None:
    llm = _llm()
    params = llm._extra_chat_params
    assert params["temperature"] == 1.0
    assert params["top_p"] == 1.0
    assert params["tool_choice"] == "none"
    assert params["tools"][0]["function"]["name"] == "bash"
    assert params["logprobs"] is True

    with pytest.raises(ValueError, match="top_p=1.0"):
        _llm(responses_create_params={"model": "test-model", "input": [], "top_p": 0.95})


@pytest.mark.parametrize("logprobs", [[-0.2], None])
def test_token_alignment_fails_closed(monkeypatch, logprobs) -> None:
    async def fake_call(self, prompt, message_history=None, **kwargs):
        self._last_prompt_token_ids = [11]
        self._last_completion_token_ids = [12, 13]
        self._last_logprobs = logprobs
        return SimpleNamespace(content="ok")

    monkeypatch.setattr(NemoGymLLM, "call", fake_call)
    llm = _llm()
    with pytest.raises(ValueError, match="unaligned token IDs"):
        asyncio.run(llm.call("prompt"))
    assert llm.total_generated_tokens == 0


def test_response_budget_caps_each_request(monkeypatch) -> None:
    requested = []

    async def fake_call(self, prompt, message_history=None, **kwargs):
        requested.append(self._extra_chat_params["max_tokens"])
        self._last_prompt_token_ids = [11]
        self._last_completion_token_ids = [12] * requested[-1]
        self._last_logprobs = [-0.2] * requested[-1]
        return SimpleNamespace(content="ok")

    monkeypatch.setattr(NemoGymLLM, "call", fake_call)
    llm = _llm(max_total_response_tokens=3, max_turn_output_tokens=2)
    asyncio.run(llm.call("one"))
    asyncio.run(llm.call("two"))
    assert requested == [2, 1]
    assert llm.total_generated_tokens == 3
    with pytest.raises(RuntimeError, match="exhausted"):
        asyncio.run(llm.call("three"))


def test_captured_response_uses_usage_for_budget_without_inline_tokens(monkeypatch) -> None:
    requested = []

    async def fake_call(self, prompt, message_history=None, **kwargs):
        requested.append(self._extra_chat_params["max_tokens"])
        self._last_prompt_token_ids = None
        self._last_completion_token_ids = None
        self._last_logprobs = None
        return SimpleNamespace(content="ok", usage=SimpleNamespace(completion_tokens=requested[-1]))

    monkeypatch.setattr(NemoGymLLM, "call", fake_call)
    llm = _llm(
        api_base="http://localhost:8000/ng-rollout/r1/training-token-capture/v1",
        max_total_response_tokens=3,
        max_turn_output_tokens=2,
    )
    asyncio.run(llm.call("one"))
    assert llm.total_generated_tokens == 2
    asyncio.run(llm.call("two"))
    assert requested == [2, 1]
    assert llm.total_generated_tokens == 3


def test_captured_response_requires_token_usage(monkeypatch) -> None:
    async def fake_call(self, prompt, message_history=None, **kwargs):
        return SimpleNamespace(content="ok", usage=None)

    monkeypatch.setattr(NemoGymLLM, "call", fake_call)
    llm = _llm(api_base="http://localhost:8000/ng-rollout/r1/training-token-capture/v1")
    with pytest.raises(ValueError, match="missing completion token usage"):
        asyncio.run(llm.call("one"))


def test_bash_call_has_120_second_limit_without_shortening_verifier(tmp_path) -> None:
    class RecordingEnvironment:
        calls = []

        async def exec(self, command, **kwargs):
            self.calls.append((command, kwargs))
            return SimpleNamespace(stdout="hello", stderr="", return_code=0)

    agent = VanilluxNemoGym(logs_dir=tmp_path, model_name="test-model", llm=_llm())
    environment = RecordingEnvironment()
    content = "<tool_call>\n" + json.dumps({"name": "bash", "arguments": {"command": "echo hello"}}) + "\n</tool_call>"
    observation = asyncio.run(agent._take_turn(content, 1, environment))
    assert "hello" in observation
    assert environment.calls[0][1] == {"timeout_sec": 120}

    config = yaml.safe_load(CONFIG.read_text())["harbor_agent"]["responses_api_agents"]["harbor_agent"]
    assert config["harbor_environment_kwargs"]["default_exec_timeout_s"] == 600
    assert config["harbor_verifier_override_timeout"] is None
    assert config["harbor_agent_override_timeout"] is None
    assert config["harbor_environment_kwargs"]["override_cpus"] is None
    assert config["harbor_environment_kwargs"]["override_memory_mb"] is None
    assert config["harbor_agent_kwargs"]["model_info"]["max_input_tokens"] == 51200
    assert config["harbor_agent_kwargs"]["max_turns"] == 64
    training = yaml.safe_load(TRAIN_CONFIG.read_text())["harbor_agent"]["responses_api_agents"]["harbor_agent"]
    assert training["harbor_agent_override_timeout"] == 14400
    assert training["harbor_verifier_override_timeout"] == 600
