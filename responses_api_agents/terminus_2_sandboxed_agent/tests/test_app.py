# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from itertools import count
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.terminus_2_sandboxed_agent import app as app_module
from responses_api_agents.terminus_2_sandboxed_agent.app import (
    NeMoGymLLM,
    NeMoGymSandboxEnvironment,
    NeMoGymTerminus2,
    Terminus2Agent,
    Terminus2AgentConfig,
    _instruction,
)


def _terminus_config(**overrides) -> Terminus2AgentConfig:
    return Terminus2AgentConfig(
        **{
            "host": "0.0.0.0",
            "port": 8080,
            "entrypoint": "app.py",
            "name": "terminus_2_1_agent",
            "resources_server": ResourcesServerRef(type="resources_servers", name="swebench_resources_server"),
            "model_server": ModelServerRef(type="responses_api_models", name="policy_model"),
            "max_turns": 100,
            "enable_summarize": True,
            "proactive_summarization_threshold": 8000,
            "tmux_pane_width": 160,
            "tmux_pane_height": 40,
            "dump_trajectory": False,
            "debug": False,
            "model_context_limit": 32_000,
            "model_output_limit": 4_000,
            "llm_request_timeout": 60,
            "sandbox_provider": "opensandbox",
            "sandbox_timeout": 10,
            "remote_tmux_binary_path": None,
            **overrides,
        }
    )


def test_instruction_joins_text_content():
    assert _instruction([{"content": [{"text": "first"}]}, {"content": "second"}]) == "first\n\nsecond"


@pytest.mark.asyncio
async def test_sandbox_environment_adapts_exec_and_is_dir():
    sandbox_calls = []

    async def sandbox_exec(command, **kwargs):
        sandbox_calls.append((command, kwargs))
        return SimpleNamespace(stdout="output", stderr=None, return_code=0)

    sandbox = SimpleNamespace(exec=sandbox_exec)
    environment = NeMoGymSandboxEnvironment(sandbox, logs_dir=SimpleNamespace(), session_id="session-1")

    result = await environment.exec("pwd", timeout_sec=12, user="root", cwd="/work")

    assert result.stdout == "output"
    assert result.stderr == ""
    assert result.return_code == 0
    assert await environment.is_dir("/workspace")
    assert sandbox_calls == [
        ("pwd", {"timeout_s": 12, "cwd": "/work", "user": "root", "env": None}),
        ('test -d "/workspace"', {"user": None}),
    ]


@pytest.mark.asyncio
async def test_sandbox_environment_uses_sandbox_exec_for_stateful_commands():
    calls = []

    async def sandbox_exec(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="output", stderr=None, return_code=0)

    sandbox = SimpleNamespace(exec=sandbox_exec)
    environment = NeMoGymSandboxEnvironment(sandbox, logs_dir=SimpleNamespace(), session_id="session-1")

    await environment.exec("tmux new-session")

    assert calls == [("tmux new-session", {"timeout_s": None, "cwd": None, "user": None, "env": None})]


def test_agent_implements_required_responses_endpoint():
    assert not getattr(Terminus2Agent, "__abstractmethods__", set())


@pytest.mark.asyncio
async def test_nemo_gym_llm_records_every_responses_request_and_output():
    class Client:
        def __init__(self):
            self.requests = []

        async def create_response(self, **kwargs):
            self.requests.append(kwargs)
            index = len(self.requests)
            return NeMoGymResponse(
                id=f"resp_{index}",
                created_at=0,
                model="policy_model",
                object="response",
                output=[
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{index}",
                        content=[
                            NeMoGymResponseOutputText(type="output_text", text=f"answer {index}", annotations=[])
                        ],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                ],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=True,
                usage=NeMoGymResponseUsage(
                    input_tokens=10,
                    input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=2),
                    output_tokens=3,
                    output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                    total_tokens=13,
                ),
            )

    client = Client()
    llm = NeMoGymLLM(
        client=client,
        model_name="policy_model",
        model_context_limit=32_000,
        model_output_limit=4_000,
        llm_request_timeout=60,
    )

    first = await llm.call("first")
    second = await llm.call(
        "second",
        message_history=[{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer 1"}],
        previous_response_id="resp_1",
    )
    third = await llm.call(
        "third",
        message_history=[{"role": "user", "content": "compacted summary"}],
        previous_response_id="resp_2",
    )

    assert first.content == "answer 1"
    assert first.usage.prompt_tokens == 10
    assert second.content == "answer 2"
    assert third.content == "answer 3"
    assert client.requests == [
        {"model": "policy_model", "input": [{"content": "first", "role": "user", "type": "message"}]},
        {
            "model": "policy_model",
            "input": [
                {"content": "first", "role": "user", "type": "message"},
                {"content": "answer 1", "role": "assistant", "type": "message"},
                {"content": "second", "role": "user", "type": "message"},
            ],
        },
        {
            "model": "policy_model",
            "input": [
                {"content": "compacted summary", "role": "user", "type": "message"},
                {"content": "third", "role": "user", "type": "message"},
            ],
        },
    ]
    assert [item.content for item in llm.trajectory if isinstance(item, NeMoGymEasyInputMessage)] == [
        "first",
        "second",
        "third",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("dump_trajectory", [False, True])
@pytest.mark.parametrize("debug", [False, True])
async def test_execute_runs_terminus_in_seeded_sandbox(monkeypatch, dump_trajectory, debug):
    config = _terminus_config(dump_trajectory=dump_trajectory, debug=debug)
    set_level = MagicMock()
    monkeypatch.setattr(app_module.harbor_logger, "setLevel", set_level)
    server = Terminus2Agent(config=config, server_client=MagicMock(spec=ServerClient))
    sandbox_calls = []

    async def sandbox_exec(command, **kwargs):
        sandbox_calls.append((command, kwargs))
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    sandbox = SimpleNamespace(exec=sandbox_exec)

    class FakeTerminus:
        session = SimpleNamespace()

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._session = SimpleNamespace(stop=self.stop)
            self._times_spent = [1.0, 3.0]
            self._num_proactive_compactions = 0
            self._num_compactions = 2

        async def stop(self):
            return None

        async def setup(self, environment):
            await environment.exec("tmux setup")

        async def run(self, instruction, environment, context):
            assert instruction == "solve this"
            assert self.kwargs["dump_trajectory"] is dump_trajectory
            await environment.exec("tmux run")
            self.kwargs["llm"]._times_spent.extend([2.0, 4.0])
            self.kwargs["llm"]._num_compactions = 2
            context.n_input_tokens = 4
            context.n_output_tokens = 3
            self.kwargs["llm"].trajectory.append(
                NeMoGymResponseOutputMessage(
                    id="msg_done",
                    content=[NeMoGymResponseOutputText(type="output_text", text="done", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

    class FakeContext:
        n_input_tokens = None
        n_cache_tokens = None
        n_output_tokens = None
        metadata = None

    monkeypatch.setattr(app_module, "NeMoGymTerminus2", FakeTerminus)
    monkeypatch.setattr(app_module, "AgentContext", FakeContext)
    monkeypatch.setattr(Terminus2Agent, "base_url_for_run", lambda *_args, **_kwargs: "http://model")
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")
    elapsed_times = iter([10.0, 20.0])
    monkeypatch.setattr(app_module, "perf_counter", lambda: next(elapsed_times))

    async def request_json():
        return {"task_id": "task"}

    request = SimpleNamespace(json=request_json, session={app_module.SESSION_ID_KEY: "session-1"})
    response, metrics = await server._execute(
        request,
        NeMoGymResponseCreateParamsNonStreaming(input="solve this"),
        sandbox,
    )

    assert metrics == {
        "terminus2_completed": True,
        "command_exec_times": [1.0, 3.0],
        "model_call_times": [2.0, 4.0],
        "average_command_exec_time": 2.0,
        "average_model_call_time": 3.0,
        "total_command_exec_time": 4.0,
        "total_model_call_time": 6.0,
        "command_exec_time_pct": 40.0,
        "model_call_time_pct": 60.0,
        "terminus2_time_taken": 10.0,
        "model_calls_gt_10min": 0,
        "num_proactive_compactions": 0,
        "num_compactions": 2,
        "error": None,
        "failure_reason": None,
        "mask_sample": False,
        "usages": [],
    }
    assert response.output[-1].content[0].text == "done"
    assert response.usage.input_tokens == 4
    assert response.usage.output_tokens == 3
    if not debug:
        set_level.assert_called_once_with(logging.WARNING)
    else:
        set_level.assert_not_called()
    assert sandbox_calls == [
        ("mkdir -p /logs/agent", {"timeout_s": None, "cwd": None, "user": "root", "env": None}),
        ("tmux setup", {"timeout_s": None, "cwd": None, "user": None, "env": None}),
        ("tmux run", {"timeout_s": None, "cwd": None, "user": None, "env": None}),
    ]


class _FakeContext:
    n_input_tokens = None
    n_cache_tokens = None
    n_output_tokens = None
    metadata = None


class _FakeTerminus:
    def __init__(self, **kwargs):
        self._times_spent = []
        self._num_proactive_compactions = 0

    async def setup(self, environment):
        return None

    async def run(self, instruction, environment, context):
        self._times_spent.append(1.0)
        context.n_input_tokens = 0
        context.n_output_tokens = 0


class _ExplodingTerminus(_FakeTerminus):
    """The harness failure this module regressed on, reproduced at `run`."""

    async def run(self, instruction, environment, context):
        raise AttributeError("'NoneType' object has no attribute 'total_tokens'")


def _llm(client=None) -> NeMoGymLLM:
    return NeMoGymLLM(
        client=client if client is not None else MagicMock(),
        model_name="policy_model",
        model_context_limit=32_000,
        model_output_limit=4_000,
        llm_request_timeout=60,
    )


def _terminus(llm: NeMoGymLLM, logs_dir) -> NeMoGymTerminus2:
    return NeMoGymTerminus2(
        logs_dir=logs_dir,
        model_name="policy_model",
        max_turns=100,
        parser_name="json",
        enable_summarize=True,
        proactive_summarization_threshold=8000,
        tmux_pane_width=160,
        tmux_pane_height=40,
        record_terminal_session=False,
        llm=llm,
        dump_trajectory=False,
    )


def _usage(total_tokens: int) -> NeMoGymResponseUsage:
    return NeMoGymResponseUsage(
        input_tokens=total_tokens - 3,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
        output_tokens=3,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
        total_tokens=total_tokens,
    )


@pytest.mark.asyncio
async def test_a_response_without_usage_is_recorded_as_a_call_with_unknown_token_counts():
    """`usage` is optional on a Responses payload, and a truncated, errored or
    timed-out completion comes back without one. The call still happened, so it
    still belongs in `usages` -- as None, meaning "tokens unknown"."""

    class Client:
        async def create_response(self, **_kwargs):
            return NeMoGymResponse(
                id="resp_1",
                created_at=0,
                model="policy_model",
                object="response",
                output=[
                    NeMoGymResponseOutputMessage(
                        id="msg_1",
                        content=[NeMoGymResponseOutputText(type="output_text", text="answer", annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                ],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=True,
                usage=None,
            )

    llm = _llm(Client())

    response = await llm.call("first")

    assert response.content == "answer"
    assert response.usage is None
    assert llm.usages == [None]


@pytest.mark.asyncio
async def test_proactive_summarization_survives_a_last_response_that_carried_no_usage(tmp_path):
    """Regression: this raised `AttributeError: 'NoneType' object has no
    attribute 'total_tokens'` out of `agent.run`, which `_execute` swallowed
    into `error` while the verifier scored the half-finished sandbox 0 -- an
    infrastructure failure charged to the model."""
    llm = _llm()
    agent = _terminus(llm, tmp_path)
    chat = SimpleNamespace(messages=[{"role": "user", "content": "hello world"}])
    llm.usages.append(None)

    assert await agent._check_proactive_summarization(chat, "solve this", MagicMock()) is None


def test_an_unusable_last_usage_falls_back_to_the_local_token_estimate(tmp_path):
    """Falling back to 0 would read as an empty context and suppress compaction
    until the model hit its real limit, so the fallback has to be the estimate
    the agent uses everywhere else."""
    llm = _llm()
    agent = _terminus(llm, tmp_path)
    chat = SimpleNamespace(messages=[{"role": "user", "content": "hello world"}])
    llm.usages.append(None)

    agent._is_check_proactive_summarization = True
    counted = agent._count_total_tokens(chat)

    agent._is_check_proactive_summarization = False
    assert counted == agent._count_total_tokens(chat)
    assert counted > 0, "0 would read as an empty conversation and suppress summarization"


def test_a_usable_last_usage_still_beats_the_local_token_estimate(tmp_path):
    """The server-reported count is why this override exists: litellm cannot
    tokenize an arbitrary policy model, so its estimate drifts from what the
    endpoint actually charged."""
    llm = _llm()
    agent = _terminus(llm, tmp_path)
    chat = SimpleNamespace(messages=[{"role": "user", "content": "hello world"}])
    llm.usages.extend([None, _usage(4242)])

    agent._is_check_proactive_summarization = True
    assert agent._count_total_tokens(chat) == 4242

    agent._is_check_proactive_summarization = False
    assert agent._count_total_tokens(chat) != 4242


class _SlowTerminus(_FakeTerminus):
    """Runs past the task's budget instead of failing."""

    async def run(self, instruction, environment, context):
        await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_an_agent_timeout_stays_in_the_score(monkeypatch):
    """A task that burned its own budget failed to solve the task. Masking it
    would drop it from scoring -- and roughly half of Terminal-Bench ends this
    way, so the benchmark would report only the tasks the model finished."""
    monkeypatch.setattr(app_module, "NeMoGymTerminus2", _SlowTerminus)
    monkeypatch.setattr(app_module, "AgentContext", _FakeContext)
    monkeypatch.setattr(Terminus2Agent, "base_url_for_run", lambda *_a, **_k: "http://model")
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")
    clock = count(step=1.0)
    monkeypatch.setattr(app_module, "perf_counter", lambda: next(clock))

    async def sandbox_exec(command, **kwargs):
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    server = Terminus2Agent(config=_terminus_config(sandbox_timeout=0.01), server_client=MagicMock(spec=ServerClient))

    async def request_json():
        return {"task_id": "task"}

    request = SimpleNamespace(json=request_json, session={app_module.SESSION_ID_KEY: "session-1"})
    _response, metrics = await server._execute(
        request,
        NeMoGymResponseCreateParamsNonStreaming(input="solve this"),
        SimpleNamespace(exec=sandbox_exec),
    )

    assert metrics["terminus2_completed"] is False
    assert metrics["mask_sample"] is False
    assert metrics["failure_reason"] is None


@pytest.mark.asyncio
async def test_an_exception_out_of_terminus_is_reported_as_an_infrastructure_failure(monkeypatch):
    """A harness bug and a model that failed the task both land here as reward
    0. Without a flag saying which, a broken agent looks like a weak model and
    silently drags the benchmark score down."""
    monkeypatch.setattr(app_module, "NeMoGymTerminus2", _ExplodingTerminus)
    monkeypatch.setattr(app_module, "AgentContext", _FakeContext)
    monkeypatch.setattr(Terminus2Agent, "base_url_for_run", lambda *_a, **_k: "http://model")
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")
    clock = count(step=1.0)
    monkeypatch.setattr(app_module, "perf_counter", lambda: next(clock))

    async def sandbox_exec(command, **kwargs):
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    server = Terminus2Agent(config=_terminus_config(), server_client=MagicMock(spec=ServerClient))

    async def request_json():
        return {"task_id": "task"}

    request = SimpleNamespace(json=request_json, session={app_module.SESSION_ID_KEY: "session-1"})
    _response, metrics = await server._execute(
        request,
        NeMoGymResponseCreateParamsNonStreaming(input="solve this"),
        SimpleNamespace(exec=sandbox_exec),
    )

    assert metrics["terminus2_completed"] is False
    assert metrics["mask_sample"] is True
    assert "AttributeError" in metrics["failure_reason"]
    assert "total_tokens" in metrics["failure_reason"]


@pytest.mark.asyncio
async def test_a_quarantined_row_reaches_the_verify_response(monkeypatch):
    """Scoring reads the rollout row, not `_execute`'s return value, so the
    classification is only useful if it survives the merge with the verifier's
    result and the response model's validation."""

    async def sandbox_exec(command, **kwargs):
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    async def sandbox_stop():
        return None

    sandbox = SimpleNamespace(exec=sandbox_exec, stop=sandbox_stop)
    posts = []

    async def _seed_session_json():
        return {"sandbox_handle": "sbx-1"}

    async def post(**kwargs):
        posts.append(kwargs)
        return SimpleNamespace(cookies={}, json=_seed_session_json)

    server_client = MagicMock(spec=ServerClient)
    server_client.post = post

    async def fake_get_response_json(_verification):
        return {
            "responses_create_params": {"input": "solve this"},
            "response": posts[-1]["json"]["response"],
            "reward": 0.0,
            "evaluation_completed": True,
        }

    monkeypatch.setattr(app_module, "NeMoGymTerminus2", _ExplodingTerminus)
    monkeypatch.setattr(app_module, "AgentContext", _FakeContext)
    monkeypatch.setattr(Terminus2Agent, "base_url_for_run", lambda *_a, **_k: "http://model")
    monkeypatch.setattr(Terminus2Agent, "_connect_sandbox", lambda _self, _id: _resolved(sandbox))
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")
    monkeypatch.setattr(app_module, "raise_for_status", _noop_async)
    monkeypatch.setattr(app_module, "get_response_json", fake_get_response_json)
    clock = count(step=1.0)
    monkeypatch.setattr(app_module, "perf_counter", lambda: next(clock))

    server = Terminus2Agent(config=_terminus_config(), server_client=server_client)
    request = SimpleNamespace(json=_task_json, cookies={}, session={app_module.SESSION_ID_KEY: "session-1"})

    result = await server.run(
        request,
        app_module.Terminus2AgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="solve this")
        ),
    )

    assert result.reward == 0.0
    assert result.mask_sample is True
    assert "AttributeError" in result.failure_reason
    assert result.model_dump()["mask_sample"] is True


async def _noop_async(*_args, **_kwargs):
    return None


async def _resolved(value):
    return value


async def _task_json():
    return {"task_id": "task"}
