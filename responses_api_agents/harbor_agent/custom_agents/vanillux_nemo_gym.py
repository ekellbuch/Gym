# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned TMax Vanillux Harbor agent through NeMo Gym's model server."""

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from harbor.llms.base import BaseLLM
from harbor.models.agent.context import AgentContext
from overfit_tbench.agents import vanillux
from overfit_tbench.agents.vanillux_harbor import VanilluxAgent

from nemo_gym.config_types import ROLLOUT_PATH_PREFIX, TOKEN_CAPTURE_PATH_SEGMENT
from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM


class VanilluxNemoGymLLM(NemoGymLLM):
    """Keep Vanillux's raw Qwen tool grammar and fail on incomplete token evidence."""

    def __init__(
        self,
        *,
        max_total_response_tokens: int = 65536,
        max_turn_output_tokens: int = 16384,
        **kwargs: Any,
    ) -> None:
        if max_total_response_tokens <= 0 or max_turn_output_tokens <= 0:
            raise ValueError("Vanillux response token budgets must be positive")
        self.max_total_response_tokens = max_total_response_tokens
        self.max_turn_output_tokens = max_turn_output_tokens
        self.total_generated_tokens = 0
        super().__init__(**kwargs)
        path = urlsplit(self._api_base).path
        self._capture_mode = f"/{ROLLOUT_PATH_PREFIX}/" in path and f"/{TOKEN_CAPTURE_PATH_SEGMENT}/" in path

    def _build_extra_chat_params(self, responses_create_params: dict[str, Any]) -> dict[str, Any]:
        params = super()._build_extra_chat_params(responses_create_params)
        for name in ("temperature", "top_p"):
            value = params.get(name)
            if value is not None and float(value) != 1.0:
                raise ValueError(f"TMax Vanillux requires {name}=1.0; got {value!r}")
            params[name] = 1.0
        # Harbor Chat passes these as call kwargs; NemoGymLLM does not forward
        # call kwargs, so set them on its actual chat-completions request.
        params["tools"] = vanillux.bash_tool_definitions()
        params["tool_choice"] = "none"
        params["logprobs"] = True
        return params

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        remaining = self.max_total_response_tokens - self.total_generated_tokens
        if remaining <= 0:
            raise RuntimeError("Vanillux response token budget is exhausted")
        self._extra_chat_params["max_tokens"] = min(self.max_turn_output_tokens, remaining)
        response = await super().call(prompt, message_history, **kwargs)
        prompt_ids = self._last_prompt_token_ids
        token_ids = self._last_completion_token_ids
        logprobs = self._last_logprobs
        if self._capture_mode:
            count = response.usage.completion_tokens if response.usage is not None else None
            if not isinstance(count, int) or count <= 0:
                raise ValueError("Captured Vanillux response is missing completion token usage")
            self.total_generated_tokens += count
        else:
            if not prompt_ids or not token_ids or logprobs is None or len(token_ids) != len(logprobs):
                raise ValueError("Vanillux rollout has missing or unaligned token IDs and behavior logprobs")
            self.total_generated_tokens += len(token_ids)
        return response


class VanilluxNemoGym(VanilluxAgent):
    """Select the released Vanillux interaction through Gym's existing Harbor bridge."""

    @staticmethod
    def name() -> str:
        return "tmax-vanillux-nemo-gym"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        api_base: str | None = None,
        collect_rollout_details: bool = True,
        model_info: dict[str, Any] | None = None,
        responses_create_params: dict[str, Any] | None = None,
        nemo_model_server_timeout_sec: float = 2400,
        max_total_response_tokens: int = 65536,
        max_turn_output_tokens: int = 16384,
        bash_timeout_sec: int = 120,
        llm: BaseLLM | None = None,
        **kwargs: Any,
    ) -> None:
        if not collect_rollout_details:
            raise ValueError("DPPO requires per-turn token IDs and behavior logprobs")
        if bash_timeout_sec <= 0:
            raise ValueError("Vanillux Bash timeout must be positive")
        if llm is None:
            if model_name is None or api_base is None:
                raise ValueError("VanilluxNemoGym requires model_name and api_base")
            llm = VanilluxNemoGymLLM(
                model_name=model_name,
                api_base=api_base,
                collect_rollout_details=True,
                model_info=model_info,
                responses_create_params=responses_create_params,
                timeout_sec=nemo_model_server_timeout_sec,
                max_total_response_tokens=max_total_response_tokens,
                max_turn_output_tokens=max_turn_output_tokens,
            )
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            api_base=api_base,
            collect_rollout_details=True,
            model_info=model_info,
            llm=llm,
            **kwargs,
        )
        self._stop_reason = "turn_cap"
        self._bash_timeout_sec = bash_timeout_sec

    async def _take_turn(self, content: str, turn: int, environment: Any) -> str | None:
        args = self._parser.parse(content)
        if args is None:
            self._turns.append({"index": turn, "format_error": "no_tool_call"})
            next_prompt = self._format_error_prompt()
        else:
            command = args.get("command", "")
            if not command:
                self._turns.append({"index": turn, "format_error": "empty_command"})
                next_prompt = vanillux.format_error_message("'command' parameter is required.")
            else:
                # Harbor's verifier omits timeout_sec, so its default must stay
                # at 600s. Bound only the agent's Bash call to the paper's 120s.
                result = await environment.exec(
                    vanillux.wrapped_bash_command(command), timeout_sec=self._bash_timeout_sec
                )
                raw = vanillux.merge_streams(result.stdout, result.stderr)
                if vanillux.SUBMIT_MARKER in raw:
                    self._record_submission(turn)
                    self._stop_reason = "submitted"
                    return None
                self._turns.append({"index": turn, "exit_code": result.return_code})
                next_prompt = vanillux.format_bash_observation(
                    result.stdout,
                    result.stderr,
                    result.return_code,
                    self._turns_remaining_message(turn),
                )
                if len(raw) > vanillux.OBS_MAX_CHARS:
                    self._truncated_observations += 1
                    self._truncation_saved_chars += len(raw) - len(vanillux.truncate_observation(raw))
                if self._last_step_warning and turn == self._max_turns - 1:
                    next_prompt = f"{next_prompt}\n\n{vanillux.LAST_STEP_WARNING}"
        if isinstance(self._llm, VanilluxNemoGymLLM) and (
            self._llm.total_generated_tokens >= self._llm.max_total_response_tokens
        ):
            self._stop_reason = "response_token_budget"
            return None
        return next_prompt

    def _finalize(self, context: AgentContext, chat: Any) -> None:
        super()._finalize(context, chat)
        context.metadata["stop_reason"] = self._stop_reason
        if isinstance(self._llm, VanilluxNemoGymLLM):
            context.metadata["generated_response_tokens"] = self._llm.total_generated_tokens

    async def run(self, instruction: str, environment: Any, context: AgentContext) -> None:
        try:
            await super().run(instruction, environment, context)
        finally:
            if isinstance(self._llm, VanilluxNemoGymLLM):
                await self._llm.aclose()
