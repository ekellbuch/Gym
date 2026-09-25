# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.vlm_eval_kit.app import (
    VlmEvalKitResourcesServer,
    VlmEvalKitResourcesServerConfig,
    VLMEvalKitVerifyRequest,
)


class TestApp:
    def test_sanity(self) -> None:
        config = VlmEvalKitResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        VlmEvalKitResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.parametrize(
        ("extracted", "masked", "failure_kind"),
        [(None, True, "judge_failed"), ("Failed to parse response", True, "judge_failed"), ("wrong", False, None)],
    )
    async def test_charxiv_judge_failure_is_excluded_from_scores(
        self, monkeypatch: pytest.MonkeyPatch, extracted: str | None, masked: bool, failure_kind: str | None
    ) -> None:
        """Exclude a failed judge while retaining a scored wrong answer."""
        vlmeval = ModuleType("vlmeval")
        vlmeval.__path__ = []
        dataset = ModuleType("vlmeval.dataset")
        dataset.__path__ = []
        charxiv = ModuleType("vlmeval.dataset.charxiv")

        def auxeval(_judge: object, _line: dict[str, str]) -> dict[str, object]:
            if extracted is None:
                raise RuntimeError("judge unavailable")
            return {"score": 0.0, "extract_answer": extracted}

        charxiv.auxeval = auxeval
        monkeypatch.setitem(sys.modules, "vlmeval", vlmeval)
        monkeypatch.setitem(sys.modules, "vlmeval.dataset", dataset)
        monkeypatch.setitem(sys.modules, "vlmeval.dataset.charxiv", charxiv)

        config = VlmEvalKitResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name="", judge_model="official-judge"
        )
        server = VlmEvalKitResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        server._judge = object()
        request = VLMEvalKitVerifyRequest.model_validate(
            {
                "benchmark_name": "CharXiv_reasoning_val",
                "category": "Text-in-Chart",
                "answer": "reference",
                "grading_query": "question and reference",
                "responses_create_params": {"input": "question"},
                "response": {
                    "id": "reference",
                    "created_at": 0.0,
                    "model": "official-reference",
                    "object": "response",
                    "output": [
                        {
                            "id": "message",
                            "content": [
                                {"annotations": [], "text": "prediction", "type": "output_text", "logprobs": None}
                            ],
                            "role": "assistant",
                            "status": "completed",
                            "type": "message",
                            "phase": "final_answer",
                        }
                    ],
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                },
            }
        )

        result = await server.verify(request)

        assert result.reward == 0.0
        assert result.mask_sample is masked
        assert result.failure_kind == failure_kind
