# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A failed judge must not turn the official matcher's random fallback into an oracle score."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from resources_servers.vlm_eval_kit.official_vstar import score_vstar


@pytest.mark.asyncio
async def test_random_matcher_fallback_is_invalid(monkeypatch):
    vlmeval = ModuleType("vlmeval")
    dataset = ModuleType("vlmeval.dataset")
    utils = ModuleType("vlmeval.dataset.utils")
    multiple_choice = ModuleType("vlmeval.dataset.utils.multiple_choice")
    multiple_choice.extract_answer_from_item = lambda judge, item: {
        "opt": item["answer"],
        "log": "Failed to predict, thus randomly generate one. ",
    }
    for name, module in {
        "vlmeval": vlmeval,
        "vlmeval.dataset": dataset,
        "vlmeval.dataset.utils": utils,
        "vlmeval.dataset.utils.multiple_choice": multiple_choice,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    body = SimpleNamespace(
        response=SimpleNamespace(output_text="ambiguous"),
        choices={"A": "yes", "B": "no"},
        answer="A",
        question="Which?",
        category="test",
    )

    result = await score_vstar(body, judge=object())

    assert result["reward"] == 1.0
    assert result["invalid_judge_response"] is True
