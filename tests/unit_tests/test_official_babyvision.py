# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from asyncio import Semaphore
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from resources_servers.vlm_eval_kit import official_babyvision
from resources_servers.vlm_eval_kit.official_babyvision import score_babyvision, verify_babyvision_source


SOURCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "results/oracle-upstream/VLMEvalKitMcore-6962c8d06b2b7b26a74a73d6212c06562b63e1b7"
)


def test_pinned_babyvision_source_hashes():
    if not SOURCE_DIR.exists():
        pytest.skip("Pinned Mcore source is downloaded during oracle preparation")
    verify_babyvision_source(SOURCE_DIR)


def test_unpinned_babyvision_source_is_rejected(tmp_path):
    for name in ("babyvision.py", "utils/babyvision.py"):
        source = tmp_path / "vlmeval/dataset" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("not the pinned scorer")
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        verify_babyvision_source(tmp_path)


@pytest.mark.parametrize(
    ("verdict", "expected_reward", "expected_invalid"),
    [
        ({"res": True, "log": "Succeed"}, 1.0, False),
        ({"res": False, "log": "Succeed"}, 0.0, False),
        ({"res": False, "log": "All retries failed.\n"}, 0.0, True),
    ],
)
def test_official_verdict_and_judge_failure_are_distinct(monkeypatch, verdict, expected_reward, expected_invalid):
    official = ModuleType("vlmeval.dataset.utils.babyvision")
    official.__file__ = str(SOURCE_DIR / "vlmeval/dataset/utils/babyvision.py")
    calls = []

    def auxeval(judge, line):
        calls.append((judge, line))
        return verdict

    official.BabyVision_auxeval = auxeval
    utils = ModuleType("vlmeval.dataset.utils")
    utils.babyvision = official
    dataset = ModuleType("vlmeval.dataset")
    dataset.utils = utils
    vlmeval = ModuleType("vlmeval")
    vlmeval.dataset = dataset
    monkeypatch.setitem(sys.modules, "vlmeval", vlmeval)
    monkeypatch.setitem(sys.modules, "vlmeval.dataset", dataset)
    monkeypatch.setitem(sys.modules, "vlmeval.dataset.utils", utils)
    monkeypatch.setitem(sys.modules, "vlmeval.dataset.utils.babyvision", official)
    monkeypatch.setattr(official_babyvision, "_check_hash", lambda path, expected: None)

    body = SimpleNamespace(
        category="Spatial Perception",
        ans_type="blank",
        question="Which cube?",
        answer="A",
        response=SimpleNamespace(output_text=r"\boxed{A}"),
    )
    judge = object()
    result = asyncio.run(score_babyvision(body, judge, Semaphore(1)))
    assert calls == [(judge, {"question": "Which cube?", "answer": "A", "prediction": r"\boxed{A}"})]
    assert result["reward"] == expected_reward
    assert result.get("invalid_judge_response", False) is expected_invalid
