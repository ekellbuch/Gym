# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Malformed judge extraction must not count as an incorrect oracle answer."""

from types import SimpleNamespace

from resources_servers.vlm_eval_kit.official_mmlongbench_doc import score_mmlongbench_reference


def test_missing_extracted_answer_is_invalid():
    official = SimpleNamespace(
        MMLongBench_auxeval=lambda judge, line: {"res": "malformed nonempty reply", "pred": ""},
        eval_score=lambda answer, pred, answer_format: 0.0,
    )

    result = score_mmlongbench_reference(
        {"question": "What is shown?", "prediction": "gold", "answer": "gold", "answer_format": "str"},
        object(),
        official,
    )

    assert result["invalid_judge_response"] is True
