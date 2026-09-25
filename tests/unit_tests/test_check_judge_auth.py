# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge preflight uses the benchmark's actual model and hides credentials."""

import importlib.util
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_oracles/check_judge_auth.py"
REPO_ROOT = SCRIPT.parents[2]
spec = importlib.util.spec_from_file_location("check_judge_auth", SCRIPT)
assert spec and spec.loader
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


@pytest.mark.parametrize(
    "benchmark",
    ["babyvision", "mmlongbench_doc", "mathvista", "charxiv_rq", "mathvision", "ocr_reasoning"],
)
def test_current_benchmark_configs_resolve_judge_settings(benchmark: str) -> None:
    config = REPO_ROOT / "benchmarks" / benchmark / "config.yaml"
    model = preflight.configured_scalar(config, "judge_model")
    endpoint = preflight.configured_scalar(REPO_ROOT / preflight.BASE_CONFIG, "judge_base_url")
    assert model and "${" not in model
    assert endpoint == "https://inference-api.nvidia.com/v1/chat/completions"


class Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_preflight_uses_configured_endpoint_and_model(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("# judge_model: ignored\njudge_model: azure/openai/gpt-4o\n")
    base = tmp_path / "base.yaml"
    base.write_text("judge_base_url: https://judge.example/v1/chat/completions\n")
    monkeypatch.setenv("INFERENCE_API_KEY", "private-test-key")
    sent = []

    def fake_urlopen(request, timeout):
        sent.append((request, timeout))
        return Response()

    monkeypatch.setattr(preflight, "urlopen", fake_urlopen)
    preflight.check_judge_auth(config, base)
    request, timeout = sent.pop()
    assert request.full_url == "https://judge.example/v1/chat/completions"
    assert json.loads(request.data)["model"] == "azure/openai/gpt-4o"
    assert request.get_header("Authorization") == "Bearer private-test-key"
    assert timeout == 20


def test_http_401_stops_without_exposing_key(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("judge_model: model-id\njudge_base_url: https://judge.example/chat\n")
    monkeypatch.setenv("INFERENCE_API_KEY", "private-test-key")

    def unauthorized(request, timeout):
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO())

    monkeypatch.setattr(preflight, "urlopen", unauthorized)
    with pytest.raises(RuntimeError, match="HTTP 401") as failure:
        preflight.check_judge_auth(config)
    assert "private-test-key" not in str(failure.value)
