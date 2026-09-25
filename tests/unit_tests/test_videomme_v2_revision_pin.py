# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video-MME-v2 preparation must not resolve a moving Hugging Face main ref."""

import json
import sys
import types
from pathlib import Path

import huggingface_hub
import pytest

from benchmarks.videomme2.prepare_data import prepare_videomme_v2


DATA_REVISION = "6e4bebb03202e1ddbf3d37703e560e51c5aa2d64"
REPO_ID = "MME-Benchmarks/Video-MME-v2"


@pytest.mark.parametrize("cached_main", [True, False])
def test_prepare_uses_pinned_hf_snapshot_even_with_newer_cached_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached_main: bool
) -> None:
    pinned_path = str(tmp_path / DATA_REVISION)
    newer_main_path = str(tmp_path / "newer-main")
    downloads = []
    official = types.ModuleType("vlmeval.dataset.videomme_v2")

    def download(*, repo_id, repo_type, revision=None):
        downloads.append((repo_id, repo_type, revision))
        if (repo_id, repo_type, revision) != (REPO_ID, "dataset", DATA_REVISION):
            raise AssertionError("Video-MME-v2 snapshot was not pinned")
        return pinned_path

    official.snapshot_download = download
    official.get_cache_path = lambda repo_id: newer_main_path if cached_main else None
    official.modelscope_flag_set = lambda: False
    (tmp_path / "video").mkdir()
    (tmp_path / "video" / "001.mp4").touch()

    class FakeVideoMMEv2:
        def __init__(self, **_kwargs):
            source = official.get_cache_path(REPO_ID)
            if source is None:
                source = official.snapshot_download(repo_id=REPO_ID, repo_type="dataset")
            assert source == pinned_path, "Video-MME-v2 selected a moving main snapshot"
            self.data_root = tmp_path
            self.data = types.SimpleNamespace(
                iterrows=lambda: iter(
                    [
                        (
                            0,
                            {
                                "index": 0,
                                "question": "Which option is correct?",
                                "candidates": "['A. first', 'B. second']",
                                "answer": "A",
                                "video": 1,
                                "video_path": "./video/001.mp4",
                                "level": 1,
                                "group_type": "test",
                                "group_structure": "{}",
                                "second_head": None,
                                "third_head": None,
                                "question_id": "q0",
                            },
                        )
                    ]
                )
            )

    official.VideoMMEv2 = FakeVideoMMEv2
    vlmeval = types.ModuleType("vlmeval")
    vlmeval.dataset = types.ModuleType("vlmeval.dataset")
    vlmeval.dataset.videomme_v2 = official
    monkeypatch.setitem(sys.modules, "vlmeval", vlmeval)
    monkeypatch.setitem(sys.modules, "vlmeval.dataset", vlmeval.dataset)
    monkeypatch.setitem(sys.modules, "vlmeval.dataset.videomme_v2", official)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)

    output = tmp_path / "rows.jsonl"
    prepare_videomme_v2(output)
    result = json.loads(output.read_text().splitlines()[0])
    assert result["answer"] == "A"
    assert result["responses_create_params"]["metadata"]["video_path"] == "./video/001.mp4"
    assert downloads and all(revision == DATA_REVISION for _, _, revision in downloads)
