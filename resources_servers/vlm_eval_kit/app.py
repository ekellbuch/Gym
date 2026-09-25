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
import hashlib
import importlib
import re
import sys
from asyncio import Event, Semaphore, to_thread
from collections import defaultdict
from pathlib import Path
from subprocess import run
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class VlmEvalKitResourcesServerConfig(BaseResourcesServerConfig):
    # CharXiv uses the official pinned mcore scorer; all existing benchmarks keep
    # the original VLMEvalKit setup when this is unset.
    charxiv_source_path: str | None = None
    mcore_source_path: str | None = None
    judge_model: str = ""
    judge_base_url: str = "https://inference-api.nvidia.com/v1/chat/completions"
    judge_api_key: str = ""
    judge_max_concurrency: int = 8
    extended_user_prompt: str = "none"
    image_resize_scale: float = 1.0
    image_resize_resample: str = "lanczos"
    image_resize_jpeg_quality: int = 95


class VLMEvalKitVerifyRequest(BaseVerifyRequest):
    # We allow extra inputs here since there are many VLMEvalKit benchmarks that are run through the same resources server.
    model_config = ConfigDict(extra="allow")

    benchmark_name: str
    category: str
    answer: Any


class VLMEvalKitVerifyResponse(VLMEvalKitVerifyRequest, BaseVerifyResponse):
    pass


class Coordinator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rewards: List[int] = Field(default_factory=list)
    event: Event = Field(default_factory=Event)


class VlmEvalKitResourcesServer(SimpleResourcesServer):
    config: VlmEvalKitResourcesServerConfig
    _mathvision_checker: Any = PrivateAttr(default=None)
    _mmlongbench_module: Any = PrivateAttr(default=None)
    _videomme_v2_scorer: Any = PrivateAttr(default=None)
    _ocr_reasoning_scorer: Any = PrivateAttr(default=None)

    MMBench_DEV_EN_V11_sets: Dict[str, Coordinator] = Field(default_factory=lambda: defaultdict(Coordinator))

    def model_post_init(self, context):
        super().model_post_init(context)
        self._judge = None
        self._judge_semaphore = Semaphore(value=self.config.judge_max_concurrency)

    def setup_webserver(self):
        if self.config.charxiv_source_path or self.config.mcore_source_path:
            source_dir = Path(self.config.charxiv_source_path or self.config.mcore_source_path)
            if not source_dir.is_absolute():
                source_dir = Path(__file__).parents[2] / source_dir
            if self.config.charxiv_source_path:
                source_file = source_dir / "vlmeval/dataset/charxiv.py"
                source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
                if source_hash != "6348576f7f21c799ccc3954be44e74bc702c1c4afe9543bc0d1ad4cfc12e75da":
                    raise RuntimeError(f"CharXiv scorer source hash mismatch: {source_file}")
            else:
                from resources_servers.vlm_eval_kit.official_mathvista import verify_mathvista_source
                from resources_servers.vlm_eval_kit.official_ocrbench_v2 import verify_ocrbench_v2_source
                from resources_servers.vlm_eval_kit.official_vstar import verify_vstar_source

                verify_vstar_source(source_dir)
                verify_mathvista_source(source_dir)
                verify_ocrbench_v2_source(source_dir)
                source_file = source_dir / "vlmeval/dataset/utils/mathv.py"
                source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
                if source_hash != "c0fa0c5a218ab136dfb668371500f5e6430569fa158cf5e1f1425ba5e2d67fd6":
                    raise RuntimeError(f"MathVision scorer source hash mismatch: {source_file}")
            sys.path.insert(0, str(source_dir))
            module_name = (
                "vlmeval.dataset.charxiv" if self.config.charxiv_source_path else "vlmeval.utils.matching_util"
            )
            imported_scorer = importlib.import_module(module_name)
            if not self.config.charxiv_source_path:
                source_file = source_dir / "vlmeval/utils/matching_util.py"
                source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
            imported_hash = hashlib.sha256(Path(imported_scorer.__file__).read_bytes()).hexdigest()
            if imported_hash != source_hash:
                raise RuntimeError(f"Imported scorer differs from pinned source: {imported_scorer.__file__}")
        else:
            self.setup_VLMEvalKit()

        return super().setup_webserver()

    def setup_VLMEvalKit(self) -> None:
        this_dir = Path(__file__).parent.absolute()
        # We freeze the commit SHA for now.
        # We pip install with no-deps since we have the deps in the pyproject.toml already.
        setup_command = f"""cd {this_dir} \
&& source .venv/bin/activate \
&& if [ ! -d VLMEvalKit ]; then git clone https://github.com/open-compass/VLMEvalKit/; fi \
&& cd VLMEvalKit \
&& git checkout 00804217f868058f871f5ff252a7b9623c3475d9 \
&& uv pip install '-e .' --no-deps \
&& sed -i '' 's/import clip/# import clip/' vlmeval/dataset/utils/SArena/FID.py
"""
        print(f"Running VLMEvalKit setup command: {setup_command}")
        run(setup_command, shell=True, check=True)

        # Dummy import to load ahead of time
        import vlmeval.utils.matching_util

        vlmeval.utils.matching_util

    async def verify(self, body: VLMEvalKitVerifyRequest) -> VLMEvalKitVerifyResponse:
        score_fn = getattr(self, f"_score_{body.benchmark_name}")

        score_dict = await score_fn(body)

        return VLMEvalKitVerifyResponse(**body.model_dump(), **score_dict)

    def _get_charxiv_judge(self):
        if not self.config.judge_model:
            raise RuntimeError("CharXiv requires its configured official judge")
        if self._judge is None:
            from vlmeval.api import OpenAIWrapper

            self._judge = OpenAIWrapper(
                self.config.judge_model,
                api_base=self.config.judge_base_url,
                key=self.config.judge_api_key,
                temperature=0,
                verbose=False,
            )
        return self._judge

    async def _score_CharXiv_reasoning_val(self, body: VLMEvalKitVerifyRequest) -> Dict[str, Any]:
        # Verbatim scoring path from internal Gym 4718d1c5bbbba3a143be60849810da5fc6ab631b:
        # the pinned mcore auxeval owns exact-match precheck, judge prompt and retries.
        from vlmeval.dataset.charxiv import auxeval

        prediction = re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>", "", body.response.output_text or "", flags=re.DOTALL | re.IGNORECASE
        ).strip()
        judge_failed = False
        if not prediction:
            reward, extracted = 0.0, ""
        else:
            line = {"grading_query": body.grading_query, "prediction": prediction, "answer": body.answer}
            try:
                async with self._judge_semaphore:
                    result = await to_thread(auxeval, self._get_charxiv_judge(), line)
                reward = float(result["score"])
                extracted = str(result.get("extract_answer", ""))
            except Exception:
                reward, extracted = 0.0, "Failed to parse response"
            judge_failed = extracted == "Failed to parse response"
        return {
            f"CharXiv_reasoning_val/{body.category}": reward,
            "CharXiv_reasoning_val": reward,
            "reward": reward,
            "extract_answer": extracted,
            "mask_sample": judge_failed,
            "failure_kind": "judge_failed" if judge_failed else None,
            "failure_reason": "CharXiv judge did not return a valid score" if judge_failed else None,
        }

    async def _score_VStarBench(self, body: VLMEvalKitVerifyRequest) -> Dict[str, Any]:
        from resources_servers.vlm_eval_kit.official_vstar import score_vstar

        return await score_vstar(body, judge=self._get_charxiv_judge(), judge_semaphore=self._judge_semaphore)

    async def _score_MathVista_MINI(self, body: VLMEvalKitVerifyRequest) -> Dict[str, Any]:
        from resources_servers.vlm_eval_kit.official_mathvista import score_mathvista

        return await score_mathvista(body, judge=self._get_charxiv_judge(), judge_semaphore=self._judge_semaphore)

    async def _score_MathVision(self, body: VLMEvalKitVerifyRequest) -> Dict[str, Any]:
        from resources_servers.vlm_eval_kit.official_mathvision import (
            MathVisionSympyChecker,
            score_mathvision_reference,
        )

        if self._mathvision_checker is None:
            self._mathvision_checker = MathVisionSympyChecker()
        prediction = re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>", "", body.response.output_text or "", flags=re.DOTALL | re.IGNORECASE
        ).strip()
        line = {"prediction": prediction, "answer": body.answer, "question": body.question, "choices": body.choices}
        async with self._judge_semaphore:
            reward = await to_thread(
                score_mathvision_reference, line, self._get_charxiv_judge(), self._mathvision_checker
            )
        return {f"MathVision/{body.category}": reward, "MathVision": reward, "reward": reward}

    async def _score_MMLongBench_DOC(self, body: VLMEvalKitVerifyRequest) -> Dict[str, Any]:
        from resources_servers.vlm_eval_kit.official_mmlongbench_doc import (
            load_official_scorer,
            score_mmlongbench_reference,
        )

        if self._mmlongbench_module is None:
            source_dir = Path(self.config.mcore_source_path)
            if not source_dir.is_absolute():
                source_dir = Path(__file__).parents[2] / source_dir
            self._mmlongbench_module = load_official_scorer(source_dir)
        prediction = re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>", "", body.response.output_text or "", flags=re.DOTALL | re.IGNORECASE
        ).strip()
        line = {
            "question": body.question,
            "prediction": prediction,
            "answer": body.answer,
            "answer_format": body.answer_format,
        }
        async with self._judge_semaphore:
            result = await to_thread(
                score_mmlongbench_reference, line, self._get_charxiv_judge(), self._mmlongbench_module
            )
        reward = result["reward"]
        return {
            f"MMLongBench_DOC/{body.category}": reward,
            "MMLongBench_DOC": reward,
            "reward": reward,
            "pred": result["pred"],
            "invalid_judge_response": result["invalid_judge_response"],
        }

    async def _score_OCRBench_v2(self, body: VLMEvalKitVerifyRequest) -> Dict[str, Any]:
        from resources_servers.vlm_eval_kit.official_ocrbench_v2 import score_ocrbench_v2

        return await score_ocrbench_v2(body)

    async def _score_OCR_Reasoning(self, body: VLMEvalKitVerifyRequest) -> Dict[str, Any]:
        from resources_servers.vlm_eval_kit.official_ocr_reasoning import (
            load_official_scorer,
            score_ocr_reasoning_reference,
        )

        if self._ocr_reasoning_scorer is None:
            source_dir = Path(self.config.mcore_source_path)
            if not source_dir.is_absolute():
                source_dir = Path(__file__).parents[2] / source_dir
            self._ocr_reasoning_scorer = load_official_scorer(source_dir)
        prediction = re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>", "", body.response.output_text or "", flags=re.DOTALL | re.IGNORECASE
        ).strip()
        line = {
            "prediction": prediction,
            "question": body.question,
            "answer": body.answer,
            "reasoning": body.reasoning,
            "question_type": body.question_type,
            "answer_type": body.answer_type,
            "choices": body.choices,
            "answer_option": body.answer_option,
        }
        async with self._judge_semaphore:
            result = await to_thread(
                score_ocr_reasoning_reference, line, self._get_charxiv_judge(), self._ocr_reasoning_scorer
            )
        reward = result["reward"]
        return {
            f"OCR_Reasoning/{body.category}": reward,
            "OCR_Reasoning": reward,
            "reward": reward,
            "reason_score": result["reason_score"],
            "invalid_judge_response": result["invalid_judge_response"],
            **({"judge_error": result["judge_error"]} if "judge_error" in result else {}),
        }

    async def _score_BabyVision(self, body: VLMEvalKitVerifyRequest) -> Dict[str, Any]:
        from resources_servers.vlm_eval_kit.official_babyvision import score_babyvision

        return await score_babyvision(body, self._get_charxiv_judge(), self._judge_semaphore)

    async def _score_Video_MME_v2(self, body: VLMEvalKitVerifyRequest) -> Dict[str, Any]:
        from resources_servers.vlm_eval_kit.official_videomme_v2 import (
            load_official_scorer,
            score_videomme_v2_reference,
        )

        if self._videomme_v2_scorer is None:
            source_dir = Path(self.config.mcore_source_path)
            if not source_dir.is_absolute():
                source_dir = Path(__file__).parents[2] / source_dir
            self._videomme_v2_scorer = load_official_scorer(source_dir)
        prediction = re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>", "", body.response.output_text or "", flags=re.DOTALL | re.IGNORECASE
        ).strip()
        result = score_videomme_v2_reference(body.answer, prediction, self._videomme_v2_scorer)
        return {
            f"Video-MME-v2/{body.category}": result["reward"],
            "Video-MME-v2": result["reward"],
            "reward": result["reward"],
            "score": result["score"],
            "prediction": prediction,
        }

    # For each of the scoring functions, we copy it over in a nicer way since the original functions
    # couple together reading from an input file path, LLM as judge, etc. It's just easier to reimplement and test e2e accuracy.
    async def _score_OCRBench(self, body: BaseVerifyRequest) -> Dict[str, Any]:
        # Reformatted from https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/dataset/image_vqa.py#L505
        reward = 0.0

        predict = body.response.output_text
        answers = body.answer
        category = body.category
        if category == "Handwritten Mathematical Expression Recognition":
            for j in range(len(answers)):
                answer = answers[j].strip().replace("\n", " ").replace(" ", "")
                predict = predict.strip().replace("\n", " ").replace(" ", "")
                if answer in predict:
                    reward = 1.0
                    break
        else:
            for j in range(len(answers)):
                answer = answers[j].lower().strip().replace("\n", " ")
                predict = predict.lower().strip().replace("\n", " ")
                if answer in predict:
                    reward = 1.0
                    break

        return {f"OCRBench/{category}": reward, "OCRBench": reward, "reward": reward}

    async def _score_MMBench_DEV_EN_V11(self, body: BaseVerifyRequest) -> Dict[str, Any]:
        # Reformatted from https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/dataset/image_mcq.py#L294
        # Each example is run 4 times and we only output score 1 if all examples are correct.
        from vlmeval.utils.matching_util import can_infer

        predict = body.response.output_text
        answer = body.answer
        category = body.category

        # Choices looks like https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/dataset/utils/multiple_choice.py#L337
        prediction = can_infer(predict, body.choices)
        this_reward = int(prediction == answer)

        coordinator = self.MMBench_DEV_EN_V11_sets[body.group]
        coordinator.rewards.append(this_reward)
        if len(coordinator.rewards) == body.group_size:
            coordinator.rewards = [int(all(coordinator.rewards))]
            self.MMBench_DEV_EN_V11_sets.pop(body.group)
            coordinator.event.set()
        else:
            await coordinator.event.wait()

        # Just take the first one since that's what we set
        reward = coordinator.rewards[0]

        # We need to return a group-level reward. Here we mark the returned reward as unweighted.
        return {f"MMBench_DEV_EN_V11/unweighted/{category}": reward, "reward": reward}

    def _aggregate_MMBench_DEV_EN_V11(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        grouped_tasks: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for group in tasks:
            for task in group:
                if task["benchmark_name"] == "MMBench_DEV_EN_V11":
                    grouped_tasks[task["group"]].append(task)

        if not grouped_tasks:
            return dict()

        # All rewards are the same for items within a group
        rewards = [group[0]["reward"] for group in grouped_tasks.values()]
        return {
            "MMBench_DEV_EN_V11": sum(rewards) / len(rewards),
        }

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        metrics = self._aggregate_MMBench_DEV_EN_V11(tasks)
        mmlong_rows = [row for group in tasks for row in group if row.get("benchmark_name") == "MMLongBench_DOC"]
        if mmlong_rows and not any(row.get("invalid_judge_response") for row in mmlong_rows):
            from resources_servers.vlm_eval_kit.official_mmlongbench_doc import mmlongbench_f1

            metrics["MMLongBench_DOC_F1"] = mmlongbench_f1(mmlong_rows)
        ocr_rows = [row for group in tasks for row in group if row.get("benchmark_name") == "OCRBench_v2"]
        if ocr_rows:
            from vlmeval.dataset.utils.ocrbrnch_v2_eval import ocrbench_v2_aggregate_accuracy

            scored = [
                {
                    "type": row["category"],
                    "score": row["reward"],
                    **({"ignore": row["ignore"]} if "ignore" in row else {}),
                }
                for row in ocr_rows
            ]
            en_scores, cn_scores = ocrbench_v2_aggregate_accuracy(scored)
            metrics.update({f"OCRBench_v2/{key}": value for key, value in en_scores.items()})
            metrics.update({f"OCRBench_v2/{key}": value for key, value in cn_scores.items()})
            # Aggregation may receive a selected subset. Reserve the official
            # Overall Score labels for a separate full-split coverage check.
            if len(en_scores) == 8:
                metrics["OCRBench_v2/selected_en_group_mean"] = sum(en_scores.values()) / 8
            if len(cn_scores) == 5:
                metrics["OCRBench_v2/selected_cn_group_mean"] = sum(cn_scores.values()) / 5
        video_rows = [row for group in tasks for row in group if row.get("benchmark_name") == "Video_MME_v2"]
        if video_rows:
            from resources_servers.vlm_eval_kit.official_videomme_v2 import official_group_rating

            source_dir = Path(self.config.mcore_source_path)
            if not source_dir.is_absolute():
                source_dir = Path(__file__).parents[2] / source_dir
            if self._videomme_v2_scorer is None:
                from resources_servers.vlm_eval_kit.official_videomme_v2 import load_official_scorer

                self._videomme_v2_scorer = load_official_scorer(source_dir)
            rating = official_group_rating(video_rows, self._videomme_v2_scorer)
            metrics["Video-MME-v2/selected_group_rating"] = rating["final_rating"]["total"]
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "mean/OCRBench",
            "MMBench_DEV_EN_V11",
        ]
        return {k: agent_metrics[k] for k in keys if k in agent_metrics}


if __name__ == "__main__":
    VlmEvalKitResourcesServer.run_webserver()
