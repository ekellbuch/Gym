# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MathVision scorer from pinned VLMEvalKitMcore mathv.py, isolated from ANTLR 4.9.

The scoring control flow was already ported in internal Gym cfafe99c83c7. Its
latex2sympy2==1.9.1 equality worker preserves the official ANTLR 4.7.2 parser.
The caller must load the matching pinned VLMEvalKitMcore checkout on sys.path.
"""

import json
import logging
import select
import threading
from pathlib import Path
from subprocess import PIPE, Popen, run
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

# --- MathVision latex2sympy subprocess isolation ------------------------------------
# The reference MathVision equality backend (mathv.is_equal) is classic latex2sympy2,
# which pins antlr4-python3-runtime==4.7.2 — mutually exclusive with the
# math-verify/latex2sympy2_extended (antlr 4.9.3+) stack this venv carries for the MCQ
# scorers. mathv.py imports latex2sympy at module level, so it cannot even be imported
# here; the equality check runs in a tiny dedicated venv instead (built lazily next to
# the server venv) via the official_mathvision_sympy_check.py worker, and the reference's
# surrounding logic (post_check / MATH_V_auxeval) is vendored below with the equality
# call delegated to that worker. The shared venv stays on antlr 4.9.3+ untouched.
MATHV_SYMPY_VENV_DIRNAME = ".venv_mathv_sympy_py312"
# Both pins mirror the reference environment: latex2sympy2 is what produced the
# reference numbers (NOT math-verify), and it hard-requires antlr 4.7.2.
MATHV_SYMPY_REQUIREMENTS = ("antlr4-python3-runtime==4.7.2", "latex2sympy2==1.9.1")
# mathv.py wraps is_equal in timeout_decorator.timeout(30); we enforce the same bound
# caller-side (a timeout is verdict False, exactly like the reference's TimeoutError ->
# post_check blanket except -> False).
MATHV_SYMPY_CHECK_TIMEOUT_SECONDS = 30.0


def ensure_mathv_sympy_venv() -> Path:
    """Build (once) and return the python of the dedicated latex2sympy2 venv.

    Created next to the server venv. `uv pip install` resolves from the configured
    index; on air-gapped compute nodes point uv at pre-staged wheels via UV_FIND_LINKS/
    UV_OFFLINE in the deploy hook. A sentinel is written only after an import check
    succeeds, so a partially-built venv is rebuilt instead of trusted.
    """
    this_dir = Path(__file__).parent.absolute()
    venv_dir = this_dir / MATHV_SYMPY_VENV_DIRNAME
    python = venv_dir / "bin" / "python"
    sentinel = venv_dir / ".setup_ok"
    if sentinel.exists():
        return python

    # ANTLR 4.7.2 imports typing.io, which Python 3.13 removed. Keep the exact
    # reference parser in its own Python 3.12 process instead of patching ANTLR.
    run(["uv", "venv", "--allow-existing", "--python", "3.12.14", str(venv_dir)], check=True)
    run(["uv", "pip", "install", "--python", str(python), *MATHV_SYMPY_REQUIREMENTS], check=True)
    run([str(python), "-c", "from latex2sympy2 import latex2sympy"], check=True)
    sentinel.write_text("")
    return python


class MathVisionSympyChecker:
    """Client for the isolated symbolic-equality worker (official_mathvision_sympy_check.py).

    Keeps one persistent worker process and streams JSON lines to it (batched protocol —
    one subprocess for the whole run, not one per row). Calls are serialized behind a
    thread lock (call sites run in to_thread); a timeout or dead worker is killed and
    restarted on the next call.
    """

    def __init__(self) -> None:
        self._proc: Optional[Popen] = None
        self._proc_lock = threading.Lock()

    def ensure_ready(self) -> None:
        """Bootstrap the venv eagerly — callers invoke this OUTSIDE the per-row error
        guards so a broken bootstrap fails verify loudly instead of silently zeroing
        every row."""
        ensure_mathv_sympy_venv()

    def _start(self) -> None:
        python = ensure_mathv_sympy_venv()
        script = Path(__file__).parent.absolute() / "official_mathvision_sympy_check.py"
        # stderr inherits (worker warnings land in the server log); line-buffered text pipes.
        self._proc = Popen([str(python), str(script)], stdin=PIPE, stdout=PIPE, text=True, bufsize=1)

    def _kill(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None

    def is_equal(self, asw: Any, gt_asw: Any, timeout: float = MATHV_SYMPY_CHECK_TIMEOUT_SECONDS) -> bool:
        """Reference mathv.is_equal verdict, computed in the isolated venv.

        Raises on worker/transport failure (TimeoutError on the 30s reference bound);
        the vendored post_check catches and scores False, exactly like the reference.
        """
        with self._proc_lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()
            proc = self._proc
            try:
                proc.stdin.write(json.dumps({"asw": asw, "gt_asw": gt_asw}) + "\n")
                proc.stdin.flush()
                ready, _, _ = select.select([proc.stdout], [], [], timeout)
                if not ready:
                    raise TimeoutError(f"mathv is_equal exceeded {timeout}s (reference timeout bound)")
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("mathvision_sympy_check worker exited unexpectedly")
            except Exception:
                self._kill()
                raise
            verdict = json.loads(line)
            if verdict.get("error"):
                logger.warning("mathvision_sympy_check worker error: %s", verdict["error"])
            return bool(verdict["equal"])


_MATHV_FAIL_MSG = "Failed to obtain answer via API."


def _mathv_get_gpt4_ICE() -> List[str]:
    # Vendored VERBATIM from VLMEvalKitMcore vlmeval/dataset/utils/mathv.py::get_gpt4_ICE
    # (pinned commit 6962c8d06b2b7b26a74a73d6212c06562b63e1b7). mathv.py cannot be
    # imported in this venv (module-level latex2sympy import; antlr clash — see the
    # isolation note above), so the judge prompt scaffolding lives here byte-identical.
    example_1 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]


def _mathv_build_gpt4_prompt(line: Dict[str, Any]) -> str:
    # Vendored VERBATIM from mathv.py::build_mathv_gpt4_prompt (same pinned commit),
    # including the reference's 'Model respone' typo — the judge prompt must be
    # byte-identical for reference comparability.
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""
    question = line["question"]
    prediction = str(line["prediction"])
    prompt = task_description
    examples = _mathv_get_gpt4_ICE()
    for example in examples:
        prompt += example + "\n"
    prompt += question + "\n"
    prompt += "Model respone: " + prediction
    prompt += "Extracted answer:"
    return prompt


def _mathv_list_to_dict(lst: List[Any]) -> Dict[str, Any]:
    # Vendored from mathv.py::list_to_dict (same pinned commit).
    return {chr(65 + i): val for i, val in enumerate(lst)}


def _mathv_post_check(line: Dict[str, Any], is_equal, prefetch: bool = False):
    # Vendored from mathv.py::post_check (same pinned commit), with the latex2sympy
    # equality delegated to the isolated worker via `is_equal` (MathVisionSympyChecker
    # .is_equal). Control flow, eval() usage and the exception envelope are unchanged:
    # any is_equal failure (including the 30s timeout) scores False, like the reference.
    from vlmeval.utils.matching_util import can_infer

    res = None
    ans = line["answer"]
    response = line["prediction"] if prefetch else line["res"]
    try:
        if len(eval(line["choices"])) > 0:
            ans = line["answer"]
            choices = _mathv_list_to_dict(eval(line["choices"]))
            res = can_infer(response, choices)
            if prefetch:
                return res
        else:
            res = str(response)
            ans = str(ans)
    except ValueError:
        pass

    try:
        if is_equal(res, ans):
            return res if prefetch else True
        else:
            return False
    except Exception as err:
        logging.warning(f"{type(err)}: {err}")
        return False


def _mathv_auxeval(model, line: Dict[str, Any], is_equal) -> Dict[str, str]:
    # Vendored from mathv.py::MATH_V_auxeval (same pinned commit), with post_check's
    # equality routed through the isolated worker. The double prefetch call is the
    # reference's own behavior — kept as-is.
    prompt = _mathv_build_gpt4_prompt(line)
    log = ""
    retry = 5
    if _mathv_post_check(line, is_equal, prefetch=True):
        res = _mathv_post_check(line, is_equal, prefetch=True)
        return dict(log="Prefetch succeed", res=res)
    for i in range(retry):
        prediction = line["prediction"]
        res = model.generate(prompt, temperature=i * 0.5)

        if _MATHV_FAIL_MSG in res:
            log += f"Try {i}: output is {prediction}, failed to parse.\n"
        else:
            log += "Succeed"
            return dict(log=log, res=res)
    log += "All 5 retries failed.\n"
    return dict(log=log, res="")


def score_mathvision_reference(line: Dict[str, Any], judge, checker: MathVisionSympyChecker) -> float:
    """Score one official MathVision row; the pinned reference requires a judge."""
    if judge is None:
        raise RuntimeError("MathVision requires the configured gpt-4o-mini extraction judge")
    checker.ensure_ready()
    candidate = dict(line)
    candidate["choices"] = candidate.get("choices") or "[]"
    aux = _mathv_auxeval(judge, candidate, checker.is_equal)
    if aux["log"].endswith("All 5 retries failed.\n"):
        raise RuntimeError("MathVision extraction judge failed all five attempts")
    candidate["res"] = aux["res"]
    return float(bool(_mathv_post_check(candidate, checker.is_equal, prefetch=False)))
