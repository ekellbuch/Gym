# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""MathVision symbolic-equality worker (runs in the dedicated latex2sympy2 venv).

The reference MathVision scorer (VLMEvalKitMcore ``vlmeval/dataset/utils/mathv.py``)
decides open-ended answers via classic ``latex2sympy2`` symbolic equality, which pins
``antlr4-python3-runtime==4.7.2`` — mutually exclusive with the
``math-verify``/``latex2sympy2_extended`` (antlr 4.9.3+) stack the shared vlm_eval_kit
venv carries for the MCQ scorers. This script is therefore executed in a tiny dedicated
venv (see ``ensure_mathv_sympy_venv`` in ``official_mathvision.py``) containing ONLY
``antlr4-python3-runtime==4.7.2`` + ``latex2sympy2``; the shared venv stays untouched.

Protocol (batched, line-oriented — the process stays alive across requests):
  stdin:  one JSON object per line: {"asw": <prediction>, "gt_asw": <reference>}
  stdout: one JSON object per line: {"equal": <bool>} (plus "error" on internal failure)

Malformed LaTeX / malformed JSON never kill the worker: the reference ``is_equal``
swallows parse errors (verdict False), and the loop answers {"equal": false, "error": ...}
for anything else. The caller enforces the reference's 30s per-check bound (mathv.py wraps
``is_equal`` in ``timeout_decorator.timeout(30)``; a timeout means verdict False there
too, via post_check's blanket except).
"""

import json
import sys
from contextlib import redirect_stdout

from latex2sympy2 import latex2sympy


def is_equal(asw: str, gt_asw: str) -> bool:
    # Vendored VERBATIM from VLMEvalKitMcore vlmeval/dataset/utils/mathv.py::is_equal
    # (pinned commit 6962c8d06b2b7b26a74a73d6212c06562b63e1b7), minus the
    # timeout_decorator wrapper (the caller enforces the same 30s bound). Do not
    # "fix" the quirks (the odd isinstance guard, eval-based numeric compare) —
    # byte-level parity with the scorer that produced the reference numbers is the point.
    if not isinstance(asw, str) != str or not isinstance(gt_asw, str):
        print("Warning: input is not string")
        print(asw, gt_asw)
    asw = str(asw).lower().strip()
    gt_asw = str(gt_asw).lower().strip()
    if gt_asw == asw:
        return True
    try:
        a = eval(gt_asw)
        b = eval(asw)
        if abs(a - b) < 1e-6:
            return True
    except:  # noqa: E722
        pass
    try:
        a = latex2sympy(gt_asw)
        b = latex2sympy(asw)
        if abs(eval(str(a)) - eval(str(b))) < 1e-6:
            return True
        if abs(a - b) < 1e-6:
            return True
    except:  # noqa: E722
        pass
    return False


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            # Anything is_equal prints (its non-string warning) must not corrupt the
            # line protocol on stdout — divert it to stderr (the server log).
            with redirect_stdout(sys.stderr):
                verdict = {"equal": bool(is_equal(request["asw"], request["gt_asw"]))}
        except Exception as err:
            verdict = {"equal": False, "error": f"{type(err).__name__}: {err}"}
        sys.stdout.write(json.dumps(verdict) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
