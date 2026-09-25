#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# gdpval_aa_v2: prepare official upstream inputs/references; never synthesize solutions.
# Usage: bash scripts/run_oracles/gdpval_aa_v2.sh [--prepare-only]
# TB2.1 phases: prepare -> start -> wait -> upstream oracle -> results -> cleanup.
# Preparation is wired below. The pinned Gym recipe is model-driven and its
# comparison mode needs reference-model deliverables not included in GDPVal.
# Source: https://github.com/NVIDIA-NeMo/Gym.git @ 1c8261080bdc881b3e9b7f870e6418f160516991
# See nemotron_recipes/lightning-3.5/instruct/gym/gdpval/gdpval.sh and
# benchmarks/gdpval/README.md for the upstream comparison-mode inputs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/gdpval_aa_v2.sh [--prepare-only]'
    echo 'Preparation writes 220 task prompts and rubrics. The official dataset publishes some expert deliverables, but AA-v2 uses pairwise Elo. Gym has the scorer and model-rating anchors, yet its recipe requires separately generated reference-model deliverables and runs a policy model; no model-free oracle runner is provided.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
if [[ "$#" -ne 0 ]]; then
    echo "Unexpected argument: $1" >&2
    exit 2
fi

# Upstream owns all dataset downloading and reference materialization.
echo '==> Preparing gdpval_aa_v2 from upstream...'
uv run --locked python -c 'from benchmarks.gdpval.prepare import prepare; print(prepare())'
if "$PREPARE_ONLY"; then exit 0; fi

# Rubric-grading an expert deliverable would produce a different metric. The
# upstream comparison recipe requires policy-generated deliverables and rated
# reference-model submissions; it cannot be used as a model-free oracle.
echo 'UPSTREAM PREPARATION COMPLETE; ORACLE EXECUTION UNSUPPORTED: No official model-free GDPVal-AA-v2 runner or reference-model deliverable set was found. Native pairwise Elo needs rated reference submissions; expert rubric grading is not the AA-v2 score. No oracle score was produced.' >&2
exit 2
