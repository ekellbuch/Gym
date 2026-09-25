#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# tau3_bank: prepare official upstream inputs/references; never synthesize solutions.
# Usage: bash scripts/run_oracles/tau3_bank.sh [--prepare-only]
# TB2.1 phases: prepare -> start -> wait -> upstream oracle -> results -> cleanup.
# Preparation is wired below. The pinned Tau source has no model-free banking
# oracle runner, so this script stops before server startup and scoring.
# Data source: https://github.com/bxyu-nvidia/tau2-bench.git @ ce4013b0afe03c873488878b72851414f92f458b
# Gym agent source: same repository @ 60c2a0dbf974ea7533456a4706f837c3a6d14afc
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/tau3_bank.sh [--prepare-only]'
    echo 'Preparation produces 97 banking_knowledge/bm25_grep tasks. No model-free banking oracle runner exists in the pinned Tau source; llm_agent_gt still calls an LLM. Preparation removes NL_ASSERTION from task_102, so Gym scores would differ from the AA scope.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
if [[ "$#" -ne 0 ]]; then
    echo "Unexpected argument: $1" >&2
    exit 2
fi

# Upstream owns all dataset downloading and reference materialization.
echo '==> Preparing tau3_bank from upstream...'
NEMO_GYM_TAU2_BENCH_DATA_REF=ce4013b0afe03c873488878b72851414f92f458b \
    uv run --locked python -c 'from benchmarks.tau2.prepare_utils.banking_bm25_grep import prepare; print(prepare())'
if "$PREPARE_ONLY"; then exit 0; fi

# The upstream gold-plan agent invokes a model. Reference actions are evaluator
# inputs, and replaying them as assistant calls would be a new oracle adapter.
echo 'UPSTREAM PREPARATION COMPLETE; ORACLE EXECUTION UNSUPPORTED: Pinned Tau source has no model-free banking oracle runner. llm_agent_gt calls an LLM, and Gym run_single_task is model-driven. No oracle score was produced.' >&2
exit 2
