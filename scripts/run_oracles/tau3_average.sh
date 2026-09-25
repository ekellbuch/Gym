#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Prepare the four domains used by Gym's Tau3-Average: airline, retail, telecom,
# and banking_knowledge/bm25_grep. Official definition:
# benchmarks/nemotron_3.5_super/export_to_csv.py (equal mean of domain rewards).
# Usage: bash scripts/run_oracles/tau3_average.sh [--prepare-only]
# Official action lists are present; the native agent has no deterministic
# reference-action runner. llm_agent_gt generates model responses, not a replay.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/tau3_average.sh [--prepare-only]'
    echo 'Prepare official airline, retail, telecom, and Banking bm25_grep data.'
    echo 'Execution requires an upstream deterministic action runner; llm_agent_gt calls a model.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
if (( $# )); then echo "Unexpected argument: $1" >&2; exit 2; fi
mkdir -p results/tau3_average_oracle
echo '==> Preparing benchmark data...'
uv run --locked python benchmarks/tau2/prepare.py tau2 \
    2>&1 | tee results/tau3_average_oracle/prepare.log
uv run --locked python benchmarks/tau2/prepare.py banking_knowledge --retrieval-config bm25_grep \
    2>&1 | tee -a results/tau3_average_oracle/prepare.log
if "$PREPARE_ONLY"; then exit 0; fi
echo 'PREPARATION COMPLETE; DETERMINISTIC RUNNER MISSING: native Gym calls tau2.runner.batch.run_single_task; upstream llm_agent_gt generates model responses. Published actions are evaluator criteria, not an executable Gym oracle trajectory.' >&2
exit 2
