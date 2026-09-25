#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ifbench: prepare official upstream inputs/references; never synthesize solutions.
# Usage: bash scripts/run_oracles/ifbench.sh [--prepare-only] [upstream overrides...]
# TB2.1 phases: prepare -> start -> wait -> upstream oracle -> results -> cleanup.
# Preparation is wired below. Later phases await an established upstream oracle command.
# Source: https://github.com/NVIDIA-NeMo/Gym.git @ 1c8261080bdc881b3e9b7f870e6418f160516991
# See SOURCES.json for the exact reference fields, variant, and remaining requirement.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/ifbench.sh [--prepare-only] [upstream overrides...]'
    echo 'Official IFBench data supplies prompts, instruction_id_list, and kwargs, but no known-correct response text to submit. Native IFBench grading requires a candidate response; generating one would invent a solution.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi

# Upstream owns all dataset downloading and reference materialization.
echo '==> Preparing benchmark data...'
uv run --locked gym eval prepare --config benchmarks/ifbench/config.yaml "$@"
if "$PREPARE_ONLY"; then exit 0; fi

# No replacement replay implementation: leave this boundary explicit.
echo 'PREPARATION COMPLETE; REFERENCE EXECUTION UNAVAILABLE: Official IFBench data supplies prompts, instruction_id_list, and kwargs, but no known-correct response text to submit. Native IFBench grading requires a candidate response; generating one would invent a solution.' >&2
exit 2
