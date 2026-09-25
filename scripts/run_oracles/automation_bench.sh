#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# automation_bench: prepare official upstream inputs/references; never synthesize solutions.
# Usage: bash scripts/run_oracles/automation_bench.sh [--prepare-only] [upstream overrides...]
# TB2.1 phases: prepare -> start -> wait -> upstream oracle -> results -> cleanup.
# Preparation is wired below. Later phases await an established upstream oracle command.
# Source: https://github.com/NVIDIA-NeMo/Gym.git @ 1c8261080bdc881b3e9b7f870e6418f160516991
# See SOURCES.json for the exact reference fields, variant, and remaining requirement.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/automation_bench.sh [--prepare-only] [upstream overrides...]'
    echo 'Official preparation retains answer/info/initial world, while native scoring checks assertions against the world after tool actions. A text answer cannot satisfy that interface; no deterministic upstream reference-action executor is exposed.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi

# Upstream owns all dataset downloading and reference materialization.
echo '==> Preparing benchmark data...'
# Native preparation imports this local environment package; a fresh root Gym
# environment does not include it (benchmarks/automationbench/README.md).
uv run --locked --with-editable benchmarks/automationbench \
    gym eval prepare --config benchmarks/automationbench/config.yaml "$@"
if "$PREPARE_ONLY"; then exit 0; fi

# No replacement replay implementation: leave this boundary explicit.
echo 'PREPARATION COMPLETE; REFERENCE EXECUTION UNAVAILABLE: Official preparation retains answer/info/initial world, while native scoring checks assertions against the world after tool actions. A text answer cannot satisfy that interface; no deterministic upstream reference-action executor is exposed.' >&2
exit 2
