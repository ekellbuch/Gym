#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# critpt: prepare official upstream inputs/references; never synthesize solutions.
# Usage: bash scripts/run_oracles/critpt.sh [--prepare-only]
# TB2.1 phases: prepare -> start -> wait -> upstream oracle -> results -> cleanup.
# Preparation is wired below. Later phases await an established upstream oracle command.
# Source: https://github.com/NVIDIA-NeMo/Gym.git @ 1c8261080bdc881b3e9b7f870e6418f160516991
# See SOURCES.json for the exact reference fields, variant, and remaining requirement.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/critpt.sh [--prepare-only]'
    echo 'Upstream scoring-only recovery command exists: uv run --locked python -m resources_servers.critpt.replay --cache-dir RUN_DIR. It requires cached candidate submissions and ARTIFICIAL_ANALYSIS_API_KEY; it does not supply correct solutions.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
if (( $# )); then echo 'This wrapper accepts no overrides.' >&2; exit 2; fi

# Upstream owns all dataset downloading and reference materialization.
echo '==> Preparing critpt from upstream...'
uv run --locked python -c 'from benchmarks.critpt.prepare import prepare; print(prepare())'
if "$PREPARE_ONLY"; then exit 0; fi

# No replacement replay implementation: leave this boundary explicit.
echo 'UPSTREAM PREPARATION COMPLETE; ORACLE EXECUTION NOT WIRED: Upstream scoring-only recovery command exists: uv run --locked python -m resources_servers.critpt.replay --cache-dir RUN_DIR. It requires cached candidate submissions and ARTIFICIAL_ANALYSIS_API_KEY; it does not supply correct solutions.' >&2
exit 2
