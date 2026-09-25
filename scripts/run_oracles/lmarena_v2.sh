#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# lmarena_v2: prepare official upstream inputs/references; never synthesize solutions.
# Usage: bash scripts/run_oracles/lmarena_v2.sh [--prepare-only] [upstream overrides...]
# TB2.1 phases: prepare -> start -> wait -> upstream oracle -> results -> cleanup.
# Preparation is wired below. Later phases await an established upstream oracle command.
# Source: https://github.com/NVIDIA-NeMo/Gym.git @ 1c8261080bdc881b3e9b7f870e6418f160516991
# See SOURCES.json for the exact reference fields, variant, and remaining requirement.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/lmarena_v2.sh [--prepare-only] [upstream overrides...]'
    echo 'The registry artifact supplies baseline answers for pairwise judging, not known-correct answers. A published reference response set and its native scoring procedure are needed for an oracle.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
if [[ -z "${GITLAB_TOKEN:-}" ]]; then
    echo 'GITLAB_TOKEN is not set in this shell; restart the session.' >&2
    exit 1
fi

# Upstream owns all dataset downloading and reference materialization.
echo '==> Preparing lmarena_v2 from upstream...'
uv run --locked gym eval prepare --config benchmarks/lmarena_v2/config.yaml \
    '++mlflow_tracking_uri=https://gitlab-master.nvidia.com/api/v4/projects/191584/ml/mlflow' \
    '++mlflow_tracking_token=${oc.env:GITLAB_TOKEN}' \
    ++head_server.port=11249 ++port_range_low=24900 ++port_range_high=24999 "$@"
if "$PREPARE_ONLY"; then exit 0; fi

# No replacement replay implementation: leave this boundary explicit.
echo 'UPSTREAM PREPARATION COMPLETE; ORACLE EXECUTION UNAVAILABLE: The registry artifact supplies baseline answers for pairwise judging, not known-correct answers. A published reference response set and its native scoring procedure are needed for an oracle.' >&2
exit 2
