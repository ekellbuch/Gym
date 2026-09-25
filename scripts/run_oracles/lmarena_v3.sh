#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# lmarena_v3: prepare official upstream inputs; no upstream oracle is available.
# Usage: bash scripts/run_oracles/lmarena_v3.sh [--prepare-only] [upstream overrides...]
# TB2.1 phases: prepare -> start -> wait -> upstream oracle -> results -> cleanup.
# Preparation is wired below. The Gym benchmark evaluates generated policy
# responses with a judge; it has no official command to score saved answers as
# oracle responses. The dataset's human verdicts are metadata, not oracle scores.
# Source: https://github.com/NVIDIA-NeMo/Gym.git @ 1c8261080bdc881b3e9b7f870e6418f160516991
# See SOURCES.json for the exact reference fields, variant, and remaining requirement.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/lmarena_v3.sh [--prepare-only] [upstream overrides...]'
    echo 'Only preparation is supported. Gym has no official LMArena v3 oracle runner for the saved dataset answers.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
if [[ -z "${GITLAB_TOKEN:-}" ]]; then
    echo 'GITLAB_TOKEN is not set in this shell; restart the session.' >&2
    exit 1
fi

# Upstream owns all dataset downloading and reference materialization.
echo '==> Preparing lmarena_v3 from upstream...'
uv run --locked gym eval prepare --config benchmarks/lmarena_v3/config.yaml \
    '++mlflow_tracking_uri=https://gitlab-master.nvidia.com/api/v4/projects/191584/ml/mlflow' \
    '++mlflow_tracking_token=${oc.env:GITLAB_TOKEN}' \
    ++head_server.port=11250 ++port_range_low=25000 ++port_range_high=25099 "$@"
if "$PREPARE_ONLY"; then exit 0; fi

# No replacement replay implementation: leave this boundary explicit.
echo 'UPSTREAM PREPARATION COMPLETE; ORACLE EXECUTION NOT WIRED: Gym has no official LMArena v3 oracle runner for the saved dataset answers. Its supported evaluation generates policy responses and calls a judge.' >&2
exit 2
