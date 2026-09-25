#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cbrne: prepare official upstream inputs/references; never synthesize solutions.
# Usage: bash scripts/run_oracles/cbrne.sh [--prepare-only] [upstream overrides...]
# TB2.1 phases: prepare -> start -> wait -> upstream oracle -> results -> cleanup.
# Preparation is wired below. Later phases await an established upstream oracle command.
# Source: https://gitlab-master.nvidia.com/dl/nemo/gym @ b1d029e0309417be599b1b66a5846bc818209b6d
# See SOURCES.json for the exact reference fields, variant, and remaining requirement.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/cbrne.sh [--prepare-only] [upstream overrides...]'
    echo 'Run from specified upstream commit; current checkout lacks this benchmark. Dataset requires GitLab registry access. No dedicated upstream oracle execution command found in pinned tree.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
if [[ -z "${GITLAB_TOKEN:-}" ]]; then
    echo 'GITLAB_TOKEN is not set in this shell; restart the session.' >&2
    exit 1
fi

# Keep the reviewed upstream revision isolated; do not port its code into this checkout.
SOURCE_DIR="$PWD/results/oracle-upstream/gym-b1d029e0309417be599b1b66a5846bc818209b6d"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    git init --quiet "$SOURCE_DIR"
    GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.extraHeader \
        GIT_CONFIG_VALUE_0="Authorization: Bearer $GITLAB_TOKEN" \
        git -C "$SOURCE_DIR" fetch --depth=1 https://gitlab-master.nvidia.com/dl/nemo/gym b1d029e0309417be599b1b66a5846bc818209b6d
    git -C "$SOURCE_DIR" checkout --quiet --detach b1d029e0309417be599b1b66a5846bc818209b6d
fi
if [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" != "b1d029e0309417be599b1b66a5846bc818209b6d" ]]; then
    echo 'Upstream source checkout does not match its reviewed revision.' >&2
    exit 1
fi
cd "$SOURCE_DIR"

# Upstream owns all dataset downloading and reference materialization.
echo '==> Preparing cbrne from upstream...'
uv run --locked gym eval prepare --config benchmarks/cbrne/config.yaml \
    '++mlflow_tracking_uri=https://gitlab-master.nvidia.com/api/v4/projects/191584/ml/mlflow' \
    '++mlflow_tracking_token=${oc.env:GITLAB_TOKEN}' \
    ++judge_api_key=dummy \
    ++head_server.port=11251 ++port_range_low=25100 ++port_range_high=25199 "$@"
if "$PREPARE_ONLY"; then exit 0; fi

# No replacement replay implementation: leave this boundary explicit.
echo 'UPSTREAM PREPARATION COMPLETE; ORACLE EXECUTION NOT WIRED: Run from specified upstream commit; current checkout lacks this benchmark. Dataset requires GitLab registry access. No dedicated upstream oracle execution command found in pinned tree.' >&2
exit 2
