#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# screenspot_pro: prepare official upstream inputs/references; never synthesize solutions.
# Usage: bash scripts/run_oracles/screenspot_pro.sh [--prepare-only] [upstream overrides...]
# TB2.1 phases: prepare -> start -> wait -> upstream oracle -> results -> cleanup.
# Preparation is wired below. Later phases await an established upstream oracle command.
# Source: https://gitlab-master.nvidia.com/dl/nemo/gym.git @ cfafe99c83c712033dc6b8105bbe7a409c592574
# See SOURCES.json for the exact reference fields, variant, and remaining requirement.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/screenspot_pro.sh [--prepare-only] [upstream overrides...]'
    echo 'Native preparation exists in pinned internal Gym commit, absent current main. No native oracle/replay command found; prepare preserves references but does not submit them as model responses.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi

# Keep the reviewed upstream revision isolated; do not port its code into this checkout.
SOURCE_DIR="$PWD/results/oracle-upstream/gym-cfafe99c83c712033dc6b8105bbe7a409c592574"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    if [[ -z "${GITLAB_TOKEN:-}" ]]; then
        echo 'GITLAB_TOKEN is not set in this shell; restart the session.' >&2
        exit 1
    fi
    git init --quiet "$SOURCE_DIR"
    GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.extraHeader \
        GIT_CONFIG_VALUE_0="Authorization: Bearer $GITLAB_TOKEN" \
        git -C "$SOURCE_DIR" fetch --depth=1 https://gitlab-master.nvidia.com/dl/nemo/gym.git cfafe99c83c712033dc6b8105bbe7a409c592574
    git -C "$SOURCE_DIR" checkout --quiet --detach cfafe99c83c712033dc6b8105bbe7a409c592574
fi
if [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" != "cfafe99c83c712033dc6b8105bbe7a409c592574" ]]; then
    echo 'Upstream source checkout does not match its reviewed revision.' >&2
    exit 1
fi
cd "$SOURCE_DIR"

# Upstream owns all dataset downloading and reference materialization.
echo '==> Preparing screenspot_pro from upstream...'
uv run --locked python benchmarks/screenspot_pro/prepare.py "$@"
if "$PREPARE_ONLY"; then exit 0; fi

# No replacement replay implementation: leave this boundary explicit.
echo 'UPSTREAM PREPARATION COMPLETE; ORACLE EXECUTION NOT WIRED: Native preparation exists in pinned internal Gym commit, absent current main. No native oracle/replay command found; prepare preserves references but does not submit them as model responses.' >&2
exit 2
