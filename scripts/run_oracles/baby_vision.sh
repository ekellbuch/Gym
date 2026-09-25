#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Published BabyVision answers -> pinned official Mcore judge -> current Gym results.
# This public author ZIP is variant-unverified against the older internal TSV.
# Usage: bash scripts/run_oracles/baby_vision.sh [--prepare-only] [positive-limit]
# Preparation downloads the authors' pinned dataset and pinned Mcore source.
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/baby_vision.sh [--prepare-only] [positive-limit]'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
if [[ $# -gt 1 || ( $# -eq 1 && ! "$1" =~ ^[1-9][0-9]*$ ) ]]; then
    echo 'Expected one positive task limit or --prepare-only.' >&2
    exit 2
fi
LIMIT_ARGS=()
if [[ $# -eq 1 ]]; then LIMIT_ARGS=(+limit="$1"); fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run_oracles/oracle_ports.sh
oracle_ports baby_vision
export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$PWD/scripts/run_oracles/build-constraints.txt}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
LOG_DIR=results/baby_vision_oracle
mkdir -p "$LOG_DIR"

echo '==> Preparing benchmark data...'
uv run --locked python benchmarks/babyvision/prepare.py 2>&1 | tee "$LOG_DIR/prepare.log"
if "$PREPARE_ONLY"; then exit 0; fi
if [[ -z "${INFERENCE_API_KEY:-}" ]]; then
    echo 'INFERENCE_API_KEY is not set in this shell; restart the session.' >&2
    exit 1
fi
# Reject an unauthenticated judge before Mcore's retries can turn it into an apparent zero score.
uv run --locked python scripts/run_oracles/check_judge_auth.py benchmarks/babyvision/config.yaml

echo '==> Starting Gym...'
set -m
uv run --locked gym env start --config benchmarks/babyvision/config.yaml \
    "${ORACLE_GYM_START_PORT_ARGS[@]}" \
    ++babyvision_simple_agent=null ++vlm_eval_kit_simple_agent=null \
    > "$LOG_DIR/server.log" 2>&1 &
GYM_ENV_PID=$!
set +m
cleanup() {
    kill -INT "$GYM_ENV_PID" 2>/dev/null || true
    wait "$GYM_ENV_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo '==> Waiting for Gym readiness...'
SERVER_READY=false
for i in $(seq 1 180); do
    if ! kill -0 "$GYM_ENV_PID" 2>/dev/null; then
        echo "Gym exited; inspect $LOG_DIR/server.log" >&2
        exit 1
    fi
    if uv run --locked gym env status --json "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" 2>/dev/null \
        | uv run --locked python scripts/run_oracles/gym_ready.py; then
        SERVER_READY=true
        break
    fi
    sleep 5
done
if ! "$SERVER_READY"; then echo 'Gym did not become ready within 15 minutes.' >&2; exit 1; fi

echo '==> Verifying official reference answers...'
uv run --locked python scripts/run_oracles/submit_references.py \
    "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" \
    +benchmark_jsonl=benchmarks/babyvision/data/babyvision_benchmark.jsonl \
    +resources_server=babyvision_resources_server +reference_field=answer \
    +output_jsonl=results/baby_vision_oracle.jsonl "${LIMIT_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/oracle.log"
echo '==> Results: results/baby_vision_oracle.jsonl'
