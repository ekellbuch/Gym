#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Published Video-MME-v2 answer letters -> native Gym verifier -> per-row results.
# Usage: bash scripts/run_oracles/videomme_v2.sh [--prepare-only] [positive-limit]
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/videomme_v2.sh [--prepare-only] [positive-limit]'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
if [[ $# -gt 1 || ( $# -eq 1 && ! "$1" =~ ^[1-9][0-9]*$ ) ]]; then
    echo 'Expected one positive task limit or --prepare-only.' >&2
    exit 2
fi
LIMIT_ARGS=()
if [[ $# -eq 1 ]]; then
    if (( $1 % 4 != 0 )); then
        echo 'Video-MME-v2 task limit must contain complete four-question groups.' >&2
        exit 2
    fi
    LIMIT_ARGS=(+limit="$1")
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run_oracles/oracle_ports.sh
oracle_ports videomme_v2
export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$PWD/scripts/run_oracles/build-constraints.txt}"
mkdir -p results/videomme2_oracle
echo '==> Preparing benchmark data...'
uv run --locked python benchmarks/videomme2/prepare.py 2>&1 | tee results/videomme2_oracle/prepare.log
if "$PREPARE_ONLY"; then exit 0; fi
echo '==> Starting Gym...'
set -m
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --locked gym env start --config benchmarks/videomme2/config.yaml \
    "${ORACLE_GYM_START_PORT_ARGS[@]}" \
    ++videomme2_simple_agent=null ++vlm_eval_kit_simple_agent=null \
    > results/videomme2_oracle/server.log 2>&1 &
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
        echo 'Gym exited; inspect results/videomme2_oracle/server.log' >&2
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
echo '==> Submitting official reference answers...'
uv run --locked python scripts/run_oracles/submit_references.py \
    "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" \
    +benchmark_jsonl=benchmarks/videomme2/data/videomme2_benchmark.jsonl \
    +resources_server=videomme2_resources_server +reference_field=answer \
    +output_jsonl=results/videomme2_oracle.jsonl "${LIMIT_ARGS[@]}" \
    2>&1 | tee results/videomme2_oracle/oracle.log
echo '==> Aggregating native Gym results...'
uv run --locked gym eval aggregate \
    --input-glob results/videomme2_oracle.jsonl \
    --output results/videomme2_oracle.jsonl \
    +merge_shards=false 2>&1 | tee results/videomme2_oracle/aggregate.log
echo '==> Computing pinned official Video-MME-v2 group rating...'
uv run --locked --project resources_servers/vlm_eval_kit python -m scripts.run_oracles.aggregate_videomme_v2 \
    --input results/videomme2_oracle.jsonl \
    --output results/videomme2_oracle_group_rating.json \
    2>&1 | tee results/videomme2_oracle/group_rating.log
echo '==> Results: results/videomme2_oracle.jsonl; official group rating: results/videomme2_oracle_group_rating.json'
