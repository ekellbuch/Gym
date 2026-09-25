#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Prepare official AA-LCR v1.1 answers, start its native Gym judge and verifier,
# then submit each unchanged answer as the candidate response to /verify.
# Usage: aa_lcr.sh [--prepare-only] [limit] [submission overrides...]
# The optional positive limit selects rows; remaining arguments go to submit_references.py.
# Reference requests run in parallel. INFERENCE_API_KEY supplies JUDGE_API_KEY.
# Results: results/aa_lcr_oracle.jsonl; logs: results/aa_lcr_oracle/.
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/aa_lcr.sh [--prepare-only] [limit] [submission overrides...]'
    echo 'Example: bash scripts/run_oracles/aa_lcr.sh 2'
    echo 'Uses unchanged official answers and the native AA-LCR v1.1 judge; requires INFERENCE_API_KEY.'
    echo 'Preparation needs no judge key. Results: results/aa_lcr_oracle.jsonl; logs: results/aa_lcr_oracle/.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
LIMIT_ARGS=()
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    if (( 10#$1 < 1 )); then echo 'The task limit must be positive.' >&2; exit 2; fi
    LIMIT_ARGS=(+limit="$((10#$1))")
    shift
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run_oracles/oracle_ports.sh
oracle_ports aa_lcr
REPO_ROOT="$PWD"
export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$REPO_ROOT/scripts/run_oracles/build-constraints.txt}"
BENCHMARK_JSONL=benchmarks/aalcr/data/aalcr_v1_1_benchmark.jsonl
OUTPUT_JSONL=results/aa_lcr_oracle.jsonl
LOG_DIR=results/aa_lcr_oracle
mkdir -p "$LOG_DIR"

echo "==> Preparing benchmark data..."
uv run --locked gym eval prepare --config benchmarks/aalcr/config_v1_1.yaml \
    2>&1 | tee "$LOG_DIR/prepare.log"
if "$PREPARE_ONLY"; then exit 0; fi
if [[ -z "${INFERENCE_API_KEY:-}" ]]; then
    echo 'INFERENCE_API_KEY is not set in this shell; restart the session.' >&2
    exit 1
fi
export JUDGE_API_KEY="$INFERENCE_API_KEY"

# Null disables only the policy-generation agent. Both native resource and judge
# instances retain their benchmark config, including official_v1_1 scoring.
echo '==> Starting native AA-LCR v1.1 judge and resource server...'
set -m
uv run --locked gym env start --config benchmarks/aalcr/config_v1_1.yaml \
    "${ORACLE_GYM_START_PORT_ARGS[@]}" \
    ++aalcr_benchmark_simple_agent=null > "$LOG_DIR/server.log" 2>&1 &
GYM_ENV_PID=$!
set +m
cleanup() {
    # SIGINT runs Gym's own worker shutdown; job control keeps it enabled.
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
        echo "Gym exited before readiness; see $LOG_DIR/server.log" >&2
        exit 1
    fi
    if uv run --locked gym env status --json "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" 2>/dev/null \
        | uv run --locked python scripts/run_oracles/gym_ready.py; then
        SERVER_READY=true
        break
    fi
    sleep 5
done
if ! "$SERVER_READY"; then echo "Gym was not ready after 15 minutes; see $LOG_DIR/server.log" >&2; exit 1; fi

echo '==> Submitting official AA-LCR answers concurrently to the native verifier...'
uv run --locked python scripts/run_oracles/submit_references.py \
    "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" \
    +benchmark_jsonl="$BENCHMARK_JSONL" \
    +resources_server=aalcr_benchmark_resources_server \
    +reference_field=answer \
    +output_jsonl="$OUTPUT_JSONL" \
    "${LIMIT_ARGS[@]}" "$@" 2>&1 | tee "$LOG_DIR/oracle.log"
echo "==> Results: $OUTPUT_JSONL; logs: $LOG_DIR"
