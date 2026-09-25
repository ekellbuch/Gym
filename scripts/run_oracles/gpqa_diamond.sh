#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Prepare official GPQA Diamond references, start the native MCQA verifier, and
# submit references through Gym. No generation model is used. The optional
# positive limit selects rows; no other arguments are forwarded.
# Submission uses the official prompt's Answer: X format with the unchanged
# reference letter; native MCQA grading stays unchanged.
# Results and preparation/server/submission logs: results/gpqa_diamond_oracle*.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run_oracles/oracle_ports.sh
oracle_ports gpqa_diamond
export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$PWD/scripts/run_oracles/build-constraints.txt}"
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [--prepare-only] [positive-task-limit]"
    echo 'Prepare official references and verify them through native Gym MCQA grading.'
    echo 'Requires uv. Results/logs: results/gpqa_diamond_oracle*.'
    echo 'Example: bash scripts/run_oracles/gpqa_diamond.sh 2'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
LIMIT="${1:-}"
if [[ $# -gt 1 || ( -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ) ]]; then
    echo "Usage: $0 [--prepare-only] [positive-task-limit]" >&2
    exit 2
fi
mkdir -p results
echo "==> Preparing benchmark data..."
uv run --locked python benchmarks/gpqa/prepare.py \
    2>&1 | tee results/gpqa_diamond_oracle.prepare.log
if "$PREPARE_ONLY"; then exit 0; fi

# Enable SIGINT for this background job so Gym shuts down its child servers.
echo '==> Starting native GPQA Diamond verifier...'
set -m
uv run --locked gym env start --config scripts/run_oracles/configs/gpqa_diamond.yaml \
    "${ORACLE_GYM_START_PORT_ARGS[@]}" \
    > results/gpqa_diamond_oracle.server.log 2>&1 &
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
        echo 'Gym exited; inspect results/gpqa_diamond_oracle.server.log' >&2
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

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then LIMIT_ARGS=(+limit="$LIMIT"); fi
# Shared submission supplies the response envelope; native MCQA owns grading.
# Hydra quoting preserves the official Answer: prefix and replacement token.
echo '==> Submitting official reference answers...'
uv run --locked python scripts/run_oracles/submit_references.py \
    "${ORACLE_GYM_CLIENT_PORT_ARGS[@]}" \
    +benchmark_jsonl=benchmarks/gpqa/data/gpqa_diamond_benchmark.jsonl \
    +resources_server=gpqa_mcqa_resources_server \
    +reference_field=expected_answer \
    +prompt_config=benchmarks/prompts/eval/aai/mcq-4choices.yaml \
    '+response_template="Answer: {answer}"' \
    +output_jsonl=results/gpqa_diamond_oracle.jsonl \
    "${LIMIT_ARGS[@]}" \
    2>&1 | tee results/gpqa_diamond_oracle.log
echo '==> Per-task oracle results: results/gpqa_diamond_oracle.jsonl'
