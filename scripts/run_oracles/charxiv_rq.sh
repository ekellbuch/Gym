#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Prepare official CharXiv RQ references, run the pinned official auxeval through
# the current Gym resource server, and save every verifier result.
# Usage: bash scripts/run_oracles/charxiv_rq.sh [--prepare-only] [positive-limit]
# Requires GITLAB_TOKEN for first-time source retrieval and INFERENCE_API_KEY for judging.
# Results: results/charxiv_rq_oracle.jsonl; logs: results/charxiv_rq_oracle/.
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/run_oracles/charxiv_rq.sh [--prepare-only] [positive-limit]'
    echo 'Runs published CharXiv_reasoning_val answers through Gym and the official pinned mcore judge.'
    exit 0
fi
PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=true; shift; fi
LIMIT_ARGS=()
if [[ $# -gt 1 || ( $# -eq 1 && ! "$1" =~ ^[1-9][0-9]*$ ) ]]; then
    echo 'Expected one positive task limit or --prepare-only.' >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then LIMIT_ARGS=(+limit="$1"); fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/run_oracles/oracle_ports.sh
oracle_ports charxiv_rq
REPO_ROOT="$PWD"
export UV_BUILD_CONSTRAINT="${UV_BUILD_CONSTRAINT:-$REPO_ROOT/scripts/run_oracles/build-constraints.txt}"
# Local Ray can read the checked-out scorer directly. Its automatic uv runtime
# upload otherwise includes the prepared 416 MiB JSONL and exceeds Ray's 512 MiB limit.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
LOG_DIR=results/charxiv_rq_oracle
SOURCE_DIR=results/oracle-upstream/VLMEvalKitMcore-6e98cbf20bd88a6469a1a298f67ec266196a4bbf
SOURCE_FILE="$SOURCE_DIR/vlmeval/dataset/charxiv.py"
mkdir -p "$LOG_DIR"

echo '==> Preparing benchmark data...'
if [[ ! -f "$SOURCE_FILE" ]]; then
    if [[ -z "${GITLAB_TOKEN:-}" ]]; then
        echo 'GITLAB_TOKEN is not set in this shell; restart the session.' >&2
        exit 1
    fi
    ARCHIVE="$LOG_DIR/VLMEvalKitMcore-6e98cbf.tar.gz"
    curl --fail --location --silent --show-error \
        --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
        --output "$ARCHIVE" \
        'https://gitlab-master.nvidia.com/api/v4/projects/matthieul%2FVLMEvalKitMcore/repository/archive.tar.gz?sha=6e98cbf20bd88a6469a1a298f67ec266196a4bbf'
    mkdir -p "$SOURCE_DIR"
    tar -xzf "$ARCHIVE" -C "$SOURCE_DIR" --strip-components=1
fi
# Reject an HTML login page, stale cache, or different scorer before preparation.
uv run --no-project python -c 'import hashlib, pathlib, sys; p=pathlib.Path(sys.argv[1]); expected="6348576f7f21c799ccc3954be44e74bc702c1c4afe9543bc0d1ad4cfc12e75da"; actual=hashlib.sha256(p.read_bytes()).hexdigest(); assert actual==expected, f"CharXiv source hash mismatch: {actual}"' "$SOURCE_FILE"
uv run --locked python benchmarks/charxiv_rq/prepare.py 2>&1 | tee "$LOG_DIR/prepare.log"
if "$PREPARE_ONLY"; then exit 0; fi
if [[ -z "${INFERENCE_API_KEY:-}" ]]; then
    echo 'INFERENCE_API_KEY is not set in this shell; restart the session.' >&2
    exit 1
fi
uv run --locked python scripts/run_oracles/check_judge_auth.py benchmarks/charxiv_rq/config.yaml

echo '==> Starting Gym...'
set -m
uv run --locked gym env start --config benchmarks/charxiv_rq/config.yaml \
    "${ORACLE_GYM_START_PORT_ARGS[@]}" \
    ++charxiv_rq_benchmark_simple_agent=null \
    ++vlm_eval_kit_simple_agent=null \
    "++charxiv_rq_benchmark_resources_server.resources_servers.vlm_eval_kit.judge_api_key=$INFERENCE_API_KEY" \
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
    +benchmark_jsonl=benchmarks/charxiv_rq/data/charxiv_rq_benchmark.jsonl \
    +resources_server=charxiv_rq_benchmark_resources_server \
    +reference_field=answer \
    +output_jsonl=results/charxiv_rq_oracle.jsonl \
    "${LIMIT_ARGS[@]}" 2>&1 | tee "$LOG_DIR/oracle.log"
echo '==> Results: results/charxiv_rq_oracle.jsonl; logs: results/charxiv_rq_oracle/'
