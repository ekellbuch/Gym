#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Invoke all benchmark wrappers concurrently, each with its own Gym ports and log.
# Each wrapper remains responsible for preparation, official scoring, and cleanup.
# Forward arguments unchanged and exit nonzero if any wrapper cannot run.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
LOG_DIR=results/run_oracles_all
mkdir -p "$LOG_DIR"
pids=()
names=()
for script in scripts/run_oracles/*.sh; do
    case "$script" in
        */run_all.sh|*/oracle_ports.sh) continue ;;
    esac
    name="${script##*/}"
    name="${name%.sh}"
    bash "$script" "$@" > "$LOG_DIR/$name.log" 2>&1 &
    pid=$!
    pids+=("$pid")
    names+=("$name")
    printf 'Started %-32s pid=%s log=%s\n' "$name" "$pid" "$LOG_DIR/$name.log"
done

stop_children() {
    local pid
    for pid in "${pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap 'stop_children; exit 130' INT
trap 'stop_children; exit 143' TERM

successful=0
unsupported=0
failed=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    status=$?
    printf '%-32s exit=%s log=%s\n' "${names[$i]}" "$status" "$LOG_DIR/${names[$i]}.log"
    case "$status" in
        0) successful=$((successful + 1)) ;;
        2) unsupported=$((unsupported + 1)) ;;
        130|143) stop_children; exit "$status" ;;
        *) failed=$((failed + 1)) ;;
    esac
done
printf '\nSuccessful commands: %s; unsupported: %s; failed: %s\n' "$successful" "$unsupported" "$failed"
[[ "$unsupported" -eq 0 && "$failed" -eq 0 ]]
