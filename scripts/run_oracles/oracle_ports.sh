#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Assign each oracle a separate Gym head port and child-server port range.
# The caller passes the name of its script without the .sh suffix.

oracle_ports() {
    local benchmark="$1"
    local names=(
        aa_lcr aa_omniscience apex_shortlist baby_vision browsecomp
        charxiv_rq deepswe gpqa_diamond graphwalks hle_text hle_vision
        mathvision mathvista mmlongbench_doc mmlu_prox ocr_reasoning
        ocrbench_v2_cn ocrbench_v2_en tb4 videomme_v2 vstar wmt24pp
    )
    local index=-1
    local i
    for i in "${!names[@]}"; do
        if [[ "${names[$i]}" == "$benchmark" ]]; then
            index="$i"
            break
        fi
    done
    if (( index < 0 )); then
        echo "No Gym ports assigned to oracle: $benchmark" >&2
        return 2
    fi

    local head_port=$((11200 + index))
    local low_port=$((22000 + 100 * index))
    local high_port=$((low_port + 99))
    ORACLE_GYM_START_PORT_ARGS=(
        "++head_server.port=$head_port"
        "++port_range_low=$low_port"
        "++port_range_high=$high_port"
    )
    ORACLE_GYM_CLIENT_PORT_ARGS=("++head_server.port=$head_port")

    # Gym installs each server's dependencies itself. Ray's automatic uv-run
    # packaging would upload unrelated local worktrees and can exceed its
    # 512 MiB working-directory limit before any verifier starts.
    export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
}
