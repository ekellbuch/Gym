# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym prepare-script entry point for OCRBench_v2 Chinese rows."""

from benchmarks.ocrbench_v2.prepare import prepare


if __name__ == "__main__":
    print(prepare("cn"))
