# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exit successfully when every Gym environment reports ready."""

import json
import sys


def main() -> int:
    try:
        environments = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 1

    if not isinstance(environments, list) or not environments:
        return 1
    return 0 if all(isinstance(env, dict) and env.get("status") == "success" for env in environments) else 1


if __name__ == "__main__":
    sys.exit(main())
