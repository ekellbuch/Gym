# BabyVision oracle

`bash scripts/run_oracles/baby_vision.sh` downloads the authors' official
`babyvision_data.zip` at commit `7f92fd4b1dc1c68b7b936a9bc09c68b4a944a55a`,
verifies its SHA256, and uses the pinned VLMEvalKitMcore
`build_babyvision_tsv` and `BabyVision_auxeval` scorer through current NeMo Gym.
It submits the published answer field for each of the 388 tasks. The prepared
rows contain 135 choice answers and 253 blank answers.

This is the **public author dataset variant**. The older internal Gym notes cite
an internal staged `BabyVision.tsv` with MD5 `b3a93182cce0e658e4f81601275e1f04`.
Rebuilding its TSV from the currently published author ZIP produces MD5
`7c66e7263e1724129d63f8879205fc6d`. The old TSV is not available here for
row-level comparison, so exact parity with that internal artifact is unverified.
Do not report a score from this runner as the old internal variant's number.

`--prepare-only` builds the JSONL without starting Gym. A full run requires
`GITLAB_TOKEN` on a fresh checkout to download pinned Mcore source and a valid
`INFERENCE_API_KEY` for the configured judge endpoint. The shell checks judge
authentication before launching because the official scorer's nested retries
would otherwise take hours on a bad key. Results are written to
`results/baby_vision_oracle.jsonl` with a summary beside them.
