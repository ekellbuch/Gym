# Oracle scripts

All 57 benchmark script names are retained in this directory, plus `run_all.sh`.
Of the 57 wrappers, 26 have an executable oracle path and 31 fail closed:
17 can prepare inputs but lack a published candidate or official oracle runner,
and 14 lack the exact current-Gym benchmark integration. A wrapper's presence
does not imply an oracle score. `SOURCES.json` names each source and blocker.
Bash owns the same phases as TB2.1: preparation, Gym startup, readiness,
upstream oracle execution, result files, and cleanup. Official answers and scoring are unchanged. Reference benchmarks use a
local submission helper; interactive replay adapters are not used.

Six entries invoke upstream golden-solution runners. Nine more submit published
reference answers to native Gym verifiers using the user-authorized local
[submit_references.py](submit_references.py): HLE text, HLE full, AA-LCR v1.1,
MMLU-ProX, GPQA Diamond, BrowseComp, WMT24++, GraphWalks,
and APEX Shortlist. Official answer values and native grading are unchanged.
Submission formatting follows the verifier's accepted format: boxed answers for
MMLU-ProX and APEX, `Answer:` for GPQA, and unquoted published node identifiers
for GraphWalks. No graph solving or interactive action replay is added.

TB4 executes the pinned packages' official `solution/solve.sh` files using a local
execution addition to the current Gym resources server. It uses the existing
sandbox provisioning, artifact collection, separate verifier, and cleanup.

Additional published-reference runners are being validated for CharXiv RQ,
VStarBench, MathVision, MathVista_MINI, MMLongBench-Doc, OCR Reasoning,
OCRBench_v2, Video-MME-v2, and BabyVision. VStarBench completed all 191 rows
with native reward 1.0 and no invalid grading. CharXiv RQ scored 998 of 1000
rows; two answers require a judge whose configured endpoint returned HTTP 401.
MathVision and MathVista_MINI each passed a one-row native Gym check. BabyVision
prepared 388 public-author answers, but its judge endpoint returned HTTP 401 and
parity with the older internal dataset variant is unverified. OCRBench_v2 EN
submits the 6,000 rows with complete published responses, excluding five
unsupported categories, so no full English score is available. Chinese completed
all 2,600 rows with an official five-group Overall Score of 0.984666. The table below
distinguishes completed scores from wired or blocked scripts.
Video-MME-v2 completed all 3,200 native Gym verifications with row mean 1.0;
its pinned official rating is 100.0 across 800 four-question groups.
Direct probes of the configured Luna `/responses` and GPT-4o-mini
`/chat/completions` routes returned HTTP 401 with this shell's exported
`INFERENCE_API_KEY` (the same value as `JUDGE_API_KEY`). This is the observed
authentication result for those routes, not a result for every judge-backed
benchmark. TB4 completed 63/66 tasks; its three H100 tasks timed out
during OpenSandbox creation before their official `solve.sh` files ran.

The new reference scripts use the same phases as TB2.1: prepare, start Gym,
wait, submit, save results, and cleanup. Native prompt materialization preserves
the question context; full HLE preserves existing image inputs. Only the verifier
and any required judge are started, with no generating policy agent.

Use `--prepare-only` to stop after preparation, or a positive positional number
for a small run. Reference results are written to `results/<name>_oracle.jsonl`
and a sibling `.summary.json`. The summary reports selected rows, valid rewards,
invalid grading, and request errors separately. Failed grading is not a zero
score. A partial run returns nonzero; `mean_reward` only describes valid rows.
The runner submits up to 64 requests concurrently with bounded memory.

[SOURCES.json](SOURCES.json) records preparation sources and execution status
for every benchmark wrapper, including the six upstream golden-solution runners.

TB4 uses this checkout's manifest and native preparation. Harbor is used only
for downloading official packages, including solutions. `tb4.sh` follows the
TB2.1 prepare/start/wait/execute/results/cleanup flow. Its local
`resources_servers/terminal_bench_4/apply_golden_patch.py` submits the selected
tasks concurrently; the server executes the entire supplied solution directory
without changing solution contents. All 66 package hashes match the manifest.
Results are `results/terminal_bench_4_oracle.jsonl`, with a sibling summary and
per-task resource artifacts under `results/terminal_bench_4_oracle/resources/`.
The full 66-task run plus one retry of its five incomplete CPU/Compose tasks
produced 63 native verifier scores: 61 rewards of 1.0 and two rewards of 0.0.
The remaining three H100 tasks (`fp8-rmsnorm-gemm`, `jax-speedrun-gpu`, and
`math-eval-grader`) remained incomplete after a separate parallel retry because
OpenSandbox sandbox creation timed out. No 66-task score is available. The
merged per-task record and coverage summary are
`results/terminal_bench_4_oracle/combined-results.jsonl` and
`results/terminal_bench_4_oracle/combined-summary.json`; the original and retry
result files are retained separately. Set `TB4_USE_PREPARED=1` to retry using
packages and Gym data previously prepared by the default download path.
A real one-task Gym smoke executed `interleaved-vigenere`'s official solution:
reward 1.0, exit 0, no infrastructure errors, and all task/verifier/log-helper
sandboxes closed without cleanup errors. Evidence:
`results/terminal_bench_4_oracle_smoke/results.jsonl` and `.summary.json`.
The smoke predates the shell-only correction making the resource artifact path
absolute; its resource logs are under
`resources_servers/terminal_bench_4/results/terminal_bench_4_oracle/resources/`.

```bash
uv sync --locked
bash scripts/run_oracles/swebench_pro.sh --help
bash scripts/run_oracles/swebench_pro.sh 2
bash scripts/run_oracles/lmarena_v3.sh --prepare-only
```

Selected tasks are submitted concurrently by upstream runners. `run_all.sh`
starts all benchmark wrappers concurrently, with a separate launcher log under
`results/run_oracles_all/` for each. Executable wrappers assign distinct Gym head
ports and child-server port ranges. The SWE-bench and Terminal-Bench wrappers
also isolate the upstream runner's fixed `temp2.jsonl` working file. `run_all.sh`
reports unsupported and failed commands with a nonzero final exit. Arguments are
forwarded to each script; use individual scripts for benchmark-specific options.

Gym source audited: `1c8261080bdc881b3e9b7f870e6418f160516991` from
`https://github.com/NVIDIA-NeMo/Gym.git`. `uv --locked` preserves root dependency
versions. Upstream server dependencies and some dataset revisions remain mutable;
this is not a promise of fully pinned fresh-machine oracle reproduction.
Credentials stay in exported environment variables. LMArena preparation reuses
`GITLAB_TOKEN` and the existing NVIDIA registry project 191584. OpenSandbox-backed
runs use the existing domain/key configuration. Resources need compatible runtime
capacity; credentials alone do not establish every benchmark's requirements.

All 59 shell files pass syntax checks. The new reference runner passes eight
focused tests, all nine executable reference configurations resolve without generating
agents, and shell process checks cover arguments, outputs, and cleanup.
MMLU-ProX completed all 341,011 native Gym verifications with mean reward 1.0.
Other live oracle workflows remain unverified; command success is not proof that
every task passed.

| Script | Upstream oracle entrypoint |
| --- | --- |
| [gdpval_aa_v2.sh](gdpval_aa_v2.sh) | Not established; exits 2 |
| [tau3_bank.sh](tau3_bank.sh) | Not established; exits 2 |
| [tb21.sh](tb21.sh) | `resources_servers/terminal_bench_2_1/apply_golden_patch.py` |
| [scicode.sh](scicode.sh) | Not established; exits 2 |
| [hle_text.sh](hle_text.sh) | Local reference submission to native Gym verifier |
| [gpqa_diamond.sh](gpqa_diamond.sh) | Native Gym score complete: 198/198 verified, mean reward 1.0 |
| [critpt.sh](critpt.sh) | Not established; exits 2 |
| [aa_omniscience.sh](aa_omniscience.sh) | Public inputs prepare; no exact non-hallucination oracle for the private 6,000-question benchmark; exits 2 |
| [aa_briefcase.sh](aa_briefcase.sh) | Not established; exits 2 |
| [automation_bench.sh](automation_bench.sh) | Not established; exits 2 |
| [tb4.sh](tb4.sh) | Local `resources_servers/terminal_bench_4/apply_golden_patch.py`; executes official `solution/solve.sh` through current Gym |
| [gdp_pdf.sh](gdp_pdf.sh) | Not established; exits 2 |
| [aa_lcr.sh](aa_lcr.sh) | Local reference submission to native Gym verifier |
| [hle_vision.sh](hle_vision.sh) | Local reference submission to native Gym verifier |
| [lcb_v6.sh](lcb_v6.sh) | Not established; exits 2 |
| [ifbench.sh](ifbench.sh) | Not established; exits 2 |
| [apex_shortlist.sh](apex_shortlist.sh) | Native Gym score complete: 47/47 verified, mean reward 1.0 |
| [swebench_pro.sh](swebench_pro.sh) | `resources_servers/swebench_pro/apply_golden_patch.py` |
| [swebench_verified.sh](swebench_verified.sh) | `resources_servers/swebench/apply_golden_patch.py` |
| [swebench_multilingual.sh](swebench_multilingual.sh) | `resources_servers/swebench/apply_golden_patch.py` |
| [tb21_3h.sh](tb21_3h.sh) | `resources_servers/terminal_bench_2_1/apply_golden_patch.py` |
| [deepswe.sh](deepswe.sh) | `resources_servers/deepswe/validate_golden.py` |
| [tau3_average.sh](tau3_average.sh) | Not established; exits 2 |
| [pinchbench.sh](pinchbench.sh) | Not established; exits 2 |
| [browsecomp.sh](browsecomp.sh) | Local reference submission to native Gym verifier |
| [browsecomp_opencode.sh](browsecomp_opencode.sh) | Not established; exits 2 |
| [lmarena_v2.sh](lmarena_v2.sh) | Not established; exits 2 |
| [lmarena_v3.sh](lmarena_v3.sh) | Not established; exits 2 |
| [kernelbench_hard.sh](kernelbench_hard.sh) | Not established; exits 2 |
| [mmlu_prox.sh](mmlu_prox.sh) | Native Gym score complete: 341,011/341,011 verified, mean reward 1.0 |
| [wmt24pp.sh](wmt24pp.sh) | Local reference submission to native Gym verifier |
| [graphwalks.sh](graphwalks.sh) | Native Gym score complete: 1,150/1,150 verified, mean reward 1.0 |
| [cs1k.sh](cs1k.sh) | Not established; exits 2 |
| [aegis_v2_reasoning.sh](aegis_v2_reasoning.sh) | Not established; exits 2 |
| [aegis_v2_no_reasoning.sh](aegis_v2_no_reasoning.sh) | Not established; exits 2 |
| [xstest.sh](xstest.sh) | Not established; exits 2 |
| [cbrne.sh](cbrne.sh) | Not established; exits 2 |
| [garak.sh](garak.sh) | Not established; exits 2 |
| [agentdyn.sh](agentdyn.sh) | Not established; exits 2 |
| [bbq.sh](bbq.sh) | Not established; exits 2 |
| [geobias.sh](geobias.sh) | Not established; exits 2 |
| [charxiv_rq.sh](charxiv_rq.sh) | 998/1000 valid rows scored 1.0; two rows unscored after judge HTTP 401; no complete score |
| [baby_vision.sh](baby_vision.sh) | 388 public-author references prepared; judge HTTP 401 and internal-variant parity unverified; no score |
| [vstar.sh](vstar.sh) | Native Gym oracle complete: 191/191 verified, mean reward 1.0 |
| [mathvision.sh](mathvision.sh) | Native Gym reference submission; one-row score 1.0, full run pending |
| [mathvista.sh](mathvista.sh) | One-row native score 1.0; 37/1000 references require judge returning HTTP 401; no full score |
| [ocr_reasoning.sh](ocr_reasoning.sh) | Native Gym runner wired; judge authentication pending |
| [mmlongbench_doc.sh](mmlongbench_doc.sh) | 1,091 references prepared; direct judge preflight HTTP 401; no score |
| [ocrbench_v2_en.sh](ocrbench_v2_en.sh) | 6,000/6,000 verified; row mean 0.978625 and five-group subset mean 0.982745; no full EN score |
| [ocrbench_v2_cn.sh](ocrbench_v2_cn.sh) | 2,600/2,600 verified; official Chinese Overall Score 0.984666; row mean 0.980543 |
| [screenspot_pro.sh](screenspot_pro.sh) | Not established; exits 2 |
| [videomme_v2.sh](videomme_v2.sh) | 3,200/3,200 native Gym rows verified, mean reward 1.0; pinned official 800-group rating 100.0 |
| [osworld.sh](osworld.sh) | Not established; exits 2 |
| [webvoyager.sh](webvoyager.sh) | Not established; exits 2 |
| [rtvlm.sh](rtvlm.sh) | Not established; exits 2 |
| [vlguard.sh](vlguard.sh) | Not established; exits 2 |
| [dehumanization.sh](dehumanization.sh) | Not established; exits 2 |

Reference-submission validation: eight focused tests cover answer/prompt fidelity,
JSONL parsing, output formatting, invalid grading, request failures, and cancelled
reruns. The full MMLU-ProX run submitted 341,011 published answers over Gym HTTP:
all verified, with mean reward 1.0 and no invalid results or request errors.
Its prepared JSONL has SHA256
`642be743fea621522453dd0adcfcf8794b67df5e77411456961a0e991b9c362c`.
The laptop run used `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` because Ray's default
packaging included old local worktrees and exceeded 512 MiB.
Judge-backed full runs and WMT's GPU evaluation have not been performed.

WMT24++ invokes native `gym eval aggregate` after verification to compute COMET;
its native GPU requirements remain in place. BrowseComp keeps its native default
400-question preparation (`BROWSECOMP_RUN_FULL=1` requests all 1,266). APEX retains
symbolic-only grading; an unused model instance satisfies Gym's required judge
reference schema. Judge-backed wrappers use `JUDGE_API_KEY=$INFERENCE_API_KEY`.
