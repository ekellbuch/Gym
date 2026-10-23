# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tomllib
from contextlib import contextmanager
from copy import deepcopy
from glob import glob
from math import ceil
from pathlib import Path, PurePosixPath
from shlex import join
from sys import stderr
from tempfile import NamedTemporaryFile
from time import time
from traceback import format_exc
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Tuple

from fastapi import Request
from pydantic import BaseModel

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY


# Bullseye security packages were removed from the live mirror after LTS ended.
# Keep the signed final-LTS repository and package versions available to task setup.
_BULLSEYE_SECURITY_SNAPSHOT_SETUP = r"""set -eu
. "$1"
if [ "${ID:-}:${VERSION_CODENAME:-}" = "debian:bullseye" ]; then
    snapshot=https://snapshot.debian.org/archive/debian-security/20260831T235959Z/
    old='deb http://deb[.]debian[.]org/debian-security bullseye-security main'
    new="deb [check-valid-until=no] $snapshot bullseye-security main"
    sed -i -E "s|^$old$|$new|" "$2"
fi
"""


def _build_free_environment(task_folder: Path, docker_image: str) -> Optional[Tuple[Optional[str], List[Tuple[Path, str]]]]:
    """WORKDIR and COPY steps of a task whose environment/Dockerfile only adds files to the row's docker_image.

    Official TB2.1 rows name an image prebuilt from the task's Dockerfile, so nothing is staged for them. A task
    whose Dockerfile is FROM the row's own docker_image with only WORKDIR and plain-file COPY steps (the Harbor
    format's small custom tasks, which Harbor builds and the sandbox providers here cannot) gets its WORKDIR made
    and its files uploaded at sandbox setup instead. Anything else returns None: such a task needs a prebuilt image.
    """
    if not task_folder.is_absolute():
        task_folder = PARENT_DIR / task_folder
    dockerfile = task_folder / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return None
    base, workdir, copies = None, None, []
    for line in dockerfile.read_text().splitlines():
        op, *args = line.split() or ["#"]
        if op.startswith("#"):
            continue
        elif op.upper() == "FROM" and len(args) == 1:
            base = args[0]
        elif op.upper() == "WORKDIR" and len(args) == 1:
            workdir = str(PurePosixPath(workdir or "/") / args[0])
        elif op.upper() == "COPY" and len(args) == 2 and not args[0].startswith("--"):
            src = task_folder / "environment" / args[0]
            if not src.is_file():
                return None
            dst = PurePosixPath(workdir or "/") / args[1]
            copies.append((src, str(dst / src.name if args[1].endswith("/") else dst)))
        else:
            return None
    return (workdir, copies) if base == docker_image else None


def _read_task_toml(task_folder: Path) -> Dict[str, Any]:
    """The task's parsed task.toml, or an empty map for a folder without one."""
    if not task_folder.is_absolute():
        task_folder = PARENT_DIR / task_folder
    task_toml = task_folder / "task.toml"
    if not task_toml.is_file():
        return {}
    with task_toml.open("rb") as f:
        return tomllib.load(f)


def _task_toml_resources(task_toml: Dict[str, Any]) -> Dict[str, int]:
    """Sandbox size a parsed task.toml's [environment] asks for, as SandboxResources keys (Harbor's sizing).

    cpus -> cpu, memory_mb -> memory_mib, storage_mb -> disk_gib rounded up to whole GiB. Keys the file does not set
    are left out.
    """
    environment = task_toml.get("environment", {})
    resources = {}
    if environment.get("cpus") is not None:
        resources["cpu"] = environment["cpus"]
    if environment.get("memory_mb") is not None:
        resources["memory_mib"] = int(environment["memory_mb"])
    if environment.get("storage_mb") is not None:
        resources["disk_gib"] = ceil(environment["storage_mb"] / 1024)
    return resources


class TerminalBench21ResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    is_verifying_golden_patch: bool = False
    evaluation_timeout: Optional[int] = None

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]

    debug: bool = False


class TerminalBench21SeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str  # @bxyu-nvidia: Just a plain string URI for now for OpenSandbox backend.
    workdir: Optional[str] = None  # the Dockerfile's WORKDIR of a build-free task; the agent connects there


class TerminalBench21SeedSessionRequest(BaseModel):
    task_name: str
    docker_image: str
    task_folder: str


class TerminalBench21VerifyRequest(TerminalBench21SeedSessionRequest, BaseVerifyRequest):
    pass


class TerminalBench21VerifyResponse(BaseVerifyResponse):
    evaluation_completed: bool

    # Misc metrics
    verification_time_taken: float

    task_name: str
    test_output: str
    golden_patch_output: Optional[str]


GOLDEN_PATCH_SOLVE_SH_PATCHES = {
    "terminal-bench/build-cython-ext": [
        (
            "pip install setuptools==80.9.0 cython==3.1.3",
            "pip install setuptools==80.9.0 cython==3.1.3 planarity==0.6",
        ),
    ],
    "terminal-bench/build-pov-ray": [
        ("wget=1.21.4-1ubuntu4.1", "wget"),
        ("ncompress=5.0-1", "ncompress"),
        (
            "wget https://www.povray.org/ftp/pub/povray/Old-Versions/Official-2.2/POVDOC.TAR.Z",
            "wget --tries=5 --timeout=60 --output-document=POVDOC.TAR.Z "
            "http://grumbeer.dyndns.org/ftp/cdroms/freebsd/freebsd-2.1.7-2/ports/distfiles/povdoc.tar.Z",
        ),
        (
            "wget https://www.povray.org/ftp/pub/povray/Old-Versions/Official-2.2/POVSCN.TAR.Z",
            "wget --tries=5 --timeout=60 --output-document=POVSCN.TAR.Z "
            "http://grumbeer.dyndns.org/ftp/cdroms/freebsd/freebsd-2.1.7-2/ports/distfiles/povscn.tar.Z",
        ),
        (
            "wget https://www.povray.org/ftp/pub/povray/Old-Versions/Official-2.2/POVSRC.TAR.Z",
            """wget --tries=5 --timeout=60 --output-document=POVSRC.TAR.Z \\
  http://grumbeer.dyndns.org/ftp/cdroms/freebsd/freebsd-2.1.7-2/ports/distfiles/povsrc.tar.Z
cat <<'EOF' | sha256sum --check -
e70e44d1fe8835c4dff7c7a55bd6629b15e6a15b2ab7f2f49ee9e2dc016cc470  POVDOC.TAR.Z
4272e2d4724d8dfd916d68827194577221d17b733d99e84e7040f3a9f7eb92a7  POVSCN.TAR.Z
4d8a7073fadaca82827f1354428393cd13e4d3f71a5a3149fd7d6fffd77293d4  POVSRC.TAR.Z
EOF""",
        ),
    ],
}

TEST_SH_PATCHES = {
    "terminal-bench/mcmc-sampling-stan": [
        ("sudo apt-get install -y \\\n    gfortran", "sudo apt-get install -y \\\n    cmake \\\n    gfortran"),
    ],
    "terminal-bench/pytorch-model-recovery": [
        ("-w torch==2.7.1", "-w torch==2.7.1 --index https://download.pytorch.org/whl/cpu"),
    ],
    "terminal-bench/torch-tensor-parallelism": [
        ("-w torch==2.7.0", "-w torch==2.7.0 --index https://download.pytorch.org/whl/cpu"),
    ],
    "terminal-bench/torch-pipeline-parallelism": [
        ("-w torch==2.7.0", "-w torch==2.7.0 --index https://download.pytorch.org/whl/cpu"),
    ],
    "terminal-bench/mteb-retrieve": [
        ("-w mteb==1.36.8", "-w mteb==1.36.8 --index https://download.pytorch.org/whl/cpu"),
    ],
}


class TerminalBench21ResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: TerminalBench21ResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        self._session_id_to_sandbox: Dict[str, AsyncSandbox] = dict()
        self._task_folder_to_resources: Dict[str, Dict[str, Any]] = dict()
        self._task_folder_to_task_toml: Dict[str, Dict[str, Any]] = dict()

    def _task_toml(self, verify_request: TerminalBench21SeedSessionRequest) -> Dict[str, Any]:
        """The task's parsed task.toml ({} without one), read once per task folder."""
        if verify_request.task_folder not in self._task_folder_to_task_toml:
            task_toml = _read_task_toml(Path(verify_request.task_folder))
            self._task_folder_to_task_toml[verify_request.task_folder] = task_toml
        return self._task_folder_to_task_toml[verify_request.task_folder]

    def _sandbox_resources(
        self, verify_request: TerminalBench21SeedSessionRequest
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """The sandbox resources (limits) map and the scheduling requests sized from the task's task.toml (None when
        the flag is off or the task.toml sets no sizes, so the configured requests apply).

        With sandbox_config.resources_from_task_toml, each task.toml [environment] key the task sets is the request for
        that field; keys it does not set keep the configured request (provider_options.resource_requests, else the
        configured limit). sandbox_config.task_toml_min_memory_mib (unset: no floor) raises the memory request to at
        least it. Each limit is max(configured sandbox_config.resources value, request): the task's size is what the
        scheduler reserves, and the generous configured limits let the sandbox burst past it (the agent runs inside).
        """
        limits = dict(self.config.sandbox_config.get("resources", {}))
        if not self.config.sandbox_config.get("resources_from_task_toml", False):
            return limits, None
        configured_requests = (self.config.sandbox_config.get("provider_options") or {}).get("resource_requests")
        if not isinstance(configured_requests, Mapping):
            configured_requests = {}
        first_read = verify_request.task_folder not in self._task_folder_to_resources
        if first_read:
            self._task_folder_to_resources[verify_request.task_folder] = _task_toml_resources(
                self._task_toml(verify_request)
            )
        from_toml = self._task_folder_to_resources[verify_request.task_folder]
        requests = self._with_memory_floor(limits | dict(configured_requests) | from_toml)
        for key, request in requests.items():
            limit = limits.get(key)
            limits[key] = request if limit is None or request is None else max(limit, request)
        if first_read:
            source = "task.toml" if from_toml else "config (no task.toml sizes)"
            print(
                f"Sandbox size for {verify_request.task_name}: requests {requests}, limits {limits} from {source}",
                file=stderr,
            )
        return limits, (requests if from_toml else None)

    def _with_memory_floor(self, resources: Dict[str, Any]) -> Dict[str, Any]:
        """resources with memory_mib raised to sandbox_config.task_toml_min_memory_mib, when that is set and
        resources_from_task_toml is on; otherwise resources unchanged. CPU and disk are never touched."""
        floor = self.config.sandbox_config.get("task_toml_min_memory_mib")
        if floor is None or not self.config.sandbox_config.get("resources_from_task_toml", False):
            return resources
        return resources | {"memory_mib": max(int(resources.get("memory_mib") or 0), int(floor))}

    def _verify_timeout(self, verify_request: TerminalBench21SeedSessionRequest) -> Optional[float]:
        """Timeout of the task's tests: evaluation_timeout, capped by the task.toml [verifier] timeout_sec.

        The cap applies with sandbox_config.resources_from_task_toml, as Harbor runs TB2.1. A task.toml without the
        key, or a task without a task.toml, keeps evaluation_timeout.
        """
        timeout = self.config.evaluation_timeout
        if not self.config.sandbox_config.get("resources_from_task_toml", False):
            return timeout
        task_timeout = self._task_toml(verify_request).get("verifier", {}).get("timeout_sec")
        if task_timeout is None:
            return timeout
        return task_timeout if timeout is None else min(timeout, task_timeout)

    def _patch_sandbox_provider_options_for_instances(
        self, task_name: str, resources: SandboxResources, provider_options: Dict[str, Any]
    ) -> None:
        # TODO @bxyu-nvidia: These patches may not be necessary eventually, but for now we need them in order for the below instance golden patches to pass.
        tasks_to_increase_initial_resources_for = {
            "terminal-bench/torch-pipeline-parallelism",
            "terminal-bench/torch-tensor-parallelism",
            "terminal-bench/pytorch-model-recovery",
            "terminal-bench/mteb-retrieve",
            "terminal-bench/caffe-cifar-10",
        }
        if task_name in tasks_to_increase_initial_resources_for:
            provider_options["resource_requests"] = {
                "cpu": resources.cpu,
                "memory_mib": resources.memory_mib,
                "disk_gib": resources.disk_gib,
            }

    async def _create_sandbox(self, verify_request: TerminalBench21SeedSessionRequest) -> AsyncSandbox:
        # TODO @bxyu-nvidia: Refactor this after Hemil's swap from Python dataclass to Pydantic BaseModel
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
        resources, task_toml_requests = self._sandbox_resources(verify_request)

        # Derive from the final resources map (after the multilingual bump); the caps follow the CPU limit, what the
        # sandbox may use, not the request. Explicit sandbox_config.env keys win over the derived caps.
        sandbox_resources = SandboxResources.from_mapping(resources)
        env = dict(self.config.sandbox_config.get("env", {}))
        if self.config.sandbox_config.get("derive_cpu_env", True):
            env = cpu_cap_env(sandbox_resources.cpu) | env

        provider_options = deepcopy(self.config.sandbox_config.get("provider_options") or {})
        if task_toml_requests is not None:
            # Requests = the task's own size (limits stay at least the configured ones); replaces the per-task requests.
            provider_options["resource_requests"] = task_toml_requests
        else:
            self._patch_sandbox_provider_options_for_instances(
                verify_request.task_name, sandbox_resources, provider_options
            )
        if (provider_options.get("resource_requests") or {}).get("memory_mib") is not None:
            # Without task.toml sizes the memory floor lifts the configured requests too (limits already cover it).
            provider_options["resource_requests"] = self._with_memory_floor(provider_options["resource_requests"])

        eval_sandbox_spec = SandboxSpec(
            image=verify_request.docker_image,
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=None,  # Default to container's WORKDIR
            env=env,
            files=dict(),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
                "instance_id": verify_request.task_name,
            },
            resources=SandboxResources.from_mapping(resources),
            entrypoint=None,
            provider_options=provider_options,
        )
        eval_sandbox = AsyncSandbox(resolved_sandbox_provider)

        async def _run_setup(sandbox: AsyncSandbox) -> None:
            result = await sandbox.exec(
                join(
                    ["bash", "-c", _BULLSEYE_SECURITY_SNAPSHOT_SETUP, "--", "/etc/os-release", "/etc/apt/sources.list"]
                ),
                timeout_s=self.config.evaluation_timeout,
            )
            if result.return_code != 0:
                raise RuntimeError(f"Failed to prepare TerminalBench package sources: {result}")

            result = await sandbox.exec("apt-get update", timeout_s=self.config.evaluation_timeout)
            if result.return_code != 0:
                print(f"Failed to apt-get update: {result}")

            environment = _build_free_environment(Path(verify_request.task_folder), verify_request.docker_image)
            if environment is not None:
                workdir, copies = environment
                for remote_dir in ([workdir] if workdir else []) + [str(PurePosixPath(dst).parent) for _, dst in copies]:
                    result = await sandbox.exec(join(["mkdir", "-p", remote_dir]), timeout_s=60)
                    if result.return_code != 0:
                        raise RuntimeError(f"Failed to make {remote_dir} for {verify_request.task_name}: {result}")
                for local_fpath, remote_fpath in copies:
                    await sandbox.upload(local_path=local_fpath, remote_path=remote_fpath)

        # start_with_setup stops the container if _run_setup raises, instead of
        # leaving it running until its TTL.
        await eval_sandbox.start_with_setup(eval_sandbox_spec, _run_setup)

        return eval_sandbox

    async def seed_session(
        self, request: Request, body: TerminalBench21SeedSessionRequest
    ) -> TerminalBench21SeedSessionResponse:
        eval_sandbox = await self._create_sandbox(body)
        self._session_id_to_sandbox[request.session[SESSION_ID_KEY]] = eval_sandbox

        environment = _build_free_environment(Path(body.task_folder), body.docker_image)
        return TerminalBench21SeedSessionResponse(
            sandbox_handle=eval_sandbox._handle.sandbox_id, workdir=environment[0] if environment else None
        )

    @contextmanager
    def _patch_golden_patch_solve_sh(
        self, task_name: str, local_fpath: Path, patches: Dict[str, List[Tuple[str, str]]]
    ):
        if task_name not in patches or local_fpath.suffix != ".sh":
            yield local_fpath
            return

        content = local_fpath.read_text()
        for old, new in patches[task_name]:
            content = content.replace(old, new)

        with NamedTemporaryFile(mode="w+", suffix=".sh", delete_on_close=False) as temp_file:
            temp_file.write(content)
            temp_file.flush()

            yield temp_file.name

    async def _upload_folder(
        self,
        sandbox: AsyncSandbox,
        local_dirpath: Path,
        target_dirpath: str,
        patches: Dict[str, List[Tuple[str, str]]],
        task_name: Optional[str] = None,
    ) -> None:
        if not local_dirpath.is_absolute():
            local_dirpath = PARENT_DIR / local_dirpath

        for file in glob("**", root_dir=str(local_dirpath), recursive=True):
            local_fpath = local_dirpath / file
            if not local_fpath.is_file():
                continue

            target_fpath = f"{target_dirpath}/{file}"
            mkdir_result = await sandbox.exec(f"mkdir -p {Path(target_fpath).parent}")
            assert mkdir_result.return_code == 0, mkdir_result

            with self._patch_golden_patch_solve_sh(task_name, local_fpath, patches) as new_local_fpath:
                await sandbox.upload(local_path=new_local_fpath, remote_path=target_fpath)

    async def verify(self, request: Request, body: TerminalBench21VerifyRequest) -> TerminalBench21VerifyResponse:
        task_folder = Path(body.task_folder)

        # An empty response is how agents with execution_failure_reward_zero grade a failed run (a timeout, the context
        # limit): score it 0 rather than the sandbox state, which the tests would grade.
        if not body.response.output and not self.config.is_verifying_golden_patch:
            sandbox = self._session_id_to_sandbox.pop(request.session[SESSION_ID_KEY], None)
            try:
                if sandbox is not None:
                    await sandbox.stop()
            except:
                print(f"Hit an exception stopping sandbox: {format_exc()}", file=stderr)
            return TerminalBench21VerifyResponse(
                **body.model_dump(),
                evaluation_completed=True,
                reward=0.0,
                verification_time_taken=0.0,
                test_output="",
                golden_patch_output=None,
            )

        if self.config.is_verifying_golden_patch:
            if self.config.debug:
                print(f"Creating eval sandbox for {body.task_name}", file=stderr)
            eval_sandbox = await self._create_sandbox(body)
            cwd = (await eval_sandbox.exec("pwd")).stdout.strip()
            await self._upload_folder(
                eval_sandbox, task_folder / "solution", cwd, GOLDEN_PATCH_SOLVE_SH_PATCHES, task_name=body.task_name
            )

            if self.config.debug:
                print(f"Running golden patch for {body.task_name}", file=stderr)
            golden_patch_result = await eval_sandbox.exec(
                f"bash {cwd}/solve.sh",
                timeout_s=self.config.evaluation_timeout,
                preserve_background_services=True,
            )
            golden_patch_output = (golden_patch_result.stderr or "") + (golden_patch_result.stdout or "")
            if self.config.debug:
                print(f"Golden patch output for {body.task_name}: {golden_patch_output}", file=stderr)
        else:
            # Re-use the original sandbox
            eval_sandbox = self._session_id_to_sandbox.pop(request.session[SESSION_ID_KEY])
            golden_patch_output = None

        if self.config.debug:
            print(f"Running tests for {body.task_name}", file=stderr)
        test_timeout = self._verify_timeout(body)
        test_start_time = None
        start_time = time()
        try:
            await self._upload_folder(eval_sandbox, task_folder / "tests", "/tests", TEST_SH_PATCHES, body.task_name)
            # Test scripts write rewards here even if their test runner fails to start.
            setup_result = await eval_sandbox.exec("mkdir -p /logs/verifier", timeout_s=self.config.evaluation_timeout)
            if setup_result.return_code != 0:
                raise RuntimeError(f"Failed to prepare TerminalBench verifier output directory: {setup_result}")
            # Harbor runs the tests in the image's WORKDIR; a build-free task's WORKDIR is not the container default.
            environment = _build_free_environment(task_folder, body.docker_image)
            test_cwd = {"cwd": environment[0]} if environment and environment[0] else {}
            test_start_time = time()
            eval_result = await eval_sandbox.exec(
                "bash /tests/test.sh",
                timeout_s=test_timeout,
                **test_cwd,
            )
            test_output = (eval_result.stderr or "") + (eval_result.stdout or "")
        except:
            print(f"Hit exception running TerminalBench 2.1 tests: {format_exc()}", file=stderr)
            eval_result = None
            test_output = ""
        verification_time_taken = time() - start_time
        # Providers return error_type "timeout", raise, or return the killed command's status at the deadline.
        tests_timed_out = test_start_time is not None and (
            getattr(eval_result, "error_type", None) == "timeout"
            or (test_timeout is not None and time() - test_start_time >= test_timeout)
        )

        if self.config.debug:
            print(f"Test output for {body.task_name}: {test_output}", file=stderr)

        evaluation_completed = False
        reward = 0.0
        if tests_timed_out and self.config.sandbox_config.get("resources_from_task_toml", False):
            # Harbor's verifier timeout: the attempt failed (e.g. the agent left a binary that hangs), not the infra.
            print(f"Tests of {body.task_name} timed out after {test_timeout}s: reward 0", file=stderr)
            evaluation_completed = True
        elif eval_result is not None:
            try:
                with NamedTemporaryFile(mode="w+", suffix=".txt") as temp_file:
                    await eval_sandbox.download("/logs/verifier/reward.txt", temp_file.name)
                    # Providers such as Docker can replace the destination file during download.
                    reward = float(Path(temp_file.name).read_text())

                evaluation_completed = True
            except:
                if self.config.debug:
                    print(f"Hit an exception downloading and converting reward: {format_exc()}", file=stderr)

        try:
            await eval_sandbox.stop()
        except:
            print(f"Hit an exception stopping sandbox: {format_exc()}", file=stderr)

        return TerminalBench21VerifyResponse(
            **body.model_dump(),
            evaluation_completed=evaluation_completed,
            reward=reward,
            verification_time_taken=verification_time_taken,
            test_output=test_output,
            golden_patch_output=golden_patch_output,
        )


if __name__ == "__main__":
    TerminalBench21ResourcesServer.run_webserver()
