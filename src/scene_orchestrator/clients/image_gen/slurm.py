"""Skyboxes as Slurm jobs on Pegasus.

For each skybox, from this machine:

    ssh login1: sbatch   a GPU job running `image-gen gen --backend local_diffusers ...`
    ssh login1: sacct    repeated until the job is COMPLETED or has failed
    ssh login1: cat      the PNG and its sidecar, into out/<scene_id>/

The GPU is held only while one picture is made, and nothing has to keep
running on Pegasus between requests. The price is the Slurm queue plus about
30-60 s of model loading per job, which is fine for scenes generated before a
session.

A job outlives whatever submitted it. Its id is reported the moment Slurm
accepts it, and a later run given that id waits for the same job rather than
submitting a second one.

Every ssh call reuses the shared connection you open by hand with your
password (an OpenSSH ControlMaster), so none of them asks for one. Windows'
ssh cannot share connections, so the orchestrator runs in WSL.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from ...config import ImageGenConfig
from ...outputs import (
    skybox_png_path,
    skybox_sidecar_path,
    write_bytes_atomic,
    write_json,
)
from .base import ImageGenError, ImageGenStatus, SkyboxRequest, SkyboxResult

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Slurm states after which the job will never complete.
_FAILED_STATES = {
    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
}


class PegasusShell:
    """Runs commands on the login node through the shared SSH connection."""

    def __init__(
        self,
        login: str,
        control_path: str,
        *,
        timeout_s: float,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self._login = login
        self._control_path = control_path
        self._timeout_s = timeout_s
        self._run = run

    @property
    def open_command(self) -> str:
        """What you type to open the shared connection."""
        return f"ssh -NM -o ControlPath={self._control_path} {self._login}"

    def check(self) -> str | None:
        """None when the shared connection is up; otherwise what to do about it."""
        if sys.platform == "win32":
            return (
                "Windows' ssh cannot share a connection. Run the orchestrator in "
                f"WSL, with `{self.open_command}` open in another WSL terminal."
            )
        try:
            result = self._run(
                self._ssh("-O", "check", self._login),
                input=b"", capture_output=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"cannot check the SSH connection: {exc}"
        if result.returncode == 0:
            return None
        return (
            "no shared SSH connection to Pegasus. Open it in a WSL terminal and "
            f"leave it open: {self.open_command}"
        )

    def run(self, command: str, *, input: bytes = b"") -> bytes:
        """Run `command` on the login node; return its stdout."""
        try:
            result = self._run(
                self._ssh(self._login, command),
                input=input, capture_output=True, timeout=self._timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise ImageGenError(
                f"no answer from Pegasus within {self._timeout_s:g}s: {command[:80]}"
            ) from exc
        except OSError as exc:
            raise ImageGenError(f"cannot run ssh: {exc}", retryable=False) from exc
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            raise ImageGenError(
                f"`{command[:80]}` failed on Pegasus: {stderr or f'exit {result.returncode}'}"
            )
        return result.stdout

    def _ssh(self, *args: str) -> list[str]:
        return [
            "ssh",
            "-o", f"ControlPath={Path(self._control_path).expanduser()}",
            "-o", "ControlMaster=no",  # reuse the shared connection, never open one
            "-o", "BatchMode=yes",  # and never wait at a password prompt
            *args,
        ]


class SlurmImageGenClient:
    def __init__(
        self,
        config: ImageGenConfig,
        *,
        shell: PegasusShell | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._shell = shell or PegasusShell(
            config.login, config.control_path, timeout_s=config.ssh_timeout_s
        )
        self._sleep = sleep
        self._clock = clock

    def status(self) -> ImageGenStatus:
        problem = self._shell.check()
        return ImageGenStatus(
            login=self._config.login, connected=problem is None, error=problem
        )

    def generate_skybox(
        self,
        request: SkyboxRequest,
        out_dir: Path,
        on_progress: Callable[[str], None] | None = None,
        *,
        resume_job_id: str | None = None,
        on_submitted: Callable[[str], None] | None = None,
    ) -> SkyboxResult:
        """`on_submitted` receives each new job's id as soon as Slurm has it.

        `resume_job_id` is such an id from an earlier run of the same request
        that never saw the job finish: the first attempt waits for that job
        instead of submitting another.
        """
        progress = on_progress or (lambda _: None)
        errors: list[str] = []
        pending: str | None = None
        for attempt in range(1, self._config.retries + 2):
            try:
                png, sidecar = self._attempt(
                    request, out_dir, progress, resume_job_id, on_submitted
                )
            except ImageGenError as exc:
                errors.append(f"attempt {attempt}: {exc}")
                pending = exc.pending_job_id
                if not exc.retryable:
                    break
            else:
                return SkyboxResult(ok=True, attempts=attempt, png=png, sidecar=sidecar)
            resume_job_id = None  # that job is over: a retry submits a new one
        return SkyboxResult(
            ok=False, attempts=len(errors), error="; ".join(errors), pending_job_id=pending
        )

    # -- one attempt: submit (or pick up) -> wait -> fetch -----------------

    def _attempt(
        self,
        request: SkyboxRequest,
        out_dir: Path,
        progress: Callable[[str], None],
        resume_job_id: str | None,
        on_submitted: Callable[[str], None] | None,
    ) -> tuple[Path, Path]:
        problem = self._shell.check()
        if problem is not None:
            # A job being picked up is still out there; keep its id for next time.
            raise ImageGenError(problem, retryable=False, pending_job_id=resume_job_id)

        if resume_job_id is not None:
            job_id = resume_job_id
            progress(f"Slurm job {job_id}: picked up from an earlier run")
        else:
            job_id = self._submit(request)
            progress(f"Slurm job {job_id}: submitted")
            if on_submitted is not None:
                on_submitted(job_id)
        self._wait(job_id, progress)

        # Where `image-gen gen` wrote them: <out>/<scene>/<scene>_skybox_<seed>.
        remote = (
            f"{self._config.remote_out}/{request.scene_id}/"
            f"{request.scene_id}_skybox_{request.seed}"
        )
        # Fetch both before writing either, so a failed attempt never leaves a
        # PNG on disk without its sidecar.
        png = self._shell.run(f"cat -- {shlex.quote(remote + '.png')}")
        if not png.startswith(PNG_SIGNATURE):
            raise ImageGenError(f"{remote}.png is not a PNG")
        try:
            metadata = json.loads(self._shell.run(f"cat -- {shlex.quote(remote + '.json')}"))
        except ValueError as exc:
            raise ImageGenError(f"{remote}.json is not JSON") from exc

        png_path = skybox_png_path(out_dir, request.scene_id, request.seed)
        sidecar_path = skybox_sidecar_path(out_dir, request.scene_id, request.seed)
        # Renamed to the orchestrator's contract; the sidecar has to agree, and
        # keeps where the file came from for traceability.
        metadata["file"] = png_path.name
        metadata["upstream_file"] = f"{remote}.png"
        metadata["slurm_job_id"] = job_id
        write_bytes_atomic(png_path, png)
        write_json(sidecar_path, metadata)
        return png_path, sidecar_path

    def _submit(self, request: SkyboxRequest) -> str:
        try:
            out = self._shell.run(
                f"mkdir -p {shlex.quote(self._logs)} && sbatch --parsable",
                input=self._job_script(request).encode(),
            )
        except ImageGenError as exc:
            # Slurm refused the job (partition, account, limits): the same
            # script would be refused again.
            raise ImageGenError(f"sbatch: {exc}", retryable=False) from exc
        job_id = out.decode(errors="replace").strip().split(";")[0]
        if not job_id.isdigit():
            raise ImageGenError(f"sbatch returned no job id: {out!r}", retryable=False)
        return job_id

    def _job_script(self, request: SkyboxRequest) -> str:
        c, q = self._config, shlex.quote
        return "\n".join([
            "#!/bin/bash",
            f"#SBATCH --job-name=skybox-{request.scene_id}",
            f"#SBATCH --partition={c.partition}",
            f"#SBATCH --account={c.account}",
            f"#SBATCH --gres=gpu:{c.gpus}",
            f"#SBATCH --time={c.time_limit}",
            f"#SBATCH --output={self._logs}/%j.out",
            "set -e",
            f"source {q(c.venv + '/bin/activate')}",
            f"cd {q(c.repo)}",
            f"export HF_HOME={q(c.hf_home)} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
            "image-gen gen --backend local_diffusers --kind skybox"
            f" --prompt {q(request.prompt)} --seed {request.seed}"
            f" --scene-id {request.scene_id} --out {q(c.remote_out)}",
            "",
        ])

    def _wait(self, job_id: str, progress: Callable[[str], None]) -> None:
        deadline = self._clock() + self._config.job_timeout_s
        last: str | None = None
        while True:
            try:
                state = self._state(job_id)
            except ImageGenError as exc:
                problem = self._shell.check()
                if problem is not None:
                    raise ImageGenError(
                        f"lost the SSH connection while Slurm job {job_id} was "
                        f"{last or 'queued'}; it keeps running on Pegasus. {problem}",
                        retryable=False,
                        pending_job_id=job_id,
                    ) from exc
                state = last or "PENDING"  # a hiccup: ask again next round

            if state != last:
                progress(f"Slurm job {job_id}: {state}")
                last = state
            if state == "COMPLETED":
                return
            if state in _FAILED_STATES:
                raise ImageGenError(f"Slurm job {job_id} {state}: {self._log_tail(job_id)}")
            if self._clock() >= deadline:
                self._cancel(job_id)
                raise ImageGenError(
                    f"Slurm job {job_id} still {state} after "
                    f"{self._config.job_timeout_s:g}s, so it was cancelled"
                )
            self._sleep(self._config.poll_interval_s)

    def _state(self, job_id: str) -> str:
        words = self._shell.run(f"sacct -j {job_id} -n -X -P -o State").decode().split()
        # Empty until accounting catches up with a fresh submission.
        # "CANCELLED by 1234" -> "CANCELLED".
        return words[0] if words else "PENDING"

    def _log_tail(self, job_id: str) -> str:
        log = f"{self._logs}/{job_id}.out"
        try:
            tail = self._shell.run(f"tail -n 5 -- {shlex.quote(log)}")
            lines = tail.decode(errors="replace").strip().splitlines()
        except ImageGenError:
            lines = []
        return f"{' | '.join(lines) or '(empty log)'} (log: {log})"

    def _cancel(self, job_id: str) -> None:
        try:
            self._shell.run(f"scancel {job_id}")
        except ImageGenError:
            pass  # best effort: the job's own time limit ends it anyway

    @property
    def _logs(self) -> str:
        return f"{self._config.remote_out}/.slurm"
