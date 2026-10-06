"""A simulated Pegasus login node, so the tests run without the cluster.

Test-only: the application itself always talks to the real Pegasus.
"""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath

from scene_orchestrator.clients.image_gen import ImageGenError, SlurmImageGenClient
from scene_orchestrator.config import ImageGenConfig

PNG = b"\x89PNG\r\n\x1a\n" + b"not really pixels"

PEGASUS = {
    "login": "demir@login1.pegasus.kl.dfki.de",
    "repo": "/netscratch/demir/image-gen-service",
    "venv": "/netscratch/demir/venvs/image-gen",
    "hf_home": "/netscratch/demir/hf",
    "remote_out": "/netscratch/demir/image-gen/out",
    "account": "ei-external",
}

NOT_CONNECTED = "no shared SSH connection to Pegasus. Open it in a WSL terminal: ssh -NM ..."


class FakePegasus:
    """Stands in for PegasusShell, answering like the login node would.

    Each submitted job plays the next script in `jobs` (the last script is
    reused once they run out): the states successive sacct calls report, the
    last one repeating. `running` holds jobs already on the cluster, submitted
    by an earlier run: job id -> such a script.
    """

    def __init__(
        self,
        jobs=(("PENDING", "RUNNING", "COMPLETED"),),
        *,
        connected=True,
        sbatch_error=None,
        lose_connection=False,
        running=None,
    ):
        self.scripts = list(jobs)
        self.connected = connected
        self.sbatch_error = sbatch_error
        self.lose_connection = lose_connection
        self.submitted: list[str] = []  # job scripts, as sbatch received them
        self.cancelled: list[str] = []
        self._states: dict[str, list[str]] = {
            job_id: list(states) for job_id, states in (running or {}).items()
        }

    def check(self):
        return None if self.connected else NOT_CONNECTED

    def run(self, command, *, input=b""):
        if "sbatch" in command:
            return self._sbatch(input.decode())
        if command.startswith("sacct"):
            return self._sacct(re.search(r"-j (\d+)", command).group(1))
        if command.startswith("cat"):
            return self._cat(command.split(" -- ", 1)[1].strip("'"))
        if command.startswith("tail"):
            return b"Traceback (most recent call last):\nRuntimeError: CUDA out of memory\n"
        if command.startswith("scancel"):
            self.cancelled.append(command.split()[1])
            return b""
        raise AssertionError(f"unexpected command on Pegasus: {command}")

    def _sbatch(self, script):
        if self.sbatch_error:
            raise ImageGenError(self.sbatch_error)
        self.submitted.append(script)
        job_id = str(123455 + len(self.submitted))
        states = self.scripts[min(len(self.submitted), len(self.scripts)) - 1]
        self._states[job_id] = list(states)
        return f"{job_id}\n".encode()

    def _sacct(self, job_id):
        if self.lose_connection:
            self.connected = False
            raise ImageGenError("ssh: connection closed")
        states = self._states[job_id]
        state = states.pop(0) if len(states) > 1 else states[0]
        return f"{state}\n".encode()

    def _cat(self, path):
        if path.endswith(".png"):
            return PNG
        png_name = PurePosixPath(path).with_suffix(".png").name
        return json.dumps({"seed": 1234, "steps": 30, "file": png_name}).encode()


class FakeClock:
    """Time that only moves when the client sleeps: timeouts cost no real time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def make_client(pegasus: FakePegasus, **config) -> SlurmImageGenClient:
    clock = FakeClock()
    settings = {**PEGASUS, "poll_interval_s": 15, "job_timeout_s": 60, **config}
    return SlurmImageGenClient(
        ImageGenConfig(**settings), shell=pegasus, sleep=clock.sleep, clock=clock
    )
