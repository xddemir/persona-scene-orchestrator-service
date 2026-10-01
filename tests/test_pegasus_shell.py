"""PegasusShell: the ssh commands it builds, and how it reports trouble.

ssh itself is replaced by a recorder, so nothing leaves this machine.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scene_orchestrator.clients.image_gen import ImageGenError, PegasusShell

LOGIN = "demir@login1.pegasus.kl.dfki.de"


class RecordingSsh:
    def __init__(self, returncode=0, stdout=b"", stderr=b"", raises=None):
        self.returncode, self.stdout, self.stderr, self.raises = returncode, stdout, stderr, raises
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if self.raises:
            raise self.raises
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


def _shell(ssh: RecordingSsh) -> PegasusShell:
    return PegasusShell(LOGIN, "~/.ssh/pegasus.sock", timeout_s=60, run=ssh)


def test_commands_go_through_the_shared_connection_and_never_prompt():
    ssh = RecordingSsh(stdout=b"123456\n")
    assert _shell(ssh).run("sbatch --parsable") == b"123456\n"

    (cmd,) = ssh.calls
    assert cmd[0] == "ssh"
    assert f"ControlPath={Path('~/.ssh/pegasus.sock').expanduser()}" in cmd
    assert "ControlMaster=no" in cmd  # reuse, never open a connection
    assert "BatchMode=yes" in cmd  # never wait at a password prompt
    assert cmd[-2:] == [LOGIN, "sbatch --parsable"]


def test_a_failing_command_raises_with_what_pegasus_said():
    ssh = RecordingSsh(returncode=1, stderr=b"sacct: error: Invalid job id\n")
    with pytest.raises(ImageGenError, match="Invalid job id"):
        _shell(ssh).run("sacct -j 1")


def test_a_hanging_command_times_out():
    ssh = RecordingSsh(raises=subprocess.TimeoutExpired("ssh", 60))
    with pytest.raises(ImageGenError, match="no answer from Pegasus within 60s"):
        _shell(ssh).run("sacct -j 1")


def test_check_asks_the_shared_connection(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    ssh = RecordingSsh()
    assert _shell(ssh).check() is None
    assert ssh.calls[0][-3:] == ["-O", "check", LOGIN]


def test_check_says_how_to_open_a_missing_connection(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    problem = _shell(RecordingSsh(returncode=255)).check()
    assert f"ssh -NM -o ControlPath=~/.ssh/pegasus.sock {LOGIN}" in problem


def test_check_on_windows_points_to_wsl(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    ssh = RecordingSsh()
    problem = _shell(ssh).check()
    assert "Run the orchestrator in WSL" in problem
    assert ssh.calls == []  # Windows' ssh cannot do this, so it is not even tried
