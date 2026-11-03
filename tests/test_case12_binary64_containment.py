"""Mock-only containment controls: no real process or OS Job is created."""

import ctypes
from types import SimpleNamespace

import pytest

from experiments.case12_binary64_v1 import containment as C


def test_windows_worker_bypasses_launcher_and_preserves_environment(
    tmp_path, monkeypatch
):
    base = tmp_path / "base-python.exe"
    base.write_bytes(b"mock interpreter")
    fake = SimpleNamespace(
        platform="win32",
        executable="venv/python.exe",
        _base_executable=str(base),
        implementation=SimpleNamespace(name="cpython"),
    )
    monkeypatch.setattr(C, "sys", fake)
    environment = {"OMP_NUM_THREADS": "1", "__PYVENV_LAUNCHER__": "stale"}
    assert C.worker_python(environment) == str(base.resolve())
    assert environment == {
        "OMP_NUM_THREADS": "1",
        "__PYVENV_LAUNCHER__": fake.executable,
    }
    fake._base_executable = str(tmp_path / "missing.exe")
    with pytest.raises(FileNotFoundError):
        C.worker_python(environment)


def test_linux_worker_retains_interpreter_and_environment(monkeypatch):
    monkeypatch.setattr(
        C, "sys", SimpleNamespace(platform="linux", executable="venv/bin/python")
    )
    environment = {"OMP_NUM_THREADS": "1"}
    assert C.worker_python(environment) == "venv/bin/python"
    assert environment == {"OMP_NUM_THREADS": "1"}


class Process:
    pid = 101
    _handle = 707

    def __init__(self, done=False):
        self.returncode = 0 if done else None
        self.kills = 0

    def poll(self):
        return self.returncode

    def kill(self):
        self.kills += 1
        self.returncode = -9


class Api:
    def __init__(self, process, members=()):
        self.process, self.members = process, list(members)
        self.closes, self.terminations = 0, 0
        self.assign_ok = self.query_ok = self.close_ok = True
        self.drain = True

    def CreateJobObjectW(self, security, name):
        return 999

    def SetInformationJobObject(self, handle, kind, pointer, size):
        assert handle == 999 and kind == 9 and size == 144
        self.limits = (
            pointer._obj.basic.flags,
            pointer._obj.process_limit,
            pointer._obj.job_limit,
        )
        return 1

    def AssignProcessToJobObject(self, handle, process_handle):
        assert (handle, process_handle) == (999, 707)
        return self.assign_ok

    def QueryInformationJobObject(self, handle, kind, pointer, size, returned):
        assert handle == 999
        if not self.query_ok:
            return 0
        if kind == 1:
            pointer._obj.active = len(self.members)
        else:
            assert kind == 3
            pointer._obj.assigned = pointer._obj.count = len(self.members)
            for i, pid in enumerate(self.members):
                pointer._obj.pids[i] = pid
        return 1

    def TerminateJobObject(self, handle, code):
        assert (handle, code) == (999, 1)
        self.terminations += 1
        if self.drain:
            self.members.clear()
            self.process.returncode = 1
        return 1

    def CloseHandle(self, handle):
        assert handle == 999
        self.closes += 1
        return self.close_ok


def windows(monkeypatch, process, members=()):
    api = Api(process, members)
    monkeypatch.setattr(C, "_platform", lambda: "windows")
    monkeypatch.setattr(C, "_windows_api", lambda: api)
    return C.WindowsJob(), api


def clock(monkeypatch):
    value = [0.0]
    monkeypatch.setattr(C.time, "monotonic", lambda: value[0])
    monkeypatch.setattr(
        C.time, "sleep", lambda seconds: value.__setitem__(0, value[0] + seconds)
    )


def test_containment_windows_structure_layout_is_exact_on_64bit_host():
    assert ctypes.sizeof(ctypes.c_void_p) == 8
    assert ctypes.sizeof(C.BasicLimits) == 64
    assert ctypes.sizeof(C.ExtendedLimits) == 144
    assert ctypes.sizeof(C.Accounting) == 48
    assert C.BasicLimits.active_limit.offset == 40
    assert C.ExtendedLimits.process_limit.offset == 112


@pytest.mark.parametrize("limit", [True, 0, -1, 1024**3 + 1, 1.0])
def test_containment_memory_limit_cannot_be_unbounded(limit):
    with pytest.raises(ValueError):
        C.WindowsJob(limit)


def test_containment_windows_ended_leader_with_child_cannot_be_clean_pass(monkeypatch):
    process = Process(done=True)
    job, api = windows(monkeypatch, process, [222])
    assert api.limits == (0x2300, 1024**3, 1024**3)
    job.assign(process)
    result = C.cleanup_owned(process, job)
    assert result["empty"] and result["terminated"] and result["unexpected_residuals"]
    assert result["residuals"] == [] and result["closed"]
    assert api.terminations == api.closes == 1
    assert job.handle is None


def test_containment_windows_finished_empty_job_does_not_terminate(monkeypatch):
    process = Process(done=True)
    job, api = windows(monkeypatch, process)
    job.assign(process)
    result = job.cleanup(process)
    assert result["empty"] and not result["terminated"]
    assert not result["unexpected_residuals"] and api.terminations == 0


def test_containment_windows_running_owned_leader_is_stopped(monkeypatch):
    process = Process()
    job, api = windows(monkeypatch, process, [process.pid])
    job.assign(process)
    result = job.cleanup(process)
    assert result["empty"] and result["terminated"]
    assert not result["unexpected_residuals"]
    assert process.kills == 0 and api.terminations == 1


def test_containment_failed_assignment_stops_only_waiting_worker(monkeypatch):
    process = Process()
    job, api = windows(monkeypatch, process)
    api.assign_ok = False
    with pytest.raises(OSError, match="AssignProcess"):
        job.assign(process)
    result = job.cleanup(process)
    assert result["empty"] and process.kills == 1 and api.terminations == 0


def test_containment_query_failure_never_becomes_empty_success(monkeypatch):
    process = Process(done=True)
    job, api = windows(monkeypatch, process)
    job.assign(process)
    api.query_ok = False
    with pytest.raises(OSError, match="QueryInformation"):
        job.cleanup(process)
    assert api.closes == 1


def test_containment_close_failure_keeps_handle_for_retry(monkeypatch):
    job, api = windows(monkeypatch, Process(done=True))
    api.close_ok = False
    with pytest.raises(OSError, match="CloseHandle"):
        job.close()
    assert job.handle == 999
    api.close_ok = True
    job.close()
    assert job.handle is None


def test_containment_idle_and_repeated_close_are_benign(monkeypatch):
    monkeypatch.setattr(C, "_platform", lambda: "linux")
    idle = C.WindowsJob()
    idle.close()
    idle.close()
    assert idle.handle is None
    job, api = windows(monkeypatch, Process(done=True))
    job.close()
    job.close()
    assert api.closes == 1


def test_containment_job_timeout_reports_actual_members(monkeypatch):
    clock(monkeypatch)
    process = Process(done=True)
    job, api = windows(monkeypatch, process, [222, 333])
    job.assign(process)
    api.drain = False
    result = job.cleanup(process, timeout=0.01)
    assert not result["empty"] and result["residuals"] == [222, 333]
    assert result["unexpected_residuals"] and api.closes == 1


def linux(monkeypatch, process, *, exists=True, drain=True):
    monkeypatch.setattr(C, "_platform", lambda: "linux")
    # Replace this module's binding, not the host's shared signal module.
    monkeypatch.setattr(C, "signal", SimpleNamespace(SIGKILL=9))
    monkeypatch.setattr(C.os, "getpgrp", lambda: 7, raising=False)
    monkeypatch.setattr(C.os, "getpgid", lambda pid: pid, raising=False)
    state = dict(exists=exists, signals=[])

    def killpg(pgid, sig):
        assert pgid == process.pid  # Nothing outside the registered owned group.
        state["signals"].append((pgid, sig))
        if not state["exists"]:
            raise ProcessLookupError()
        if sig == C.signal.SIGKILL and drain:
            state["exists"] = False
            process.returncode = -9

    monkeypatch.setattr(C.os, "killpg", killpg, raising=False)
    return state


def test_containment_linux_ended_leader_does_not_skip_owned_group(monkeypatch):
    process = Process(done=True)
    state = linux(monkeypatch, process)
    result = C.cleanup_owned(process, None)
    assert (101, C.signal.SIGKILL) in state["signals"]
    assert result["empty"] and result["unexpected_residuals"]


def test_containment_linux_empty_finished_group_is_clean(monkeypatch):
    process = Process(done=True)
    state = linux(monkeypatch, process, exists=False)
    result = C.cleanup_owned(process, None)
    assert result["empty"] and not result["terminated"]
    assert all(sig == 0 for _, sig in state["signals"])


def test_containment_linux_cannot_kill_current_or_nonisolated_group(monkeypatch):
    process = Process()
    state = linux(monkeypatch, process)
    monkeypatch.setattr(C.os, "getpgrp", lambda: process.pid)
    with pytest.raises(ValueError, match="unsafe"):
        C.cleanup_owned(process, None)
    monkeypatch.setattr(C.os, "getpgrp", lambda: 7)
    monkeypatch.setattr(C.os, "getpgid", lambda pid: 8)
    with pytest.raises(ValueError, match="isolated"):
        C.cleanup_owned(process, None)
    assert state["signals"] == []


def test_containment_linux_timeout_retains_residuals(monkeypatch):
    clock(monkeypatch)
    process = Process(done=True)
    linux(monkeypatch, process, drain=False)
    monkeypatch.setattr(C, "_linux_members", lambda pgid: [222])
    result = C.cleanup_owned(process, None, timeout=0.01)
    assert not result["empty"] and result["residuals"] == [222]
    assert result["unexpected_residuals"]


def test_containment_linux_permission_failure_is_not_empty(monkeypatch):
    process = Process(done=True)
    linux(monkeypatch, process)

    def denied(pgid, sig):
        raise PermissionError("query denied")

    monkeypatch.setattr(C.os, "killpg", denied)
    with pytest.raises(PermissionError):
        C.cleanup_owned(process, None)


def test_containment_proc_parser_handles_spaces_and_parentheses(monkeypatch):
    entry = SimpleNamespace(name="222")

    class Item:
        name = entry.name

        def __truediv__(self, name):
            assert name == "stat"
            return SimpleNamespace(
                read_text=lambda: "222 (worker (child)) Z 1 101 101 0"
            )

    monkeypatch.setattr(C.Path, "iterdir", lambda path: iter([Item()]))
    assert C._linux_members(101) == [222]
    assert C._linux_members(7) == []


@pytest.mark.parametrize("timeout", [0, -1, 6, True, float("nan"), float("inf")])
def test_containment_cleanup_timeout_is_bounded(timeout):
    with pytest.raises(ValueError):
        C.cleanup_owned(None, None, timeout=timeout)
