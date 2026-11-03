"""Owned-worker containment and verified cleanup, using only the standard library.

Linux callers must create the worker with start_new_session=True. The registered
PID is then the owned process-group ID, even after its leader exits. This module
never kills a process discovered outside that group or the assigned Windows Job.
"""

from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
import os
from pathlib import Path
import signal
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def worker_python(environment):
    """Launch CPython itself so Job assignment cannot miss a venv launcher child."""
    if sys.platform != "win32":
        return sys.executable
    require(sys.implementation.name == "cpython", "Windows worker requires CPython")
    executable = Path(sys._base_executable).resolve(strict=True)
    require(executable.is_file(), "base interpreter must be a regular file")
    environment["__PYVENV_LAUNCHER__"] = sys.executable
    return str(executable)


def _platform():
    if sys.platform == "win32":
        return "windows"
    require(sys.platform.startswith("linux"), "unsupported containment platform")
    return "linux"


class BasicLimits(C.Structure):
    _fields_ = [
        ("process_time", C.c_longlong),
        ("job_time", C.c_longlong),
        ("flags", C.c_uint32),
        ("min_ws", C.c_size_t),
        ("max_ws", C.c_size_t),
        ("active_limit", C.c_uint32),
        ("affinity", C.c_size_t),
        ("priority", C.c_uint32),
        ("scheduling", C.c_uint32),
    ]


class ExtendedLimits(C.Structure):
    _fields_ = [
        ("basic", BasicLimits),
        ("io", C.c_ulonglong * 6),
        ("process_limit", C.c_size_t),
        ("job_limit", C.c_size_t),
        ("peak_process", C.c_size_t),
        ("peak_job", C.c_size_t),
    ]


class Accounting(C.Structure):
    _fields_ = [("times", C.c_longlong * 4)] + [
        (name, C.c_uint32) for name in ("faults", "total", "active", "terminated")
    ]


def _windows_api():
    api = C.WinDLL("kernel32", use_last_error=True)
    definitions = {
        "CreateJobObjectW": ((C.c_void_p, W.LPCWSTR), W.HANDLE),
        "SetInformationJobObject": ((W.HANDLE, C.c_int, C.c_void_p, W.DWORD), W.BOOL),
        "QueryInformationJobObject": (
            (W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p),
            W.BOOL,
        ),
        "AssignProcessToJobObject": ((W.HANDLE, W.HANDLE), W.BOOL),
        "TerminateJobObject": ((W.HANDLE, W.UINT), W.BOOL),
        "CloseHandle": ((W.HANDLE,), W.BOOL),
    }
    for name, (args, result) in definitions.items():
        function = getattr(api, name)
        function.argtypes, function.restype = args, result
    return api


def _checked(value, operation):
    if not value:
        error = getattr(C, "get_last_error", lambda: 0)()
        raise OSError(error, operation + " failed")
    return value


def _timeout(value):
    require(type(value) in (int, float) and 0 < value <= 5, "cleanup timeout in (0,5]")
    return time.monotonic() + value


def _result(platform, process, before, terminated, residuals, *, closed):
    return dict(
        platform=platform,
        owned_pid=None if process is None else process.pid,
        leader_exited_before_cleanup=before,
        terminated=terminated,
        residuals=residuals,
        empty=not residuals,
        closed=closed,
        unexpected_residuals=before and terminated,
    )


class WindowsJob:
    def __init__(self, memory_bytes=1024**3):
        require(
            type(memory_bytes) is int and 0 < memory_bytes <= 1024**3,
            "positive memory limit at most 1 GiB",
        )
        self.handle, self.assigned_pid, self.api = None, None, None
        if _platform() != "windows":
            return
        self.api = _windows_api()
        self.handle = _checked(
            self.api.CreateJobObjectW(None, None), "CreateJobObjectW"
        )
        limits = ExtendedLimits()
        # KILL_ON_JOB_CLOSE | PROCESS_MEMORY | JOB_MEMORY: the memory envelope
        # covers both each process and the aggregate owned descendant tree.
        limits.basic.flags = 0x2000 | 0x100 | 0x200
        limits.process_limit = memory_bytes
        limits.job_limit = memory_bytes
        try:
            _checked(
                self.api.SetInformationJobObject(
                    self.handle, 9, C.byref(limits), C.sizeof(limits)
                ),
                "SetInformationJobObject",
            )
        except BaseException:
            self.close()
            raise

    def assign(self, process):
        if self.api is None:
            return
        require(
            self.handle is not None and self.assigned_pid is None,
            "fresh owned Job required",
        )
        _checked(
            self.api.AssignProcessToJobObject(self.handle, int(process._handle)),
            "AssignProcessToJobObject",
        )
        self.assigned_pid = process.pid

    def _active(self):
        record = Accounting()
        _checked(
            self.api.QueryInformationJobObject(
                self.handle, 1, C.byref(record), C.sizeof(record), None
            ),
            "QueryInformationJobObject(accounting)",
        )
        return int(record.active)

    def _members(self):
        capacity = max(1, self._active())
        for _ in range(8):

            class Pids(C.Structure):
                _fields_ = [
                    ("assigned", C.c_uint32),
                    ("count", C.c_uint32),
                    ("pids", C.c_size_t * capacity),
                ]

            record = Pids()
            ok = self.api.QueryInformationJobObject(
                self.handle, 3, C.byref(record), C.sizeof(record), None
            )
            if ok:
                require(record.count <= capacity, "invalid Job PID count")
                return sorted(int(record.pids[i]) for i in range(record.count))
            # ERROR_MORE_DATA is the sole permitted retry, with a larger buffer.
            error = getattr(C, "get_last_error", lambda: 0)()
            if error != 234:
                _checked(False, "QueryInformationJobObject(process list)")
            capacity = max(capacity * 2, int(record.assigned))
        raise RuntimeError("Job process list did not stabilize")

    def cleanup(self, process, timeout=5):
        deadline = _timeout(timeout)
        require(
            self.api is not None and self.handle is not None,
            "open Windows Job required",
        )
        require(
            process is None or self.assigned_pid in (None, process.pid),
            "different owned process",
        )
        before = process is None or process.poll() is not None
        terminated = False
        try:
            if self._active():
                _checked(
                    self.api.TerminateJobObject(self.handle, 1), "TerminateJobObject"
                )
                terminated = True
            # Assignment can fail before the release handshake. Such a worker
            # cannot have started scientific work; terminate its exact PID only.
            if (
                process is not None
                and self.assigned_pid is None
                and process.poll() is None
            ):
                process.kill()
                terminated = True
            while True:
                active = self._active()
                leader_running = process is not None and process.poll() is None
                if not active and not leader_running:
                    residuals = []
                    break
                if time.monotonic() >= deadline:
                    residuals = self._members() if active else []
                    if leader_running and process.pid not in residuals:
                        residuals.append(process.pid)
                    require(
                        not active or residuals, "active Job with unavailable PID list"
                    )
                    break
                time.sleep(min(0.02, max(0, deadline - time.monotonic())))
            return _result(
                "windows", process, before, terminated, residuals, closed=True
            )
        finally:
            # This is also a last containment action on a failed query. The
            # exception remains a failure; it cannot stand in for verified empty.
            self.close()

    def close(self):
        if self.handle is not None:
            _checked(self.api.CloseHandle(self.handle), "CloseHandle")
            self.handle = None


def _group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _linux_members(pgid):
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            raw = (entry / "stat").read_text()
        except FileNotFoundError:
            continue
        fields = raw[raw.rfind(")") + 2 :].split()
        require(len(fields) >= 3, "unreadable /proc PID identity")
        if int(fields[2]) == pgid:
            members.append(int(entry.name))
    return sorted(members)


def cleanup_owned(process, job, timeout=5):
    """Verify the entire owned container, including a group whose leader exited."""
    deadline = _timeout(timeout)
    if _platform() == "windows":
        if job is not None:
            return job.cleanup(process, timeout=timeout)
        require(process is None, "Windows worker has no owned Job")
        return _result("windows", None, True, False, [], closed=True)
    if process is None:
        return _result("linux", None, True, False, [], closed=True)
    pgid = process.pid
    require(
        type(pgid) is int and pgid > 0 and pgid != os.getpgrp(), "unsafe owned PGID"
    )
    before = process.poll() is not None
    if not before:
        try:
            require(
                os.getpgid(pgid) == pgid,
                "worker is not an isolated process-group leader",
            )
        except ProcessLookupError:
            require(process.poll() is not None, "worker group unavailable")
    terminated = False
    if _group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
            terminated = True
        except ProcessLookupError:
            pass
    while True:
        leader_running = process.poll() is None
        exists = _group_exists(pgid)
        if not exists and not leader_running:
            return _result("linux", process, before, terminated, [], closed=True)
        if time.monotonic() >= deadline:
            residuals = _linux_members(pgid) if exists else []
            if leader_running and pgid not in residuals:
                residuals.append(pgid)
            require(
                not exists or residuals,
                "owned group exists but members are unavailable",
            )
            return _result("linux", process, before, terminated, residuals, closed=True)
        time.sleep(min(0.02, max(0, deadline - time.monotonic())))
