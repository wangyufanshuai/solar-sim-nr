"""Prepared metadata/resource fault controls; no scientific worker is required."""

import importlib.util
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.case12_binary64_v1 import runtime as rt


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def deterministic_resource_queries(monkeypatch):
    """These controls never inspect the host's free RAM, disk or process RSS."""
    original_resources = rt.resources
    monkeypatch.setattr(rt.shutil, "disk_usage", lambda path: SimpleNamespace(free=64 * 1024**3))
    monkeypatch.setattr(rt, "resources", lambda pid=None: {
        "available_ram_bytes": rt.MIN_FREE_RAM,
        "total_ram_bytes": 2 * rt.MIN_FREE_RAM,
        "worker_rss_bytes": 1024,
        "worker_private_bytes": 1024,
        "worker_peak_rss_bytes": 1024,
    })
    return original_resources


class ScanResult:
    def __init__(self, entries):
        self.entries = entries

    def __enter__(self):
        return iter(self.entries)

    def __exit__(self, *exception):
        return False


class VanishedEntry:
    """Inject stat failure without depending on platform DirEntry caches."""

    def __init__(self, path):
        self.path = str(path)

    def stat(self, *, follow_symlinks):
        assert follow_symlinks is False
        raise FileNotFoundError("injected disappearance after enumeration")


def entrypoint():
    spec = importlib.util.spec_from_file_location(
        "binary64_fault_control_entry", ROOT / "examples/reproduce_binary64_alignment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("token", ["1e999", "-1e999", "1.7976931348623159e308"])
def test_exponent_overflow_is_rejected_as_nonfinite_json(token):
    with pytest.raises(ValueError, match="nonfinite"):
        rt.decode(('{"value":' + token + "}").encode("ascii"))


@pytest.mark.parametrize("token,value", [("1.25e2", 125.0), ("-2.5e-3", -0.0025)])
def test_finite_exponent_metadata_remains_valid(token, value):
    assert rt.decode(('{"value":' + token + "}").encode("ascii")) == {"value": value}


def test_manifest_dimension_float_alias_is_not_the_registered_integer(tmp_path, monkeypatch):
    manifest = rt.decode((ROOT / rt.MANIFEST_PATH).read_bytes())
    assert type(manifest["array_shapes"]["wave_numbers"][0]) is int
    manifest["array_shapes"]["wave_numbers"][0] = 2.0
    altered = tmp_path / "float-dimension-manifest.json"
    altered.write_bytes(rt.encode(manifest))
    original_safe_path = rt.safe_path

    def substituted_path(root, name, *, exists=True):
        if name == rt.MANIFEST_PATH:
            return altered
        return original_safe_path(root, name, exists=exists)

    monkeypatch.setattr(rt, "safe_path", substituted_path)
    with pytest.raises(ValueError, match="layout"):
        rt.verify_inputs(ROOT)


def test_deadline_is_rechecked_after_a_slow_census(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(rt.time, "monotonic", lambda: clock[0])
    writer = rt.Evidence(tmp_path, 1.0)

    def slow_census():
        clock[0] = 2.0
        return {}

    monkeypatch.setattr(writer, "census", slow_census)
    with pytest.raises(TimeoutError):
        writer.check()


def test_metadata_mode_cannot_report_success_after_budget_work_exceeds_deadline(monkeypatch):
    entry = entrypoint()
    args = entry.parse_args(["--verify-inputs-only", "--timeout-seconds", "1"])
    clock = [0.0]
    monkeypatch.setattr(entry.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(entry.rt, "verify_inputs", lambda root, check: {})

    def slow_budget(verified):
        clock[0] = 2.0
        return {}

    monkeypatch.setattr(entry.rt, "output_budget", slow_budget)
    with pytest.raises(TimeoutError):
        entry.supervise(args)


def test_scandir_open_failure_is_not_an_empty_successful_census(tmp_path, monkeypatch):
    (tmp_path / "partial.bin").write_bytes(b"preserve")
    writer = rt.Evidence(tmp_path, 60.0)

    def denied(path):
        raise PermissionError("injected scan denial")

    monkeypatch.setattr(rt.os, "scandir", denied)
    with pytest.raises(PermissionError, match="scan denial"):
        writer.census()
    assert (tmp_path / "partial.bin").read_bytes() == b"preserve"


def test_scandir_iteration_failure_does_not_return_a_partial_inventory(tmp_path, monkeypatch):
    (tmp_path / "partial.bin").write_bytes(b"preserve")
    with rt.os.scandir(tmp_path) as stream:
        first = next(stream)
    writer = rt.Evidence(tmp_path, 60.0)

    def interrupted():
        yield first
        raise OSError("injected iterator failure")

    monkeypatch.setattr(rt.os, "scandir", lambda path: ScanResult(interrupted()))
    with pytest.raises(OSError, match="iterator failure"):
        writer.census()


@pytest.mark.parametrize("filename", ["partial.npy", "progress.next", "not-owned.next"])
def test_unregistered_disappearing_entry_fails_closed(tmp_path, monkeypatch, filename):
    path = tmp_path / filename
    entry = VanishedEntry(path)
    writer = rt.Evidence(tmp_path, 60.0)
    monkeypatch.setattr(rt.os, "scandir", lambda directory: ScanResult([entry]))
    with pytest.raises(ValueError, match="ordinary evidence disappeared"):
        writer.census()


@pytest.mark.parametrize("filename", [
    "start.next", "production/progress.next", "replay/progress.next",
])
def test_only_registered_atomic_staging_disappearance_is_tolerated(tmp_path, monkeypatch, filename):
    path = tmp_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    real_scandir = rt.os.scandir
    vanished = VanishedEntry(path)
    writer = rt.Evidence(tmp_path, 60.0)

    def scan(directory):
        return ScanResult([vanished]) if Path(directory) == path.parent else real_scandir(directory)

    monkeypatch.setattr(rt.os, "scandir", scan)
    assert writer.census() == {}


@pytest.mark.parametrize("slow_stage", ["disk", "ram"])
def test_deadline_is_rechecked_after_resource_queries(tmp_path, monkeypatch, slow_stage):
    clock = [0.0]
    monkeypatch.setattr(rt.time, "monotonic", lambda: clock[0])
    writer = rt.Evidence(tmp_path, 1.0, check_resources=True)

    def slow_disk(path):
        clock[0] = 2.0
        return SimpleNamespace(free=64 * 1024**3)

    def slow_ram(pid=None):
        clock[0] = 2.0
        return {"available_ram_bytes": rt.MIN_FREE_RAM,
                "worker_rss_bytes": 1024, "worker_private_bytes": 1024}

    if slow_stage == "disk":
        monkeypatch.setattr(rt.shutil, "disk_usage", slow_disk)
    else:
        monkeypatch.setattr(rt, "resources", slow_ram)
    with pytest.raises(TimeoutError):
        writer.check()


@pytest.mark.parametrize("startup,additional", [(False, 0), (False, 17), (True, 0)])
def test_disk_floor_exact_boundary_and_one_byte_short(tmp_path, monkeypatch, startup, additional):
    monkeypatch.setattr(rt, "sys", SimpleNamespace(platform="linux"))
    floor = rt.MIN_FREE_DISK if startup else rt.RUN_FREE_DISK + additional
    free = [floor - 1]
    monkeypatch.setattr(rt.shutil, "disk_usage", lambda path: SimpleNamespace(free=free[0]))
    with pytest.raises(ValueError, match="output-volume"):
        rt.disk_headroom(tmp_path, additional=additional, startup=startup)
    free[0] = floor
    assert rt.disk_headroom(tmp_path, additional=additional, startup=startup)["output_free_bytes"] == floor


def test_windows_system_disk_floor_is_independent_of_output_volume(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "sys", SimpleNamespace(platform="win32"))
    reads = iter([rt.MIN_FREE_DISK + 1, 3 * 1024**3 - 1])
    monkeypatch.setattr(rt.shutil, "disk_usage", lambda path: SimpleNamespace(free=next(reads)))
    with pytest.raises(ValueError, match="3 GiB"):
        rt.disk_headroom(tmp_path, startup=True)


def test_ordinary_write_rechecks_disk_with_its_entire_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(rt.time, "monotonic", lambda: 0.0)
    payload = b"four"
    monkeypatch.setattr(rt.shutil, "disk_usage", lambda path: SimpleNamespace(
        free=rt.RUN_FREE_DISK + len(payload) - 1))
    writer = rt.Evidence(tmp_path, 60.0)
    with pytest.raises(ValueError, match="output-volume"):
        writer.write("not-written.bin", payload)
    assert not (tmp_path / "not-written.bin").exists()


def test_terminal_slot_is_durable_initial_failure_and_fixed_two_mib(tmp_path, monkeypatch):
    calls = []
    fsync = rt.os.fsync

    def observed_fsync(fd):
        calls.append(fd)
        return fsync(fd)

    monkeypatch.setattr(rt.os, "fsync", observed_fsync)
    slot = rt.TerminalSlot(tmp_path)
    try:
        path = tmp_path / "terminal.json"
        assert path.stat().st_size == rt.RESERVE_BYTES == 2 * 1024**2
        initial = rt.decode(path.read_bytes())
        assert initial["accepted"] is False and initial["completed"] is False
        assert initial["state"] == "NOT_QUALIFIED"
        assert calls
        identity = (path.stat().st_dev, path.stat().st_ino)
        slot.write({"accepted": False, "state": "NOT_QUALIFIED", "failure": "injected"})
        assert (path.stat().st_dev, path.stat().st_ino) == identity
        assert path.stat().st_size == 2 * 1024**2
        assert len(calls) >= 2
    finally:
        slot.close()
    assert rt.decode(path.read_bytes())["failure"] == "injected"


@pytest.mark.parametrize("fault", ["census", "low_disk"])
def test_terminal_failure_survives_broken_census_or_no_disk_headroom(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(rt.time, "monotonic", lambda: 0.0)
    slot = rt.TerminalSlot(tmp_path)
    path = tmp_path / "terminal.json"
    writer = rt.Evidence(tmp_path, 60.0, terminal_reserved=True)
    before = path.stat()
    try:
        if fault == "census":
            def failed_census():
                raise OSError("injected directory scan failure")
            monkeypatch.setattr(writer, "census", failed_census)
            expected = OSError
        else:
            monkeypatch.setattr(rt.shutil, "disk_usage", lambda parent: SimpleNamespace(free=0))
            expected = ValueError
        with pytest.raises(expected):
            writer.check()
        with monkeypatch.context() as local:
            def no_reopen(*args, **kwargs):
                raise AssertionError("terminal update must use its existing descriptor")
            local.setattr(Path, "open", no_reopen)
            slot.write({"accepted": False, "state": "NOT_QUALIFIED", "fault": fault})
    finally:
        slot.close()
    after = path.stat()
    assert (after.st_dev, after.st_ino, after.st_size) == (
        before.st_dev, before.st_ino, 2 * 1024**2)
    assert rt.decode(path.read_bytes())["fault"] == fault


def test_oversize_terminal_record_keeps_preallocated_failure_unchanged(tmp_path):
    slot = rt.TerminalSlot(tmp_path)
    path = tmp_path / "terminal.json"
    initial = path.read_bytes()
    try:
        with pytest.raises(ValueError, match="exceeds reserved slot"):
            slot.write({"accepted": True, "too_large": "x" * rt.RESERVE_BYTES})
        assert path.read_bytes() == initial
    finally:
        slot.close()


def test_reserved_terminal_is_counted_and_cannot_be_written_by_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(rt.time, "monotonic", lambda: 0.0)
    slot = rt.TerminalSlot(tmp_path)
    writer = rt.Evidence(tmp_path, 60.0, terminal_reserved=True)
    try:
        assert writer.census() == {"terminal.json": rt.RESERVE_BYTES}
        with pytest.raises(ValueError, match="TerminalSlot"):
            writer.json("terminal", {"accepted": True}, terminal=True)
        monkeypatch.setattr(rt, "MAX_BYTES", rt.RESERVE_BYTES + 3)
        writer.write("fits.bin", b"abc")
        with pytest.raises(ValueError, match="output cap"):
            writer.write("exceeds.bin", b"x")
        assert not (tmp_path / "exceeds.bin").exists()
        assert sum(writer.census().values()) == rt.MAX_BYTES
        assert rt.decode((tmp_path / "terminal.json").read_bytes())["accepted"] is False
    finally:
        slot.close()


def test_windows_memory_query_close_failure_rejects_snapshot(
    monkeypatch, deterministic_resource_queries
):
    close_calls = []
    kernel = SimpleNamespace(
        OpenProcess=lambda *a: 123,
        CloseHandle=lambda handle: close_calls.append(handle) or 0,
    )
    psapi = SimpleNamespace(GetProcessMemoryInfo=lambda *a: 1)
    monkeypatch.setattr(rt, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(rt.ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(
        GlobalMemoryStatusEx=lambda p: 1)), raising=False)
    monkeypatch.setattr(rt.ctypes, "WinDLL",
        lambda name, **kw: kernel if name == "kernel32" else psapi, raising=False)
    with pytest.raises(ValueError, match="memory-query handle close failed"):
        deterministic_resource_queries(123)
    assert close_calls == [123]


def test_bounded_read_rejects_oversized_stat_without_open(monkeypatch):
    monkeypatch.setattr(rt, "regular_metadata", lambda path: SimpleNamespace(st_size=5))

    def forbidden_open(*args, **kwargs):
        raise AssertionError("oversized source must not be opened")

    monkeypatch.setattr(Path, "open", forbidden_open)
    with pytest.raises(ValueError, match="file exceeds read limit"):
        rt.bounded_read(Path("mock-source"), 4)


def test_bounded_read_rejects_growth_with_limit_plus_one_read(monkeypatch):
    reads = []

    class Reader(BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    monkeypatch.setattr(rt, "regular_metadata", lambda path: SimpleNamespace(st_size=4))
    monkeypatch.setattr(Path, "open", lambda *a, **k: Reader(b"abcde-more"))
    with pytest.raises(ValueError, match="file grew beyond read limit"):
        rt.bounded_read(Path("mock-source"), 4)
    assert reads == [5]


def test_bounded_read_rechecks_deadline_after_successful_read(monkeypatch):
    finished = []

    class Reader(BytesIO):
        def read(self, size=-1):
            result = super().read(size)
            finished.append(size)
            return result

    def check():
        if finished:
            raise TimeoutError("expired while reading")

    monkeypatch.setattr(rt, "regular_metadata", lambda path: SimpleNamespace(st_size=4))
    monkeypatch.setattr(Path, "open", lambda *a, **k: Reader(b"abcd"))
    with pytest.raises(TimeoutError, match="expired while reading"):
        rt.bounded_read(Path("mock-source"), 4, check)
    assert finished == [5]
