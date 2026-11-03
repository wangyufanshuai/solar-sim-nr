"""Standard-library resource, input and evidence guards; no scientific imports."""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import sys
import time
import zipfile

MAX_SECONDS = 600
MAX_BYTES = 64 * 1024**2
RESERVE_BYTES = 2 * 1024**2
JSON_BUDGET = 4 * 1024**2
SNAPSHOT_BYTES = 8 * 1024**2
MIN_FREE_RAM = 4 * 1024**3
RUN_FREE_RAM = 2 * 1024**3
WORKER_BYTES = 1024**3
RUN_FREE_DISK = 4 * 1024**3 + 32 * 1024**2
MIN_FREE_DISK = RUN_FREE_DISK + MAX_BYTES
MANIFEST_PATH = "experiments/case12_binary64_v1/manifest.json"
# Existing Git blob uses LF; its Windows worktree copy uses CRLF. Only this
# metadata file has LF text representation. Scientific sources/parents stay raw.
EXPORT_SHA256 = "9ffd2abaf0c3070ace18981a8b8d4bb7f97531273408faa72dd124bba5e738b3"
ARRAY_SHAPES = {
    "q0": (58,),
    "scales": (58,),
    "shape_field": (8, 8, 8),
    "shape_projection": (13, 2),
    "wave_numbers": (2, 13, 3),
    "reference_rhs": (4, 58, 58),
    "reference_constraints": (10, 43, 58),
    "background_fields": (2, 101, 8, 8, 8),
    "background_projections": (2, 101, 13, 2),
    "main_projections": (2, 2, 58, 2, 101, 13, 2),
    "negative_projections": (2, 2, 2, 2, 101, 13, 2),
    **{
        n: (2, 2, 4, 58, 58)
        for n in ("rhs_coefficients", "rhs_errors", "rhs_resolution")
    },
    **{
        n: (2, 2, 10, 43, 58)
        for n in (
            "constraint_coefficients",
            "constraint_errors",
            "constraint_resolution",
        )
    },
    "modal_errors": (2, 2, 58, 101, 13, 2),
    "modal_limits": (2, 2, 58, 101, 13),
    "field_stats": (2, 2, 58, 101, 11),
    "roundtrip_stats": (482, 58, 3),
    "negative_changes": (2, 2, 2, 101, 13, 2),
    "negative_errors": (2, 2, 2, 101, 13, 2),
    "negative_resolution": (2, 2, 2, 101, 13),
    "negative_limits": (2, 2, 2, 101, 13),
    "negative_field_scales": (2, 2, 2, 101),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(record):
    return (
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON token: " + value)

    def finite(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON exponent")
        return result

    return json.loads(
        data, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite
    )


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bounded_read(path, limit, check=lambda: None):
    """Reject source growth before allocation and bound the read across races."""
    require(type(limit) is int and 0 <= limit <= SNAPSHOT_BYTES, "invalid read limit")
    check()
    require(regular_metadata(path).st_size <= limit, "file exceeds read limit")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    check()
    require(len(data) <= limit, "file grew beyond read limit")
    return data


def safe_path(root, name, *, exists=True):
    relative = PurePosixPath(name)
    require(
        type(name) is str
        and str(relative) == name
        and not relative.is_absolute()
        and all(
            p not in ("", ".", "..") and ":" not in p and "\\" not in p
            for p in relative.parts
        ),
        "unsafe relative path",
    )
    path = root
    for part in relative.parts:
        path = path / part
        if path.exists():
            metadata = path.lstat()
            require(
                not stat.S_ISLNK(metadata.st_mode)
                and not (getattr(metadata, "st_file_attributes", 0) & 0x400),
                "symlink/reparse input or output forbidden",
            )
    require(path.resolve().is_relative_to(root.resolve()), "path leaves root")
    if exists:
        require(path.is_file(), "required regular file missing: " + name)
    return path


def verify_inputs(root, check=lambda: None):
    """Hash all frozen export files and all new registered experiment files."""
    check()
    manifest_bytes = bounded_read(safe_path(root, MANIFEST_PATH), 1024**2, check)
    manifest = decode(manifest_bytes)
    require(manifest["schema"] == "case12-binary64-experiment-v1", "manifest schema")
    require(manifest["execution_status"] == "UNEXECUTED_CANDIDATE", "candidate status")
    fixed_contract = dict(
        grid=[8, 8, 8],
        orders=[2, 4],
        epsilons=["1/1024", "1/4096"],
        state_evaluations_per_path=482,
        total_state_evaluations=964,
        timeout_seconds=MAX_SECONDS,
        max_output_bytes=MAX_BYTES,
        terminal_reserve_bytes=RESERVE_BYTES,
        snapshot_limit_bytes=SNAPSHOT_BYTES,
        json_budget_bytes=JSON_BUDGET,
        min_available_ram_bytes=MIN_FREE_RAM,
        runtime_free_ram_bytes=RUN_FREE_RAM,
        startup_free_disk_bytes=MIN_FREE_DISK,
        runtime_free_disk_bytes=RUN_FREE_DISK,
        worker_memory_bytes=WORKER_BYTES,
        driver49_executed=False,
        time_evolution_executed=False,
    )
    require(
        all(
            encode(manifest.get(key)) == encode(expected)
            for key, expected in fixed_contract.items()
        ),
        "registered resource/science contract drift",
    )
    export_bytes = bounded_read(
        safe_path(root, "provenance/export-manifest.json"), 1024**2, check
    ).replace(b"\r\n", b"\n")
    require(digest(export_bytes) == EXPORT_SHA256, "original export manifest drift")
    exported = {f["path"]: f for f in decode(export_bytes)["files"]}
    records = manifest["files"]
    require(type(records) is list and records, "manifest entries required")
    require(
        all(type(r["bytes"]) is int and r["bytes"] >= 0 for r in records)
        and sum(r["bytes"] for r in records) + len(manifest_bytes) <= SNAPSHOT_BYTES,
        "registered inputs exceed read allocation budget",
    )
    by_path, raw = {}, {}
    for record in records:
        check()
        name = record["path"]
        require(
            name != MANIFEST_PATH and name not in by_path,
            "duplicate/self manifest entry",
        )
        data = bounded_read(
            safe_path(root, name),
            1024**2 if name == "provenance/export-manifest.json" else record["bytes"],
            check,
        )
        if name == "provenance/export-manifest.json":
            require(
                record.get("representation") == "lf-text",
                "export metadata LF representation required",
            )
            data = data.replace(b"\r\n", b"\n")
        else:
            require(
                record.get("representation", "verbatim") == "verbatim",
                "scientific bytes must remain raw",
            )
        require(
            type(record["bytes"]) is int
            and len(data) == record["bytes"]
            and digest(data) == record["sha256"],
            "bound input changed: " + name,
        )
        by_path[name], raw[name] = record, data
    for name, item in exported.items():
        require(
            name in by_path
            and all(by_path[name][k] == item[k] for k in ("bytes", "sha256")),
            "export identity drift",
        )
    required = {
        "provenance/export-manifest.json",
        "examples/reproduce_binary64_alignment.py",
    }
    required.update(
        "experiments/case12_binary64_v1/" + n
        for n in (
            "runtime.py",
            "containment.py",
            "kernel.py",
            "replay.py",
            "comparison.py",
            "provenance.py",
            "protocol-note.md",
        )
    )
    require(required <= by_path.keys(), "required implementation not bound")
    aliases = manifest["input_aliases"]
    expected_aliases = {
        "volume/" + name: "experiments/case12_binary64_v1/inputs/volume/" + name
        for name in (
            "baseline__F0.json",
            "baseline__J.json",
            "baseline__Ax.json",
            "baseline__Ay.json",
            "baseline__Az.json",
            "baseline__execution.json",
        )
    }
    expected_aliases.update(
        {
            "constraint/C"
            + str(i)
            + ".json": "examples/data/constraint_bridge/parents/C" + str(i) + ".json"
            for i in range(3)
        }
    )
    require(aliases == expected_aliases, "registered parent aliases changed")
    require(
        type(aliases) is dict and all(v in raw for v in aliases.values()),
        "input aliases",
    )
    require(
        encode(manifest["array_shapes"])
        == encode({k: list(v) for k, v in ARRAY_SHAPES.items()}),
        "artifact layout drift",
    )
    from .provenance import verify_parent_metadata

    parent_summary = verify_parent_metadata(raw)
    raw[MANIFEST_PATH] = manifest_bytes
    check()
    return dict(
        manifest=manifest,
        manifest_sha256=digest(manifest_bytes),
        raw=raw,
        inputs={k: raw[v] for k, v in aliases.items()},
        parents=parent_summary,
    )


def npy_header(shape):
    # NumPy v1 format: explicit little-endian, C order, fixed 64-byte alignment.
    value = repr(
        {"descr": "<f8", "fortran_order": False, "shape": tuple(shape)}
    ).encode("ascii")
    padding = (-(10 + len(value) + 1)) % 64
    value += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(value)) + value


def output_budget(verified):
    arrays = {
        name: len(npy_header(shape)) + 8 * math.prod(shape)
        for name, shape in ARRAY_SHAPES.items()
    }
    # ZIP_STORED, no comments, no extra fields, no ZIP64, ASCII paths.
    archive = 22 + sum(
        76 + 2 * len(name.encode("ascii")) + len(data)
        for name, data in verified["raw"].items()
    )
    require(archive <= SNAPSHOT_BYTES, "registered snapshot exceeds 8 MiB")
    total = 2 * sum(arrays.values()) + archive + JSON_BUDGET + RESERVE_BYTES
    require(total <= MAX_BYTES, "registered uncompressed output budget exceeds 64 MiB")
    return dict(
        per_path_arrays=arrays,
        two_path_array_bytes=2 * sum(arrays.values()),
        archive_bytes=archive,
        json_budget=JSON_BUDGET,
        terminal_reserve=RESERVE_BYTES,
        snapshot_limit_bytes=SNAPSHOT_BYTES,
        registered_upper_bytes=total,
        limit_bytes=MAX_BYTES,
    )


def resources(pid=None):
    if sys.platform == "win32":
        from ctypes import wintypes

        class Memory(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page_total",
                    "page_available",
                    "virtual_total",
                    "virtual_available",
                    "extended",
                )
            ]

        data = Memory()
        data.length = ctypes.sizeof(data)
        require(
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(data)),
            "RAM query failed",
        )
        result = {
            "available_ram_bytes": int(data.available),
            "total_ram_bytes": int(data.total),
        }
        if pid is not None:

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                    (n, ctypes.c_size_t)
                    for n in (
                        "peak_rss",
                        "rss",
                        "peak_pool",
                        "pool",
                        "peak_nonpaged",
                        "nonpaged",
                        "pagefile",
                        "peak_pagefile",
                        "private",
                    )
                ]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.OpenProcess.argtypes = (
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            )
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel.CloseHandle.restype = wintypes.BOOL
            handle = kernel.OpenProcess(0x1000 | 0x0010, False, pid)
            require(handle, "worker process query failed")
            try:
                counters = Counters()
                counters.cb = ctypes.sizeof(counters)
                query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
                query.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
                require(
                    query(handle, ctypes.byref(counters), counters.cb),
                    "worker memory query failed",
                )
                result.update(
                    worker_rss_bytes=int(counters.rss),
                    worker_private_bytes=int(counters.private),
                    worker_peak_rss_bytes=int(counters.peak_rss),
                )
            finally:
                require(kernel.CloseHandle(handle), "memory-query handle close failed")
        return result
    require(
        sys.platform.startswith("linux"),
        "resource monitor supports Windows and Linux only",
    )
    items = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    result = {
        "available_ram_bytes": int(items["MemAvailable"].split()[0]) * 1024,
        "total_ram_bytes": int(items["MemTotal"].split()[0]) * 1024,
    }
    if pid is not None:
        items = dict(
            line.split(":", 1)
            for line in Path(f"/proc/{pid}/status").read_text().splitlines()
        )
        result.update(
            worker_rss_bytes=int(items["VmRSS"].split()[0]) * 1024,
            worker_peak_rss_bytes=int(items["VmHWM"].split()[0]) * 1024,
        )
    return result


def disk_headroom(parent, *, additional=0, startup=False):
    require(
        type(additional) is int and additional >= 0, "nonnegative write reservation"
    )
    result = {"output_free_bytes": shutil.disk_usage(parent).free}
    floor = MIN_FREE_DISK if startup else RUN_FREE_DISK + additional
    require(result["output_free_bytes"] >= floor, "insufficient output-volume space")
    if sys.platform == "win32":
        system = Path(os.environ.get("SystemRoot", r"C:\Windows")).anchor
        result["system_free_bytes"] = shutil.disk_usage(system).free
        require(
            result["system_free_bytes"] >= 3 * 1024**3,
            "need 3 GiB system-volume free space",
        )
    return result


def preflight(parent):
    require(parent.is_dir(), "output parent directory must already exist")
    result = resources()
    require(
        result["available_ram_bytes"] >= MIN_FREE_RAM,
        "need at least 4 GiB available RAM",
    )
    result.update(disk_headroom(parent, startup=True))
    require(struct.calcsize("P") == 8, "64-bit Python required")
    return result


def regular_metadata(path, *, directory=False):
    metadata = path.lstat()
    require(
        not stat.S_ISLNK(metadata.st_mode)
        and not (getattr(metadata, "st_file_attributes", 0) & 0x400)
        and (
            stat.S_ISDIR(metadata.st_mode)
            if directory
            else stat.S_ISREG(metadata.st_mode)
        ),
        "link/reparse/special output forbidden",
    )
    return metadata


class TerminalSlot:
    """Allocate failure storage before launch; finalization needs no free blocks.

    A bounded JSON record plus whitespace occupies exactly the old 2 MiB reserve.
    The open descriptor and identity checks avoid reopening or growing the slot
    when an ordinary census/disk check has failed. Hardware/permission failures
    can still prevent persistence and must be reported as such.
    """

    def __init__(self, directory):
        self.path = Path(directory) / "terminal.json"
        regular_metadata(Path(directory), directory=True)
        self.stream = self.path.open("x+b")
        try:
            payload = encode(
                dict(
                    state="NOT_QUALIFIED",
                    accepted=False,
                    completed=False,
                    phase="preallocated_before_worker_start",
                )
            )
            self.stream.write(payload + b" " * (RESERVE_BYTES - len(payload)))
            self.stream.flush()
            os.fsync(self.stream.fileno())
            metadata = os.fstat(self.stream.fileno())
            require(metadata.st_size == RESERVE_BYTES, "terminal allocation incomplete")
            self.identity = (metadata.st_dev, metadata.st_ino)
        except BaseException:
            self.stream.close()
            raise

    def write(self, record):
        payload = encode(record)
        require(len(payload) <= RESERVE_BYTES, "terminal record exceeds reserved slot")
        padded = payload + b" " * (RESERVE_BYTES - len(payload))
        sha256 = digest(padded)
        metadata = regular_metadata(self.path)
        require(
            (metadata.st_dev, metadata.st_ino) == self.identity
            and metadata.st_size == RESERVE_BYTES,
            "terminal reserve changed",
        )
        self.stream.seek(0)
        self.stream.write(padded)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        final = regular_metadata(self.path)
        require(
            (final.st_dev, final.st_ino) == self.identity
            and final.st_size == RESERVE_BYTES,
            "terminal identity changed while writing",
        )
        return sha256

    def close(self):
        self.stream.close()


class Evidence:
    """Single worker writes ordinary files; parent writes reserved terminal last."""

    def __init__(
        self, directory, deadline, *, check_resources=False, terminal_reserved=False
    ):
        self.directory = Path(directory)
        metadata = self.directory.lstat()
        require(
            stat.S_ISDIR(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
            and not (getattr(metadata, "st_file_attributes", 0) & 0x400),
            "output root must be a regular directory",
        )
        self.deadline = deadline
        self.check_resources = check_resources
        self.root_identity = (metadata.st_dev, metadata.st_ino)
        self.terminal_reserved = terminal_reserved

    def census(self):
        result = {}
        root = regular_metadata(self.directory, directory=True)
        require(
            (root.st_dev, root.st_ino) == self.root_identity, "output root replaced"
        )
        staging = {"production/progress.next", "replay/progress.next", "start.next"}

        def walk(directory, identity):
            # New iterator and fresh stat each scan. Enumeration errors propagate;
            # Path.rglob may suppress some permission errors on supported Python.
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    name = path.relative_to(self.directory).as_posix()
                    try:
                        # Windows DirEntry can reuse enumeration metadata.
                        metadata = os.stat(entry.path, follow_symlinks=False)
                    except FileNotFoundError:
                        require(
                            name in staging,
                            "ordinary evidence disappeared during census",
                        )
                        continue  # only the three owned atomic publication names
                    require(
                        not stat.S_ISLNK(metadata.st_mode)
                        and not (getattr(metadata, "st_file_attributes", 0) & 0x400),
                        "unexpected output link/reparse",
                    )
                    if stat.S_ISDIR(metadata.st_mode):
                        walk(path, (metadata.st_dev, metadata.st_ino))
                    else:
                        require(
                            stat.S_ISREG(metadata.st_mode), "special output forbidden"
                        )
                        result[name] = metadata.st_size
            current = regular_metadata(directory, directory=True)
            require(
                (current.st_dev, current.st_ino) == identity,
                "output directory replaced during scan",
            )

        walk(self.directory, self.root_identity)
        if self.terminal_reserved:
            require(
                result.get("terminal.json") == RESERVE_BYTES,
                "terminal reservation missing/changed",
            )
        return result

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("bounded deadline")

    def ordinary_limit(self):
        return MAX_BYTES if self.terminal_reserved else MAX_BYTES - RESERVE_BYTES

    def check(self):
        self.check_deadline()
        require(sum(self.census().values()) <= self.ordinary_limit(), "output limit")
        disk_headroom(self.directory)
        if self.check_resources:
            snapshot = resources(os.getpid())
            require(
                snapshot["available_ram_bytes"] >= RUN_FREE_RAM,
                "runtime RAM reserve below 2 GiB",
            )
            require(
                max(
                    snapshot.get("worker_private_bytes", 0),
                    snapshot["worker_rss_bytes"],
                )
                <= WORKER_BYTES,
                "worker memory limit",
            )
        self.check_deadline()

    def write(self, name, data, *, progress=False, terminal=False, publish=False):
        require(type(data) is bytes, "evidence must be bytes")
        require(
            not self.terminal_reserved or name != "terminal.json",
            "only TerminalSlot may update reserve",
        )
        if not terminal:
            self.check()
        path = safe_path(self.directory, name, exists=False)
        require(not path.exists() or progress, "refuse to overwrite existing evidence")
        limit = MAX_BYTES if terminal else self.ordinary_limit()
        # Include both old and replacement progress bytes during atomic replacement.
        require(sum(self.census().values()) + len(data) <= limit, "evidence output cap")
        if name.endswith(".json") and not terminal:
            json_bytes = sum(
                size
                for key, size in self.census().items()
                if key.endswith(".json")
                and not (self.terminal_reserved and key == "terminal.json")
            )
            require(json_bytes + len(data) <= JSON_BUDGET, "registered JSON budget")
        require(
            not progress or (name.endswith("/progress.json") and len(data) <= 8192),
            "only small live counters may be replaced",
        )
        require(
            not publish or name == "start.json",
            "only worker release uses atomic publication",
        )
        if not terminal:
            disk_headroom(self.directory, additional=len(data))
            self.check_deadline()
        path.parent.mkdir(parents=True, exist_ok=True)
        if progress or publish:
            temporary = path.with_suffix(".next")
            with temporary.open("xb") as stream:
                stream.write(data)
            if progress:
                os.replace(temporary, path)
            else:
                require(not path.exists(), "worker release already exists")
                os.rename(temporary, path)
        else:
            with path.open("xb") as stream:
                stream.write(data)

    def json(self, name, value, **options):
        self.write(name + ".json", encode(value), **options)

    def array(self, name, value):
        # Called only by the worker after the RAM gate; no NumPy import here.
        short = name.split("/")[-1]
        require(
            short in ARRAY_SHAPES and tuple(value.shape) == ARRAY_SHAPES[short],
            "array shape",
        )
        require(
            value.dtype.str == "<f8" and value.flags.c_contiguous,
            "little-endian contiguous float64",
        )
        self.write(name + ".npy", npy_header(value.shape) + value.tobytes(order="C"))

    def snapshot(self, verified):
        archive = io.BytesIO()
        with zipfile.ZipFile(
            archive, "w", compression=zipfile.ZIP_STORED, allowZip64=False
        ) as stream:
            for name, data in sorted(verified["raw"].items()):
                self.check()
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                stream.writestr(info, data)
        payload = archive.getvalue()
        require(
            len(payload) == output_budget(verified)["archive_bytes"],
            "archive size accounting",
        )
        self.write("source-input-snapshot.zip", payload)

    def seal(self, *, enforce_deadline):
        result = {}
        for name, size in sorted(self.census().items()):
            if self.terminal_reserved and name == "terminal.json":
                continue  # terminal cannot contain its own digest
            hasher = hashlib.sha256()
            path = safe_path(self.directory, name)
            before = regular_metadata(path)
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    if enforce_deadline:
                        self.check()
                    hasher.update(block)
            after = regular_metadata(path)
            require(
                before.st_size == after.st_size == size
                and (before.st_dev, before.st_ino, before.st_mtime_ns)
                == (after.st_dev, after.st_ino, after.st_mtime_ns),
                "output changed while sealing",
            )
            result[name] = dict(bytes=size, sha256=hasher.hexdigest())
        return result
