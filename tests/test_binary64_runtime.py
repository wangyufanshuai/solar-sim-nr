"""Controls prepared for later execution; no scientific worker is needed here."""

import importlib.util
import io
from pathlib import Path
import struct
import time
import zipfile

import pytest

from experiments.case12_binary64_v1 import runtime as rt

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_covers_frozen_export_and_candidate():
    result = rt.verify_inputs(ROOT)
    assert set(result["inputs"]) >= {
        "constraint/C0.json",
        "constraint/C1.json",
        "constraint/C2.json",
    }
    assert "science/numerical_relativity/fo_ccz4_rhs.py" in result["raw"]
    assert "experiments/case12_binary64_v1/comparison.py" in result["raw"]
    assert rt.output_budget(result)["registered_upper_bytes"] <= rt.MAX_BYTES


@pytest.mark.parametrize("data", [b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_rejects_ambiguous_json(data):
    with pytest.raises(ValueError):
        rt.decode(data)


@pytest.mark.parametrize(
    "name", ["../outside", "a/../b", "C:/outside", "a\\b", "/tmp/out", ""]
)
def test_refuses_path_escape(tmp_path, name):
    with pytest.raises(ValueError):
        rt.safe_path(tmp_path, name, exists=False)


def test_deadline_refuses_ordinary_output_but_reserves_terminal(tmp_path):
    writer = rt.Evidence(tmp_path, time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        writer.json("result", {"accepted": True})
    writer.json("terminal", {"accepted": False}, terminal=True)
    assert rt.decode((tmp_path / "terminal.json").read_bytes())["accepted"] is False


def test_evidence_never_overwrites_except_small_progress(tmp_path):
    writer = rt.Evidence(tmp_path, time.monotonic() + 60)
    writer.json("production/progress", {"started": 1, "completed": 0}, progress=True)
    writer.json("production/progress", {"started": 1, "completed": 1}, progress=True)
    writer.json("result", {"accepted": False})
    with pytest.raises(ValueError):
        writer.json("result", {"accepted": True})
    with pytest.raises(ValueError):
        writer.json("result", {"accepted": True}, progress=True)
    assert (
        rt.decode((tmp_path / "production/progress.json").read_bytes())["completed"]
        == 1
    )
    assert rt.decode((tmp_path / "result.json").read_bytes())["accepted"] is False


def test_output_reserve_accounted_before_write(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "MAX_BYTES", 100)
    monkeypatch.setattr(rt, "RESERVE_BYTES", 20)
    writer = rt.Evidence(tmp_path, time.monotonic() + 60)
    writer.write("first", b"x" * 79)
    with pytest.raises(ValueError):
        writer.write("overflow", b"xx")
    assert not (tmp_path / "overflow").exists()
    writer.write("terminal", b"x" * 20, terminal=True)
    assert sum(writer.census().values()) == 99


def test_available_memory_is_not_total_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rt,
        "resources",
        lambda: {
            "total_ram_bytes": 64 * 1024**3,
            "available_ram_bytes": rt.MIN_FREE_RAM - 1,
        },
    )
    with pytest.raises(ValueError, match="available RAM"):
        rt.preflight(tmp_path)


def test_release_is_published_only_after_complete_write(tmp_path, monkeypatch):
    original = rt.os.rename
    observations = []

    def publish(source, destination):
        assert not destination.exists()
        observations.append(rt.decode(source.read_bytes()))
        original(source, destination)

    monkeypatch.setattr(rt.os, "rename", publish)
    writer = rt.Evidence(tmp_path, time.monotonic() + 60)
    writer.json("start", {"ready": True}, publish=True)
    assert observations == [{"ready": True}]
    assert rt.decode((tmp_path / "start.json").read_bytes()) == {"ready": True}
    assert not (tmp_path / "start.next").exists()


def test_seal_binds_every_byte_and_preserves_failed_evidence(tmp_path):
    writer = rt.Evidence(tmp_path, time.monotonic() + 60)
    writer.write("partial.npy", b"unfinished-payload")
    sealed = writer.seal(enforce_deadline=True)
    assert sealed == {
        "partial.npy": {"bytes": 18, "sha256": rt.digest(b"unfinished-payload")}
    }
    assert (tmp_path / "partial.npy").read_bytes() == b"unfinished-payload"


def test_unknown_memory_query_fails_closed(tmp_path, monkeypatch):
    def unavailable():
        raise OSError("no supported resource query")

    monkeypatch.setattr(rt, "resources", unavailable)
    with pytest.raises(OSError):
        rt.preflight(tmp_path)


def test_registered_npy_headers_are_complete_binary64_v1():
    for shape in rt.ARRAY_SHAPES.values():
        header = rt.npy_header(shape)
        assert header[:8] == b"\x93NUMPY\x01\x00"
        assert len(header) == 10 + struct.unpack("<H", header[8:10])[0]
        assert len(header) % 64 == 0
        assert b"'<f8'" in header and b"False" in header


def test_snapshot_is_uncompressed_and_byte_exact(tmp_path):
    data = {"a.py": b"line\r\n", "folder/input.json": b'{"x":0}\n'}
    verified = {"raw": data}
    writer = rt.Evidence(tmp_path, time.monotonic() + 60)
    writer.snapshot(verified)
    payload = (tmp_path / "source-input-snapshot.zip").read_bytes()
    assert len(payload) == rt.output_budget(verified)["archive_bytes"]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == data
        assert all(
            item.compress_type == zipfile.ZIP_STORED for item in archive.infolist()
        )


def test_timeout_cannot_expand_registered_limit():
    spec = importlib.util.spec_from_file_location(
        "binary64_cli", ROOT / "examples/reproduce_binary64_alignment.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    args = entry.parse_args(["--timeout-seconds", "601", "--verify-inputs-only"])
    with pytest.raises(ValueError, match="600"):
        entry.supervise(args)
