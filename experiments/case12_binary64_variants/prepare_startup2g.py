"""Prepare a separately bound copy; never mutate the original registration.

Usage (from the repository root): python experiments/case12_binary64_variants/prepare_startup2g.py [dest-name]
Writes .release-local/<dest-name> (default binary64-startup2g-20261029).
"""
import hashlib
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = ROOT / ".release-local" / (sys.argv[1] if len(sys.argv) > 1 else "binary64-startup2g-20261029")
MANIFEST = "experiments/case12_binary64_v1/manifest.json"
RUNTIME = "experiments/case12_binary64_v1/runtime.py"
EXPECTED = "dba2b44c3e010b2a0f56514ae7af291b8b6c42d2b5613583272a02b69cd89e02"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


raw = (ROOT / MANIFEST).read_bytes()
assert sha(raw) == EXPECTED, "original frozen registration changed"
original = json.loads(raw)
assert not DEST.exists(), "fresh candidate directory required"
records = original["files"]
before = {}
for row in records:
    data = (ROOT / row["path"]).read_bytes()
    represented = data.replace(b"\r\n", b"\n") if row.get("representation") == "lf-text" else data
    assert len(represented) == row["bytes"] and sha(represented) == row["sha256"], row["path"]
    before[row["path"]] = sha(data)
    target = DEST / row["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

runtime = (DEST / RUNTIME).read_bytes()
assert runtime.count(b"MIN_FREE_RAM = 4 * 1024**3") == 1
assert runtime.count(b"need at least 4 GiB available RAM") == 1
updated = runtime.replace(b"MIN_FREE_RAM = 4 * 1024**3", b"MIN_FREE_RAM = 2 * 1024**3").replace(
    b"need at least 4 GiB available RAM", b"need at least 2 GiB available RAM")
(DEST / RUNTIME).write_bytes(updated)
manifest = json.loads(raw)
manifest["registration_date"] = "2026-10-29"
manifest["min_available_ram_bytes"] = 2 * 1024**3
manifest["resource_profile"] = {
    "id": "startup2g-runtime2g-20261029",
    "authorization": "user requested try with 2g available RAM",
    "parent_manifest_sha256": EXPECTED,
    "original_startup_ram_bytes": 4 * 1024**3,
    "changed_source_paths": [RUNTIME],
    "runtime_ram_floor_unchanged": True,
    "numerical_thresholds_and_scientific_sources_unchanged": True,
    "original_v1_qualification": False,
    "note": "Isolated resource variant. Original protocol-note startup floor applies to parent v1; this registration overrides startup only. No automatic retry."
}
for row in manifest["files"]:
    if row["path"] == RUNTIME:
        row.update(bytes=len(updated), sha256=sha(updated))
new_raw = encode(manifest)
assert sum(r["bytes"] for r in manifest["files"]) + len(new_raw) <= 8 * 1024**2
(DEST / MANIFEST).write_bytes(new_raw)
(DEST / "outputs").mkdir()
(DEST / ".release-local").mkdir()
controls = ".release-local/run_bounded_controls.py"
(DEST / controls).write_bytes((HERE / "run_bounded_controls.py").read_bytes())
(DEST / "parent-manifest.json").write_bytes(raw)
assert (ROOT / MANIFEST).read_bytes() == raw
assert all(sha((ROOT / name).read_bytes()) == digest for name, digest in before.items())
receipt = dict(
    schema="isolated-binary64-resource-variant-v1", created_utc=datetime.now(timezone.utc).isoformat(),
    candidate_root=str(DEST), parent_root=str(ROOT), parent_manifest_sha256=EXPECTED,
    manifest_sha256=sha(new_raw), parent_files_unchanged=True,
    bound_files=len(records), changed_source_paths=[RUNTIME],
    startup_ram_bytes=2 * 1024**3, runtime_ram_bytes=2 * 1024**3,
    worker_limit_bytes=1024**3, timeout_seconds=600, output_limit_bytes=64 * 1024**2,
    numerical_execution=False, original_v1_qualification=False,
    preparation_script_sha256=sha(Path(__file__).read_bytes()))
(DEST / "preparation.json").write_bytes(encode(receipt))
print(json.dumps(receipt))
