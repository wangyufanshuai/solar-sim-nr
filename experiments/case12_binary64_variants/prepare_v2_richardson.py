"""Prepare the isolated v2 (Richardson slot gate) candidate from the startup-2-GiB variant.

Copies every bound file of the startup2g candidate, applies exact single-match source
patches to kernel.py and replay.py, adds richardson.py and its control test, and writes a
new manifest. The startup2g candidate and the original v1 registration are not modified.

Usage (from the repository root, after prepare_startup2g.py):
    python experiments/case12_binary64_variants/prepare_v2_richardson.py [dest-name] [startup2g-name]
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parents[1] / ".release-local"
SRC = LOCAL / (sys.argv[2] if len(sys.argv) > 2 else "binary64-startup2g-20261029")
DEST = LOCAL / (sys.argv[1] if len(sys.argv) > 1 else "binary64-v2-richardson-20261103")
NEW = HERE / "v2"
EXP = "experiments/case12_binary64_v1/"
MANIFEST = EXP + "manifest.json"
PARENT = "1e148ddf37b916da22428c665c366daebaff8496fc27875c1f6f33a9ae98d0e7"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def patch(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


KERNEL = [
    (
        "import time\nfrom types import SimpleNamespace\n",
        "import time\nfrom types import SimpleNamespace\n\nfrom .richardson import richardson_slots\n",
    ),
    (
        "    all_slots = bool(np.all(error <= limits))\n",
        "    all_slots_v1 = bool(np.all(error <= limits))\n"
        "    v2 = [\n"
        "        richardson_slots(coefficients[oi], reference, resolution[oi], weights[..., None])\n"
        "        for oi in range(2)\n"
        "    ]\n"
        "    all_slots = all(r[\"all_slots\"] for r in v2)\n",
    ),
    (
        "    return dict(\n        all_slots=all_slots,\n",
        "    return dict(\n        all_slots=all_slots,\n        all_slots_v1=all_slots_v1,\n"
        "        richardson_failed_slots=sum(r[\"failed_slots\"] for r in v2),\n"
        "        richardson_worst_ratio=max(r[\"worst_ratio\"] for r in v2),\n",
    ),
    (
        "    modal_pass, field_pass = True, True\n",
        "    modal_pass, field_pass = True, True\n    modal_v1, modal_buffer = True, {}\n",
    ),
    (
        "                modal_pass = modal_pass and bool(\n"
        "                    np.all(np.abs(error_mode) <= mode_limit)\n                )\n",
        "                modal_v1 = modal_v1 and bool(np.all(np.abs(error_mode) <= mode_limit))\n"
        "                modal_buffer[oi, ei, column] = (error_mode, mode_weights, nu_mode)\n",
    ),
    (
        "    arrays[\"rhs_errors\"] = arrays[\"rhs_coefficients\"] - reference[\"rhs\"]\n",
        "    modal_v2 = [\n"
        "        richardson_slots(\n"
        "            np.stack([modal_buffer[oi, ei, c][0] for ei in (0, 1)]),\n"
        "            0.0,\n"
        "            np.stack([modal_buffer[oi, ei, c][2] for ei in (0, 1)]),\n"
        "            modal_buffer[oi, 0, c][1],\n"
        "        )\n"
        "        for oi in range(2)\n"
        "        for c in range(58)\n"
        "    ]\n"
        "    modal_pass = all(r[\"all_slots\"] for r in modal_v2)\n"
        "    modal_summary = dict(\n"
        "        all_slots=modal_pass,\n"
        "        all_slots_v1=modal_v1,\n"
        "        richardson_failed_slots=sum(r[\"failed_slots\"] for r in modal_v2),\n"
        "        richardson_worst_ratio=max(r[\"worst_ratio\"] for r in modal_v2),\n"
        "    )\n"
        "    del modal_buffer\n"
        "    arrays[\"rhs_errors\"] = arrays[\"rhs_coefficients\"] - reference[\"rhs\"]\n",
    ),
    (
        "        physical43=constraint_acceptance,\n        negative_controls=negative_records,\n",
        "        physical43=constraint_acceptance,\n        modal=modal_summary,\n"
        "        acceptance_profile=\"v2-richardson-20261103\",\n"
        "        negative_controls=negative_records,\n",
    ),
]

REPLAY = [
    (
        "from types import SimpleNamespace\n",
        "from types import SimpleNamespace\n\nfrom .richardson import richardson_slots\n",
    ),
    (
        "    coefficient_results = []\n",
        "    coefficient_results = []\n    modal_nu, modal_w = {}, {}\n",
    ),
    (
        "                    arrays[\"modal_limits\"][oi, ei, slot] = (\n"
        "                        64.0 * epsilon * epsilon * modal_weight + 8.0 * nu\n"
        "                    )\n",
        "                    arrays[\"modal_limits\"][oi, ei, slot] = (\n"
        "                        64.0 * epsilon * epsilon * modal_weight + 8.0 * nu\n"
        "                    )\n"
        "                    modal_nu[oi, ei, slot] = nu\n"
        "                    modal_w[oi] = modal_weight\n",
    ),
    (
        "        acceptance = {\n            \"schedule\": len(calls) == started == 482,\n",
        "        richardson = {\n"
        "            label: [\n"
        "                richardson_slots(\n"
        "                    arrays[label + \"_coefficients\"][oi],\n"
        "                    ref,\n"
        "                    arrays[label + \"_resolution\"][oi],\n"
        "                    np.maximum(1.0, np.sum(np.abs(ref), axis=2))[..., None],\n"
        "                )\n"
        "                for oi in range(2)\n"
        "            ]\n"
        "            for label, ref in (\n"
        "                (\"rhs\", rhs_reference),\n"
        "                (\"constraint\", constraint_reference),\n"
        "            )\n"
        "        }\n"
        "        richardson[\"modal\"] = [\n"
        "            richardson_slots(\n"
        "                modal_error[oi],\n"
        "                0.0,\n"
        "                np.stack(\n"
        "                    [\n"
        "                        np.stack([modal_nu[oi, ei, s] for s in range(58)])\n"
        "                        for ei in (0, 1)\n"
        "                    ]\n"
        "                ),\n"
        "                modal_w[oi],\n"
        "            )\n"
        "            for oi in range(2)\n"
        "        ]\n"
        "        acceptance = {\n            \"schedule\": len(calls) == started == 482,\n",
    ),
    (
        "            \"coefficients\": all(row[\"accepted\"] for row in coefficient_results),\n",
        "            \"coefficients\": all(\n"
        "                row[\"resolved_nonzero\"] and row[\"relative_nonzero\"]\n"
        "                for row in coefficient_results\n"
        "            )\n"
        "            and all(\n"
        "                r[\"all_slots\"] for k in (\"rhs\", \"constraint\") for r in richardson[k]\n"
        "            ),\n",
    ),
    (
        "            \"modal\": bool(np.all(np.abs(modal_error) <= arrays[\"modal_limits\"])),\n",
        "            \"modal\": all(r[\"all_slots\"] for r in richardson[\"modal\"]),\n",
    ),
    (
        "                \"coefficient_gates\": coefficient_results,\n",
        "                \"coefficient_gates\": coefficient_results,\n"
        "                \"richardson\": richardson,\n"
        "                \"acceptance_profile\": \"v2-richardson-20261103\",\n",
    ),
]

KERNEL_TEST = [
    (
        "    measured[:, 0, 0, 0, 0] += 1e-5\n"
        "    measured[:, 1, 0, 0, 0] += 1e-7\n"
        "    shrinking = K.matrix_acceptance(measured, reference, nu)\n"
        "    assert shrinking[\"accepted\"] and shrinking[\"measurable_rows\"] == 2\n",
        "    measured[:, 0, 0, 0, 0] += 1e-5\n"
        "    measured[:, 1, 0, 0, 0] += 1e-7\n"
        "    # v2: a 100x shrink for a 4x amplitude step is not the symmetric-difference\n"
        "    # eps^2 + eps^4 form within the registered c4 bound; v1 accepted it.\n"
        "    off_form = K.matrix_acceptance(measured, reference, nu)\n"
        "    assert off_form[\"all_slots_v1\"] and not off_form[\"all_slots\"]\n"
        "    measured[:, 1, 0, 0, 0] = 1 + 1e-5 / 16\n"
        "    shrinking = K.matrix_acceptance(measured, reference, nu)\n"
        "    assert shrinking[\"accepted\"] and shrinking[\"measurable_rows\"] == 2\n",
    ),
]

parent_raw = (SRC / MANIFEST).read_bytes()
assert sha(parent_raw) == PARENT, "startup2g registration changed"
assert not DEST.exists(), "fresh candidate directory required"
manifest = json.loads(parent_raw)
before = {}
for row in manifest["files"]:
    data = (SRC / row["path"]).read_bytes()
    represented = data.replace(b"\r\n", b"\n") if row.get("representation") == "lf-text" else data
    assert len(represented) == row["bytes"] and sha(represented) == row["sha256"], row["path"]
    before[row["path"]] = sha(data)
    target = DEST / row["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

changed = {}
for name, patches in (
    (EXP + "kernel.py", KERNEL),
    (EXP + "replay.py", REPLAY),
    ("tests/test_case12_binary64_kernel.py", KERNEL_TEST),
):
    text = (DEST / name).read_bytes().decode("utf-8")
    for old, new in patches:
        text = patch(text, old, new)
    changed[name] = text.encode("utf-8")
changed[EXP + "richardson.py"] = (NEW / "richardson.py").read_bytes()
changed["tests/test_case12_binary64_richardson.py"] = (
    NEW / "test_case12_binary64_richardson.py"
).read_bytes()
records = {row["path"]: row for row in manifest["files"]}
for name, data in changed.items():
    (DEST / name).write_bytes(data)
    if name in records:
        records[name].update(bytes=len(data), sha256=sha(data))
    else:
        manifest["files"].append(dict(path=name, bytes=len(data), sha256=sha(data)))
manifest["files"].sort(key=lambda r: r["path"])
manifest["registration_date"] = "2026-11-03"
manifest["acceptance_profile"] = {
    "id": "v2-richardson-20261103",
    "authorization": "user requested: register v2 then run",
    "parent_manifest_sha256": PARENT,
    "changed_source_paths": sorted(changed),
    "replaces": "per-amplitude 64*eps^2 slot bound in rhs, physical43 and modal checks",
    "rule": "|(16*C_L - C_H)/15 - ref| <= 256*eps_H^4*w + 8*(16*nu_L + nu_H)/15",
    "unchanged": "grid, modes, amplitudes, signs, resources, field/offmode, resolved, relative, contraction, negative-control gates",
    "registered_after_seeing_v1_run04": True,
    "original_v1_qualification": False,
}
new_raw = encode(manifest)
assert sum(r["bytes"] for r in manifest["files"]) + len(new_raw) <= 8 * 1024**2
(DEST / MANIFEST).write_bytes(new_raw)
(DEST / "outputs").mkdir()
(DEST / ".release-local").mkdir()
controls = ".release-local/run_bounded_controls.py"
(DEST / controls).write_bytes((SRC / controls).read_bytes())
(DEST / "parent-manifest.json").write_bytes(parent_raw)
assert (SRC / MANIFEST).read_bytes() == parent_raw
assert all(sha((SRC / n).read_bytes()) == d for n, d in before.items())
receipt = dict(
    schema="isolated-binary64-acceptance-variant-v2",
    created_utc=datetime.now(timezone.utc).isoformat(),
    candidate_root=str(DEST),
    parent_root=str(SRC),
    parent_manifest_sha256=PARENT,
    manifest_sha256=sha(new_raw),
    bound_files=len(manifest["files"]),
    changed_source_paths=sorted(changed),
    numerical_execution=False,
    preparation_script_sha256=sha(Path(__file__).read_bytes()),
)
(DEST / "preparation.json").write_bytes(encode(receipt))
print(json.dumps(receipt))
