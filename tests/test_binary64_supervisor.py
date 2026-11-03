"""Mock-only terminal commit controls; no worker, resource query or fsync runs."""

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import pytest

from experiments.case12_binary64_v1 import runtime as rt


@pytest.fixture
def entry():
    path = Path(__file__).resolve().parents[1] / "examples/reproduce_binary64_alignment.py"
    spec = importlib.util.spec_from_file_location("binary64_commit_controls", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemorySlot:
    def __init__(self, phase=None, persistent=False):
        self.phase, self.persistent = phase, persistent
        self.failed = self.closed = False
        self.events, self.records = [], []
        self.payload = b""

    def fault(self, phase):
        if self.phase == phase and (not self.failed or self.persistent):
            self.failed = True
            raise OSError("injected " + phase)

    def write(self, record):
        self.events.append("write")
        if self.closed:
            raise ValueError("descriptor already closed")
        self.fault("write")
        self.records.append(deepcopy(record))
        payload = rt.encode(record)
        self.payload = payload + b" " * (rt.RESERVE_BYTES - len(payload))
        self.fault("fsync")  # Bytes already written when durability fails.
        return hashlib.sha256(self.payload).hexdigest()

    def close(self):
        self.events.append("close")
        self.closed = True
        self.fault("close")

    def deadline(self):
        self.events.append("deadline")
        if self.phase == "deadline":
            raise TimeoutError("late deadline")


def test_success_requires_write_close_deadline_and_matching_padded_digest(entry):
    slot, terminal = MemorySlot(), {"accepted": True, "state": "PASSED"}
    digest = entry.finalize_terminal(slot, terminal, slot.deadline)
    assert slot.events == ["write", "close", "deadline"]
    assert len(slot.payload) == 2 * 1024**2
    assert digest == hashlib.sha256(slot.payload).hexdigest()
    durable = rt.decode(slot.payload)
    assert durable["accepted"] is False
    assert durable["checks_passed"] is True
    assert durable["acceptance_requires_supervisor_commit"] is True
    assert durable["state"] == "CHECKS_PASSED_AWAITING_SUPERVISOR_EXIT"


@pytest.mark.parametrize("phase", ["write", "fsync", "close", "deadline"])
def test_late_failure_cannot_leave_a_standalone_pass_or_return_commit(entry, phase):
    slot, terminal = MemorySlot(phase), {"accepted": True, "state": "PASSED"}
    assert entry.finalize_terminal(slot, terminal, slot.deadline) is None
    assert terminal["accepted"] is False
    assert terminal["state"] == "NOT_QUALIFIED"
    assert terminal["terminal_available"] is False
    assert all(row["accepted"] is False and row["state"] != "PASSED"
               and row["acceptance_requires_supervisor_commit"] is True
               for row in slot.records)
    if phase in ("close", "deadline"):
        # A closed descriptor prevents the corrective write; the earlier bytes
        # must still require an absent supervisor commit, never qualify alone.
        assert rt.decode(slot.payload)["state"] == "CHECKS_PASSED_AWAITING_SUPERVISOR_EXIT"


def test_failed_best_effort_downgrade_still_cannot_return_a_commit(entry):
    slot, terminal = MemorySlot("fsync", persistent=True), {"accepted": True}
    assert entry.finalize_terminal(slot, terminal, slot.deadline) is None
    assert len(slot.records) == 2
    assert all(row["accepted"] is False for row in slot.records)
    assert rt.decode(slot.payload)["state"] == "NOT_QUALIFIED"


@pytest.mark.parametrize("accepted", [False, 1, "true", None])
def test_only_literal_true_checks_can_reach_commit(entry, accepted):
    slot, terminal = MemorySlot(), {"accepted": accepted}
    assert entry.finalize_terminal(slot, terminal, slot.deadline) is None
    assert slot.events == ["write", "close"]
    assert rt.decode(slot.payload)["checks_passed"] is False
    assert rt.decode(slot.payload)["state"] == "NOT_QUALIFIED"
