"""Memory observations must not retain a ctypes type per numerical RHS stage."""
import ctypes
from experiments.b3_native.long_diagnostics import peak_rss
from experiments.b3_native.memory_probe import memory, run
import pytest


def test_repeated_peak_queries_reuse_pointer_types():
    peak_rss()
    before=len(ctypes._pointer_type_cache)
    for _ in range(4096):peak_rss()
    assert len(ctypes._pointer_type_cache)==before


def test_peak_query_has_same_units_and_order_as_independent_observer():
    before=memory()["peak_bytes"]
    actual=peak_rss()
    after=memory()["peak_bytes"]
    assert isinstance(actual,int) and 0<before<=actual<=after


def test_probe_writes_fresh_bounded_result_and_does_not_evolve(tmp_path):
    result=run(tmp_path/"probe.json","monitor")
    assert result["status"]=="COMPLETED"
    assert result["scientific_evolution"] is False and result["input_unchanged"]
    samples=result["samples"]
    assert samples[-1]["pointer_type_cache_entries"]==samples[0]["pointer_type_cache_entries"]
    assert samples[-1]["calls"]==8192
    assert result["peak_bytes"]<=512*2**20
    with pytest.raises(FileExistsError):run(tmp_path/"probe.json","monitor")


def test_probe_rejects_unknown_workload_before_output(tmp_path):
    with pytest.raises(ValueError):run(tmp_path/"bad.json","unregistered")
    assert not (tmp_path/"bad.json").exists()
