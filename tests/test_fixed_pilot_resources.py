"""TEST_ONLY stand-ins verify resource archival; no genuine CUDA execution."""
import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper4_kbvqa.data.manifest import VQASample


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts/run_fixed_candidate_pilot.py"
    spec = importlib.util.spec_from_file_location("resource_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CUDAStandIn:
    """TEST_ONLY known counter values; never a model/benchmark fixture."""
    def __init__(self, available=True, fail_reset=False, fail_counter=False, fail_sync=False):
        self.available = available
        self.fail_reset = fail_reset
        self.fail_counter = fail_counter
        self.fail_sync = fail_sync
        self.reset_calls = []

    def is_available(self): return self.available
    def device_count(self): return 2
    def get_device_name(self, index): return f"TEST_ONLY_DEVICE_{index}"
    def memory_allocated(self, index): return 100 + index
    def memory_reserved(self, index): return 200 + index
    def synchronize(self, index):
        if self.fail_sync: raise RuntimeError("TEST_ONLY synchronization failure")
    def reset_peak_memory_stats(self, index):
        if self.fail_reset: raise RuntimeError("TEST_ONLY reset failure")
        self.reset_calls.append(index)
    def max_memory_allocated(self, index):
        if self.fail_counter: raise RuntimeError("TEST_ONLY counter failure")
        return 1024 + index
    def max_memory_reserved(self, index): return 2048 + index


def test_actual_runtime_counter_api_used_and_each_session_appended(cli, tmp_path):
    stand_in = CUDAStandIn()
    session = cli.CUDASessionResources(tmp_path)
    session.begin(SimpleNamespace(cuda=stand_in))
    record = session.finish("COMPLETED_VALID", pilot_status="DEVELOPMENT_VALID")
    assert stand_in.reset_calls == [0, 1]
    assert record["session_wall_time_s"] >= 0
    assert record["devices"][0]["peak_allocated_bytes"] == 1024
    assert record["devices"][1]["peak_reserved_bytes"] == 2049
    assert record["devices"][0]["initial_allocated_bytes"] == 100
    assert record["multi_device_peaks_are_not_summed"] is True
    assert record["benchmark_result"] is record["calibration_implemented"] is False
    assert "run_identity_sha256" not in record
    other = cli.CUDASessionResources(tmp_path)
    other.begin(SimpleNamespace(cuda=stand_in))
    other.finish("COMPLETED_INVALID", pilot_status="DEVELOPMENT_INVALID")
    rows = [json.loads(x) for x in (tmp_path / "resource_sessions.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0] == record
    assert rows[0]["session_id"] != rows[1]["session_id"]
    assert rows[1]["status"] == "COMPLETED_INVALID"


def test_unavailable_runtime_does_not_invent_zero_vram_measurements(cli, tmp_path):
    session = cli.CUDASessionResources(tmp_path)
    session.begin(SimpleNamespace(cuda=CUDAStandIn(available=False)))
    row = session.finish("FAILED", error={"type": "RuntimeError", "message": "No CUDA"})
    assert row["cuda_available"] is False and row["devices"] == []
    assert row["counter_errors"] and row["error"]["message"] == "No CUDA"


@pytest.mark.parametrize("failure", ["reset", "counter", "sync"])
def test_measurement_failure_keeps_nulls_and_explicit_error(cli, tmp_path, failure):
    stand_in = CUDAStandIn(fail_reset=failure == "reset", fail_counter=failure == "counter", fail_sync=failure == "sync")
    session = cli.CUDASessionResources(tmp_path)
    session.begin(SimpleNamespace(cuda=stand_in))
    row = session.finish("FAILED", error={"type": "RuntimeError", "message": "TEST_ONLY execution error"})
    assert row["devices"][0]["peak_allocated_bytes"] is None
    assert row["devices"][0]["errors"]
    if failure == "counter":
        assert row["devices"][0]["peak_reserved_bytes"] == 2048
    else:
        assert row["devices"][0]["peak_reset_succeeded"] is False


def test_main_archives_import_failure_then_reraises_original(cli, tmp_path, monkeypatch):
    sample = VQASample("TEST_ONLY_QID", str(tmp_path / "image.png"), "TEST_ONLY question", metadata={"source_split": "train"})
    monkeypatch.setattr(cli, "load_pilot_inputs", lambda args: ([sample], {"TEST_ONLY": True}))
    original_import = builtins.__import__
    def missing_torch(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("TEST_ONLY missing torch")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing_torch)
    out = tmp_path / "output"
    with pytest.raises(ModuleNotFoundError, match="TEST_ONLY missing torch"):
        cli.main(["--pilot-dir", str(tmp_path / "prepared"), "--out-dir", str(out), "--max-samples", "1"])
    row = json.loads((out / "resource_sessions.jsonl").read_text())
    assert row["status"] == "FAILED" and row["cuda_available"] is False
    assert row["error"]["type"] == "ModuleNotFoundError" and row["devices"] == []
    assert row["session_wall_time_s"] >= 0
    assert not (out / "run_identity.json").exists()
