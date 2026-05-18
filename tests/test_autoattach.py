"""Tests for the autoattach background session pipeline."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap
import time
import types

import pytest
from unittest.mock import patch


def _make_torch_stub() -> types.ModuleType:
    torch_mod = types.ModuleType("torch")
    nn_mod = types.ModuleType("torch.nn")
    dynamo_mod = types.ModuleType("torch._dynamo")

    class _FakeModule:
        pass

    class _FakeLinear(_FakeModule):
        pass

    nn_mod.Module = _FakeModule
    nn_mod.Linear = _FakeLinear
    dynamo_mod.disable = lambda f: f

    class _FakeTensor:
        ndim = 2

    torch_mod.nn = nn_mod
    torch_mod._dynamo = dynamo_mod
    torch_mod.Tensor = _FakeTensor
    torch_mod.tensor = lambda *a, **kw: None
    torch_mod.topk = lambda *a, **kw: (None, None)
    return torch_mod


def _real_torch_available() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import torch; torch.randn(1)"],
        capture_output=True,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# snapshot writer — no torch required
# ---------------------------------------------------------------------------

def test_snapshot_writer_flushes_correct_shape(tmp_path, monkeypatch):
    """Snapshot writer thread serialises the counter into the expected JSON shape."""
    stub = _make_torch_stub()
    monkeypatch.setitem(sys.modules, "torch", stub)
    monkeypatch.setitem(sys.modules, "torch.nn", stub.nn)
    monkeypatch.setitem(sys.modules, "torch._dynamo", stub._dynamo)

    if "plumb.autoattach" in sys.modules:
        importlib.reload(sys.modules["plumb.autoattach"])
    import plumb.autoattach as autoattach_mod

    registry_dir = tmp_path / "plumb"
    registry_dir.mkdir()
    monkeypatch.setattr("plumb.registry.REGISTRY_DIR", registry_dir)
    monkeypatch.setattr(autoattach_mod, "REGISTRY_DIR", registry_dir, raising=False)

    class _StubHooks:
        def __init__(self, counter):
            self.counter = counter

        def attach(self, model, top_k=2):
            for lid in range(4):
                for eid in range(8):
                    self.counter.record(lid, eid, token_count=10 + eid)
            for _ in range(3):
                self.counter.increment_pass()
            return 4

        def detach(self):
            pass

    import plumb.hook as hook_mod
    monkeypatch.setattr(hook_mod, "ProfilingHooks", _StubHooks, raising=False)

    import numa_topology
    monkeypatch.setattr(numa_topology, "_count_gpus", lambda: 0)

    class TinyMoEModel:
        pass

    started = autoattach_mod._attach_session(TinyMoEModel(), n_moe=4)
    assert started is True

    deadline = time.time() + 5.0
    snap_path = None
    while time.time() < deadline:
        snaps = list(registry_dir.glob("*_snapshot.json"))
        if snaps:
            data = json.loads(snaps[0].read_text())
            if data.get("pass_count", 0) > 0 and data.get("expert_loads"):
                snap_path = snaps[0]
                break
        time.sleep(0.2)

    assert snap_path is not None, (
        f"Snapshot writer never flushed valid data; registry: {list(registry_dir.iterdir())}"
    )

    snap = json.loads(snap_path.read_text())
    assert snap["pass_count"] >= 1
    assert snap["n_layers"] == 4
    assert snap["model_name"] == "TinyMoEModel"
    assert len(snap["expert_counts"]) == 32       # 4 layers × 8 experts
    assert snap["expert_counts"]["0:0"] == 10
    assert snap["expert_counts"]["0:7"] == 17
    assert len(snap["imbalance"]) == 4


# ---------------------------------------------------------------------------
# end-to-end with real torch (skipped in CI without GPU)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_pipeline_records_activations_and_writes_snapshot(tmp_path):
    """Full autoattach path: sitecustomize → hooks → snapshot. Requires real torch + GPU env."""
    if not _real_torch_available():
        pytest.skip("real torch not available — autoattach e2e requires it")

    from plumb.launcher import launch

    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent("""\
        import time, torch, torch.nn as nn

        class MixtralSparseMoeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 8
            def forward(self, x):
                return x, torch.randn(x.numel() // x.size(-1), self.num_experts)

        class TinyMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 16)
                self.layers = nn.ModuleList([MixtralSparseMoeBlock() for _ in range(4)])
                self.head = nn.Linear(16, 100)
            def forward(self, ids):
                x = self.embed(ids)
                for layer in self.layers:
                    x, _ = layer(x)
                return self.head(x)

        model = TinyMoE()
        model.eval()
        time.sleep(7)   # wait for scanner (SAI_PROFILER_SCAN_INTERVAL=5)

        with torch.no_grad():
            for _ in range(5):
                model(torch.randint(0, 100, (2,)).unsqueeze(0))

        time.sleep(4)   # allow snapshot writer to flush
    """))

    before = time.time()
    rc = launch([sys.executable, str(child)], extra_env={"SAI_PROFILER_SCAN_INTERVAL": "5"})
    assert rc == 0, f"child exited with rc={rc}"

    from pathlib import Path
    new_snaps = [
        p for p in Path("/tmp/plumb").glob("*_snapshot.json")
        if p.stat().st_mtime >= before
    ]
    assert new_snaps, "No snapshot written during test run"

    snap = json.loads(new_snaps[0].read_text())
    assert snap["pass_count"] > 0
    assert snap["expert_loads"], "expert_loads must be non-empty"
