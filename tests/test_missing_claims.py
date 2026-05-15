"""Tests for features claimed in the plumb README / spec that lacked coverage."""
from __future__ import annotations

import json
import subprocess
import sys
import types
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_torch_stub() -> types.ModuleType:
    """Return a minimal torch stub sufficient to import plumb.hook."""
    torch_mod = types.ModuleType("torch")
    nn_mod = types.ModuleType("torch.nn")
    dynamo_mod = types.ModuleType("torch._dynamo")

    # torch.nn.Module + torch.nn.Linear stubs
    class _FakeModule:
        pass

    class _FakeLinear(_FakeModule):
        pass

    nn_mod.Module = _FakeModule
    nn_mod.Linear = _FakeLinear

    # torch._dynamo.disable is used as a decorator in hook.py
    dynamo_mod.disable = lambda f: f

    # Tensor stub with .ndim attribute
    class _FakeTensor:
        ndim = 2

    torch_mod.nn = nn_mod
    torch_mod._dynamo = dynamo_mod
    torch_mod.Tensor = _FakeTensor
    torch_mod.tensor = lambda *a, **kw: None
    torch_mod.topk = lambda *a, **kw: (None, None)

    return torch_mod


def _import_hook_with_stub():
    """Import (or reload) plumb.hook with a torch stub injected."""
    stub = _make_torch_stub()
    mods_to_inject = {
        "torch": stub,
        "torch.nn": stub.nn,
        "torch._dynamo": stub._dynamo,
    }
    # Only inject stubs for modules not already loaded (real torch absent in this env)
    for mod_name, mod in mods_to_inject.items():
        if mod_name not in sys.modules:
            sys.modules[mod_name] = mod

    # If plumb.hook is already cached, use it directly (may have been loaded with real torch)
    if "plumb.hook" in sys.modules:
        return sys.modules["plumb.hook"]

    import importlib
    return importlib.import_module("plumb.hook")


# ---------------------------------------------------------------------------
# test_pause_file_pauses_and_resumes
# ---------------------------------------------------------------------------

def test_pause_file_pauses_and_resumes(tmp_path, monkeypatch):
    """Creating PAUSE_FILE → recording disabled; removing it → recording re-enabled."""
    hook_mod = _import_hook_with_stub()

    pause_file = tmp_path / "plumb_paused"

    # Point the module at our temp path and reset the cache
    monkeypatch.setattr(hook_mod, "PAUSE_FILE", str(pause_file))
    monkeypatch.setattr(hook_mod, "_last_toggle_check", 0.0)

    # File does not exist → recording enabled
    assert hook_mod.recording_enabled() is True

    # Create the pause file → recording disabled
    pause_file.touch()
    monkeypatch.setattr(hook_mod, "_last_toggle_check", 0.0)
    assert hook_mod.recording_enabled() is False

    # Remove the pause file → recording re-enabled
    pause_file.unlink()
    monkeypatch.setattr(hook_mod, "_last_toggle_check", 0.0)
    assert hook_mod.recording_enabled() is True


# ---------------------------------------------------------------------------
# test_cli_help
# ---------------------------------------------------------------------------

def test_cli_help():
    """plumb CLI --help exits 0 and 'run' appears in stdout."""
    from click.testing import CliRunner
    from plumb.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0, f"exit_code={result.exit_code}, output={result.output}"
    assert "run" in result.output


# ---------------------------------------------------------------------------
# test_eplb_output_float32_shape
# ---------------------------------------------------------------------------

def test_eplb_output_float32_shape(tmp_path):
    """_write_eplb_output writes a float32 (num_layers, num_experts) .npy file."""
    from plumb.cli import _write_eplb_output
    from plumb.registry import SessionInfo

    # Build a fake snapshot JSON with 2 layers × 2 experts
    expert_loads = {"0:0": 5, "0:1": 3, "1:0": 4, "1:1": 2}
    snapshot_data = {
        "pid": 99999,
        "model_name": "FakeModel",
        "pass_count": 10,
        "updated_at": time.time(),
        "expert_loads": expert_loads,
        "expert_counts": expert_loads,
    }
    snap_path = tmp_path / "snap.json"
    snap_path.write_text(json.dumps(snapshot_data))

    session = SessionInfo(
        pid=99999,
        model_name="FakeModel",
        n_layers=2,
        socket_path="",
        started_at=time.time(),
        snapshot_path=str(snap_path),
    )

    out_npy = tmp_path / "out.npy"

    with patch("plumb.cli.list_sessions", return_value=[session]):
        _write_eplb_output(str(out_npy))

    assert out_npy.exists(), "Output .npy file was not created"
    arr = np.load(str(out_npy))
    assert arr.shape == (2, 2), f"Expected shape (2, 2), got {arr.shape}"
    assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"
    assert arr[0, 0] == pytest.approx(5.0)
    assert arr[0, 1] == pytest.approx(3.0)
    assert arr[1, 0] == pytest.approx(4.0)
    assert arr[1, 1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# test_low_imbalance_warning
# ---------------------------------------------------------------------------

def test_low_imbalance_warning():
    """Uniform expert load → recommend_placement returns method='none' with warning."""
    from plumb.analysis.placement import recommend_placement
    from plumb.counter import ActivationCounter
    from plumb.topology import Topology

    # All experts equal — peak imbalance = 1.0 (well below threshold of 3.0)
    c = ActivationCounter(window_size=100_000)
    for layer in range(2):
        for expert in range(8):
            c.record(layer, expert, 100)

    topology = Topology.flat(8)
    rec = recommend_placement(c, topology, num_gpus=8)

    assert rec is not None, "Should return a recommendation, not None"
    assert rec.method == "none", f"Expected method='none', got {rec.method!r}"
    assert rec.expert_placement == {}
    assert rec.warning != "", "Warning should be non-empty"
    assert "not recommended" in rec.warning.lower() or "imbalance" in rec.warning.lower()


# ---------------------------------------------------------------------------
# test_prometheus_port_env
# ---------------------------------------------------------------------------

def test_prometheus_port_env():
    """The 'run' CLI command exposes a --prometheus-port option."""
    from plumb.cli import run

    param_names = {p.name for p in run.params}
    assert "prometheus_port" in param_names, (
        f"--prometheus-port not found in run.params; found: {param_names}"
    )


# ---------------------------------------------------------------------------
# test_qwen_transformers_in_block_extractors
# ---------------------------------------------------------------------------

def test_qwen_transformers_in_block_extractors():
    """Qwen2/3 MoE blocks should appear in _BLOCK_EXTRACTORS for transformers path."""
    hook_mod = _import_hook_with_stub()
    _BLOCK_EXTRACTORS = hook_mod._BLOCK_EXTRACTORS

    assert "Qwen2MoeSparseMoeBlock" in _BLOCK_EXTRACTORS, (
        "Qwen2MoeSparseMoeBlock missing from _BLOCK_EXTRACTORS"
    )
    assert "Qwen3MoeSparseMoeBlock" in _BLOCK_EXTRACTORS, (
        "Qwen3MoeSparseMoeBlock missing from _BLOCK_EXTRACTORS"
    )

    # Verify the extractor returns index 1 for tuples (same pattern as Mixtral)
    extractor2 = _BLOCK_EXTRACTORS["Qwen2MoeSparseMoeBlock"]
    extractor3 = _BLOCK_EXTRACTORS["Qwen3MoeSparseMoeBlock"]

    fake_hidden = object()
    fake_logits = object()

    assert extractor2((fake_hidden, fake_logits)) is fake_logits
    assert extractor3((fake_hidden, fake_logits)) is fake_logits
    assert extractor2(fake_hidden) is None   # non-tuple → None
    assert extractor3(fake_hidden) is None


# ---------------------------------------------------------------------------
# test_deepseek_v3_in_vllm_blocks
# ---------------------------------------------------------------------------

def test_deepseek_v3_in_vllm_blocks():
    """DeepseekV3MoE should be in _VLLM_GATE_BLOCKS pointing to 'gate'."""
    hook_mod = _import_hook_with_stub()
    _VLLM_GATE_BLOCKS = hook_mod._VLLM_GATE_BLOCKS

    assert "DeepseekV3MoE" in _VLLM_GATE_BLOCKS, (
        "DeepseekV3MoE missing from _VLLM_GATE_BLOCKS"
    )
    assert _VLLM_GATE_BLOCKS["DeepseekV3MoE"] == "gate", (
        f"Expected 'gate', got {_VLLM_GATE_BLOCKS['DeepseekV3MoE']!r}"
    )
    # Also verify the V2 entry is still present (not accidentally removed)
    assert _VLLM_GATE_BLOCKS["DeepseekV2MoE"] == "gate"


# ---------------------------------------------------------------------------
# test_no_gpu_message
# ---------------------------------------------------------------------------

def test_no_gpu_message():
    """scan_gpu_processes() returns empty list gracefully when nvidia-smi is missing."""
    from plumb.scanner import scan_gpu_processes

    with patch("plumb.scanner.subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        result = scan_gpu_processes()

    assert result == [], f"Expected empty list, got {result}"


def test_no_gpu_nonzero_exit():
    """scan_gpu_processes() returns empty list when nvidia-smi exits non-zero."""
    from plumb.scanner import scan_gpu_processes

    with patch(
        "plumb.scanner.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
    ):
        result = scan_gpu_processes()

    assert result == []


# ---------------------------------------------------------------------------
# SIGTERM → EPLB write
# ---------------------------------------------------------------------------

def test_eplb_written_when_child_exits_via_sigterm(tmp_path):
    """_write_eplb_output is called even when the child process exits via SIGTERM.

    The CLI calls _write_eplb_output unconditionally after launch() returns,
    regardless of the child's exit code. This verifies that a SIGTERM-killed
    child (exit code -15 on Linux) does not suppress the weights.npy write.
    """
    import signal
    from click.testing import CliRunner
    from plumb.cli import run

    written: list[str] = []

    def fake_write(path: str) -> None:
        written.append(path)

    out = str(tmp_path / "weights.npy")

    with patch("plumb.launcher.launch", return_value=-signal.SIGTERM) as mock_launch, \
         patch("plumb.cli._write_eplb_output", side_effect=fake_write), \
         patch("sys.exit"):
        runner = CliRunner()
        runner.invoke(run, ["--eplb-output", out, "--", "vllm", "serve", "model"])

    mock_launch.assert_called_once()
    assert written == [out], (
        f"_write_eplb_output was not called after SIGTERM exit; written={written}"
    )


def test_launcher_forwards_sigterm_to_child(tmp_path):
    """SIGTERM received by the launcher process is forwarded to the child subprocess.

    Spawns a real helper script that calls launcher.launch() around a long-running
    child, then sends SIGTERM to the helper. Verifies that launch() returns promptly
    (child was terminated) rather than blocking until the 30-second sleep finishes.
    """
    import signal
    import textwrap

    script = tmp_path / "helper.py"
    script.write_text(textwrap.dedent(f"""\
        import sys
        sys.executable  # ensure real interpreter
        from plumb.launcher import launch
        rc = launch([sys.executable, "-c", "import time; time.sleep(30)"])
        print(f"rc={{rc}}", flush=True)
    """))

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    # Give child time to start, then send SIGTERM to the helper wrapper
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)

    try:
        stdout, _ = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise AssertionError(
            "launcher.launch() did not return within 5s after SIGTERM — "
            "child was not forwarded the signal"
        )

    # launch() returned and printed an rc line
    assert "rc=" in stdout, (
        f"Expected 'rc=<n>' in helper stdout, got: {stdout!r}"
    )


# ---------------------------------------------------------------------------
# test_plumb_run_autoattach_e2e
# ---------------------------------------------------------------------------

def test_plumb_run_autoattach_e2e(tmp_path):
    """plumb launcher injects autoattach hooks, records activations, writes snapshot JSON.

    Exercises the full `plumb run` autoattach path end-to-end:
      sitecustomize → background scanner → ProfilingHooks.attach → snapshot writer
    Uses SAI_PROFILER_SCAN_INTERVAL=1 so the scanner fires within a second.
    """
    import textwrap
    from plumb.launcher import launch

    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent("""\
        import time, torch, torch.nn as nn

        # MixtralSparseMoeBlock is recognised by _BLOCK_EXTRACTORS in hook.py
        class MixtralSparseMoeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 8
            def forward(self, x):
                num_tokens = x.numel() // x.size(-1)
                return x, torch.randn(num_tokens, self.num_experts)

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

        # Wait for the autoattach scanner to fire and hook the model.
        # SAI_PROFILER_SCAN_INTERVAL=5 → first scan at ~5s.  We also need torch
        # to be fully imported before the background thread tries `import torch.nn`,
        # so the 5s interval serves double duty (torch typically imports in 2–4s).
        time.sleep(7)

        # Run forward passes AFTER hooks are attached so pass_count > 0
        with torch.no_grad():
            for _ in range(5):
                model(torch.randint(0, 100, (2,)).unsqueeze(0))

        # Allow snapshot writer (fires every 2s) to flush at least once
        time.sleep(4)
    """))

    before = time.time()
    rc = launch(
        [sys.executable, str(child)],
        extra_env={"SAI_PROFILER_SCAN_INTERVAL": "5"},
    )

    assert rc == 0, f"child script exited with rc={rc}"

    from pathlib import Path
    registry_dir = Path("/tmp/plumb")
    new_snaps = [
        p for p in registry_dir.glob("*_snapshot.json")
        if p.stat().st_mtime >= before
    ]
    assert new_snaps, (
        f"No snapshot file written during test run (reference time={before:.1f}). "
        f"Contents of {registry_dir}: {list(registry_dir.glob('*'))}"
    )

    snap = json.loads(new_snaps[0].read_text())
    assert snap["pass_count"] > 0, (
        f"Expected pass_count > 0, got {snap['pass_count']}"
    )
    assert snap["expert_loads"], "Expected non-empty expert_loads in snapshot"
