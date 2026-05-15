import torch
import torch.nn as nn
from unittest.mock import patch

from plumb.counter import ActivationCounter
from plumb.hook import ProfilingHooks, _layer_id_from_path, detect_transformers_version


class MixtralSparseMoeBlock(nn.Module):
    """Minimal stand-in for MixtralSparseMoeBlock — name must match exactly for hook detection."""

    def __init__(self, num_experts: int = 8) -> None:
        super().__init__()
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor):
        router_logits = torch.randn(x.size(0), self.num_experts)
        return x, router_logits


class TinyMoEModel(nn.Module):
    def __init__(self, num_layers: int = 4, num_experts: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MixtralSparseMoeBlock(num_experts) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x, _ = layer(x)
        return x


def test_attach_detects_layers():
    model = TinyMoEModel(num_layers=4)
    counter = ActivationCounter()
    hooks = ProfilingHooks(counter)
    n = hooks.attach(model, top_k=2)
    assert n == 4
    hooks.detach()


def test_hook_records_activations():
    model = TinyMoEModel(num_layers=2, num_experts=8)
    counter = ActivationCounter()
    hooks = ProfilingHooks(counter)
    hooks.attach(model, top_k=2)

    x = torch.randn(4, 16)  # 4 tokens
    with torch.no_grad():
        model(x)

    hooks.detach()

    snap = counter.snapshot()
    # 2 layers × 4 tokens × top-k=2 = 16 total activations spread across experts
    total = sum(snap.values())
    assert total == 16, f"expected 16 activations, got {total}"


def test_pass_count_incremented():
    model = TinyMoEModel(num_layers=2)
    counter = ActivationCounter()
    hooks = ProfilingHooks(counter)
    hooks.attach(model, top_k=2)

    x = torch.randn(3, 16)
    with torch.no_grad():
        model(x)
        model(x)

    hooks.detach()
    # is_last fires on the final layer only → one increment per full forward pass
    assert counter.pass_count == 2


# ---------------------------------------------------------------------------
# vLLM gate hook tests
# ---------------------------------------------------------------------------

class _FakeReplicatedLinear(nn.Module):
    """Mimics vLLM's ReplicatedLinear: forward returns (output, bias)."""

    def __init__(self, num_experts: int = 8) -> None:
        super().__init__()
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor):
        logits = torch.randn(x.size(0), self.num_experts)
        bias = torch.zeros(self.num_experts)
        return logits, bias


class MixtralMoE(nn.Module):
    """Mimics vLLM's MixtralMoE block: has a .gate (ReplicatedLinear), returns single Tensor."""

    def __init__(self, num_experts: int = 8) -> None:
        super().__init__()
        self.gate = _FakeReplicatedLinear(num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _logits, _ = self.gate(x)
        return x


class TinyVllmModel(nn.Module):
    def __init__(self, num_layers: int = 4, num_experts: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({"mlp": MixtralMoE(num_experts)})
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer["mlp"](x)
        return x


def test_vllm_attach_detects_layers():
    model = TinyVllmModel(num_layers=4)
    counter = ActivationCounter()
    hooks = ProfilingHooks(counter)
    n = hooks.attach(model, top_k=2)
    assert n == 4, f"expected 4 vLLM layers, got {n}"
    hooks.detach()


def test_vllm_hook_records_activations():
    model = TinyVllmModel(num_layers=2, num_experts=8)
    counter = ActivationCounter()
    hooks = ProfilingHooks(counter)
    hooks.attach(model, top_k=2)

    x = torch.randn(4, 16)
    with torch.no_grad():
        model(x)

    hooks.detach()
    snap = counter.snapshot()
    total = sum(snap.values())
    # 2 layers × 4 tokens × top-k=2 = 16
    assert total == 16, f"expected 16 vLLM activations, got {total}"


def test_vllm_layer_ids_from_path():
    assert _layer_id_from_path("model.layers.7.mlp") == 7
    assert _layer_id_from_path("model.layers.0.mlp") == 0
    assert _layer_id_from_path("transformer.h.3.mlp") == 3


# ---------------------------------------------------------------------------
# transformers v5 BatchLinear compat tests
# ---------------------------------------------------------------------------

class BatchLinear(nn.Module):
    """Simulates transformers v5 fused expert weights — all experts in one tensor."""

    def __init__(self, num_experts: int = 8, hidden_size: int = 16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # output shape irrelevant for hook tests


class V5MoeBlock(nn.Module):
    """Simulates a transformers v5 MoE block: BatchLinear for experts, nn.Linear for routing.

    Unlike v4, forward() returns only hidden states — router_logits are NOT in the output tuple.
    Uses `moe_router` (not in _GATE_LEAF_NAMES) to isolate the v5 compat code path in tests.
    """

    def __init__(self, hidden_size: int = 16, num_experts: int = 8) -> None:
        super().__init__()
        self.moe_router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = BatchLinear(num_experts, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.moe_router(x)         # logits computed but not returned
        return self.experts(x)     # v5 block returns only hidden states


class TinyV5Model(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_size: int = 16, num_experts: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            V5MoeBlock(hidden_size, num_experts) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def test_v5_attach_detects_layers():
    model = TinyV5Model(num_layers=4)
    counter = ActivationCounter()
    hooks = ProfilingHooks(counter)
    with patch("plumb.hook.detect_transformers_version", return_value=(5, 0)):
        n = hooks.attach(model, top_k=2)
    assert n == 4, f"expected 4 v5 layers, got {n}"
    hooks.detach()


def test_v5_hook_records_activations():
    model = TinyV5Model(num_layers=2, num_experts=8)
    counter = ActivationCounter()
    hooks = ProfilingHooks(counter)
    with patch("plumb.hook.detect_transformers_version", return_value=(5, 0)):
        hooks.attach(model, top_k=2)

    x = torch.randn(4, 16)
    with torch.no_grad():
        model(x)

    hooks.detach()
    snap = counter.snapshot()
    total = sum(snap.values())
    # 2 layers × 4 tokens × top-k=2 = 16
    assert total == 16, f"expected 16 v5 activations, got {total}"


def test_v5_does_not_interfere_with_v4_layout():
    """v4 blocks must still be hooked via _BLOCK_EXTRACTORS even if v5 compat runs."""
    model = TinyMoEModel(num_layers=3)
    counter = ActivationCounter()
    hooks = ProfilingHooks(counter)
    n = hooks.attach(model, top_k=2)
    assert n == 3


def test_detect_transformers_version_returns_tuple():
    major, minor = detect_transformers_version()
    assert isinstance(major, int)
    assert isinstance(minor, int)


def test_detect_transformers_version_unavailable():
    """When transformers is not installed, returns (0, 0) without raising."""
    from importlib.metadata import PackageNotFoundError
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError("transformers")):
        major, minor = detect_transformers_version()
    assert (major, minor) == (0, 0)


def test_v5_compat_skipped_when_transformers_is_v4():
    """v5 compat path must not run if transformers reports version 4.x."""
    model = TinyV5Model(num_layers=2)
    counter = ActivationCounter()
    hooks = ProfilingHooks(counter)
    # Pretend transformers 4.x is installed — v5 compat should return 0 layers for v5 blocks
    with patch("plumb.hook.detect_transformers_version", return_value=(4, 51)):
        n = hooks.attach(model, top_k=2)
    # v5 blocks have no v4 class name and no vLLM gate, so 0 layers expected
    assert n == 0


def test_context_manager():
    model = TinyMoEModel(num_layers=2)
    counter = ActivationCounter()
    hooks = ProfilingHooks(counter)
    with hooks:
        hooks.attach(model, top_k=2)
        with torch.no_grad():
            model(torch.randn(2, 8))
    # hooks detached — verify no new activations after exit
    before = sum(counter.snapshot().values())
    with torch.no_grad():
        model(torch.randn(2, 8))
    after = sum(counter.snapshot().values())
    assert before == after
