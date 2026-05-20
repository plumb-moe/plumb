"""
End-to-end smoke test for plumb — uses a tiny synthetic MoE model so it
runs on any GPU (or CPU-only) without downloading large checkpoints.

Tests the full pipeline:
  hook attach → inference → activation recording → imbalance → NUMA stats → detach
"""
import sys

sys.path = [p for p in sys.path if '/usr/lib/python3' not in p and '/usr/local/lib/python3' not in p]

import torch
import torch.nn as nn

from plumb.analysis.imbalance import compute_imbalance
from plumb.analysis.numa import compute_cross_numa
from plumb.counter import ActivationCounter
from plumb.hook import ProfilingHooks
from plumb.topology import Topology

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")


# ---------------------------------------------------------------------------
# Tiny synthetic MoE model (Mixtral-style block name so hooks recognise it)
# ---------------------------------------------------------------------------

class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, num_experts: int = 64) -> None:
        super().__init__()
        self.num_experts = num_experts
        # Simulate an imbalanced router: expert 0 is always hot
        self._call_count = 0

    def forward(self, x: torch.Tensor):
        # x may be (batch, seq, hidden) or (tokens, hidden) — count all tokens
        num_tokens = x.numel() // x.size(-1)
        logits = torch.randn(num_tokens, self.num_experts, device=x.device)
        # Make expert 0 strongly preferred to create measurable imbalance
        logits[:, 0] += 5.0
        self._call_count += 1
        return x, logits


class TinyOLMoE(nn.Module):
    def __init__(self, num_layers: int = 16, num_experts: int = 64, hidden: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(1000, hidden)
        self.layers = nn.ModuleList([
            MixtralSparseMoeBlock(num_experts) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden, 1000)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x, _ = layer(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Run the smoke test
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nBuilding TinyOLMoE (16 layers, 64 experts) on {device} ...")
model = TinyOLMoE(num_layers=16, num_experts=64).to(device)
model.eval()

counter = ActivationCounter(window_size=1000)
hooks = ProfilingHooks(counter)
n_layers = hooks.attach(model, top_k=8)
print(f"Hooked {n_layers} MoE layers")
assert n_layers == 16, f"expected 16 hooked layers, got {n_layers}"

# Run 20 forward passes with 4-token sequences
print("Running 20 forward passes ...")
for _ in range(20):
    tokens = torch.randint(0, 1000, (4,), device=device).unsqueeze(0)
    with torch.no_grad():
        model(tokens)

hooks.detach()

snap = counter.snapshot()
total_activations = sum(snap.values())
print(f"\nTotal activation records : {total_activations}")
print(f"Pass count               : {counter.pass_count}")
assert counter.pass_count == 20, f"expected 20 passes, got {counter.pass_count}"
# 20 passes × 4 tokens × 16 layers × top-8 = 10240
assert total_activations == 10240, f"expected 10240 activations, got {total_activations}"

# Imbalance
imbalance = compute_imbalance(counter)
imbalance.sort(key=lambda x: x.imbalance_ratio, reverse=True)
print("\nImbalance per layer (top 5 worst):")
for r in imbalance[:5]:
    print(f"  Layer {r.layer_id:2d}: ratio={r.imbalance_ratio:.3f}  max_expert={r.max_expert_id}")
# Expert 0 is hot — every layer should show significant imbalance
assert imbalance[0].imbalance_ratio > 3.0, \
    f"expected imbalance >3x, got {imbalance[0].imbalance_ratio:.2f}x"
assert imbalance[0].max_expert_id == 0, \
    f"expected max expert 0, got {imbalance[0].max_expert_id}"

# NUMA / placement
topology = Topology.flat(1)
placement = {(layer, expert): 0 for (layer, expert) in snap}
numa_stats = compute_cross_numa(placement, snap, topology, src_gpu=0)
print(f"\nCross-NUMA rate: {numa_stats[0].cross_numa_rate:.1%}  (flat single-GPU — expected 0%)")
assert numa_stats[0].cross_numa_rate == 0.0

print("\n✓ All smoke test assertions passed.")
