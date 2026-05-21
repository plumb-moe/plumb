"""Expert weight reordering for MoE models in safetensors format.

Computes and applies a permutation over expert slots so that expert weights
are laid out in the order expected by a given GPU placement.

Usage::

    from plumb.tools.reorder_experts import compute_reorder_map, reorder_safetensors
    from plumb.analysis.placement import recommend_placement

    rec = recommend_placement(counter, topology, num_gpus=4)
    rmap = compute_reorder_map(rec.expert_placement,
                               layers=[0, 1, ..., 47],
                               num_experts_per_layer=128,
                               num_gpus=4)
    reorder_safetensors("Qwen/Qwen3-30B-A3B", rmap, "/tmp/model_optimal")
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Expert weight keys: "...layers.N.{block_sparse_moe|mlp}.experts.EXPERT_ID.SUFFIX"
_EXPERT_KEY_RE = re.compile(
    r"^(.*layers\.(\d+)\.(?:block_sparse_moe|mlp)\.experts\.)(\d+)(\..*)?$"
)
# Gate / router weight keys: "...layers.N.{block_sparse_moe|mlp}.gate.weight"
_GATE_KEY_RE = re.compile(
    r"^.*layers\.(\d+)\.(?:block_sparse_moe|mlp)\.gate\.weight$"
)


def _resolve_model_path(model: str) -> Path:
    """Return a local filesystem path for model weights.

    If *model* is already a local directory with safetensors files, return it
    as-is.  Otherwise resolve through the HuggingFace Hub cache (downloading if
    necessary).
    """
    p = Path(model)
    if p.is_dir() and any(p.glob("*.safetensors")):
        return p

    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(model, local_files_only=True)
    except Exception:
        from huggingface_hub import snapshot_download

        local = snapshot_download(model)

    return Path(local)


def compute_reorder_map(
    placement: dict[tuple[int, int], list[int]],
    layers: list[int],
    num_experts_per_layer: int,
    num_gpus: int,
) -> dict[int, list[int]]:
    """Compute a per-layer expert permutation from a placement dict.

    Returns ``{layer_id: perm}`` where ``perm[new_slot] = old_expert_id``.

    GPU *g* owns slots ``g * experts_per_gpu .. (g+1) * experts_per_gpu - 1``
    in the reordered weights.  Within each GPU's slice the original expert
    order is preserved.

    An empty *placement* (e.g. when imbalance is below threshold) returns an
    empty dict — callers should treat this as the identity permutation.
    """
    if not placement:
        return {}

    experts_per_gpu = max(1, num_experts_per_layer // num_gpus)
    result: dict[int, list[int]] = {}

    for layer_id in layers:
        gpu_experts: dict[int, list[int]] = {g: [] for g in range(num_gpus)}
        for expert_id in range(num_experts_per_layer):
            gpus = placement.get((layer_id, expert_id), [expert_id % num_gpus])
            gpu = gpus[0] if gpus else expert_id % num_gpus
            gpu_experts[gpu].append(expert_id)

        perm: list[int] = []
        for g in range(num_gpus):
            perm.extend(gpu_experts.get(g, [])[:experts_per_gpu])

        # Handle uneven assignments: fill remaining slots with unassigned experts
        assigned = set(perm)
        for eid in range(num_experts_per_layer):
            if eid not in assigned:
                perm.append(eid)

        result[layer_id] = perm[:num_experts_per_layer]

    return result


def reorder_safetensors(
    src_model: str,
    reorder_map: dict[int, list[int]],
    out_dir: str,
) -> None:
    """Write a permuted copy of the model weights to *out_dir*.

    For each layer in *reorder_map*:
    - Expert weight tensors are written under their new slot index.
    - Gate (router) weight rows are permuted to match the new slot order.
    Shards that contain no expert or gate tensors are hard-linked to save disk.

    *src_model* may be a local path or a HuggingFace model ID.
    """
    src = _resolve_model_path(src_model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load shard index
    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text())
        weight_map: dict[str, str] = index_data["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = ["model.safetensors"]
        weight_map = {}
        with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = "model.safetensors"

    # Build inverse perm: old_expert_id → new_slot, per layer
    inv_perm: dict[int, dict[int, int]] = {}
    for layer_id, perm in reorder_map.items():
        inv_perm[layer_id] = {old: new_slot for new_slot, old in enumerate(perm)}

    # Find which shards contain expert/gate tensors (need rewriting)
    expert_shards: set[str] = set()
    for key, shard in weight_map.items():
        if _EXPERT_KEY_RE.match(key) or _GATE_KEY_RE.match(key):
            expert_shards.add(shard)

    new_weight_map: dict[str, str] = {}

    for shard_file in shard_files:
        shard_src = src / shard_file
        shard_dst = out / shard_file

        if shard_file not in expert_shards or not reorder_map:
            # No expert tensors in this shard — hard-link instead of copying
            try:
                os.link(str(shard_src), str(shard_dst))
            except OSError:
                shutil.copy2(shard_src, shard_dst)
            # Keys from this shard carry over unchanged
            for key, sf in weight_map.items():
                if sf == shard_file:
                    new_weight_map[key] = shard_file
            continue

        # Load, permute expert/gate tensors, write new shard
        tensors: dict[str, torch.Tensor] = {}

        with safe_open(str(shard_src), framework="pt", device="cpu") as f:
            for key in f.keys():
                orig = f.get_tensor(key)

                gm = _GATE_KEY_RE.match(key)
                if gm:
                    layer_id = int(gm.group(1))
                    perm = reorder_map.get(layer_id)
                    if perm is not None:
                        perm_t = torch.tensor(perm, dtype=torch.long)
                        tensors[key] = orig[perm_t]
                    else:
                        tensors[key] = orig
                    new_weight_map[key] = shard_file
                    continue

                em = _EXPERT_KEY_RE.match(key)
                if em:
                    prefix = em.group(1)       # "...layers.N.mlp.experts."
                    layer_id = int(em.group(2))
                    old_eid = int(em.group(3))
                    suffix = em.group(4) or ""  # ".gate_proj.weight"

                    layer_inv = inv_perm.get(layer_id)
                    if layer_inv is not None and old_eid in layer_inv:
                        new_slot = layer_inv[old_eid]
                        new_key = f"{prefix}{new_slot}{suffix}"
                    else:
                        new_key = key

                    tensors[new_key] = orig
                    new_weight_map[new_key] = shard_file
                    continue

                tensors[key] = orig
                new_weight_map[key] = shard_file

        save_file(tensors, str(shard_dst))

    # Write updated index
    if index_path.exists():
        new_index = {
            "metadata": index_data.get("metadata", {}),
            "weight_map": new_weight_map,
        }
        (out / "model.safetensors.index.json").write_text(
            json.dumps(new_index, indent=2)
        )

    # Copy config, tokenizer, and other non-weight files
    for f in src.iterdir():
        if f.name in set(shard_files) or f.name == "model.safetensors.index.json":
            continue
        dest = out / f.name
        if not dest.exists():
            if f.is_file():
                shutil.copy2(f, dest)
            elif f.is_dir():
                shutil.copytree(str(f), str(dest))
