from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any, Callable

import torch
import torch.nn as nn

from .counter import ActivationCounter

logger = logging.getLogger(__name__)

# ── Async drain queue ─────────────────────────────────────────────────────────
# Hooks push (counter, layer_idx, logits, pass_flag, top_k) here — raw GPU
# tensor, no topk, no sync on the hot path.
# The drain thread does topk() + tolist() + record() off the critical path.
#
# queue.Queue (not SimpleQueue) is used so that task_done()/join() give us a
# reliable flush on detach() — important for tests and clean shutdown.
_drain_queue: queue.Queue = queue.Queue()
_drain_started = False
_drain_lock = threading.Lock()


def _ensure_drain_thread() -> None:
    global _drain_started
    if _drain_started:
        return
    with _drain_lock:
        if _drain_started:
            return
        t = threading.Thread(target=_drain_worker, name="sai-hook-drain", daemon=True)
        t.start()
        _drain_started = True


def _drain_worker() -> None:
    while True:
        item = _drain_queue.get()      # blocking get — no spin
        try:
            _process_item(item)
        finally:
            _drain_queue.task_done()


def _flush_drain(timeout: float = 10.0) -> None:
    """Block until every queued item has been fully processed."""
    _drain_queue.join()


def _process_item(item: tuple) -> None:
    counter, layer_idx, logits, pass_flag, top_k = item
    try:
        k = min(top_k, logits.size(-1))
        _, selected = torch.topk(logits, k, dim=-1)  # topk off the hot path
        for eid in selected.view(-1).tolist():        # CPU-GPU sync off the hot path
            counter.record(layer_idx, int(eid))
        if pass_flag:
            counter.increment_pass()
    except Exception:
        pass

# ── Recording toggle ─────────────────────────────────────────────────────────
# External processes can pause/resume recording by creating/removing this file.
# Used by the hook-toggle benchmark to measure overhead on a live server.
PAUSE_FILE: str = os.environ.get("SAI_PROFILER_PAUSE_FILE", "/tmp/plumb_paused")
_recording_enabled: bool = True
_last_toggle_check: float = 0.0
_TOGGLE_CHECK_INTERVAL_S: float = 0.25   # re-stat the file at most 4×/sec


def recording_enabled() -> bool:
    """Return whether hooks should record. Re-checks the pause file periodically."""
    global _recording_enabled, _last_toggle_check
    now = time.monotonic()
    if now - _last_toggle_check >= _TOGGLE_CHECK_INTERVAL_S:
        _last_toggle_check = now
        _recording_enabled = not os.path.exists(PAUSE_FILE)
    return _recording_enabled


def detect_transformers_version() -> tuple[int, int]:
    """Return (major, minor) of the installed transformers package, or (0, 0) if unavailable."""
    try:
        from importlib.metadata import version as pkg_version
        parts = pkg_version("transformers").split(".")
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        return 0, 0

# MoE block/router classes -> callable that extracts router_logits.
# router_logits shape must be (num_tokens, num_experts).
#
# Block-level (return (hidden, logits, ...)): logits at index 1.
# Router-level (return (logits, scores, ...)): logits at index 0.
# DFS traversal of named_modules() assigns correct layer_idx for both patterns.
_BLOCK_EXTRACTORS: dict[str, Callable[[Any], torch.Tensor | None]] = {
    # transformers block-level — return (hidden, router_logits)
    "MixtralSparseMoeBlock":    lambda out: out[1] if isinstance(out, tuple) else None,
    "PhimoeSparseMoeBlock":     lambda out: out[1] if isinstance(out, tuple) else None,
    # Llama 4 Scout / Maverick (transformers 4.57+) — returns (hidden, router_logits)
    # Verified against transformers 4.57.6: Llama4TextMoe.forward returns (out, router_logits)
    # router_logits shape: (num_tokens, num_experts); num_experts=16 Scout, 128 Maverick
    "Llama4TextMoe":            lambda out: out[1] if isinstance(out, tuple) else None,
    # transformers 5.x router-level — OlmoeTopKRouter returns (logits, scores, indices)
    # OlmoeSparseMoeBlock is NOT listed for 5.x: it returns only hidden states there.
    # In transformers 4.x, OlmoeSparseMoeBlock returns (hidden, router_logits) — added below.
    "OlmoeTopKRouter":          lambda out: out[0] if isinstance(out, tuple) else None,
    # Phi-3.5-MoE router-level — PhimoeTopKRouter returns (logits, scores, indices).
    # PhimoeSparseMoeBlock is NOT listed: forward() returns only hidden states (logits discarded).
    "PhimoeTopKRouter":         lambda out: out[0] if isinstance(out, tuple) else None,
    # Gemma 4 — Gemma4Router is a sibling module on Gemma4DecoderLayer, not a gate attr of Gemma4MoE.
    # forward() returns a plain logit tensor [T, E]; Gemma4MoE.forward() takes logits as input.
    "Gemma4Router":             lambda out: out if isinstance(out, torch.Tensor) and out.ndim == 2 else None,
    # NOTE: In vLLM, Qwen2/3MoeSparseMoeBlock returns a plain tensor (not a tuple)
    # so these extractors would return None in that context. In transformers they
    # return (hidden_states, router_logits) like Mixtral, so the extractor works.
    # They're also listed in _VLLM_GATE_BLOCKS for the vLLM path.
    "Qwen2MoeSparseMoeBlock":  lambda out: out[1] if isinstance(out, tuple) else None,
    "Qwen3MoeSparseMoeBlock":  lambda out: out[1] if isinstance(out, tuple) else None,
}

# In transformers 4.x, OlmoeSparseMoeBlock.forward() returns (hidden, router_logits).
# In transformers 5.x it returns only hidden states — OlmoeTopKRouter handles that case.
_tx_major, _ = detect_transformers_version()
if _tx_major < 5:
    _BLOCK_EXTRACTORS["OlmoeSparseMoeBlock"] = lambda out: out[1] if isinstance(out, tuple) else None

# Router-level fallback dict (kept for future models with the same pattern).
_ROUTER_EXTRACTORS: dict[str, Callable[[Any], torch.Tensor | None]] = {}

# vLLM MoE block class names → gate attribute name.
# These blocks return a single Tensor (not a tuple), so we hook self.gate instead.
# gate is vLLM's ReplicatedLinear, which returns (output, bias); index 0 is router logits.
_VLLM_GATE_BLOCKS: dict[str, str] = {
    "MixtralMoE":              "gate",
    "DeepseekV2MoE":           "gate",
    "DeepseekV3MoE":           "gate",   # vLLM DeepSeek-V3
    "OlmoeMoE":                "gate",
    "PhiMoE":                  "gate",
    # Llama 4 Scout / Maverick — vLLM uses self.router (ReplicatedLinear), not self.gate
    "Llama4MoE":               "router",
    "Qwen2MoeSparseMoeBlock":  "gate",   # vLLM Qwen1.5/2-MoE
    "Qwen3MoeSparseMoeBlock":  "gate",   # vLLM Qwen3-MoE
}

# Fallback: hook inner gate/router linear layers by name.
_GATE_LEAF_NAMES = frozenset({"gate", "router", "router_weights", "w_gate", "router_linear"})

# transformers ≥5.0: expert weights are fused into BatchLinear, removing per-expert submodules.
# The router remains a separate nn.Linear inside the block. We detect v5 blocks by the presence
# of a BatchLinear child, then hook the router linear to capture logits.
_V5_ROUTER_ATTR_NAMES = ("router", "gate", "router_linear", "expert_gate", "moe_router")


def _layer_id_from_path(name: str) -> int | None:
    """Extract layer index from paths like 'model.layers.7.mlp.router' → 7."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part in ("layers", "h", "blocks", "layer") and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


class ProfilingHooks:
    """Attach/detach forward hooks on all MoE layers of a model."""

    def __init__(self, counter: ActivationCounter) -> None:
        self._counter = counter
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    def attach(self, model: nn.Module, top_k: int = 2) -> int:
        """Attach hooks. Returns number of MoE layers found."""
        _ensure_drain_thread()
        self._handles.clear()
        layer_idx = 0
        _is_last_refs: list[list[bool]] = []

        for name, module in model.named_modules():
            mname = type(module).__name__
            # Skip modules that appear in _VLLM_GATE_BLOCKS: their _BLOCK_EXTRACTORS
            # extractor returns None in vLLM (plain-tensor forward), and the gate
            # sub-module path in _attach_vllm_gates works for both vLLM and transformers.
            if mname in _VLLM_GATE_BLOCKS:
                continue
            extractor = _BLOCK_EXTRACTORS.get(mname)
            if extractor is None:
                continue
            hook, is_last = self._block_hook(layer_idx, extractor, top_k)
            handle = module.register_forward_hook(hook)
            _is_last_refs.append(is_last)
            self._handles.append(handle)
            logger.debug("hook → %s (layer %d)", name, layer_idx)
            layer_idx += 1

        if layer_idx == 0:
            layer_idx, _is_last_refs = self._attach_router_extractors(model, top_k)

        # Always try vLLM gate hooks — not gated on layer_idx == 0.
        # Modules in _VLLM_GATE_BLOCKS were skipped above, so no double-counting.
        # Gate extractor (out[0] or out) handles both ReplicatedLinear (vLLM) and
        # nn.Linear (transformers) gate/router sub-modules correctly.
        _vllm_n, _vllm_is_last = self._attach_vllm_gates(model, top_k)
        if _vllm_n > 0:
            layer_idx = _vllm_n
            _is_last_refs = _vllm_is_last

        if layer_idx == 0:
            layer_idx = self._attach_gate_fallback(model, top_k)
            _is_last_refs = []

        if layer_idx == 0:
            layer_idx, _is_last_refs = self._attach_v5_compat(model, top_k)

        if _is_last_refs:
            _is_last_refs[-1][0] = True  # mark last layer to trigger pass count

        if layer_idx == 0:
            logger.warning(
                "No MoE layers found. Known classes: %s",
                list(_BLOCK_EXTRACTORS) + list(_ROUTER_EXTRACTORS),
            )
        else:
            logger.info("Attached to %d MoE layers", layer_idx)
        return layer_idx

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        _flush_drain()   # wait for all queued logits to be processed before returning
        logger.info("Hooks removed")

    def _block_hook(self, layer_idx: int, extractor: Callable, top_k: int):
        counter = self._counter
        is_last = [False]   # set to True on the final layer hook to trigger pass count

        @torch._dynamo.disable
        def hook(_module: nn.Module, _inp: Any, output: Any) -> None:
            if not recording_enabled():
                return
            try:
                logits = extractor(output)
                if logits is None or logits.ndim != 2:
                    return
                # Move to CPU on the hook thread before queuing. Gate tensors are
                # small (num_tokens × num_experts), so the sync is cheap. Keeping raw
                # GPU tensors across threads races with vLLM's CUDA stream management
                # and triggers device-side asserts on subsequent attention kernels.
                _drain_queue.put_nowait((counter, layer_idx, logits.detach().cpu(), is_last[0], top_k))
            except Exception:
                logger.debug("Hook error layer %d", layer_idx, exc_info=True)

        return hook, is_last

    def _attach_router_extractors(self, model: nn.Module, top_k: int) -> tuple[int, list]:
        """Hook router-level classes (e.g. OlmoeTopKRouter) using path-derived layer IDs."""
        seen: dict[int, bool] = {}
        is_last_refs: list[list[bool]] = []
        for name, module in model.named_modules():
            extractor = _ROUTER_EXTRACTORS.get(type(module).__name__)
            if extractor is None:
                continue
            lid = _layer_id_from_path(name)
            if lid is None or lid in seen:
                continue
            seen[lid] = True
            hook, is_last = self._block_hook(lid, extractor, top_k)
            handle = module.register_forward_hook(hook)
            is_last_refs.append(is_last)
            self._handles.append(handle)
            logger.debug("router hook → %s (layer %d)", name, lid)
        n = max(seen) + 1 if seen else 0
        return n, is_last_refs

    def _attach_vllm_gates(self, model: nn.Module, top_k: int) -> tuple[int, list]:
        """Hook vLLM MoE gate sub-layers (ReplicatedLinear → (output, bias))."""
        seen: dict[int, bool] = {}
        is_last_refs: list[list[bool]] = []
        # ReplicatedLinear returns (output, bias); router logits are at index 0.
        def extractor(out: Any) -> torch.Tensor | None:
            return out[0] if isinstance(out, tuple) else out
        for name, module in model.named_modules():
            gate_attr = _VLLM_GATE_BLOCKS.get(type(module).__name__)
            if gate_attr is None:
                continue
            gate = getattr(module, gate_attr, None)
            if gate is None:
                continue
            lid = _layer_id_from_path(name)
            if lid is None or lid in seen:
                continue
            seen[lid] = True
            hook, is_last = self._block_hook(lid, extractor, top_k)
            handle = gate.register_forward_hook(hook)
            is_last_refs.append(is_last)
            self._handles.append(handle)
            logger.debug("vllm gate hook → %s.%s (layer %d)", name, gate_attr, lid)
        n = max(seen) + 1 if seen else 0
        return n, is_last_refs

    def _attach_gate_fallback(self, model: nn.Module, top_k: int) -> int:
        layer_idx = 0
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name.split(".")[-1] not in _GATE_LEAF_NAMES:
                continue
            handle = module.register_forward_hook(self._gate_hook(layer_idx, top_k))
            self._handles.append(handle)
            logger.debug("gate fallback hook → %s (layer %d)", name, layer_idx)
            layer_idx += 1
        return layer_idx

    def _attach_v5_compat(self, model: nn.Module, top_k: int) -> tuple[int, list]:
        """transformers ≥5.0 compat: hook router linears inside BatchLinear-fused MoE blocks.

        In v5, per-expert submodules are replaced by a single BatchLinear; the routing linear
        remains as a separate attribute. We detect v5 blocks by the presence of a BatchLinear
        child, then attach to the router linear so we can extract top-k expert selections.
        """
        major, _ = detect_transformers_version()
        if major > 0 and major < 5:
            return 0, []

        seen: set[int] = set()
        is_last_refs: list[list[bool]] = []
        for name, module in model.named_modules():
            has_batch_linear = any(
                type(child).__name__ == "BatchLinear"
                for child in module.children()
            )
            if not has_batch_linear:
                continue

            for attr in _V5_ROUTER_ATTR_NAMES:
                router = getattr(module, attr, None)
                if not isinstance(router, nn.Linear):
                    continue
                lid = _layer_id_from_path(name)
                if lid is None or lid in seen:
                    continue
                seen.add(lid)
                def extractor(out: Any) -> torch.Tensor | None:  # noqa: E306
                    return out if isinstance(out, torch.Tensor) and out.ndim == 2 else None
                hook, is_last = self._block_hook(lid, extractor, top_k)
                handle = router.register_forward_hook(hook)
                is_last_refs.append(is_last)
                self._handles.append(handle)
                logger.debug("v5 BatchLinear router hook → %s.%s (layer %d)", name, attr, lid)
                break

        return max(seen) + 1 if seen else 0, is_last_refs

    def _gate_hook(self, layer_idx: int, top_k: int):
        counter = self._counter

        @torch._dynamo.disable
        def hook(_module: nn.Module, _inp: Any, output: torch.Tensor) -> None:
            if not recording_enabled():
                return
            try:
                if isinstance(output, tuple):
                    output = output[0]
                if output.ndim != 2:
                    return
                _drain_queue.put_nowait((counter, layer_idx, output.detach().cpu(), False, top_k))
            except Exception:
                logger.debug("Gate hook error layer %d", layer_idx, exc_info=True)

        return hook

    def __enter__(self) -> ProfilingHooks:
        return self

    def __exit__(self, *_: Any) -> None:
        self.detach()
