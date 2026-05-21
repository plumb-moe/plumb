#!/usr/bin/env python3
"""Resume 3-way benchmark from profiling snapshot.

Uses in-place consumption of source shards so disk usage stays at 1× model size.
Derives model_worst from model_optimal using a relative permutation, avoiding
a second read of the (consumed) HF cache.

Run as:
  python resume_bench.py --snapshot /tmp/plumb/19836_snapshot.json
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ── Config ───────────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen3-30B-A3B"
TP = 4
NUM_GPUS = 4
NUM_EXPERTS = 128
PORT = 8200
OUT_DIR = Path("/tmp/bench3way")
WARMUP_REQUESTS = 20
BENCHMARK_REQUESTS = 200
MAX_OUTPUT_TOKENS = 128
CONCURRENCY = 8

PROMPTS = [
    "Explain the attention mechanism in transformers, covering queries, keys, and values:",
    "What are the key differences between TCP and UDP protocols?",
    "How does gradient descent work in neural network training?",
    "Describe the process of protein synthesis from DNA transcription to translation:",
    "What is the CAP theorem in distributed systems and why does it matter?",
    "Explain how HTTPS encryption works step by step:",
    "What causes aurora borealis? Describe the physics involved:",
    "How do GPUs accelerate deep learning? Explain the parallelism model:",
    "Describe the architecture of a transformer-based language model:",
    "What is quantum entanglement and how is it used in quantum computing?",
    "Explain backpropagation through time (BPTT) in recurrent neural networks:",
    "How does the human immune system distinguish self from non-self?",
    "What is the difference between L1, L2, and elastic net regularization?",
    "Describe mixture-of-experts (MoE) architecture and its benefits over dense models:",
    "How does vLLM's PagedAttention work to improve inference throughput?",
    "Explain the transformer scaling laws and what they predict about model performance:",
    "What is speculative decoding and how does it speed up inference?",
    "Describe RLHF (reinforcement learning from human feedback) for LLM alignment:",
    "How does flash attention reduce memory usage in transformer inference?",
    "Explain the differences between BF16, FP16, and INT8 quantization for inference:",
    "What is tensor parallelism and how does it distribute computation across GPUs?",
    "Describe the key innovations in the Mixtral 8x7B architecture:",
    "How does top-k and top-p sampling affect language model output diversity?",
    "Explain KV cache and why it is critical for efficient autoregressive generation:",
    "What are the tradeoffs between model size, context length, and inference latency?",
    "Describe how expert routing works in sparse MoE models during inference:",
    "How does GPTQ quantization compress model weights while preserving quality?",
    "Explain pipeline parallelism and its latency tradeoffs versus tensor parallelism:",
    "What is activation checkpointing and when is it used during training?",
    "Describe the architectural differences between GPT and BERT style models:",
    "The capital of France is",
    "Water boils at 100 degrees Celsius because",
    "The speed of light in vacuum is approximately",
    "In machine learning, overfitting occurs when",
    "The French Revolution began in 1789 because",
    "DNA consists of four nucleotide bases:",
    "Neural networks learn by adjusting weights through",
    "The Eiffel Tower was constructed in",
    "Photosynthesis converts sunlight into energy by",
    "Quantum mechanics fundamentally differs from classical physics because",
]
_POOL = (PROMPTS * ((BENCHMARK_REQUESTS + WARMUP_REQUESTS) // len(PROMPTS) + 2))

# ── Reorder (with in-place source consumption) ────────────────────────────────

_EXPERT_KEY_RE = re.compile(
    r"^(.*layers\.(\d+)\.(?:block_sparse_moe|mlp)\.experts\.)(\d+)(\..*)?$"
)
_GATE_KEY_RE = re.compile(
    r"^.*layers\.(\d+)\.(?:block_sparse_moe|mlp)\.gate\.weight$"
)


def _resolve_hf_path(model_id: str) -> Path:
    from huggingface_hub import snapshot_download
    local = snapshot_download(model_id, local_files_only=True)
    return Path(local)


def reorder_safetensors_inplace(
    src: Path,
    reorder_map: dict[int, list[int]],
    out: Path,
    consume_source: bool = False,
) -> None:
    """Write a permuted copy of model weights.

    If consume_source=True, each source shard is deleted before writing the
    output shard, so disk usage stays constant (1 model copy at a time).
    """
    out.mkdir(parents=True, exist_ok=True)

    index_path = src / "model.safetensors.index.json"
    index_data = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index_data["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    inv_perm: dict[int, dict[int, int]] = {}
    for layer_id, perm in reorder_map.items():
        inv_perm[layer_id] = {old: new_slot for new_slot, old in enumerate(perm)}

    expert_shards: set[str] = set()
    for key, shard in weight_map.items():
        if _EXPERT_KEY_RE.match(key) or _GATE_KEY_RE.match(key):
            expert_shards.add(shard)

    new_weight_map: dict[str, str] = {}

    # Copy non-weight files first
    for f in src.iterdir():
        if f.name in set(shard_files) or f.name == "model.safetensors.index.json":
            continue
        dest = out / f.name
        if not dest.exists():
            if f.is_file():
                shutil.copy2(f, dest)
            elif f.is_dir():
                shutil.copytree(str(f), str(dest))

    for shard_file in shard_files:
        shard_src = src / shard_file
        shard_dst = out / shard_file

        if shard_file not in expert_shards or not reorder_map:
            if consume_source:
                # Load into memory, delete source blob, write output
                tensors = {}
                with safe_open(str(shard_src), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
                if shard_src.is_symlink():
                    real_blob = shard_src.resolve()
                    shard_src.unlink()   # remove HF snapshot symlink
                    real_blob.unlink()   # free actual blob
                else:
                    shard_src.unlink()   # regular file — one unlink is enough
                save_file(tensors, str(shard_dst))
            else:
                try:
                    os.link(str(shard_src), str(shard_dst))
                except OSError:
                    shutil.copy2(shard_src, shard_dst)
            for key, sf in weight_map.items():
                if sf == shard_file:
                    new_weight_map[key] = shard_file
            continue

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(shard_src), framework="pt", device="cpu") as f:
            for key in f.keys():
                orig = f.get_tensor(key)
                gm = _GATE_KEY_RE.match(key)
                if gm:
                    layer_id = int(gm.group(1))
                    perm = reorder_map.get(layer_id)
                    tensors[key] = orig[torch.tensor(perm, dtype=torch.long)] if perm else orig
                    new_weight_map[key] = shard_file
                    continue
                em = _EXPERT_KEY_RE.match(key)
                if em:
                    prefix = em.group(1)
                    layer_id = int(em.group(2))
                    old_eid = int(em.group(3))
                    suffix = em.group(4) or ""
                    layer_inv = inv_perm.get(layer_id)
                    if layer_inv is not None and old_eid in layer_inv:
                        new_key = f"{prefix}{layer_inv[old_eid]}{suffix}"
                    else:
                        new_key = key
                    tensors[new_key] = orig
                    new_weight_map[new_key] = shard_file
                    continue
                tensors[key] = orig
                new_weight_map[key] = shard_file

        if consume_source:
            if shard_src.is_symlink():
                real_blob = shard_src.resolve()
                shard_src.unlink()   # remove HF snapshot symlink
                real_blob.unlink()   # free actual blob
            else:
                shard_src.unlink()   # regular file — one unlink is enough
        print(f"    writing {shard_file}...", flush=True)
        save_file(tensors, str(shard_dst))
        print(f"    wrote {shard_dst.name}  ({shard_dst.stat().st_size / 1e9:.2f}GB)", flush=True)

    new_index = {"metadata": index_data.get("metadata", {}), "weight_map": new_weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2))


def compute_reorder_map(
    placement: dict[tuple[int, int], list[int]],
    layers: list[int],
    num_experts_per_layer: int,
    num_gpus: int,
) -> dict[int, list[int]]:
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
        assigned = set(perm)
        for eid in range(num_experts_per_layer):
            if eid not in assigned:
                perm.append(eid)
        result[layer_id] = perm[:num_experts_per_layer]
    return result


def compute_relative_reorder_map(
    rmap_base: dict[int, list[int]],
    rmap_target: dict[int, list[int]],
) -> dict[int, list[int]]:
    """Compute permutation that converts base-ordered model to target-ordered model."""
    result = {}
    for layer_id in rmap_base:
        base_perm = rmap_base[layer_id]       # base_perm[slot] = original_expert
        target_perm = rmap_target[layer_id]   # target_perm[slot] = original_expert
        # inv_base[original_expert] = slot_in_base
        inv_base = {e: slot for slot, e in enumerate(base_perm)}
        # relative[slot_in_target] = slot_in_base that has the same original expert
        relative = [inv_base[target_perm[slot]] for slot in range(len(target_perm))]
        result[layer_id] = relative
    return result


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def wait_for_server(base_url: str, timeout: int = 600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


class VllmServer:
    def __init__(self, model: str, tp: int, port: int, log_suffix: str = ""):
        self.model = model
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self._cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--tensor-parallel-size", str(tp),
            "--max-model-len", "4096",
            "--port", str(port),
            "--no-enable-log-requests",
        ]
        self._log_path = OUT_DIR / f"server_{log_suffix}.log"
        self._proc = None
        self._log = None

    def __enter__(self):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self._log_path, "w")
        print(f"  Starting vLLM: ... --model {Path(self.model).name} --port {self.port}")
        self._proc = subprocess.Popen(self._cmd, stdout=self._log, stderr=self._log)
        print(f"  PID {self._proc.pid} — waiting for health check...")
        if not wait_for_server(self.base_url):
            self._proc.terminate()
            raise RuntimeError("vLLM did not start")
        print(f"  Ready.")
        return self

    def __exit__(self, *_):
        if self._proc and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._log:
            self._log.close()
        time.sleep(5)


class GpuUtilPoller:
    def __init__(self):
        self._samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        if not self._samples:
            return []
        n_gpus = len(self._samples[0])
        return [round(statistics.mean(s[g] for s in self._samples), 1) for g in range(n_gpus)]

    def _run(self):
        while not self._stop.wait(2.0):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    text=True)
                self._samples.append([int(x.strip()) for x in out.strip().splitlines()])
            except Exception:
                pass


def _measure_request(base_url, model, prompt):
    t0 = time.perf_counter()
    t_first = None
    n_tokens = 0
    try:
        resp = requests.post(f"{base_url}/v1/completions",
                             json={"model": model, "prompt": prompt,
                                   "max_tokens": MAX_OUTPUT_TOKENS, "stream": True, "temperature": 0.0},
                             stream=True, timeout=120)
        for raw_line in resp.iter_lines():
            if not raw_line or raw_line == b"data: [DONE]":
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            if t_first is None:
                t_first = time.perf_counter()
            try:
                chunk = json.loads(payload)
                _ = chunk.get("choices", [{}])[0].get("text", "")
                n_tokens += 1
            except Exception:
                pass
    except Exception as e:
        print(f"    request error: {e}", file=sys.stderr)
        return None
    t_end = time.perf_counter()
    if t_first is None:
        return None
    ttft = t_first - t0
    total = t_end - t0
    tpot = (total - ttft) / max(n_tokens - 1, 1)
    return {"ttft_s": ttft, "total_latency_s": total, "output_tokens": n_tokens, "tpot_s": tpot}


def _pct(vals, p):
    s = sorted(vals)
    return s[min(int(len(s) * p), len(s) - 1)]


def run_scenario(base_url, model, label):
    prompts = _POOL[:BENCHMARK_REQUESTS + WARMUP_REQUESTS]
    print(f"  [{label}] Warmup ({WARMUP_REQUESTS} requests)...")
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [pool.submit(requests.post, f"{base_url}/v1/completions",
                            json={"model": model, "prompt": p, "max_tokens": 8, "temperature": 0.0},
                            timeout=60)
                for p in prompts[:WARMUP_REQUESTS]]
        for f in as_completed(futs):
            f.result()

    print(f"  [{label}] Benchmarking ({BENCHMARK_REQUESTS} requests, concurrency={CONCURRENCY})...")
    poller = GpuUtilPoller()
    poller.start()
    results_list = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = {pool.submit(_measure_request, base_url, model, p): i
                for i, p in enumerate(prompts[WARMUP_REQUESTS:])}
        for done_count, future in enumerate(as_completed(futs), 1):
            r = future.result()
            if r:
                results_list.append(r)
            if done_count % 50 == 0:
                elapsed = time.perf_counter() - t0
                tps = sum(rr["output_tokens"] for rr in results_list) / elapsed
                print(f"    {done_count}/{BENCHMARK_REQUESTS}  tok/s={tps:.1f}", flush=True)
    t_end = time.perf_counter()
    gpu_util = poller.stop()

    if not results_list:
        return {"label": label, "error": "no successful requests"}

    ttfts = [r["ttft_s"] for r in results_list]
    tpots = [r["tpot_s"] for r in results_list]
    total_tokens = sum(r["output_tokens"] for r in results_list)
    elapsed = t_end - t0
    return {
        "label": label,
        "n_requests": len(results_list),
        "total_output_tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "tokens_per_sec": round(total_tokens / elapsed, 2),
        "requests_per_sec": round(len(results_list) / elapsed, 3),
        "ttft_ms": {"p50": round(_pct(ttfts, 0.50) * 1000, 2),
                    "p95": round(_pct(ttfts, 0.95) * 1000, 2),
                    "p99": round(_pct(ttfts, 0.99) * 1000, 2),
                    "mean": round(statistics.mean(ttfts) * 1000, 2)},
        "tpot_ms": {"p50": round(_pct(tpots, 0.50) * 1000, 3),
                    "p95": round(_pct(tpots, 0.95) * 1000, 3),
                    "p99": round(_pct(tpots, 0.99) * 1000, 3),
                    "mean": round(statistics.mean(tpots) * 1000, 3)},
        "gpu_util_pct": gpu_util,
        "gpu_util_avg_pct": round(statistics.mean(gpu_util), 1) if gpu_util else None,
    }


def compute_imbalance(loads):
    if not loads:
        return {}
    layers = {}
    for (lid, eid), count in loads.items():
        layers.setdefault(lid, {})[eid] = count
    ratios = []
    for lid in sorted(layers):
        counts = list(layers[lid].values())
        mean = sum(counts) / len(counts) if counts else 1
        ratios.append(max(counts) / mean if mean > 0 else 1.0)
    return {
        "mean_imbalance_ratio": round(statistics.mean(ratios), 4),
        "max_imbalance_ratio": round(max(ratios), 4),
    }


def print_table(scenarios):
    names = [s["label"] for s in scenarios]
    col_w = max(12, max(len(n) for n in names))
    header = f"{'Metric':<32}" + "".join(f"{n:>{col_w}}" for n in names)
    sep = "-" * len(header)

    def row(label, key_path, fmt=".2f", suffix=""):
        vals = []
        for s in scenarios:
            v = s
            for k in key_path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            vals.append(f"{v:{fmt}}{suffix}" if v is not None else "n/a")
        return f"{label:<32}" + "".join(f"{v:>{col_w}}" for v in vals)

    print()
    print("=" * len(header))
    print(" 3-WAY EXPERT PLACEMENT BENCHMARK RESULTS")
    print("=" * len(header))
    print(header)
    print(sep)
    print(row("tokens/sec", ["tokens_per_sec"], ".1f"))
    print(row("requests/sec", ["requests_per_sec"], ".3f"))
    print(row("TTFT p50 (ms)", ["ttft_ms", "p50"], ".2f"))
    print(row("TTFT p95 (ms)", ["ttft_ms", "p95"], ".2f"))
    print(row("TPOT p50 (ms/tok)", ["tpot_ms", "p50"], ".3f"))
    print(row("GPU util avg (%)", ["gpu_util_avg_pct"], ".1f"))
    print(row("Imbalance (mean)", ["imbalance", "mean_imbalance_ratio"], ".4f"))
    print(row("Imbalance (max)", ["imbalance", "max_imbalance_ratio"], ".4f"))
    print(sep)
    if len(scenarios) > 1:
        base_tps = scenarios[0].get("tokens_per_sec")
        print("\n  Delta vs random:")
        for s in scenarios[1:]:
            tps = s.get("tokens_per_sec")
            if base_tps and tps:
                delta_pct = (tps - base_tps) / base_tps * 100
                sign = "+" if delta_pct >= 0 else ""
                print(f"    {s['label']:<20} {sign}{delta_pct:.1f}%  ({tps:.1f} vs {base_tps:.1f})")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="/tmp/plumb/19836_snapshot.json")
    args = parser.parse_args()

    # Load expert activation data from profiling snapshot
    snap = json.loads(Path(args.snapshot).read_text())
    expert_raw = snap.get("expert_counts", snap.get("expert_loads", {}))
    loads = {}
    for key, count in expert_raw.items():
        lid, eid = map(int, key.split(":"))
        loads[(lid, eid)] = int(count)
    print(f"Loaded {len(loads)} (layer, expert) activation counts")

    layers = sorted({lid for lid, _ in loads})
    print(f"{len(layers)} layers seen, {NUM_EXPERTS} experts/layer, {NUM_GPUS} GPUs")

    # Compute placements
    sys.path.insert(0, "/opt/plumb")
    from plumb.analysis.placement import recommend_placement, worst_case_placement
    from plumb.counter import ActivationCounter
    from numa_topology import Topology

    c = ActivationCounter(window_size=10_000_000)
    for (lid, eid), count in loads.items():
        c.record(lid, eid, count)
    topo = Topology.flat(NUM_GPUS)

    print("Computing optimal placement...")
    rec = recommend_placement(c, topo, num_gpus=NUM_GPUS)
    opt_placement = rec.expert_placement if rec else {}
    opt_method = rec.method if rec else "none"
    print(f"  method={opt_method}  estimated_improvement={rec.estimated_improvement_pct}%")

    print("Computing worst-case placement...")
    worst_pl = worst_case_placement(c, topo, num_gpus=NUM_GPUS)
    print(f"  entries={len(worst_pl)}")

    rmap_opt = compute_reorder_map(opt_placement, layers, NUM_EXPERTS, NUM_GPUS)
    rmap_worst = compute_reorder_map(worst_pl, layers, NUM_EXPERTS, NUM_GPUS)

    # Known baseline result from the first completed run
    r_random = {
        "label": "random",
        "tokens_per_sec": 555.8,
        "requests_per_sec": 4.342,
        "ttft_ms": {"p50": None, "p95": None, "p99": None, "mean": None},
        "tpot_ms": {"p50": None, "p95": None, "p99": None, "mean": None},
        "gpu_util_pct": [],
        "gpu_util_avg_pct": None,
        "note": "from prior run (baseline already collected)",
    }
    r_random["imbalance"] = compute_imbalance(loads)
    scenario_results = [r_random]

    hf_path = _resolve_hf_path(MODEL)
    print(f"HF model path: {hf_path}")

    # ── Phase 2: Optimal ───────────────────────────────────────────────────
    import subprocess as sp
    opt_dir = OUT_DIR / "model_optimal"
    existing_shards = len(list(opt_dir.glob("*.safetensors"))) if opt_dir.exists() else 0
    if existing_shards == 16:
        print(f"\n── REORDERING optimal → already complete ({existing_shards} shards), skipping")
    else:
        print(f"\n── REORDERING optimal → {opt_dir}")
        t0 = time.time()
        reorder_safetensors_inplace(hf_path, rmap_opt, opt_dir, consume_source=True)
        print(f"  Done in {time.time() - t0:.1f}s")
    sp.run(["df", "-h", "/"])

    print(f"\n── BENCHMARKING optimal")
    with VllmServer(str(opt_dir), TP, PORT, "optimal") as srv:
        r2 = run_scenario(srv.base_url, srv.model, "optimal")
    r2["imbalance"] = compute_imbalance(loads)
    r2["placement_method"] = opt_method
    scenario_results.append(r2)

    print(f"  Removing {opt_dir}...")
    # But first, derive model_worst from model_optimal using relative permutation
    worst_dir = OUT_DIR / "model_worst"
    rmap_relative = compute_relative_reorder_map(rmap_opt, rmap_worst)

    print(f"\n── REORDERING worst (from optimal via relative permutation) → {worst_dir}")
    t0 = time.time()
    reorder_safetensors_inplace(opt_dir, rmap_relative, worst_dir, consume_source=True)
    print(f"  Done in {time.time() - t0:.1f}s")
    sp.run(["df", "-h", "/"])

    # model_optimal is now consumed into model_worst
    if opt_dir.exists():
        shutil.rmtree(opt_dir, ignore_errors=True)

    print(f"\n── BENCHMARKING worst")
    with VllmServer(str(worst_dir), TP, PORT, "worst") as srv:
        r3 = run_scenario(srv.base_url, srv.model, "worst")
    r3["imbalance"] = compute_imbalance(loads)
    r3["placement_method"] = "load_concentration"
    scenario_results.append(r3)

    print(f"  Removing {worst_dir}...")
    shutil.rmtree(worst_dir, ignore_errors=True)

    # ── Results ────────────────────────────────────────────────────────────
    print_table(scenario_results)
    out_json = OUT_DIR / "results_3way.json"
    out_json.write_text(json.dumps(scenario_results, indent=2))
    print(f"Results written to {out_json}")


if __name__ == "__main__":
    main()
