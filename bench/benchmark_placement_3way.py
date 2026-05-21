#!/usr/bin/env python3
"""3-way expert placement benchmark: random vs optimal vs worst-case.

Measures the throughput impact of expert weight reordering under three scenarios:

  random    — original model, unmodified weights (control + profiling phase)
  optimal   — weights reordered to balance load across GPUs
  worst     — weights reordered to concentrate load (adversarial imbalance)

Headline metric is tokens/sec. Additional metrics per scenario:
  - req/s, TTFT p50/p95/p99 ms, TPOT p50/p95/p99 ms
  - GPU utilization avg % (polled via nvidia-smi)
  - Expert imbalance ratio (max/mean per layer)
  - Estimated communication overhead µs
  - Cross-GPU co-activation rate (if co-activation data available)

Usage on Vast.ai instance:
  python bench/benchmark_placement_3way.py \\
    --model Qwen/Qwen3.5-35B-A3B \\
    --tp 4 --num-gpus 4 --num-experts 128

Requires: vllm, safetensors, torch, plumb
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------
WARMUP_REQUESTS = 20
BENCHMARK_REQUESTS = 200
PROFILING_REQUESTS = 300   # longer profiling run for stable load counts
MAX_OUTPUT_TOKENS = 128    # representative decode length

# ShareGPT-style prompts: varied length, diverse content
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

# Repeat to reach PROFILING_REQUESTS
_PROMPT_POOL = (PROMPTS * ((PROFILING_REQUESTS // len(PROMPTS)) + 2))


# ---------------------------------------------------------------------------
# Server / measurement helpers (reused from hardware_benchmark.py patterns)
# ---------------------------------------------------------------------------

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
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--tensor-parallel-size", str(tp),
            "--enforce-eager",
            "--port", str(port),
            "--no-enable-log-requests",
        ]
        self._cmd = cmd
        self._log_path = Path(f"/tmp/bench3way/server_{log_suffix}.log")
        self._proc: subprocess.Popen | None = None
        self._log = None

    def __enter__(self):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self._log_path, "w")
        print(f"  Starting vLLM: {self._cmd[0]} ... --port {self.port}")
        self._proc = subprocess.Popen(self._cmd, stdout=self._log, stderr=self._log)
        print(f"  PID {self._proc.pid} — waiting for health check...")
        if not wait_for_server(self.base_url):
            self._proc.terminate()
            raise RuntimeError(f"vLLM server at {self.base_url} did not start")
        print(f"  Ready at {self.base_url}")
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
        time.sleep(5)  # drain GPU memory before next scenario


class GpuUtilPoller:
    """Background thread: sample GPU utilization every 2 seconds."""

    def __init__(self):
        self._samples: list[list[int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self) -> list[float]:
        self._stop.set()
        self._thread.join(timeout=5)
        if not self._samples:
            return []
        # Average utilization per GPU across all samples
        n_gpus = len(self._samples[0])
        return [
            round(statistics.mean(s[g] for s in self._samples), 1)
            for g in range(n_gpus)
        ]

    def _run(self):
        while not self._stop.wait(2.0):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    text=True,
                )
                self._samples.append([int(x.strip()) for x in out.strip().splitlines()])
            except Exception:
                pass


def _measure_request(base_url: str, model: str, prompt: str) -> dict | None:
    """Send one streaming request; return TTFT, total latency, output tokens."""
    t0 = time.perf_counter()
    t_first = None
    n_tokens = 0
    try:
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "stream": True,
                "temperature": 0.0,
            },
            stream=True,
            timeout=120,
        )
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
                text = chunk.get("choices", [{}])[0].get("text", "")
                # Approximate token count from chunk text (not exact but consistent)
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
    total_latency = t_end - t0
    tpot = (total_latency - ttft) / max(n_tokens - 1, 1)
    return {
        "ttft_s": ttft,
        "total_latency_s": total_latency,
        "output_tokens": n_tokens,
        "tpot_s": tpot,
    }


def _percentile(vals: list[float], p: float) -> float:
    s = sorted(vals)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def run_scenario(base_url: str, model: str, label: str) -> dict:
    """Warmup then benchmark one scenario. Returns metrics dict."""
    prompts = _PROMPT_POOL[:BENCHMARK_REQUESTS + WARMUP_REQUESTS]

    print(f"  [{label}] Warmup ({WARMUP_REQUESTS} requests)...")
    for p in prompts[:WARMUP_REQUESTS]:
        requests.post(
            f"{base_url}/v1/completions",
            json={"model": model, "prompt": p, "max_tokens": 8, "temperature": 0.0},
            timeout=60,
        )

    print(f"  [{label}] Benchmarking ({BENCHMARK_REQUESTS} requests, max_tokens={MAX_OUTPUT_TOKENS})...")
    poller = GpuUtilPoller()
    poller.start()

    results_list: list[dict] = []
    t_bench_start = time.perf_counter()
    bench_prompts = prompts[WARMUP_REQUESTS:WARMUP_REQUESTS + BENCHMARK_REQUESTS]
    for i, p in enumerate(bench_prompts):
        r = _measure_request(base_url, model, p)
        if r:
            results_list.append(r)
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_bench_start
            done = i + 1
            tps_so_far = sum(rr["output_tokens"] for rr in results_list) / elapsed
            print(f"    {done}/{BENCHMARK_REQUESTS}  tok/s={tps_so_far:.1f}")

    t_bench_end = time.perf_counter()
    gpu_util = poller.stop()

    if not results_list:
        return {"label": label, "error": "no successful requests"}

    ttfts = [r["ttft_s"] for r in results_list]
    tpots = [r["tpot_s"] for r in results_list]
    total_tokens = sum(r["output_tokens"] for r in results_list)
    elapsed = t_bench_end - t_bench_start

    return {
        "label": label,
        "n_requests": len(results_list),
        "total_output_tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "tokens_per_sec": round(total_tokens / elapsed, 2),
        "requests_per_sec": round(len(results_list) / elapsed, 3),
        "ttft_ms": {
            "p50": round(_percentile(ttfts, 0.50) * 1000, 2),
            "p95": round(_percentile(ttfts, 0.95) * 1000, 2),
            "p99": round(_percentile(ttfts, 0.99) * 1000, 2),
            "mean": round(statistics.mean(ttfts) * 1000, 2),
        },
        "tpot_ms": {
            "p50": round(_percentile(tpots, 0.50) * 1000, 3),
            "p95": round(_percentile(tpots, 0.95) * 1000, 3),
            "p99": round(_percentile(tpots, 0.99) * 1000, 3),
            "mean": round(statistics.mean(tpots) * 1000, 3),
        },
        "gpu_util_pct": gpu_util,
        "gpu_util_avg_pct": round(statistics.mean(gpu_util), 1) if gpu_util else None,
    }


# ---------------------------------------------------------------------------
# Profiling: collect expert load counts from the random-placement run
# ---------------------------------------------------------------------------

def run_profiling_phase(base_url: str, model: str) -> dict[tuple[int, int], int]:
    """Send PROFILING_REQUESTS to a live server; parse plumb snapshot for loads.

    Returns {(layer_id, expert_id): token_count}.
    Falls back to empty dict (bench continues without placement reordering).
    """
    print(f"  [profile] Sending {PROFILING_REQUESTS} requests for expert load data...")
    for i, p in enumerate(_PROMPT_POOL[:PROFILING_REQUESTS]):
        requests.post(
            f"{base_url}/v1/completions",
            json={"model": model, "prompt": p, "max_tokens": MAX_OUTPUT_TOKENS,
                  "temperature": 0.0},
            timeout=120,
        )
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{PROFILING_REQUESTS}")

    # Read latest snapshot written by plumb autoattach
    snap_dir = Path("/tmp/plumb")
    snaps = sorted(snap_dir.glob("*_snapshot.json"), key=lambda p: p.stat().st_mtime)
    if not snaps:
        print("  [profile] No snapshot found — skipping placement reordering", file=sys.stderr)
        return {}

    snap = json.loads(snaps[-1].read_text())
    expert_raw = snap.get("expert_counts", snap.get("expert_loads", {}))
    loads: dict[tuple[int, int], int] = {}
    for key, count in expert_raw.items():
        lid, eid = map(int, key.split(":"))
        loads[(lid, eid)] = int(count)
    print(f"  [profile] Loaded {len(loads)} (layer, expert) counts from {snaps[-1].name}")
    return loads


def loads_to_counter(loads: dict[tuple[int, int], int]):
    """Reconstruct an ActivationCounter from snapshot load data."""
    from plumb.counter import ActivationCounter
    c = ActivationCounter(window_size=10_000_000)
    for (lid, eid), count in loads.items():
        c.record(lid, eid, count)
    return c


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_imbalance(loads: dict[tuple[int, int], int]) -> dict:
    """Per-layer imbalance ratio and aggregate stats."""
    if not loads:
        return {}
    layers: dict[int, dict[int, int]] = {}
    for (lid, eid), count in loads.items():
        layers.setdefault(lid, {})[eid] = count

    ratios = []
    layer_stats = []
    for lid in sorted(layers):
        counts = list(layers[lid].values())
        total = sum(counts)
        mean = total / len(counts) if counts else 1
        ratio = max(counts) / mean if mean > 0 else 1.0
        ratios.append(ratio)
        layer_stats.append({"layer_id": lid, "imbalance_ratio": round(ratio, 4)})

    return {
        "mean_imbalance_ratio": round(statistics.mean(ratios), 4) if ratios else None,
        "max_imbalance_ratio": round(max(ratios), 4) if ratios else None,
        "per_layer": layer_stats,
    }


def compute_comm_stats(
    placement: dict[tuple[int, int], list[int]],
    loads: dict[tuple[int, int], int],
    num_gpus: int,
) -> dict:
    """Estimated communication overhead using plumb's comms model."""
    try:
        from plumb.analysis.comms import CommunicationConstants, compute_communication_cost
        from numa_topology import Topology
        from numa_topology.pcie import PCIeTopology

        topo = Topology.flat(num_gpus)
        pcie = PCIeTopology.discover()
        constants = CommunicationConstants()
        result = compute_communication_cost(
            placement, placement, loads, topo, pcie,
            constants=constants, src_gpu=0,
        )
        return {
            "total_overhead_us": round(result.current_overhead_us, 2),
            "cross_numa_pcie_constant_us": round(constants.cross_numa_pcie_us, 3),
            "same_numa_constant_us": round(constants.same_numa_us, 3),
        }
    except Exception as e:
        return {"error": str(e)}


def build_placement_from_loads(
    loads: dict[tuple[int, int], int],
    num_gpus: int,
    strategy: str,
) -> tuple[dict[tuple[int, int], list[int]], str]:
    """Compute placement dict from snapshot loads.

    strategy: "optimal" | "worst"
    Returns (placement, method_name).
    """
    from plumb.analysis.placement import recommend_placement, worst_case_placement
    from numa_topology import Topology

    topo = Topology.flat(num_gpus)
    counter = loads_to_counter(loads)

    if strategy == "optimal":
        rec = recommend_placement(counter, topo, num_gpus=num_gpus)
        if rec is None:
            return {}, "none"
        return rec.expert_placement, rec.method

    if strategy == "worst":
        pl = worst_case_placement(counter, topo, num_gpus=num_gpus)
        return pl, "load_concentration"

    raise ValueError(f"Unknown strategy: {strategy!r}")


def reorder_model(
    src_model: str,
    placement: dict[tuple[int, int], list[int]],
    out_dir: str,
    num_gpus: int,
    layers: list[int],
    num_experts: int,
) -> str:
    """Write permuted model weights to out_dir. Returns out_dir."""
    from plumb.tools.reorder_experts import compute_reorder_map, reorder_safetensors

    rmap = compute_reorder_map(placement, layers=layers, num_experts_per_layer=num_experts,
                               num_gpus=num_gpus)
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    reorder_safetensors(src_model, rmap, str(out))
    return str(out)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_table(scenarios: list[dict]) -> None:
    names = [s["label"] for s in scenarios]
    col_w = max(12, max(len(n) for n in names))
    header = f"{'Metric':<32}" + "".join(f"{n:>{col_w}}" for n in names)
    sep = "-" * len(header)

    def row(label: str, key_path: list[str], fmt: str = ".2f", suffix: str = "") -> str:
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
    print(" PLACEMENT BENCHMARK RESULTS")
    print("=" * len(header))
    print(header)
    print(sep)
    print(row("tokens/sec (headline)", ["tokens_per_sec"], ".1f"))
    print(row("requests/sec",          ["requests_per_sec"], ".3f"))
    print(row("TTFT p50 (ms)",         ["ttft_ms", "p50"], ".2f"))
    print(row("TTFT p95 (ms)",         ["ttft_ms", "p95"], ".2f"))
    print(row("TTFT p99 (ms)",         ["ttft_ms", "p99"], ".2f"))
    print(row("TPOT p50 (ms/tok)",     ["tpot_ms", "p50"], ".3f"))
    print(row("TPOT p95 (ms/tok)",     ["tpot_ms", "p95"], ".3f"))
    print(row("GPU util avg (%)",      ["gpu_util_avg_pct"], ".1f"))
    print(row("Imbalance ratio (mean)",["imbalance", "mean_imbalance_ratio"], ".4f"))
    print(row("Imbalance ratio (max)", ["imbalance", "max_imbalance_ratio"], ".4f"))
    print(row("Comm overhead (µs)",    ["comm", "total_overhead_us"], ".1f"))
    print(sep)

    # Delta vs random (baseline)
    if len(scenarios) > 1:
        base_tps = scenarios[0].get("tokens_per_sec")
        print()
        print(f"  Delta vs random (tok/s):")
        for s in scenarios[1:]:
            tps = s.get("tokens_per_sec")
            if base_tps and tps:
                delta_pct = (tps - base_tps) / base_tps * 100
                sign = "+" if delta_pct >= 0 else ""
                print(f"    {s['label']:<20} {sign}{delta_pct:.1f}%  ({tps:.1f} vs {base_tps:.1f})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3-way expert placement benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs (defaults to --tp)")
    parser.add_argument("--num-experts", type=int, default=128,
                        help="Number of experts per layer (Qwen3 MoE: 128)")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--out-dir", default="/tmp/bench3way")
    parser.add_argument("--skip-profiling", action="store_true",
                        help="Skip profiling phase, use random placement only")
    parser.add_argument("--skip-reorder", action="store_true",
                        help="Skip model reordering, benchmark original weights 3x")
    args = parser.parse_args()

    num_gpus = args.num_gpus or args.tp
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print(" 3-WAY EXPERT PLACEMENT BENCHMARK")
    print(f" Model:    {args.model}")
    print(f" TP={args.tp}  GPUs={num_gpus}  Experts/layer={args.num_experts}")
    try:
        host = subprocess.check_output("hostname", text=True).strip()
        print(f" Host:     {host}")
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"], text=True
        ).strip().splitlines()
        for i, g in enumerate(gpu_info):
            print(f" GPU {i}:    {g}")
    except Exception:
        pass
    print("=" * 60)

    scenario_results = []

    # ── Phase 1: Random placement (baseline + optional profiling) ────────────
    print()
    print("── PHASE 1: Random placement (baseline + profiling) ──────────────")

    model_path = args.model
    loads: dict[tuple[int, int], int] = {}

    with_profiler = not args.skip_profiling
    srv_cmd_extra = []
    snap_dir = Path("/tmp/plumb")
    snap_dir.mkdir(parents=True, exist_ok=True)

    if with_profiler:
        # Launch with plumb autoattach to capture expert loads
        cmd = [
            "plumb", "run", "--",
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--tensor-parallel-size", str(args.tp),
            "--enforce-eager",
            "--port", str(args.port),
            "--no-enable-log-requests",
        ]
        log_path = out_dir / "server_random_profiling.log"
        print(f"  Starting vLLM + plumb profiler (log: {log_path})")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        base_url = f"http://localhost:{args.port}"
        print(f"  PID {proc.pid} — waiting for health check...")
        if not wait_for_server(base_url):
            proc.terminate()
            log_file.close()
            sys.exit("ERROR: vLLM server did not start")
        print(f"  Ready. Running profiling requests...")
        loads = run_profiling_phase(base_url, args.model)
        print(f"  Running baseline benchmark...")
        r1 = run_scenario(base_url, args.model, "random")
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                proc.kill()
        log_file.close()
        time.sleep(5)
    else:
        with VllmServer(args.model, args.tp, args.port, "random") as srv:
            r1 = run_scenario(srv.base_url, srv.model, "random")
        loads = {}

    if loads:
        r1["imbalance"] = compute_imbalance(loads)
        # Build identity placement (original order) for comm stats
        layers_seen = sorted({lid for lid, _ in loads})
        eids_seen = sorted({eid for _, eid in loads})
        identity_pl = {(lid, eid): [eid % num_gpus] for lid in layers_seen for eid in eids_seen}
        r1["comm"] = compute_comm_stats(identity_pl, loads, num_gpus)
    scenario_results.append(r1)

    if args.skip_reorder or not loads:
        print()
        if not loads:
            print("  No profile data — skipping optimal/worst reordering.")
        print("  Reporting random baseline only.")
        print_table(scenario_results)
        out_json = out_dir / "results_3way.json"
        out_json.write_text(json.dumps(scenario_results, indent=2))
        print(f"Results written to {out_json}")
        return

    # ── Compute placements from profile data ──────────────────────────────
    print()
    print("── COMPUTING PLACEMENTS ──────────────────────────────────────────")
    layers = sorted({lid for lid, _ in loads})
    num_experts = args.num_experts

    print(f"  {len(layers)} layers, {num_experts} experts/layer, {num_gpus} GPUs")

    print("  Computing optimal placement...")
    opt_placement, opt_method = build_placement_from_loads(loads, num_gpus, "optimal")
    print(f"  Method: {opt_method}")

    print("  Computing worst-case placement...")
    worst_placement, worst_method = build_placement_from_loads(loads, num_gpus, "worst")
    print(f"  Method: {worst_method}")

    # ── Reorder model weights ──────────────────────────────────────────────
    print()
    print("── REORDERING MODEL WEIGHTS ──────────────────────────────────────")
    opt_model_dir = str(out_dir / "model_optimal")
    worst_model_dir = str(out_dir / "model_worst")

    print(f"  Writing optimal model to {opt_model_dir}...")
    t0 = time.time()
    reorder_model(model_path, opt_placement, opt_model_dir, num_gpus, layers, num_experts)
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"  Writing worst-case model to {worst_model_dir}...")
    t0 = time.time()
    reorder_model(model_path, worst_placement, worst_model_dir, num_gpus, layers, num_experts)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Phase 2: Optimal placement ─────────────────────────────────────────
    print()
    print("── PHASE 2: Optimal placement ────────────────────────────────────")
    with VllmServer(opt_model_dir, args.tp, args.port, "optimal") as srv:
        r2 = run_scenario(srv.base_url, srv.model, "optimal")
    r2["imbalance"] = compute_imbalance(loads)
    r2["comm"] = compute_comm_stats(opt_placement, loads, num_gpus)
    r2["placement_method"] = opt_method
    scenario_results.append(r2)

    # ── Phase 3: Worst-case placement ─────────────────────────────────────
    print()
    print("── PHASE 3: Worst-case placement ─────────────────────────────────")
    with VllmServer(worst_model_dir, args.tp, args.port, "worst") as srv:
        r3 = run_scenario(srv.base_url, srv.model, "worst")
    r3["imbalance"] = compute_imbalance(loads)
    r3["comm"] = compute_comm_stats(worst_placement, loads, num_gpus)
    r3["placement_method"] = worst_method
    scenario_results.append(r3)

    # ── Results ───────────────────────────────────────────────────────────
    print_table(scenario_results)

    out_json = out_dir / "results_3way.json"
    out_json.write_text(json.dumps(scenario_results, indent=2))
    print(f"Results written to {out_json}")


if __name__ == "__main__":
    main()
