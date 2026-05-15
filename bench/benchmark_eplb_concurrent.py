#!/usr/bin/env python3
"""
EPLB before/after benchmark under concurrent load.

EPLB expert rebalancing only helps when GPU compute is saturated — i.e. when
multiple requests are queued and one GPU's experts are the bottleneck.  This
benchmark fires N concurrent requests per round and measures throughput (req/s)
and median TTFT under DEFAULT vs EPLB expert placement.

Usage
-----
  python bench/benchmark_eplb_concurrent.py \\
      --model deepseek-ai/DeepSeek-V2-Lite-Chat \\
      --tp 4 --concurrency 8 --n 300 \\
      --snapshot-dir /tmp/plumb
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import requests

RESULTS_DIR = Path("/tmp/sai-bench")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_PROMPTS = [
    "The capital of France is",
    "Explain how transformers work in machine learning:",
    "What are the main causes of climate change?",
    "Describe the transformer architecture in detail:",
    "How does reinforcement learning from human feedback work?",
    "What is mixture-of-experts and why does it improve efficiency?",
    "Explain how expert load imbalance affects MoE inference latency:",
    "Write a detailed explanation of attention mechanisms:",
    "Describe protein synthesis from DNA to functional protein:",
    "Explain quantum computing vs classical computing:",
    "What is backpropagation and why does it work?",
    "How does HTTPS protect data transmission?",
    "Explain gradient descent in one paragraph:",
    "What causes the northern lights?",
    "How do GPUs accelerate deep learning?",
    "Describe supervised vs unsupervised learning:",
    "What is the significance of the Turing test?",
    "Explain large language model pre-training and fine-tuning:",
    "What is the difference between RAM and ROM?",
    "How does the human immune system fight viruses?",
]


def wait_for_server(base_url: str, timeout: int = 480) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _send_request(base_url: str, model: str, prompt: str,
                  results: list, errors: list) -> None:
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": 1,
                  "stream": True, "temperature": 0.0},
            stream=True, timeout=60,
        )
        ttft = None
        for raw in resp.iter_lines():
            if raw and raw != b"data: [DONE]":
                s = raw.decode("utf-8", errors="replace")
                if s.startswith("data: ") and s[6:] != "[DONE]":
                    ttft = time.perf_counter() - t0
                    break
        if ttft is not None:
            results.append(ttft)
        else:
            errors.append("no first token")
    except Exception as e:
        errors.append(str(e))


def run_concurrent_phase(base_url: str, model: str, n: int,
                         concurrency: int, label: str,
                         warmup: int = 40) -> dict:
    """Fire requests at `concurrency` in-flight at a time. Returns stats dict."""
    prompts_pool = (_PROMPTS * (max(n, warmup) // len(_PROMPTS) + 2))

    # Warmup
    print(f"  Warmup ({warmup} req, concurrency={concurrency})...", flush=True)
    _run_batch(base_url, model, prompts_pool[:warmup], concurrency)

    # Measurement
    print(f"  Measuring {n} req ({label}, concurrency={concurrency})...", flush=True)
    t_start = time.perf_counter()
    results, errors = _run_batch(base_url, model, prompts_pool[warmup:warmup+n], concurrency)
    elapsed = time.perf_counter() - t_start

    if not results:
        print(f"  ERROR: all {len(errors)} requests failed. First: {errors[0]}")
        return {}

    ms = [t * 1000 for t in results]
    p = lambda pct: sorted(ms)[int(len(ms) * pct / 100)]

    throughput = len(results) / elapsed
    print(f"  {label}: {len(results)} ok / {len(errors)} err  "
          f"p50={p(50):.1f}ms  p90={p(90):.1f}ms  "
          f"throughput={throughput:.2f} req/s")

    return {
        "n": len(results),
        "errors": len(errors),
        "mean_ms": statistics.mean(ms),
        "p50_ms": p(50),
        "p90_ms": p(90),
        "p95_ms": p(95) if len(ms) >= 20 else p(90),
        "p99_ms": p(99) if len(ms) >= 100 else p(95),
        "throughput_rps": throughput,
        "elapsed_s": elapsed,
    }


def _run_batch(base_url: str, model: str, prompts: list[str],
               concurrency: int) -> tuple[list, list]:
    results: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()

    prompt_q: queue.Queue = queue.Queue()
    for p in prompts:
        prompt_q.put(p)

    def worker():
        while True:
            try:
                prompt = prompt_q.get_nowait()
            except queue.Empty:
                return
            local_r: list[float] = []
            local_e: list[str] = []
            _send_request(base_url, model, prompt, local_r, local_e)
            with lock:
                results.extend(local_r)
                errors.extend(local_e)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results, errors


def load_expert_counts(snapshot_dir: Path, model_filter: str | None) -> dict:
    merged: dict[str, int] = defaultdict(int)
    n_files = 0
    for f in sorted(snapshot_dir.glob("*_snapshot.json")):
        d = json.loads(f.read_text())
        if model_filter and model_filter.lower() not in d.get("model_name", "").lower():
            continue
        for k, v in d.get("expert_counts", {}).items():
            merged[k] += v
        n_files += 1
    print(f"  Loaded {len(merged)} expert slots from {n_files} snapshot(s)")
    return dict(merged)


def compute_eplb(expert_counts: dict, n_gpus: int) -> tuple[object, str]:
    from plumb_pro.rebalance.engine import ExpertLoad, compute_plan
    loads = []
    for key, count in expert_counts.items():
        parts = key.split(":")
        if len(parts) == 2:
            loads.append(ExpertLoad(layer_id=int(parts[0]),
                                    expert_id=int(parts[1]),
                                    token_count=count))
    plan = compute_plan(loads, n_gpus=n_gpus)
    mapping: dict[str, dict[str, int]] = {}
    for m in plan.moves:
        mapping.setdefault(str(m.layer), {})[str(m.expert_id)] = m.to_gpu
    env_val = json.dumps(mapping, separators=(",", ":")) if mapping else ""
    print(f"  EPLB: {len(plan.moves)} moves, ~{plan.estimated_ttft_improvement_pct:.1f}% est. gain "
          f"(conf {plan.confidence:.0%})")
    return plan, env_val


def start_server(cmd: list[str], env_extra: dict | None = None) -> subprocess.Popen:
    log = RESULTS_DIR / "eplb_server.log"
    full_env = os.environ.copy()
    if env_extra:
        full_env.update(env_extra)
    with open(log, "w") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=f, env=full_env)


def stop_server(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Concurrent in-flight requests per phase")
    ap.add_argument("--n", type=int, default=300, help="Requests per phase")
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--snapshot-dir", default="/tmp/plumb")
    ap.add_argument("--model-filter", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    base_url = f"http://localhost:{args.port}"

    print("\n" + "=" * 65)
    print(" EPLB CONCURRENT BENCHMARK")
    print(f" Model       : {args.model}")
    print(f" TP/EP size  : {args.tp}")
    print(f" Concurrency : {args.concurrency} in-flight")
    print(f" N per phase : {args.n}")
    print("=" * 65)

    # Load activation data + compute plan
    print("\n[1/4] Loading snapshot & computing EPLB plan...")
    ec = load_expert_counts(Path(args.snapshot_dir), args.model_filter)
    if not ec:
        print("ERROR: no snapshot data found. Run plumb first.")
        sys.exit(1)
    plan, expert_map_json = compute_eplb(ec, n_gpus=args.tp)
    if not expert_map_json:
        print("Placement already optimal — nothing to test.")
        sys.exit(0)

    server_cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--tensor-parallel-size", str(args.tp),
        "--enable-expert-parallel",
        "--enforce-eager",
        "--port", str(args.port),
        "--disable-log-requests",
        "--max-num-seqs", str(args.concurrency * 2),  # allow enough queue depth
    ]
    if args.trust_remote_code:
        server_cmd.append("--trust-remote-code")

    # Phase 1: default placement
    print("\n[2/4] DEFAULT placement...")
    proc = start_server(server_cmd)
    print(f"  PID {proc.pid} — waiting for ready...")
    if not wait_for_server(base_url):
        print("ERROR: server not ready (default)")
        stop_server(proc); sys.exit(1)
    print("  Ready.")
    try:
        default_stats = run_concurrent_phase(
            base_url, args.model, args.n, args.concurrency,
            "DEFAULT", warmup=args.warmup)
    finally:
        stop_server(proc)
        time.sleep(5)

    # Phase 2: EPLB placement
    print("\n[3/4] EPLB placement...")
    proc = start_server(server_cmd, env_extra={"VLLM_EXPERT_MAP": expert_map_json})
    print(f"  PID {proc.pid} — waiting for ready...")
    if not wait_for_server(base_url):
        print("ERROR: server not ready (EPLB)")
        stop_server(proc); sys.exit(1)
    print("  Ready.")
    try:
        eplb_stats = run_concurrent_phase(
            base_url, args.model, args.n, args.concurrency,
            "EPLB  ", warmup=args.warmup)
    finally:
        stop_server(proc)

    # Report
    print("\n" + "=" * 65)
    print(" RESULTS")
    print("=" * 65)
    print(f"\n  {'Metric':<22} {'DEFAULT':>14} {'EPLB':>14} {'Δ':>10}")
    print("  " + "-" * 60)
    for m in ["mean_ms", "p50_ms", "p90_ms", "p95_ms", "throughput_rps"]:
        dv = default_stats.get(m, 0)
        ev = eplb_stats.get(m, 0)
        delta = ev - dv
        pct = (delta / dv * 100) if dv else 0
        unit = "req/s" if "rps" in m else "ms"
        sign = "+" if delta >= 0 else ""
        print(f"  {m:<22} {dv:>11.2f}{unit} {ev:>11.2f}{unit} {sign}{delta:>+.2f} ({sign}{pct:.1f}%)")

    p50_delta_pct = (eplb_stats.get("p50_ms",0) - default_stats.get("p50_ms",0)) / default_stats.get("p50_ms",1) * 100
    tput_delta_pct = (eplb_stats.get("throughput_rps",0) - default_stats.get("throughput_rps",0)) / default_stats.get("throughput_rps",1) * 100
    print(f"\n  P50 TTFT change     : {p50_delta_pct:+.1f}%")
    print(f"  Throughput change   : {tput_delta_pct:+.1f}%")
    print(f"  Algorithm estimate  : {plan.estimated_ttft_improvement_pct:.1f}%  (conf {plan.confidence:.0%})")
    print("=" * 65)

    result = {
        "model": args.model, "tp": args.tp,
        "concurrency": args.concurrency, "n_per_phase": args.n,
        "default": default_stats, "eplb": eplb_stats,
        "p50_delta_pct": p50_delta_pct,
        "throughput_delta_pct": tput_delta_pct,
        "plan_estimated_pct": plan.estimated_ttft_improvement_pct,
        "plan_confidence": plan.confidence,
        "n_expert_moves": len(plan.moves),
    }
    out = RESULTS_DIR / "eplb_concurrent_benchmark.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n  Full results: {out}")
    print("=" * 65)


if __name__ == "__main__":
    main()
