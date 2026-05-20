#!/usr/bin/env python3
"""
ShareGPT MoE Benchmark — Mixtral-8x7B expert load imbalance + plumb placement.

Measures TTFT and throughput at multiple concurrency levels using real
ShareGPT conversations, runs plumb profiling to capture expert activation
imbalance, then optionally repeats after applying the recommended placement.

Usage
-----
  python bench/sharegpt_moe_bench.py \\
      --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
      --tp 2 \\
      --concurrency 1,4,16,64 \\
      --num-requests 500 \\
      --output-dir bench/results/mixtral-sharegpt-$(date +%Y%m%d) \\
      [--sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json] \\
      [--skip-phase2]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ── Optional rich import ──────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn  # noqa: F401
    from rich.table import Table
    _console = Console()
    _rich = True
except ImportError:
    _console = None  # type: ignore[assignment]
    _rich = False

try:
    import requests as _requests_lib
except ImportError:
    _requests_lib = None  # type: ignore[assignment]

SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered"
    "/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)

# Rough chars-per-token estimate for simple length filter
_CHARS_PER_TOKEN = 4
_MAX_PROMPT_CHARS = 512 * _CHARS_PER_TOKEN  # ~2048 chars


# ── Data loading ──────────────────────────────────────────────────────────────

def _download_sharegpt(dest: Path) -> None:
    """Download ShareGPT JSON to dest with a progress indicator."""
    print(f"Downloading ShareGPT dataset to {dest} ...", flush=True)
    try:
        urllib.request.urlretrieve(SHAREGPT_URL, dest)
    except Exception as exc:
        print(
            f"ERROR: download failed: {exc}\n"
            "Retry with --sharegpt-path pointing to a local copy.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  Saved {dest.stat().st_size // 1024 // 1024} MiB", flush=True)


def load_sharegpt_prompts(
    path: Path | None,
    num_requests: int,
    cache_dir: Path,
) -> list[dict]:
    """Return list of {prompt: str, output_len: int} dicts sampled from ShareGPT."""
    if path is None:
        cached = cache_dir / "ShareGPT_V3_unfiltered_cleaned_split.json"
        if not cached.exists():
            _download_sharegpt(cached)
        path = cached

    print(f"Loading ShareGPT from {path} ...", flush=True)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    prompts: list[dict] = []
    for conv in raw:
        turns = conv.get("conversations") or conv.get("conversation") or []
        human_turns = [t for t in turns if t.get("from") in ("human", "user")]
        assistant_turns = [t for t in turns if t.get("from") in ("gpt", "assistant")]
        if not human_turns or not assistant_turns:
            continue
        prompt_text = human_turns[0].get("value", "")
        asst_text = assistant_turns[0].get("value", "")
        if not prompt_text or len(prompt_text) > _MAX_PROMPT_CHARS:
            continue
        output_len = max(1, len(asst_text) // _CHARS_PER_TOKEN)
        prompts.append({"prompt": prompt_text, "output_len": output_len})
        if len(prompts) >= num_requests * 3:
            break  # gather extra to have headroom after filtering

    if len(prompts) < num_requests:
        print(
            f"WARNING: only {len(prompts)} usable ShareGPT turns found "
            f"(wanted {num_requests}). Proceeding with available data.",
            file=sys.stderr,
        )

    # Deterministic sample
    result = prompts[:num_requests]
    print(f"  Loaded {len(result)} prompts (filtered from {len(prompts)} candidates)")
    return result


# ── Server lifecycle ───────────────────────────────────────────────────────────

def start_server(
    model: str, tp: int, port: int, output_dir: Path,
    extra_env: dict | None = None, max_num_seqs: int = 256,
) -> subprocess.Popen:
    log = output_dir / "vllm_server.log"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--tensor-parallel-size", str(tp),
        "--dtype", "float16",
        "--port", str(port),
        "--disable-log-requests",
        "--max-num-seqs", str(max_num_seqs),
        "--max-model-len", "4096",
    ]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"  Launching: {' '.join(cmd[:8])} ...", flush=True)
    with open(log, "w") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=f, env=env)


def stop_server(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def wait_for_server(base_url: str, timeout: int = 600) -> bool:
    if _requests_lib is None:
        raise RuntimeError("requests library not installed")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if _requests_lib.get(f"{base_url}/health", timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


# ── HTTP benchmark helpers ────────────────────────────────────────────────────

def _send_request(
    base_url: str, model: str, prompt: str,
    results: list, errors: list,
) -> None:
    t0 = time.perf_counter()
    try:
        resp = _requests_lib.post(
            f"{base_url}/v1/completions",
            json={
                "model": model, "prompt": prompt,
                "max_tokens": 1, "stream": True, "temperature": 0.0,
            },
            stream=True, timeout=90,
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
    except Exception as exc:
        errors.append(str(exc))


def _run_batch(
    base_url: str, model: str, prompts: list[str], concurrency: int,
) -> tuple[list[float], list[str]]:
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
            lr: list[float] = []
            le: list[str] = []
            _send_request(base_url, model, prompt, lr, le)
            with lock:
                results.extend(lr)
                errors.extend(le)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


def _pct(ms: list[float], p: int) -> float:
    return sorted(ms)[int(len(ms) * p / 100)]


def run_concurrent_phase(
    base_url: str,
    model: str,
    prompts: list[dict],
    concurrency_levels: list[int],
    label: str,
    warmup: int = 40,
) -> list[dict]:
    """Run benchmark at each concurrency level. Returns list of stat dicts."""
    prompt_texts = [p["prompt"] for p in prompts]
    pool = (prompt_texts * (max(len(prompt_texts), warmup + 200) // len(prompt_texts) + 2))

    # Warmup (highest concurrency to ensure KV-cache is warm)
    top_c = max(concurrency_levels)
    print(f"  Warmup ({warmup} req, concurrency={top_c})...", flush=True)
    _run_batch(base_url, model, pool[:warmup], top_c)

    all_stats: list[dict] = []
    for c in concurrency_levels:
        n = len(prompt_texts)
        print(f"  [{label}] concurrency={c}, {n} requests...", flush=True)
        t_start = time.perf_counter()
        results, errors = _run_batch(base_url, model, pool[warmup:warmup + n], c)
        elapsed = time.perf_counter() - t_start

        if not results:
            print(f"    ERROR: all {len(errors)} requests failed. First: {errors[0]}")
            all_stats.append({"concurrency": c, "error": errors[0] if errors else "unknown"})
            continue

        ms = [t * 1000 for t in results]
        throughput = len(results) / elapsed
        stat = {
            "concurrency": c,
            "n": len(results),
            "errors": len(errors),
            "mean_ms": round(statistics.mean(ms), 2),
            "p50_ms": round(_pct(ms, 50), 2),
            "p95_ms": round(_pct(ms, 95) if len(ms) >= 20 else _pct(ms, 90), 2),
            "p99_ms": round(_pct(ms, 99) if len(ms) >= 100 else _pct(ms, 95), 2),
            "throughput_rps": round(throughput, 3),
            "elapsed_s": round(elapsed, 2),
        }
        print(
            f"    p50={stat['p50_ms']:.1f}ms  p95={stat['p95_ms']:.1f}ms  "
            f"throughput={stat['throughput_rps']:.2f} req/s  errors={len(errors)}"
        )
        all_stats.append(stat)
    return all_stats


# ── Plumb profiling pass ──────────────────────────────────────────────────────

def _attach_to_vllm(hooks, llm) -> int:
    """Try known vLLM internal paths to find the model and attach hooks."""
    candidates = []
    try:
        candidates.append(
            llm.llm_engine.model_executor.driver_worker.model_runner.model
        )
    except AttributeError:
        pass
    try:
        candidates.append(
            llm.llm_engine.model_executor.driver_worker.model
        )
    except AttributeError:
        pass
    # vLLM 0.5+ multiprocessing executor path
    try:
        exec_obj = llm.llm_engine.model_executor
        for attr in ("worker", "workers"):
            w = getattr(exec_obj, attr, None)
            if w is None:
                continue
            worker = w[0] if isinstance(w, list) else w
            for mr_attr in ("model_runner", ""):
                mr = getattr(worker, mr_attr, worker) if mr_attr else worker
                m = getattr(mr, "model", None)
                if m is not None:
                    candidates.append(m)
    except Exception:
        pass

    for model in candidates:
        try:
            n = hooks.attach(model)
            if n > 0:
                return n
        except Exception:
            pass

    print(
        "WARNING: could not attach plumb hooks to vLLM model — "
        "no known path worked. Profiling data will be empty.",
        file=sys.stderr,
    )
    return 0


def run_profiling_pass(
    model_name: str,
    prompts: list[dict],
    num_profile: int = 200,
) -> tuple[object, object, float]:
    """Spawn a TP=1 LLM, attach plumb hooks, run prompts.

    Returns (ProfileReport, ActivationCounter, duration_seconds).
    Counter is needed for hetero re-analysis; report may be None on failure.
    """
    try:
        from vllm import LLM, SamplingParams  # type: ignore[import]
    except ImportError:
        print("ERROR: vllm not installed. Cannot run profiling pass.", file=sys.stderr)
        return None

    try:
        from plumb.counter import ActivationCounter
        from plumb.hook import ProfilingHooks
        from plumb.report.generator import generate_report
        from plumb.topology import Topology
    except ImportError as exc:
        print(f"ERROR: plumb not installed: {exc}", file=sys.stderr)
        return None, None, 0.0

    profile_prompts = [p["prompt"] for p in prompts[:num_profile]]
    print(f"  Spawning TP=1 LLM for profiling ({len(profile_prompts)} prompts)...", flush=True)

    t_start = time.time()
    try:
        llm = LLM(
            model=model_name,
            dtype="float16",
            tensor_parallel_size=1,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )
    except Exception as exc:
        print(f"ERROR: failed to load LLM for profiling: {exc}", file=sys.stderr)
        return None, None, 0.0

    counter = ActivationCounter(window_size=500)
    hooks = ProfilingHooks(counter)
    n_hooks = _attach_to_vllm(hooks, llm)
    print(f"  Plumb attached to {n_hooks} MoE layers", flush=True)

    sampling = SamplingParams(temperature=0.0, max_tokens=50)
    try:
        llm.generate(profile_prompts, sampling)
    except Exception as exc:
        print(f"WARNING: profiling generation error: {exc}", file=sys.stderr)

    hooks.detach()
    duration = time.time() - t_start

    print(f"  Profiling done in {duration:.1f}s  "
          f"({counter.pass_count} forward passes recorded)", flush=True)

    topology = Topology.flat(1)
    report = generate_report(
        counter=counter,
        topology=topology,
        model_name=model_name,
        duration_seconds=duration,
        num_gpus=1,
    )
    return report, counter, duration


# ── Placement scenario analysis ───────────────────────────────────────────────

def _greedy_placement(
    sorted_pairs: list[tuple[tuple[int, int], float]],
    num_experts: int,
    num_gpus: int,
    maximize: bool,
) -> dict[int, int]:
    assignment: dict[int, int] = {}
    gpu_load = [0] * num_gpus
    per_gpu = max(1, (num_experts + num_gpus - 1) // num_gpus)

    def available() -> list[int]:
        return [g for g in range(num_gpus) if gpu_load[g] < per_gpu]

    def assign(e: int, g: int) -> None:
        assignment[e] = g
        gpu_load[g] += 1

    for (a, b), _ in sorted_pairs:
        a_done = a in assignment
        b_done = b in assignment
        avail = available()
        if not avail:
            break
        if not a_done and not b_done:
            assign(a, avail[0])
            avail = available()
            if not avail:
                continue
            if maximize:
                diff = [g for g in avail if g != assignment[a]]
                assign(b, diff[0] if diff else avail[0])
            else:
                same = [g for g in avail if g == assignment[a]]
                assign(b, same[0] if same else avail[0])
        elif a_done and not b_done:
            if maximize:
                diff = [g for g in avail if g != assignment[a]]
                assign(b, diff[0] if diff else avail[0])
            else:
                same = [g for g in avail if g == assignment[a]]
                assign(b, same[0] if same else avail[0])
        elif b_done and not a_done:
            if maximize:
                diff = [g for g in avail if g != assignment[b]]
                assign(a, diff[0] if diff else avail[0])
            else:
                same = [g for g in avail if g == assignment[b]]
                assign(a, same[0] if same else avail[0])

    for e in range(num_experts):
        if e not in assignment:
            avail = available()
            if avail:
                assign(e, avail[0])
            else:
                g = min(range(num_gpus), key=lambda g: gpu_load[g])
                assign(e, g)
                gpu_load[g] += 1
    return assignment


def _cross_gpu_rate(
    assignment: dict[int, int],
    counts: dict[tuple[int, int], float],
) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    cross = sum(cnt for (a, b), cnt in counts.items() if assignment.get(a) != assignment.get(b))
    return cross / total


def compute_placement_scenarios(report, num_gpus: int) -> dict:
    """
    Given a plumb ProfileReport, compute expected cross-GPU dispatch rates for
    three expert-placement strategies (worst / random / optimized) using the
    co-activation pairs collected during profiling.  No extra GPU time required.
    """
    if report is None:
        return {}
    coact = getattr(report, "coactivation", None)
    if coact is None:
        return {}
    coact_layers = getattr(coact, "layers", None) or []
    if not coact_layers:
        return {}

    layer_worst: list[float] = []
    layer_rand: list[float] = []
    layer_opt: list[float] = []

    for lc in coact_layers:
        pairs_raw = getattr(lc, "top_misplaced_pairs", None) or []
        if not pairs_raw:
            continue
        all_ids: set[int] = set()
        counts: dict[tuple[int, int], float] = {}
        for p in pairs_raw:
            a = getattr(p, "expert_a", None)
            b = getattr(p, "expert_b", None)
            cnt = float(getattr(p, "coactivation_count", 0) or 0)
            if a is None or b is None:
                continue
            all_ids |= {a, b}
            key = (min(a, b), max(a, b))
            counts[key] = counts.get(key, 0) + cnt
        if not all_ids:
            continue
        n = max(all_ids) + 1
        actual_gpus = min(num_gpus, n)
        if actual_gpus < 2:
            continue
        sorted_pairs = sorted(counts.items(), key=lambda x: -x[1])

        worst_assign = _greedy_placement(sorted_pairs, n, actual_gpus, maximize=True)
        layer_worst.append(_cross_gpu_rate(worst_assign, counts))

        opt_assign = _greedy_placement(sorted_pairs, n, actual_gpus, maximize=False)
        layer_opt.append(_cross_gpu_rate(opt_assign, counts))

        rand_rates: list[float] = []
        experts = list(range(n))
        for _ in range(50):
            shuffled = experts[:]
            random.shuffle(shuffled)
            rand_assign = {e: (i * actual_gpus) // n for i, e in enumerate(shuffled)}
            rand_rates.append(_cross_gpu_rate(rand_assign, counts))
        layer_rand.append(statistics.mean(rand_rates))

    def _avg(vals: list[float]) -> float | None:
        return round(statistics.mean(vals), 4) if vals else None

    return {
        "num_gpus": num_gpus,
        "worst_cross_gpu_rate": _avg(layer_worst),
        "random_cross_gpu_rate": _avg(layer_rand),
        "optimized_cross_gpu_rate": _avg(layer_opt),
        "layers_analyzed": len(layer_worst),
    }


# ── Output helpers ────────────────────────────────────────────────────────────

def _serialize_report(report) -> dict:
    """Convert a ProfileReport to a plain dict for JSON serialisation."""
    try:
        return json.loads(report.model_dump_json())
    except Exception:
        try:
            return report.dict()
        except Exception:
            return {}


def _top3_hot_experts(report) -> list[dict]:
    """Return the 3 experts with highest token_count across all layers."""
    items: list[dict] = []
    if report is None:
        return items
    for layer in (report.layers or []):
        for e in (layer.experts or []):
            items.append({
                "layer_id": layer.layer_id,
                "expert_id": e.expert_id,
                "token_count": e.token_count,
                "activation_fraction": e.activation_fraction,
            })
    items.sort(key=lambda x: x["token_count"], reverse=True)
    return items[:3]


def _cross_gpu_dispatch_rate(report) -> float | None:
    if report is None:
        return None
    coact = getattr(report, "coactivation", None)
    if coact is None:
        return None
    return getattr(coact, "total_cross_gpu_coactivation_rate", None)


def _max_imbalance_ratio(report) -> float | None:
    if report is None or not report.layers:
        return None
    return max(la.imbalance_ratio for la in report.layers)


def _print_summary_table(
    phase1: list[dict],
    phase2: list[dict] | None,
    report,
    output_dir: Path,
) -> None:
    if not _rich:
        _print_summary_plain(phase1, phase2, report, output_dir)
        return

    _console.print()
    _console.rule("[bold cyan]Plumb ShareGPT MoE Benchmark — Results")

    # Imbalance summary
    max_imb = _max_imbalance_ratio(report)
    top3 = _top3_hot_experts(report)
    cgdr = _cross_gpu_dispatch_rate(report)
    if max_imb is not None:
        _console.print(f"  Max imbalance ratio   : [bold red]{max_imb:.3f}[/]")
    if cgdr is not None:
        _console.print(f"  Cross-GPU dispatch    : [bold yellow]{cgdr:.1%}[/]")
    if top3:
        _console.print("  Top-3 hot experts:")
        for item in top3:
            _console.print(
                f"    layer={item['layer_id']} expert={item['expert_id']} "
                f"tokens={item['token_count']} ({item['activation_fraction']:.1%})"
            )
    _console.print()

    table = Table(title="Throughput & Latency", show_header=True, header_style="bold magenta")
    table.add_column("Concurrency", style="cyan", justify="right")
    table.add_column("Phase", style="bold")
    table.add_column("Throughput (req/s)", justify="right")
    table.add_column("p50 TTFT (ms)", justify="right")
    table.add_column("p95 TTFT (ms)", justify="right")
    table.add_column("p99 TTFT (ms)", justify="right")

    p1_by_c = {s["concurrency"]: s for s in phase1}
    p2_by_c = {s["concurrency"]: s for s in phase2} if phase2 else {}
    all_concurrencies = sorted(p1_by_c)

    for c in all_concurrencies:
        s1 = p1_by_c.get(c, {})
        table.add_row(
            str(c), "Baseline",
            f"{s1.get('throughput_rps', 0):.2f}",
            f"{s1.get('p50_ms', 0):.1f}",
            f"{s1.get('p95_ms', 0):.1f}",
            f"{s1.get('p99_ms', 0):.1f}",
        )
        if p2_by_c:
            s2 = p2_by_c.get(c, {})
            t1 = s1.get("throughput_rps", 0) or 1
            t2 = s2.get("throughput_rps", 0)
            gain = (t2 - t1) / t1 * 100
            gain_str = f"[green]+{gain:.1f}%[/]" if gain >= 0 else f"[red]{gain:.1f}%[/]"
            table.add_row(
                "", f"Plumb {gain_str}",
                f"{t2:.2f}",
                f"{s2.get('p50_ms', 0):.1f}",
                f"{s2.get('p95_ms', 0):.1f}",
                f"{s2.get('p99_ms', 0):.1f}",
            )

    _console.print(table)
    _console.print(f"\n  Output dir: [bold]{output_dir}[/]")
    _console.rule()


def _print_summary_plain(
    phase1: list[dict],
    phase2: list[dict] | None,
    report,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 72)
    print(" PLUMB SHAREGPT MOE BENCHMARK — RESULTS")
    print("=" * 72)
    max_imb = _max_imbalance_ratio(report)
    if max_imb is not None:
        print(f"  Max imbalance ratio : {max_imb:.3f}")
    cgdr = _cross_gpu_dispatch_rate(report)
    if cgdr is not None:
        print(f"  Cross-GPU dispatch  : {cgdr:.1%}")
    print()
    print(f"  {'Concurrency':>12}  {'Phase':12}  {'RPS':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}")
    print("  " + "-" * 65)
    p1_by_c = {s["concurrency"]: s for s in phase1}
    p2_by_c = {s["concurrency"]: s for s in phase2} if phase2 else {}
    for c in sorted(p1_by_c):
        s = p1_by_c[c]
        print(f"  {c:>12}  {'Baseline':12}  {s.get('throughput_rps',0):>8.2f}  "
              f"{s.get('p50_ms',0):>8.1f}  {s.get('p95_ms',0):>8.1f}  {s.get('p99_ms',0):>8.1f}")
        if c in p2_by_c:
            s2 = p2_by_c[c]
            print(f"  {'':>12}  {'Plumb':12}  {s2.get('throughput_rps',0):>8.2f}  "
                  f"{s2.get('p50_ms',0):>8.1f}  {s2.get('p95_ms',0):>8.1f}  {s2.get('p99_ms',0):>8.1f}")
    print(f"\n  Output dir: {output_dir}")
    print("=" * 72)


# ── Heterogeneous topology simulation ────────────────────────────────────────

def build_simulated_hetero_topology(num_gpus: int) -> object:
    """
    Build a HeterogeneousTopology representing a realistic mixed-GPU node.

    For num_gpus=4: 2x A100 PCIe (fast) + 2x RTX 3060 (slow).
    For num_gpus=2: 1x A100 PCIe + 1x RTX 3060.
    Relative compute scores are normalised so A100=1.0, RTX 3060≈0.31
    (reflecting SM-clock × compute-cap ratio: 1410×8.0 vs 1777×8.6 × VRAM penalty).

    This exercises plumb's heterogeneous placement analysis on real profiling
    data without requiring actual mixed-GPU hardware.
    """
    try:
        from numa_topology.gpu_capabilities import GPUCapability, HeterogeneousTopology
    except ImportError:
        print("  WARNING: numa_topology.gpu_capabilities not available — "
              "skipping hetero sim.", file=sys.stderr)
        return None

    a100_score = 1.0
    rtx3060_score = round((1777 * 8.6) / (1410 * 8.0), 4)  # ~1.357 unnorm → norm below

    # Normalise: fastest gets 1.0
    max_score = max(a100_score, rtx3060_score)
    a100_norm = round(a100_score / max_score, 4)
    rtx_norm = round(rtx3060_score / max_score, 4)

    fast_count = max(1, num_gpus // 2)
    slow_count = num_gpus - fast_count

    gpus = []
    for i in range(fast_count):
        gpus.append(GPUCapability(
            index=i, name="A100 PCIe 40GB",
            memory_total_mib=40960, memory_free_mib=36000,
            compute_cap="8.0", max_sm_clock_mhz=1410, max_mem_clock_mhz=1215,
            relative_compute_score=a100_norm,
        ))
    for i in range(slow_count):
        gpus.append(GPUCapability(
            index=fast_count + i, name="RTX 3060",
            memory_total_mib=12288, memory_free_mib=10000,
            compute_cap="8.6", max_sm_clock_mhz=1777, max_mem_clock_mhz=937,
            relative_compute_score=rtx_norm,
        ))

    return HeterogeneousTopology(
        gpus=gpus,
        is_homogeneous=False,
        mixed_vendor=False,
        compute_score_range=(min(a100_norm, rtx_norm), max(a100_norm, rtx_norm)),
    )


def run_hetero_analysis(report, hetero_topo, counter, model_name: str,
                        duration: float, num_gpus: int) -> dict:
    """Re-run generate_report with hetero_topology; return serialised result."""
    try:
        from plumb.report.generator import generate_report
        from plumb.topology import Topology
    except ImportError as exc:
        print(f"  WARNING: plumb not importable for hetero analysis: {exc}", file=sys.stderr)
        return {}

    if counter is None or hetero_topo is None:
        return {}

    try:
        hetero_report = generate_report(
            counter=counter,
            topology=Topology.flat(num_gpus),
            model_name=model_name,
            duration_seconds=duration,
            num_gpus=num_gpus,
            hetero_topology=hetero_topo,
        )
        result = _serialize_report(hetero_report)

        # Extract key hetero fields for summary
        ht = getattr(hetero_report, "heterogeneous_topology", None)
        hp = getattr(hetero_report, "heterogeneous_placement", None)
        violations = getattr(hp, "violations", []) if hp else []
        print(f"  Hetero sim: {len(hetero_topo.gpus)} GPUs  "
              f"homogeneous={hetero_topo.is_homogeneous}  "
              f"compute_range={hetero_topo.compute_score_range}")
        print(f"  Placement violations: {len(violations)}")
        for v in violations[:3]:
            layer = getattr(v, "layer_id", "?")
            expert = getattr(v, "expert_id", "?")
            gpu = getattr(v, "current_gpu", "?")
            rec = getattr(v, "recommended_gpu", "?")
            print(f"    layer={layer} expert={expert}: gpu{gpu}→gpu{rec}")
        return result
    except Exception as exc:
        print(f"  WARNING: hetero analysis failed: {exc}", file=sys.stderr)
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Guard: vllm must be importable ────────────────────────────────────────
    try:
        import vllm  # noqa: F401
    except ImportError:
        print(
            "ERROR: vllm is not installed.\n"
            "Install it with: pip install vllm\n"
            "See https://docs.vllm.ai/en/latest/getting_started/installation.html",
            file=sys.stderr,
        )
        sys.exit(1)

    if _requests_lib is None:
        print("ERROR: requests library not installed. Run: pip install requests", file=sys.stderr)
        sys.exit(1)

    # ── CLI ───────────────────────────────────────────────────────────────────
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    help="HuggingFace model ID (default: Mixtral-8x7B-Instruct-v0.1)")
    ap.add_argument("--tp", type=int, default=2,
                    help="Tensor-parallel size for the benchmark server (default: 2)")
    ap.add_argument("--concurrency", default="1,4,16,64",
                    help="Comma-separated concurrency levels (default: 1,4,16,64)")
    ap.add_argument("--num-requests", type=int, default=500,
                    help="Number of ShareGPT prompts to sample for benchmarking (default: 500)")
    ap.add_argument("--output-dir", default=None,
                    help="Directory to write results (default: bench/results/sharegpt-DATE)")
    ap.add_argument("--sharegpt-path", default=None,
                    help="Path to local ShareGPT JSON. Downloads if omitted.")
    ap.add_argument("--skip-phase2", action="store_true",
                    help="Skip phase 2 (placement benchmark). OSS-only mode.")
    ap.add_argument("--port", type=int, default=8100,
                    help="Port for the vLLM OpenAI-compatible server (default: 8100)")
    ap.add_argument("--warmup", type=int, default=40,
                    help="Warmup requests before each measurement phase (default: 40)")
    ap.add_argument("--num-profile", type=int, default=200,
                    help="Requests for the plumb profiling pass (default: 200)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument(
        "--hetero-sim", action="store_true",
        help="Simulate a heterogeneous topology (2x A100 + 2x RTX 3060) for the "
             "plumb heterogeneous placement analysis. Exercises the hetero codepath "
             "without needing real mixed-GPU hardware.",
    )
    args = ap.parse_args()

    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",")]

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"bench/results/mixtral-sharegpt-{timestamp}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"http://localhost:{args.port}"

    print("\n" + "=" * 72)
    print(" SHAREGPT MOE BENCHMARK — plumb + Mixtral-8x7B")
    print(f"  Model       : {args.model}")
    print(f"  TP size     : {args.tp}")
    print(f"  Concurrency : {concurrency_levels}")
    print(f"  Requests    : {args.num_requests}")
    print(f"  Output dir  : {output_dir}")
    print("=" * 72 + "\n")

    # ── Step 1: Load ShareGPT data ────────────────────────────────────────────
    print("[1/5] Loading ShareGPT prompts...")
    prompts = load_sharegpt_prompts(
        path=Path(args.sharegpt_path) if args.sharegpt_path else None,
        num_requests=args.num_requests,
        cache_dir=output_dir,
    )
    if not prompts:
        print("ERROR: no usable prompts loaded. Aborting.", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: Plumb profiling pass (TP=1, separate LLM instance) ───────────
    print("\n[2/5] Running plumb profiling pass (TP=1 LLM)...")
    report, _profile_counter, _profile_duration = run_profiling_pass(
        model_name=args.model,
        prompts=prompts,
        num_profile=args.num_profile,
    )

    report_dict = _serialize_report(report) if report else {}
    (output_dir / "plumb_report.json").write_text(
        json.dumps(report_dict, indent=2, default=str)
    )
    print(f"  Plumb report saved → {output_dir / 'plumb_report.json'}")
    if report:
        summ = report.summary()
        print(f"  Layers profiled     : {summ.get('num_layers_profiled', 0)}")
        print(f"  Mean imbalance ratio: {summ.get('mean_imbalance_ratio', 0):.3f}")
        print(f"  Max imbalance ratio : {summ.get('max_imbalance_ratio', 0):.3f}")
        print(f"  Worst layer         : {summ.get('worst_layer_id', 'n/a')}")

    # ── Heterogeneous simulation ──────────────────────────────────────────────
    hetero_report_dict: dict = {}
    if args.hetero_sim:
        print("\n  [hetero-sim] Simulating mixed-GPU topology "
              f"(2×A100 + 2×RTX 3060, {args.tp} slots)...")
        hetero_topo = build_simulated_hetero_topology(args.tp)
        hetero_report_dict = run_hetero_analysis(
            report=report, hetero_topo=hetero_topo,
            counter=_profile_counter,
            model_name=args.model,
            duration=_profile_duration or 1.0,
            num_gpus=args.tp,
        )
        if hetero_report_dict:
            (output_dir / "hetero_report.json").write_text(
                json.dumps(hetero_report_dict, indent=2, default=str)
            )
            print(f"  Hetero report saved → {output_dir / 'hetero_report.json'}")

    placement_scenarios = compute_placement_scenarios(report, num_gpus=args.tp)
    if placement_scenarios.get("layers_analyzed", 0) > 0:
        w = placement_scenarios.get("worst_cross_gpu_rate")
        r = placement_scenarios.get("random_cross_gpu_rate")
        o = placement_scenarios.get("optimized_cross_gpu_rate")
        print(f"  Placement scenarios ({args.tp} GPUs, "
              f"{placement_scenarios['layers_analyzed']} layers):")
        if w is not None:
            print(f"    Worst     cross-GPU dispatch: {w:.1%}")
        if r is not None:
            print(f"    Random    cross-GPU dispatch: {r:.1%}")
        if o is not None:
            print(f"    Optimized cross-GPU dispatch: {o:.1%}")
        if w and o:
            print(f"    Headroom  (worst→opt)       : -{(w - o) / w * 100:.1f}%")

    # ── Step 3: Phase 1 — Baseline benchmark ─────────────────────────────────
    print("\n[3/5] Phase 1 — Baseline benchmark...")
    max_c = max(concurrency_levels)
    proc = start_server(
        model=args.model, tp=args.tp, port=args.port,
        output_dir=output_dir, max_num_seqs=max_c * 2,
    )
    print(f"  Server PID {proc.pid} — waiting for ready (timeout 600s)...", flush=True)
    try:
        if not wait_for_server(base_url):
            print("ERROR: server did not become ready (Phase 1). Check vllm_server.log.",
                  file=sys.stderr)
            stop_server(proc)
            sys.exit(1)
        print("  Server ready.")
        phase1_stats = run_concurrent_phase(
            base_url=base_url,
            model=args.model,
            prompts=prompts,
            concurrency_levels=concurrency_levels,
            label="Baseline",
            warmup=args.warmup,
        )
    finally:
        stop_server(proc)
        time.sleep(5)

    phase1_out = {
        "model": args.model,
        "tp": args.tp,
        "timestamp": timestamp,
        "concurrency_results": phase1_stats,
    }
    (output_dir / "phase1_benchmark.json").write_text(json.dumps(phase1_out, indent=2))
    print(f"  Phase 1 results saved → {output_dir / 'phase1_benchmark.json'}")

    # ── Step 4: Phase 2 — Recommended placement ───────────────────────────────
    phase2_stats: list[dict] | None = None
    placement_env: dict = {}

    if args.skip_phase2:
        print("\n[4/5] Phase 2 skipped (--skip-phase2).")
    else:
        print("\n[4/5] Phase 2 — Recommended placement benchmark...")
        placement_report = getattr(report, "placement", None) if report else None
        if placement_report is None:
            print("  WARNING: no placement recommendation from plumb — "
                  "running Phase 2 with identical (default) placement.")
        else:
            est = getattr(placement_report, "estimated_improvement_pct", None)
            if est is not None:
                print(f"  Estimated improvement from plumb: {est:.1f}%")
            # Build VLLM_EXPERT_MAP env var from placement report
            raw_ep = getattr(placement_report, "expert_placement", {})
            # raw_ep keys: "layer_id:expert_id" → list of GPU indices
            mapping: dict[str, dict[str, int]] = {}
            for key, gpus in raw_ep.items():
                if ":" in key and gpus:
                    lid_str, eid_str = key.split(":", 1)
                    mapping.setdefault(lid_str, {})[eid_str] = gpus[0]
            if mapping:
                placement_env = {"VLLM_EXPERT_MAP": json.dumps(mapping, separators=(",", ":"))}
                print(f"  Placement covers {sum(len(v) for v in mapping.values())} expert slots "
                      f"across {len(mapping)} layers.")
            else:
                print("  WARNING: expert_placement is empty — Phase 2 uses default placement.")

        proc2 = start_server(
            model=args.model, tp=args.tp, port=args.port,
            output_dir=output_dir, extra_env=placement_env or None,
            max_num_seqs=max_c * 2,
        )
        print(f"  Server PID {proc2.pid} — waiting for ready...", flush=True)
        try:
            if not wait_for_server(base_url):
                print("ERROR: server did not become ready (Phase 2). Check vllm_server.log.",
                      file=sys.stderr)
                stop_server(proc2)
            else:
                print("  Server ready.")
                phase2_stats = run_concurrent_phase(
                    base_url=base_url,
                    model=args.model,
                    prompts=prompts,
                    concurrency_levels=concurrency_levels,
                    label="Plumb",
                    warmup=args.warmup,
                )
        finally:
            stop_server(proc2)
            time.sleep(5)

        if phase2_stats:
            phase2_out = {
                "model": args.model,
                "tp": args.tp,
                "timestamp": timestamp,
                "placement_applied": bool(placement_env),
                "concurrency_results": phase2_stats,
            }
            (output_dir / "phase2_benchmark.json").write_text(json.dumps(phase2_out, indent=2))
            print(f"  Phase 2 results saved → {output_dir / 'phase2_benchmark.json'}")

    # ── Step 5: Write summary ─────────────────────────────────────────────────
    print("\n[5/5] Writing summary...")

    # Pick highest-concurrency result for headline numbers
    def _best_throughput(stats: list[dict] | None) -> float:
        if not stats:
            return 0.0
        valid = [s for s in stats if "throughput_rps" in s]
        return max((s["throughput_rps"] for s in valid), default=0.0)

    p1_rps = _best_throughput(phase1_stats)
    p2_rps = _best_throughput(phase2_stats) if phase2_stats else None
    improvement_pct = ((p2_rps - p1_rps) / p1_rps * 100) if (p2_rps and p1_rps) else None

    summary = {
        "model": args.model,
        "tp": args.tp,
        "timestamp": timestamp,
        "num_prompts": len(prompts),
        "phase1_throughput_rps": round(p1_rps, 3),
        "phase2_throughput_rps": round(p2_rps, 3) if p2_rps else None,
        "improvement_pct": round(improvement_pct, 2) if improvement_pct is not None else None,
        "cross_gpu_dispatch_rate": _cross_gpu_dispatch_rate(report),
        "max_imbalance_ratio": _max_imbalance_ratio(report),
        "top3_hot_experts": _top3_hot_experts(report),
        "plumb_estimated_improvement_pct": (
            getattr(getattr(report, "placement", None), "estimated_improvement_pct", None)
            if report else None
        ),
        "placement_scenarios": placement_scenarios if placement_scenarios else None,
        "hetero_sim_run": bool(hetero_report_dict),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  Summary saved → {output_dir / 'summary.json'}")

    # ── Print rich results table ──────────────────────────────────────────────
    _print_summary_table(phase1_stats, phase2_stats, report, output_dir)


if __name__ == "__main__":
    main()
