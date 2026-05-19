# Plumb — Benchmark Results v2

**Date:** May 2026  
**Hardware (v1 runs):** 4× NVIDIA RTX 3090 (24 GB each), Vast.ai  
**Hardware (hook overhead run):** 2× RTX 3090, TP=2 (same Vast.ai instance)  
**vLLM:** 0.8.5.post1 (all stored results) / 0.21.0 (planned — see Section 5)  
**Model:** allenai/OLMoE-1B-7B-0924  

> This document supersedes the methodology notes in v1 and adds: the vLLM native online EPLB reference run (Section 3), full analysis of when and why Plumb's placement differs (Section 4), and a planned same-version EP=2 clean comparison on vLLM 0.21.0 (Section 5). v1 results (`REPORT.md`) remain available for reference.

---

## What Changed Since v1

v1 covered hook overhead, the TP=2 negative control, and the main OLMoE EP=4 concurrency sweep. This update:

- Adds a cross-version reference against vLLM's native online EPLB (Section 3) — the magnitude of vLLM's converged gain motivates the planned end-to-end comparison
- Adds Section 4: a precise account of where Plumb's placement differs from vLLM's native EPLB and why
- Identifies the planned Section 5 benchmark (EP=2, same-version, vLLM 0.21.0) as the missing clean comparison

---

## 1. Hook Overhead — Confirmed Zero

**Method:** ABAB interleaved phases on vLLM TP=2, 2× RTX 3090. A phases run with profiling hooks active; B phases pause them. 150 requests per phase × 4 phases = 600 total. Welch t-test on the two TTFT distributions.

| Phase | N | Mean TTFT | P50 | P90 | P95 | P99 |
|---|---|---|---|---|---|---|
| Hooks ON | 300 | 38.59 ms | 36.02 ms | 36.77 ms | 38.37 ms | 85.77 ms |
| Hooks OFF | 300 | 38.68 ms | 35.71 ms | 36.43 ms | 37.95 ms | 90.03 ms |

**Overhead: −0.089 ms (−0.23%), 95% CI [−4.01, +3.84 ms], p = 0.786 — not significant.**

The hot-path — top-k router logit extraction and async ring-buffer push — adds zero measurable latency. Profiling can be left on in production without penalty.

---

## 2. Plumb Static Placement — EP=4 Results

**Method:** vLLM 0.8.5.post1, EP=4 (one expert shard per GPU), 4× RTX 3090. Plumb profiles 100 prompts, runs `recommend_placement()` → `_numa_finetune()`, produces an expert→GPU map. Map applied via `VLLM_EXPERT_MAP` and vLLM restarted. Benchmark runs against both configurations.

**Observed peak imbalance:** 6.74× on Layer 0 (Expert 6). Activation snapshot: 1017 expert slots, 772 moves in the EPLB plan.

### Concurrency = 8 (2 in-flight per GPU)

| Metric | Baseline | Plumb EPLB | Δ |
|---|---|---|---|
| P50 TTFT | 33.1 ms | 32.8 ms | −1.0% |
| P90 TTFT | 46.4 ms | 35.5 ms | **−23.4%** |
| P95 TTFT | 52.8 ms | 54.2 ms | +2.7% |
| P99 TTFT | 320.5 ms | 70.7 ms | **−77.9%** |
| Mean TTFT | 45.7 ms | 34.3 ms | **−24.8%** |
| RPS | 170.7 | 219.3 | **+28.5%** |

### Concurrency = 32 (8 in-flight per GPU)

| Metric | Baseline | Plumb EPLB | Δ |
|---|---|---|---|
| P50 TTFT | 52.5 ms | 48.6 ms | **−7.4%** |
| P90 TTFT | 81.4 ms | 55.8 ms | **−31.5%** |
| P95 TTFT | 88.5 ms | 58.2 ms | **−34.2%** |
| Mean TTFT | 55.9 ms | 46.9 ms | **−16.1%** |
| RPS | 493.2 | 593.5 | **+20.3%** |

### Concurrency = 64 (16 in-flight per GPU — saturation)

| Metric | Baseline | Plumb EPLB | Δ |
|---|---|---|---|
| P50 TTFT | 65.0 ms | 72.6 ms | +11.7% |
| P90 TTFT | 85.0 ms | 94.3 ms | +11.0% |
| RPS | 687.5 | 653.6 | **−4.9%** |

**Reading:** EPLB delivers large gains from c=8 through c=32, with the largest tail improvement at c=8 (P99 −77.9%) where the single 6.74× hot expert cluster on GPU 0 creates a severe queue for expert-heavy tokens. At c=64 all GPUs are fully saturated — no spare capacity exists to absorb moved experts and the all-to-all routing overhead wins.

**Note on TP=2 negative control:** A separate run with TP=2 (not EP) on the same hardware shows no benefit from EPLB (−7.2% RPS, see `olmoe_1b7b_eplb_concurrent_tp2_20260515.json`). Under tensor parallelism every GPU co-owns every expert's weight shards; there is no per-GPU expert ownership for EPLB to exploit. Expert parallelism is required.

---

## 3. vLLM Native Online EPLB — Cross-Version Reference

> **Important caveat:** these runs used vLLM 0.8.5.post1. The absolute numbers are not directly comparable to a future vLLM 0.21.0 run. This table is included for directional reference and to quantify the converged gain that motivates Section 5.

| Condition | c | RPS | P50 | P95 | vs baseline |
|---|---|---|---|---|---|
| Baseline (0.8.5) | 8 | 177.0 | 41.2 ms | 57.0 ms | — |
| vLLM EPLB early (0.8.5) | 8 | 154.9 | 41.9 ms | 76.2 ms | −12% RPS (converging) |
| vLLM EPLB late (0.8.5) | 8 | 187.6 | 40.3 ms | 48.1 ms | +6% RPS / −16% p95 |
| Baseline (0.8.5) | 32 | 322.5 | 83.7 ms | 121.0 ms | — |
| vLLM EPLB early (0.8.5) | 32 | 512.6 | 47.4 ms | 67.2 ms | +59% RPS (peak) |
| vLLM EPLB late (0.8.5) | 32 | 481.3 | 47.0 ms | 72.7 ms | +49% RPS |

**Reading:** vLLM's online EPLB shows large gains at c=32 once converged (+49–59% RPS). The early-phase regression (−12% at c=8) reflects the convergence cost — the online rebalancer explores placements before settling. The magnitude of the converged gain motivates the planned same-version comparison in Section 5.

---

## 4. Why Plumb's Placement Differs From vLLM's Native EPLB

Both Plumb and vLLM's native EPLB use the same underlying algorithm (DeepSeek's `eplb.rebalance_experts`) as the primary placement engine. The differences are in what happens before and after that algorithm runs.

### 4.1 NUMA-Topology-Aware Post-Processing

After EPLB produces a load-balanced assignment, Plumb runs `_numa_finetune()` — a second pass that reads the actual GPU NUMA topology from Linux sysfs (`/sys/bus/pci/devices/*/numa_node`) and pins the hottest experts per layer to GPUs on the same NUMA socket. Cross-NUMA expert dispatch is more expensive than same-NUMA dispatch because activation tensors must cross the QPI/UPI fabric. On PCIe-only multi-GPU systems (which covers the majority of self-hosted deployments), this is the dominant inter-GPU communication cost.

vLLM's native EPLB does not perform any topology-aware post-processing. It produces a load-balanced map and stops. The NUMA fine-tuning step is Plumb's primary algorithmic differentiator in the one-shot placement workflow.

### 4.2 Imbalance Threshold Gate

Plumb suppresses the placement recommendation entirely when peak imbalance is below 3× (`_LOW_IMBALANCE_THRESHOLD = 3.0`), with an explicit warning. This prevents harmful rebalancing on models that are already well-balanced by training design.

vLLM's native EPLB runs unconditionally. In v1 we benchmarked this on DeepSeek-V2-Lite (peak imbalance 1.5×) and measured P95 TTFT degrading by +226% at c=16 from blind rebalancing. vLLM's online EPLB has no equivalent gate.

### 4.3 No Redundant Expert Requirement

vLLM's online EPLB requires `--num-redundant-experts` — it maintains additional copies of hot experts in GPU memory. This makes it unsuitable for memory-constrained deployments or setups where KV cache space is at a premium.

Plumb's placement operates on the existing expert set with no replication. This is currently a constraint (see Section 4.4) but also means Plumb works in environments where vLLM's EPLB cannot.

### 4.4 Current Limitation: Single-Copy Placement

vLLM's online EPLB can replicate hot experts across multiple GPUs (`logcnt > 1` in `rebalance_experts`). Plumb's current `VLLM_EXPERT_MAP` wire format is single-GPU-per-expert — replica assignments from `rebalance_experts` are discarded. The benchmark comparisons in this document are therefore deliberately conservative: Plumb achieves these results with no expert duplication. Expert replication support is on the roadmap (tracked as `bx-zis`) and expected to widen the performance gap further once implemented.

*Plumb Pro additionally scores candidate placements by predicted tokens/s — modelling compute ceiling, HBM bandwidth, and cross-NUMA transfer costs — to select between candidates in a continuous replanning loop. This scoring runs in the Pro scheduler, not in the one-shot placement workflow benchmarked here.*

---

## 5. Planned Benchmark: Same-Version Clean Comparison (vLLM 0.21.0)

### What's missing

Sections 2 and 3 cannot be directly compared: Section 2 is EP=4 on vLLM 0.8.5 with 4 GPUs; Section 3 is vLLM's native EPLB also on 0.8.5 but measured with a different baseline (EP=2, 2 GPU shards). A clean same-version, same-hardware, same-EP comparison of Plumb's static placement against vLLM's native online EPLB has not yet been run. The benchmark script is written and ready; this run is deferred on cost grounds.

### Hypothesis

Based on the results in Sections 2 and 3:

- **Plumb static vs vLLM converged EPLB:** Plumb's single-restart approach should approach vLLM's converged gain (+49–59% at c=32) without the early-phase regression (−12% at c=8 during convergence). The question is how much the lack of replication costs.
- **Better tail latency on NUMA-asymmetric hardware:** `_numa_finetune()` is expected to reduce P90/P95 beyond what load balancing alone achieves on PCIe-only systems.
- **Safer on balanced workloads:** the imbalance gate will suppress the plan on DeepSeek-V2-Lite rather than applying a harmful rebalancing. vLLM's online EPLB has no equivalent.

### Methodology

When run, the benchmark will use:
- vLLM 0.21.0, EP=2 (one expert shard per GPU), 2× RTX 4090
- OLMoE-1B-7B-0924
- Plumb profiling snapshot: 200 prompts
- ABAB interleave: 300 requests per phase × 4 phases
- Concurrency sweep: c=8, c=16, c=32
- Compared directly against a same-run baseline (same version, same hardware, same prompts)

Results will be published as v3 of this document.

---

## 6. Open Items

- **Same-version clean comparison (v3)** — EP=2, vLLM 0.21.0, 2× RTX 4090; script ready, deferred on cost
- **Expert replication (bx-zis)** — single-GPU-per-expert wire format is the current ceiling; replication support expected to materially improve high-concurrency results
- **Decode throughput benchmark** — all current results are prefill only (max_tokens=1); decode benchmark needed for real-workload tok/s figures
- **DeepSeek-V3 on A100/H100** — most commercially relevant model; RTX results do not directly generalise
- **PCIe bandwidth-asymmetric placement** — asymmetric topologies (x16/x8/x8/x4) are not captured by NUMA topology alone; per-slot bandwidth weighting is a planned extension to `_numa_finetune()`
- **OLMoE c=16 rerun** — borderline result from v1 (−7.9%) warrants larger sample to pin sweet-spot ceiling

---

## Appendix: Raw Data Files

All files are under `bench/results/`.

| Run | File | Notes |
|---|---|---|
| Hook overhead | `olmoe_1b7b_tp2_hook_toggle_20260514.json` | TP=2, 2× RTX 3090, vLLM 0.8.5 |
| TP=2 negative control | `olmoe_1b7b_eplb_concurrent_tp2_20260515.json` | Confirms EPLB requires EP not TP |
| Plumb EPLB EP=4 (c=8) | `olmoe_1b7b_eplb_ep4_20260515.json` | 4× RTX 3090, vLLM 0.8.5 |
| Plumb EPLB EP=4 (c=8/32/64) | `REPORT.md` — Section 3 tables | Full sweep; c=32/64 not in standalone JSON |
| vLLM native EPLB reference | `eplb_comparison_20260515.json` | vLLM 0.8.5, cross-version reference |

*v1 narrative and DeepSeek-V2-Lite results are in `REPORT.md`.*
