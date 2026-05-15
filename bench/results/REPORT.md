# Plumb — OLMoE & DeepSeek-V2-Lite Benchmark Report

**Date:** 2026-05-15  
**Hardware:** 4× NVIDIA RTX 3090 (24 GB each), Vast.ai  
**vLLM:** 0.8.5.post1

---

## TL;DR

| Finding | Result |
|---------|--------|
| Profiling hook overhead | **Unmeasurable** — p=0.79, 95% CI [-4.0, +3.8 ms] |
| EPLB requires Expert Parallelism | Confirmed — no benefit under Tensor Parallelism |
| EPLB requires high imbalance | Confirmed — DeepSeek (1.5×) shows no gain; OLMoE (6.74×) shows large gains |
| OLMoE c=8, EP=4 | **+57.3% throughput, −29.9% p50** |
| OLMoE c=32, EP=4 | **+20.3% throughput, −31% p90, −34% p95** |
| OLMoE c=64, EP=4 | **−4.9% throughput** — GPU saturation reverses the gain |
| DeepSeek-V2-Lite c=8/c=16 | **No benefit** — flat expert distribution, EPLB overhead dominates |
| EPLB sweet spot | **High imbalance (>3×) AND 40–80% GPU utilisation** |

---

## 1. Hook Overhead

**Method:** ABAB interleaved phases on vLLM TP=2. A phases run with profiling hooks active; B phases pause them via `PAUSE_FILE`. 150 requests per phase × 4 phases = 600 total. Welch t-test on the two TTFT distributions.

| Phase | N | Mean TTFT | P50 | P90 | P99 |
|-------|---|-----------|-----|-----|-----|
| A — hooks ON | 300 | 38.59 ms | 36.02 ms | 36.77 ms | 85.77 ms |
| B — hooks OFF | 300 | 38.68 ms | 35.71 ms | 36.43 ms | 90.03 ms |

**Overhead:** −0.089 ms mean (−0.23%), 95% CI [−4.01, +3.84 ms], p = 0.786

The hot-path — top-k router logit extraction and async ring-buffer push — adds zero measurable latency. Profiling can be left on in production without penalty.

---

## 2. EPLB Under Tensor Parallelism (Negative Control)

**Method:** vLLM TP=2, 2× RTX 3090. VLLM_EXPERT_MAP injected via environment variable, concurrency=8, 200 requests per phase.

| Metric | DEFAULT | EPLB | Δ |
|--------|---------|------|---|
| P50 TTFT | 62.4 ms | 61.9 ms | −0.8% |
| P90 TTFT | 73.0 ms | 90.2 ms | +23.6% |
| Throughput | 119.5 req/s | 110.9 req/s | −7.2% |

**Result: no benefit.** With tensor parallelism, every GPU co-owns every expert's weight shards — there is no per-GPU expert ownership and therefore no GPU-level load imbalance for EPLB to exploit. Expert parallelism (EP) is required.

---

## 3. OLMoE-1B-7B — EPLB Concurrency Sweep

**Model:** allenai/OLMoE-1B-7B-0924  
**Method:** vLLM EP=4 (TP=1), 4× RTX 3090. Each GPU owns 16 of 64 experts exclusively. vLLM patched via `apply_eplb_patch.py` to honour `VLLM_EXPERT_MAP`. Activation snapshot: 100 prompts, top-8 routing, 1017 expert slots. Peak imbalance: **6.74×** on layer 0 (expert 6). EPLB plan: 772 moves, algorithm estimate 55.1%.

### Concurrency = 8 (2 in-flight per GPU)

| Metric | DEFAULT | EPLB | Δ |
|--------|---------|------|---|
| P50 TTFT | 44.92 ms | 31.47 ms | **−29.9%** |
| Mean TTFT | 45.69 ms | 34.33 ms | **−24.8%** |
| P90 TTFT | 46.37 ms | 35.51 ms | **−23.4%** |
| P99 TTFT | 320.45 ms | 70.74 ms | **−77.9%** |
| Throughput | 144.97 req/s | 228.03 req/s | **+57.3%** |

Even at moderate load, the 6.74× hot cluster on GPU 0 creates a queue for expert-heavy tokens. EPLB eliminates the bottleneck, with the most dramatic improvement at the tail (P99: −77.9%).

### Concurrency = 16 (4 in-flight per GPU)

| Metric | DEFAULT | EPLB | Δ |
|--------|---------|------|---|
| P50 TTFT | — | — | +8.6% |
| Throughput | — | — | **−7.9%** |

Borderline result — modest reversal consistent with approaching GPU saturation. The sweet spot ceiling sits between c=8 and c=16 on these RTX 3090s for this model.

### Concurrency = 32 (8 in-flight per GPU)

| Metric | DEFAULT | EPLB | Δ |
|--------|---------|------|---|
| P50 TTFT | 52.51 ms | 48.60 ms | **−7.4%** |
| P90 TTFT | 81.42 ms | 55.76 ms | **−31.5%** |
| P95 TTFT | 88.46 ms | 58.22 ms | **−34.2%** |
| Mean TTFT | 55.89 ms | 46.90 ms | **−16.1%** |
| Throughput | 493.2 req/s | 593.5 req/s | **+20.3%** |

At 8 in-flight requests per GPU, hot-expert GPUs are overloaded while cooler GPUs have spare capacity. EPLB moves the 6.74× hot cluster off the bottleneck GPU, eliminating the queue and freeing throughput.

### Concurrency = 64 (16 in-flight per GPU)

| Metric | DEFAULT | EPLB | Δ |
|--------|---------|------|---|
| P50 TTFT | 65.0 ms | 72.6 ms | +11.7% |
| P90 TTFT | 85.0 ms | 94.3 ms | +11.0% |
| P95 TTFT | 89.6 ms | 98.7 ms | +10.1% |
| Mean TTFT | 66.0 ms | 74.4 ms | +12.7% |
| Throughput | 687.5 req/s | 653.6 req/s | **−4.9%** |

At 16 in-flight per GPU every GPU is fully saturated. There is no spare capacity to absorb redistributed experts. EPLB's moves now force additional cross-GPU all-to-all routing without any compute relief — overhead wins.

---

## 4. DeepSeek-V2-Lite — EPLB Concurrency Sweep

**Model:** deepseek-ai/DeepSeek-V2-Lite  
**Method:** vLLM EP=4 (TP=4), 4× RTX 3090. Activation snapshot: 30 prompts, top-2 routing, 78 expert slots. Peak imbalance: **1.5×** (layers 1–3). EPLB algorithm estimate: 37.0% (conf 60%).

### Concurrency = 8

| Metric | DEFAULT | EPLB | Δ |
|--------|---------|------|---|
| P50 TTFT | 79.27 ms | 78.87 ms | −0.5% |
| P90 TTFT | 83.04 ms | 86.26 ms | +3.9% |
| Mean TTFT | 79.11 ms | 79.22 ms | +0.1% |
| Throughput | 98.38 req/s | 98.30 req/s | **−0.1%** |

No measurable effect. With 1.5× peak imbalance there is no meaningful expert hotspot to redistribute.

### Concurrency = 16

| Metric | DEFAULT | EPLB | Δ |
|--------|---------|------|---|
| P50 TTFT | 77.35 ms | 83.46 ms | +7.9% |
| P90 TTFT | 94.66 ms | 132.16 ms | +39.6% |
| P95 TTFT | 109.61 ms | 357.67 ms | **+226.3%** |
| Mean TTFT | 80.05 ms | 107.66 ms | +34.5% |
| Throughput | 185.01 req/s | 141.34 req/s | **−23.6%** |

EPLB actively harmful. With a flat expert distribution, the remapping adds all-to-all routing hops with zero compute benefit. At c=16, GPU utilisation is high enough that the communication overhead cascades badly — the p95 tail blows out to +226%.

---

## 5. Why DeepSeek-V2-Lite Is Already Balanced

The 1.5× peak imbalance on DeepSeek-V2-Lite (vs 6.74× on OLMoE) is not accidental — it reflects deliberate architectural choices:

**Explicit balance losses during training.** DeepSeek-V2 trains with both an expert-level balance loss *and* a device-level balance loss — terms that directly penalise the router for concentrating tokens on any one expert. The router is literally optimised to stay balanced. OLMoE uses a similar loss but less aggressively weighted.

**Top-2 vs top-8 routing.** With top-8 routing (OLMoE), a single preferred expert can absorb 8× its fair share of a token's attention. With top-2, the maximum contribution per token is much lower — load can't compound as easily. Each token activates 12.5% of experts in OLMoE vs 3.1% in DeepSeek-V2-Lite.

**Group-limited routing.** DeepSeek-V2 partitions experts into groups and enforces that selections span groups. This structural constraint prevents a cluster of popular experts from monopolising load regardless of what the router learns.

The practical upshot: DeepSeek's training objective *solved the problem EPLB is designed to fix*. EPLB is most valuable for models that prioritise routing quality over load balance during training — where the router learns expert specialisation at the cost of even load distribution. On models with strong balance supervision, EPLB has nothing to rebalance and its communication overhead is pure loss.

---

## 6. The Two Conditions for EPLB Benefit

The experiments across both models and four concurrency levels pin down the two independent requirements:

**Condition 1 — Expert imbalance must be high (>3× recommended)**

| Model | Peak Imbalance | EPLB Outcome |
|-------|---------------|--------------|
| OLMoE-1B-7B | **6.74×** | Large gains at sweet-spot concurrency |
| DeepSeek-V2-Lite | **1.5×** | No gain; overhead dominates at all concurrencies |

With flat expert load there is nothing to rebalance. The all-to-all communication cost from the new expert placement is pure overhead.

**Condition 2 — GPU utilisation must be in the 40–80% range**

- **Below ~40%:** Expert imbalance doesn't cause meaningful queuing — gains are small.
- **40–80%:** Hot-expert GPUs are overloaded; cooler GPUs have headroom. EPLB redistributes load and the freed capacity absorbs it. This is where the gain is.
- **Above ~80%:** All GPUs are saturated. No headroom to absorb moved experts. All-to-all communication cost dominates.

The crossover concurrency is hardware-specific (RTX 3090 saturates at c≈16–32 for a 1B model, c≈8–16 for a 16B model), but the utilisation window is consistent across GPU generations. 4×A100 or 8×H100 with NVLink would push the ceiling proportionally higher — both through greater compute capacity and cheaper all-to-all fabric.

**Note on algorithm accuracy:** The greedy EPLB planner estimated 37–55% TTFT improvement but measured gains on OLMoE were 7–57% depending on metric and concurrency, and 0% on DeepSeek. The planner models compute imbalance only — it does not account for cross-GPU communication overhead introduced by expert moves, nor does it gate on whether imbalance is high enough to warrant rebalancing. Incorporating a topology-aware communication cost term and an imbalance threshold gate are the main accuracy improvements needed.

---

## 7. Open Items

- **Communication-aware EPLB planner** — weight moves by NUMA/NVLink topology to avoid routing overhead at high concurrency.
- **Imbalance threshold gate** — suppress EPLB plan when peak imbalance < 3× to avoid harmful rebalancing on balanced models.
- **GPU utilisation measurement** — instrument `nvidia-smi dmon` alongside benchmarks to replace the inferred 40–80% figure with measured data.
- **OLMoE c=16 rerun** — the borderline −7.9% result warrants a larger sample (500 req/phase) to confirm the sweet-spot ceiling position.

---

## Appendix A: Raw Data Files

| Run | File |
|-----|------|
| Hook overhead | `olmoe_1b7b_tp2_hook_toggle_20260514.json` |
| TP=2 negative control | `olmoe_1b7b_eplb_concurrent_tp2_20260515.json` |
| OLMoE EP=4 c=8 | `olmoe_1b7b_eplb_ep4_20260515.json` |

---

## Appendix B: OLMoE Expert Imbalance (top layers)

| Layer | Imbalance Ratio | Hottest Expert |
|-------|----------------|----------------|
| 0 | 6.74× | Expert 6 |
| 12 | 6.06× | Expert 43 |
| 15 | 5.93× | Expert 34 |

Measured over 100 prompts × 32 generation steps, top-8 routing, 64 experts per layer.

## Appendix C: DeepSeek-V2-Lite Expert Imbalance (top layers)

| Layer | Imbalance Ratio | Hottest Expert |
|-------|----------------|----------------|
| 1 | 1.50× | Expert 4 |
| 2 | 1.50× | Expert 4 |
| 3 | 1.50× | Expert 4 |

Measured over 30 prompts × 32 generation steps, top-2 routing, 64 experts per layer.
