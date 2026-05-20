#!/usr/bin/env python3
"""
plot_results.py — Publication-quality charts for Mixtral-8x7B / ShareGPT benchmark.

Usage:
    python bench/charts/plot_results.py \\
        --results-dir bench/results/mixtral-sharegpt-DATE \\
        --output-dir  bench/charts/output/DATE \\
        [--style dark|light]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────

_STEEL_BLUE = "#2563eb"
_FOREST_GREEN = "#16a34a"
_WARM_RED = "#dc2626"
_LIGHT_GRAY = "#e5e7eb"
_MID_GRAY = "#9ca3af"
_DARK = "#111827"
_CREDIT = "plumb-oss — github.com/plumb-moe/plumb"

# ── rcParams (applied once at import time, overridable per figure) ─────────────

def _apply_style(dark: bool = False) -> None:
    bg = "#1a1a2e" if dark else "#ffffff"
    fg = "#e2e8f0" if dark else _DARK
    grid_color = "#374151" if dark else _LIGHT_GRAY

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "text.color": fg,
        "axes.labelcolor": fg,
        "xtick.color": fg,
        "ytick.color": fg,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": bg,
        "figure.facecolor": bg,
        "axes.grid": True,
        "grid.color": grid_color,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "axes.edgecolor": _MID_GRAY,
    })


# ── JSON loading helpers ──────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    """Load JSON file; return None and print a warning on failure."""
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"  [skip] {path.name} could not be parsed: {exc}")
        return None


def _credit(ax: plt.Axes) -> None:
    """Add a small plumb-oss credit line inside the axes."""
    ax.text(
        0.99, 0.01, _CREDIT,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7, color=_MID_GRAY, alpha=0.8,
    )


# ── Chart 1: Expert utilisation heatmap ──────────────────────────────────────

def chart_expert_utilization(plumb: dict, out_dir: Path) -> bool:
    """
    Heatmap of activation_fraction across all (layer, expert) cells.
    Top-3 hottest cells are marked with a star.
    """
    layers = plumb.get("layers")
    if not layers:
        print("  [skip] expert_utilization_heatmap.png — 'layers' missing in plumb_report.json")
        return False

    # Build matrix: rows = layer_id, cols = expert_id
    num_layers = max(l["layer_id"] for l in layers) + 1
    num_experts_per_layer = max(len(l.get("experts", [])) for l in layers)
    if num_experts_per_layer == 0:
        print("  [skip] expert_utilization_heatmap.png — no expert data in layers")
        return False
    num_experts = max(
        max((e["expert_id"] for e in l.get("experts", [])), default=0)
        for l in layers
    ) + 1

    matrix = np.full((num_layers, num_experts), np.nan)
    for layer in layers:
        lid = layer["layer_id"]
        for exp in layer.get("experts", []):
            eid = exp["expert_id"]
            matrix[lid, eid] = exp["activation_fraction"]

    # Balanced activation for Mixtral-8x7B (top-2 / 8 experts)
    balanced = 1.0 / num_experts

    # Top-3 hottest cells
    flat = [(matrix[r, c], r, c)
            for r in range(num_layers)
            for c in range(num_experts)
            if not np.isnan(matrix[r, c])]
    flat.sort(reverse=True)
    top3 = flat[:3]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Diverging: blue=cold, white=balanced, red=hot
    vmax = max(balanced * 3, np.nanmax(matrix))
    vmin = 0.0
    vcenter = balanced
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = plt.cm.RdBu_r

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm,
                   origin="upper", interpolation="nearest")

    # Star annotations on top-3 hot cells
    for rank, (val, r, c) in enumerate(top3):
        ax.text(c, r, "★", ha="center", va="center",
                fontsize=13, color="white", fontweight="bold",
                path_effects=[
                    matplotlib.patheffects.withStroke(linewidth=1.5, foreground="black")
                ])
        ax.text(c, r + 0.42, f"{val:.3f}", ha="center", va="top",
                fontsize=6.5, color="white", alpha=0.9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Activation fraction", fontsize=10)
    cbar.ax.axhline(balanced, color="black", lw=1.2, ls="--")
    cbar.ax.text(1.05, balanced, f"balanced\n({balanced:.3f})",
                 transform=cbar.ax.transData, va="center", fontsize=7.5, color=_DARK)

    ax.set_xlabel("Expert ID", fontsize=11)
    ax.set_ylabel("Layer ID", fontsize=11)
    ax.set_xticks(range(num_experts))
    ax.set_yticks(range(0, num_layers, max(1, num_layers // 16)))
    ax.set_title(
        "Expert Activation Imbalance — Mixtral-8x7B (ShareGPT workload)",
        fontsize=13, fontweight="bold", pad=12,
    )
    _credit(ax)
    fig.tight_layout()
    dest = out_dir / "expert_utilization_heatmap.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok]   {dest.name}")
    return True


# ── Chart 2: Co-activation heatmap ───────────────────────────────────────────

def chart_coactivation(plumb: dict, out_dir: Path) -> bool:
    coact = plumb.get("coactivation")
    if not coact:
        print("  [skip] coactivation_heatmap.png — 'coactivation' field absent in plumb_report.json")
        return False

    layers = coact.get("layers", [])
    if not layers:
        print("  [skip] coactivation_heatmap.png — coactivation.layers is empty")
        return False

    layer0 = layers[0]
    pairs = layer0.get("top_misplaced_pairs", [])
    if not pairs:
        print("  [skip] coactivation_heatmap.png — top_misplaced_pairs is empty for layer 0")
        return False

    # Determine expert count from pairs
    all_ids = set()
    for p in pairs:
        all_ids.add(p["expert_a"])
        all_ids.add(p["expert_b"])
    n = max(all_ids) + 1 if all_ids else 8

    mat = np.zeros((n, n))
    cross_gpu_mask = np.zeros((n, n), dtype=bool)

    for p in pairs:
        a, b = p["expert_a"], p["expert_b"]
        cnt = p["coactivation_count"]
        mat[a, b] += cnt
        mat[b, a] += cnt
        if p.get("cross_gpu"):
            cross_gpu_mask[a, b] = True
            cross_gpu_mask[b, a] = True

    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(mat, cmap="YlOrRd", aspect="equal", origin="upper",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Co-activation count", fontsize=10)

    # Border colour: green=same GPU, red=cross-GPU
    for r in range(n):
        for c in range(n):
            if r == c:
                continue
            color = _WARM_RED if cross_gpu_mask[r, c] else _FOREST_GREEN
            rect = mpatches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                linewidth=1.4, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(i) for i in range(n)])
    ax.set_yticklabels([str(i) for i in range(n)])
    ax.set_xlabel("Expert ID", fontsize=11)
    ax.set_ylabel("Expert ID", fontsize=11)
    ax.set_title(
        f"Expert Co-Activation Frequency — Layer {layer0.get('layer_id', 0)} (Mixtral-8x7B)",
        fontsize=12, fontweight="bold", pad=10,
    )

    # Legend for border colours
    same_patch = mpatches.Patch(edgecolor=_FOREST_GREEN, facecolor="none",
                                linewidth=1.5, label="Same GPU")
    cross_patch = mpatches.Patch(edgecolor=_WARM_RED, facecolor="none",
                                 linewidth=1.5, label="Cross-GPU")
    ax.legend(handles=[same_patch, cross_patch], loc="lower right",
              frameon=True, fontsize=9)
    _credit(ax)
    fig.tight_layout()
    dest = out_dir / "coactivation_heatmap.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok]   {dest.name}")
    return True


# ── Chart 3: Throughput comparison ───────────────────────────────────────────

def chart_throughput(phase1: dict | None, phase2: dict | None, out_dir: Path) -> bool:
    if not phase1:
        print("  [skip] throughput_comparison.png — phase1_benchmark.json not found")
        return False

    p1_results: list[dict] = phase1.get("concurrency_results", [])
    p1_valid = [r for r in p1_results if "throughput_rps" in r]
    if not p1_valid:
        print("  [skip] throughput_comparison.png — no valid concurrency_results in phase1")
        return False

    p2_results: list[dict] = (phase2 or {}).get("concurrency_results", [])
    p2_by_c = {r["concurrency"]: r for r in p2_results if "throughput_rps" in r}
    have_phase2 = bool(p2_by_c)

    concurrencies = sorted(r["concurrency"] for r in p1_valid)
    p1_tps = [next(r["throughput_rps"] for r in p1_valid if r["concurrency"] == c)
              for c in concurrencies]

    fig, ax = plt.subplots(figsize=(10, 5))

    if have_phase2:
        p2_tps = [p2_by_c[c]["throughput_rps"] if c in p2_by_c else 0.0
                  for c in concurrencies]

        x = np.arange(len(concurrencies))
        w = 0.35
        bars1 = ax.bar(x - w / 2, p1_tps, w, color=_STEEL_BLUE,
                       label="Phase 1 — Default EPLB", zorder=3)
        bars2 = ax.bar(x + w / 2, p2_tps, w, color=_FOREST_GREEN,
                       label="Phase 2 — Plumb-Optimized", zorder=3)

        for bar2, t1, t2 in zip(bars2, p1_tps, p2_tps):
            if t1 > 0 and t2 > 0:
                pct = (t2 - t1) / t1 * 100
                sign = "+" if pct >= 0 else ""
                color = _FOREST_GREEN if pct >= 0 else _WARM_RED
                ax.text(
                    bar2.get_x() + bar2.get_width() / 2,
                    bar2.get_height() + max(p1_tps + p2_tps) * 0.015,
                    f"{sign}{pct:.1f}%",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color=color,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in concurrencies])
        ax.set_title(
            "Throughput: Default EPLB vs Plumb-Optimized Placement",
            fontsize=12, fontweight="bold",
        )
        ax.legend(frameon=False, fontsize=10)
    else:
        # Single-phase: line chart
        ax.plot(concurrencies, p1_tps, color=_STEEL_BLUE, marker="o",
                linewidth=2, markersize=7, label="Phase 1 — Default EPLB")
        ax.set_xticks(concurrencies)
        ax.set_title(
            "Throughput vs. Concurrency — Default EPLB (Phase 1 only)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(frameon=False, fontsize=10)

    ax.set_xlabel("Concurrency (simultaneous requests)", fontsize=11)
    ax.set_ylabel("Throughput (req/s)", fontsize=11)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    _credit(ax)
    fig.tight_layout()
    dest = out_dir / "throughput_comparison.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok]   {dest.name}")
    return True


# ── Chart 4: TTFT CDF ─────────────────────────────────────────────────────────

def _lognormal_cdf_from_quantiles(
    p50: float, p95: float, p99: float, n_points: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a log-normal distribution to three quantiles and return (x, cdf) arrays.
    Uses scipy if available; falls back to a simple two-point estimate.
    """
    try:
        from scipy import stats as sp_stats  # type: ignore[import]
        from scipy.optimize import brentq  # type: ignore[import]

        # Fit mu / sigma using p50 and p95
        mu = np.log(p50)
        z95 = sp_stats.norm.ppf(0.95)
        sigma = (np.log(p95) - mu) / z95
        if sigma <= 0:
            raise ValueError("sigma <= 0")
        x = np.linspace(max(1.0, p50 * 0.1), p99 * 2.5, n_points)
        cdf = sp_stats.lognorm.cdf(x, s=sigma, scale=np.exp(mu))
        return x, cdf
    except Exception:
        # Minimal fallback: piecewise linear through three quantiles
        x = np.array([0.0, p50, p95, p99, p99 * 3])
        y = np.array([0.0, 0.50, 0.95, 0.99, 1.0])
        x_fine = np.linspace(0.0, p99 * 2.5, n_points)
        y_fine = np.interp(x_fine, x, y)
        return x_fine, y_fine


def chart_latency_cdf(phase1: dict | None, phase2: dict | None, out_dir: Path) -> bool:
    if not phase1:
        print("  [skip] latency_cdf.png — phase1_benchmark.json not found")
        return False

    p1_results = phase1.get("concurrency_results", [])
    p1_valid = [r for r in p1_results if "p50_ms" in r]
    if not p1_valid:
        print("  [skip] latency_cdf.png — no latency data in phase1_benchmark.json")
        return False

    # Use highest concurrency level
    p1_peak = max(p1_valid, key=lambda r: r["concurrency"])

    p2_peak = None
    if phase2:
        p2_valid = [r for r in phase2.get("concurrency_results", []) if "p50_ms" in r]
        if p2_valid:
            p2_peak = max(p2_valid, key=lambda r: r["concurrency"])

    fig, ax = plt.subplots(figsize=(10, 5))

    series = [("Phase 1 — Default EPLB", p1_peak, _STEEL_BLUE)]
    if p2_peak:
        series.append(("Phase 2 — Plumb-Optimized", p2_peak, _FOREST_GREEN))

    for label, row, color in series:
        p50 = row.get("p50_ms", 0.0)
        p95 = row.get("p95_ms", 0.0)
        p99 = row.get("p99_ms", 0.0)

        if p50 <= 0 or p95 <= 0 or p99 <= 0:
            print(f"  [warn] latency_cdf — missing percentiles for {label}, skipping series")
            continue

        x, cdf = _lognormal_cdf_from_quantiles(p50, p95, p99)
        ax.plot(x, cdf, color=color, linewidth=2, label=label)

        # Mark p50 / p95 / p99 with vertical dotted lines
        for pval, ptxt in [(p50, "p50"), (p95, "p95"), (p99, "p99")]:
            ax.axvline(pval, color=color, linestyle=":", linewidth=1.2, alpha=0.7)
            ax.text(pval + max(x) * 0.005, 0.05, f"{ptxt}\n{pval:.0f}ms",
                    fontsize=7.5, color=color, va="bottom", alpha=0.9)

    ax.set_xlabel("Time to First Token — TTFT (ms)", fontsize=11)
    ax.set_ylabel("Cumulative fraction", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "TTFT CDF at Peak Concurrency — Default vs Optimized",
        fontsize=12, fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=10)

    # Note about synthetic CDF
    ax.text(
        0.01, 0.98,
        "Note: CDF synthesised from p50/p95/p99 via log-normal fit",
        transform=ax.transAxes, fontsize=7.5, color=_MID_GRAY,
        va="top", style="italic",
    )
    _credit(ax)
    fig.tight_layout()
    dest = out_dir / "latency_cdf.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok]   {dest.name}")
    return True


# ── Chart 5: Cross-GPU dispatch rate by layer ─────────────────────────────────

def chart_dispatch_rate(plumb: dict, phase2_plumb: dict | None, out_dir: Path) -> bool:
    """
    Per-layer cross-NUMA/cross-GPU dispatch rate.
    Tries communication_cost field first; falls back to layers[*].cross_numa_rate.
    """
    layers = plumb.get("layers", [])
    comm_cost = plumb.get("communication_cost")

    # Prefer cross_numa_rate from each LayerReport (direct field)
    layer_rates: list[tuple[int, float]] = []
    for layer in layers:
        rate = layer.get("cross_numa_rate")
        if rate is not None:
            layer_rates.append((layer["layer_id"], float(rate)))

    if not layer_rates and not comm_cost:
        print("  [skip] dispatch_rate_comparison.png — "
              "neither 'communication_cost' nor per-layer 'cross_numa_rate' found")
        return False

    if not layer_rates:
        # Scalar summary only — show a single-bar summary chart instead
        fig, ax = plt.subplots(figsize=(7, 4))
        current = comm_cost.get("current_overhead_us", 0)
        recommended = comm_cost.get("recommended_overhead_us", 0)

        bars = ax.bar(
            ["Current placement", "Plumb recommended"],
            [current, recommended],
            color=[_STEEL_BLUE, _FOREST_GREEN],
            width=0.4, zorder=3,
        )
        for bar, v in zip(bars, [current, recommended]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{v:.0f} µs", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Cross-GPU overhead (µs)", fontsize=11)
        ax.set_title("Cross-GPU Dispatch Rate by Layer", fontsize=12, fontweight="bold")
        caveat = comm_cost.get("caveat")
        if caveat:
            ax.text(0.5, -0.14, f"Note: {caveat}", transform=ax.transAxes,
                    ha="center", fontsize=8, color=_MID_GRAY, style="italic")
        _credit(ax)
        fig.tight_layout()
        dest = out_dir / "dispatch_rate_comparison.png"
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [ok]   {dest.name}")
        return True

    # Per-layer bar chart
    layer_rates.sort(key=lambda t: t[0])
    layer_ids = [t[0] for t in layer_rates]
    rates_before = [t[1] for t in layer_rates]

    # Phase 2 rates if available
    rates_after: list[float] | None = None
    if phase2_plumb:
        p2_layers = phase2_plumb.get("layers", [])
        p2_map = {l["layer_id"]: l.get("cross_numa_rate") for l in p2_layers}
        candidate = [p2_map.get(lid) for lid in layer_ids]
        if any(v is not None for v in candidate):
            rates_after = [v if v is not None else 0.0 for v in candidate]

    x = np.arange(len(layer_ids))
    fig, ax = plt.subplots(figsize=(14, 5))

    if rates_after:
        w = 0.4
        ax.bar(x - w / 2, rates_before, w, color=_STEEL_BLUE,
               label="Before (default placement)", zorder=3)
        ax.bar(x + w / 2, rates_after, w, color=_FOREST_GREEN,
               label="After (Plumb placement)", zorder=3)
        ax.legend(frameon=False, fontsize=10)
    else:
        ax.bar(x, rates_before, color=_STEEL_BLUE,
               label="Cross-NUMA dispatch rate", zorder=3)
        ax.legend(frameon=False, fontsize=10)

    ax.set_xticks(x[::max(1, len(x) // 16)])
    ax.set_xticklabels([str(layer_ids[i]) for i in range(0, len(layer_ids), max(1, len(layer_ids) // 16))])
    ax.set_xlabel("Layer ID", fontsize=11)
    ax.set_ylabel("Cross-GPU dispatch rate", fontsize=11)
    ax.set_title("Cross-GPU Dispatch Rate by Layer", fontsize=12, fontweight="bold")
    _credit(ax)
    fig.tight_layout()
    dest = out_dir / "dispatch_rate_comparison.png"
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok]   {dest.name}")
    return True


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--results-dir", required=True,
        help="Directory containing phase1_benchmark.json, phase2_benchmark.json, "
             "plumb_report.json, summary.json",
    )
    ap.add_argument(
        "--output-dir", required=True,
        help="Directory to write PNG charts (created if absent)",
    )
    ap.add_argument(
        "--style", choices=["light", "dark"], default="light",
        help="Color scheme: light (default, best for blog posts) or dark",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"ERROR: results directory does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    _apply_style(dark=(args.style == "dark"))

    # Suppress matplotlib font warnings that clutter output
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    print(f"\nplumb-oss chart generator")
    print(f"  Results : {results_dir}")
    print(f"  Output  : {out_dir}")
    print(f"  Style   : {args.style}")
    print()

    plumb = _load_json(results_dir / "plumb_report.json") or {}
    phase1 = _load_json(results_dir / "phase1_benchmark.json")
    phase2 = _load_json(results_dir / "phase2_benchmark.json")
    # phase2 plumb report is not produced by the bench script today; reserved for future use
    phase2_plumb: dict | None = None

    generated: list[str] = []
    skipped: list[str] = []

    # ── Chart 1 ───────────────────────────────────────────────────────────────
    if chart_expert_utilization(plumb, out_dir):
        generated.append("expert_utilization_heatmap.png")
    else:
        skipped.append("expert_utilization_heatmap.png")

    # ── Chart 2 ───────────────────────────────────────────────────────────────
    if chart_coactivation(plumb, out_dir):
        generated.append("coactivation_heatmap.png")
    else:
        skipped.append("coactivation_heatmap.png")

    # ── Chart 3 ───────────────────────────────────────────────────────────────
    if chart_throughput(phase1, phase2, out_dir):
        generated.append("throughput_comparison.png")
    else:
        skipped.append("throughput_comparison.png")

    # ── Chart 4 ───────────────────────────────────────────────────────────────
    if chart_latency_cdf(phase1, phase2, out_dir):
        generated.append("latency_cdf.png")
    else:
        skipped.append("latency_cdf.png")

    # ── Chart 5 ───────────────────────────────────────────────────────────────
    if chart_dispatch_rate(plumb, phase2_plumb, out_dir):
        generated.append("dispatch_rate_comparison.png")
    else:
        skipped.append("dispatch_rate_comparison.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(f"Generated ({len(generated)}):")
    for name in generated:
        print(f"  {out_dir / name}")
    if skipped:
        print(f"\nSkipped ({len(skipped)}) — data not present:")
        for name in skipped:
            print(f"  {name}")
    print()


if __name__ == "__main__":
    main()
