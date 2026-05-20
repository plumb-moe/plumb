"""Generate benchmark charts for the plumb OSS README."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUT = ROOT / "charts"
OUT.mkdir(exist_ok=True)

BLUE = "#2563eb"
GRAY = "#9ca3af"
DARK = "#111827"
LIGHT_GRAY = "#f3f4f6"
RED = "#dc2626"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#e5e7eb",
    "grid.linewidth": 0.8,
    "grid.alpha": 1.0,
    "axes.axisbelow": True,
})


# ── Chart 1: Throughput sweep ─────────────────────────────────────────────────

def chart_throughput_sweep():
    ep4 = json.loads((RESULTS / "olmoe_1b7b_eplb_ep4_20260515.json").read_text())

    # data from REPORT.md (c=32 and c=64 are in text, c=8 is in JSON)
    data = {
        "c=8":  {"baseline": 170.7, "eplb": 219.3, "delta": +28.5},
        "c=32": {"baseline": 493.2, "eplb": 593.5, "delta": +20.3},
        "c=64": {"baseline": 687.5, "eplb": 653.6, "delta": -4.9},
    }

    labels = list(data.keys())
    baseline_vals = [data[k]["baseline"] for k in labels]
    eplb_vals = [data[k]["eplb"] for k in labels]
    deltas = [data[k]["delta"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars_b = ax.bar(x - width / 2, baseline_vals, width, color=GRAY, label="Baseline", zorder=3)
    bars_e = ax.bar(x + width / 2, eplb_vals, width, color=BLUE, label="Plumb EPLB", zorder=3)

    # delta annotations
    for bar, d in zip(bars_e, deltas):
        color = BLUE if d > 0 else RED
        sign = "+" if d > 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"{sign}{d}%",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=color,
        )

    y_max = max(eplb_vals) * 1.18
    ax.set_ylim(0, y_max)

    # sweet spot shading
    ax.axvspan(-0.5, 1.5, alpha=0.06, color=BLUE, zorder=0)
    ax.text(0.5, y_max * 0.97, "sweet spot",
            ha="center", va="top", fontsize=8.5, color=BLUE, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Requests / second", fontsize=11)
    ax.set_title("OLMoE EP=4: Throughput vs. Concurrency\n(4× RTX 3090, vLLM 0.8.5)", fontsize=12)
    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "throughput_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote throughput_sweep.png")


# ── Chart 2: Latency percentiles at c=8 and c=32 ─────────────────────────────

def chart_latency():
    # c=8 data from ep4 JSON; c=32 from REPORT.md
    c8_base = {"p50": 44.92, "p90": 46.37, "p99": 320.45}
    c8_eplb = {"p50": 31.47, "p90": 35.51, "p99": 70.74}
    c32_base = {"p50": 52.51, "p90": 81.42, "p99": None}  # p99 not in report for c=32
    c32_eplb = {"p50": 48.60, "p90": 55.76, "p99": None}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    for ax, (label, base, eplb) in zip(
        axes,
        [("Concurrency = 8", c8_base, c8_eplb),
         ("Concurrency = 32", c32_base, c32_eplb)],
    ):
        pcts = ["P50", "P90", "P99"]
        base_vals = [base["p50"], base["p90"], base["p99"] or 0]
        eplb_vals = [eplb["p50"], eplb["p90"], eplb["p99"] or 0]

        x = np.arange(len(pcts))
        w = 0.35
        b_bars = ax.bar(x - w / 2, base_vals, w, color=GRAY, label="Baseline", zorder=3)
        e_bars = ax.bar(x + w / 2, eplb_vals, w, color=BLUE, label="Plumb EPLB", zorder=3)

        for bar_b, bar_e, bv, ev in zip(b_bars, e_bars, base_vals, eplb_vals):
            if bv > 0 and ev > 0:
                delta = (ev - bv) / bv * 100
                color = RED if delta > 0 else BLUE
                sign = "+" if delta > 0 else ""
                ax.text(
                    bar_e.get_x() + bar_e.get_width() / 2,
                    bar_e.get_height() + max(base_vals) * 0.02,
                    f"{sign}{delta:.0f}%",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(pcts, fontsize=11)
        ax.set_ylabel("TTFT (ms)", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.legend(frameon=False, fontsize=9)

        if label == "Concurrency = 8":
            # annotate p99 improvement
            ax.annotate(
                "P99: −77.9%\nhot expert\nqueue drained",
                xy=(e_bars[2].get_x() + e_bars[2].get_width() / 2, eplb_vals[2]),
                xytext=(2.6, eplb_vals[2] + base_vals[2] * 0.35),
                fontsize=8, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2),
            )

    fig.suptitle("OLMoE EP=4: TTFT Latency Percentiles\n(4× RTX 3090, vLLM 0.8.5)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "latency_percentiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote latency_percentiles.png")


# ── Chart 3: Hook overhead (forest plot style) ────────────────────────────────

def chart_hook_overhead():
    hooks_on  = {"mean": 38.586, "ci_lo": 38.586 - 4.014, "ci_hi": 38.586 + 3.836}
    hooks_off = {"mean": 38.675, "ci_lo": 38.675 - 4.014, "ci_hi": 38.675 + 3.836}

    fig, ax = plt.subplots(figsize=(6, 3))

    for i, (label, d, color) in enumerate([
        ("Hooks ON", hooks_on, BLUE),
        ("Hooks OFF", hooks_off, GRAY),
    ]):
        ax.errorbar(
            d["mean"], i,
            xerr=[[d["mean"] - d["ci_lo"]], [d["ci_hi"] - d["mean"]]],
            fmt="o", color=color, capsize=6, capthick=2, elinewidth=2, markersize=9, zorder=4,
        )
        ax.text(d["mean"] + 0.25, i, f'{d["mean"]:.2f} ms', va="center", fontsize=10)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Hooks ON", "Hooks OFF"], fontsize=11)
    ax.set_xlabel("Mean TTFT (ms)", fontsize=10)
    ax.set_title("Profiling Hook Overhead — OLMoE TP=2\n95% CI, Welch t-test p=0.786", fontsize=11)
    ax.set_xlim(30, 48)
    ax.axvline(38.586, color=BLUE, lw=0.8, ls="--", alpha=0.5)

    ax.text(
        0.98, 0.12, "overhead: −0.09 ms (−0.23%)\nnot significant",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#374151",
        bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT_GRAY, ec="none"),
    )

    fig.tight_layout()
    fig.savefig(OUT / "hook_overhead.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote hook_overhead.png")


# ── Chart 4: Expert imbalance comparison ─────────────────────────────────────

def chart_imbalance_comparison():
    models = ["OLMoE-1B-7B\n(top-8 routing)", "DeepSeek-V2-Lite\n(top-2 routing)"]
    imbalances = [6.74, 1.50]
    colors = [BLUE, GRAY]

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(models, imbalances, color=colors, width=0.45, zorder=3)

    # threshold line
    ax.axhline(3.0, color=RED, lw=1.8, ls="--", zorder=4)
    ax.text(1.35, 3.12, "imbalance gate (3×)", color=RED, fontsize=9, va="bottom")

    for bar, v in zip(bars, imbalances):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.1,
            f"{v}×", ha="center", va="bottom", fontsize=12, fontweight="bold",
            color=bar.get_facecolor(),
        )

    ax.set_ylabel("Peak imbalance ratio  (max / mean)", fontsize=10)
    ax.set_title("Expert Load Imbalance: OLMoE vs DeepSeek-V2-Lite\n(100 prompts, top-k routing)", fontsize=11)
    ax.set_ylim(0, 8.5)

    # annotation boxes
    ax.text(0, 0.4, "EPLB beneficial", ha="center", fontsize=9, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.25", fc="#eff6ff", ec="none"))
    ax.text(1, 0.4, "EPLB harmful", ha="center", fontsize=9, color="#6b7280",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f3f4f6", ec="none"))

    fig.tight_layout()
    fig.savefig(OUT / "imbalance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote imbalance_comparison.png")


# ── Chart 5: Test coverage donut ─────────────────────────────────────────────

def chart_test_coverage():
    sizes = [41, 3, 3]
    labels = ["Fully verified\n41 claims (87%)", "Partial\n3 claims (6%)", "Not tested\n3 claims (6%)"]
    colors = ["#16a34a", "#f59e0b", "#e5e7eb"]
    explode = (0.04, 0, 0)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    wedges, texts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=9.5),
    )

    ax.text(0, 0, "47\nclaims", ha="center", va="center", fontsize=13, fontweight="bold", color=DARK)
    ax.set_title("plumb Test Coverage\n(plumb/tests/ + smoke_test.py)", fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT / "test_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote test_coverage.png")


if __name__ == "__main__":
    chart_throughput_sweep()
    chart_latency()
    chart_hook_overhead()
    chart_imbalance_comparison()
    chart_test_coverage()
    print("done — charts in", OUT)
