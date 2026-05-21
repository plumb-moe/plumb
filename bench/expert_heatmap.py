#!/usr/bin/env python3
"""Expert activation heatmap — static image or animated GIF.

Usage:
  # Animated GIF from timestamped frames saved during a benchmark run:
  python bench/expert_heatmap.py --frames /tmp/bench3way/snapshots/ --out expert_activation.gif

  # Static heatmap from a single snapshot:
  python bench/expert_heatmap.py --snapshot /tmp/plumb/12345_snapshot.json --out heatmap.png

  # Watch live snapshot dir and regenerate on change:
  python bench/expert_heatmap.py --watch /tmp/plumb/ --out live_heatmap.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _load_counts(path: str | Path) -> tuple[np.ndarray, int, int]:
    """Load expert_counts from a snapshot JSON. Returns (matrix, n_layers, n_experts)."""
    d = json.loads(Path(path).read_text())
    n_layers = d.get("n_layers", 48)
    raw = d.get("expert_counts", d.get("expert_loads", {}))

    n_experts = 128
    if raw:
        max_eid = max(int(k.split(":")[1]) for k in raw)
        n_experts = max(n_experts, max_eid + 1)

    mat = np.zeros((n_layers, n_experts), dtype=np.float32)
    for k, v in raw.items():
        lid, eid = map(int, k.split(":"))
        if lid < n_layers and eid < n_experts:
            mat[lid, eid] = v
    return mat, n_layers, n_experts


def _render_frame(ax, mat: np.ndarray, title: str, vmax: float | None = None) -> None:
    import matplotlib.pyplot as plt  # noqa: F401 — local import for headless compat

    ax.clear()
    log_mat = np.log1p(mat)
    vm = vmax if vmax is not None else log_mat.max() or 1.0
    im = ax.imshow(log_mat, aspect="auto", origin="upper",
                   cmap="inferno", vmin=0, vmax=vm)
    ax.set_xlabel("Expert ID", fontsize=9)
    ax.set_ylabel("Layer", fontsize=9)
    ax.set_title(title, fontsize=10, pad=4)
    ax.tick_params(labelsize=7)
    return im, vm


def make_static(snapshot_path: str, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat, n_layers, n_experts = _load_counts(snapshot_path)
    fig, ax = plt.subplots(figsize=(14, 5))
    im, _ = _render_frame(ax, mat, f"Expert activation counts — {Path(snapshot_path).name}")
    fig.colorbar(im, ax=ax, label="log(1 + count)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_gif(frames_dir: str, out_path: str, fps: int = 4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frame_files = sorted(Path(frames_dir).glob("frame_*.json"))
    if not frame_files:
        print(f"No frame_*.json files found in {frames_dir}")
        return

    print(f"Building GIF from {len(frame_files)} frames...")

    # Load all matrices, compute shared vmax for consistent colour scale
    matrices = []
    for f in frame_files:
        try:
            mat, n_layers, n_experts = _load_counts(f)
            matrices.append(mat)
        except Exception as e:
            print(f"  skip {f.name}: {e}")

    if not matrices:
        print("No valid frames loaded.")
        return

    vmax = float(max(np.log1p(m).max() for m in matrices)) or 1.0

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    cbar_ax = fig.add_axes([0.96, 0.15, 0.015, 0.7])

    # Initial frame
    im, _ = _render_frame(ax, matrices[0],
                          f"Expert activations — frame 0/{len(matrices)}", vmax=vmax)
    cb = fig.colorbar(im, cax=cbar_ax, label="log(1 + count)")

    def update(frame_idx: int):
        im, _ = _render_frame(ax, matrices[frame_idx],
                               f"Expert activations — {frame_idx * 50} requests processed",
                               vmax=vmax)
        cb.update_normal(im)
        return [im]

    anim = FuncAnimation(fig, update, frames=len(matrices),
                         interval=1000 // fps, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved: {out_path}  ({len(matrices)} frames @ {fps}fps)")


def watch_live(snap_dir: str, out_path: str, interval_s: float = 2.0) -> None:
    """Regenerate PNG every interval_s from the freshest snapshot in snap_dir."""
    import matplotlib
    matplotlib.use("Agg")

    print(f"Watching {snap_dir} → {out_path}  (Ctrl-C to stop)")
    last_mtime = 0.0
    while True:
        try:
            snaps = sorted(Path(snap_dir).glob("*_snapshot.json"),
                           key=lambda p: p.stat().st_mtime)
            if snaps:
                newest = snaps[-1]
                mtime = newest.stat().st_mtime
                if mtime != last_mtime:
                    make_static(str(newest), out_path)
                    last_mtime = mtime
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  error: {e}")
        time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert activation heatmap / GIF")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--frames", metavar="DIR",
                     help="Directory of frame_NNNN.json snapshots → animated GIF")
    grp.add_argument("--snapshot", metavar="FILE",
                     help="Single snapshot JSON → static PNG")
    grp.add_argument("--watch", metavar="DIR",
                     help="Live-watch snapshot dir, regenerate PNG on change")
    parser.add_argument("--out", required=True, help="Output file (.gif or .png)")
    parser.add_argument("--fps", type=int, default=4, help="GIF frames per second (default 4)")
    args = parser.parse_args()

    if args.frames:
        make_gif(args.frames, args.out, fps=args.fps)
    elif args.snapshot:
        make_static(args.snapshot, args.out)
    elif args.watch:
        watch_live(args.watch, args.out)


if __name__ == "__main__":
    main()
