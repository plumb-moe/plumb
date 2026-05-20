from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.table import Table

from .registry import REGISTRY_DIR, deregister, list_sessions

console = Console()


@click.group(invoke_without_command=True)
@click.version_option()
@click.pass_context
def main(ctx: click.Context) -> None:
    """plumb — MoE inference profiler.

    Run without a subcommand to detect running models interactively.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(detect)


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------

@main.command()
def detect() -> None:
    """Scan for running MoE inference processes on this machine."""
    from .registry import list_sessions
    from .scanner import scan_gpu_processes

    console.rule("[bold cyan]plumb — detect[/bold cyan]")
    processes = scan_gpu_processes()
    sessions  = list_sessions()

    gpu_pids = {p.pid for p in processes}
    # Sessions whose process isn't using the GPU (CPU inference or not visible to nvidia-smi)
    cpu_sessions = [s for s in sessions if s.pid not in gpu_pids]

    if not processes and not cpu_sessions:
        console.print("[yellow]No inference processes found.[/yellow]")
        console.print(
            "  Start one with:\n"
            "  [bold]plumb run -- python my_inference.py[/bold]"
        )
        return

    t = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    t.add_column("PID",     style="dim", width=8)
    t.add_column("VRAM",    width=10)
    t.add_column("Model",   width=22)
    t.add_column("Status",  width=20)
    t.add_column("Command", overflow="fold", max_width=48)

    for p in processes:
        model_str = p.detected_model or "[dim]unknown[/dim]"
        if p.session:
            status = f"[green]profiling ({p.session.n_layers}L)[/green]"
        else:
            status = "[dim]not attached[/dim]"
        t.add_row(str(p.pid), f"{p.gpu_memory_mb:,} MB", model_str, status, p.cmdline[:80])

    for s in cpu_sessions:
        t.add_row(str(s.pid), "[dim]CPU[/dim]", s.model_name,
                  f"[green]profiling ({s.n_layers}L)[/green]", "")

    console.print(t)

    unattached = [p for p in processes if p.session is None]
    if unattached:
        console.print(
            "\nTo profile unattached processes, wrap the command with:\n"
            "  [bold]plumb run -- <your inference command>[/bold]"
        )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@main.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option(
    "--eplb-output",
    default=None,
    type=click.Path(),
    metavar="PATH",
    help="After the wrapped process exits, write a float32 [num_layers, num_experts] "
         "numpy .npy file from the session snapshot (for DeepSeek EPLB rebalancing).",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.option("--prometheus-port", default=None, type=int,
              help="Expose Prometheus metrics at this port (e.g. 9000). "
                   "Metrics: vllm:moe_expert_activation_count, vllm:moe_imbalance_ratio.")
def run(eplb_output: str | None, args: tuple[str, ...], prometheus_port: int | None) -> None:
    """Wrap any inference command and inject live profiling.

    \b
    Examples:
      plumb run -- python inference.py
      plumb run -- vllm serve mistralai/Mixtral-8x7B-v0.1
      plumb run --eplb-output weights.npy -- vllm serve deepseek-ai/DeepSeek-V3
      plumb run --prometheus-port 9000 -- vllm serve mistralai/Mixtral-8x7B-v0.1
    """
    if not args:
        console.print("[red]No command given. Usage: plumb run -- <cmd>[/red]")
        sys.exit(1)

    from .launcher import launch

    extra_env: dict[str, str] = {}
    if prometheus_port is not None:
        extra_env["SAI_PROFILER_PROMETHEUS_PORT"] = str(prometheus_port)
        console.print(f"  Prometheus metrics → http://localhost:{prometheus_port}/metrics")

    console.print(f"[cyan]plumb[/cyan] wrapping: [bold]{' '.join(args)}[/bold]")
    console.print("  Auto-attach will scan for MoE models every ~8 s after startup.")
    console.print("  Run [bold]plumb detect[/bold] in another terminal to see live status.\n")
    rc = launch(list(args), extra_env=extra_env or None)

    if eplb_output is not None:
        _write_eplb_output(eplb_output)

    sys.exit(rc)


def _write_eplb_output(path: str) -> None:
    """Read the most recent session snapshot and write float32 [num_layers, num_experts] .npy."""
    from pathlib import Path as _Path

    import numpy as np

    snapshot_data: dict | None = None

    # Prefer a live session snapshot, fall back to most recent file in registry dir
    try:
        active = list_sessions()
        if active:
            try:
                snapshot_data = json.loads(_Path(active[0].snapshot_path).read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                pass
    except Exception:
        pass

    if snapshot_data is None:
        candidates = sorted(REGISTRY_DIR.glob("*_snapshot.json")) if REGISTRY_DIR.exists() else []
        if candidates:
            try:
                snapshot_data = json.loads(candidates[-1].read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                pass

    if snapshot_data is None:
        console.print("[yellow]--eplb-output: no snapshot found — skipping.[/yellow]")
        return

    expert_loads: dict[str, int] = snapshot_data.get("expert_loads", {})
    if not expert_loads:
        console.print("[yellow]--eplb-output: snapshot has no expert_loads — skipping.[/yellow]")
        return

    # Determine matrix dimensions from key space "layer_id:expert_id"
    layer_ids: set[int] = set()
    expert_ids: set[int] = set()
    for key in expert_loads:
        try:
            lid, eid = key.split(":", 1)
            layer_ids.add(int(lid))
            expert_ids.add(int(eid))
        except ValueError:
            continue

    if not layer_ids or not expert_ids:
        console.print("[yellow]--eplb-output: could not parse expert_loads keys — skipping.[/yellow]")
        return

    num_layers  = max(layer_ids) + 1
    num_experts = max(expert_ids) + 1
    weight = np.zeros((num_layers, num_experts), dtype=np.float32)
    for key, count in expert_loads.items():
        try:
            lid, eid = key.split(":", 1)
            weight[int(lid), int(eid)] = float(count)
        except (ValueError, IndexError):
            continue

    out_path = _Path(path)
    np.save(out_path, weight)
    console.print(
        f"[green]EPLB weights → {out_path}[/green]  "
        f"[dim]shape=({num_layers}, {num_experts})  dtype=float32[/dim]"
    )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@main.command()
@click.option("--pid", default=None, type=int, help="Generate from a running session PID.")
@click.option("--snapshot", default=None, type=click.Path(exists=True), help="Read from snapshot JSON directly.")
@click.option("--output", "-o", default=None, show_default=False, help="Output path (default: report.json or report.html).")
@click.option("--format", "fmt", default="json", show_default=True, type=click.Choice(["json", "html"]), help="Output format.")
@click.option("--open-dashboard", is_flag=True, help="Open the dashboard after writing the report.")
@click.option("--port", default=8080, show_default=True)
def report(pid: int | None, snapshot: str | None, output: str | None, fmt: str, open_dashboard: bool, port: int) -> None:
    """Generate a profiling report.

    \b
    Sources (in priority order):
      --pid         live session by PID
      --snapshot    raw snapshot JSON file
      (none)        first active session, else exits with error

    Use --format html for a self-contained HTML file (default: json).
    """
    from .report.generator import generate_report_from_snapshot

    snapshot_path: str | None = snapshot

    if pid is not None:
        sessions = [s for s in list_sessions() if s.pid == pid]
        if not sessions:
            console.print(f"[red]No active session for PID {pid}.[/red]")
            sys.exit(1)
        snapshot_path = sessions[0].snapshot_path
    elif snapshot_path is None:
        active = list_sessions()
        if not active:
            console.print("[red]No active sessions found. Provide --pid or --snapshot.[/red]")
            sys.exit(1)
        snapshot_path = active[0].snapshot_path
        console.print(f"Using session: PID {active[0].pid} ({active[0].model_name})")

    assert snapshot_path is not None
    try:
        data = json.loads(Path(snapshot_path).read_text())
    except FileNotFoundError:
        console.print(f"[red]Snapshot not found: {snapshot_path}[/red]")
        sys.exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid snapshot JSON: {e}[/red]")
        sys.exit(1)

    try:
        prof_report = generate_report_from_snapshot(data)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    default_ext = ".html" if fmt == "html" else ".json"
    out_path = Path(output) if output else Path(f"report{default_ext}")

    if fmt == "html":
        from .report.html import generate_html_report
        html = generate_html_report(prof_report)
        out_path.write_text(html, encoding="utf-8")
        size_kb = len(html.encode("utf-8")) / 1024
        console.print(f"[green]HTML report written → {out_path}[/green] ({size_kb:.0f} KB)")
    else:
        out_path.write_text(prof_report.model_dump_json(indent=2))
        console.print(f"[green]Report written → {out_path}[/green]")

    s = prof_report.summary()
    if s:
        console.print(
            f"  layers={s['num_layers_profiled']}  "
            f"mean_imbalance={s['mean_imbalance_ratio']}×  "
            f"max_imbalance={s['max_imbalance_ratio']}× (layer {s['worst_layer_id']})  "
            f"passes={s['total_forward_passes']}"
        )

    if open_dashboard:
        from .dashboard.server import serve
        console.print(f"Dashboard → http://localhost:{port}")
        serve(report_path=str(out_path), port=port)


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------

@main.command()
@click.argument("pid", type=int, required=False, default=None)
@click.option("--all", "attach_all", is_flag=True, help="Stream all registered sessions.")
@click.option("--interval", default=2, show_default=True, help="Refresh interval (seconds).")
def attach(pid: int | None, attach_all: bool, interval: int) -> None:
    """Stream live metrics from a running profiled session.

    \b
    With no arguments: interactive picker.
    With a PID:        stream that session.
    With --all:        stream all sessions.
    """
    from .registry import list_sessions

    sessions = list_sessions()

    if not sessions:
        console.print("[yellow]No active plumb sessions found.[/yellow]")
        console.print("Start one with: [bold]plumb run -- <your inference command>[/bold]")
        return

    if pid is not None:
        targets = [s for s in sessions if s.pid == pid]
        if not targets:
            console.print(f"[red]No session found for PID {pid}.[/red]")
            sys.exit(1)
    elif attach_all:
        targets = sessions
    elif len(sessions) == 1:
        targets = sessions
    else:
        # Interactive picker
        console.print("Active sessions:")
        for i, s in enumerate(sessions, 1):
            console.print(f"  [{i}] PID {s.pid} — {s.model_name} ({s.n_layers} layers)")
        choice = click.prompt("Select session", type=click.IntRange(1, len(sessions)))
        targets = [sessions[choice - 1]]

    console.rule("[bold cyan]plumb — live[/bold cyan]")
    _stream_sessions(targets, interval)


def _stream_sessions(sessions, interval: int) -> None:
    try:
        while True:
            for s in sessions:
                snap_path = Path(s.snapshot_path)
                if not snap_path.exists():
                    console.print(f"[dim]PID {s.pid}: waiting for first snapshot…[/dim]")
                    continue
                try:
                    data = json.loads(snap_path.read_text())
                except json.JSONDecodeError:
                    continue
                _print_snapshot(data)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


def _print_snapshot(data: dict) -> None:
    pid        = data.get("pid", "?")
    model      = data.get("model_name", "?")
    passes     = data.get("pass_count", 0)
    updated    = data.get("updated_at", 0)
    imbalance  = data.get("imbalance", [])
    age        = time.time() - updated

    console.print(
        f"\n[bold]{model}[/bold]  pid={pid}  passes={passes}  "
        f"[dim]{age:.0f}s ago[/dim]"
    )
    if not imbalance:
        console.print("  [dim]no data yet[/dim]")
        return

    worst = max(imbalance, key=lambda r: r["ratio"])
    mean  = sum(r["ratio"] for r in imbalance) / len(imbalance)

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    t.add_column("Layer", width=6)
    t.add_column("Imbalance ratio", width=16)
    t.add_column("Hot expert", width=12)

    for r in imbalance:
        ratio_str = f"{r['ratio']:.3f}"
        if r["ratio"] >= worst["ratio"]:
            ratio_str = f"[red]{ratio_str}[/red]"
        elif r["ratio"] >= mean * 1.5:
            ratio_str = f"[yellow]{ratio_str}[/yellow]"
        t.add_row(str(r["layer_id"]), ratio_str, str(r["max_expert"]))

    console.print(t)
    console.print(
        f"  mean ratio: {mean:.3f}   worst layer: {worst['layer_id']} "
        f"({worst['ratio']:.3f}×, expert {worst['max_expert']})"
    )


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

@main.command()
@click.option("--report", default=None, type=click.Path(exists=True), help="Static report JSON (optional)")
@click.option("--pid", default=None, type=int, help="Stream from a running session PID")
@click.option("--port", default=8080, show_default=True)
def dashboard(report: str | None, pid: int | None, port: int) -> None:
    """Serve the profiling dashboard.

    \b
    With --report: render a saved report.
    With --pid:    stream live from a running session.
    No args:       uses the first active session if available, else last report.json.
    """
    from .dashboard.server import serve
    from .registry import list_sessions

    snapshot_path: str | None = None

    if pid is not None:
        sessions = [s for s in list_sessions() if s.pid == pid]
        if not sessions:
            console.print(f"[red]No session for PID {pid}.[/red]")
            sys.exit(1)
        snapshot_path = sessions[0].snapshot_path
    elif report is None:
        active = list_sessions()
        if active:
            snapshot_path = active[0].snapshot_path
            console.print(f"Streaming live session: PID {active[0].pid} ({active[0].model_name})")
        elif Path("report.json").exists():
            report = "report.json"
        else:
            console.print("[yellow]No active session and no report.json found.[/yellow]")
            console.print("Run [bold]plumb run -- <cmd>[/bold] first.")
            sys.exit(1)

    console.print(f"Dashboard → http://localhost:{port}")
    serve(report_path=report, snapshot_path=snapshot_path, port=port)


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------

@main.command()
@click.option("--pid", default=None, type=int, help="PID of the session to stop (default: first active session).")
def stop(pid: int | None) -> None:
    """Stop a running profiled session cleanly.

    \b
    Sends SIGTERM and waits up to 10 s for the session to deregister.
    Falls back to SIGKILL if the process does not exit in time.
    """
    sessions = list_sessions()
    if not sessions:
        console.print("[yellow]No active plumb sessions found.[/yellow]")
        sys.exit(1)

    if pid is not None:
        targets = [s for s in sessions if s.pid == pid]
        if not targets:
            console.print(f"[red]No active session for PID {pid}.[/red]")
            sys.exit(1)
    else:
        if len(sessions) == 1:
            targets = sessions
        else:
            console.print("Active sessions:")
            for i, s in enumerate(sessions, 1):
                console.print(f"  [{i}] PID {s.pid} — {s.model_name} ({s.n_layers} layers)")
            choice = click.prompt("Select session to stop", type=click.IntRange(1, len(sessions)))
            targets = [sessions[choice - 1]]

    session = targets[0]
    registry_file = REGISTRY_DIR / f"{session.pid}.json"

    console.print(f"Stopping PID {session.pid} ({session.model_name})…")
    try:
        try:
            os.kill(session.pid, signal.SIGTERM)
        except ProcessLookupError:
            console.print(f"[yellow]PID {session.pid} already gone.[/yellow]")
            deregister(session.pid)
            return

        # Wait up to 10 s for the registry file to disappear (process deregisters on exit)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if not registry_file.exists():
                break
            try:
                os.kill(session.pid, 0)
            except OSError:
                break
            time.sleep(0.25)
        else:
            console.print(f"[yellow]PID {session.pid} did not exit — sending SIGKILL.[/yellow]")
            try:
                os.kill(session.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — session may still be running.[/yellow]")
        sys.exit(1)

    deregister(session.pid)
    console.print(f"[green]Session {session.pid} stopped.[/green]")


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

@main.command()
@click.argument("report_a", type=click.Path(exists=True))
@click.argument("report_b", type=click.Path(exists=True))
@click.option("--format", "fmt", default="text", show_default=True,
              type=click.Choice(["text", "html"]), help="Output format.")
@click.option("--out", "-o", default=None, type=click.Path(), help="Output file (HTML only).")
def diff(report_a: str, report_b: str, fmt: str, out: str | None) -> None:
    """Compare two ProfileReport JSON files and show per-expert deltas."""
    from .diff import compute_diff
    from .report.schema import ProfileReport

    try:
        a = ProfileReport.model_validate_json(Path(report_a).read_text())
        b = ProfileReport.model_validate_json(Path(report_b).read_text())
    except Exception as exc:
        console.print(f"[red]Failed to load report: {exc}[/red]")
        sys.exit(1)

    result = compute_diff(a, b)

    if fmt == "html":
        from .report.diff_html import render_diff_html
        html = render_diff_html(result)
        out_path = Path(out) if out else Path("diff.html")
        out_path.write_text(html, encoding="utf-8")
        console.print(f"[green]Diff written → {out_path}[/green]")
        return

    console.print(f"[bold]plumb diff[/bold]: {a.model_name} → {b.model_name}")
    console.print(
        f"  Mean imbalance ratio: {result.mean_imbalance_before:.4f}× → {result.mean_imbalance_after:.4f}×"
    )
    console.print(
        f"  Max imbalance ratio:  {result.max_imbalance_before:.4f}× → {result.max_imbalance_after:.4f}×"
    )
    if result.ttft_est_before is not None and result.ttft_est_after is not None:
        console.print(
            f"  Est. TTFT improvement: ~{result.ttft_est_before:.0f}% → ~{result.ttft_est_after:.0f}%"
        )
