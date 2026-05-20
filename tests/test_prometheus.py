from __future__ import annotations

from prometheus_client import generate_latest

from plumb.counter import ActivationCounter
from plumb.exporters.prometheus import PrometheusExporter


def _counter(data: dict[tuple[int, int], int], passes: int = 10) -> ActivationCounter:
    c = ActivationCounter(window_size=100_000)
    for (layer, expert), count in data.items():
        c.record(layer, expert, count)
    for _ in range(passes):
        c.increment_pass()
    return c


def _metrics_text(exporter: PrometheusExporter) -> str:
    return generate_latest(exporter._registry).decode()


# ---------------------------------------------------------------------------
# update() — metric values
# ---------------------------------------------------------------------------

def test_activation_count_labels_present():
    c = _counter({(0, 3): 500, (1, 7): 200})
    exp = PrometheusExporter(c, port=19001)
    exp.update()
    text = _metrics_text(exp)
    assert 'layer="0",expert="3"' in text or 'expert="3",layer="0"' in text
    assert 'layer="1",expert="7"' in text or 'expert="7",layer="1"' in text


def test_activation_count_values():
    c = _counter({(0, 0): 300, (0, 1): 100})
    exp = PrometheusExporter(c, port=19002)
    exp.update()
    text = _metrics_text(exp)
    # Counter value should match token counts
    assert "300.0" in text or "300" in text
    assert "100.0" in text or "100" in text


def test_imbalance_gauge_present():
    c = _counter({(0, 0): 800, (0, 1): 100, (0, 2): 100})
    exp = PrometheusExporter(c, port=19003)
    exp.update()
    text = _metrics_text(exp)
    assert "moe_imbalance_ratio" in text
    assert 'layer="0"' in text


def test_imbalance_gauge_value_reasonable():
    # Expert 0 gets 9x more than others → ratio > 1
    c = _counter({(0, 0): 900, (0, 1): 100})
    exp = PrometheusExporter(c, port=19004)
    exp.update()
    text = _metrics_text(exp)
    assert "moe_imbalance_ratio" in text
    # Extract gauge value: should be > 1.0
    for line in text.splitlines():
        if "moe_imbalance_ratio" in line and not line.startswith("#"):
            value = float(line.split()[-1])
            assert value > 1.0, f"Expected imbalance > 1.0, got {value}"
            break


def test_metric_names_use_vllm_namespace():
    c = _counter({(0, 0): 100})
    exp = PrometheusExporter(c, port=19005)
    exp.update()
    text = _metrics_text(exp)
    assert "vllm:moe_expert_activation_count" in text
    assert "vllm:moe_imbalance_ratio" in text


# ---------------------------------------------------------------------------
# Delta tracking — Counter only goes up
# ---------------------------------------------------------------------------

def test_counter_increments_on_second_update():
    c = _counter({(0, 0): 100})
    exp = PrometheusExporter(c, port=19006)
    exp.update()

    # Simulate more tokens arriving
    c.record(0, 0, 50)
    exp.update()

    text = _metrics_text(exp)
    # Total should be at least 150
    for line in text.splitlines():
        if "moe_expert_activation_count" in line and 'expert="0"' in line and not line.startswith("#"):
            value = float(line.split()[-1])
            assert value >= 150.0
            break


def test_counter_does_not_decrease_on_window_expiry():
    """If rolling window shrinks, Counter must not decrement."""
    c = ActivationCounter(window_size=3)
    c.record(0, 0, 100)
    c.record(0, 0, 100)
    c.record(0, 0, 100)

    exp = PrometheusExporter(c, port=19007)
    exp.update()

    text_before = _metrics_text(exp)
    val_before = _extract_activation(text_before, layer=0, expert=0)

    # Overflow the window — old entries fall off
    c.record(0, 0, 1)
    c.record(0, 0, 1)
    c.record(0, 0, 1)
    c.record(0, 0, 1)

    exp.update()

    text_after = _metrics_text(exp)
    val_after = _extract_activation(text_after, layer=0, expert=0)
    # Counter should never go backward
    assert val_after >= val_before


# ---------------------------------------------------------------------------
# Multi-layer
# ---------------------------------------------------------------------------

def test_multiple_layers_each_get_gauge():
    c = _counter({
        (0, 0): 500, (0, 1): 100,
        (1, 0): 200, (1, 1): 200,
        (2, 0): 300, (2, 1): 100,
    })
    exp = PrometheusExporter(c, port=19008)
    exp.update()
    text = _metrics_text(exp)
    assert 'layer="0"' in text
    assert 'layer="1"' in text
    assert 'layer="2"' in text


def test_empty_counter_produces_no_metrics():
    c = ActivationCounter()
    exp = PrometheusExporter(c, port=19009)
    exp.update()
    text = _metrics_text(exp)
    # No data points — no sample lines (only HELP/TYPE headers if any)
    data_lines = [l for l in text.splitlines()
                  if l and not l.startswith("#")]
    assert data_lines == []


# ---------------------------------------------------------------------------
# start / stop (no actual port bind needed for unit tests)
# ---------------------------------------------------------------------------

def test_stop_sets_event():
    c = _counter({(0, 0): 100})
    exp = PrometheusExporter(c, port=19010, interval=60.0)
    exp.stop()
    assert exp._stop.is_set()


# ---------------------------------------------------------------------------
# CLI integration — --prometheus-port passes env var to launcher
# ---------------------------------------------------------------------------

def test_cli_run_prometheus_port_env(monkeypatch):
    """--prometheus-port should be forwarded as SAI_PROFILER_PROMETHEUS_PORT."""
    from click.testing import CliRunner

    from plumb.cli import main

    captured: dict[str, str] = {}

    def fake_launch(args, extra_env=None):
        captured.update(extra_env or {})
        return 0

    monkeypatch.setattr("plumb.cli.launch", fake_launch, raising=False)

    # Patch launch in the run command's import scope
    import plumb.launcher as launcher_mod
    monkeypatch.setattr(launcher_mod, "launch", fake_launch)

    runner = CliRunner()
    result = runner.invoke(main, ["run", "--prometheus-port", "9000", "--", "echo", "hi"])
    assert captured.get("SAI_PROFILER_PROMETHEUS_PORT") == "9000"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _extract_activation(text: str, layer: int, expert: int) -> float:
    for line in text.splitlines():
        if (
            "moe_expert_activation_count" in line
            and f'layer="{layer}"' in line
            and f'expert="{expert}"' in line
            and not line.startswith("#")
        ):
            return float(line.split()[-1])
    return 0.0
