from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from plumb.cli import main
from plumb.counter import ActivationCounter
from plumb.diff import DiffResult, compute_diff
from plumb.report.diff_html import render_diff_html
from plumb.report.generator import generate_report
from plumb.report.schema import ProfileReport
from plumb.topology import Topology


def _report(expert_loads: dict[tuple[int, int], int], model: str = "TestModel",
            passes: int = 100, num_gpus: int | None = None) -> ProfileReport:
    c = ActivationCounter(window_size=100_000)
    for (layer, expert), count in expert_loads.items():
        c.record(layer, expert, count)
    for _ in range(passes):
        c.increment_pass()
    return generate_report(c, Topology.flat(num_gpus or 1), model, 10.0, num_gpus=num_gpus)


# ---------------------------------------------------------------------------
# compute_diff
# ---------------------------------------------------------------------------

def test_diff_basic_structure():
    a = _report({(0, 0): 500, (0, 1): 100})
    b = _report({(0, 0): 300, (0, 1): 200})
    result = compute_diff(a, b)
    assert isinstance(result, DiffResult)
    assert result.model_name_a == "TestModel"
    assert result.model_name_b == "TestModel"
    assert len(result.expert_deltas) == 2


def test_diff_delta_values():
    a = _report({(0, 0): 400, (0, 1): 100})
    b = _report({(0, 0): 200, (0, 1): 300})
    result = compute_diff(a, b)
    by_expert = {d.expert_id: d for d in result.expert_deltas if d.layer_id == 0}
    assert by_expert[0].delta == 200 - 400          # -200
    assert by_expert[1].delta == 300 - 100          # +200


def test_diff_delta_pct():
    a = _report({(0, 0): 200, (0, 1): 100})
    b = _report({(0, 0): 100, (0, 1): 100})
    result = compute_diff(a, b)
    d0 = next(d for d in result.expert_deltas if d.layer_id == 0 and d.expert_id == 0)
    assert d0.delta_pct == pytest.approx(-50.0)


def test_diff_imbalance_summary():
    # a has high imbalance; b has low imbalance
    a = _report({(0, 0): 800, (0, 1): 100, (0, 2): 100, (0, 3): 100})
    b = _report({(0, 0): 250, (0, 1): 250, (0, 2): 250, (0, 3): 250})
    result = compute_diff(a, b)
    assert result.mean_imbalance_before > result.mean_imbalance_after


def test_diff_multilayer():
    loads_a = {(l, e): 100 + e * 50 for l in range(4) for e in range(8)}
    loads_b = {(l, e): 100 + e * 10 for l in range(4) for e in range(8)}
    a = _report(loads_a)
    b = _report(loads_b)
    result = compute_diff(a, b)
    assert len(result.expert_deltas) == 4 * 8


def test_diff_missing_layer_in_b():
    """Expert in layer_a with no matching layer_b gets token_count_after=0."""
    a = _report({(0, 0): 100, (0, 1): 100, (1, 0): 200, (1, 1): 200})
    b = _report({(0, 0): 150, (0, 1): 50})   # layer 1 missing from b
    result = compute_diff(a, b)
    layer1 = [d for d in result.expert_deltas if d.layer_id == 1]
    assert all(d.token_count_after == 0 for d in layer1)


def test_diff_ttft_none_when_no_placement():
    """Reports with no placement (empty counter) produce None TTFT estimates."""
    from plumb.report.schema import ProfileReport, LayerReport, ExpertLoad
    from datetime import datetime, timezone

    def _bare(model: str) -> ProfileReport:
        return ProfileReport(
            model_name=model,
            hardware_config="test",
            profiling_duration_seconds=10.0,
            total_forward_passes=100,
            layers=[LayerReport(layer_id=0, imbalance_ratio=1.5, max_expert_id=0,
                                min_expert_id=1, experts=[
                                    ExpertLoad(expert_id=0, token_count=150, activation_fraction=0.6),
                                    ExpertLoad(expert_id=1, token_count=100, activation_fraction=0.4),
                                ])],
            placement=None,
            generated_at=datetime.now(tz=timezone.utc),
        )

    result = compute_diff(_bare("A"), _bare("B"))
    assert result.ttft_est_before is None
    assert result.ttft_est_after is None


def test_diff_ttft_present_with_placement():
    a = _report({(l, e): 100 + e * 50 for l in range(3) for e in range(8)}, num_gpus=8)
    b = _report({(l, e): 100 + e * 20 for l in range(3) for e in range(8)}, num_gpus=8)
    result = compute_diff(a, b)
    assert result.ttft_est_before is not None
    assert result.ttft_est_after is not None


# ---------------------------------------------------------------------------
# render_diff_html
# ---------------------------------------------------------------------------

def test_html_render_contains_key_elements():
    a = _report({(0, e): 100 + e * 30 for e in range(4)})
    b = _report({(0, e): 100 + e * 10 for e in range(4)})
    result = compute_diff(a, b)
    html = render_diff_html(result)
    assert "plumb diff" in html
    assert "TestModel" in html
    assert "Mean Imbalance" in html
    assert "Per-Expert Delta" in html


def test_html_is_self_contained():
    """No external resource references."""
    a = _report({(0, 0): 500, (0, 1): 100})
    b = _report({(0, 0): 300, (0, 1): 200})
    result = compute_diff(a, b)
    html = render_diff_html(result)
    assert "http" not in html
    assert "<script src" not in html
    assert "<link rel" not in html


def test_html_contains_row_data():
    a = _report({(0, 0): 400, (0, 1): 200})
    b = _report({(0, 0): 200, (0, 1): 400})
    result = compute_diff(a, b)
    html = render_diff_html(result)
    # Row data is embedded as JSON
    assert '"layer"' in html
    assert '"delta"' in html


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def test_cli_diff_text(tmp_path: Path):
    a = _report({(0, 0): 500, (0, 1): 100})
    b = _report({(0, 0): 300, (0, 1): 200})
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(a.model_dump_json())
    path_b.write_text(b.model_dump_json())

    runner = CliRunner()
    result = runner.invoke(main, ["diff", str(path_a), str(path_b)])
    assert result.exit_code == 0, result.output
    assert "Mean imbalance ratio" in result.output


def test_cli_diff_html(tmp_path: Path):
    a = _report({(0, e): 100 + e * 40 for e in range(4)})
    b = _report({(0, e): 100 + e * 15 for e in range(4)})
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    out    = tmp_path / "out.html"
    path_a.write_text(a.model_dump_json())
    path_b.write_text(b.model_dump_json())

    runner = CliRunner()
    result = runner.invoke(main, ["diff", str(path_a), str(path_b), "--format", "html", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    html = out.read_text()
    assert "plumb diff" in html


def test_cli_diff_invalid_json(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    good = tmp_path / "good.json"
    good.write_text(_report({(0, 0): 100}).model_dump_json())

    runner = CliRunner()
    result = runner.invoke(main, ["diff", str(bad), str(good)])
    assert result.exit_code != 0 or "Failed" in result.output
