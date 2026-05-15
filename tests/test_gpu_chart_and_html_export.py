"""Tests for bx-5i9 (gpu_to_numa in report) and bx-248 (HTML export)."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from plumb.cli import main
from plumb.counter import ActivationCounter
from plumb.report.generator import generate_report
from plumb.report.html import generate_html_report
from plumb.report.schema import ProfileReport
from plumb.topology import Topology


def _counter(data: dict[tuple[int, int], int], passes: int = 50) -> ActivationCounter:
    c = ActivationCounter(window_size=100_000)
    for (layer, expert), count in data.items():
        c.record(layer, expert, count)
    for _ in range(passes):
        c.increment_pass()
    return c


def _dual_epyc() -> Topology:
    return Topology({0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1})


# ---------------------------------------------------------------------------
# bx-5i9: gpu_to_numa in ProfileReport
# ---------------------------------------------------------------------------

class TestGpuToNuma:
    def test_multi_numa_topology_included_in_report(self):
        c = _counter({(l, e): 100 for l in range(4) for e in range(8)})
        report = generate_report(c, _dual_epyc(), "Model", 10.0, num_gpus=8)
        assert report.gpu_to_numa is not None
        assert report.gpu_to_numa[0] == 0
        assert report.gpu_to_numa[4] == 1

    def test_flat_topology_gives_none_gpu_to_numa(self):
        # All GPUs on same NUMA node — no meaningful NUMA split
        c = _counter({(0, e): 100 for e in range(8)})
        report = generate_report(c, Topology.flat(8), "Model", 10.0, num_gpus=8)
        assert report.gpu_to_numa is None

    def test_gpu_to_numa_survives_json_round_trip(self):
        c = _counter({(l, e): 50 for l in range(2) for e in range(8)})
        report = generate_report(c, _dual_epyc(), "Model", 5.0, num_gpus=8)
        data = json.loads(report.model_dump_json())
        # Pydantic serialises int keys as strings in JSON
        assert data["gpu_to_numa"] is not None
        assert "0" in data["gpu_to_numa"] or 0 in data["gpu_to_numa"]

    def test_gpu_to_numa_not_in_schema_when_none(self):
        c = _counter({(0, e): 100 for e in range(4)})
        report = generate_report(c, Topology.flat(4), "Model", 5.0)
        data = json.loads(report.model_dump_json())
        assert data.get("gpu_to_numa") is None


# ---------------------------------------------------------------------------
# bx-248: HTML export — generate_html_report()
# ---------------------------------------------------------------------------

class TestHtmlExport:
    def _make_report(self) -> ProfileReport:
        c = _counter({(l, e): (e + 1) * 20 for l in range(4) for e in range(8)})
        return generate_report(c, _dual_epyc(), "MixtralTest", 60.0, num_gpus=8)

    def test_html_contains_model_name(self):
        report = self._make_report()
        html = generate_html_report(report)
        assert "MixtralTest" in html

    def test_html_is_self_contained_no_fetch(self):
        report = self._make_report()
        html = generate_html_report(report)
        # Should not contain fetch('/api/report') — that would make it non-self-contained
        assert "fetch('/api/report')" not in html
        assert "/api/report" not in html

    def test_html_has_no_setinterval_poll(self):
        report = self._make_report()
        html = generate_html_report(report)
        assert "setInterval" not in html

    def test_html_injects_static_data_const(self):
        report = self._make_report()
        html = generate_html_report(report)
        assert "_STATIC_DATA" in html

    def test_html_contains_full_report_json(self):
        report = self._make_report()
        html = generate_html_report(report)
        # The model name should appear in the inlined JSON
        assert report.model_name in html
        assert str(report.total_forward_passes) in html

    def test_html_is_valid_html_structure(self):
        report = self._make_report()
        html = generate_html_report(report)
        assert html.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<script>" in html
        assert "</script>" in html

    def test_html_size_under_500kb(self):
        report = self._make_report()
        html = generate_html_report(report)
        size_bytes = len(html.encode("utf-8"))
        assert size_bytes < 500 * 1024, f"HTML export too large: {size_bytes / 1024:.0f} KB"

    def test_html_report_validates_against_schema(self):
        """The inlined JSON must be a valid ProfileReport."""
        report = self._make_report()
        html = generate_html_report(report)
        # Extract the JSON from const _STATIC_DATA = {...};
        m = re.search(r'const _STATIC_DATA = (\{.*?\});', html, re.DOTALL)
        assert m, "Could not find _STATIC_DATA in HTML"
        parsed = ProfileReport.model_validate_json(m.group(1))
        assert parsed.model_name == report.model_name
        assert len(parsed.layers) == len(report.layers)


# ---------------------------------------------------------------------------
# bx-248: CLI --format html
# ---------------------------------------------------------------------------

class TestReportCliHtmlFormat:
    def _make_snapshot(self, tmp_path: Path) -> Path:
        snap = tmp_path / "12345_snapshot.json"
        loads = {f"{l}:{e}": (e + 1) * 10 for l in range(4) for e in range(8)}
        snap.write_text(json.dumps({
            "pid": 12345,
            "model_name": "TestMoE",
            "n_layers": 4,
            "pass_count": 80,
            "updated_at": time.time(),
            "started_at": time.time() - 90.0,
            "expert_counts": loads,
            "gpu_to_numa": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 1, "5": 1, "6": 1, "7": 1},
        }))
        return snap

    def test_format_html_writes_html_file(self, tmp_path):
        from plumb.registry import SessionInfo
        snap = self._make_snapshot(tmp_path)
        out = tmp_path / "out.html"
        session = SessionInfo(
            pid=12345, model_name="TestMoE", n_layers=4,
            socket_path="", started_at=time.time() - 90.0,
            snapshot_path=str(snap),
        )
        runner = CliRunner()
        with patch("plumb.cli.list_sessions", return_value=[session]):
            result = runner.invoke(main, ["report", "--pid", "12345", "--format", "html", "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()
        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        assert "TestMoE" in content
        assert "/api/report" not in content

    def test_format_json_default_unchanged(self, tmp_path):
        from plumb.registry import SessionInfo
        snap = self._make_snapshot(tmp_path)
        out = tmp_path / "out.json"
        session = SessionInfo(
            pid=12345, model_name="TestMoE", n_layers=4,
            socket_path="", started_at=time.time() - 90.0,
            snapshot_path=str(snap),
        )
        runner = CliRunner()
        with patch("plumb.cli.list_sessions", return_value=[session]):
            result = runner.invoke(main, ["report", "--pid", "12345", "--format", "json", "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["model_name"] == "TestMoE"

    def test_html_report_size_reported_in_output(self, tmp_path):
        from plumb.registry import SessionInfo
        snap = self._make_snapshot(tmp_path)
        out = tmp_path / "out.html"
        session = SessionInfo(
            pid=12345, model_name="TestMoE", n_layers=4,
            socket_path="", started_at=time.time() - 90.0,
            snapshot_path=str(snap),
        )
        runner = CliRunner()
        with patch("plumb.cli.list_sessions", return_value=[session]):
            result = runner.invoke(main, ["report", "--pid", "12345", "--format", "html", "--output", str(out)])
        assert "KB" in result.output
