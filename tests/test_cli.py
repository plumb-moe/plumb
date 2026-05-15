"""Tests for plumb stop and report CLI commands."""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from plumb.cli import main
from plumb.registry import SessionInfo


def _make_session(pid: int = 12345, tmp_path: Path | None = None) -> tuple[SessionInfo, Path]:
    """Return a SessionInfo and a snapshot path written to tmp_path."""
    snap_path = (tmp_path or Path("/tmp")) / f"{pid}_snapshot.json"
    session = SessionInfo(
        pid=pid,
        model_name="MixtralForCausalLM",
        n_layers=8,
        socket_path="",
        started_at=time.time() - 120.0,
        snapshot_path=str(snap_path),
    )
    return session, snap_path


def _write_snapshot(snap_path: Path, pid: int = 12345, pass_count: int = 100) -> None:
    # Use expert_counts — the field required by generate_report_from_snapshot (≥0.1.0)
    expert_counts = {f"{lid}:{eid}": (eid + 1) * 10 for lid in range(4) for eid in range(8)}
    payload = {
        "pid": pid,
        "model_name": "MixtralForCausalLM",
        "n_layers": 8,
        "pass_count": pass_count,
        "updated_at": time.time(),
        "started_at": time.time() - 120.0,
        "imbalance": [
            {"layer_id": lid, "ratio": 3.0, "max_expert": 7}
            for lid in range(4)
        ],
        "expert_counts": expert_counts,
        "gpu_to_numa": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 1, "5": 1, "6": 1, "7": 1},
    }
    snap_path.write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------

class TestStop:
    def test_stop_no_sessions(self):
        runner = CliRunner()
        with patch("plumb.cli.list_sessions", return_value=[]):
            result = runner.invoke(main, ["stop"])
        assert result.exit_code == 1
        assert "No active" in result.output

    def test_stop_wrong_pid(self):
        runner = CliRunner()
        session, _ = _make_session(99999)
        with patch("plumb.cli.list_sessions", return_value=[session]):
            result = runner.invoke(main, ["stop", "--pid", "12345"])
        assert result.exit_code == 1
        assert "No active session for PID 12345" in result.output

    def test_stop_sends_sigterm_and_deregisters(self, tmp_path):
        runner = CliRunner()
        session, snap_path = _make_session(12345, tmp_path)
        registry_file = tmp_path / "12345.json"
        registry_file.write_text("{}")  # simulate registered session

        kill_calls = []

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))
            if sig == signal.SIGTERM:
                registry_file.unlink(missing_ok=True)  # process exits, removes registry file

        with (
            patch("plumb.cli.list_sessions", return_value=[session]),
            patch("plumb.cli.REGISTRY_DIR", tmp_path),
            patch("plumb.cli.deregister") as mock_deregister,
            patch("plumb.cli.os.kill", side_effect=fake_kill),
        ):
            result = runner.invoke(main, ["stop", "--pid", "12345"])

        assert result.exit_code == 0
        assert "stopped" in result.output
        assert (12345, signal.SIGTERM) in kill_calls
        mock_deregister.assert_called_once_with(12345)

    def test_stop_falls_back_to_sigkill_when_process_lingers(self, tmp_path):
        runner = CliRunner()
        session, _ = _make_session(12345, tmp_path)
        registry_file = tmp_path / "12345.json"
        registry_file.write_text("{}")

        kill_calls = []

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))
            # Never remove the registry file — process never exits

        with (
            patch("plumb.cli.list_sessions", return_value=[session]),
            patch("plumb.cli.REGISTRY_DIR", tmp_path),
            patch("plumb.cli.deregister"),
            patch("plumb.cli.os.kill", side_effect=fake_kill),
            # Speed up the 10 s wait
            patch("plumb.cli.time.monotonic", side_effect=[0.0, 11.0, 11.0]),
            patch("plumb.cli.time.sleep"),
        ):
            result = runner.invoke(main, ["stop", "--pid", "12345"])

        assert (12345, signal.SIGKILL) in kill_calls

    def test_stop_already_gone_process(self, tmp_path):
        runner = CliRunner()
        session, _ = _make_session(12345, tmp_path)

        with (
            patch("plumb.cli.list_sessions", return_value=[session]),
            patch("plumb.cli.REGISTRY_DIR", tmp_path),
            patch("plumb.cli.deregister") as mock_deregister,
            patch("plumb.cli.os.kill", side_effect=ProcessLookupError),
        ):
            result = runner.invoke(main, ["stop", "--pid", "12345"])

        assert result.exit_code == 0
        assert "already gone" in result.output
        mock_deregister.assert_called_once_with(12345)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_no_active_session_exits(self, tmp_path):
        runner = CliRunner()
        with patch("plumb.cli.list_sessions", return_value=[]):
            result = runner.invoke(main, ["report"])
        assert result.exit_code == 1
        assert "No active sessions" in result.output

    def test_report_generates_json_and_prints_summary(self, tmp_path):
        runner = CliRunner()
        session, snap_path = _make_session(12345, tmp_path)
        _write_snapshot(snap_path)
        out_file = tmp_path / "out.json"

        with patch("plumb.cli.list_sessions", return_value=[session]):
            result = runner.invoke(main, ["report", "--pid", "12345", "--output", str(out_file)])

        assert result.exit_code == 0, result.output
        assert out_file.exists()
        report_data = json.loads(out_file.read_text())
        assert report_data["model_name"] == "MixtralForCausalLM"
        assert "layers" in report_data
        assert len(report_data["layers"]) == 4
        assert "Report written" in result.output

    def test_report_validates_against_schema(self, tmp_path):
        from plumb.report.schema import ProfileReport

        runner = CliRunner()
        session, snap_path = _make_session(12345, tmp_path)
        _write_snapshot(snap_path)
        out_file = tmp_path / "out.json"

        with patch("plumb.cli.list_sessions", return_value=[session]):
            result = runner.invoke(main, ["report", "--pid", "12345", "--output", str(out_file)])

        assert result.exit_code == 0, result.output
        report = ProfileReport.model_validate_json(out_file.read_text())
        assert report.total_forward_passes == 100
        assert len(report.layers) == 4

    def test_report_missing_expert_counts_exits(self, tmp_path):
        runner = CliRunner()
        session, snap_path = _make_session(12345, tmp_path)
        # Snapshot without expert_counts — pre-0.1.0 format
        snap_path.write_text(json.dumps({"pid": 12345, "model_name": "X", "pass_count": 10}))

        with patch("plumb.cli.list_sessions", return_value=[session]):
            result = runner.invoke(main, ["report", "--pid", "12345"])

        assert result.exit_code == 1
        assert "expert_counts" in result.output

    def test_report_reads_snapshot_by_path(self, tmp_path):
        snap_path = tmp_path / "snap.json"
        _write_snapshot(snap_path)
        out_file = tmp_path / "out.json"

        result = CliRunner().invoke(main, ["report", "--snapshot", str(snap_path), "--output", str(out_file)])

        assert result.exit_code == 0, result.output
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["model_name"] == "MixtralForCausalLM"
