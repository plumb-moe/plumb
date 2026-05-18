"""Tests for GPU process scanner."""
import subprocess
from unittest.mock import patch

from plumb.scanner import scan_gpu_processes


def test_returns_empty_when_nvidia_smi_missing():
    with patch("plumb.scanner.subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        assert scan_gpu_processes() == []


def test_returns_empty_when_nvidia_smi_exits_nonzero():
    with patch(
        "plumb.scanner.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
    ):
        assert scan_gpu_processes() == []
