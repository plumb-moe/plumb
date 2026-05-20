from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .registry import SessionInfo, list_sessions


@dataclass
class GpuProcess:
    pid: int
    gpu_memory_mb: int
    cmdline: str
    detected_model: str | None
    session: SessionInfo | None


def scan_gpu_processes() -> list[GpuProcess]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    sessions_by_pid = {s.pid: s for s in list_sessions()}

    processes: list[GpuProcess] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0].strip())
            mem_mb = int(parts[1].strip())
        except ValueError:
            continue

        cmdline = _read_cmdline(pid)
        processes.append(
            GpuProcess(
                pid=pid,
                gpu_memory_mb=mem_mb,
                cmdline=cmdline,
                detected_model=_detect_model_from_cmdline(cmdline),
                session=sessions_by_pid.get(pid),
            )
        )

    return sorted(processes, key=lambda p: p.gpu_memory_mb, reverse=True)


def _read_cmdline(pid: int) -> str:
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
        return data.replace(b"\x00", b" ").decode(errors="replace").strip()
    except (PermissionError, FileNotFoundError):
        return ""


def _detect_model_from_cmdline(cmdline: str) -> str | None:
    lower = cmdline.lower()

    m = re.search(r"mixtral.*?8x(\d+)b|8x(\d+)b", lower)
    if m:
        n = m.group(1) or m.group(2)
        return f"Mixtral-8x{n}B"

    if re.search(r"olmoe", lower):
        return "OLMoE"

    if re.search(r"qwen.*moe", lower):
        return "Qwen-MoE"

    if re.search(r"deepseek.*v3", lower):
        return "DeepSeek-V3"

    if re.search(r"deepseek.*v2", lower):
        return "DeepSeek-V2"

    if re.search(r"deepseek.*moe", lower):
        return "DeepSeek-MoE"

    if re.search(r"phi.*moe|phimoe", lower):
        return "Phi-MoE"

    return None
