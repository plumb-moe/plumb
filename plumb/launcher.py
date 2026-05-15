from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

_SITECUSTOMIZE_DIR = str(Path(__file__).parent / "_sitecustomize")
_PACKAGE_DIR = str(Path(__file__).parent.parent)


def launch(args: list[str], extra_env: dict[str, str] | None = None) -> int:
    """Run args as a subprocess with plumb auto-attach injected."""
    if not args:
        return 0

    env = os.environ.copy()
    env["SAI_PROFILER_AUTO"] = "1"
    if extra_env:
        env.update(extra_env)

    # Prepend sitecustomize dir and package root so plumb is importable
    # even if not installed in the target env.
    existing = env.get("PYTHONPATH", "")
    parts = [_SITECUSTOMIZE_DIR, _PACKAGE_DIR]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    try:
        proc = subprocess.Popen(args, env=env)

        # Forward SIGTERM to child so the wrapper can finish (write eplb output, etc.)
        # after the child dies. PEP 475 restarts proc.wait() after the handler returns.
        def _sigterm(sig, frame):
            proc.terminate()

        old_sigterm = signal.signal(signal.SIGTERM, _sigterm)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
        finally:
            signal.signal(signal.SIGTERM, old_sigterm)

        return proc.returncode
    except FileNotFoundError:
        print(f"plumb: command not found: {args[0]}", file=sys.stderr)
        return 127
