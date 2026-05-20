"""Tests for the subprocess launcher (signal forwarding, exit code propagation)."""
import signal
import subprocess
import sys
import textwrap
import time


def test_sigterm_forwarded_to_child(tmp_path):
    """SIGTERM sent to the launcher is forwarded to the child; launch() returns promptly."""
    script = tmp_path / "helper.py"
    script.write_text(textwrap.dedent("""\
        import sys
        from plumb.launcher import launch
        rc = launch([sys.executable, "-c", "import time; time.sleep(30)"])
        print(f"rc={rc}", flush=True)
    """))

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)

    try:
        stdout, _ = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise AssertionError(
            "launch() did not return within 5s after SIGTERM — child signal was not forwarded"
        )

    assert "rc=" in stdout, f"Expected 'rc=<n>' in stdout, got: {stdout!r}"
