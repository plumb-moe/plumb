import os

if os.environ.get("SAI_PROFILER_AUTO") == "1":
    try:
        from plumb.autoattach import start_background_profiler
        start_background_profiler()
    except Exception as exc:
        import sys
        print(f"[plumb] autoattach failed: {exc}", file=sys.stderr)
