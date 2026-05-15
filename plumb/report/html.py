"""Generate a self-contained HTML report from a ProfileReport."""
from __future__ import annotations

import re
from pathlib import Path

from .schema import ProfileReport

_TEMPLATE = Path(__file__).parent.parent / "dashboard" / "static" / "index.html"

# Anchors in index.html that bracket the live-fetch function.
_LIVE_START = "/*LIVE_START*/"
_LIVE_END = "/*LIVE_END*/"

# The two poll lines at the end of the script block.
_POLL_LINES = "load();\nsetInterval(load, POLL_MS);"

_STATIC_BLOCK = """\
// --- static export: no polling ---
const _STATIC_DATA = {data_json};
function load() {{ render(_STATIC_DATA); }}
load();"""


def generate_html_report(profile: ProfileReport) -> str:
    """Return a self-contained HTML string embedding the ProfileReport data."""
    html = _TEMPLATE.read_text(encoding="utf-8")
    data_json = profile.model_dump_json()

    # Strip the live fetch function (between the anchor comments)
    if _LIVE_START not in html or _LIVE_END not in html:
        raise RuntimeError(
            "Dashboard template missing LIVE_START/LIVE_END markers. "
            "Update plumb/report/html.py to match the new template."
        )
    html = re.sub(
        re.escape(_LIVE_START) + r".*?" + re.escape(_LIVE_END),
        "",
        html,
        flags=re.DOTALL,
    )

    # Replace the poll invocation with inline data + static render call
    if _POLL_LINES not in html:
        raise RuntimeError(
            "Dashboard template missing poll invocation. "
            "Update plumb/report/html.py to match the new template."
        )
    html = html.replace(_POLL_LINES, _STATIC_BLOCK.format(data_json=data_json))

    return html
