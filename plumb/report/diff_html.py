from __future__ import annotations

from ..diff import DiffResult

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>plumb diff</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; padding: 24px; }}
  h1 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; }}
  .meta {{ color: #64748b; font-size: 0.85rem; margin-bottom: 24px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 32px; }}
  .card {{ background: #1e2130; border-radius: 8px; padding: 16px 20px; min-width: 180px; }}
  .card-label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em; }}
  .card-value {{ font-size: 1.4rem; font-weight: 700; margin-top: 4px; }}
  .card-value.warn {{ color: #f59e0b; }}
  .card-value.ok   {{ color: #34d399; }}
  .card-value.better {{ color: #34d399; }}
  .card-value.worse  {{ color: #f87171; }}
  h2 {{ font-size: 1rem; font-weight: 600; margin-bottom: 12px; color: #94a3b8;
        text-transform: uppercase; letter-spacing: .05em; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ text-align: left; padding: 6px 10px; color: #64748b; border-bottom: 1px solid #2d3748;
        cursor: pointer; user-select: none; }}
  th:hover {{ color: #94a3b8; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #1a1f2e; }}
  tr:hover td {{ background: #1e2130; }}
  .pos {{ color: #f87171; }}
  .neg {{ color: #34d399; }}
  .zero {{ color: #475569; }}
  .filter-row {{ margin-bottom: 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  input[type=text] {{ background: #1e2130; color: #e2e8f0; border: 1px solid #2d3748;
                      border-radius: 4px; padding: 6px 10px; font-size: 0.85rem; width: 200px; }}
  input[type=text]:focus {{ outline: none; border-color: #4f6ef7; }}
  label {{ color: #64748b; font-size: 0.8rem; }}
  .section {{ margin-bottom: 40px; }}
</style>
</head>
<body>
<h1>plumb diff</h1>
<div class="meta">{meta_a} &rarr; {meta_b}</div>

<div class="cards">
  <div class="card">
    <div class="card-label">Mean Imbalance (before)</div>
    <div class="card-value warn">{mean_before}&times;</div>
  </div>
  <div class="card">
    <div class="card-label">Mean Imbalance (after)</div>
    <div class="card-value {mean_cls}">{mean_after}&times;</div>
  </div>
  <div class="card">
    <div class="card-label">Max Imbalance (before)</div>
    <div class="card-value warn">{max_before}&times;</div>
  </div>
  <div class="card">
    <div class="card-label">Max Imbalance (after)</div>
    <div class="card-value {max_cls}">{max_after}&times;</div>
  </div>
  {ttft_cards}
</div>

<div class="section">
<h2>Per-Expert Delta</h2>
<div class="filter-row">
  <label>Filter layer: <input type="text" id="filter-layer" placeholder="e.g. 0, 3"></label>
  <label>Filter expert: <input type="text" id="filter-expert" placeholder="e.g. 5"></label>
  <label>Min |delta%|: <input type="text" id="filter-pct" placeholder="e.g. 10"></label>
</div>
<table id="delta-table">
  <thead>
    <tr>
      <th onclick="sortBy('layer')">Layer &uarr;&darr;</th>
      <th onclick="sortBy('expert')">Expert &uarr;&darr;</th>
      <th onclick="sortBy('before')">Tokens Before &uarr;&darr;</th>
      <th onclick="sortBy('after')">Tokens After &uarr;&darr;</th>
      <th onclick="sortBy('delta')">Delta &uarr;&darr;</th>
      <th onclick="sortBy('pct')">Delta % &uarr;&darr;</th>
    </tr>
  </thead>
  <tbody id="table-body"></tbody>
</table>
</div>

<script>
const RAW = {rows_json};

let sortKey = 'pct';
let sortDir = -1;

function render() {{
  const layerFilter  = document.getElementById('filter-layer').value.trim();
  const expertFilter = document.getElementById('filter-expert').value.trim();
  const pctFilter    = parseFloat(document.getElementById('filter-pct').value) || 0;

  let rows = RAW.filter(r => {{
    if (layerFilter  && String(r.layer)  !== layerFilter)  return false;
    if (expertFilter && String(r.expert) !== expertFilter) return false;
    if (Math.abs(r.pct) < pctFilter) return false;
    return true;
  }});

  rows.sort((a, b) => {{
    const av = a[sortKey], bv = b[sortKey];
    return sortDir * (av < bv ? -1 : av > bv ? 1 : 0);
  }});

  const tbody = document.getElementById('table-body');
  tbody.innerHTML = rows.map(r => {{
    const cls = r.delta > 0 ? 'pos' : r.delta < 0 ? 'neg' : 'zero';
    const sign = r.delta > 0 ? '+' : '';
    return `<tr>
      <td>${{r.layer}}</td>
      <td>${{r.expert}}</td>
      <td>${{r.before.toLocaleString()}}</td>
      <td>${{r.after.toLocaleString()}}</td>
      <td class="${{cls}}">${{sign}}${{r.delta.toLocaleString()}}</td>
      <td class="${{cls}}">${{sign}}${{r.pct.toFixed(2)}}%</td>
    </tr>`;
  }}).join('');
}}

function sortBy(key) {{
  if (sortKey === key) sortDir = -sortDir;
  else {{ sortKey = key; sortDir = -1; }}
  render();
}}

['filter-layer', 'filter-expert', 'filter-pct'].forEach(id => {{
  document.getElementById(id).addEventListener('input', render);
}});

render();
</script>
</body>
</html>
"""

_TTFT_CARD = """\
  <div class="card">
    <div class="card-label">Est. TTFT Improvement (before)</div>
    <div class="card-value warn">~{before:.0f}%</div>
  </div>
  <div class="card">
    <div class="card-label">Est. TTFT Improvement (after)</div>
    <div class="card-value {cls}">~{after:.0f}%</div>
  </div>"""


def render_diff_html(result: DiffResult) -> str:
    import json

    rows = [
        {
            "layer":  d.layer_id,
            "expert": d.expert_id,
            "before": d.token_count_before,
            "after":  d.token_count_after,
            "delta":  d.delta,
            "pct":    d.delta_pct,
        }
        for d in result.expert_deltas
    ]

    mean_better = result.mean_imbalance_after <= result.mean_imbalance_before
    max_better  = result.max_imbalance_after  <= result.max_imbalance_before

    ttft_cards = ""
    if result.ttft_est_before is not None and result.ttft_est_after is not None:
        ttft_cls = "better" if result.ttft_est_after >= result.ttft_est_before else "worse"
        ttft_cards = _TTFT_CARD.format(
            before=result.ttft_est_before,
            after=result.ttft_est_after,
            cls=ttft_cls,
        )

    return _TEMPLATE.format(
        meta_a=result.model_name_a,
        meta_b=result.model_name_b,
        mean_before=f"{result.mean_imbalance_before:.4f}",
        mean_after=f"{result.mean_imbalance_after:.4f}",
        mean_cls="better" if mean_better else "worse",
        max_before=f"{result.max_imbalance_before:.4f}",
        max_after=f"{result.max_imbalance_after:.4f}",
        max_cls="better" if max_better else "worse",
        ttft_cards=ttft_cards,
        rows_json=json.dumps(rows),
    )
