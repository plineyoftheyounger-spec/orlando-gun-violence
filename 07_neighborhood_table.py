"""
07_neighborhood_table.py
────────────────────────
Generates output/maps/neighborhood_table.html — a searchable, sortable
DataTables page showing killed/injured by neighborhood, era, and year.

Eras match 05_era_maps.ipynb:
  Era 1: 2018–2022
  Era 2: 2023–Present
"""

import json
import pandas as pd
import config

ERA1_YEARS = list(range(2018, 2023))
ERA2_YEARS = list(range(2023, 2027))
ERA1_LABEL = "2018–2022"
ERA2_LABEL = "2023–Present"

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(config.DATA_PROCESSED / "orlando_incidents_with_neighborhoods.csv")
df["neighborhood"] = df["neighborhood"].fillna("Unknown / Outside Neighborhoods")

all_years = sorted(df["year"].unique())

# ── Aggregate ─────────────────────────────────────────────────────────────────
def agg(sub):
    return pd.Series({
        "killed": int(sub["killed"].sum()),
        "injured": int(sub["injured"].sum()),
        "incidents": len(sub),
    })

# Era summaries
era1 = df[df["year"].isin(ERA1_YEARS)].groupby("neighborhood").apply(agg).reset_index()
era2 = df[df["year"].isin(ERA2_YEARS)].groupby("neighborhood").apply(agg).reset_index()

# Year-by-year
by_year = df.groupby(["neighborhood", "year"]).apply(agg).reset_index()

# All neighborhoods (union)
all_nbds = sorted(df["neighborhood"].unique())

# ── Build row dicts ────────────────────────────────────────────────────────────
rows = []
for nbd in all_nbds:
    e1 = era1[era1["neighborhood"] == nbd].iloc[0] if len(era1[era1["neighborhood"] == nbd]) else None
    e2 = era2[era2["neighborhood"] == nbd].iloc[0] if len(era2[era2["neighborhood"] == nbd]) else None

    row = {
        "neighborhood": nbd,
        "e1_killed":    int(e1["killed"])    if e1 is not None else 0,
        "e1_injured":   int(e1["injured"])   if e1 is not None else 0,
        "e1_incidents": int(e1["incidents"]) if e1 is not None else 0,
        "e2_killed":    int(e2["killed"])    if e2 is not None else 0,
        "e2_injured":   int(e2["injured"])   if e2 is not None else 0,
        "e2_incidents": int(e2["incidents"]) if e2 is not None else 0,
    }
    for yr in all_years:
        yr_row = by_year[(by_year["neighborhood"] == nbd) & (by_year["year"] == yr)]
        row[f"{yr}_killed"]  = int(yr_row["killed"].iloc[0])  if len(yr_row) else 0
        row[f"{yr}_injured"] = int(yr_row["injured"].iloc[0]) if len(yr_row) else 0

    rows.append(row)

# ── Build column header HTML ───────────────────────────────────────────────────
era_headers = f"""
    <th rowspan="2" class="nbd-col">Neighborhood</th>
    <th colspan="3" class="era-header era1-header">{ERA1_LABEL}</th>
    <th colspan="3" class="era-header era2-header">{ERA2_LABEL}</th>
    {"".join(f'<th colspan="2" class="year-header">{yr}</th>' for yr in all_years)}
"""

sub_headers = """
    <th class="era1">Killed</th><th class="era1">Injured</th><th class="era1">Incidents</th>
    <th class="era2">Killed</th><th class="era2">Injured</th><th class="era2">Incidents</th>
""" + "".join(
    f'<th class="yr-col">Killed</th><th class="yr-col">Injured</th>'
    for _ in all_years
)

# ── Build table rows HTML ──────────────────────────────────────────────────────
def cell(val, cls=""):
    display = str(val) if val else '<span class="zero">—</span>'
    return f'<td class="{cls}">{display}</td>'

tbody_rows = []
for r in rows:
    cells = f'<td class="nbd-name">{r["neighborhood"]}</td>'
    cells += cell(r["e1_killed"],    "era1 killed")
    cells += cell(r["e1_injured"],   "era1")
    cells += cell(r["e1_incidents"], "era1")
    cells += cell(r["e2_killed"],    "era2 killed")
    cells += cell(r["e2_injured"],   "era2")
    cells += cell(r["e2_incidents"], "era2")
    for yr in all_years:
        cells += cell(r[f"{yr}_killed"],  "yr-col killed")
        cells += cell(r[f"{yr}_injured"], "yr-col")
    tbody_rows.append(f"<tr>{cells}</tr>")

tbody = "\n".join(tbody_rows)

# ── Assemble HTML ──────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Orlando Gun Violence — Neighborhood Table</title>
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
  <link rel="stylesheet" href="https://cdn.datatables.net/fixedcolumns/4.3.0/css/fixedColumns.dataTables.min.css"/>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f0f0f;
      color: #e0e0e0;
      margin: 0;
      padding: 20px 24px;
    }}
    h1 {{ font-size: 1.3rem; margin: 0 0 4px; color: #fff; }}
    p.sub {{ color: #888; font-size: 0.85rem; margin: 0 0 20px; }}

    .dataTables_wrapper .dataTables_filter input {{
      background: #1e1e1e;
      border: 1px solid #444;
      color: #e0e0e0;
      border-radius: 4px;
      padding: 4px 8px;
    }}
    .dataTables_wrapper .dataTables_length select {{
      background: #1e1e1e;
      border: 1px solid #444;
      color: #e0e0e0;
    }}
    .dataTables_wrapper .dataTables_info,
    .dataTables_wrapper .dataTables_paginate {{ color: #888; font-size: 0.82rem; }}
    .dataTables_wrapper .dataTables_paginate .paginate_button {{ color: #aaa !important; }}
    .dataTables_wrapper .dataTables_paginate .paginate_button.current {{ background: #2a2a2a !important; color: #fff !important; border: 1px solid #555 !important; }}

    table.dataTable {{
      background: #141414;
      border-collapse: collapse;
      width: 100%;
    }}
    table.dataTable thead tr:first-child th {{
      background: #1a1a1a;
      color: #ccc;
      border-bottom: 1px solid #333;
      text-align: center;
      font-size: 0.78rem;
      white-space: nowrap;
      padding: 8px 10px;
    }}
    table.dataTable thead tr:nth-child(2) th {{
      background: #1e1e1e;
      color: #999;
      font-size: 0.72rem;
      text-align: center;
      padding: 4px 8px;
      border-bottom: 2px solid #333;
    }}
    table.dataTable tbody tr {{ background: #141414; }}
    table.dataTable tbody tr:hover {{ background: #1c1c1c; }}
    table.dataTable tbody tr:nth-child(even) {{ background: #181818; }}
    table.dataTable tbody tr:nth-child(even):hover {{ background: #1c1c1c; }}
    table.dataTable td, table.dataTable th {{
      border-right: 1px solid #2a2a2a;
      padding: 5px 8px;
      font-size: 0.82rem;
    }}

    .nbd-col  {{ min-width: 180px; text-align: left !important; }}
    .nbd-name {{ color: #d0d0d0; white-space: nowrap; }}

    .era1-header {{ background: #1a2840 !important; color: #7ab3f0 !important; }}
    .era2-header {{ background: #1a2d1a !important; color: #7acc7a !important; }}
    .year-header  {{ background: #1e1e1e !important; color: #aaa !important; }}

    th.era1, td.era1 {{ background-color: rgba(100,160,240,0.05); }}
    th.era2, td.era2 {{ background-color: rgba(100,200,100,0.05); }}

    td.killed {{ color: #e07070; font-weight: 600; }}
    td.zero   {{ color: #444; }}
    span.zero {{ color: #444; }}
  </style>
</head>
<body>
  <h1>Orlando Gun Violence — By Neighborhood</h1>
  <p class="sub">Gun Violence Archive · 2014–2026 · Orlando, FL &nbsp;|&nbsp; Eras: {ERA1_LABEL} vs {ERA2_LABEL}</p>

  <div style="overflow-x:auto;">
  <table id="nbd-table" class="display nowrap" style="width:100%">
    <thead>
      <tr>{era_headers}</tr>
      <tr>{sub_headers}</tr>
    </thead>
    <tbody>
{tbody}
    </tbody>
  </table>
  </div>

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/fixedcolumns/4.3.0/js/dataTables.fixedColumns.min.js"></script>
  <script>
    $(document).ready(function() {{
      $('#nbd-table').DataTable({{
        paging: true,
        pageLength: 25,
        scrollX: true,
        fixedColumns: {{ leftColumns: 1 }},
        order: [[1, 'desc']],
        columnDefs: [{{ targets: 0, orderable: true }}],
        language: {{
          search: "Search neighborhoods:"
        }}
      }});
    }});
  </script>
</body>
</html>
"""

out = config.OUTPUT_MAPS / "neighborhood_table.html"
out.write_text(html, encoding="utf-8")
print(f"Saved: {out}")
print(f"  {len(rows)} neighborhoods | {len(all_years)} years")
