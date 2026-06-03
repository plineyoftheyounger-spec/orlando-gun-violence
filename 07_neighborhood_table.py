"""
07_neighborhood_table.py
────────────────────────
Generates output/maps/neighborhood_table.html — a searchable, sortable
DataTables page showing killed/injured by neighborhood, era, and year.

Eras match 05_era_maps.ipynb:
  Era 1: 2018–2022
  Era 2: 2023–Present
"""

import pandas as pd
import geopandas as gpd
import config

ERA1_YEARS = list(range(2018, 2023))
ERA2_YEARS = list(range(2023, 2027))
ERA1_LABEL = "2018–2022"
ERA2_LABEL = "2023–Present"

UNKNOWN_LABEL = "Unknown / Outside Neighborhoods"

# ── Load incidents ─────────────────────────────────────────────────────────────
df = pd.read_csv(config.DATA_PROCESSED / "orlando_incidents_with_neighborhoods.csv")
df["neighborhood"] = df["neighborhood"].fillna(UNKNOWN_LABEL)

all_years = sorted(df["year"].unique())

# ── Build Kidz Zone → neighborhood mapping via spatial join ────────────────────
nbds_gdf = gpd.read_file(config.NEIGHBORHOODS_DIR / "orlando_neighborhoods.geojson")
kz_gdf   = gpd.read_file(config.NEIGHBORHOODS_DIR / "kidz_zones_official.geojson")
kz_gdf   = kz_gdf.to_crs(nbds_gdf.crs)

kz_join = gpd.sjoin(
    nbds_gdf[["NeighborhoodName", "geometry"]],
    kz_gdf[["KZ_Name", "geometry"]],
    how="inner", predicate="intersects"
)[["NeighborhoodName", "KZ_Name"]].drop_duplicates()

# Collapse multiple KZ hits per neighborhood into one label
kz_map = (
    kz_join.groupby("NeighborhoodName")["KZ_Name"]
    .apply(lambda x: " / ".join(sorted(set(x.str.replace(" Kidz Zone", "", regex=False)))))
    .to_dict()
)

# ── Aggregate ─────────────────────────────────────────────────────────────────
def agg(sub):
    return pd.Series({
        "killed":    int(sub["killed"].sum()),
        "injured":   int(sub["injured"].sum()),
        "incidents": len(sub),
    })

era1    = df[df["year"].isin(ERA1_YEARS)].groupby("neighborhood").apply(agg, include_groups=False).reset_index()
era2    = df[df["year"].isin(ERA2_YEARS)].groupby("neighborhood").apply(agg, include_groups=False).reset_index()
by_year = df.groupby(["neighborhood", "year"]).apply(agg, include_groups=False).reset_index()

# ── Build row dicts ────────────────────────────────────────────────────────────
UNKNOWN = UNKNOWN_LABEL
all_nbds = sorted([n for n in df["neighborhood"].unique() if n != UNKNOWN])

rows = []
for nbd in all_nbds:
    e1 = era1[era1["neighborhood"] == nbd]
    e2 = era2[era2["neighborhood"] == nbd]

    row = {
        "neighborhood": nbd,
        "kidz_zone":    kz_map.get(nbd, ""),
        "e1_killed":    int(e1["killed"].iloc[0])    if len(e1) else 0,
        "e1_injured":   int(e1["injured"].iloc[0])   if len(e1) else 0,
        "e1_incidents": int(e1["incidents"].iloc[0]) if len(e1) else 0,
        "e2_killed":    int(e2["killed"].iloc[0])    if len(e2) else 0,
        "e2_injured":   int(e2["injured"].iloc[0])   if len(e2) else 0,
        "e2_incidents": int(e2["incidents"].iloc[0]) if len(e2) else 0,
        "is_unknown": False,
    }
    for yr in all_years:
        yr_row = by_year[(by_year["neighborhood"] == nbd) & (by_year["year"] == yr)]
        row[f"{yr}_killed"]  = int(yr_row["killed"].iloc[0])  if len(yr_row) else 0
        row[f"{yr}_injured"] = int(yr_row["injured"].iloc[0]) if len(yr_row) else 0

    rows.append(row)

# Unknown row goes at the end
u_e1 = era1[era1["neighborhood"] == UNKNOWN]
u_e2 = era2[era2["neighborhood"] == UNKNOWN]
unknown_row = {
    "neighborhood": UNKNOWN,
    "kidz_zone":    "",
    "e1_killed":    int(u_e1["killed"].iloc[0])    if len(u_e1) else 0,
    "e1_injured":   int(u_e1["injured"].iloc[0])   if len(u_e1) else 0,
    "e1_incidents": int(u_e1["incidents"].iloc[0]) if len(u_e1) else 0,
    "e2_killed":    int(u_e2["killed"].iloc[0])    if len(u_e2) else 0,
    "e2_injured":   int(u_e2["injured"].iloc[0])   if len(u_e2) else 0,
    "e2_incidents": int(u_e2["incidents"].iloc[0]) if len(u_e2) else 0,
    "is_unknown":   True,
}
for yr in all_years:
    yr_row = by_year[(by_year["neighborhood"] == UNKNOWN) & (by_year["year"] == yr)]
    unknown_row[f"{yr}_killed"]  = int(yr_row["killed"].iloc[0])  if len(yr_row) else 0
    unknown_row[f"{yr}_injured"] = int(yr_row["injured"].iloc[0]) if len(yr_row) else 0
rows.append(unknown_row)

# ── Build column headers ───────────────────────────────────────────────────────
era_headers = (
    '<th rowspan="2" class="nbd-col">Neighborhood</th>'
    '<th rowspan="2" class="kz-col">Kidz Zone</th>'
    f'<th colspan="3" class="era-header era1-header">{ERA1_LABEL}</th>'
    f'<th colspan="3" class="era-header era2-header">{ERA2_LABEL}</th>'
    + "".join(f'<th colspan="2" class="year-header">{yr}</th>' for yr in all_years)
)

sub_headers = (
    '<th class="kz-sub"></th>'
    '<th class="era1">Killed</th><th class="era1">Injured</th><th class="era1">Incidents</th>'
    '<th class="era2">Killed</th><th class="era2">Injured</th><th class="era2">Incidents</th>'
    + "".join('<th class="yr-col">Killed</th><th class="yr-col">Injured</th>' for _ in all_years)
)

# ── Build table rows HTML ──────────────────────────────────────────────────────
def cell(val, cls=""):
    display = str(val) if val else '<span class="zero">—</span>'
    return f'<td class="{cls}">{display}</td>'

tbody_rows = []
for r in rows:
    row_cls = "unknown-row" if r["is_unknown"] else ("kz-row" if r["kidz_zone"] else "")
    kz_label = f'<span class="kz-badge">{r["kidz_zone"]}</span>' if r["kidz_zone"] else ""
    cells  = f'<td class="nbd-name">{r["neighborhood"]}</td>'
    cells += f'<td class="kz-cell">{kz_label}</td>'
    cells += cell(r["e1_killed"],    "era1 killed")
    cells += cell(r["e1_injured"],   "era1")
    cells += cell(r["e1_incidents"], "era1")
    cells += cell(r["e2_killed"],    "era2 killed")
    cells += cell(r["e2_injured"],   "era2")
    cells += cell(r["e2_incidents"], "era2")
    for yr in all_years:
        cells += cell(r[f"{yr}_killed"],  "yr-col killed")
        cells += cell(r[f"{yr}_injured"], "yr-col")
    tbody_rows.append(f'<tr class="{row_cls}">{cells}</tr>')

tbody = "\n".join(tbody_rows)

kz_count = sum(1 for r in rows if r["kidz_zone"])

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
      background: #fff;
      color: #1a1a1a;
      margin: 0;
      padding: 20px 24px;
    }}
    h1 {{ font-size: 1.3rem; margin: 0 0 4px; color: #111; }}
    p.sub {{ color: #666; font-size: 0.85rem; margin: 0 0 20px; }}

    .dataTables_wrapper .dataTables_filter input {{
      border: 1px solid #ccc;
      border-radius: 4px;
      padding: 4px 8px;
      color: #111;
    }}
    .dataTables_wrapper .dataTables_info,
    .dataTables_wrapper .dataTables_paginate {{ color: #666; font-size: 0.82rem; }}
    .dataTables_wrapper .dataTables_paginate .paginate_button {{ color: #333 !important; }}
    .dataTables_wrapper .dataTables_paginate .paginate_button.current {{
      background: #e8e8e8 !important; color: #111 !important; border: 1px solid #bbb !important;
    }}

    table.dataTable {{
      background: #fff;
      border-collapse: collapse;
      width: 100%;
    }}
    table.dataTable thead tr:first-child th {{
      background: #f0f0f0;
      color: #222;
      border-bottom: 1px solid #ccc;
      text-align: center;
      font-size: 0.78rem;
      white-space: nowrap;
      padding: 8px 10px;
    }}
    table.dataTable thead tr:nth-child(2) th {{
      background: #f7f7f7;
      color: #555;
      font-size: 0.72rem;
      text-align: center;
      padding: 4px 8px;
      border-bottom: 2px solid #ccc;
    }}
    table.dataTable tbody tr {{ background: #fff; }}
    table.dataTable tbody tr:hover {{ background: #f5f8ff; }}
    table.dataTable tbody tr:nth-child(even) {{ background: #fafafa; }}
    table.dataTable tbody tr:nth-child(even):hover {{ background: #f5f8ff; }}
    table.dataTable td, table.dataTable th {{
      border-right: 1px solid #e0e0e0;
      border-bottom: 1px solid #e8e8e8;
      padding: 5px 8px;
      font-size: 0.82rem;
    }}

    .nbd-col  {{ min-width: 180px; text-align: left !important; }}
    .kz-col   {{ min-width: 120px; text-align: left !important; }}
    .kz-sub   {{ min-width: 120px; }}
    .nbd-name {{ color: #222; white-space: nowrap; font-weight: 500; }}

    .era1-header {{ background: #dbeafe !important; color: #1e40af !important; }}
    .era2-header {{ background: #dcfce7 !important; color: #166534 !important; }}
    .year-header  {{ background: #f0f0f0 !important; color: #555 !important; }}

    th.era1, td.era1 {{ background-color: rgba(59,130,246,0.04); }}
    th.era2, td.era2 {{ background-color: rgba(34,197,94,0.04); }}

    td.killed {{ color: #c0392b; font-weight: 600; }}
    td.zero   {{ color: #bbb; }}
    span.zero {{ color: #bbb; }}

    /* Kidz Zone rows */
    tr.kz-row td {{ background-color: #fffbeb !important; }}
    tr.kz-row:hover td {{ background-color: #fef3c7 !important; }}
    .kz-badge {{
      background: #f59e0b;
      color: #fff;
      font-size: 0.68rem;
      font-weight: 600;
      padding: 2px 6px;
      border-radius: 3px;
      white-space: nowrap;
    }}
    .kz-cell {{ white-space: nowrap; }}

    /* Unknown row — muted, italic */
    tr.unknown-row td {{
      color: #999 !important;
      font-style: italic;
      background-color: #f9f9f9 !important;
    }}
    tr.unknown-row td.killed {{ color: #c0a0a0 !important; }}
  </style>
</head>
<body>
  <h1>Orlando Gun Violence — By Neighborhood</h1>
  <p class="sub">
    Gun Violence Archive · 2014–2026 · Orlando, FL &nbsp;|&nbsp; Eras: {ERA1_LABEL} vs {ERA2_LABEL}
    &nbsp;|&nbsp; <span style="background:#f59e0b;color:#fff;font-size:0.75rem;padding:1px 6px;border-radius:3px;font-weight:600;">KZ</span> = Kidz Zone neighborhood ({kz_count} neighborhoods)
    &nbsp;|&nbsp; <em style="color:#999;">Italicized row</em> = incidents outside mapped boundaries
  </p>

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
      var table = $('#nbd-table').DataTable({{
        paging: true,
        pageLength: 25,
        scrollX: true,
        fixedColumns: {{ leftColumns: 2 }},
        order: [[2, 'desc']],
        columnDefs: [{{ targets: [0,1], orderable: true }}],
        language: {{ search: "Search neighborhoods:" }},
        rowCallback: function(row, data, index) {{
          // Keep unknown row always at the bottom regardless of sort
          if ($(row).hasClass('unknown-row')) {{
            $(row).data('order', 9999);
          }}
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
print(f"  {len(rows)-1} named neighborhoods + 1 unknown row | {len(all_years)} years")
print(f"  {kz_count} Kidz Zone neighborhoods: {sorted(kz_map.keys())}")
