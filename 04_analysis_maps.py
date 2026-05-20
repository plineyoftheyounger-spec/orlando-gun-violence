"""
04_analysis_maps.py
────────────────────
Generates all analysis maps for the Orlando gun violence study.

PREREQUISITES:
  python 01_prepare_data.py    ← builds data/processed/orlando_incidents.csv
  python 00_fetch_boundaries.py ← downloads GeoJSON boundary files

OUTPUT FILES (all in output/maps/):
  Dot density by era:
    dot_all_by_era.html
    dot_homicides_by_era.html
    dot_injury_by_era.html

  Heat maps by era (smooth + grid-count toggle):
    heat_all_by_era.html
    heat_homicides_by_era.html
    heat_injury_by_era.html

  Neighborhood zooms (2018–present):
    zoom_parramore.html
    zoom_holden_heights.html
    zoom_mercy_drive.html
    zoom_carver_shores.html

  Kidz Zones analysis 2024–2025:
    kidz_zones_2024_2025.html
    kidz_zones_2024_2025_table.csv

HEAT MAP NOTE:
  Two modes are available (toggle in layer control):
  "Smooth" — Gaussian blur (KDE-style). Good for visual density patterns.
             max_val is fixed globally so a single isolated incident won't
             turn red when you zoom in.
  "Grid count" — Square ~200m cells, colored by exact incident count.
             Zoom-stable and shows actual numbers on hover. Better for
             comparing exact concentrations across neighborhoods.
"""

import json
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster, DualMap
import numpy as np
import branca.colormap as cm

import config

# ── Era definitions ─────────────────────────────────────────────────────────────
ERA_1_LABEL = "2018–2022"
ERA_1_YEARS = list(range(2018, 2023))
ERA_2_LABEL = "2023–Present"
ERA_2_YEARS = list(range(2023, 2026))

ERA_1_COLOR = "#2166ac"   # blue
ERA_2_COLOR = "#d6604d"   # red-orange

# Neighborhood display name → exact name in OrlandoPoliticalNeighborhoods dataset
NEIGHBORHOODS = {
    "Parramore":      "Holden/Parramore",
    "Holden Heights": "Holden Heights",
    "Mercy Drive":    "Mercy Drive",
    "Carver Shores":  "Carver Shores",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_incidents():
    path = config.DATA_PROCESSED / "orlando_incidents.csv"
    if not path.exists():
        raise FileNotFoundError("Run 01_prepare_data.py first.")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["lat", "lon"])
    df["year"] = df["date"].dt.year
    return df

def load_kidz_zones():
    path = config.NEIGHBORHOODS_DIR / "kidz_zones_official.geojson"
    if not path.exists():
        raise FileNotFoundError("Run 00_fetch_boundaries.py first.")
    return gpd.read_file(path).to_crs("EPSG:4326")

def load_neighborhoods():
    path = config.NEIGHBORHOODS_DIR / "orlando_neighborhoods.geojson"
    if not path.exists():
        raise FileNotFoundError("Run 00_fetch_boundaries.py first.")
    return gpd.read_file(path).to_crs("EPSG:4326")


# ═══════════════════════════════════════════════════════════════════════════════
# Incident filters
# ═══════════════════════════════════════════════════════════════════════════════

def era1(df):        return df[df["year"].isin(ERA_1_YEARS)]
def era2(df):        return df[df["year"].isin(ERA_2_YEARS)]
def homicides(df):   return df[df["killed"] > 0]
def injury_only(df): return df[(df["injured"] > 0) & (df["killed"] == 0)]


# ═══════════════════════════════════════════════════════════════════════════════
# Map component helpers
# ═══════════════════════════════════════════════════════════════════════════════

def base_map(zoom=config.DEFAULT_ZOOM):
    return folium.Map(
        location=[config.ORLANDO_LAT, config.ORLANDO_LON],
        zoom_start=zoom,
        tiles="CartoDB positron",
    )

def add_title(m, html_text):
    html = f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
         background:white;padding:8px 18px;border-radius:6px;border:1px solid #aaa;
         z-index:1000;font-family:Arial,sans-serif;font-size:14px;text-align:center;
         white-space:nowrap;">
        {html_text}
    </div>"""
    m.get_root().html.add_child(folium.Element(html))

def add_legend(m, items):
    rows = "".join(
        f'<div><span style="color:{c};font-size:18px;line-height:1.3;">●</span>&nbsp;{label}</div>'
        for label, c in items
    )
    html = f"""
    <div style="position:fixed;bottom:30px;left:30px;background:white;
         padding:10px 14px;border-radius:6px;border:1px solid #aaa;
         z-index:1000;font-family:Arial,sans-serif;font-size:13px;">{rows}</div>"""
    m.get_root().html.add_child(folium.Element(html))

def save_map(m, filename):
    config.OUTPUT_MAPS.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_MAPS / filename
    m.save(str(out))
    print(f"  Saved: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer builders
# ═══════════════════════════════════════════════════════════════════════════════

def dot_layer(df, name, color, show=True):
    group = folium.FeatureGroup(name=name, show=show)
    for _, row in df.iterrows():
        popup = (
            f"<b>{row['date'].strftime('%b %d, %Y')}</b><br>"
            f"{row.get('address', '')}<br>"
            f"Killed: {row['killed']} &nbsp;|&nbsp; Injured: {row['injured']}"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.72,
            popup=folium.Popup(popup, max_width=280),
        ).add_to(group)
    return group


def shaped_layer(df, name, color, shape, show=True):
    """Marker layer using a Unicode glyph (★ fatal, ✕ injury) with white outline."""
    group = folium.FeatureGroup(name=name, show=show)
    for _, row in df.iterrows():
        popup = (
            f"<b>{row['date'].strftime('%b %d, %Y')}</b><br>"
            f"{row.get('address', '')}<br>"
            f"Killed: {row['killed']} &nbsp;|&nbsp; Injured: {row['injured']}"
        )
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.DivIcon(
                html=(
                    f'<span style="font-size:22px;font-weight:bold;color:{color};'
                    f'text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,'
                    f'-1px 1px 0 #fff,1px 1px 0 #fff;">{shape}</span>'
                ),
                icon_size=(22, 22),
                icon_anchor=(11, 12),
            ),
            popup=folium.Popup(popup, max_width=280),
        ).add_to(group)
    return group


def smooth_heat_layer(df, name, show=True):
    if df.empty:
        return folium.FeatureGroup(name=name, show=show)
    data = [[row["lat"], row["lon"], 1.0] for _, row in df.iterrows()]
    return HeatMap(
        data,
        name=name,
        show=show,
        radius=config.HEAT_RADIUS,
        blur=config.HEAT_BLUR,
        min_opacity=config.HEAT_MIN_OPACITY,
    )


def grid_count_layer(df, name, cell_deg=0.002, show=False):
    """
    Square grid choropleth (~200m cells). Hover to see exact count.
    Zoom-stable: colors represent absolute counts, not relative to viewport.
    """
    if df.empty:
        return folium.FeatureGroup(name=name, show=show)

    lats = df["lat"].values
    lons = df["lon"].values

    lat_min = lats.min() - cell_deg
    lat_max = lats.max() + cell_deg
    lon_min = lons.min() - cell_deg
    lon_max = lons.max() + cell_deg

    lat_bins = np.arange(lat_min, lat_max + cell_deg, cell_deg)
    lon_bins = np.arange(lon_min, lon_max + cell_deg, cell_deg)

    lat_idx = np.clip(np.digitize(lats, lat_bins) - 1, 0, len(lat_bins) - 2)
    lon_idx = np.clip(np.digitize(lons, lon_bins) - 1, 0, len(lon_bins) - 2)

    counts = {}
    for li, lo in zip(lat_idx, lon_idx):
        counts[(li, lo)] = counts.get((li, lo), 0) + 1

    if not counts:
        return folium.FeatureGroup(name=name, show=show)

    max_c = max(counts.values())
    colormap = cm.linear.YlOrRd_09.scale(0, max_c)

    features = []
    for (li, lo), count in counts.items():
        lat1, lat2 = lat_bins[li], lat_bins[li + 1]
        lon1, lon2 = lon_bins[lo], lon_bins[lo + 1]
        features.append({
            "type": "Feature",
            "properties": {"count": int(count), "fill": colormap(count)},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[lon1, lat1], [lon2, lat1], [lon2, lat2],
                                  [lon1, lat2], [lon1, lat1]]],
            },
        })

    return folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name=name,
        show=show,
        style_function=lambda f: {
            "fillColor":   f["properties"]["fill"],
            "color":       "none",
            "fillOpacity": 0.72,
            "weight":      0,
        },
        tooltip=folium.GeoJsonTooltip(fields=["count"], aliases=["Incidents:"]),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Map builders
# ═══════════════════════════════════════════════════════════════════════════════

def make_era_dot_map(df, title, filename):
    m = base_map()
    dot_layer(era1(df), ERA_1_LABEL, ERA_1_COLOR, show=True).add_to(m)
    dot_layer(era2(df), ERA_2_LABEL, ERA_2_COLOR, show=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    add_title(m, title)
    add_legend(m, [(ERA_1_LABEL, ERA_1_COLOR), (ERA_2_LABEL, ERA_2_COLOR)])
    save_map(m, filename)


def make_era_heat_map(df, title, filename):
    m = base_map()

    # Smooth heat map — era 1 on by default
    smooth_heat_layer(era1(df), f"Smooth — {ERA_1_LABEL}", show=True).add_to(m)
    smooth_heat_layer(era2(df), f"Smooth — {ERA_2_LABEL}", show=False).add_to(m)
    # Grid count — off by default (toggle in layer control)
    grid_count_layer(era1(df), f"Grid count — {ERA_1_LABEL}", show=False).add_to(m)
    grid_count_layer(era2(df), f"Grid count — {ERA_2_LABEL}", show=False).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    add_title(m, title + "<br><small style='font-weight:normal;'>Toggle Smooth ↔ Grid count in layer control</small>")
    save_map(m, filename)


def make_neighborhood_zoom(df, neighborhoods_gdf, display_name, dataset_name):
    nbd = neighborhoods_gdf[neighborhoods_gdf["NeighborhoodName"] == dataset_name]
    if nbd.empty:
        print(f"  '{dataset_name}' not found — skipping.")
        return

    bounds = nbd.geometry.total_bounds   # [minx, miny, maxx, maxy]
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    # Spatial clip: incidents within the neighborhood (+ 400m buffer for edge cases)
    # Project to UTM Zone 17N (meters) for accurate buffer, then back to WGS84
    nbd_buf = nbd.to_crs("EPSG:32617")
    nbd_buf = nbd_buf.copy()
    nbd_buf["geometry"] = nbd_buf.geometry.buffer(400)
    nbd_buf = nbd_buf.to_crs("EPSG:4326")
    inc_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    )
    clipped = gpd.sjoin(inc_gdf, nbd_buf[["geometry"]], how="inner", predicate="within")
    clipped = clipped.drop(columns=["geometry", "index_right"], errors="ignore")
    local_df = pd.DataFrame(clipped)

    m = folium.Map(location=center, zoom_start=14, tiles="CartoDB positron")

    # Neighborhood boundary overlay
    folium.GeoJson(
        nbd.to_json(),
        name=f"{display_name} boundary",
        style_function=lambda x: {"color": "#333", "weight": 2.5, "fillOpacity": 0.04},
    ).add_to(m)

    # Era dot layers
    dot_layer(era1(local_df), ERA_1_LABEL, ERA_1_COLOR, show=True).add_to(m)
    dot_layer(era2(local_df), ERA_2_LABEL, ERA_2_COLOR, show=True).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    n1, n2 = len(era1(local_df)), len(era2(local_df))
    add_title(m,
        f"<b>{display_name}</b><br>"
        f"<small style='font-weight:normal;'>"
        f"{ERA_1_LABEL}: {n1} &nbsp;|&nbsp; {ERA_2_LABEL}: {n2}</small>"
    )
    add_legend(m, [(ERA_1_LABEL, ERA_1_COLOR), (ERA_2_LABEL, ERA_2_COLOR)])

    safe = display_name.lower().replace(" ", "_").replace("/", "_")
    save_map(m, f"zoom_{safe}.html")


def make_kidz_zones_map(df, kidz_zones):
    # 2024–2025 fatal + injury shootings
    recent = df[df["year"].isin([2024, 2025])]
    significant = recent[(recent["killed"] > 0) | (recent["injured"] > 0)].copy()

    if significant.empty:
        print("  No incidents in 2024–2025 — skipping Kidz Zones map.")
        return

    # Spatial join: which incidents fall inside a Kidz Zone polygon?
    inc_gdf = gpd.GeoDataFrame(
        significant,
        geometry=gpd.points_from_xy(significant["lon"], significant["lat"]),
        crs="EPSG:4326",
    )
    inside = gpd.sjoin(
        inc_gdf,
        kidz_zones[["KZ_Name", "geometry"]],
        how="inner",
        predicate="within",
    )
    inside = inside.drop(columns=["geometry", "index_right"], errors="ignore")

    # Summary table
    if not inside.empty:
        table = (
            pd.DataFrame(inside)
            .groupby("KZ_Name")
            .agg(
                incidents=("date", "count"),
                killed=("killed", "sum"),
                injured=("injured", "sum"),
            )
            .reset_index()
            .sort_values("incidents", ascending=False)
        )
        csv_out = config.OUTPUT_MAPS / "kidz_zones_2024_2025_table.csv"
        table.to_csv(csv_out, index=False)
        print(f"  Saved table: kidz_zones_2024_2025_table.csv")
        print(f"\n  {len(inside)} incidents in {len(table)} Kidz Zones (2024–2025):")
        print(table.to_string(index=False))
    else:
        print("  No incidents fell within Kidz Zone boundaries.")
        table = pd.DataFrame()

    # Build map
    m = base_map()

    # All Kidz Zone boundaries (green)
    folium.GeoJson(
        kidz_zones.to_json(),
        name="Kidz Zones (all 121)",
        show=True,
        style_function=lambda x: {
            "color":       "#2ca25f",
            "weight":      1.5,
            "fillColor":   "#2ca25f",
            "fillOpacity": 0.12,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["KZ_Name"], aliases=["Zone:"]
        ),
    ).add_to(m)

    # Incidents INSIDE zones
    if not inside.empty:
        inside_layer = folium.FeatureGroup(name="Incidents inside Kidz Zones", show=True)
        for _, row in pd.DataFrame(inside).iterrows():
            color = "darkred" if row["killed"] > 0 else "orange"
            popup = (
                f"<b>Zone: {row.get('KZ_Name', '')}</b><br>"
                f"<b>{row['date'].strftime('%b %d, %Y')}</b><br>"
                f"{row.get('address', '')}<br>"
                f"Killed: {row['killed']} &nbsp;|&nbsp; Injured: {row['injured']}"
            )
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=7, color=color, fill=True, fill_opacity=0.88,
                popup=folium.Popup(popup, max_width=300),
            ).add_to(inside_layer)
        inside_layer.add_to(m)

    # All 2024–2025 incidents as context (off by default)
    dot_layer(significant, "All 2024–2025 (context, off by default)", "#888", show=False).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    zone_count = len(table) if not table.empty else 0
    add_title(m,
        f"<b>Orlando Kidz Zones — Shootings 2024–2025</b><br>"
        f"<small style='font-weight:normal;'>"
        f"{len(inside)} incidents across {zone_count} zones</small>"
    )
    add_legend(m, [
        ("Fatal shooting", "darkred"),
        ("Injury shooting", "orange"),
        ("Kidz Zone boundary", "#2ca25f"),
    ])
    save_map(m, "kidz_zones_2024_2025.html")


# ═══════════════════════════════════════════════════════════════════════════════
# Clustered era dot map
# ═══════════════════════════════════════════════════════════════════════════════

def _incident_color(row) -> str:
    if row.get("mass") or row["killed"] >= 4:
        return "darkred"
    if row["killed"] > 0:
        return "red"
    if row["injured"] >= 3:
        return "orange"
    return "cadetblue"


def _cluster_icon_fn(hex_color: str) -> str:
    return f"""
    function(cluster) {{
        var n = cluster.getChildCount();
        var size = n < 10 ? 30 : n < 100 ? 40 : 50;
        return L.divIcon({{
            html: '<div style="background:{hex_color};color:#fff;border-radius:50%;'
                + 'width:' + size + 'px;height:' + size + 'px;'
                + 'display:flex;align-items:center;justify-content:center;'
                + 'font-size:13px;font-weight:bold;'
                + 'border:2px solid rgba(255,255,255,0.6);">' + n + '</div>',
            className: '',
            iconSize: L.point(size, size)
        }});
    }}"""


def make_era_clustered_dot_map(df, title, filename):
    m = base_map()

    e1_df = era1(df)
    e2_df = era2(df)

    for era_df, label, color in [
        (e1_df, ERA_1_LABEL, ERA_1_COLOR),
        (e2_df, ERA_2_LABEL, ERA_2_COLOR),
    ]:
        cluster = MarkerCluster(name=label, icon_create_function=_cluster_icon_fn(color))
        for _, row in era_df.iterrows():
            popup = (
                f"<b>{row['date'].strftime('%b %d, %Y')}</b><br>"
                f"{row.get('address', '')}<br>"
                f"Killed: {row['killed']} &nbsp;|&nbsp; Injured: {row['injured']}"
                + (" &nbsp;<b>⚠ Mass Shooting</b>" if row.get("mass") else "")
            )
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color=_incident_color(row),
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(popup, max_width=280),
            ).add_to(cluster)
        cluster.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    n1 = len(e1_df); k1 = int(e1_df["killed"].sum()); i1 = int(e1_df["injured"].sum())
    n2 = len(e2_df); k2 = int(e2_df["killed"].sum()); i2 = int(e2_df["injured"].sum())
    add_title(m,
        f"<b>{title}</b><br>"
        f"<small style='font-weight:normal;'>"
        f"<span style='color:{ERA_1_COLOR};'>●</span>&nbsp;{ERA_1_LABEL}: "
        f"{n1:,} incidents, {k1:,} killed, {i1:,} injured"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"<span style='color:{ERA_2_COLOR};'>●</span>&nbsp;{ERA_2_LABEL}: "
        f"{n2:,} incidents, {k2:,} killed, {i2:,} injured"
        f"</small>"
    )
    add_legend(m, [
        ("Fatal (no mass)", "red"),
        ("Multiple injured", "orange"),
        ("Injury only", "cadetblue"),
        ("Mass shooting", "darkred"),
    ])
    save_map(m, filename)


# ═══════════════════════════════════════════════════════════════════════════════
# Advancing Peace side-by-side map
# ═══════════════════════════════════════════════════════════════════════════════

def _nbd_style(color, weight=1.5, fill_opacity=0.06):
    return lambda x: {"color": color, "weight": weight,
                       "fillColor": color, "fillOpacity": fill_opacity}


def make_advancing_peace_sidebyside(df, neighborhoods_gdf, kidz_zones_gdf):
    """
    Synced side-by-side dot map: 2018-2022 (left) vs 2023-Present (right).
    Single universal control bar across the top controls both maps.
    Neighborhood search zooms both maps and shows an incident stats table.
    """
    e1 = era1(df)
    e2 = era2(df)

    # ── Pre-compute neighborhood stats for the search table ────────────────────
    inc_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    )
    joined = gpd.sjoin(
        inc_gdf,
        neighborhoods_gdf[["NeighborhoodName", "geometry"]],
        how="left", predicate="within"
    ).drop(columns=["geometry", "index_right"], errors="ignore")

    nbd_stats = {}
    for name, grp in pd.DataFrame(joined).groupby("NeighborhoodName"):
        b = grp[grp["year"].isin(ERA_1_YEARS)]
        a = grp[grp["year"].isin(ERA_2_YEARS)]
        nbd_stats[name] = {
            "total":          len(grp),
            "total_killed":   int(grp["killed"].sum()),
            "total_injured":  int(grp["injured"].sum()),
            "before_count":   len(b),
            "before_killed":  int(b["killed"].sum()),
            "before_injured": int(b["injured"].sum()),
            "after_count":    len(a),
            "after_killed":   int(a["killed"].sum()),
            "after_injured":  int(a["injured"].sum()),
        }

    nbd_bounds = {}
    for _, row in neighborhoods_gdf.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        nbd_bounds[row["NeighborhoodName"]] = [miny, minx, maxy, maxx]

    # ── Pre-compute Kidz Zone stats + bounds ───────────────────────────────────
    kz_dissolved = kidz_zones_gdf.dissolve(by="KZ_Name").reset_index()

    kz_joined = gpd.sjoin(
        inc_gdf,
        kz_dissolved[["KZ_Name", "geometry"]],
        how="left", predicate="within"
    ).drop(columns=["geometry", "index_right"], errors="ignore")

    kz_stats = {}
    for name, grp in pd.DataFrame(kz_joined).groupby("KZ_Name"):
        b = grp[grp["year"].isin(ERA_1_YEARS)]
        a = grp[grp["year"].isin(ERA_2_YEARS)]
        kz_stats[name] = {
            "total":          len(grp),
            "total_killed":   int(grp["killed"].sum()),
            "total_injured":  int(grp["injured"].sum()),
            "before_count":   len(b),
            "before_killed":  int(b["killed"].sum()),
            "before_injured": int(b["injured"].sum()),
            "after_count":    len(a),
            "after_killed":   int(a["killed"].sum()),
            "after_injured":  int(a["injured"].sum()),
        }

    kz_bounds = {}
    for _, row in kz_dissolved.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        kz_bounds[row["KZ_Name"]] = [miny, minx, maxy, maxx]

    # ── Build map ──────────────────────────────────────────────────────────────
    m = DualMap(
        location=[config.ORLANDO_LAT, config.ORLANDO_LON],
        zoom_start=config.DEFAULT_ZOOM,
        layout="horizontal",
        tiles=None,
    )

    for side in (m.m1, m.m2):
        folium.TileLayer("CartoDB positron", name="Base map").add_to(side)

    # Build layers and track JS variable names for each so toggling is reliable.
    def make_layers(side, era_df):
        reg = {}
        for label, sub_df, color in [
            ("All incidents",    era_df,              "#555555"),
            ("Fatal shootings",  homicides(era_df),   "darkred"),
            ("Injury shootings", injury_only(era_df), "orange"),
        ]:
            lyr = dot_layer(sub_df, label, color, show=True)
            lyr.add_to(side)
            reg[label] = lyr.get_name()

        nbd = folium.GeoJson(
            neighborhoods_gdf.to_json(), name="All neighborhoods", show=True,
            style_function=_nbd_style("#6b7280"),
            tooltip=folium.GeoJsonTooltip(fields=["NeighborhoodName"], aliases=["Neighborhood:"]),
        )
        nbd.add_to(side)
        reg["All neighborhoods"] = nbd.get_name()

        kz = folium.GeoJson(
            kidz_zones_gdf.to_json(), name="Kidz Zone neighborhoods", show=True,
            style_function=_nbd_style("#2ca25f", weight=2.5, fill_opacity=0.12),
            tooltip=folium.GeoJsonTooltip(fields=["KZ_Name"], aliases=["Zone:"]),
        )
        kz.add_to(side)
        reg["Kidz Zone neighborhoods"] = kz.get_name()
        return reg

    reg1 = make_layers(m.m1, e1)
    reg2 = make_layers(m.m2, e2)

    # ── Era titles ─────────────────────────────────────────────────────────────
    n1 = len(e1); k1 = int(e1["killed"].sum()); i1 = int(e1["injured"].sum())
    n2 = len(e2); k2 = int(e2["killed"].sum()); i2 = int(e2["injured"].sum())

    def _title(text, left_pct):
        return (f'<div style="position:fixed;top:10px;left:{left_pct}%;transform:translateX(-50%);'
                f'background:white;padding:6px 14px;border-radius:6px;border:1px solid #ccc;'
                f'z-index:900;font-family:Arial,sans-serif;font-size:13px;text-align:center;'
                f'white-space:nowrap;pointer-events:none;">{text}</div>')

    m.m1.get_root().html.add_child(folium.Element(_title(
        f"<b>Before · {ERA_1_LABEL}</b> &nbsp;·&nbsp; "
        f"<small>{n1:,} incidents &nbsp;| {k1:,} killed &nbsp;| {i1:,} injured</small>", 25)))
    m.m2.get_root().html.add_child(folium.Element(_title(
        f"<b>After · {ERA_2_LABEL}</b> &nbsp;·&nbsp; "
        f"<small>{n2:,} incidents &nbsp;| {k2:,} killed &nbsp;| {i2:,} injured</small>", 75)))

    # ── Universal control bar + neighborhood search ────────────────────────────
    map1_id = m.m1.get_name()
    map2_id = m.m2.get_name()
    stats_json    = json.dumps(nbd_stats,  ensure_ascii=False)
    bounds_json   = json.dumps(nbd_bounds, ensure_ascii=False)
    kz_stats_json = json.dumps(kz_stats,   ensure_ascii=False)
    kz_bounds_json= json.dumps(kz_bounds,  ensure_ascii=False)

    # Store string names only — looked up lazily via window[] at call time
    def reg_js(reg):
        pairs = ", ".join(f'"{k}": "{v}"' for k, v in reg.items())
        return "{" + pairs + "}"

    control_html = f"""
<style>
  .leaflet-control-zoom {{ display:none !important; }}
  #ap-control {{
    position:fixed; top:46px; left:50%; transform:translateX(-50%);
    background:white; border:1px solid #bbb; border-radius:8px;
    padding:8px 16px; z-index:1000; font-family:Arial,sans-serif; font-size:13px;
    display:flex; align-items:center; gap:14px;
    box-shadow:0 2px 8px rgba(0,0,0,.14); white-space:nowrap;
  }}
  #ap-control label {{ cursor:pointer; display:flex; align-items:center; gap:4px; }}
  #ap-control .sep {{ color:#ccc; font-size:16px; }}
  #ap-control input[type=search] {{
    padding:4px 8px; border:1px solid #ccc; border-radius:4px; width:190px; font-size:13px;
  }}
  #ap-table {{
    position:fixed; bottom:0; left:50%; transform:translateX(-50%);
    background:white; border:1px solid #bbb; border-radius:8px 8px 0 0;
    padding:12px 20px 14px; z-index:1000; font-family:Arial,sans-serif; font-size:13px;
    box-shadow:0 -2px 10px rgba(0,0,0,.12); min-width:440px; display:none;
  }}
  #ap-table table {{ border-collapse:collapse; width:100%; margin-top:8px; }}
  #ap-table th,#ap-table td {{ border:1px solid #ddd; padding:5px 12px; }}
  #ap-table th {{ background:#f5f5f5; text-align:left; }}
  #ap-table td:not(:first-child) {{ text-align:right; }}
  #ap-table .total-row {{ font-weight:bold; background:#f5f5f5; }}
  #ap-close {{ float:right; background:none; border:none; font-size:18px; cursor:pointer; line-height:1; }}
</style>

<div id="ap-control">
  <span><b>Incidents:</b></span>
  <label><input type="checkbox" id="cb-all" checked> All incidents</label>
  <label><input type="checkbox" id="cb-fatal"> Fatal shootings</label>
  <label><input type="checkbox" id="cb-injury"> Injury shootings</label>
  <span class="sep">|</span>
  <span><b>Boundaries:</b></span>
  <label><input type="checkbox" id="cb-nbds"> All neighborhoods</label>
  <label><input type="checkbox" id="cb-kz"> Kidz Zone neighborhoods</label>
  <span class="sep">|</span>
  <input type="search" id="nbd-search" list="nbd-list" placeholder="Search neighborhood...">
  <datalist id="nbd-list"></datalist>
</div>

<div id="ap-table">
  <button id="ap-close">&#x00D7;</button>
  <span id="ap-table-title" style="font-size:14px;font-weight:bold;"></span>
  <table>
    <thead><tr><th>Period</th><th>Incidents</th><th>Killed</th><th>Injured</th></tr></thead>
    <tbody id="ap-table-body"></tbody>
  </table>
</div>

<script>
(function() {{
  var MAP1_ID = '{map1_id}';
  var MAP2_ID = '{map2_id}';
  var LAYERS1    = {reg_js(reg1)};
  var LAYERS2    = {reg_js(reg2)};
  var NBD_STATS  = {stats_json};
  var NBD_BOUNDS = {bounds_json};
  var KZ_STATS   = {kz_stats_json};
  var KZ_BOUNDS  = {kz_bounds_json};

  var DEFAULT_OFF = ['Fatal shootings','Injury shootings','All neighborhoods','Kidz Zone neighborhoods'];
  var highlightLayers = [null, null];

  function setLayer(name, show) {{
    [[MAP1_ID, LAYERS1],[MAP2_ID, LAYERS2]].forEach(function(pair) {{
      var map = window[pair[0]], layer = window[pair[1][name]];
      if (!map || !layer) return;
      if (show && !map.hasLayer(layer)) map.addLayer(layer);
      else if (!show && map.hasLayer(layer)) map.removeLayer(layer);
    }});
  }}

  // ── Highlight ────────────────────────────────────────────────────────────────
  function getFeatureGeoJSON(layerVarName, propField, propValue) {{
    var lyr = window[layerVarName];
    if (!lyr) return null;
    var found = null;
    lyr.eachLayer(function(sub) {{
      var p = sub.feature && sub.feature.properties;
      if (p && p[propField] === propValue) found = sub.toGeoJSON();
    }});
    return found;
  }}

  function clearHighlight() {{
    highlightLayers.forEach(function(lyr, i) {{
      var map = window[[MAP1_ID, MAP2_ID][i]];
      if (lyr && map) map.removeLayer(lyr);
    }});
    highlightLayers = [null, null];
  }}

  function highlightArea(layerVarPair, propField, propValue) {{
    clearHighlight();
    layerVarPair.forEach(function(varName, i) {{
      var map = window[[MAP1_ID, MAP2_ID][i]];
      if (!map) return;
      var feat = getFeatureGeoJSON(varName, propField, propValue);
      if (!feat) return;
      var hl = L.geoJSON(feat, {{
        style: {{ color:'#e63946', weight:3, fillColor:'#e63946', fillOpacity:0.22 }}
      }}).addTo(map);
      highlightLayers[i] = hl;
    }});
  }}

  // ── Datalist: neighborhoods + Kidz Zones ─────────────────────────────────────
  var dl = document.getElementById('nbd-list');
  var allNames = Object.keys(NBD_BOUNDS).sort().concat(Object.keys(KZ_BOUNDS).sort());
  allNames.forEach(function(name) {{
    var opt = document.createElement('option');
    opt.value = name;
    dl.appendChild(opt);
  }});

  // ── Checkboxes ───────────────────────────────────────────────────────────────
  var cbMap = {{
    'cb-all':   'All incidents',
    'cb-fatal': 'Fatal shootings',
    'cb-injury':'Injury shootings',
    'cb-nbds':  'All neighborhoods',
    'cb-kz':    'Kidz Zone neighborhoods'
  }};
  Object.keys(cbMap).forEach(function(id) {{
    document.getElementById(id).addEventListener('change', function() {{
      setLayer(cbMap[id], this.checked);
    }});
  }});

  // ── Search ───────────────────────────────────────────────────────────────────
  document.getElementById('nbd-search').addEventListener('change', function() {{
    var name = this.value.trim();
    var isKZ  = KZ_BOUNDS[name] !== undefined;
    var isNbd = NBD_BOUNDS[name] !== undefined;
    if (!isKZ && !isNbd) return;

    var b = isKZ ? KZ_BOUNDS[name] : NBD_BOUNDS[name];
    [MAP1_ID, MAP2_ID].forEach(function(id) {{
      var map = window[id]; if (map) map.fitBounds([[b[0],b[1]],[b[2],b[3]]]);
    }});

    if (isKZ) {{
      highlightArea([LAYERS1['Kidz Zone neighborhoods'], LAYERS2['Kidz Zone neighborhoods']], 'KZ_Name', name);
      showTable(name, KZ_STATS[name], 'Kidz Zone');
    }} else {{
      highlightArea([LAYERS1['All neighborhoods'], LAYERS2['All neighborhoods']], 'NeighborhoodName', name);
      showTable(name, NBD_STATS[name], 'Neighborhood');
    }}
  }});

  // ── Stats table ──────────────────────────────────────────────────────────────
  function showTable(name, s, type) {{
    if (!s) return;
    document.getElementById('ap-table-title').textContent = name + ' (' + type + ')';
    document.getElementById('ap-table-body').innerHTML = [
      ['Before (2018–2022)',      s.before_count, s.before_killed, s.before_injured],
      ['After (2023–Present)',    s.after_count,  s.after_killed,  s.after_injured],
      ['All years (2014–present)',s.total,         s.total_killed,  s.total_injured]
    ].map(function(r, i) {{
      var cls = i===2 ? ' class="total-row"' : '';
      return '<tr'+cls+'><td>'+r[0]+'</td><td>'+r[1]+'</td><td>'+r[2]+'</td><td>'+r[3]+'</td></tr>';
    }}).join('');
    document.getElementById('ap-table').style.display = 'block';
  }}

  document.getElementById('ap-close').addEventListener('click', function() {{
    document.getElementById('ap-table').style.display = 'none';
    document.getElementById('nbd-search').value = '';
    clearHighlight();
  }});

  window.addEventListener('load', function() {{
    DEFAULT_OFF.forEach(function(name) {{ setLayer(name, false); }});
  }});
}})();
</script>
"""
    m.get_root().html.add_child(folium.Element(control_html))
    save_map(m, "sidebyside_advanced_peace.html")


# ═══════════════════════════════════════════════════════════════════════════════
# Unified single map — both eras, yellow vs blue (colorblind-safe)
# ═══════════════════════════════════════════════════════════════════════════════

COLOR_BEFORE = "#E69F00"   # warm yellow  — Wong colorblind-safe palette
COLOR_AFTER  = "#0072B2"   # deep blue    — Wong colorblind-safe palette


def make_unified_map(df, neighborhoods_gdf, kidz_zones_gdf):
    """
    Single-map view of both eras.
    Yellow = 2018-2022 (Before Advancing Peace)
    Blue   = 2023-Present (After Advancing Peace)
    Toggle era (checkboxes) and incident type (radio) independently.
    Neighborhood/Kidz Zone search with highlight and stats table.
    """
    e1 = era1(df)
    e2 = era2(df)

    # ── Pre-compute stats (shared with sidebyside logic) ───────────────────────
    inc_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    )
    joined = gpd.sjoin(
        inc_gdf, neighborhoods_gdf[["NeighborhoodName", "geometry"]],
        how="left", predicate="within"
    ).drop(columns=["geometry", "index_right"], errors="ignore")

    nbd_stats = {}
    for name, grp in pd.DataFrame(joined).groupby("NeighborhoodName"):
        b = grp[grp["year"].isin(ERA_1_YEARS)]
        a = grp[grp["year"].isin(ERA_2_YEARS)]
        nbd_stats[name] = {
            "total": len(grp), "total_killed": int(grp["killed"].sum()),
            "total_injured": int(grp["injured"].sum()),
            "before_count": len(b), "before_killed": int(b["killed"].sum()),
            "before_injured": int(b["injured"].sum()),
            "after_count": len(a), "after_killed": int(a["killed"].sum()),
            "after_injured": int(a["injured"].sum()),
        }

    nbd_bounds = {}
    for _, row in neighborhoods_gdf.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        nbd_bounds[row["NeighborhoodName"]] = [miny, minx, maxy, maxx]

    kz_dissolved = kidz_zones_gdf.dissolve(by="KZ_Name").reset_index()
    kz_joined = gpd.sjoin(
        inc_gdf, kz_dissolved[["KZ_Name", "geometry"]],
        how="left", predicate="within"
    ).drop(columns=["geometry", "index_right"], errors="ignore")

    kz_stats = {}
    for name, grp in pd.DataFrame(kz_joined).groupby("KZ_Name"):
        b = grp[grp["year"].isin(ERA_1_YEARS)]
        a = grp[grp["year"].isin(ERA_2_YEARS)]
        kz_stats[name] = {
            "total": len(grp), "total_killed": int(grp["killed"].sum()),
            "total_injured": int(grp["injured"].sum()),
            "before_count": len(b), "before_killed": int(b["killed"].sum()),
            "before_injured": int(b["injured"].sum()),
            "after_count": len(a), "after_killed": int(a["killed"].sum()),
            "after_injured": int(a["injured"].sum()),
        }

    kz_bounds = {}
    for _, row in kz_dissolved.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        kz_bounds[row["KZ_Name"]] = [miny, minx, maxy, maxx]

    # ── Map ────────────────────────────────────────────────────────────────────
    m = base_map()

    reg = {}
    for label, sub_df, color, shape in [
        ("Before — Fatal shootings",  homicides(e1),   COLOR_BEFORE, "★"),
        ("Before — Injury shootings", injury_only(e1), COLOR_BEFORE, "✕"),
        ("After — Fatal shootings",   homicides(e2),   COLOR_AFTER,  "★"),
        ("After — Injury shootings",  injury_only(e2), COLOR_AFTER,  "✕"),
    ]:
        lyr = shaped_layer(sub_df, label, color, shape, show=True)
        lyr.add_to(m)
        reg[label] = lyr.get_name()

    nbd_lyr = folium.GeoJson(
        neighborhoods_gdf.to_json(), name="All neighborhoods", show=True,
        style_function=_nbd_style("#6b7280"),
        tooltip=folium.GeoJsonTooltip(fields=["NeighborhoodName"], aliases=["Neighborhood:"]),
    )
    nbd_lyr.add_to(m)
    reg["All neighborhoods"] = nbd_lyr.get_name()

    kz_lyr = folium.GeoJson(
        kidz_zones_gdf.to_json(), name="Kidz Zone neighborhoods", show=True,
        style_function=_nbd_style("#2ca25f", weight=2.5, fill_opacity=0.12),
        tooltip=folium.GeoJsonTooltip(fields=["KZ_Name"], aliases=["Zone:"]),
    )
    kz_lyr.add_to(m)
    reg["Kidz Zone neighborhoods"] = kz_lyr.get_name()

    # ── Stats in title ─────────────────────────────────────────────────────────
    k1 = int(e1["killed"].sum()); i1 = int(e1["injured"].sum())
    k2 = int(e2["killed"].sum()); i2 = int(e2["injured"].sum())
    title_html = (
        f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
        f'background:white;padding:6px 18px;border-radius:6px;border:1px solid #aaa;'
        f'z-index:1000;font-family:Arial,sans-serif;font-size:13px;'
        f'text-align:center;white-space:nowrap;">'
        f'<span style="color:{COLOR_BEFORE};font-weight:bold;">★✕</span> '
        f'Before {ERA_1_LABEL}: {k1:,} killed · {i1:,} injured'
        f'&nbsp;&nbsp;&nbsp;'
        f'<span style="color:{COLOR_AFTER};font-weight:bold;">★✕</span> '
        f'After {ERA_2_LABEL}: {k2:,} killed · {i2:,} injured'
        f'&nbsp;&nbsp; <small style="color:#666;">★ fatal &nbsp; ✕ injury</small>'
        f'</div>'
    )
    m.get_root().html.add_child(folium.Element(title_html))

    # ── Control bar + search ───────────────────────────────────────────────────
    map_id      = m.get_name()
    reg_js_str  = "{" + ", ".join(f'"{k}": "{v}"' for k, v in reg.items()) + "}"
    stats_json  = json.dumps(nbd_stats, ensure_ascii=False)
    bounds_json = json.dumps(nbd_bounds, ensure_ascii=False)
    kz_stats_j  = json.dumps(kz_stats,  ensure_ascii=False)
    kz_bounds_j = json.dumps(kz_bounds, ensure_ascii=False)

    html = f"""
<style>
  .leaflet-control-zoom {{ display:none !important; }}
  #um-control {{
    position:fixed; top:46px; left:50%; transform:translateX(-50%);
    background:white; border:1px solid #bbb; border-radius:8px;
    padding:8px 16px; z-index:1000; font-family:Arial,sans-serif; font-size:13px;
    display:flex; align-items:center; gap:14px;
    box-shadow:0 2px 8px rgba(0,0,0,.14); white-space:nowrap;
  }}
  #um-control label {{ cursor:pointer; display:flex; align-items:center; gap:4px; }}
  #um-control .sep {{ color:#ccc; font-size:16px; margin:0 2px; }}
  #um-control input[type=search] {{
    padding:4px 8px; border:1px solid #ccc; border-radius:4px; width:190px; font-size:13px;
  }}
  .sym-before {{ color:{COLOR_BEFORE}; font-weight:bold; }}
  .sym-after  {{ color:{COLOR_AFTER};  font-weight:bold; }}
  #um-table {{
    position:fixed; bottom:0; left:50%; transform:translateX(-50%);
    background:white; border:1px solid #bbb; border-radius:8px 8px 0 0;
    padding:12px 20px 14px; z-index:1000; font-family:Arial,sans-serif; font-size:13px;
    box-shadow:0 -2px 10px rgba(0,0,0,.12); min-width:440px; display:none;
  }}
  #um-table table {{ border-collapse:collapse; width:100%; margin-top:8px; }}
  #um-table th,#um-table td {{ border:1px solid #ddd; padding:5px 12px; }}
  #um-table th {{ background:#f5f5f5; text-align:left; }}
  #um-table td:not(:first-child) {{ text-align:right; }}
  #um-table .total-row {{ font-weight:bold; background:#f5f5f5; }}
  #um-close {{ float:right; background:none; border:none; font-size:18px; cursor:pointer; line-height:1; }}
</style>

<div id="um-control">
  <span><b>Era:</b></span>
  <label><input type="checkbox" id="cb-before" checked>
    <span class="sym-before">&#9733;&#10005;</span> Before 2018–2022</label>
  <label><input type="checkbox" id="cb-after" checked>
    <span class="sym-after">&#9733;&#10005;</span> After 2023–Present</label>
  <span class="sep">|</span>
  <span><b>Type:</b></span>
  <label><input type="checkbox" id="cb-fatal" checked> Fatal &#9733;</label>
  <label><input type="checkbox" id="cb-injury" checked> Injury &#10005;</label>
  <span class="sep">|</span>
  <span><b>Boundaries:</b></span>
  <label><input type="checkbox" id="cb-nbds"> All neighborhoods</label>
  <label><input type="checkbox" id="cb-kz"> Kidz Zone neighborhoods</label>
  <span class="sep">|</span>
  <input type="search" id="um-search" list="um-nbd-list" placeholder="Search neighborhood or Kidz Zone...">
  <datalist id="um-nbd-list"></datalist>
</div>

<div id="um-table">
  <button id="um-close">&#x00D7;</button>
  <span id="um-table-title" style="font-size:14px;font-weight:bold;"></span>
  <table>
    <thead><tr><th>Period</th><th>Incidents</th><th>Killed</th><th>Injured</th></tr></thead>
    <tbody id="um-table-body"></tbody>
  </table>
</div>

<script>
(function() {{
  var MAP_ID  = '{map_id}';
  var LAYERS  = {reg_js_str};
  var NBD_STATS  = {stats_json};
  var NBD_BOUNDS = {bounds_json};
  var KZ_STATS   = {kz_stats_j};
  var KZ_BOUNDS  = {kz_bounds_j};

  var DEFAULT_OFF = ['All neighborhoods', 'Kidz Zone neighborhoods'];

  var highlightLayer = null;

  // ── Layer control ─────────────────────────────────────────────────────────
  function setLayer(name, show) {{
    var map   = window[MAP_ID];
    var layer = window[LAYERS[name]];
    if (!map || !layer) return;
    if (show && !map.hasLayer(layer)) map.addLayer(layer);
    else if (!show && map.hasLayer(layer)) map.removeLayer(layer);
  }}

  var STATE = {{ before: true, after: true, fatal: true, injury: true }};

  function syncLayers() {{
    ['Before', 'After'].forEach(function(era) {{
      var eraOn = era === 'Before' ? STATE.before : STATE.after;
      setLayer(era + ' — Fatal shootings',  eraOn && STATE.fatal);
      setLayer(era + ' — Injury shootings', eraOn && STATE.injury);
    }});
  }}

  // ── Era checkboxes ────────────────────────────────────────────────────────
  document.getElementById('cb-before').addEventListener('change', function() {{
    STATE.before = this.checked; syncLayers();
  }});
  document.getElementById('cb-after').addEventListener('change', function() {{
    STATE.after = this.checked; syncLayers();
  }});

  // ── Type checkboxes ───────────────────────────────────────────────────────
  document.getElementById('cb-fatal').addEventListener('change', function() {{
    STATE.fatal = this.checked; syncLayers();
  }});
  document.getElementById('cb-injury').addEventListener('change', function() {{
    STATE.injury = this.checked; syncLayers();
  }});

  // ── Boundary checkboxes ───────────────────────────────────────────────────
  document.getElementById('cb-nbds').addEventListener('change', function() {{
    setLayer('All neighborhoods', this.checked);
  }});
  document.getElementById('cb-kz').addEventListener('change', function() {{
    setLayer('Kidz Zone neighborhoods', this.checked);
  }});

  // ── Datalist ──────────────────────────────────────────────────────────────
  var dl = document.getElementById('um-nbd-list');
  Object.keys(NBD_BOUNDS).sort().concat(Object.keys(KZ_BOUNDS).sort()).forEach(function(n) {{
    var o = document.createElement('option'); o.value = n; dl.appendChild(o);
  }});

  // ── Highlight ─────────────────────────────────────────────────────────────
  function getFeatureGeoJSON(layerVarName, propField, propValue) {{
    var lyr = window[layerVarName]; if (!lyr) return null;
    var found = null;
    lyr.eachLayer(function(sub) {{
      var p = sub.feature && sub.feature.properties;
      if (p && p[propField] === propValue) found = sub.toGeoJSON();
    }});
    return found;
  }}

  function clearHighlight() {{
    var map = window[MAP_ID];
    if (highlightLayer && map) {{ map.removeLayer(highlightLayer); highlightLayer = null; }}
  }}

  function highlightArea(layerVarName, propField, propValue) {{
    clearHighlight();
    var map = window[MAP_ID]; if (!map) return;
    var feat = getFeatureGeoJSON(layerVarName, propField, propValue);
    if (!feat) return;
    highlightLayer = L.geoJSON(feat, {{
      style: {{ color:'#e63946', weight:3, fillColor:'#e63946', fillOpacity:0.2 }}
    }}).addTo(map);
  }}

  // ── Search ────────────────────────────────────────────────────────────────
  document.getElementById('um-search').addEventListener('change', function() {{
    var name = this.value.trim();
    var isKZ  = KZ_BOUNDS[name] !== undefined;
    var isNbd = NBD_BOUNDS[name] !== undefined;
    if (!isKZ && !isNbd) return;
    var b = isKZ ? KZ_BOUNDS[name] : NBD_BOUNDS[name];
    var map = window[MAP_ID]; if (map) map.fitBounds([[b[0],b[1]],[b[2],b[3]]]);
    if (isKZ) {{
      highlightArea(LAYERS['Kidz Zone neighborhoods'], 'KZ_Name', name);
      showTable(name, KZ_STATS[name], 'Kidz Zone');
    }} else {{
      highlightArea(LAYERS['All neighborhoods'], 'NeighborhoodName', name);
      showTable(name, NBD_STATS[name], 'Neighborhood');
    }}
  }});

  // ── Stats table ───────────────────────────────────────────────────────────
  function showTable(name, s, type) {{
    if (!s) return;
    document.getElementById('um-table-title').textContent = name + ' (' + type + ')';
    document.getElementById('um-table-body').innerHTML = [
      ['Before (2018–2022)',      s.before_count, s.before_killed, s.before_injured],
      ['After (2023–Present)',    s.after_count,  s.after_killed,  s.after_injured],
      ['All years (2014–present)',s.total,         s.total_killed,  s.total_injured]
    ].map(function(r, i) {{
      var cls = i===2 ? ' class="total-row"' : '';
      return '<tr'+cls+'><td>'+r[0]+'</td><td>'+r[1]+'</td><td>'+r[2]+'</td><td>'+r[3]+'</td></tr>';
    }}).join('');
    document.getElementById('um-table').style.display = 'block';
  }}

  document.getElementById('um-close').addEventListener('click', function() {{
    document.getElementById('um-table').style.display = 'none';
    document.getElementById('um-search').value = '';
    clearHighlight();
  }});

  // ── Init: hide default-off layers after maps load ─────────────────────────
  window.addEventListener('load', function() {{
    DEFAULT_OFF.forEach(function(n) {{ setLayer(n, false); }});
  }});
}})();
</script>
"""
    m.get_root().html.add_child(folium.Element(html))
    save_map(m, "unified_advanced_peace.html")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    df            = load_incidents()
    kidz_zones    = load_kidz_zones()
    neighborhoods = load_neighborhoods()
    print(f"  {len(df):,} mappable incidents | years: {sorted(df['year'].unique())}")
    print(f"  {len(kidz_zones)} Kidz Zones | {len(neighborhoods)} neighborhoods")

    # ── Dot density maps by era ───────────────────────────────────────────────
    print("\n── Dot density maps ──────────────────────────────────────────────")
    make_era_dot_map(df,
        "Orlando Gun Violence — All Incidents",
        "dot_all_by_era.html")
    make_era_dot_map(homicides(df),
        "Orlando Gun Violence — Fatal Shootings",
        "dot_homicides_by_era.html")
    make_era_dot_map(injury_only(df),
        "Orlando Gun Violence — Injury Shootings (Non-Fatal)",
        "dot_injury_by_era.html")

    # ── Clustered era maps (era-colored bubbles, incident-type dot colors) ────
    print("\n── Clustered era maps ────────────────────────────────────────────")
    make_era_clustered_dot_map(df,
        "Orlando Gun Violence — All Incidents",
        "dot_all_by_era_clustered.html")
    make_era_clustered_dot_map(homicides(df),
        "Orlando Gun Violence — Fatal Shootings",
        "dot_homicides_by_era_clustered.html")
    make_era_clustered_dot_map(injury_only(df),
        "Orlando Gun Violence — Injury Shootings (Non-Fatal)",
        "dot_injury_by_era_clustered.html")

    # ── Heat maps by era ──────────────────────────────────────────────────────
    print("\n── Heat maps ────────────────────────────────────────────────────")
    make_era_heat_map(df,
        "Orlando Gun Violence — Heat Map, All Incidents",
        "heat_all_by_era.html")
    make_era_heat_map(homicides(df),
        "Orlando Gun Violence — Heat Map, Fatal Shootings",
        "heat_homicides_by_era.html")
    make_era_heat_map(injury_only(df),
        "Orlando Gun Violence — Heat Map, Injury Shootings",
        "heat_injury_by_era.html")

    # ── Neighborhood zooms ────────────────────────────────────────────────────
    print("\n── Neighborhood zoom maps ───────────────────────────────────────")
    for display_name, dataset_name in NEIGHBORHOODS.items():
        make_neighborhood_zoom(df, neighborhoods, display_name, dataset_name)

    # ── Kidz Zones analysis ───────────────────────────────────────────────────
    print("\n── Kidz Zones 2024–2025 ─────────────────────────────────────────")
    make_kidz_zones_map(df, kidz_zones)

    # ── Advancing Peace side-by-side ─────────────────────────────────────────
    print("\n── Advancing Peace side-by-side ─────────────────────────────────")
    make_advancing_peace_sidebyside(df, neighborhoods, kidz_zones)

    # ── Advancing Peace unified single map ────────────────────────────────────
    print("\n── Advancing Peace unified map ──────────────────────────────────")
    make_unified_map(df, neighborhoods, kidz_zones)

    print(f"\nAll done. Files in: {config.OUTPUT_MAPS}")


if __name__ == "__main__":
    main()
