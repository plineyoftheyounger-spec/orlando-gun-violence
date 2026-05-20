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
from folium.plugins import HeatMap, MarkerCluster
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

    print(f"\nAll done. Files in: {config.OUTPUT_MAPS}")


if __name__ == "__main__":
    main()
