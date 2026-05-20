"""
02_create_maps.py
──────────────────
Generates interactive HTML maps from the processed incident data.
Open any output file in a browser — no server required.

MAPS PRODUCED:
  Per-year incident maps  →  output/maps/orlando_YYYY_incidents.html
  Per-year heat maps      →  output/maps/orlando_YYYY_heatmap.html
  All-years heat map      →  output/maps/orlando_all_years_heatmap.html
  All-years combined dot  →  output/maps/orlando_all_years_combined.html

TWEAK IDEAS:
  - CLUSTER_MARKERS: False gives raw dots, True groups them at low zoom
  - WEIGHT_BY_SEVERITY: counts fatality incidents more heavily on heat maps
  - Change HEAT_RADIUS / HEAT_BLUR in config.py to adjust heat map spread
"""

import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import branca.colormap as cm

import config

# ── Settings ───────────────────────────────────────────────────────────────────
CLUSTER_MARKERS    = True    # Cluster dots at low zoom (cleaner for dense areas)
WEIGHT_BY_SEVERITY = True    # Fatality incidents weighted heavier on heat maps
SHOW_MASS_SHOOTINGS = True   # Add a separate layer highlighting mass shootings


# ── Data ───────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = config.DATA_PROCESSED / "orlando_incidents.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}\n"
            "Run 01_prepare_data.py first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    # Keep only rows with valid coordinates
    df = df.dropna(subset=["lat", "lon"])
    print(f"Loaded {len(df):,} mappable incidents.")
    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

def _marker_color(row) -> str:
    if row["killed"] >= 4 or row["total"] >= 4:
        return "darkred"    # Mass shooting
    if row["killed"] > 0:
        return "red"        # Fatal
    if row["injured"] >= 3:
        return "orange"     # Multiple injured
    return "cadetblue"      # Injury only


def _popup(row) -> str:
    date_str = row["date"].strftime("%B %d, %Y") if pd.notna(row["date"]) else "Unknown"
    return (
        f"<b>{date_str}</b><br>"
        f"{row.get('address', '')}<br>"
        f"<b>Killed:</b> {row['killed']} &nbsp; <b>Injured:</b> {row['injured']}"
        + (" &nbsp; <b>⚠ Mass Shooting</b>" if row.get("mass") else "")
    )


def _heat_weight(row) -> float:
    if not WEIGHT_BY_SEVERITY:
        return 1.0
    return float(1 + row["killed"] * (config.FATALITY_WEIGHT - 1))


def _legend(colors: dict) -> folium.Element:
    items = "".join(
        f'<div><span style="color:{c};font-size:16px;">●</span> {label}</div>'
        for label, c in colors.items()
    )
    html = f"""
    <div style="position:fixed;bottom:30px;left:30px;background:white;
         padding:10px 14px;border-radius:6px;border:1px solid #aaa;
         z-index:1000;font-family:Arial,sans-serif;font-size:13px;">
        <b>Incident Type</b><br>{items}
    </div>"""
    return folium.Element(html)


def _base_map() -> folium.Map:
    return folium.Map(
        location=[config.ORLANDO_LAT, config.ORLANDO_LON],
        zoom_start=config.DEFAULT_ZOOM,
        tiles="CartoDB positron",
    )


# ── Per-year incident dot map ──────────────────────────────────────────────────

def map_by_year(df: pd.DataFrame, year: int) -> None:
    year_df = df[df["year"] == year]
    if year_df.empty:
        print(f"  {year}: no data — skipping.")
        return

    m = _base_map()
    container = MarkerCluster(name="Incidents") if CLUSTER_MARKERS else folium.FeatureGroup(name="Incidents")

    for _, row in year_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color=_marker_color(row),
            fill=True,
            fill_opacity=0.75,
            popup=folium.Popup(_popup(row), max_width=300),
        ).add_to(container)

    container.add_to(m)

    if SHOW_MASS_SHOOTINGS:
        mass_df = year_df[year_df["mass"] == True]
        if not mass_df.empty:
            mass_layer = folium.FeatureGroup(name="Mass Shootings (4+ victims)")
            for _, row in mass_df.iterrows():
                folium.Marker(
                    location=[row["lat"], row["lon"]],
                    icon=folium.Icon(color="black", icon="warning-sign", prefix="glyphicon"),
                    popup=folium.Popup(_popup(row), max_width=300),
                ).add_to(mass_layer)
            mass_layer.add_to(m)

    folium.LayerControl().add_to(m)
    m.get_root().html.add_child(_legend({
        "Fatal (no mass)": "red",
        "Multiple injured": "orange",
        "Injury only": "cadetblue",
        "Mass shooting": "darkred",
    }))

    count = len(year_df)
    killed = year_df["killed"].sum()
    injured = year_df["injured"].sum()
    title_html = f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
         background:white;padding:8px 16px;border-radius:6px;border:1px solid #aaa;
         z-index:1000;font-family:Arial,sans-serif;font-size:14px;text-align:center;">
        <b>Orlando Gun Violence {year}</b><br>
        {count:,} incidents &nbsp;|&nbsp; {killed:,} killed &nbsp;|&nbsp; {injured:,} injured
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    out = config.OUTPUT_MAPS / f"orlando_{year}_incidents.html"
    m.save(str(out))
    print(f"  Saved: {out.name}")


# ── Per-year heat map ──────────────────────────────────────────────────────────

def heatmap_by_year(df: pd.DataFrame, year: int) -> None:
    year_df = df[df["year"] == year]
    if year_df.empty:
        return

    m = _base_map()
    heat_data = [[row["lat"], row["lon"], _heat_weight(row)] for _, row in year_df.iterrows()]

    HeatMap(
        heat_data,
        name="Heat Map",
        radius=config.HEAT_RADIUS,
        blur=config.HEAT_BLUR,
        min_opacity=config.HEAT_MIN_OPACITY,
    ).add_to(m)

    title_html = f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
         background:white;padding:8px 16px;border-radius:6px;border:1px solid #aaa;
         z-index:1000;font-family:Arial,sans-serif;font-size:14px;text-align:center;">
        <b>Orlando Gun Violence Heat Map — {year}</b>
        {"<br><small>Weighted by fatalities</small>" if WEIGHT_BY_SEVERITY else ""}
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    out = config.OUTPUT_MAPS / f"orlando_{year}_heatmap.html"
    m.save(str(out))
    print(f"  Saved: {out.name}")


# ── All-years heat map (layer per year, toggleable) ────────────────────────────

def heatmap_all_years(df: pd.DataFrame) -> None:
    m = _base_map()
    years = sorted(df["year"].unique())

    for i, year in enumerate(years):
        year_df = df[df["year"] == year]
        heat_data = [[row["lat"], row["lon"], _heat_weight(row)] for _, row in year_df.iterrows()]
        # Show only the most recent year by default
        HeatMap(
            heat_data,
            name=str(year),
            show=(i == len(years) - 1),
            radius=config.HEAT_RADIUS,
            blur=config.HEAT_BLUR,
            min_opacity=config.HEAT_MIN_OPACITY,
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    title_html = """
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
         background:white;padding:8px 16px;border-radius:6px;border:1px solid #aaa;
         z-index:1000;font-family:Arial,sans-serif;font-size:14px;text-align:center;">
        <b>Orlando Gun Violence — Heat Maps by Year</b><br>
        <small>Use layer control (top right) to switch years</small>
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    out = config.OUTPUT_MAPS / "orlando_all_years_heatmap.html"
    m.save(str(out))
    print(f"  Saved: {out.name}")


# ── All-years combined dot map (layer per year) ────────────────────────────────

def map_all_years_combined(df: pd.DataFrame) -> None:
    m = _base_map()
    years = sorted(df["year"].unique())

    year_colormap = cm.linear.YlOrRd_09.scale(min(years), max(years))
    year_colormap.caption = "Year"

    for i, year in enumerate(years):
        year_df = df[df["year"] == year]
        group = folium.FeatureGroup(name=str(year), show=(i == len(years) - 1))
        color = year_colormap(year)

        for _, row in year_df.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.65,
                popup=folium.Popup(_popup(row), max_width=280),
            ).add_to(group)

        group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    year_colormap.add_to(m)

    title_html = """
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
         background:white;padding:8px 16px;border-radius:6px;border:1px solid #aaa;
         z-index:1000;font-family:Arial,sans-serif;font-size:14px;text-align:center;">
        <b>Orlando Gun Violence — All Years</b><br>
        <small>Toggle years using the layer control (top right)</small>
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    out = config.OUTPUT_MAPS / "orlando_all_years_combined.html"
    m.save(str(out))
    print(f"  Saved: {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    config.OUTPUT_MAPS.mkdir(parents=True, exist_ok=True)
    df = load_data()
    years = sorted(df["year"].unique())

    print(f"\nGenerating per-year maps for: {years}")
    for year in years:
        map_by_year(df, year)
        heatmap_by_year(df, year)

    print("\nGenerating multi-year maps...")
    heatmap_all_years(df)
    map_all_years_combined(df)

    print(f"\nDone. Open files in output/maps/ in any browser.")


if __name__ == "__main__":
    main()
