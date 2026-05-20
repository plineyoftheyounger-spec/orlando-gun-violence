"""
03_neighborhood_analysis.py
────────────────────────────
Joins incident data to a neighborhood boundary file and produces:
  - Choropleth maps (incident count / rate per neighborhood, per year)
  - Per-neighborhood incident dot maps
  - Summary CSV by neighborhood

GETTING A NEIGHBORHOOD FILE:
  Orlando official neighborhoods (GeoJSON) can often be found at:
    - Orlando Open Data portal: https://data.orlando.gov
    - Orange County GIS: https://www.orangecountyfl.net/CultureParks/GISProgramServices.aspx
    - US Census TIGER shapefiles (use "Census Tracts" or "ZCTAs" as a proxy)

  Save the file to neighborhoods/ as orlando_neighborhoods.geojson
  (or adjust NEIGHBORHOOD_FILE below).

  The file needs a column with a unique neighborhood name — adjust NBD_NAME_COL
  to match whatever your GeoJSON uses.
"""

import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import branca.colormap as cm

import config

# ── Settings ───────────────────────────────────────────────────────────────────
NEIGHBORHOOD_FILE = config.NEIGHBORHOODS_DIR / "orlando_neighborhoods.geojson"
NBD_NAME_COL      = "NeighborhoodName"  # Column in the GeoJSON with neighborhood name
# If you have population data, set a path here to enable per-capita rates
POPULATION_FILE   = None          # e.g., config.NEIGHBORHOODS_DIR / "population.csv"
POPULATION_COL    = "population"  # Column name in the population CSV


# ── Load data ──────────────────────────────────────────────────────────────────

def load_incidents() -> gpd.GeoDataFrame:
    path = config.DATA_PROCESSED / "orlando_incidents.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["lat", "lon"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    return gdf


def load_neighborhoods() -> gpd.GeoDataFrame:
    if not NEIGHBORHOOD_FILE.exists():
        raise FileNotFoundError(
            f"Neighborhood file not found: {NEIGHBORHOOD_FILE}\n"
            "See the docstring at the top of this script for download instructions."
        )
    nbd = gpd.read_file(NEIGHBORHOOD_FILE).to_crs("EPSG:4326")
    print(f"Loaded {len(nbd)} neighborhoods.")
    print(f"Available columns: {list(nbd.columns)}")
    return nbd


# ── Spatial join ───────────────────────────────────────────────────────────────

def join_to_neighborhoods(incidents: gpd.GeoDataFrame, neighborhoods: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(incidents, neighborhoods[[NBD_NAME_COL, "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={NBD_NAME_COL: "neighborhood"})
    unmatched = joined["neighborhood"].isna().sum()
    if unmatched:
        print(f"  {unmatched:,} incidents fell outside neighborhood boundaries.")
    return joined


# ── Summary stats ──────────────────────────────────────────────────────────────

def summarize_by_neighborhood(joined: gpd.GeoDataFrame, year: int | None = None) -> pd.DataFrame:
    df = joined.copy()
    if year:
        df = df[df["date"].dt.year == year]

    summary = df.groupby("neighborhood").agg(
        incidents=("date", "count"),
        killed=("killed", "sum"),
        injured=("injured", "sum"),
        mass_shootings=("mass", "sum"),
    ).reset_index()

    if POPULATION_FILE:
        pop = pd.read_csv(POPULATION_FILE)
        summary = summary.merge(pop[["neighborhood", POPULATION_COL]], on="neighborhood", how="left")
        summary["incidents_per_1k"] = (summary["incidents"] / summary[POPULATION_COL] * 1000).round(2)
        summary["killed_per_1k"]    = (summary["killed"]    / summary[POPULATION_COL] * 1000).round(2)

    return summary.sort_values("incidents", ascending=False)


# ── Choropleth map ─────────────────────────────────────────────────────────────

def choropleth_map(neighborhoods: gpd.GeoDataFrame, summary: pd.DataFrame,
                   value_col: str = "incidents", year: int | None = None) -> None:
    nbd_with_data = neighborhoods.merge(summary, left_on=NBD_NAME_COL, right_on="neighborhood", how="left")
    nbd_with_data[value_col] = nbd_with_data[value_col].fillna(0)

    m = folium.Map(
        location=[config.ORLANDO_LAT, config.ORLANDO_LON],
        zoom_start=config.DEFAULT_ZOOM,
        tiles="CartoDB positron",
    )

    label = value_col.replace("_", " ").title()
    year_str = str(year) if year else "All Years"

    folium.Choropleth(
        geo_data=nbd_with_data.to_json(),
        data=nbd_with_data,
        columns=[NBD_NAME_COL, value_col],
        key_on=f"feature.properties.{NBD_NAME_COL}",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.4,
        legend_name=f"{label} — {year_str}",
        name="Choropleth",
    ).add_to(m)

    # Tooltip: show neighborhood name + value on hover
    tooltip_cols = [NBD_NAME_COL, value_col]
    folium.GeoJson(
        nbd_with_data.to_json(),
        style_function=lambda x: {"fillOpacity": 0, "color": "transparent"},
        tooltip=folium.GeoJsonTooltip(fields=tooltip_cols, aliases=["Neighborhood", label]),
    ).add_to(m)

    folium.LayerControl().add_to(m)

    suffix = str(year) if year else "all_years"
    out = config.OUTPUT_MAPS / f"orlando_{suffix}_choropleth_{value_col}.html"
    m.save(str(out))
    print(f"  Saved: {out.name}")


# ── Per-neighborhood dot map (zoom in on one neighborhood) ─────────────────────

def map_neighborhood(joined: gpd.GeoDataFrame, neighborhoods: gpd.GeoDataFrame,
                     name: str, year: int | None = None) -> None:
    nbd_row = neighborhoods[neighborhoods[NBD_NAME_COL] == name]
    if nbd_row.empty:
        print(f"  Neighborhood '{name}' not found.")
        return

    bounds = nbd_row.geometry.total_bounds  # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    df = joined[joined["neighborhood"] == name].copy()
    if year:
        df = df[df["date"].dt.year == year]

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

    # Neighborhood boundary
    folium.GeoJson(
        nbd_row.to_json(),
        style_function=lambda x: {"color": "#333", "fillOpacity": 0.05, "weight": 2},
    ).add_to(m)

    # Incidents
    for _, row in df.iterrows():
        color = "red" if row["killed"] > 0 else "cadetblue"
        date_str = row["date"].strftime("%B %d, %Y") if pd.notna(row["date"]) else "Unknown"
        popup = f"<b>{date_str}</b><br>{row.get('address','')}<br>Killed: {row['killed']} | Injured: {row['injured']}"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6, color=color, fill=True, fill_opacity=0.75,
            popup=folium.Popup(popup, max_width=280),
        ).add_to(m)

    year_str = str(year) if year else "all_years"
    safe_name = name.lower().replace(" ", "_")
    out = config.OUTPUT_MAPS / f"orlando_{year_str}_{safe_name}.html"
    m.save(str(out))
    print(f"  Saved: {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    config.OUTPUT_MAPS.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    incidents    = load_incidents()
    neighborhoods = load_neighborhoods()

    print("Joining incidents to neighborhoods...")
    joined = join_to_neighborhoods(incidents, neighborhoods)

    # Save enriched CSV
    out_csv = config.DATA_PROCESSED / "orlando_incidents_with_neighborhoods.csv"
    joined.drop(columns="geometry").to_csv(out_csv, index=False)
    print(f"Saved enriched CSV: {out_csv.name}")

    # All-years choropleth
    print("\nGenerating all-years choropleth...")
    summary_all = summarize_by_neighborhood(joined)
    choropleth_map(neighborhoods, summary_all, value_col="incidents")
    choropleth_map(neighborhoods, summary_all, value_col="killed")

    # Per-year choropleths
    years = sorted(joined["date"].dt.year.unique())
    for year in years:
        print(f"\nYear {year}:")
        summary = summarize_by_neighborhood(joined, year=year)
        choropleth_map(neighborhoods, summary, value_col="incidents", year=year)

        # Print top 5 neighborhoods for the year
        print(summary.head(5).to_string(index=False))

    # Example: zoom into a specific neighborhood
    # Uncomment and change the name to match your data:
    # map_neighborhood(joined, neighborhoods, name="Downtown", year=2023)

    print("\nDone.")


if __name__ == "__main__":
    main()
