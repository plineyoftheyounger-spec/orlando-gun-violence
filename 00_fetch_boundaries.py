"""
00_fetch_boundaries.py
──────────────────────
Downloads boundary GeoJSON files from the City of Orlando's ArcGIS services.
Run this once before running 04_analysis_maps.py.

Saves:
  neighborhoods/kidz_zones.geojson            — 121 Orlando parks (Kidz Zones program)
  neighborhoods/orlando_neighborhoods.geojson — official neighborhood boundaries
"""

import json
import urllib.request
import config

SOURCES = {
    "kidz_zones.geojson": (
        "https://services5.arcgis.com/mMuoPCaIYD4wEgDl/arcgis/rest/services"
        "/Orlando_Kidz_Zones/FeatureServer/0/query"
        "?where=1%3D1&outFields=*&f=geojson"
    ),
    "orlando_neighborhoods.geojson": (
        "https://services5.arcgis.com/mMuoPCaIYD4wEgDl/arcgis/rest/services"
        "/OrlandoPoliticalNeighborhoods/FeatureServer/0/query"
        "?where=1%3D1&outFields=*&f=geojson"
    ),
}

def fetch(url, out_path):
    print(f"  Fetching {out_path.name} ...")
    with urllib.request.urlopen(url) as r:
        data = json.loads(r.read().decode())
    n = len(data.get("features", []))
    if n == 0:
        print(f"  Warning: 0 features returned — check the URL.")
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  Saved {n} features → {out_path.name}")

def main():
    config.NEIGHBORHOODS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCES.items():
        out = config.NEIGHBORHOODS_DIR / filename
        if out.exists():
            print(f"  {filename} already exists — delete it to re-download")
        else:
            fetch(url, out)
    print(f"\nDone. Files in: {config.NEIGHBORHOODS_DIR}")

if __name__ == "__main__":
    main()
