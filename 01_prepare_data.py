"""
01_prepare_data.py
──────────────────
Reads all GVA CSVs from data/raw/, cleans and merges them,
and writes a single processed CSV to data/processed/orlando_incidents.csv.

HOW TO DOWNLOAD FROM GVA:
  1. Go to https://www.gunviolencearchive.org/query
  2. Add filter: State = Florida, City = Orlando
  3. Set a date range for a single year (e.g., 01/01/2020 – 12/31/2020)
  4. Click Search, then Export CSV
  5. Save the file to data/raw/ with a descriptive name like gva_orlando_2020.csv
  6. Repeat for each year you want
  7. Run this script once to merge everything

GVA COLUMN NOTES:
  - Latitude / Longitude are included in most GVA exports.
  - If your CSV is missing those columns, set GEOCODE_MISSING = True below
    and the script will geocode addresses via Nominatim (free, slow, rate-limited).
"""

import pandas as pd
import re
from pathlib import Path

import config

# ── Settings ───────────────────────────────────────────────────────────────────
GEOCODE_MISSING = False   # Set True if your CSVs lack Latitude/Longitude
GEOCODE_DELAY   = 1.5     # Seconds between geocode requests (be polite to Nominatim)

# ── Load & Merge ───────────────────────────────────────────────────────────────

def load_raw_csvs() -> pd.DataFrame:
    csv_files = list(config.DATA_RAW.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {config.DATA_RAW}\n"
            "Download data from GVA and place CSVs in that folder."
        )

    frames = []
    for f in sorted(csv_files):
        print(f"  Reading {f.name}...")
        df = pd.read_csv(f, dtype=str)
        df.columns = df.columns.str.strip()
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(merged):,} rows from {len(csv_files)} file(s).")
    return merged


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # ── Required columns check ─────────────────────────────────────────────────
    required = [config.COL_DATE, config.COL_KILLED, config.COL_INJURED]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing expected columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}\n"
            "Check config.py and adjust COL_* names to match your export."
        )

    # ── Rename to internal names ───────────────────────────────────────────────
    rename_map = {
        config.COL_ID:      "incident_id",
        config.COL_DATE:    "date",
        config.COL_STATE:   "state",
        config.COL_CITY:    "city",
        config.COL_ADDRESS: "address",
        config.COL_KILLED:  "killed",
        config.COL_INJURED: "injured",
    }
    # Only rename columns that actually exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # ── Lat / Lon ──────────────────────────────────────────────────────────────
    if config.COL_LAT in df.columns and config.COL_LON in df.columns:
        df = df.rename(columns={config.COL_LAT: "lat", config.COL_LON: "lon"})
    else:
        print("  Warning: Latitude/Longitude columns not found in CSV.")
        df["lat"] = None
        df["lon"] = None

    # ── Types ──────────────────────────────────────────────────────────────────
    df["date"]    = pd.to_datetime(df["date"], errors="coerce")
    df["killed"]  = pd.to_numeric(df["killed"],  errors="coerce").fillna(0).astype(int)
    df["injured"] = pd.to_numeric(df["injured"], errors="coerce").fillna(0).astype(int)
    df["lat"]     = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"]     = pd.to_numeric(df["lon"], errors="coerce")

    # ── Boolean flag columns (child_involved, officer_involved, etc.) ──────────
    for col in config.COL_FLAGS:
        if col in df.columns:
            df[col] = df[col].map({"True": True, "False": False, True: True, False: False}).fillna(False).astype(bool)

    # ── Derived fields ─────────────────────────────────────────────────────────
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["total"] = df["killed"] + df["injured"]
    df["fatal"] = df["killed"] > 0
    # Use the pre-flagged column if present, otherwise fall back to FBI definition (4+ victims)
    if "mass_shooting" in df.columns:
        df["mass"] = df["mass_shooting"]
    else:
        df["mass"] = df["total"] >= 4

    # ── Dedup ──────────────────────────────────────────────────────────────────
    before = len(df)
    if "incident_id" in df.columns:
        df = df.drop_duplicates(subset=["incident_id"])
    else:
        df = df.drop_duplicates(subset=["date", "address"])
    after = len(df)
    if before != after:
        print(f"  Removed {before - after:,} duplicate rows.")

    # ── Drop rows with no valid date ───────────────────────────────────────────
    df = df.dropna(subset=["date"])

    print(f"  Cleaned: {len(df):,} incidents | "
          f"{df['killed'].sum():,} killed | {df['injured'].sum():,} injured")
    return df.sort_values("date").reset_index(drop=True)


# ── Optional geocoding ─────────────────────────────────────────────────────────

def geocode_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Geocode rows that are missing lat/lon using Nominatim (free)."""
    try:
        from geopy.geocoders import Nominatim
        from geopy.extra.rate_limiter import RateLimiter
        import time
    except ImportError:
        print("  Install geopy to use geocoding: pip install geopy")
        return df

    missing = df["lat"].isna() | df["lon"].isna()
    if not missing.any():
        print("  All rows already have lat/lon — skipping geocoding.")
        return df

    count = missing.sum()
    print(f"  Geocoding {count:,} rows missing coordinates (this may take a while)...")

    geolocator = Nominatim(user_agent="orlando_gv_analysis")
    geocode    = RateLimiter(geolocator.geocode, min_delay_seconds=GEOCODE_DELAY)

    def get_coords(row):
        if pd.notna(row["lat"]) and pd.notna(row["lon"]):
            return row["lat"], row["lon"]
        query = f"{row.get('address', '')}, Orlando, FL"
        try:
            loc = geocode(query)
            if loc:
                return loc.latitude, loc.longitude
        except Exception:
            pass
        return None, None

    for idx in df[missing].index:
        lat, lon = get_coords(df.loc[idx])
        df.at[idx, "lat"] = lat
        df.at[idx, "lon"] = lon

    resolved = df["lat"].notna().sum() - (len(df) - count)
    print(f"  Geocoded {resolved:,} of {count:,} missing locations.")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("Loading raw CSVs...")
    df = load_raw_csvs()

    print("Cleaning data...")
    df = clean(df)

    if GEOCODE_MISSING:
        print("Geocoding missing coordinates...")
        df = geocode_missing(df)

    no_coords = df["lat"].isna().sum()
    if no_coords:
        print(f"  Note: {no_coords:,} rows have no coordinates and won't appear on maps.")

    out = config.DATA_PROCESSED / "orlando_incidents.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # ── Quick summary ──────────────────────────────────────────────────────────
    print("\n── Incidents by Year ──────────────────────────────────────────────")
    agg = {
        "incidents":      ("date",     "count"),
        "killed":         ("killed",   "sum"),
        "injured":        ("injured",  "sum"),
        "mass_shootings": ("mass",     "sum"),
    }
    for flag in ["child_involved", "officer_involved", "domestic_violence", "drive_by"]:
        if flag in df.columns:
            agg[flag] = (flag, "sum")
    summary = df.groupby("year").agg(**agg)
    print(summary.to_string())


if __name__ == "__main__":
    main()
