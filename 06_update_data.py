"""
06_update_data.py
─────────────────
Monthly updater for the Orlando GVA dataset.

Workflow:
  1. Run this script — it shows you exactly what date range to download.
  2. Go to https://www.gunviolencearchive.org/query
       Filter: State = Florida  |  City/County = Orlando
       Date range: (shown below)
       Click Search → Export CSV → save anywhere in data/raw/
  3. Press Enter here — new rows are merged in without duplicates.
"""

import sys
import importlib.util
from pathlib import Path
from datetime import date, timedelta

import pandas as pd

import config

# ── Import clean() from 01_prepare_data.py (numeric filename, can't import normally) ──
_spec = importlib.util.spec_from_file_location(
    "prepare_data", config.ROOT / "01_prepare_data.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
clean = _mod.clean


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_processed() -> pd.DataFrame:
    path = config.DATA_PROCESSED / "orlando_incidents.csv"
    if not path.exists():
        print("No processed file found — run 01_prepare_data.py first.")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def last_covered_date(df: pd.DataFrame) -> date:
    return df["date"].max().date()


def load_all_raw() -> pd.DataFrame:
    csvs = list(config.DATA_RAW.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {config.DATA_RAW}")
        sys.exit(1)
    frames = []
    for f in sorted(csvs):
        raw = pd.read_csv(f, dtype=str)
        raw.columns = raw.columns.str.strip()
        frames.append(raw)
    return pd.concat(frames, ignore_index=True)


def merge(existing: pd.DataFrame, fresh: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Combine existing processed data with freshly cleaned data; return (merged, new_count)."""
    combined = pd.concat([existing, fresh], ignore_index=True)

    before = len(combined)
    if "incident_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["incident_id"])
    else:
        combined = combined.drop_duplicates(subset=["date", "address"])
    combined = combined.sort_values("date").reset_index(drop=True)

    new_count = len(combined) - len(existing)
    return combined, new_count


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    existing = load_processed()
    last_date = last_covered_date(existing)
    today = date.today()
    start = last_date + timedelta(days=1)

    print("=" * 60)
    print("  Orlando GVA — Monthly Data Update")
    print("=" * 60)
    print(f"  Current data through : {last_date}")
    print(f"  Today                : {today}")

    if start > today:
        print("\n  Data is already up to date. Nothing to do.")
        return

    print(f"\n  Missing data         : {start}  →  {today}")
    print(f"  Gap                  : {(today - last_date).days} days\n")

    print("─" * 60)
    print("  DOWNLOAD INSTRUCTIONS")
    print("─" * 60)
    print("  1. Open: https://www.gunviolencearchive.org/query")
    print("  2. Add filters:")
    print("       State        = Florida")
    print("       City/County  = Orlando")
    print(f"      Date range   = {start.strftime('%m/%d/%Y')}  to  {today.strftime('%m/%d/%Y')}")
    print("  3. Click Search → Export CSV")
    print(f"  4. Save the file to:  {config.DATA_RAW}")
    print("─" * 60)
    print()

    input("  Press Enter once you've saved the CSV to data/raw/ ...  ")
    print()

    print("  Loading raw CSVs...")
    raw_df = load_all_raw()

    print("  Cleaning...")
    fresh = clean(raw_df)

    print("  Merging with existing data...")
    updated, new_count = merge(existing, fresh)

    if new_count <= 0:
        print(f"\n  No new incidents found after deduplication.")
        print("  Make sure the downloaded CSV contains dates after", last_date)
        return

    out = config.DATA_PROCESSED / "orlando_incidents.csv"
    updated.to_csv(out, index=False)

    new_last = updated["date"].max().date()
    print(f"\n  Done!")
    print(f"  Added       : {new_count:,} new incidents")
    print(f"  Total rows  : {len(updated):,}")
    print(f"  Now covers  : {updated['date'].min().date()}  →  {new_last}")
    print(f"  Saved to    : {out}")
    print()
    print("  Re-run 02_create_maps.py through 05_era_maps.py to refresh outputs.")


if __name__ == "__main__":
    main()
