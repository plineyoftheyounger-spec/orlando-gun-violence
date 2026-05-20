from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT              = Path(__file__).parent
DATA_RAW          = ROOT / "data" / "raw"
DATA_PROCESSED    = ROOT / "data" / "processed"
OUTPUT_MAPS       = ROOT / "output" / "maps"
NEIGHBORHOODS_DIR = ROOT / "neighborhoods"

# ── Orlando map center ─────────────────────────────────────────────────────────
ORLANDO_LAT  = 28.5383
ORLANDO_LON  = -81.3792
DEFAULT_ZOOM = 12

# ── GVA CSV column names ───────────────────────────────────────────────────────
# Matched to the actual column names in the downloaded CSV.
COL_ID      = "id"
COL_DATE    = "date"
COL_STATE   = "state"
COL_CITY    = "city"
COL_ADDRESS = "address"
COL_KILLED  = "number_killed"
COL_INJURED = "number_injured"
COL_LAT     = "latitude"
COL_LON     = "longitude"

# Extra boolean flag columns present in this export (kept as-is in processed CSV)
COL_FLAGS = ["child_involved", "officer_involved", "mass_shooting",
             "accidental", "domestic_violence", "drive_by"]

# ── Heatmap settings ───────────────────────────────────────────────────────────
HEAT_RADIUS      = 15
HEAT_BLUR        = 10
HEAT_MIN_OPACITY = 0.3
# Fatality weight multiplier vs. injury-only incidents
FATALITY_WEIGHT  = 3
