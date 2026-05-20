"""
06_auto_update.py
─────────────────
Fully automated monthly update for the Orlando GVA dataset.

Downloads the latest incidents from gunviolencearchive.org using
Playwright browser automation, then merges them into the processed CSV.

Run manually:
  python 06_auto_update.py

Scheduled via Task Scheduler — see setup_schedule.ps1 to install.
Logs written to: output/update_log.txt
Debug screenshots saved to: output/debug/ on failure.
"""

import sys
import importlib.util
import logging
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

import config

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_FILE = config.OUTPUT_MAPS.parent / "update_log.txt"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

DEBUG_DIR = config.OUTPUT_MAPS.parent / "debug"

# ── Import clean() from 01_prepare_data.py ────────────────────────────────────
_spec = importlib.util.spec_from_file_location("prepare_data", config.ROOT / "01_prepare_data.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
clean = _mod.clean


# ── GVA automation ─────────────────────────────────────────────────────────────

def download_gva_csv(start: date, end: date) -> Path:
    """Automate GVA query for Florida/Orlando and download the result CSV."""
    start_str = start.strftime("%m/%d/%Y")
    end_str   = end.strftime("%m/%d/%Y")
    filename  = f"gva_orlando_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    out_path  = config.DATA_RAW / filename

    log.info(f"GVA query: Florida/Orlando  {start_str} to {end_str}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            accept_downloads=True,
        )
        page = context.new_page()

        try:
            # ── Query page ────────────────────────────────────────────────────
            page.goto("https://www.gunviolencearchive.org/query",
                      wait_until="networkidle", timeout=30_000)

            # ── Location filter ───────────────────────────────────────────────
            page.evaluate("document.querySelector('a.filter-dropdown-trigger').click()")
            page.wait_for_timeout(600)
            page.evaluate("document.querySelector('a[data-value=\"IncidentLocation\"]').click()")
            page.wait_for_timeout(2_000)

            page.locator('[name$="[state][select]"]').first.select_option(label="Florida")
            page.wait_for_timeout(300)
            page.locator('[name$="[city_county][city][textfield]"]').first.fill("Orlando")

            # ── Date filter ───────────────────────────────────────────────────
            page.evaluate("document.querySelector('a.filter-dropdown-trigger').click()")
            page.wait_for_timeout(600)
            page.evaluate("document.querySelector('a[data-value=\"IncidentDate\"]').click()")
            page.wait_for_timeout(2_000)

            page.locator('[name$="[date-from]"]').last.fill(start_str)
            page.locator('[name$="[date-to]"]').last.fill(end_str)

            # ── Search ────────────────────────────────────────────────────────
            page.locator("#edit-actions-execute").click()
            page.wait_for_load_state("networkidle", timeout=30_000)
            log.info(f"Search returned: {page.url}")

            # ── Trigger export ────────────────────────────────────────────────
            # The export link contains zero-width characters; click via JS.
            page.locator('a[href="#"]').first.click()
            page.wait_for_load_state("networkidle", timeout=60_000)

            if "export-finished" not in page.url:
                _save_debug(page, "export_redirect_failed")
                raise RuntimeError(f"Unexpected URL after export click: {page.url}")

            # ── Download ──────────────────────────────────────────────────────
            dl_link = page.locator('a[href*="export-finished/download"]').first
            dl_href = dl_link.get_attribute("href")
            if not dl_href:
                _save_debug(page, "no_download_link")
                raise RuntimeError("Download link not found on export-finished page")

            dl_url = f"https://www.gunviolencearchive.org{dl_href}"
            log.info(f"Downloading: {dl_url}")

            with page.expect_download(timeout=60_000) as dl_info:
                dl_link.click()
            dl = dl_info.value
            dl.save_as(str(out_path))
            log.info(f"Saved to: {out_path}")

        except Exception:
            _save_debug(page, "error")
            browser.close()
            raise

        browser.close()
    return out_path


def _save_debug(page, label: str):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    path = DEBUG_DIR / f"{label}_{date.today()}.png"
    try:
        page.screenshot(path=str(path), full_page=True)
        log.info(f"Debug screenshot: {path}")
    except Exception:
        pass


# ── Column normalisation ───────────────────────────────────────────────────────
# GVA's export format changed; new exports use human-readable headers.
# Map them to the internal names that config.py and clean() expect.
_NEW_COL_MAP = {
    "Incident ID":       config.COL_ID,       # "id"
    "Incident Date":     config.COL_DATE,      # "date"
    "State":             config.COL_STATE,     # "state"
    "City Or County":    config.COL_CITY,      # "city"
    "Address":           config.COL_ADDRESS,   # "address"
    "Victims Killed":    config.COL_KILLED,    # "number_killed"
    "Victims Injured":   config.COL_INJURED,   # "number_injured"
}

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename new-format GVA columns to the names config.py expects."""
    rename = {k: v for k, v in _NEW_COL_MAP.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df


# ── Data merge ─────────────────────────────────────────────────────────────────

def load_processed() -> pd.DataFrame:
    path = config.DATA_PROCESSED / "orlando_incidents.csv"
    if not path.exists():
        raise FileNotFoundError("Processed CSV not found — run 01_prepare_data.py first.")
    return pd.read_csv(path, parse_dates=["date"])


def merge(existing: pd.DataFrame, fresh: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    combined = pd.concat([existing, fresh], ignore_index=True)
    before = len(combined)
    if "incident_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["incident_id"])
    else:
        combined = combined.drop_duplicates(subset=["date", "address"])
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined, len(combined) - len(existing)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 50)
    log.info("Orlando GVA auto-update starting")

    existing = load_processed()
    last_date = existing["date"].max().date()
    today     = date.today()
    start     = last_date + timedelta(days=1)

    if start > today:
        log.info(f"Data already current through {last_date}. Nothing to do.")
        return

    log.info(f"Current data through {last_date}; fetching {start} to {today}")

    # Download
    csv_path = download_gva_csv(start, today)

    # Clean
    raw_df = pd.read_csv(csv_path, dtype=str)
    raw_df.columns = raw_df.columns.str.strip()
    raw_df = normalise_columns(raw_df)
    fresh = clean(raw_df)

    # Merge
    updated, new_count = merge(existing, fresh)

    if new_count <= 0:
        log.info("No new incidents after deduplication — CSV may overlap existing data.")
        return

    out = config.DATA_PROCESSED / "orlando_incidents.csv"
    updated.to_csv(out, index=False)

    new_last = updated["date"].max().date()
    log.info(f"Added {new_count:,} new incidents | total {len(updated):,} | now through {new_last}")
    log.info("Update complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception(f"Update failed: {e}")
        sys.exit(1)
