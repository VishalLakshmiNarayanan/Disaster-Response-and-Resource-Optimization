"""
03_ingest_ncei.py
Downloads NOAA NCEI Storm Events CSV files for hurricane/tropical storm events
from 1996 to 2023.
Output: data/raw/ncei/storm_events.parquet
"""
import re
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW = Path("data/raw/ncei")
RAW.mkdir(parents=True, exist_ok=True)

BASE      = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
START_YR  = 1996
END_YR    = 2023

TARGET_TYPES = {
    "Hurricane (Typhoon)", "Tropical Storm",
    "Marine Tropical Storm", "Marine Hurricane/Typhoon",
}

KEEP_COLS = [
    "BEGIN_YEARMONTH", "BEGIN_DAY", "END_YEARMONTH", "END_DAY",
    "STATE", "STATE_FIPS", "CZ_FIPS", "CZ_NAME", "CZ_TYPE",
    "EVENT_TYPE",
    "DEATHS_DIRECT", "DEATHS_INDIRECT",
    "INJURIES_DIRECT", "INJURIES_INDIRECT",
    "DAMAGE_PROPERTY", "DAMAGE_CROPS",
    "BEGIN_LAT", "BEGIN_LON",
]


def damage_to_float(val) -> float:
    if pd.isna(val) or str(val).strip() in ("", "0"):
        return 0.0
    val = str(val).upper().strip()
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9}
    if val and val[-1] in multipliers:
        try:
            return float(val[:-1]) * multipliers[val[-1]]
        except ValueError:
            return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


if __name__ == "__main__":
    out_path = RAW / "storm_events.parquet"
    if out_path.exists():
        print("storm_events.parquet already exists, skipping download.")
        df_check = pd.read_parquet(out_path)
        print(f"Loaded {len(df_check):,} rows")
        print(df_check["EVENT_TYPE"].value_counts())
        exit()

    print("Fetching file listing from NCEI...")
    r = requests.get(BASE, timeout=30)
    r.raise_for_status()

    # Find all detail files and extract years
    all_files = re.findall(
        r'(StormEvents_details-ftp_v1\.0_d(\d{4})_c\d+\.csv\.gz)',
        r.text
    )
    # For each year keep only the latest version (last in list)
    year_to_file: dict[int, str] = {}
    for fname, yr in all_files:
        y = int(yr)
        if START_YR <= y <= END_YR:
            year_to_file[y] = fname  # later entries overwrite earlier ones

    print(f"Found files for {len(year_to_file)} years ({START_YR}–{END_YR})")

    frames = []
    for year in tqdm(sorted(year_to_file.keys()), desc="NCEI years"):
        url = BASE + year_to_file[year]
        try:
            df = pd.read_csv(url, compression="gzip", low_memory=False)
            df = df[df["EVENT_TYPE"].isin(TARGET_TYPES)].copy()
            if df.empty:
                continue
            available = [c for c in KEEP_COLS if c in df.columns]
            df = df[available].copy()
            df["year"] = year
            frames.append(df)
        except Exception as e:
            print(f"  {year}: ERROR — {e}")

    if not frames:
        print("ERROR: No data retrieved. Check internet connection.")
        exit(1)

    ncei = pd.concat(frames, ignore_index=True)

    # Parse damage columns
    for col in ["DAMAGE_PROPERTY", "DAMAGE_CROPS"]:
        if col in ncei.columns:
            ncei[col] = ncei[col].apply(damage_to_float)

    # Build 5-digit county FIPS
    ncei["fips"] = (
        ncei["STATE_FIPS"].astype(str).str.zfill(2) +
        ncei["CZ_FIPS"].astype(str).str.zfill(3)
    )

    ncei.to_parquet(out_path, index=False)
    print(f"\n[OK] NCEI saved: {len(ncei):,} event-county rows")
    print(ncei["EVENT_TYPE"].value_counts())
    print(ncei[["STATE", "fips", "DAMAGE_PROPERTY"]].head(5).to_string())
