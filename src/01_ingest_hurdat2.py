"""
01_ingest_hurdat2.py
Downloads and parses the HURDAT2 Atlantic hurricane track database.
Output: data/raw/hurdat2/tracks.parquet
"""
import requests
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/hurdat2")
RAW.mkdir(parents=True, exist_ok=True)

URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"


def download():
    print("Downloading HURDAT2...")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    (RAW / "hurdat2.txt").write_text(r.text, encoding="utf-8")
    print(f"  Saved {len(r.text):,} chars to {RAW / 'hurdat2.txt'}")


def parse(path: Path) -> pd.DataFrame:
    rows = []
    storm_id = storm_name = None

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split(",")]

        # Header line: AL122005, KATRINA, 34
        if len(parts) >= 3 and len(parts[0]) == 8 and parts[0][:2] in ("AL", "EP", "CP"):
            storm_id   = parts[0]
            storm_name = parts[1]
            continue

        # Data line: 20050823, 1800, , HU, 23.1N, 75.1W, 100, 957, ...
        if len(parts) < 8 or storm_id is None:
            continue
        try:
            date    = parts[0].strip()
            time    = parts[1].strip().zfill(4)
            status  = parts[3].strip()
            lat_str = parts[4].strip()
            lon_str = parts[5].strip()
            wind    = int(parts[6].strip())
            pres    = int(parts[7].strip()) if parts[7].strip() else None

            lat = float(lat_str[:-1]) * (1 if lat_str[-1] == "N" else -1)
            lon = float(lon_str[:-1]) * (-1 if lon_str[-1] == "W" else 1)

            rows.append({
                "storm_id":   storm_id,
                "storm_name": storm_name,
                "datetime":   pd.to_datetime(date + time, format="%Y%m%d%H%M"),
                "status":     status,
                "lat":        lat,
                "lon":        lon,
                "wind_kt":    wind,
                "pressure":   pres,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    df["year"] = df["datetime"].dt.year
    return df


if __name__ == "__main__":
    if not (RAW / "hurdat2.txt").exists():
        download()
    else:
        print("HURDAT2 already downloaded, parsing...")

    df = parse(RAW / "hurdat2.txt")

    # Keep Atlantic basin (AL), 1980+, tropical storms and hurricanes
    df = df[
        (df["storm_id"].str.startswith("AL")) &
        (df["year"] >= 1980) &
        (df["status"].isin(["TS", "HU", "SS", "TY"]))
    ].copy()

    df.to_parquet(RAW / "tracks.parquet", index=False)
    print(f"\n[OK] Saved {len(df):,} track waypoints from {df['storm_id'].nunique()} storms "
          f"({df['year'].min()}–{df['year'].max()})")
    print(df.head(3).to_string())
