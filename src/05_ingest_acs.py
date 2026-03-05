"""
05_ingest_acs.py
Pulls ACS 5-year county-level estimates (2012–2022) via the Census API.
Output: data/raw/acs/acs_county.parquet
"""
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from census import Census

load_dotenv()
API_KEY = os.getenv("CENSUS_API_KEY")

RAW = Path("data/raw/acs")
RAW.mkdir(parents=True, exist_ok=True)

# ACS variable codes > readable names
VARS = {
    "B01003_001E": "total_pop",
    "B25044_003E": "renter_no_vehicle",
    "B25044_010E": "owner_no_vehicle",
    # Age 65+ (male)
    "B01001_020E": "m_65_66",
    "B01001_021E": "m_67_69",
    "B01001_022E": "m_70_74",
    "B01001_023E": "m_75_79",
    "B01001_024E": "m_80_84",
    "B01001_025E": "m_85plus",
    # Age 65+ (female)
    "B01001_044E": "f_65_66",
    "B01001_045E": "f_67_69",
    "B01001_046E": "f_70_74",
    "B01001_047E": "f_75_79",
    "B01001_048E": "f_80_84",
    "B01001_049E": "f_85plus",
    # Economic / housing
    "B19013_001E": "median_hh_income",
    "B25035_001E": "median_year_built",
    "B25001_001E": "total_housing_units",
}


def pull_year(c: Census, year: int) -> pd.DataFrame:
    fields = list(VARS.keys())
    try:
        data = c.acs5.get(fields, {"for": "county:*", "in": "state:*"}, year=year)
    except Exception as e:
        print(f"  ACS {year} failed: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    df = df.rename(columns=VARS)
    df["year"] = year

    # Convert to numeric, coerce errors
    num_cols = list(VARS.values())
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived features
    age65_cols = ["m_65_66","m_67_69","m_70_74","m_75_79","m_80_84","m_85plus",
                  "f_65_66","f_67_69","f_70_74","f_75_79","f_80_84","f_85plus"]
    available_age = [c2 for c2 in age65_cols if c2 in df.columns]
    df["pop_65plus"] = df[available_age].fillna(0).sum(axis=1)

    df["no_vehicle"] = (
        df.get("renter_no_vehicle", pd.Series(0, index=df.index)).fillna(0) +
        df.get("owner_no_vehicle",  pd.Series(0, index=df.index)).fillna(0)
    )

    keep = ["fips", "year", "total_pop", "no_vehicle", "pop_65plus",
            "median_hh_income", "median_year_built", "total_housing_units"]
    available_keep = [c2 for c2 in keep if c2 in df.columns]
    return df[available_keep]


if __name__ == "__main__":
    if not API_KEY:
        raise ValueError("CENSUS_API_KEY not found in .env — check your .env file")

    out_path = RAW / "acs_county.parquet"
    if out_path.exists():
        print("acs_county.parquet already exists, skipping.")
        df = pd.read_parquet(out_path)
        print(f"Loaded {len(df):,} county-year rows")
        exit()

    c = Census(API_KEY)
    frames = []

    for year in tqdm(range(2012, 2023), desc="ACS years"):
        df = pull_year(c, year)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("ERROR: No ACS data retrieved. Check Census API key.")
        exit(1)

    acs = pd.concat(frames, ignore_index=True)
    # Replace Census sentinel -666666666 with NaN
    acs = acs.replace(-666666666, float("nan"))
    acs = acs.replace(-666666666.0, float("nan"))

    acs.to_parquet(out_path, index=False)
    print(f"\n[OK] ACS saved: {len(acs):,} county-year rows, {acs['year'].nunique()} vintages")
    print(acs.head(3).to_string())
