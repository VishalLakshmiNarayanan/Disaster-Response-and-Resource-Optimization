"""
06_build_panel.py
Builds the event-county panel by:
  1. Spatially matching HURDAT2 storm tracks to counties (via centroids)
  2. Joining FEMA declarations and PA project costs (demand label)
  3. Joining SVI and ACS features
Output: data/processed/panel.parquet
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import requests
import io

PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

MAX_DIST_KM   = 250   # counties within this radius of any track point
MIN_WIND_KT   = 34    # only events that reached tropical storm strength
LABEL_COL     = "demand_proxy"


def haversine_vectorised(lat1, lon1, lat2_arr, lon2_arr):
    """
    Compute distances (km) from a single point (lat1, lon1)
    to an array of points (lat2_arr, lon2_arr).
    """
    R = 6371.0
    dlat = np.radians(lat2_arr - lat1)
    dlon = np.radians(lon2_arr - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2_arr))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_county_centroids() -> pd.DataFrame:
    """Download Census Gazetteer county centroids."""
    gaz_path = Path("data/raw/county_centroids.parquet")
    if gaz_path.exists():
        return pd.read_parquet(gaz_path)

    print("Downloading county centroids from Census Gazetteer...")
    url = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
           "2023_Gazetteer/2023_Gaz_counties_national.zip")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    import zipfile
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        fname = [f for f in z.namelist() if f.endswith(".txt")][0]
        gaz = pd.read_csv(z.open(fname), sep="\t", dtype={"GEOID": str}, encoding="latin-1")

    # Strip trailing whitespace from column names (Census Gazetteer quirk)
    gaz.columns = gaz.columns.str.strip()

    gaz = gaz[["GEOID", "INTPTLAT", "INTPTLONG"]].rename(columns={
        "GEOID":     "fips",
        "INTPTLAT":  "county_lat",
        "INTPTLONG": "county_lon",
    })
    gaz["fips"] = gaz["fips"].astype(str).str.zfill(5)
    gaz.to_parquet(gaz_path, index=False)
    print(f"  [OK] {len(gaz):,} county centroids saved")
    return gaz


if __name__ == "__main__":
    out_path = PROC / "panel.parquet"
    if out_path.exists():
        print("panel.parquet already exists.")
        df = pd.read_parquet(out_path)
        print(f"Loaded {len(df):,} rows")
        print(df.head(3).to_string())
        exit()

    # - Load all raw data --------------------------─
    print("Loading raw data...")
    tracks = pd.read_parquet("data/raw/hurdat2/tracks.parquet")
    decl   = pd.read_parquet("data/raw/fema/declarations.parquet")
    pa     = pd.read_parquet("data/raw/fema/pa_projects.parquet")
    ncei   = pd.read_parquet("data/raw/ncei/storm_events.parquet")
    svi    = pd.read_parquet("data/raw/svi/svi_2020.parquet")
    acs    = pd.read_parquet("data/raw/acs/acs_county.parquet")

    centroids = load_county_centroids()

    county_lats = centroids["county_lat"].values
    county_lons = centroids["county_lon"].values

    # - Step 1: Match storms > counties -------------------
    print("\nMatching storm tracks to counties (may take a few minutes)...")
    storm_ids = tracks["storm_id"].unique()
    pairs = []

    for sid in tqdm(storm_ids, desc="Storms"):
        st = tracks[tracks["storm_id"] == sid]
        if st["wind_kt"].max() < MIN_WIND_KT:
            continue

        storm_year  = int(st["year"].iloc[0])
        storm_name  = st["storm_name"].iloc[0]
        storm_date  = st["datetime"].min()
        peak_wind   = int(st["wind_kt"].max())

        # Min distance from each county centroid to any track point
        min_dist = np.full(len(centroids), np.inf)
        for _, waypoint in st.iterrows():
            d = haversine_vectorised(
                waypoint["lat"], waypoint["lon"],
                county_lats, county_lons
            )
            min_dist = np.minimum(min_dist, d)

        affected_idx = np.where(min_dist <= MAX_DIST_KM)[0]
        for idx in affected_idx:
            row = centroids.iloc[idx]
            pairs.append({
                "storm_id":    sid,
                "storm_name":  storm_name,
                "storm_year":  storm_year,
                "storm_date":  storm_date,
                "fips":        row["fips"],
                "county_lat":  row["county_lat"],
                "county_lon":  row["county_lon"],
                "min_dist_km": round(min_dist[idx], 1),
                "peak_wind_kt": peak_wind,
            })

    panel = pd.DataFrame(pairs)
    print(f"  Raw storm-county pairs: {len(panel):,}")

    # - Step 2: FEMA label --------------------------
    print("\nAttaching FEMA labels...")

    # Inspect column names to handle API variations
    print("  Declaration columns:", decl.columns.tolist())
    print("  PA columns:", pa.columns.tolist())

    # Build county FIPS in declarations
    state_col  = next((c for c in decl.columns if "state" in c.lower() and "fips" in c.lower()), None)
    county_col = next((c for c in decl.columns if "county" in c.lower() and ("fips" in c.lower() or "code" in c.lower())), None)
    date_col   = next((c for c in decl.columns if "incidentbegin" in c.lower() or "declarationdate" in c.lower()), None)
    dnum_col   = next((c for c in decl.columns if "disasternumber" in c.lower()), None)

    if state_col and county_col:
        decl["fips"] = (
            decl[state_col].astype(str).str.zfill(2) +
            decl[county_col].astype(str).str.zfill(3)
        )
    else:
        print("  WARNING: Could not find FIPS columns in declarations")
        decl["fips"] = "00000"

    if date_col:
        decl["incident_year"] = pd.to_datetime(decl[date_col], errors="coerce").dt.year
    else:
        decl["incident_year"] = None

    # Aggregate PA dollars to county + disaster number
    pa_cost_col = next((c for c in pa.columns if "amount" in c.lower() or "cost" in c.lower()
                        or "obligated" in c.lower()), None)
    pa_state_col   = next((c for c in pa.columns if "state" in c.lower() and "fips" in c.lower() or
                           c.lower() in ("statenumbercode","statefipscode")), None)
    pa_county_col  = next((c for c in pa.columns if "county" in c.lower() and ("fips" in c.lower() or "code" in c.lower())), None)
    pa_dnum_col    = next((c for c in pa.columns if "disasternumber" in c.lower()), None)

    print(f"  PA cost column: {pa_cost_col}")
    print(f"  PA state col: {pa_state_col}, county col: {pa_county_col}")

    if pa_state_col and pa_county_col and pa_cost_col and pa_dnum_col:
        pa["fips"] = (
            pa[pa_state_col].astype(str).str.zfill(2) +
            pa[pa_county_col].astype(str).str.zfill(3)
        )
        pa[pa_cost_col] = pd.to_numeric(pa[pa_cost_col], errors="coerce").fillna(0)
        pa_agg = (
            pa.groupby([pa_dnum_col, "fips"])[pa_cost_col]
              .sum()
              .reset_index()
              .rename(columns={pa_dnum_col: "disasterNumber", pa_cost_col: "pa_total_usd"})
        )
    else:
        print("  WARNING: Could not build PA aggregation — label will be zero")
        pa_agg = pd.DataFrame(columns=["disasterNumber", "fips", "pa_total_usd"])

    # Join disaster number onto panel via year + fips
    if dnum_col:
        decl_link = decl[[dnum_col, "fips", "incident_year"]].drop_duplicates()
        decl_link = decl_link.rename(columns={dnum_col: "disasterNumber"})
    else:
        decl_link = pd.DataFrame(columns=["disasterNumber", "fips", "incident_year"])

    panel = panel.merge(
        decl_link,
        left_on=["fips", "storm_year"],
        right_on=["fips", "incident_year"],
        how="left"
    )
    panel = panel.merge(pa_agg, on=["disasterNumber", "fips"], how="left")
    panel["pa_total_usd"] = panel["pa_total_usd"].fillna(0)

    # Also attach NCEI damage as cross-validation signal
    ncei_agg = (
        ncei.groupby(["fips", "year"])
            .agg(
                ncei_damage_property=("DAMAGE_PROPERTY", "sum"),
                ncei_deaths=("DEATHS_DIRECT", "sum"),
                ncei_injuries=("INJURIES_DIRECT", "sum"),
            )
            .reset_index()
    )
    panel = panel.merge(
        ncei_agg,
        left_on=["fips", "storm_year"],
        right_on=["fips", "year"],
        how="left"
    )

    # - Step 3: SVI -----------------------------─
    print("Attaching SVI...")
    panel = panel.merge(svi, on="fips", how="left")

    # - Step 4: ACS (nearest vintage) --------------------
    print("Attaching ACS...")
    acs_years = sorted(acs["year"].unique())
    def nearest_acs(y):
        return min(acs_years, key=lambda a: abs(a - y))

    panel["acs_year"] = panel["storm_year"].apply(nearest_acs)
    panel = panel.merge(
        acs.rename(columns={"year": "acs_year"}),
        on=["fips", "acs_year"],
        how="left"
    )

    # - Step 5: Build demand label ----------------------
    print("Building demand label...")
    panel["pa_per_capita"] = (
        panel["pa_total_usd"] /
        panel["total_pop"].replace(0, np.nan)
    ).clip(lower=0)

    # 1 supply unit = $1,000 PA per capita (documented conversion)
    panel[LABEL_COL] = (panel["pa_per_capita"] / 1_000).fillna(0)

    # Mark whether the county had a FEMA disaster declaration
    panel["fema_declared"] = panel["disasterNumber"].notna().astype(int)

    print(f"\n- Panel summary ----------------")
    print(f"  Total rows:          {len(panel):,}")
    print(f"  Storms:              {panel['storm_id'].nunique()}")
    print(f"  Counties touched:    {panel['fips'].nunique():,}")
    print(f"  FEMA declared rows:  {panel['fema_declared'].sum():,}")
    print(f"  Label > 0:           {(panel[LABEL_COL] > 0).sum():,}")
    print(f"  Label stats:\n{panel[LABEL_COL].describe().round(4)}")

    panel.to_parquet(out_path, index=False)
    print(f"\n[OK] Panel saved to {out_path}")
