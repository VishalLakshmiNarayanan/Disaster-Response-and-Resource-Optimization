"""
02_ingest_fema.py
Pulls FEMA disaster declarations and Public Assistance funded project details
for hurricane incidents via the OpenFEMA API (no key required).
Outputs:
  data/raw/fema/declarations.parquet
  data/raw/fema/pa_projects.parquet
"""
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

RAW = Path("data/raw/fema")
RAW.mkdir(parents=True, exist_ok=True)

BASE = "https://www.fema.gov/api/open/v2"


def fetch_all(endpoint: str, params: dict, top: int = 1000) -> pd.DataFrame:
    """Page through FEMA API and return all records as a DataFrame."""
    params = {**params, "$top": top, "$inlinecount": "allpages", "$format": "json"}
    frames = []
    skip   = 0
    total  = None

    with tqdm(desc=endpoint, unit=" records") as bar:
        while True:
            params["$skip"] = skip
            for attempt in range(3):
                try:
                    r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=60)
                    r.raise_for_status()
                    break
                except Exception as e:
                    print(f"\n  Attempt {attempt+1} failed: {e}")
                    time.sleep(5)
            else:
                print(f"\n  Giving up on {endpoint} at skip={skip}")
                break

            data = r.json()
            # Metadata key
            if total is None:
                total = int(data.get("metadata", {}).get("count", 0))
                bar.total = total

            # Find the data key (everything except 'metadata')
            key = next((k for k in data if k != "metadata"), None)
            if key is None:
                break

            records = data[key]
            if not records:
                break

            frames.append(pd.DataFrame(records))
            skip += len(records)
            bar.update(len(records))

            if skip >= total:
                break

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    # - 1. Disaster Declarations (filter to Hurricane) ------------
    decl_path = RAW / "declarations.parquet"
    if not decl_path.exists():
        print("\n[1/2] Fetching hurricane disaster declarations...")
        decl = fetch_all(
            "DisasterDeclarationsSummaries",
            {"$filter": "incidentType eq 'Hurricane'"}
        )
        decl.to_parquet(decl_path, index=False)
        print(f"[OK] Declarations: {len(decl):,} rows saved")
    else:
        print("[1/2] declarations.parquet already exists, loading...")
        decl = pd.read_parquet(decl_path)
        print(f"  Loaded {len(decl):,} rows")

    print("\nDeclaration columns:", decl.columns.tolist())
    print(decl.head(2).to_string())

    # - 2. PA Funded Project Details ---------------------
    pa_path = RAW / "pa_projects.parquet"
    if not pa_path.exists():
        print("\n[2/2] Fetching PA funded project details (this takes a while)...")

        hurricane_ids = decl["disasterNumber"].dropna().astype(int).unique().tolist()
        print(f"  Found {len(hurricane_ids)} hurricane disaster numbers")

        pa_frames = []
        batch_size = 30
        for i in tqdm(range(0, len(hurricane_ids), batch_size), desc="PA batches"):
            batch = hurricane_ids[i: i + batch_size]
            id_filter = " or ".join([f"disasterNumber eq {d}" for d in batch])
            chunk = fetch_all(
                "PublicAssistanceFundedProjectsDetails",
                {"$filter": id_filter}
            )
            if not chunk.empty:
                pa_frames.append(chunk)
            time.sleep(0.5)  # be polite to the API

        if pa_frames:
            pa = pd.concat(pa_frames, ignore_index=True)
            pa.to_parquet(pa_path, index=False)
            print(f"[OK] PA Projects: {len(pa):,} rows saved")
        else:
            print("WARNING: No PA data retrieved")
    else:
        print("[2/2] pa_projects.parquet already exists, loading...")
        pa = pd.read_parquet(pa_path)
        print(f"  Loaded {len(pa):,} rows")

    print("\nPA columns:", pa.columns.tolist())
    print(pa.head(2).to_string())
