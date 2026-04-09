"""
web/app.py  --  Flask backend for DisasterAI dashboard
Run:  python web/app.py
      (from project root or from web/ folder)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pulp
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Paths  (web/data/ is bundled for deployment; fall back to project root for dev)
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
BASE = HERE.parent

app = Flask(__name__, template_folder=str(HERE / "templates"),
            static_folder=str(HERE / "static"))

def _find(bundled_name, fallback):
    """Return bundled path if it exists, otherwise fall back to project-root path."""
    p = HERE / "data" / bundled_name
    return p if p.exists() else BASE / fallback

# ---------------------------------------------------------------------------
# Load data once at startup
# ---------------------------------------------------------------------------
preds   = pd.read_parquet(_find("predictions.parquet",         "data/processed/predictions.parquet"))
results = pd.read_parquet(_find("all_storms_comparison.parquet","outputs/allocations/all_storms_comparison.parquet"))
panel   = pd.read_parquet(_find("panel.parquet",               "data/processed/panel.parquet"))

svi_raw = pd.read_parquet(_find("svi_2020.parquet",            "data/raw/svi/svi_2020.parquet"))
if svi_raw.index.name == "fips":
    svi_raw = svi_raw.reset_index()
svi_lookup = svi_raw.set_index("fips")["svi_overall"].to_dict()

# Population lookup (for synthetic demand in custom scenario)
pop_lookup = {}
if "E_TOTPOP" in svi_raw.columns:
    pop_lookup = svi_raw.set_index("fips")["E_TOTPOP"].fillna(10000).to_dict()

# Average predicted demand per county across all storms (for custom scenario fallback)
preds_avg_demand = preds.groupby("fips")["pred_demand"].mean().to_dict()
# Scale factor: mean demand per 1000 population (for unseen counties)
mean_demand = float(preds["pred_demand"].mean()) if not preds.empty else 0.07
mean_pop    = float(np.nanmean(list(pop_lookup.values()))) if pop_lookup else 50000.0
demand_per_pop = mean_demand / (mean_pop / 1000.0)   # demand units per 1k people

# County centroids
coords_df = (
    panel[["fips", "county_lat", "county_lon"]]
    .drop_duplicates("fips")
    .dropna(subset=["county_lat", "county_lon"])
)
coords_lookup = coords_df.set_index("fips")[["county_lat", "county_lon"]].to_dict("index")

# Pre-join coords into preds for fast responses
preds = preds.merge(coords_df, on="fips", how="left")

# Overall averages
avg_policy = (
    results.groupby("policy")[
        ["pct_unmet", "coverage_overall", "coverage_high_svi", "coverage_low_svi", "equity_gap"]
    ].mean()
)

SVI_THRESHOLD   = 0.75
SUPPLY_FRACTION = 0.80
EQUITY_FRAC     = 0.40

RESOURCE_TYPES = {
    "hurricane": ["Water & Filtration", "Emergency Shelter", "Medical Supplies",
                  "Food & MREs", "Power Generators", "Search & Rescue"],
    "flood":     ["Water Pumps & Boats", "Emergency Shelter", "Sanitation Kits",
                  "Medical Supplies", "Food & MREs", "Debris Removal"],
    "tornado":   ["Search & Rescue", "Emergency Shelter", "Medical Supplies",
                  "Structural Assessment", "Food & MREs", "Power Restoration"],
    "wildfire":  ["Evacuation Support", "Air Quality Kits", "Medical Supplies",
                  "Temporary Housing", "Food & MREs", "Animal Rescue"],
}

# ---------------------------------------------------------------------------
# Helper: run LP
# ---------------------------------------------------------------------------
def run_lp(demand: pd.Series, svi: pd.Series, equity_frac: float,
           supply_frac: float, svi_threshold: float):
    S        = demand.sum() * supply_frac
    counties = demand.index.tolist()
    high_svi = svi[svi >= svi_threshold].index.tolist()
    svi_d    = svi.to_dict()
    weights  = {i: (2.0 if float(svi_d.get(i, 0)) >= svi_threshold else 1.0) for i in counties}

    prob  = pulp.LpProblem("alloc", pulp.LpMinimize)
    alloc = pulp.LpVariable.dicts("a", counties, lowBound=0)
    unmet = pulp.LpVariable.dicts("u", counties, lowBound=0)

    prob += pulp.lpSum(weights[i] * unmet[i] for i in counties)
    for i in counties:
        prob += unmet[i] >= demand[i] - alloc[i]
    prob += pulp.lpSum(alloc[i] for i in counties) <= S
    if high_svi:
        prob += (pulp.lpSum(alloc[i] for i in high_svi if i in alloc)
                 >= equity_frac * S)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    lp_alloc = pd.Series(
        {i: max(0.0, pulp.value(alloc[i]) or 0.0) for i in counties}
    )
    return lp_alloc, S


def coverage(unmet_s: pd.Series, demand_s: pd.Series) -> float:
    total = demand_s.sum()
    return float(1 - unmet_s.sum() / total) if total > 0 else 1.0


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    pop  = avg_policy.loc["population_proportional"]
    lp   = avg_policy.loc["model_lp_equity"]
    kpis = {
        "unmet_reduction":       round((pop["pct_unmet"] - lp["pct_unmet"]) / pop["pct_unmet"] * 100, 1),
        "hisvi_coverage":        round(lp["coverage_high_svi"] * 100, 1),
        "equity_gap_improvement":round((lp["equity_gap"] - pop["equity_gap"]) * 100, 1),
        "storms_analyzed":       int(results["storm_id"].nunique()),
        "counties_covered":      int(preds["fips"].nunique()),
    }
    return render_template("index.html", kpis=kpis)


@app.route("/simulator")
def simulator():
    storms = (
        preds.groupby(["storm_id", "storm_name", "storm_year"])
        .size()
        .reset_index(name="n_counties")
        .sort_values(["storm_year", "storm_name"], ascending=[False, True])
        .to_dict("records")
    )
    return render_template("simulator.html", storms=storms)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.route("/api/counties/svi")
def api_counties_svi():
    """FIPS -> SVI score for map coloring (all US counties)."""
    return jsonify(svi_lookup)


@app.route("/api/storms")
def api_storms():
    storms = (
        preds.groupby(["storm_id", "storm_name", "storm_year"])
        .size()
        .reset_index(name="n_counties")
        .sort_values(["storm_year", "storm_name"], ascending=[False, True])
    )
    return jsonify(storms.to_dict("records"))


@app.route("/api/storm/<storm_id>")
def api_storm(storm_id):
    sub = preds[preds["storm_id"] == storm_id].copy()
    if sub.empty:
        return jsonify({"error": "Storm not found"}), 404

    cols = ["fips", "storm_name", "storm_year", "demand_proxy", "pred_demand",
            "svi_overall", "fema_declared", "p_nonzero",
            "county_lat", "county_lon", "peak_wind_kt", "min_dist_km"]
    cols = [c for c in cols if c in sub.columns]
    records = sub[cols].fillna(0).to_dict("records")

    return jsonify({
        "storm_id":   storm_id,
        "storm_name": sub["storm_name"].iloc[0],
        "storm_year": int(sub["storm_year"].iloc[0]),
        "n_counties": len(sub),
        "counties":   records,
    })


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    body          = request.get_json(force=True)
    storm_id      = body.get("storm_id")
    fips_list     = body.get("fips", [])
    equity_frac   = float(body.get("equity_frac",   EQUITY_FRAC))
    supply_frac   = float(body.get("supply_frac",   SUPPLY_FRACTION))
    svi_threshold = float(body.get("svi_threshold", SVI_THRESHOLD))
    disaster_type = body.get("disaster_type", "hurricane")

    if storm_id:
        sub    = preds[preds["storm_id"] == storm_id].set_index("fips")
        demand = sub["pred_demand"].clip(lower=0)
        svi    = sub["svi_overall"].fillna(0.5)
    elif fips_list:
        # Build demand + SVI for every selected FIPS
        demand_rows, svi_rows = {}, {}
        for f in fips_list:
            # 1. Historical average from predictions (best)
            if f in preds_avg_demand:
                demand_rows[f] = max(preds_avg_demand[f], 0.0)
            else:
                # 2. Synthetic: population * demand_per_pop * svi_weight
                pop   = float(pop_lookup.get(f, mean_pop))
                svi_v = float(svi_lookup.get(f, 0.5))
                # High-SVI counties get proportionally more demand
                weight = 0.5 + svi_v          # ranges 0.5 – 1.5
                demand_rows[f] = max((pop / 1000.0) * demand_per_pop * weight, 0.001)
            svi_rows[f] = float(svi_lookup.get(f, 0.5))

        demand = pd.Series(demand_rows)
        svi    = pd.Series(svi_rows)
    else:
        return jsonify({"error": "Provide storm_id or fips list"}), 400

    if demand.empty or demand.sum() == 0:
        return jsonify({"error": "No demand data for selection — try selecting different counties"}), 400

    lp_alloc, S  = run_lp(demand, svi, equity_frac, supply_frac, svi_threshold)
    lp_unmet     = (demand - lp_alloc).clip(lower=0)

    # Pop-proportional baseline
    pop_alloc = demand / demand.sum() * S
    pop_unmet = (demand - pop_alloc).clip(lower=0)

    high = svi >= svi_threshold
    low  = ~high

    def metrics(unmet_s):
        return {
            "pct_unmet":        float(unmet_s.sum() / demand.sum()) if demand.sum() > 0 else 0,
            "coverage_overall": coverage(unmet_s, demand),
            "coverage_high_svi":coverage(unmet_s[high], demand[high]),
            "coverage_low_svi": coverage(unmet_s[low],  demand[low]),
            "equity_gap":       coverage(unmet_s[high], demand[high]) - coverage(unmet_s[low], demand[low]),
        }

    # Resource type breakdown (heuristic fractions by disaster type)
    resource_fractions = {
        "hurricane": [0.20, 0.25, 0.18, 0.15, 0.12, 0.10],
        "flood":     [0.22, 0.20, 0.18, 0.15, 0.15, 0.10],
        "tornado":   [0.25, 0.22, 0.18, 0.12, 0.13, 0.10],
        "wildfire":  [0.18, 0.20, 0.15, 0.22, 0.15, 0.10],
    }.get(disaster_type, [0.20, 0.20, 0.20, 0.15, 0.15, 0.10])

    res_names = RESOURCE_TYPES.get(disaster_type, RESOURCE_TYPES["hurricane"])
    resources = [
        {"name": name, "fraction": frac, "units": int(S * frac * 1000)}
        for name, frac in zip(res_names, resource_fractions)
    ]

    # Top counties by LP allocation
    top_fips = lp_alloc.nlargest(15).index
    top_rows = []
    for fips in top_fips:
        row = {
            "fips":        fips,
            "allocation":  float(lp_alloc[fips]),
            "demand":      float(demand[fips]),
            "svi":         float(svi.get(fips, 0)),
            "is_high_svi": bool(svi.get(fips, 0) >= svi_threshold),
        }
        if fips in coords_lookup:
            row["lat"] = coords_lookup[fips]["county_lat"]
            row["lon"] = coords_lookup[fips]["county_lon"]
        top_rows.append(row)

    return jsonify({
        "n_counties":     int(len(demand)),
        "n_high_svi":     int(high.sum()),
        "total_demand":   float(demand.sum()),
        "total_supply":   float(S),
        "lp":             metrics(lp_unmet),
        "pop":            metrics(pop_unmet),
        "resources":      resources,
        "top_counties":   top_rows,
        "improvement": {
            "unmet_reduction":  float((metrics(pop_unmet)["pct_unmet"] - metrics(lp_unmet)["pct_unmet"])
                                      / max(metrics(pop_unmet)["pct_unmet"], 1e-9) * 100),
            "hisvi_gain":       float((metrics(lp_unmet)["coverage_high_svi"]
                                       - metrics(pop_unmet)["coverage_high_svi"]) * 100),
            "equity_gap_gain":  float((metrics(lp_unmet)["equity_gap"]
                                       - metrics(pop_unmet)["equity_gap"]) * 100),
        },
    })


@app.route("/api/overview")
def api_overview():
    lp_df  = (results[results["policy"] == "model_lp_equity"]
              .sort_values(["storm_year", "storm_name"]))
    pop_df = (results[results["policy"] == "population_proportional"]
              .sort_values(["storm_year", "storm_name"]))

    def safe_records(df):
        return df[["storm_name", "storm_year", "pct_unmet",
                   "coverage_high_svi", "equity_gap"]].to_dict("records")

    return jsonify({
        "averages":      avg_policy.to_dict(),
        "per_storm_lp":  safe_records(lp_df),
        "per_storm_pop": safe_records(pop_df),
    })


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
