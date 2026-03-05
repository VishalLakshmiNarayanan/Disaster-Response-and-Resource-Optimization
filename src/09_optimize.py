"""
09_optimize.py
Runs equity-constrained LP allocation and compares against baselines
for every storm in the test set.
Output: outputs/allocations/all_storms_comparison.parquet
"""
import pandas as pd
import numpy as np
import pulp
from pathlib import Path
from tqdm import tqdm

PROC    = Path("data/processed")
OUT_DIR = Path("outputs/allocations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPLY_FRACTION = 0.80   # total supply = 80% of predicted total demand
EQUITY_FRAC     = 0.40   # fraction of supply reserved for high-SVI counties
SVI_THRESHOLD   = 0.75   # top quartile = "high vulnerability"


def lp_allocate(
    demands:      pd.Series,   # index=fips
    svi:          pd.Series,   # index=fips
    total_supply: float,
    equity_frac:  float = EQUITY_FRAC,
    svi_threshold:float = SVI_THRESHOLD,
) -> pd.DataFrame:
    """Solve LP to minimise weighted unmet demand with equity constraint."""
    counties = demands.index.tolist()
    high_svi = svi[svi >= svi_threshold].index.tolist()

    # Convert svi to plain dict to avoid pandas ambiguity on lookup
    svi_dict = svi.to_dict()

    # Equity weight: high-SVI counties get weight 2, others weight 1
    weights = {i: (2.0 if float(svi_dict.get(i, 0)) >= svi_threshold else 1.0)
               for i in counties}

    prob = pulp.LpProblem("allocation", pulp.LpMinimize)

    alloc = pulp.LpVariable.dicts("a", counties, lowBound=0)
    unmet  = pulp.LpVariable.dicts("u", counties, lowBound=0)

    # Objective
    prob += pulp.lpSum(weights[i] * unmet[i] for i in counties)

    # Constraints
    for i in counties:
        prob += unmet[i] >= demands[i] - alloc[i]

    prob += pulp.lpSum(alloc[i] for i in counties) <= total_supply

    if high_svi:
        prob += (pulp.lpSum(alloc[i] for i in high_svi if i in alloc)
                 >= equity_frac * total_supply)

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    return pd.DataFrame({
        "fips":      counties,
        "allocated": [max(0, pulp.value(alloc[i]) or 0) for i in counties],
        "unmet":     [max(0, pulp.value(unmet[i]) or 0)  for i in counties],
    }).set_index("fips")


def pop_proportional(pop: pd.Series, total_supply: float) -> pd.Series:
    """Allocate in proportion to population."""
    safe = pop.clip(lower=0).fillna(0)
    if safe.sum() == 0:
        return pd.Series(total_supply / len(pop), index=pop.index)
    return (safe / safe.sum() * total_supply)


def svi_weighted(svi: pd.Series, total_supply: float) -> pd.Series:
    """Allocate in proportion to SVI score."""
    safe = svi.clip(lower=0.01).fillna(0.01)
    return (safe / safe.sum() * total_supply)


if __name__ == "__main__":
    preds   = pd.read_parquet(PROC / "predictions.parquet")
    svi_df  = pd.read_parquet("data/raw/svi/svi_2020.parquet").set_index("fips")
    acs_df  = pd.read_parquet("data/raw/acs/acs_county.parquet")

    # Latest ACS per county for population-proportional baseline
    pop_ser = (
        acs_df.sort_values("year")
              .drop_duplicates("fips", keep="last")
              .set_index("fips")["total_pop"]
    )

    storm_ids = preds["storm_id"].unique()
    print(f"Running allocation for {len(storm_ids)} test storms...")

    all_rows = []

    for sid in tqdm(storm_ids, desc="Storms"):
        subset = preds[preds["storm_id"] == sid].copy()
        subset = subset.set_index("fips")

        demand   = subset["pred_demand"].clip(lower=0)
        svi_vals = svi_df["svi_overall"].reindex(demand.index).fillna(0.5)
        pop_vals = pop_ser.reindex(demand.index).fillna(1.0)
        S        = demand.sum() * SUPPLY_FRACTION

        if S <= 0 or len(demand) == 0:
            continue

        # Policy 1: population-proportional
        alloc_pop = pop_proportional(pop_vals, S)
        unmet_pop = (demand - alloc_pop).clip(lower=0)

        # Policy 2: SVI-weighted
        alloc_svi_w = svi_weighted(svi_vals, S)
        unmet_svi_w = (demand - alloc_svi_w).clip(lower=0)

        # Policy 3: LP with equity constraint
        lp_result    = lp_allocate(demand, svi_vals, S)
        alloc_lp     = lp_result["allocated"]
        unmet_lp     = lp_result["unmet"]

        # Robustness: re-run LP under +20% / -20% demand shock
        demand_high  = demand * 1.20
        demand_low   = demand * 0.80
        unmet_lp_high = lp_allocate(demand_high, svi_vals, S)["unmet"]
        unmet_lp_low  = lp_allocate(demand_low,  svi_vals, S)["unmet"]

        storm_name = subset["storm_name"].iloc[0]
        storm_year = subset["storm_year"].iloc[0]
        high_mask  = svi_vals >= SVI_THRESHOLD
        low_mask   = svi_vals <  SVI_THRESHOLD

        def coverage(unmet_s, demand_s):
            total_d = demand_s.sum()
            return 1 - unmet_s.sum() / total_d if total_d > 0 else 1.0

        for policy, alloc_s, unmet_s in [
            ("population_proportional", alloc_pop,   unmet_pop),
            ("svi_weighted",            alloc_svi_w, unmet_svi_w),
            ("model_lp_equity",         alloc_lp,    unmet_lp),
        ]:
            all_rows.append({
                "storm_id":         sid,
                "storm_name":       storm_name,
                "storm_year":       storm_year,
                "policy":           policy,
                "total_demand":     demand.sum(),
                "total_supply":     S,
                "total_unmet":      unmet_s.sum(),
                "pct_unmet":        unmet_s.sum() / demand.sum() if demand.sum() > 0 else 0,
                "coverage_overall": coverage(unmet_s, demand),
                "coverage_high_svi":coverage(unmet_s[high_mask], demand[high_mask]),
                "coverage_low_svi": coverage(unmet_s[low_mask],  demand[low_mask]),
                "equity_gap":       (coverage(unmet_s[high_mask], demand[high_mask]) -
                                     coverage(unmet_s[low_mask],  demand[low_mask])),
                # Robustness (only for LP policy)
                "unmet_shock_plus20":  (unmet_lp_high.sum() if policy == "model_lp_equity"
                                        else np.nan),
                "unmet_shock_minus20": (unmet_lp_low.sum()  if policy == "model_lp_equity"
                                        else np.nan),
            })

    results = pd.DataFrame(all_rows)
    results.to_parquet(OUT_DIR / "all_storms_comparison.parquet", index=False)
    print(f"\n[OK] Saved results for {len(storm_ids)} storms")

    # - Summary table ----------------------------─
    summary = (
        results.groupby("policy")[[
            "pct_unmet", "coverage_high_svi", "coverage_low_svi", "equity_gap"
        ]].mean().round(3)
    )
    print("\n- Policy Comparison (averages across all test storms) -------")
    print(summary.to_string())
