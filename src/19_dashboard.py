# Run with: streamlit run src/19_dashboard.py
# Install:  pip install streamlit plotly

"""
19_dashboard.py
Interactive Streamlit dashboard for real-time exploration of the disaster
resource allocation system.

Purpose: equity-based predictive optimization explorer allowing analysts to
select a storm, adjust LP parameters live, and compare allocation policies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pulp
from pathlib import Path

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False
    st.error(
        "plotly is not installed. Run: pip install plotly\n"
        "Charts will be unavailable until plotly is installed."
    )

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Disaster Response Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent.parent   # project root

# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading predictions...")
def load_predictions() -> pd.DataFrame:
    path = BASE / "data" / "processed" / "predictions.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading SVI data...")
def load_svi() -> pd.DataFrame:
    path = BASE / "data" / "raw" / "svi" / "svi_2020.parquet"
    svi = pd.read_parquet(path)
    if svi.index.name == "fips":
        svi = svi.reset_index()
    return svi


@st.cache_data(show_spinner="Loading storm comparison results...")
def load_results() -> pd.DataFrame:
    path = BASE / "outputs" / "allocations" / "all_storms_comparison.parquet"
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# LP allocation (mirrors 09_optimize.py)
# ---------------------------------------------------------------------------

def lp_allocate(
    demands: pd.Series,
    svi: pd.Series,
    total_supply: float,
    equity_frac: float = 0.40,
    svi_threshold: float = 0.75,
) -> pd.Series:
    """
    Solve LP to minimise weighted unmet demand with equity constraint.
    Returns allocated supply as a Series indexed by fips.
    """
    counties = demands.index.tolist()
    high_svi = svi[svi >= svi_threshold].index.tolist()
    svi_dict = svi.to_dict()

    weights = {
        i: (2.0 if float(svi_dict.get(i, 0)) >= svi_threshold else 1.0)
        for i in counties
    }

    prob  = pulp.LpProblem("alloc", pulp.LpMinimize)
    alloc = pulp.LpVariable.dicts("a", counties, lowBound=0)
    unmet = pulp.LpVariable.dicts("u", counties, lowBound=0)

    prob += pulp.lpSum(weights[i] * unmet[i] for i in counties)

    for i in counties:
        prob += unmet[i] >= demands[i] - alloc[i]

    prob += pulp.lpSum(alloc[i] for i in counties) <= total_supply

    if high_svi:
        prob += (
            pulp.lpSum(alloc[i] for i in high_svi if i in alloc)
            >= equity_frac * total_supply
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    return pd.Series(
        {i: max(0.0, pulp.value(alloc[i]) or 0.0) for i in counties}
    )


# ---------------------------------------------------------------------------
# Allocation baselines
# ---------------------------------------------------------------------------

def pop_proportional(pop: pd.Series, total_supply: float) -> pd.Series:
    safe = pop.clip(lower=0).fillna(0)
    if safe.sum() == 0:
        return pd.Series(total_supply / max(len(pop), 1), index=pop.index)
    return safe / safe.sum() * total_supply


def svi_weighted_alloc(svi: pd.Series, total_supply: float) -> pd.Series:
    safe = svi.clip(lower=0.01).fillna(0.01)
    return safe / safe.sum() * total_supply


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(
    demands: pd.Series,
    allocated: pd.Series,
    svi: pd.Series,
    svi_threshold: float = 0.75,
) -> dict:
    """
    Returns dict with pct_unmet, coverage_overall, coverage_high_svi,
    coverage_low_svi, equity_gap.
    """
    unmet_mask  = demands > 0
    covered     = (allocated[unmet_mask] >= demands[unmet_mask]).sum()
    total_need  = unmet_mask.sum()
    pct_unmet   = 1.0 - (covered / total_need) if total_need > 0 else 0.0

    high_mask   = (svi >= svi_threshold) & unmet_mask
    low_mask    = (svi < svi_threshold) & unmet_mask

    cov_high = (
        (allocated[high_mask] >= demands[high_mask]).sum() / high_mask.sum()
        if high_mask.sum() > 0 else float("nan")
    )
    cov_low = (
        (allocated[low_mask] >= demands[low_mask]).sum() / low_mask.sum()
        if low_mask.sum() > 0 else float("nan")
    )
    coverage_overall = (covered / total_need) if total_need > 0 else 0.0
    equity_gap       = (cov_low - cov_high) if not (
        np.isnan(cov_high) or np.isnan(cov_low)
    ) else float("nan")

    return {
        "pct_unmet":        round(pct_unmet, 4),
        "coverage_overall": round(coverage_overall, 4),
        "coverage_high_svi": round(cov_high, 4) if not np.isnan(cov_high) else None,
        "coverage_low_svi":  round(cov_low, 4) if not np.isnan(cov_low) else None,
        "equity_gap":        round(equity_gap, 4) if not np.isnan(equity_gap) else None,
    }


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Controls")

# load data (will cache after first run)
try:
    predictions = load_predictions()
    data_ok = True
except Exception as e:
    st.sidebar.error(f"Could not load predictions.parquet:\n{e}")
    data_ok = False
    predictions = pd.DataFrame()

try:
    svi_df = load_svi()
    svi_ok = True
except Exception as e:
    st.sidebar.warning(f"Could not load SVI data:\n{e}")
    svi_ok = False
    svi_df = pd.DataFrame()

try:
    results_df = load_results()
    results_ok = True
except Exception as e:
    results_ok = False
    results_df = pd.DataFrame()

# --- Storm selector ---
if data_ok and not predictions.empty:
    storm_opts = (
        predictions[["storm_name", "storm_year"]]
        .drop_duplicates()
        .sort_values("storm_year", ascending=False)
    )
    storm_labels = [
        f"{row['storm_name']} ({int(row['storm_year'])})"
        for _, row in storm_opts.iterrows()
    ]
    selected_label = st.sidebar.selectbox("Storm", storm_labels)
    sel_name = selected_label.split(" (")[0]
    sel_year = int(selected_label.split("(")[1].rstrip(")"))
else:
    selected_label = "N/A"
    sel_name = None
    sel_year = None

st.sidebar.markdown("---")
st.sidebar.subheader("LP Parameters")

equity_frac    = st.sidebar.slider(
    "Equity Fraction",
    min_value=0.0, max_value=0.80, value=0.40, step=0.05,
    help="Fraction of supply reserved for high-SVI counties."
)
supply_frac    = st.sidebar.slider(
    "Supply Fraction",
    min_value=0.60, max_value=1.00, value=0.80, step=0.05,
    help="Total supply as a fraction of predicted total demand."
)
svi_threshold  = st.sidebar.slider(
    "SVI Threshold",
    min_value=0.50, max_value=0.90, value=0.75, step=0.05,
    help="SVI score above which a county is classified as high vulnerability."
)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Disaster Response Resource Allocation Dashboard")
st.markdown(
    "**Equity-Based Predictive Optimization System** -- "
    "Select a storm in the sidebar and adjust LP parameters to explore "
    "how allocation policies affect coverage and equity."
)

if not data_ok:
    st.error(
        "predictions.parquet could not be loaded. "
        "Run 08_train_model.py first to generate this file."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Filter to selected storm
# ---------------------------------------------------------------------------
storm_df = predictions[
    (predictions["storm_name"] == sel_name) &
    (predictions["storm_year"] == sel_year)
].copy()

if storm_df.empty:
    st.warning(f"No data found for storm: {selected_label}")
    st.stop()

# Merge SVI if available
if svi_ok and not svi_df.empty and "svi_overall" not in storm_df.columns:
    svi_merge = svi_df[["fips", "svi_overall"]].drop_duplicates("fips")
    storm_df = storm_df.merge(svi_merge, on="fips", how="left")

# Fill missing SVI with median
if "svi_overall" not in storm_df.columns:
    storm_df["svi_overall"] = 0.5
storm_df["svi_overall"] = storm_df["svi_overall"].fillna(
    storm_df["svi_overall"].median() if storm_df["svi_overall"].notna().any() else 0.5
)

storm_df = storm_df.set_index("fips")
demands  = storm_df["pred_demand"].clip(lower=0).fillna(0)
svi_ser  = storm_df["svi_overall"].fillna(0.5)

total_supply = float(demands.sum()) * supply_frac

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["Allocation Map", "Policy Comparison", "System Overview"]
)

# ===========================================================================
# TAB 1 -- Allocation Map
# ===========================================================================
with tab1:
    st.subheader(f"County Map -- {sel_name} ({sel_year})")

    has_geo = (
        "county_lat" in storm_df.columns and
        "county_lon" in storm_df.columns
    )

    if PLOTLY_OK:
        if has_geo:
            map_df = storm_df.reset_index().copy()
            map_df["pred_demand_scaled"] = (
                map_df["pred_demand"].clip(lower=0) * 100
            ).clip(lower=5)
            map_df["fema_declared_label"] = map_df.get(
                "fema_declared", pd.Series(0, index=map_df.index)
            ).map({0: "No", 1: "Yes"}).fillna("Unknown")

            fig_map = px.scatter(
                map_df,
                x="county_lon",
                y="county_lat",
                color="svi_overall",
                size="pred_demand_scaled",
                hover_data={
                    col: True
                    for col in [
                        "fips", "storm_name", "svi_overall",
                        "pred_demand", "demand_proxy", "fema_declared_label"
                    ]
                    if col in map_df.columns
                },
                color_continuous_scale="YlOrRd",
                title=f"County-Level Allocation Map -- {sel_name} ({sel_year})",
                labels={
                    "county_lon":  "Longitude",
                    "county_lat":  "Latitude",
                    "svi_overall": "SVI Score",
                },
                size_max=30,
            )
            fig_map.update_layout(
                height=520,
                coloraxis_colorbar=dict(title="SVI Score"),
                plot_bgcolor="#f8f8f8",
            )
            st.plotly_chart(fig_map, use_container_width=True)

        else:
            # Fallback: bar chart of top-20 counties by pred_demand
            st.info(
                "Latitude/longitude columns (county_lat, county_lon) not found "
                "in predictions.parquet. Showing top-20 counties by predicted "
                "demand instead."
            )
            top20 = (
                storm_df["pred_demand"]
                .nlargest(20)
                .reset_index()
            )
            top20.columns = ["fips", "pred_demand"]
            top20["svi"] = svi_ser.reindex(top20["fips"]).values

            fig_bar = px.bar(
                top20,
                x="fips",
                y="pred_demand",
                color="svi",
                color_continuous_scale="YlOrRd",
                title=f"Top 20 Counties by Predicted Demand -- {sel_name} ({sel_year})",
                labels={"fips": "County FIPS", "pred_demand": "Predicted Demand",
                        "svi": "SVI Score"},
            )
            fig_bar.update_layout(height=420, xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.warning("plotly is required for map/chart visualizations.")

    # Quick stats below map
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Counties", f"{len(storm_df):,}")
    col_b.metric("Total Predicted Demand", f"{demands.sum():,.1f}")
    col_c.metric("Total Supply (LP)", f"{total_supply:,.1f}")
    col_d.metric(
        "High-SVI Counties",
        f"{(svi_ser >= svi_threshold).sum():,}"
    )

    # Demand distribution
    if PLOTLY_OK:
        st.subheader("Predicted Demand Distribution")
        col_left, col_right = st.columns(2)

        with col_left:
            fig_hist = px.histogram(
                storm_df.reset_index(),
                x="pred_demand",
                nbins=40,
                title="Predicted Demand Distribution",
                labels={"pred_demand": "Predicted Demand"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig_hist.update_layout(height=320)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_right:
            scatter_df = storm_df.reset_index().copy()
            if "pred_q10" in scatter_df.columns:
                fig_ci = go.Figure()
                sorted_df = scatter_df.sort_values("pred_demand")
                fig_ci.add_trace(go.Scatter(
                    x=list(range(len(sorted_df))),
                    y=sorted_df["pred_q90"],
                    fill=None,
                    mode="lines",
                    line_color="rgba(31,119,180,0.3)",
                    name="Q90",
                ))
                fig_ci.add_trace(go.Scatter(
                    x=list(range(len(sorted_df))),
                    y=sorted_df["pred_q10"],
                    fill="tonexty",
                    mode="lines",
                    line_color="rgba(31,119,180,0.3)",
                    fillcolor="rgba(31,119,180,0.15)",
                    name="Q10",
                ))
                fig_ci.add_trace(go.Scatter(
                    x=list(range(len(sorted_df))),
                    y=sorted_df["pred_demand"],
                    mode="lines",
                    line=dict(color="#1f77b4", width=2),
                    name="Median",
                ))
                fig_ci.update_layout(
                    title="Prediction Intervals (sorted by demand)",
                    xaxis_title="County (sorted)",
                    yaxis_title="Demand",
                    height=320,
                    showlegend=True,
                )
                st.plotly_chart(fig_ci, use_container_width=True)
            else:
                fig_svi = px.scatter(
                    scatter_df,
                    x="svi_overall",
                    y="pred_demand",
                    title="SVI vs Predicted Demand",
                    labels={
                        "svi_overall": "SVI Score",
                        "pred_demand": "Predicted Demand",
                    },
                    color_discrete_sequence=["#d62728"],
                    opacity=0.5,
                )
                fig_svi.update_layout(height=320)
                st.plotly_chart(fig_svi, use_container_width=True)


# ===========================================================================
# TAB 2 -- Policy Comparison
# ===========================================================================
with tab2:
    st.subheader(f"Policy Comparison -- {sel_name} ({sel_year})")
    st.markdown(
        f"Comparing three allocation policies for **{sel_name} ({sel_year})** "
        f"with supply fraction **{supply_frac:.0%}**, "
        f"equity fraction **{equity_frac:.0%}**, "
        f"SVI threshold **{svi_threshold:.2f}**."
    )

    # Derive population proxy from log_total_pop if available
    if "log_total_pop" in storm_df.columns:
        pop_ser = np.exp(storm_df["log_total_pop"].fillna(0))
    else:
        pop_ser = pd.Series(1.0, index=storm_df.index)

    # Run all three policies
    with st.spinner("Optimizing allocation..."):
        alloc_pop  = pop_proportional(pop_ser, total_supply)
        alloc_svi  = svi_weighted_alloc(svi_ser, total_supply)
        alloc_lp   = lp_allocate(
            demands,
            svi_ser,
            total_supply,
            equity_frac=equity_frac,
            svi_threshold=svi_threshold,
        )

    metrics_pop = compute_metrics(demands, alloc_pop, svi_ser, svi_threshold)
    metrics_svi = compute_metrics(demands, alloc_svi, svi_ser, svi_threshold)
    metrics_lp  = compute_metrics(demands, alloc_lp,  svi_ser, svi_threshold)

    policy_labels = ["Pop-Proportional", "SVI-Weighted", f"LP+Equity (ef={equity_frac:.2f})"]
    all_metrics   = [metrics_pop, metrics_svi, metrics_lp]

    # --- Metric cards ---
    st.markdown("#### Coverage Metrics (per policy)")
    header_cols = st.columns([2, 2, 2])
    header_cols[0].markdown(f"**{policy_labels[0]}**")
    header_cols[1].markdown(f"**{policy_labels[1]}**")
    header_cols[2].markdown(f"**{policy_labels[2]}**")

    def fmt_pct(v):
        if v is None:
            return "N/A"
        return f"{v * 100:.1f}%"

    def delta_str(val, base):
        if val is None or base is None:
            return None
        d = (val - base) * 100
        return f"{d:+.1f}pp vs Pop-Prop"

    metric_rows = [
        ("Pct Unmet",         "pct_unmet",         False),
        ("Coverage Overall",  "coverage_overall",  True),
        ("Coverage High-SVI", "coverage_high_svi", True),
        ("Coverage Low-SVI",  "coverage_low_svi",  True),
        ("Equity Gap",        "equity_gap",         False),
    ]

    for label, key, higher_is_better in metric_rows:
        row_cols = st.columns([1, 2, 2, 2])
        row_cols[0].markdown(f"*{label}*")
        for col_idx, (mc, pl) in enumerate(zip(all_metrics, policy_labels)):
            val  = mc.get(key)
            base = metrics_pop.get(key)
            if col_idx == 0:
                row_cols[col_idx + 1].metric(label="", value=fmt_pct(val))
            else:
                d = delta_str(val, base)
                row_cols[col_idx + 1].metric(label="", value=fmt_pct(val), delta=d)

    st.markdown("---")

    # --- Grouped bar chart ---
    if PLOTLY_OK:
        st.markdown("#### Policy Comparison Chart")

        chart_metrics = ["pct_unmet", "coverage_overall", "coverage_high_svi", "equity_gap"]
        chart_labels  = ["Pct Unmet", "Coverage Overall", "Coverage High-SVI", "Equity Gap"]

        bar_data = []
        for policy, mc in zip(policy_labels, all_metrics):
            for m_key, m_label in zip(chart_metrics, chart_labels):
                v = mc.get(m_key)
                bar_data.append({
                    "Policy": policy,
                    "Metric": m_label,
                    "Value":  (v * 100) if v is not None else 0.0,
                })
        bar_df = pd.DataFrame(bar_data)

        fig_bar = px.bar(
            bar_df,
            x="Metric",
            y="Value",
            color="Policy",
            barmode="group",
            title="Policy Comparison on Key Metrics",
            labels={"Value": "Value (%)", "Metric": "Metric"},
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
        )
        fig_bar.update_layout(height=420, legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Allocation distribution
        st.markdown("#### Allocation Distribution by SVI Quartile")
        alloc_compare = pd.DataFrame({
            "fips":      storm_df.index,
            "svi":       svi_ser.values,
            "demand":    demands.values,
            "Pop-Prop":  alloc_pop.reindex(storm_df.index).fillna(0).values,
            "SVI-Wtd":   alloc_svi.reindex(storm_df.index).fillna(0).values,
            "LP+Equity": alloc_lp.reindex(storm_df.index).fillna(0).values,
        })
        alloc_compare["svi_quartile"] = pd.qcut(
            alloc_compare["svi"], q=4,
            labels=["Q1 (Low SVI)", "Q2", "Q3", "Q4 (High SVI)"],
            duplicates="drop",
        )
        quartile_agg = (
            alloc_compare.groupby("svi_quartile", observed=True)[
                ["Pop-Prop", "SVI-Wtd", "LP+Equity"]
            ].sum()
            .reset_index()
        )
        fig_q = px.bar(
            quartile_agg.melt(id_vars="svi_quartile", var_name="Policy", value_name="Allocation"),
            x="svi_quartile",
            y="Allocation",
            color="Policy",
            barmode="group",
            title="Total Allocation by SVI Quartile",
            labels={"svi_quartile": "SVI Quartile", "Allocation": "Total Allocated"},
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
        )
        fig_q.update_layout(height=360)
        st.plotly_chart(fig_q, use_container_width=True)

    else:
        st.warning("plotly is required for comparison charts.")

    # --- Raw allocation table (expandable) ---
    with st.expander("Raw Allocation Data (first 50 rows)"):
        display_cols = ["svi_overall", "pred_demand"]
        if "demand_proxy" in storm_df.columns:
            display_cols.append("demand_proxy")
        disp = storm_df[display_cols].copy()
        disp["alloc_pop_prop"] = alloc_pop.reindex(disp.index).round(3)
        disp["alloc_svi_wtd"]  = alloc_svi.reindex(disp.index).round(3)
        disp["alloc_lp_equity"] = alloc_lp.reindex(disp.index).round(3)
        st.dataframe(disp.head(50), use_container_width=True)


# ===========================================================================
# TAB 3 -- System Overview
# ===========================================================================
with tab3:
    st.subheader("System Overview -- All Storms")

    if not results_ok or results_df.empty:
        st.warning(
            "all_storms_comparison.parquet could not be loaded. "
            "Run src/09_optimize.py first to generate it."
        )
    else:
        # --- Overall stats table ---
        st.markdown("#### Average Metrics Across All Storms (by Policy)")

        num_cols = ["pct_unmet", "coverage_overall", "coverage_high_svi",
                    "coverage_low_svi", "equity_gap"]
        available_num = [c for c in num_cols if c in results_df.columns]

        if "policy" in results_df.columns and available_num:
            avg_table = (
                results_df.groupby("policy")[available_num]
                .mean()
                .round(4)
                .reset_index()
            )
            # Convert to percentages for readability
            for c in available_num:
                avg_table[c] = (avg_table[c] * 100).round(2)
            col_rename = {c: c.replace("_", " ").title() + " (%)" for c in available_num}
            avg_table = avg_table.rename(columns=col_rename | {"policy": "Policy"})
            st.dataframe(avg_table, use_container_width=True, hide_index=True)
        else:
            st.dataframe(results_df.describe(), use_container_width=True)

        # --- Big metric cards ---
        st.markdown("#### Key System Metrics")
        lp_rows = results_df[results_df.get("policy", pd.Series()).str.contains(
            "lp|equity|LP|Equity", na=False
        )] if "policy" in results_df.columns else results_df

        kpi_cols = st.columns(4)
        n_storms = results_df["storm_id"].nunique() if "storm_id" in results_df.columns else len(results_df)
        kpi_cols[0].metric("Storms Analyzed", f"{n_storms}")

        if "pct_unmet" in results_df.columns and "policy" in results_df.columns:
            avg_unmet = lp_rows["pct_unmet"].mean() * 100 if not lp_rows.empty else float("nan")
            kpi_cols[1].metric("Avg Pct Unmet (LP)", f"{avg_unmet:.1f}%")

        if "coverage_high_svi" in results_df.columns and not lp_rows.empty:
            avg_cov_hi = lp_rows["coverage_high_svi"].mean() * 100
            kpi_cols[2].metric("Avg High-SVI Coverage (LP)", f"{avg_cov_hi:.1f}%")

        if "equity_gap" in results_df.columns and not lp_rows.empty:
            avg_gap = lp_rows["equity_gap"].mean() * 100
            kpi_cols[3].metric("Avg Equity Gap (LP)", f"{avg_gap:.1f}pp")

        # --- Equity tradeoff scatter ---
        if PLOTLY_OK and "coverage_low_svi" in results_df.columns and \
                "coverage_high_svi" in results_df.columns:
            st.markdown("#### Equity Tradeoff: Low-SVI vs High-SVI Coverage")
            st.markdown(
                "Each point is a (storm, policy) pair. Points above the "
                "diagonal line favour high-SVI counties; points below favour "
                "low-SVI counties."
            )

            scatter_res = results_df.copy()
            hover_cols  = [c for c in ["storm_name", "storm_year", "policy",
                                        "pct_unmet", "equity_gap"]
                           if c in scatter_res.columns]
            color_col   = "policy" if "policy" in scatter_res.columns else None

            fig_eq = px.scatter(
                scatter_res,
                x="coverage_low_svi",
                y="coverage_high_svi",
                color=color_col,
                hover_data={c: True for c in hover_cols},
                title="Coverage Equity Tradeoff Across All Storms",
                labels={
                    "coverage_low_svi":  "Coverage -- Low SVI Counties",
                    "coverage_high_svi": "Coverage -- High SVI Counties",
                },
                opacity=0.75,
            )

            # Add diagonal reference line
            lo = min(
                scatter_res["coverage_low_svi"].min(),
                scatter_res["coverage_high_svi"].min(),
            )
            hi = max(
                scatter_res["coverage_low_svi"].max(),
                scatter_res["coverage_high_svi"].max(),
            )
            fig_eq.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi],
                mode="lines",
                line=dict(dash="dash", color="gray", width=1),
                name="Equal coverage",
                showlegend=True,
            ))
            fig_eq.update_layout(height=500)
            st.plotly_chart(fig_eq, use_container_width=True)

        # --- Policy trends over storms ---
        if PLOTLY_OK and "storm_year" in results_df.columns and \
                "pct_unmet" in results_df.columns:
            st.markdown("#### Pct Unmet by Storm Year")
            trend_df = results_df.sort_values("storm_year")
            color_col = "policy" if "policy" in trend_df.columns else None

            name_col = "storm_name" if "storm_name" in trend_df.columns else "storm_id"
            hover_extra = {name_col: True, "pct_unmet": True}
            if color_col:
                hover_extra["policy"] = True

            fig_trend = px.scatter(
                trend_df,
                x="storm_year",
                y="pct_unmet",
                color=color_col,
                hover_data=hover_extra,
                title="Percent Unmet Demand by Storm Year",
                labels={"storm_year": "Storm Year", "pct_unmet": "Pct Unmet"},
                opacity=0.7,
            )
            fig_trend.update_layout(height=380)
            st.plotly_chart(fig_trend, use_container_width=True)

        # --- Full results table (expandable) ---
        with st.expander("Full Storm Comparison Table"):
            st.dataframe(results_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Disaster Response Resource Allocation Dashboard | "
    "Equity-Based Predictive Optimization System | "
    "Data: FEMA, CDC SVI, HURDAT2"
)
