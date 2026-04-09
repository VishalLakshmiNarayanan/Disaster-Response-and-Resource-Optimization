# DisasterAI — Equity-Based Predictive Disaster Response & Resource Optimization

> **Live demo:** https://disaster-ai-gamma.vercel.app

A capstone project that combines machine learning and equity-constrained linear programming to optimize disaster resource allocation across US counties — prioritizing the most socially vulnerable communities.

---

## What the Website Does

### Landing Page (`/`)
Gives a high-level overview of the project's key results:
- **43.6% reduction** in unmet demand vs. population-proportional baseline
- **97.3% coverage** for high-vulnerability counties
- How the two-stage model + LP optimizer works
- Bootstrap-validated confidence intervals on all headline numbers

### Simulator (`/simulator`)
The main interactive tool. Two modes:

#### Historical Storm Mode
1. Select a **disaster type** (Hurricane, Flood, Tornado, Wildfire)
2. Pick any of **46 historical storms** from the dropdown (2017–2023)
3. The US county map highlights affected counties, colored by predicted demand
4. Adjust the three sliders:
   - **Equity Reserve** — % of supply ring-fenced for high-SVI counties
   - **Supply Availability** — total supply as % of predicted demand
   - **SVI Threshold** — minimum score to classify a county as high-vulnerability
5. Click **Run Optimizer** — the LP solves in < 1 second and shows:
   - Unmet demand, high-SVI coverage, equity gap (vs. pop-proportional baseline)
   - LP vs. population-proportional comparison bars
   - Resource breakdown by type (water, shelter, medical, etc.)
   - Top 15 allocated counties with coverage %

#### Custom Scenario Mode
Build any imaginary disaster scenario from scratch:
1. Switch to **Custom Scenario** tab
2. Click any counties on the US map to select them (click again to deselect)
3. Choose disaster type and adjust sliders
4. Click **Run Optimizer** — works for any county in the US, even ones with no storm history (demand is synthesized from SVI × population)

---

## How the Algorithm Works

### Stage 1 — Demand Prediction (XGBoost Hurdle Model)
A two-stage model handles the 79% zero-inflation in FEMA Public Assistance data:
- **Classifier** (XGBClassifier): predicts P(demand > 0) using storm track, wind speed, distance, SVI features
- **Regressor** (XGBRegressor): predicts log(demand) on non-zero rows only
- Combined: `prediction = P(nonzero) × exp(log_demand)`

### Stage 2 — Equity-Constrained LP (PuLP / CBC)
Given predicted demand per county, solves:

```
minimize  Σ weight_i × unmet_i
subject to:
  unmet_i  ≥  demand_i − allocation_i     (for all counties i)
  Σ allocation_i  ≤  S                    (total supply)
  Σ allocation_i  ≥  equity_frac × S      (for high-SVI counties only)
  allocation_i, unmet_i ≥ 0
```

Where `weight_i = 2.0` for high-SVI counties (SVI ≥ threshold), `1.0` otherwise — forcing the optimizer to prioritize vulnerable communities.

### Key Findings
| Metric | Pop-Proportional | Our LP | Improvement |
|--------|-----------------|--------|-------------|
| Unmet demand | 72.4% | 28.8% | **−43.6pp** |
| High-SVI coverage | 53.7% | 97.3% | **+43.6pp** |
| Equity gap | 56.2pp | 0.0pp | **−56.2pp** |

*All improvements statistically significant (n=10,000 bootstrap, 95% CI)*

---

## Running Locally

```bash
# 1. Clone
git clone https://github.com/VishalLakshmiNarayanan/Disaster-Response-and-Resource-Optimization.git
cd Disaster-Response-and-Resource-Optimization
git checkout shubbh/hurricane

# 2. Install web dependencies
pip install flask pandas numpy pyarrow pulp

# 3. Run
python web/app.py

# 4. Open http://localhost:5000
```

---

## Project Structure

```
web/
├── app.py                  # Flask backend + REST API
├── requirements.txt        # Web-only dependencies
├── data/                   # Bundled parquet files (for deployment)
│   ├── predictions.parquet         # Model predictions per county per storm
│   ├── panel.parquet               # Full county panel with SVI features
│   ├── svi_2020.parquet            # CDC Social Vulnerability Index 2020
│   └── all_storms_comparison.parquet  # LP vs baseline results for 46 storms
├── templates/
│   ├── base.html           # Shared nav, fonts, footer
│   ├── index.html          # Landing page
│   └── simulator.html      # Interactive simulator
└── static/
    ├── css/main.css        # Full design system (glassmorphism dark theme)
    └── js/
        ├── map.js          # D3 v7 US county choropleth + click-to-select
        └── simulator.js    # Simulator interactions + LP result rendering

src/
├── 08_train_model.py       # XGBoost hurdle model training
├── 12_equity_frontier.py   # Pareto frontier sweep
├── 13_stochastic_lp.py     # Stochastic LP + two-stage recourse
├── 14_fairness_analysis.py # Fairness by minority/income/rural
├── 15_spatial_cv.py        # Spatial block cross-validation
├── 16_counterfactual.py    # FEMA vs LP comparison (IAN, IDA, LAURA)
├── 17_bootstrap_ci.py      # 10,000-resample bootstrap CIs
└── 19_dashboard.py         # Streamlit dashboard (alternate UI)
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/counties/svi` | GET | SVI score for all 3,143 US counties (for map coloring) |
| `/api/storms` | GET | List of 46 historical storms |
| `/api/storm/<storm_id>` | GET | County-level data for a specific storm |
| `/api/optimize` | POST | Run LP optimizer, returns allocations + metrics |

**POST `/api/optimize` body:**
```json
{
  "storm_id": "IAN_2022",        // historical storm (OR use fips below)
  "fips": ["48201", "22071"],    // custom county list (alternative to storm_id)
  "equity_frac": 0.40,           // 0.0 – 0.80
  "supply_frac": 0.80,           // 0.40 – 1.20
  "svi_threshold": 0.75,         // 0.50 – 0.90
  "disaster_type": "hurricane"   // hurricane | flood | tornado | wildfire
}
```

---

## Data Sources

| Dataset | Source | Use |
|---------|--------|-----|
| FEMA Public Assistance | OpenFEMA API | Demand proxy (project costs) |
| CDC Social Vulnerability Index 2020 | CDC GRASP | Equity weights, high-SVI classification |
| NOAA HURDAT2 / IBTrACS | NOAA | Storm track, wind speed, landfall distance |
| US Census TIGER | Census Bureau | County geometries, population |
| NFIP Flood Claims | OpenFEMA API | Supplementary demand validation |

---

## Tech Stack

**Backend:** Python · Flask · PuLP (CBC solver) · pandas · pyarrow
**Frontend:** D3.js v7 · TopoJSON · Vanilla JS
**Fonts:** Syne · DM Sans · JetBrains Mono
**Deployment:** Vercel (serverless Python)
