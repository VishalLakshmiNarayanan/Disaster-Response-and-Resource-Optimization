/* ============================================================
   simulator.js  —  Simulator page interactions
   ============================================================ */

// ── SIM namespace (used by map.js callbacks) ──────────────────
window.SIM = { onSelectionChange };

// ── State ─────────────────────────────────────────────────────
let appMode              = "storm";   // "storm" | "custom"
let selectedDisasterType = "hurricane";
let currentStormId       = null;

// ── Mode tabs ─────────────────────────────────────────────────
document.querySelectorAll(".mode-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode-tab").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    appMode = btn.dataset.mode;
    switchMode(appMode);
  });
});

function switchMode(m) {
  document.getElementById("stormModePanel").style.display  = m === "storm"  ? "" : "none";
  document.getElementById("customModePanel").style.display = m === "custom" ? "" : "none";

  if (m === "custom") {
    MAP.setSelectMode(true);
    MAP.resetStorm();
    currentStormId = null;
    document.getElementById("stormInfo").style.display = "none";
    document.getElementById("runBtn").disabled = true;
    updateSelectionStatus(0);
  } else {
    MAP.setSelectMode(false);
    document.getElementById("runBtn").disabled = !currentStormId;
  }
}

// ── Selection change callback (from map.js) ───────────────────
function onSelectionChange(count) {
  updateSelectionStatus(count);
  document.getElementById("runBtn").disabled = (count === 0);
}

function updateSelectionStatus(count) {
  document.getElementById("selectionCount").textContent = count;
  const statusEl = document.getElementById("selectionStatus");
  if (count > 0) {
    statusEl.classList.add("has-selection");
  } else {
    statusEl.classList.remove("has-selection");
  }
}

// ── Disaster type cards (both grids) ─────────────────────────
["disasterGrid", "disasterGridCustom"].forEach(gridId => {
  document.getElementById(gridId)?.addEventListener("click", e => {
    const card = e.target.closest(".disaster-card");
    if (!card) return;
    // sync both grids
    document.querySelectorAll(".disaster-card").forEach(c => c.classList.remove("active"));
    document.querySelectorAll(`[data-type="${card.dataset.type}"]`).forEach(c => c.classList.add("active"));
    selectedDisasterType = card.dataset.type;
  });
});

// ── Sliders ───────────────────────────────────────────────────
document.getElementById("equitySlider").addEventListener("input", e => {
  document.getElementById("equityVal").textContent = e.target.value + "%";
});
document.getElementById("supplySlider").addEventListener("input", e => {
  document.getElementById("supplyVal").textContent = e.target.value + "%";
});
document.getElementById("sviSlider").addEventListener("input", e => {
  document.getElementById("sviVal").textContent = parseFloat(e.target.value).toFixed(2);
});

// ── Storm selector ────────────────────────────────────────────
document.getElementById("stormSelect").addEventListener("change", async e => {
  const stormId = e.target.value;
  currentStormId = stormId || null;
  document.getElementById("runBtn").disabled = !stormId;

  if (!stormId) {
    MAP.resetStorm();
    document.getElementById("stormInfo").style.display = "none";
    return;
  }

  MAP.showOverlay("Loading storm data…");
  try {
    const res = await fetch(`/api/storm/${encodeURIComponent(stormId)}`);
    if (!res.ok) throw new Error("Storm not found");
    const data = await res.json();

    MAP.highlightStorm(data.counties);

    document.getElementById("stormInfoText").textContent =
      `${data.storm_name} (${data.storm_year}) · ${data.n_counties} affected counties`;
    document.getElementById("stormInfo").style.display = "block";

    // Auto-switch map to demand view
    document.querySelectorAll(".map-toggle").forEach(b => {
      b.classList.toggle("active", b.dataset.mode === "demand");
    });
    MAP.setMode("demand");
  } catch (err) {
    showToast("Error: " + err.message, "error");
  } finally {
    MAP.hideOverlay();
  }
});

// ── Run optimizer ─────────────────────────────────────────────
document.getElementById("runBtn").addEventListener("click", runOptimizer);

async function runOptimizer() {
  const btn = document.getElementById("runBtn");
  btn.classList.add("loading");
  btn.disabled = true;
  MAP.showOverlay("Solving LP…");

  // Build request body
  const body = {
    equity_frac:   parseFloat(document.getElementById("equitySlider").value) / 100,
    supply_frac:   parseFloat(document.getElementById("supplySlider").value) / 100,
    svi_threshold: parseFloat(document.getElementById("sviSlider").value),
    disaster_type: selectedDisasterType,
  };

  if (appMode === "storm") {
    if (!currentStormId) { btn.classList.remove("loading"); btn.disabled = false; MAP.hideOverlay(); return; }
    body.storm_id = currentStormId;
  } else {
    const fips = MAP.getSelectedFips();
    if (!fips.length) { btn.classList.remove("loading"); btn.disabled = false; MAP.hideOverlay(); return; }
    body.fips = fips;
  }

  try {
    const res = await fetch("/api/optimize", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Optimization failed");
    }
    const data = await res.json();

    renderResults(data);
    MAP.applyAllocations(data.top_counties);

    document.querySelectorAll(".map-toggle").forEach(b => {
      b.classList.toggle("active", b.dataset.mode === "allocation");
    });
    MAP.setMode("allocation");

    showToast("Optimization complete!", "success");
  } catch (err) {
    showToast("Error: " + err.message, "error");
  } finally {
    btn.classList.remove("loading");
    btn.disabled = false;
    MAP.hideOverlay();
  }
}

// ── Render results ────────────────────────────────────────────
function renderResults(d) {
  document.getElementById("resultsPanel").classList.add("visible");
  document.getElementById("emptyState").style.display   = "none";
  document.getElementById("resultsContent").style.display = "block";

  const lp  = d.lp;
  const pop = d.pop;

  // KPI cards
  setKpi("kpiUnmet",
    pct(lp.pct_unmet),
    delta(-(lp.pct_unmet - pop.pct_unmet) * 100, "pp", false)
  );
  setKpi("kpiHiSvi",
    pct(lp.coverage_high_svi),
    delta((lp.coverage_high_svi - pop.coverage_high_svi) * 100, "pp")
  );
  setKpi("kpiEquity",
    ((lp.equity_gap) * 100).toFixed(1) + "pp",
    delta((lp.equity_gap - pop.equity_gap) * 100, "pp")
  );

  // Comparison bars (delayed for animation)
  setTimeout(() => {
    animBar("cbarLpOverall",  lp.coverage_overall  * 100, "cbarLpOverallPct");
    animBar("cbarPopOverall", pop.coverage_overall * 100, "cbarPopOverallPct");
    animBar("cbarLpHiSvi",   lp.coverage_high_svi * 100, "cbarLpHiSviPct");
    animBar("cbarPopHiSvi",  pop.coverage_high_svi * 100, "cbarPopHiSviPct");
  }, 150);

  // Resources
  const rl = document.getElementById("resourceList");
  rl.innerHTML = "";
  document.getElementById("resourceTitle").textContent = cap(selectedDisasterType) + " Resource Breakdown";
  const maxFrac = Math.max(...d.resources.map(r => r.fraction));
  d.resources.forEach((r, i) => {
    const div = document.createElement("div");
    div.className = "resource-item";
    div.innerHTML = `
      <div class="resource-item__name">${r.name}</div>
      <div class="resource-item__bar">
        <div class="resource-item__fill" data-w="${(r.fraction/maxFrac*100).toFixed(1)}%"></div>
      </div>
      <div class="resource-item__val">${fmtUnits(r.units)}</div>
    `;
    rl.appendChild(div);
  });
  // Stagger bar animation
  rl.querySelectorAll(".resource-item__fill").forEach((el, i) => {
    setTimeout(() => { el.style.width = el.dataset.w; }, 100 + i * 60);
  });

  // Counties subtitle
  document.getElementById("countiesSubtitle").textContent =
    `${d.n_counties} counties · ${d.n_high_svi} high-SVI`;

  // Table
  const tbody = document.getElementById("countiesTbody");
  tbody.innerHTML = "";
  const maxAlloc = Math.max(...d.top_counties.map(c => c.allocation), 1);
  d.top_counties.forEach(c => {
    const cov = c.demand > 0 ? Math.min(c.allocation / c.demand * 100, 100) : 100;
    const badge = c.is_high_svi
      ? `<span class="badge-high-svi">HIGH</span>`
      : `<span class="badge-low-svi">LOW</span>`;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="mono">${c.fips}</span>${badge}</td>
      <td class="mono">${c.svi.toFixed(3)}</td>
      <td class="mono">${c.demand.toFixed(1)}</td>
      <td>
        <div class="alloc-bar-inline">
          <div class="alloc-bar-track">
            <div class="alloc-bar-fill" style="width:${(c.allocation/maxAlloc*100).toFixed(1)}%"></div>
          </div>
          <span class="mono">${c.allocation.toFixed(1)}</span>
        </div>
      </td>
      <td class="mono" style="color:${cov>=90?'var(--green)':cov>=60?'var(--amber)':'var(--red)'}">${cov.toFixed(0)}%</td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Utilities ─────────────────────────────────────────────────
function pct(v)  { return (v * 100).toFixed(1) + "%"; }
function cap(s)  { return s.charAt(0).toUpperCase() + s.slice(1); }
function fmtUnits(n) {
  if (n >= 1e6) return (n/1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n/1e3).toFixed(0) + "K";
  return String(n);
}

function delta(val, unit, higherBetter = true) {
  const pos = higherBetter ? val >= 0 : val <= 0;
  const cls = pos ? "delta--pos" : "delta--neg";
  const sign = val >= 0 ? "+" : "";
  return `<span class="${cls}">${sign}${val.toFixed(1)}${unit} vs baseline</span>`;
}

function setKpi(id, val, deltaHtml) {
  document.getElementById(id).textContent = val;
  const d = document.getElementById(id + "Delta");
  if (d) d.innerHTML = deltaHtml;
}

function animBar(id, pct, pctLabelId) {
  const el = document.getElementById(id);
  if (el) el.style.width = Math.min(pct, 100).toFixed(1) + "%";
  const lbl = document.getElementById(pctLabelId);
  if (lbl) lbl.textContent = pct.toFixed(1) + "%";
}

// ── Toast ─────────────────────────────────────────────────────
let toastTimer = null;
function showToast(msg, type = "") {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "toast" + (type ? " toast--" + type : "");
  void t.offsetWidth;
  t.classList.add("visible");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("visible"), 3400);
}
