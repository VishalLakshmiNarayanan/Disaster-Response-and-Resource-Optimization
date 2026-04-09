/* ============================================================
   map.js  —  D3 v7 interactive US county choropleth
   Modes: svi / demand / allocation
   Selection: click counties for custom scenario
   ============================================================ */

const MAP = (() => {
  // ── internal state ────────────────────────────────────────────
  let geoData      = null;
  let sviData      = {};
  let stormData    = null;
  let allocData    = {};
  let demandData   = {};
  let mode         = "svi";
  let selectMode   = false;   // custom-scenario click-select
  let selectedFips = new Set();
  let affectedFips = new Set();

  let projection, path, svg, g, countyPaths, zoomBehavior;
  let countyNames  = {};
  let W = 960, H = 500;

  const TOPO_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json";

  // ── colour scales ─────────────────────────────────────────────
  // Use more visible, higher-contrast gradients
  const colorSvi = v => {
    if (v < 0.25)  return `hsl(220,30%,${18 + v*24}%)`;
    if (v < 0.5)   return `hsl(210,55%,${28 + (v-0.25)*40}%)`;
    if (v < 0.75)  return `hsl(200,80%,${38 + (v-0.5)*30}%)`;
    return `hsl(${200 - (v-0.75)*160},90%,${50 + (v-0.75)*20}%)`;
  };

  const colorDemand = v =>
    d3.interpolateRgb("#192840", "#22d3a5")(Math.pow(v, 0.45));

  const colorAlloc = v =>
    d3.interpolateRgb("#1e1a10", "#fbbf24")(Math.pow(v, 0.45));

  // ── helpers ──────────────────────────────────────────────────
  function fipsStr(id) { return String(id).padStart(5, "0"); }

  function getFill(fipsId) {
    const fips = fipsStr(fipsId);

    // Custom-scenario selection highlight
    if (selectMode && selectedFips.has(fips)) return "#4f8dff";

    // Dim non-affected counties when a storm is loaded
    if (stormData && !affectedFips.has(fips) && !selectMode) {
      return "#101118";
    }

    if (mode === "svi") {
      const v = sviData[fips] ?? sviData[parseInt(fips)];
      return v != null ? colorSvi(+v) : "#18192a";
    }
    if (mode === "demand") {
      const v = demandData[fips];
      return v != null ? colorDemand(v) : "#18192a";
    }
    if (mode === "allocation") {
      const v = allocData[fips];
      return v != null ? colorAlloc(v) : "#18192a";
    }
    return "#18192a";
  }

  function normalise(dict) {
    const vals = Object.values(dict).map(Number).filter(isFinite);
    if (!vals.length) return dict;
    const mx = Math.max(...vals) || 1;
    const out = {};
    for (const [k, v] of Object.entries(dict)) out[k] = +v / mx;
    return out;
  }

  // ── resize helper ─────────────────────────────────────────────
  function resize() {
    const container = document.getElementById("mapContainer");
    W = container.clientWidth  || 960;
    H = container.clientHeight || 500;

    svg.attr("width",  W)
       .attr("height", H);

    // Recalculate AlbersUSA to fill container
    const scale = Math.min(W * 1.25, H * 2.2);
    projection
      .scale(scale)
      .translate([W / 2, H / 2]);

    path = d3.geoPath().projection(projection);

    if (countyPaths) countyPaths.attr("d", path);
    g.selectAll(".state-borders").attr("d", path);

    // Reset zoom so map fits after resize
    svg.call(zoomBehavior.transform, d3.zoomIdentity);
  }

  // ── init ─────────────────────────────────────────────────────
  async function init() {
    showOverlay("Loading map…");

    svg = d3.select("#mapSvg");

    const container = document.getElementById("mapContainer");
    W = container.clientWidth  || 960;
    H = container.clientHeight || 500;

    svg.attr("width", W).attr("height", H);

    // No viewBox — use pixel coordinates directly for correct zoom
    const scale = Math.min(W * 1.25, H * 2.2);
    projection = d3.geoAlbersUsa()
      .scale(scale)
      .translate([W / 2, H / 2]);

    path = d3.geoPath().projection(projection);

    // Zoom behaviour — applied to the <svg>, transforms <g>
    zoomBehavior = d3.zoom()
      .scaleExtent([1, 12])
      .translateExtent([[0, 0], [W, H]])
      .on("zoom", e => g.attr("transform", e.transform));

    svg.call(zoomBehavior)
       .on("dblclick.zoom", null); // disable double-click zoom (use our own reset)

    const [topo, svi] = await Promise.all([
      d3.json(TOPO_URL),
      fetch("/api/counties/svi").then(r => r.json()),
    ]);

    geoData = topo;
    sviData = svi;

    topojson.feature(topo, topo.objects.counties).features.forEach(f => {
      countyNames[fipsStr(f.id)] = f.properties.name || f.id;
    });

    g = svg.append("g");

    // County fill paths
    countyPaths = g.append("g")
      .selectAll("path")
      .data(topojson.feature(topo, topo.objects.counties).features)
      .join("path")
        .attr("class", "county")
        .attr("d", path)
        .attr("fill", d => getFill(d.id))
        .on("mousemove", onMouseMove)
        .on("mouseleave", () => {
          document.getElementById("mapTooltip").classList.remove("visible");
        })
        .on("click", onCountyClick);

    // State borders on top
    g.append("path")
      .datum(topojson.mesh(topo, topo.objects.states, (a, b) => a !== b))
      .attr("class", "state-borders")
      .attr("d", path);

    hideOverlay();
    updateLegend();

    // Zoom reset button
    document.getElementById("zoomReset")?.addEventListener("click", () => {
      svg.transition().duration(500).call(zoomBehavior.transform, d3.zoomIdentity);
    });

    // Resize observer
    new ResizeObserver(resize).observe(container);
  }

  // ── click to select county ────────────────────────────────────
  function onCountyClick(event, d) {
    if (!selectMode) return;
    const fips = fipsStr(d.id);
    if (selectedFips.has(fips)) {
      selectedFips.delete(fips);
      d3.select(event.currentTarget).attr("fill", getFill(d.id));
    } else {
      selectedFips.add(fips);
      d3.select(event.currentTarget).attr("fill", "#4f8dff");
    }
    // Notify simulator
    if (window.SIM) SIM.onSelectionChange(selectedFips.size);
  }

  // ── tooltip ──────────────────────────────────────────────────
  function onMouseMove(event, d) {
    const fips = fipsStr(d.id);
    const name = countyNames[fips] || fips;
    const sviV = sviData[fips] ?? sviData[parseInt(fips)];
    const dem  = demandData[fips];
    const alc  = allocData[fips];

    document.getElementById("ttName").textContent    = name + " County";
    document.getElementById("ttSvi").textContent     = sviV != null ? (+sviV).toFixed(3) : "—";
    document.getElementById("ttDemand").textContent  = dem  != null ? (+dem).toFixed(1)   : "—";
    document.getElementById("ttAlloc").textContent   = alc  != null ? (+alc).toFixed(1)   : "—";
    document.getElementById("ttFips").textContent    = fips;
    document.getElementById("ttSelected").textContent =
      selectMode ? (selectedFips.has(fips) ? "✓ selected" : "click to select") : "";

    const rect = document.getElementById("mapContainer").getBoundingClientRect();
    let x = event.clientX - rect.left + 14;
    let y = event.clientY - rect.top  + 14;
    const tt = document.getElementById("mapTooltip");
    if (x + 200 > rect.width)  x = event.clientX - rect.left - 206;
    if (y + 140 > rect.height) y = event.clientY - rect.top  - 136;
    tt.style.left = x + "px";
    tt.style.top  = y + "px";
    tt.classList.add("visible");
  }

  // ── overlays ─────────────────────────────────────────────────
  function showOverlay(msg) {
    document.getElementById("overlayText").textContent = msg;
    document.getElementById("mapOverlay").classList.add("visible");
  }
  function hideOverlay() {
    document.getElementById("mapOverlay").classList.remove("visible");
  }

  // ── storm highlight ──────────────────────────────────────────
  function highlightStorm(counties) {
    stormData = counties;
    affectedFips = new Set(counties.map(c => fipsStr(c.fips)));

    const rawDemand = {};
    counties.forEach(c => { rawDemand[fipsStr(c.fips)] = c.pred_demand ?? 0; });
    demandData = normalise(rawDemand);
    allocData  = {};

    repaint();
  }

  // ── allocations overlay ──────────────────────────────────────
  function applyAllocations(topCounties) {
    const raw = {};
    topCounties.forEach(c => { raw[fipsStr(c.fips)] = c.allocation ?? 0; });
    allocData = normalise(raw);
    if (mode === "allocation") repaint();
  }

  // ── selection mode ────────────────────────────────────────────
  function setSelectMode(enabled) {
    selectMode = enabled;
    if (!enabled) {
      selectedFips.clear();
    }
    svg.style("cursor", enabled ? "crosshair" : "grab");
    repaint();
  }

  function getSelectedFips() { return [...selectedFips]; }

  function clearSelection() {
    selectedFips.clear();
    repaint();
    if (window.SIM) SIM.onSelectionChange(0);
  }

  // ── repaint & legend ─────────────────────────────────────────
  function repaint() {
    if (!countyPaths) return;
    countyPaths.attr("fill", d => getFill(d.id));
  }

  function setMode(m) {
    mode = m;
    repaint();
    updateLegend();
  }

  function updateLegend() {
    const cfg = {
      svi:        { title: "SVI Vulnerability",  lo: "Low", hi: "High" },
      demand:     { title: "Predicted Demand",   lo: "Low", hi: "High" },
      allocation: { title: "LP Allocation",      lo: "Low", hi: "High" },
    };
    const c = cfg[mode] || cfg.svi;
    document.getElementById("legendTitle").textContent = c.title;
    document.getElementById("legendLow").textContent   = c.lo;
    document.getElementById("legendHigh").textContent  = c.hi;

    const gradients = {
      svi:        "linear-gradient(90deg,#18192a,#4f8dff,#f87171)",
      demand:     "linear-gradient(90deg,#192840,#22d3a5)",
      allocation: "linear-gradient(90deg,#1e1a10,#fbbf24)",
    };
    document.getElementById("legendGradient").style.background = gradients[mode] || gradients.svi;
  }

  function resetStorm() {
    stormData   = null;
    affectedFips = new Set();
    demandData  = {};
    allocData   = {};
    repaint();
  }

  return {
    init, highlightStorm, applyAllocations,
    setMode, resetStorm,
    setSelectMode, getSelectedFips, clearSelection,
    showOverlay, hideOverlay,
  };
})();

// ── Map mode toggle buttons ──────────────────────────────────
document.querySelectorAll(".map-toggle").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".map-toggle").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    MAP.setMode(btn.dataset.mode);
  });
});

document.addEventListener("DOMContentLoaded", () => MAP.init());
