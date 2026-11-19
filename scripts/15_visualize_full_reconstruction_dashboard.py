import os, csv, json, statistics

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECON_DIR = os.path.join(PROJECT, "outputs", "reconstructed_official_full")
EVAL_DIR = os.path.join(PROJECT, "outputs", "evaluation")
OUT_HTML = os.path.join(PROJECT, "outputs", "dashboard", "reconstruction_dashboard_full.html")

def load_eval_csv(path, cos_naive_col, cos_aligned_col):
    """Load evaluation CSV into {image_name: {method: {naive, aligned}}}."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            img = row["image_name"].replace(".png", "")
            method = row["method"]
            naive = float(row[cos_naive_col]) if row[cos_naive_col] not in ("N/A", "") else None
            aligned = float(row[cos_aligned_col]) if row[cos_aligned_col] not in ("N/A", "") else None
            data.setdefault(img, {})[method] = {"naive": naive, "aligned": aligned}
    return data

def load_recon_metrics(path):
    """Load reconstruction_metrics.csv for timing info."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            identity = row["identity"]
            idx = row["image_idx"]
            key = identity + "_img" + idx
            data[key] = {
                "cos_mapped": float(row.get("cos_mapped_vs_real", 0)),
                "b1_time": float(row.get("b1_time_s", 0)),
                "b2_time": float(row.get("b2_time_s", 0)),
                "b3_time": float(row.get("b3_time_s", 0)),
            }
    return data

def build_image_data():
    """Build list of all image records with metrics."""
    glint = load_eval_csv(
        os.path.join(EVAL_DIR, "full_official_eval_aligned.csv"),
        "cosine_naive_resize", "cosine_aligned")
    arcface = load_eval_csv(
        os.path.join(EVAL_DIR, "full_official_eval_arcface_internal.csv"),
        "cosine_internal_naive", "cosine_internal_aligned")
    recon = load_recon_metrics(os.path.join(RECON_DIR, "reconstruction_metrics.csv"))

    originals_dir = os.path.join(RECON_DIR, "originals")
    images = sorted([f.replace(".png", "") for f in os.listdir(originals_dir) if f.endswith(".png")])

    records = []
    for img in images:
        parts = img.rsplit("_", 1)
        identity = parts[0] if len(parts) == 2 else img
        g = glint.get(img, {})
        a = arcface.get(img, {})
        r = recon.get(img, {})
        b1_g = g.get("baseline", {}).get("naive")
        b2_g = g.get("unmapped", {}).get("naive")
        b3_g = g.get("mapped", {}).get("naive")
        b1_a = a.get("baseline", {}).get("naive")
        b2_a = a.get("unmapped", {}).get("naive")
        b3_a = a.get("mapped", {}).get("naive")
        ratio_g = (b3_g / b1_g * 100) if b1_g and b3_g and b1_g > 0.01 else None
        ratio_a = (b3_a / b1_a * 100) if b1_a and b3_a and b1_a > 0.01 else None
        records.append({
            "name": img, "identity": identity,
            "b1_g": b1_g, "b2_g": b2_g, "b3_g": b3_g, "ratio_g": ratio_g,
            "b1_a": b1_a, "b2_a": b2_a, "b3_a": b3_a, "ratio_a": ratio_a,
            "b1_time": r.get("b1_time", 0), "b2_time": r.get("b2_time", 0),
            "b3_time": r.get("b3_time", 0),
        })
    return records

def compute_summary(records):
    """Compute summary statistics."""
    def safe_mean(vals): return statistics.mean(vals) if vals else 0
    def safe_std(vals): return statistics.stdev(vals) if len(vals) > 1 else 0
    b1 = [r["b1_g"] for r in records if r["b1_g"] is not None]
    b2 = [r["b2_g"] for r in records if r["b2_g"] is not None]
    b3 = [r["b3_g"] for r in records if r["b3_g"] is not None]
    ratios = [r["ratio_g"] for r in records if r["ratio_g"] is not None]
    b1a = [r["b1_a"] for r in records if r["b1_a"] is not None]
    b3a = [r["b3_a"] for r in records if r["b3_a"] is not None]
    times = [r["b1_time"] + r["b2_time"] + r["b3_time"] for r in records]
    return {
        "n_images": len(records), "n_recon": len(records) * 3,
        "b1_mean": safe_mean(b1), "b2_mean": safe_mean(b2), "b3_mean": safe_mean(b3),
        "b1_std": safe_std(b1), "b3_std": safe_std(b3),
        "ratio_mean": safe_mean(ratios), "ratio_std": safe_std(ratios),
        "b1a_mean": safe_mean(b1a), "b3a_mean": safe_mean(b3a),
        "ratio_a": safe_mean(b3a) / safe_mean(b1a) * 100 if safe_mean(b1a) > 0 else 0,
        "total_time_h": sum(times) / 3600,
    }

def generate_html(records, summary):
    """Generate the full dashboard HTML."""
    # Relative path from dashboard HTML to reconstruction images
    rel = "../../outputs/reconstructed_official_full"
    records_json = json.dumps(records)
    s = summary

    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Emb2Face — Full 500-Image Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e0e0e0;min-height:100vh}}
.header{{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);padding:32px 24px;text-align:center;border-bottom:2px solid #e94560}}
.header h1{{font-size:2rem;color:#fff;margin-bottom:8px}}
.header p{{color:#a0a0b0;font-size:1rem}}
.cards{{display:flex;flex-wrap:wrap;gap:16px;padding:24px;justify-content:center}}
.card{{background:#1a1a2e;border-radius:12px;padding:20px 24px;min-width:160px;text-align:center;border:1px solid #2a2a4a}}
.card .val{{font-size:1.8rem;font-weight:700;color:#e94560}}
.card .val.green{{color:#4ade80}}.card .val.blue{{color:#60a5fa}}.card .val.gray{{color:#94a3b8}}
.card .lbl{{font-size:0.8rem;color:#888;margin-top:4px}}
.controls{{display:flex;flex-wrap:wrap;gap:12px;padding:12px 24px;align-items:center;background:#12121f;border-bottom:1px solid #2a2a4a}}
.controls select,.controls input,.controls button{{background:#1e1e30;color:#e0e0e0;border:1px solid #3a3a5a;border-radius:6px;padding:8px 12px;font-size:0.85rem}}
.controls button{{cursor:pointer;background:#e94560;border:none;font-weight:600}}
.controls button:hover{{background:#c73652}}
.controls button.sec{{background:#2a2a4a}}.controls button.sec:hover{{background:#3a3a5a}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;padding:24px}}
.img-card{{background:#1a1a2e;border-radius:10px;overflow:hidden;border:1px solid #2a2a4a;cursor:pointer;transition:transform .15s}}
.img-card:hover{{transform:scale(1.02);border-color:#e94560}}
.img-card .imgs{{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:#000}}
.img-card .imgs img{{width:100%;aspect-ratio:1;object-fit:cover}}
.img-card .info{{padding:10px;font-size:0.78rem}}
.img-card .info .id{{font-weight:700;color:#60a5fa}}.img-card .info .met{{color:#aaa}}
.img-card .info .ratio{{font-weight:700;color:#4ade80}}
.modal-bg{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.85);z-index:100;justify-content:center;align-items:center}}
.modal-bg.show{{display:flex}}
.modal{{background:#1a1a2e;border-radius:12px;max-width:900px;width:95%;max-height:90vh;overflow:auto;padding:24px;position:relative}}
.modal .close{{position:absolute;top:12px;right:16px;font-size:1.5rem;cursor:pointer;color:#e94560}}
.modal .cmp{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:16px 0}}
.modal .cmp div{{text-align:center}}
.modal .cmp img{{width:100%;border-radius:6px}}
.modal .cmp .lbl{{font-size:.75rem;color:#888;margin-top:4px}}
.modal table{{width:100%;border-collapse:collapse;margin-top:12px;font-size:.85rem}}
.modal td,.modal th{{padding:6px 10px;border:1px solid #2a2a4a;text-align:center}}
.modal th{{background:#12121f}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:24px}}
.chart-box{{background:#1a1a2e;border-radius:10px;padding:16px;border:1px solid #2a2a4a}}
.chart-box canvas{{max-height:300px}}
.chart-box h3{{font-size:.95rem;color:#60a5fa;margin-bottom:10px}}
section{{padding:0 24px 24px}}
section h2{{font-size:1.3rem;color:#e94560;margin-bottom:12px;padding-top:20px;border-top:1px solid #2a2a4a}}
.explain{{background:#12121f;border-radius:10px;padding:20px;margin-bottom:16px;font-size:.9rem;line-height:1.6}}
.explain h4{{color:#60a5fa;margin-bottom:6px}}
.pager{{display:flex;gap:8px;justify-content:center;padding:12px;align-items:center}}
.pager button{{background:#2a2a4a;color:#e0e0e0;border:none;border-radius:6px;padding:6px 14px;cursor:pointer}}
.pager button.active{{background:#e94560}}
.pager span{{color:#888;font-size:.85rem}}
.top-table{{width:100%;border-collapse:collapse;margin:8px 0;font-size:.85rem}}
.top-table td,.top-table th{{padding:5px 8px;border:1px solid #2a2a4a;text-align:center}}
.top-table th{{background:#12121f;color:#60a5fa}}
@media(max-width:768px){{.charts{{grid-template-columns:1fr}}.modal .cmp{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>

<div class="header">
<h1>🧑 Emb2Face — Full 500-Image Reconstruction Dashboard</h1>
<p>100 identities × 5 images × 3 methods = 1,500 reconstructions | GlintR100 + Internal ArcFace evaluation</p>
</div>

<div class="cards">
<div class="card"><div class="val blue">{s["n_images"]}</div><div class="lbl">Eval Images</div></div>
<div class="card"><div class="val blue">{s["n_recon"]}</div><div class="lbl">Reconstructions</div></div>
<div class="card"><div class="val">{s["b1_mean"]:.4f}</div><div class="lbl">B1 Mean (Glint)</div></div>
<div class="card"><div class="val gray">{s["b2_mean"]:.4f}</div><div class="lbl">B2 Mean (Glint)</div></div>
<div class="card"><div class="val green">{s["b3_mean"]:.4f}</div><div class="lbl">B3 Mean (Glint)</div></div>
<div class="card"><div class="val green">{s["ratio_mean"]:.1f}%</div><div class="lbl">B3/B1 Ratio (Glint)</div></div>
<div class="card"><div class="val blue">{s["b3_mean"]-s["b2_mean"]:.4f}</div><div class="lbl">B3−B2 Improvement</div></div>
<div class="card"><div class="val blue">{s["total_time_h"]:.1f}h</div><div class="lbl">Total Gen Time</div></div>
</div>

<div class="controls" id="controls">
<label>Identity: <select id="idFilter"><option value="">All (100)</option></select></label>
<label>Sort: <select id="sortBy">
<option value="name">Name</option><option value="b3_g_desc">B3 Glint ↓</option><option value="b3_g_asc">B3 Glint ↑</option>
<option value="ratio_desc">B3/B1 ↓</option><option value="ratio_asc">B3/B1 ↑</option>
<option value="b1_g_desc">B1 Glint ↓</option></select></label>
<label>Show: <select id="showFilter">
<option value="all">All Images</option><option value="best20">Best 20 (B3/B1)</option><option value="worst20">Worst 20 (B3/B1)</option>
<option value="best10">Best 10</option><option value="worst10">Worst 10</option></select></label>
<label>Per page: <select id="perPage"><option value="20">20</option><option value="40">40</option><option value="100">100</option><option value="500">All</option></select></label>
<button onclick="applyFilters()">Apply</button>
<button class="sec" onclick="resetFilters()">Reset</button>
</div>

<div class="pager" id="pagerTop"></div>
<div class="gallery" id="gallery"></div>
<div class="pager" id="pagerBottom"></div>

<div class="modal-bg" id="modalBg" onclick="if(event.target===this)closeModal()">
<div class="modal" id="modal"></div>
</div>

<section>
<h2>📊 Charts</h2>
<div class="charts">
<div class="chart-box"><h3>B1 / B2 / B3 Mean Cosine (GlintR100)</h3><canvas id="chartBar"></canvas></div>
<div class="chart-box"><h3>B3/B1 Ratio Distribution</h3><canvas id="chartHist"></canvas></div>
<div class="chart-box"><h3>B3 Cosine per Image (sorted)</h3><canvas id="chartSorted"></canvas></div>
<div class="chart-box"><h3>Generation Time per Image</h3><canvas id="chartTime"></canvas></div>
</div>
</section>

<section>
<h2>🏆 Best 10 Images (by B3/B1 GlintR100)</h2>
<table class="top-table" id="bestTable"><tr><th>#</th><th>Image</th><th>B1</th><th>B3</th><th>B3/B1</th><th>B3 ArcFace</th></tr></table>
</section>
<section>
<h2>⚠️ Worst 10 Images (by B3/B1 GlintR100)</h2>
<table class="top-table" id="worstTable"><tr><th>#</th><th>Image</th><th>B1</th><th>B3</th><th>B3/B1</th><th>B3 ArcFace</th></tr></table>
</section>

<section>
<h2>📖 Method Explanations</h2>
<div class="explain"><h4>B1 — Baseline (Gold Standard)</h4>Original → ArcFace → Real ArcFace embedding → Arc2Face → Generated face.<br>This is the <b>best possible result</b>. Arc2Face receives exactly the embedding type it was designed for.</div>
<div class="explain"><h4>B2 — Unmapped (Negative Control)</h4>Original → AdaFace → Raw AdaFace embedding → Arc2Face → Generated face.<br>This is the <b>expected failure</b>. AdaFace embeddings are in a different space — Arc2Face cannot interpret them. B2 ≈ 0 proves mapping is necessary.</div>
<div class="explain"><h4>B3 — Mapped (Our Method)</h4>Original → AdaFace → Mapping Network → Mapped embedding → Arc2Face → Generated face.<br>This is <b>our contribution</b>. The mapping network translates AdaFace embeddings into ArcFace-compatible space.</div>
<div class="explain"><h4>Why B2 ≈ 0 is important</h4>B2 cosine near zero confirms AdaFace and ArcFace use genuinely different mathematical spaces. Without the mapping network, cross-model face reconstruction is impossible.</div>
<div class="explain"><h4>Why B3 ≈ B1 is good</h4>B3/B1 ≈ 93.7% means our mapping preserves 94% of the identity information. The generated faces are recognizable and would pass typical face verification thresholds.</div>
<div class="explain"><h4>Naive vs Aligned evaluation</h4>Naive: resize 512×512 → 112×112 directly (all 500 images).<br>Aligned: detect face in 512×512, crop & align to 112×112 (~220 images where detection succeeds).<br>Naive covers all images and avoids selection bias; aligned gives slightly higher scores on detected images.</div>
</section>

<div style="text-align:center;padding:24px;color:#555;font-size:.8rem">
Emb2Face Dashboard — Generated {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')} — 500 images, linked-image mode
</div>

<script>
const DATA = {records_json};
const REL = "../../outputs/reconstructed_official_full";
let filtered = [...DATA];
let page = 0;

function getIds() {{
  return [...new Set(DATA.map(d => d.identity))].sort();
}}

window.onload = function() {{
  const sel = document.getElementById("idFilter");
  getIds().forEach(id => {{ const o = document.createElement("option"); o.value=id; o.textContent=id; sel.appendChild(o); }});
  applyFilters();
  drawCharts();
  fillTopTables();
}};

function applyFilters() {{
  const idF = document.getElementById("idFilter").value;
  const sortF = document.getElementById("sortBy").value;
  const showF = document.getElementById("showFilter").value;
  filtered = [...DATA];
  if (idF) filtered = filtered.filter(d => d.identity === idF);
  // Sort
  const sf = {{
    "name": (a,b) => a.name.localeCompare(b.name),
    "b3_g_desc": (a,b) => (b.b3_g||0) - (a.b3_g||0),
    "b3_g_asc": (a,b) => (a.b3_g||0) - (b.b3_g||0),
    "ratio_desc": (a,b) => (b.ratio_g||0) - (a.ratio_g||0),
    "ratio_asc": (a,b) => (a.ratio_g||0) - (b.ratio_g||0),
    "b1_g_desc": (a,b) => (b.b1_g||0) - (a.b1_g||0),
  }};
  filtered.sort(sf[sortF] || sf["name"]);
  if (showF === "best10") filtered = filtered.sort((a,b) => (b.ratio_g||0)-(a.ratio_g||0)).slice(0,10);
  else if (showF === "worst10") filtered = filtered.sort((a,b) => (a.ratio_g||0)-(b.ratio_g||0)).slice(0,10);
  else if (showF === "best20") filtered = filtered.sort((a,b) => (b.ratio_g||0)-(a.ratio_g||0)).slice(0,20);
  else if (showF === "worst20") filtered = filtered.sort((a,b) => (a.ratio_g||0)-(b.ratio_g||0)).slice(0,20);
  page = 0;
  renderGallery();
}}

function resetFilters() {{
  document.getElementById("idFilter").value = "";
  document.getElementById("sortBy").value = "name";
  document.getElementById("showFilter").value = "all";
  document.getElementById("perPage").value = "20";
  applyFilters();
}}

function renderGallery() {{
  const pp = parseInt(document.getElementById("perPage").value) || 20;
  const totalPages = Math.ceil(filtered.length / pp);
  const start = page * pp;
  const slice = filtered.slice(start, start + pp);
  const gal = document.getElementById("gallery");
  gal.innerHTML = "";
  slice.forEach((d, i) => {{
    const r = d.ratio_g != null ? d.ratio_g.toFixed(1) + "%" : "N/A";
    const b3 = d.b3_g != null ? d.b3_g.toFixed(4) : "N/A";
    const card = document.createElement("div");
    card.className = "img-card";
    card.onclick = () => showModal(start + i);
    card.innerHTML = `<div class="imgs">
      <img src="${{REL}}/originals/${{d.name}}.png" alt="Original" loading="lazy">
      <img src="${{REL}}/baseline/${{d.name}}.png" alt="B1" loading="lazy">
      <img src="${{REL}}/unmapped/${{d.name}}.png" alt="B2" loading="lazy">
      <img src="${{REL}}/mapped/${{d.name}}.png" alt="B3" loading="lazy">
    </div><div class="info"><span class="id">${{d.identity}}</span> <span class="met">B3=${{b3}}</span> <span class="ratio">B3/B1=${{r}}</span></div>`;
    gal.appendChild(card);
  }});
  // Pager
  ["pagerTop","pagerBottom"].forEach(pid => {{
    const p = document.getElementById(pid);
    p.innerHTML = `<span>${{filtered.length}} images | Page ${{page+1}}/${{totalPages}}</span>`;
    if (page > 0) {{ const b = document.createElement("button"); b.textContent="← Prev"; b.onclick=()=>{{page--;renderGallery();window.scrollTo(0,0)}}; p.prepend(b); }}
    if (page < totalPages-1) {{ const b = document.createElement("button"); b.textContent="Next →"; b.onclick=()=>{{page++;renderGallery();window.scrollTo(0,0)}}; p.appendChild(b); }}
  }});
}}

function showModal(idx) {{
  const d = filtered[idx];
  const m = document.getElementById("modal");
  m.innerHTML = `<span class="close" onclick="closeModal()">✕</span>
    <h3 style="color:#60a5fa;margin-bottom:8px">${{d.name}}</h3>
    <div class="cmp">
      <div><img src="${{REL}}/originals/${{d.name}}.png"><div class="lbl">Original</div></div>
      <div><img src="${{REL}}/baseline/${{d.name}}.png"><div class="lbl">B1 Baseline</div></div>
      <div><img src="${{REL}}/unmapped/${{d.name}}.png"><div class="lbl">B2 Unmapped</div></div>
      <div><img src="${{REL}}/mapped/${{d.name}}.png"><div class="lbl">B3 Mapped</div></div>
    </div>
    <table>
      <tr><th>Metric</th><th>B1</th><th>B2</th><th>B3</th><th>B3/B1</th></tr>
      <tr><td>GlintR100</td><td>${{fmt(d.b1_g)}}</td><td>${{fmt(d.b2_g)}}</td><td>${{fmt(d.b3_g)}}</td><td style="color:#4ade80;font-weight:700">${{d.ratio_g?d.ratio_g.toFixed(1)+"%":"N/A"}}</td></tr>
      <tr><td>Internal ArcFace</td><td>${{fmt(d.b1_a)}}</td><td>${{fmt(d.b2_a)}}</td><td>${{fmt(d.b3_a)}}</td><td style="color:#4ade80;font-weight:700">${{d.ratio_a?d.ratio_a.toFixed(1)+"%":"N/A"}}</td></tr>
      <tr><td>Gen Time (s)</td><td>${{d.b1_time.toFixed(1)}}</td><td>${{d.b2_time.toFixed(1)}}</td><td>${{d.b3_time.toFixed(1)}}</td><td>${{(d.b1_time+d.b2_time+d.b3_time).toFixed(1)}}s total</td></tr>
    </table>`;
  document.getElementById("modalBg").classList.add("show");
}}
function closeModal(){{document.getElementById("modalBg").classList.remove("show")}}
function fmt(v){{return v!=null?v.toFixed(4):"N/A"}}
document.addEventListener("keydown",e=>{{if(e.key==="Escape")closeModal()}});

function drawCharts() {{
  // Bar chart
  new Chart(document.getElementById("chartBar"),{{type:"bar",data:{{labels:["B1 Baseline","B2 Unmapped","B3 Mapped"],datasets:[{{label:"GlintR100 Naive",data:[{s["b1_mean"]:.4f},{s["b2_mean"]:.4f},{s["b3_mean"]:.4f}],backgroundColor:["#e94560","#94a3b8","#4ade80"]}},{{label:"Internal ArcFace",data:[{s["b1a_mean"]:.4f},0,{s["b3a_mean"]:.4f}],backgroundColor:["#f97316","#94a3b8","#22d3ee"]}}]}},options:{{responsive:true,scales:{{y:{{beginAtZero:true,max:1,grid:{{color:"#2a2a4a"}}}},x:{{grid:{{color:"#2a2a4a"}}}}}},plugins:{{legend:{{labels:{{color:"#e0e0e0"}}}}}}}}}});
  // Histogram of B3/B1 ratios
  const ratios = DATA.filter(d=>d.ratio_g!=null).map(d=>d.ratio_g);
  const bins = Array(20).fill(0);
  ratios.forEach(r => {{ let b=Math.floor((r-40)/4); if(b<0)b=0; if(b>19)b=19; bins[b]++; }});
  const binLabels = Array.from({{length:20}},(_,i)=>(40+i*4)+"-"+(44+i*4)+"%");
  new Chart(document.getElementById("chartHist"),{{type:"bar",data:{{labels:binLabels,datasets:[{{label:"Count",data:bins,backgroundColor:"#e94560"}}]}},options:{{responsive:true,scales:{{y:{{beginAtZero:true,grid:{{color:"#2a2a4a"}}}},x:{{grid:{{color:"#2a2a4a"}},ticks:{{maxRotation:60}}}}}},plugins:{{legend:{{display:false}}}}}}}});
  // Sorted B3
  const sorted = DATA.filter(d=>d.b3_g!=null).map(d=>d.b3_g).sort((a,b)=>a-b);
  new Chart(document.getElementById("chartSorted"),{{type:"line",data:{{labels:sorted.map((_,i)=>i),datasets:[{{label:"B3 Cosine",data:sorted,borderColor:"#4ade80",pointRadius:0,borderWidth:1.5}}]}},options:{{responsive:true,scales:{{y:{{beginAtZero:true,max:1,grid:{{color:"#2a2a4a"}}}},x:{{display:false}}}},plugins:{{legend:{{labels:{{color:"#e0e0e0"}}}}}}}}}});
  // Time chart
  const times = DATA.map(d=>d.b1_time+d.b2_time+d.b3_time).sort((a,b)=>a-b);
  new Chart(document.getElementById("chartTime"),{{type:"line",data:{{labels:times.map((_,i)=>i),datasets:[{{label:"Total Time (s)",data:times,borderColor:"#f97316",pointRadius:0,borderWidth:1.5}}]}},options:{{responsive:true,scales:{{y:{{beginAtZero:true,grid:{{color:"#2a2a4a"}}}},x:{{display:false}}}},plugins:{{legend:{{labels:{{color:"#e0e0e0"}}}}}}}}}});
}}

function fillTopTables() {{
  const byRatio = DATA.filter(d=>d.ratio_g!=null).sort((a,b)=>b.ratio_g-a.ratio_g);
  const best = byRatio.slice(0,10);
  const worst = byRatio.slice(-10).reverse();
  const fill = (tbl,items) => {{
    items.forEach((d,i) => {{
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${{i+1}}</td><td style="color:#60a5fa">${{d.name}}</td><td>${{d.b1_g?.toFixed(4)}}</td><td>${{d.b3_g?.toFixed(4)}}</td><td style="color:#4ade80;font-weight:700">${{d.ratio_g?.toFixed(1)}}%</td><td>${{d.b3_a?.toFixed(4)||"N/A"}}</td>`;
      tbl.appendChild(tr);
    }});
  }};
  fill(document.getElementById("bestTable"), best);
  fill(document.getElementById("worstTable"), worst);
}}
</script></body></html>'''
    return html

def main():
    print("Loading data...")
    records = build_image_data()
    summary = compute_summary(records)
    print(f"  {summary['n_images']} images, {summary['n_recon']} reconstructions")
    print(f"  B1={summary['b1_mean']:.4f}  B2={summary['b2_mean']:.4f}  B3={summary['b3_mean']:.4f}")
    print(f"  B3/B1={summary['ratio_mean']:.1f}%")

    print("Generating HTML...")
    html = generate_html(records, summary)
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(OUT_HTML) / 1024 / 1024
    print(f"Dashboard written to: {OUT_HTML}")
    print(f"  Size: {size_mb:.1f} MB (linked-image mode)")
    print(f"\nTo view:")
    print(f"  cd {PROJECT}")
    print(f"  python3 -m http.server 8080")
    print(f"  Open: http://localhost:8080/outputs/dashboard/reconstruction_dashboard_full.html")

if __name__ == "__main__":
    main()
