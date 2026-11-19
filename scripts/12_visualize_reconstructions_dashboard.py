import os
import sys
import csv
import base64
import json
from pathlib import Path

# ── Project root ──
PROJECT = Path(__file__).resolve().parent.parent
RECON = PROJECT / "outputs" / "reconstructed_official"
EVAL_DIR = PROJECT / "outputs" / "evaluation"
OUT_DIR = PROJECT / "outputs" / "dashboard"

# ── Data paths (read-only) ──
DIRS = {
    "original": RECON / "originals",
    "baseline": RECON / "baseline",
    "unmapped": RECON / "unmapped",
    "mapped":   RECON / "mapped",
}
CSV_ALIGNED  = EVAL_DIR / "reconstructed_official_eval_aligned.csv"
CSV_INTERNAL = EVAL_DIR / "reconstructed_official_eval_arcface_internal.csv"
CSV_BASELINE = RECON / "baseline_metrics.csv"

def img_to_b64(path):
    """Encode image file to base64 data URI."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = path.suffix.lstrip(".").lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    return f"data:{mime};base64,{data}"

def load_csv(path):
    """Load CSV as list of dicts."""
    if not path.exists():
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))

def safe_float(val, default=None):
    """Convert to float, returning default on failure."""
    try:
        if val in (None, "", "N/A"):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

def build_identity_data():
    """Build structured data for all 10 identities."""
    # Get identity list from originals
    orig_dir = DIRS["original"]
    if not orig_dir.exists():
        print(f"ERROR: {orig_dir} not found"); sys.exit(1)
    
    images = sorted([f.name for f in orig_dir.iterdir() if f.suffix == ".png"])
    identities = []
    
    # Load eval CSVs
    aligned_rows = load_csv(CSV_ALIGNED)
    internal_rows = load_csv(CSV_INTERNAL)
    baseline_rows = load_csv(CSV_BASELINE)
    
    # Index eval data by (identity, method)
    def index_eval(rows, cos_key):
        idx = {}
        for r in rows:
            k = (r.get("identity", ""), r.get("method", ""))
            idx[k] = safe_float(r.get(cos_key))
        return idx
    
    aligned_naive = index_eval(aligned_rows, "cosine_naive_resize")
    internal_naive = index_eval(internal_rows, "cosine_internal_naive")
    
    # Index baseline metrics
    baseline_idx = {}
    for r in baseline_rows:
        baseline_idx[r.get("identity", "")] = r
    
    for img_name in images:
        identity = img_name.replace(".png", "")
        
        # Encode images
        imgs = {}
        for key, d in DIRS.items():
            imgs[key] = img_to_b64(d / img_name)
        
        # Metrics
        b1_aligned = aligned_naive.get((identity, "baseline"))
        b2_aligned = aligned_naive.get((identity, "unmapped"))
        b3_aligned = aligned_naive.get((identity, "mapped"))
        b1_internal = internal_naive.get((identity, "baseline"))
        b2_internal = internal_naive.get((identity, "unmapped"))
        b3_internal = internal_naive.get((identity, "mapped"))
        
        # Timing from baseline_metrics
        bm = baseline_idx.get(identity.rsplit("_img", 1)[0], {})
        
        entry = {
            "id": identity,
            "short_id": identity.rsplit("_img", 1)[0],
            "images": imgs,
            "b1_glint": b1_aligned,
            "b2_glint": b2_aligned,
            "b3_glint": b3_aligned,
            "b1_internal": b1_internal,
            "b2_internal": b2_internal,
            "b3_internal": b3_internal,
            "cos_adaface_mapped": safe_float(bm.get("cos_adaface_mapped")),
            "b1_time": safe_float(bm.get("b1_time_s")),
            "b2_time": safe_float(bm.get("b2_time_s")),
            "b3_time": safe_float(bm.get("b3_time_s")),
        }
        identities.append(entry)
    
    return identities

def compute_summary(identities):
    """Compute summary statistics."""
    def mean_of(key):
        vals = [e[key] for e in identities if e[key] is not None]
        return sum(vals) / len(vals) if vals else None
    
    b1g = mean_of("b1_glint")
    b2g = mean_of("b2_glint")
    b3g = mean_of("b3_glint")
    b1i = mean_of("b1_internal")
    b2i = mean_of("b2_internal")
    b3i = mean_of("b3_internal")
    
    return {
        "b1_glint": b1g, "b2_glint": b2g, "b3_glint": b3g,
        "b1_internal": b1i, "b2_internal": b2i, "b3_internal": b3i,
        "ratio_glint": (b3g / b1g * 100) if b1g and b3g else None,
        "ratio_internal": (b3i / b1i * 100) if b1i and b3i else None,
        "improve_glint": (b3g - b2g) if b3g is not None and b2g is not None else None,
        "improve_internal": (b3i - b2i) if b3i is not None and b2i is not None else None,
        "mean_b1_time": mean_of("b1_time"),
        "mean_b2_time": mean_of("b2_time"),
        "mean_b3_time": mean_of("b3_time"),
    }

def fmt(v, d=4):
    return f"{v:.{d}f}" if v is not None else "N/A"

def pct(v):
    return f"{v:.1f}%" if v is not None else "N/A"

def gen_card(label, value, color, sub=""):
    return f'<div class="card" style="border-top:3px solid {color}"><div class="card-label">{label}</div><div class="card-value" style="color:{color}">{value}</div>{f"<div class=card-sub>{sub}</div>" if sub else ""}</div>'

def gen_identity_panel(e, idx):
    """Generate HTML for one identity's image comparison row."""
    labels = [("Original","original"),("B1 Baseline","baseline"),("B2 Unmapped","unmapped"),("B3 Mapped","mapped")]
    imgs_html = ""
    for lbl, key in labels:
        src = e["images"].get(key, "")
        imgs_html += f'<div class="img-cell"><div class="img-label">{lbl}</div><img src="{src}" alt="{lbl}" loading="lazy"/></div>'
    
    metrics_html = '<div class="metrics-row">'
    for lbl, gk, ik in [("B1",e["b1_glint"],e["b1_internal"]),("B2",e["b2_glint"],e["b2_internal"]),("B3",e["b3_glint"],e["b3_internal"])]:
        metrics_html += f'<div class="metric-badge"><b>{lbl}</b> GlintR100: {fmt(gk)} | ArcFace: {fmt(ik)}</div>'
    
    ratio_g = (e["b3_glint"]/e["b1_glint"]*100) if e["b1_glint"] and e["b3_glint"] else None
    ratio_i = (e["b3_internal"]/e["b1_internal"]*100) if e["b1_internal"] and e["b3_internal"] else None
    metrics_html += f'<div class="metric-badge ratio">B3/B1: {pct(ratio_g)} (G) | {pct(ratio_i)} (I)</div>'
    
    if e["b1_time"]:
        metrics_html += f'<div class="metric-badge time">⏱ B1:{e["b1_time"]:.0f}s B2:{e["b2_time"]:.0f}s B3:{e["b3_time"]:.0f}s</div>'
    metrics_html += '</div>'
    
    return f'''<div class="identity-panel" id="panel-{idx}" data-id="{e["short_id"]}">
<h3>🆔 {e["short_id"]}</h3>
<div class="img-row">{imgs_html}</div>
{metrics_html}</div>'''

def generate_html(identities, summary):
    """Generate the full HTML dashboard."""
    # Build identity panels
    panels = "\n".join(gen_identity_panel(e, i) for i, e in enumerate(identities))
    
    # Selector options
    opts = "".join(f'<option value="{i}">{e["short_id"]}</option>' for i, e in enumerate(identities))
    
    # Chart data
    labels_js = json.dumps([e["short_id"] for e in identities])
    b1g = json.dumps([e["b1_glint"] or 0 for e in identities])
    b2g = json.dumps([e["b2_glint"] or 0 for e in identities])
    b3g = json.dumps([e["b3_glint"] or 0 for e in identities])
    b1i = json.dumps([e["b1_internal"] or 0 for e in identities])
    b2i = json.dumps([e["b2_internal"] or 0 for e in identities])
    b3i = json.dumps([e["b3_internal"] or 0 for e in identities])
    ratios_g = json.dumps([(e["b3_glint"]/e["b1_glint"]*100) if e["b1_glint"] and e["b3_glint"] else 0 for e in identities])
    ratios_i = json.dumps([(e["b3_internal"]/e["b1_internal"]*100) if e["b1_internal"] and e["b3_internal"] else 0 for e in identities])
    t1 = json.dumps([e["b1_time"] or 0 for e in identities])
    t2 = json.dumps([e["b2_time"] or 0 for e in identities])
    t3 = json.dumps([e["b3_time"] or 0 for e in identities])
    
    # Summary cards
    cards = ""
    cards += gen_card("B1 Mean (GlintR100)", fmt(summary["b1_glint"]), "#6366f1")
    cards += gen_card("B2 Mean (GlintR100)", fmt(summary["b2_glint"]), "#ef4444")
    cards += gen_card("B3 Mean (GlintR100)", fmt(summary["b3_glint"]), "#10b981")
    cards += gen_card("B3/B1 Ratio (Glint)", pct(summary["ratio_glint"]), "#f59e0b", "Target: ≥90%")
    cards += gen_card("B3−B2 Improvement (Glint)", fmt(summary["improve_glint"]), "#8b5cf6")
    cards += gen_card("B1 Mean (ArcFace)", fmt(summary["b1_internal"]), "#6366f1", "Internal")
    cards += gen_card("B2 Mean (ArcFace)", fmt(summary["b2_internal"]), "#ef4444", "Internal")
    cards += gen_card("B3 Mean (ArcFace)", fmt(summary["b3_internal"]), "#10b981", "Internal")
    cards += gen_card("B3/B1 Ratio (ArcFace)", pct(summary["ratio_internal"]), "#f59e0b", "Internal eval")
    cards += gen_card("B3−B2 Improvement (ArcFace)", fmt(summary["improve_internal"]), "#8b5cf6", "Internal eval")

    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Emb2Face Reconstruction Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;line-height:1.6}}
.header{{background:linear-gradient(135deg,#1e1b4b,#312e81,#1e3a5f);padding:2rem;text-align:center;border-bottom:2px solid #6366f1}}
.header h1{{font-size:2rem;background:linear-gradient(90deg,#818cf8,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.3rem}}
.header p{{color:#94a3b8;font-size:.95rem}}
.container{{max-width:1400px;margin:0 auto;padding:1.5rem}}
.section{{margin-bottom:2rem}}
.section h2{{font-size:1.4rem;color:#818cf8;margin-bottom:1rem;padding-bottom:.5rem;border-bottom:1px solid #334155}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:1rem;margin-bottom:2rem}}
.card{{background:#1e293b;border-radius:12px;padding:1.2rem;text-align:center;transition:transform .2s,box-shadow .2s}}
.card:hover{{transform:translateY(-3px);box-shadow:0 8px 25px rgba(99,102,241,.2)}}
.card-label{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.4rem}}
.card-value{{font-size:1.5rem;font-weight:700}}
.card-sub{{font-size:.7rem;color:#64748b;margin-top:.3rem}}
.controls{{display:flex;gap:1rem;align-items:center;margin-bottom:1.5rem;flex-wrap:wrap}}
.controls select,.controls button{{background:#1e293b;color:#e2e8f0;border:1px solid #475569;border-radius:8px;padding:.6rem 1rem;font-size:.9rem;cursor:pointer}}
.controls button{{background:linear-gradient(135deg,#6366f1,#8b5cf6);border:none;font-weight:600}}
.controls button:hover{{opacity:.9}}
.controls button.active{{background:linear-gradient(135deg,#10b981,#059669)}}
.identity-panel{{background:#1e293b;border-radius:16px;padding:1.5rem;margin-bottom:1.5rem;border:1px solid #334155;transition:all .3s}}
.identity-panel h3{{color:#a5b4fc;margin-bottom:1rem;font-size:1.1rem}}
.img-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1rem}}
.img-cell{{text-align:center}}
.img-cell img{{width:100%;border-radius:8px;border:2px solid #334155;transition:transform .3s}}
.img-cell img:hover{{transform:scale(1.05);border-color:#6366f1}}
.img-label{{font-size:.75rem;color:#94a3b8;margin-bottom:.4rem;text-transform:uppercase;letter-spacing:.03em}}
.metrics-row{{display:flex;gap:.6rem;flex-wrap:wrap}}
.metric-badge{{background:#0f172a;border-radius:8px;padding:.4rem .8rem;font-size:.78rem;color:#cbd5e1;border:1px solid #334155}}
.metric-badge.ratio{{border-color:#f59e0b;color:#fbbf24}}
.metric-badge.time{{border-color:#06b6d4;color:#67e8f9}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:1.5rem}}
.chart-box{{background:#1e293b;border-radius:16px;padding:1.5rem;border:1px solid #334155}}
.chart-box h3{{color:#a5b4fc;font-size:1rem;margin-bottom:1rem}}
.chart-box canvas{{max-height:320px}}
.explain{{background:#1e293b;border-radius:16px;padding:2rem;border:1px solid #334155}}
.explain h3{{color:#818cf8;margin-bottom:1rem}}
.explain-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1.5rem}}
.explain-item{{background:#0f172a;border-radius:12px;padding:1.2rem;border-left:3px solid #6366f1}}
.explain-item h4{{margin-bottom:.5rem}}
.explain-item.b1 h4{{color:#818cf8}} .explain-item.b2 h4{{color:#ef4444}} .explain-item.b3 h4{{color:#10b981}} .explain-item.why h4{{color:#f59e0b}}
.explain-item.b1{{border-color:#6366f1}} .explain-item.b2{{border-color:#ef4444}} .explain-item.b3{{border-color:#10b981}} .explain-item.why{{border-color:#f59e0b}}
.hidden{{display:none!important}}
.footer{{text-align:center;color:#475569;font-size:.8rem;padding:2rem;border-top:1px solid #1e293b;margin-top:2rem}}
</style></head><body>
<div class="header"><h1>🔬 Emb2Face Reconstruction Dashboard</h1>
<p>10-Identity Official Reconstruction Results &mdash; Visual Analysis &amp; Metrics</p></div>
<div class="container">

<div class="section"><h2>📊 Summary Metrics</h2><div class="cards">{cards}</div></div>

<div class="section"><h2>🖼️ Identity Comparison</h2>
<div class="controls">
<select id="idSelect" onchange="filterIdentity()"><option value="all">All Identities</option>{opts}</select>
<button id="gridBtn" class="active" onclick="setMode('grid')">Grid View</button>
<button id="singleBtn" onclick="setMode('single')">Single View</button>
</div>
<div id="panelsContainer">{panels}</div></div>

<div class="section"><h2>📈 Charts</h2><div class="charts">
<div class="chart-box"><h3>Cosine Similarity — GlintR100</h3><canvas id="chartGlint"></canvas></div>
<div class="chart-box"><h3>Cosine Similarity — Internal ArcFace</h3><canvas id="chartInternal"></canvas></div>
<div class="chart-box"><h3>B3/B1 Ratio per Identity</h3><canvas id="chartRatio"></canvas></div>
<div class="chart-box"><h3>Generation Time (seconds)</h3><canvas id="chartTime"></canvas></div>
</div></div>

<div class="section"><h2>📖 Method Explanation</h2>
<div class="explain"><div class="explain-grid">
<div class="explain-item b1"><h4>B1 — Baseline</h4><p>Real ArcFace embedding → Arc2Face generator. This is the <b>upper bound</b>: what Arc2Face produces when given the correct embedding type it was trained on.</p></div>
<div class="explain-item b2"><h4>B2 — Unmapped</h4><p>Raw AdaFace embedding → Arc2Face generator <b>without mapping</b>. Expected to fail because Arc2Face expects ArcFace embeddings, not AdaFace.</p></div>
<div class="explain-item b3"><h4>B3 — Mapped</h4><p>AdaFace embedding → <b>Learned Mapping Network</b> → Arc2Face generator. The mapping translates AdaFace embeddings into ArcFace-compatible space.</p></div>
<div class="explain-item why"><h4>Why B3 ≈ B1 is Good</h4><p>If B3/B1 ratio is high (≥90%), the mapping network successfully bridges the embedding gap. This proves we can reconstruct faces from <b>any</b> face recognition model via learned translation.</p></div>
</div></div></div>
</div>

<div class="footer">Emb2Face Dashboard — Generated by 12_visualize_reconstructions_dashboard.py — Read-only, no outputs modified</div>

<script>
const LABELS={labels_js};
const B1G={b1g},B2G={b2g},B3G={b3g};
const B1I={b1i},B2I={b2i},B3I={b3i};
const RATG={ratios_g},RATI={ratios_i};
const T1={t1},T2={t2},T3={t3};

Chart.defaults.color='#94a3b8';Chart.defaults.borderColor='#334155';

function mkBar(id,datasets,labels){{
  new Chart(document.getElementById(id),{{type:'bar',data:{{labels:labels||LABELS,datasets}},options:{{responsive:true,plugins:{{legend:{{position:'top'}}}},scales:{{y:{{beginAtZero:true}}}}}}}});
}}

mkBar('chartGlint',[
  {{label:'B1 Baseline',data:B1G,backgroundColor:'rgba(99,102,241,0.7)'}},
  {{label:'B2 Unmapped',data:B2G,backgroundColor:'rgba(239,68,68,0.7)'}},
  {{label:'B3 Mapped',data:B3G,backgroundColor:'rgba(16,185,129,0.7)'}}
]);
mkBar('chartInternal',[
  {{label:'B1 Baseline',data:B1I,backgroundColor:'rgba(99,102,241,0.7)'}},
  {{label:'B2 Unmapped',data:B2I,backgroundColor:'rgba(239,68,68,0.7)'}},
  {{label:'B3 Mapped',data:B3I,backgroundColor:'rgba(16,185,129,0.7)'}}
]);
mkBar('chartRatio',[
  {{label:'GlintR100 B3/B1 %',data:RATG,backgroundColor:'rgba(245,158,11,0.7)'}},
  {{label:'ArcFace B3/B1 %',data:RATI,backgroundColor:'rgba(139,92,246,0.7)'}}
]);
mkBar('chartTime',[
  {{label:'B1 Time',data:T1,backgroundColor:'rgba(99,102,241,0.6)'}},
  {{label:'B2 Time',data:T2,backgroundColor:'rgba(239,68,68,0.6)'}},
  {{label:'B3 Time',data:T3,backgroundColor:'rgba(16,185,129,0.6)'}}
]);

let mode='grid';
function setMode(m){{
  mode=m;
  document.getElementById('gridBtn').classList.toggle('active',m==='grid');
  document.getElementById('singleBtn').classList.toggle('active',m==='single');
  filterIdentity();
}}
function filterIdentity(){{
  const sel=document.getElementById('idSelect').value;
  document.querySelectorAll('.identity-panel').forEach((p,i)=>{{
    if(sel==='all'){{p.classList.remove('hidden')}}
    else{{p.classList.toggle('hidden',String(i)!==sel)}}
  }});
}}
</script></body></html>'''
    return html


def main():
    print("=" * 60)
    print("  Emb2Face Reconstruction Dashboard Generator")
    print("  READ-ONLY — does NOT modify any existing outputs")
    print("=" * 60)
    
    # Verify source data exists
    for name, d in DIRS.items():
        if not d.exists():
            print(f"ERROR: Missing directory: {d}")
            sys.exit(1)
        count = len(list(d.glob("*.png")))
        print(f"  ✓ {name}: {count} images")
    
    # Build data
    print("\n📊 Loading metrics...")
    identities = build_identity_data()
    summary = compute_summary(identities)
    print(f"  ✓ Loaded {len(identities)} identities")
    print(f"  ✓ B1 mean (Glint): {fmt(summary['b1_glint'])}")
    print(f"  ✓ B3 mean (Glint): {fmt(summary['b3_glint'])}")
    print(f"  ✓ B3/B1 ratio (Glint): {pct(summary['ratio_glint'])}")
    
    # Generate HTML
    print("\n🎨 Generating dashboard...")
    html = generate_html(identities, summary)
    
    # Write output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "reconstruction_dashboard.html"
    with open(out_path, "w") as f:
        f.write(html)
    
    size_kb = out_path.stat().st_size / 1024
    print(f"\n✅ Dashboard saved: {out_path}")
    print(f"   Size: {size_kb:.0f} KB")
    print(f"\n🌐 Open in browser:")
    print(f"   file://{out_path}")
    print(f"\n   Or serve locally:")
    print(f"   python3 -m http.server 8080 -d {OUT_DIR}")
    print(f"   Then open: http://localhost:8080/reconstruction_dashboard.html")
    
    # Streamlit note
    print(f"\n📝 Note: Streamlit was not available.")
    print(f"   To install: pip install streamlit")
    print(f"\n{'=' * 60}")
    print(f"  Dashboard generation complete — no files were modified")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

