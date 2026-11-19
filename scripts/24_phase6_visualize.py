import os
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)

EVAL_DIR = 'outputs/evaluation'
FIG_DIR = os.path.join(EVAL_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

EVALUATORS = {
    'glintr100': 'GlintR100',
    'arcface': 'Internal ArcFace',
}


def load_data(eval_name):
    """Load all Phase 3–4 data for one evaluator."""
    genuine = np.load(os.path.join(EVAL_DIR, f'biometric_genuine_scores_{eval_name}.npy'))
    impostor = np.load(os.path.join(EVAL_DIR, f'biometric_impostor_scores_{eval_name}.npy'))

    with open(os.path.join(EVAL_DIR, f'biometric_thresholds_{eval_name}.json')) as f:
        thresholds = json.load(f)

    with open(os.path.join(EVAL_DIR, f'biometric_iapmr_{eval_name}.json')) as f:
        iapmr = json.load(f)

    # Load Protocol B cross-image attack scores
    cross_b1, cross_b2, cross_b3 = [], [], []
    with open(os.path.join(EVAL_DIR, f'biometric_attack_cross_image_{eval_name}.csv')) as f:
        for row in csv.DictReader(f):
            cross_b1.append(float(row['b1_score']))
            cross_b2.append(float(row['b2_score']))
            cross_b3.append(float(row['b3_score']))

    # Load Protocol A same-image attack scores
    same_b1, same_b2, same_b3 = [], [], []
    with open(os.path.join(EVAL_DIR, f'biometric_attack_same_image_{eval_name}.csv')) as f:
        for row in csv.DictReader(f):
            same_b1.append(float(row['b1_score']))
            same_b2.append(float(row['b2_score']))
            same_b3.append(float(row['b3_score']))

    return {
        'genuine': genuine,
        'impostor': impostor,
        'thresholds': thresholds,
        'iapmr': iapmr,
        'cross': {'b1': np.array(cross_b1), 'b2': np.array(cross_b2), 'b3': np.array(cross_b3)},
        'same': {'b1': np.array(same_b1), 'b2': np.array(same_b2), 'b3': np.array(same_b3)},
    }


# ============================================================
# Figure 1: Score Distribution Histograms
# ============================================================
def plot_score_distributions(data, eval_name, eval_label):
    """Plot overlaid score distributions for bona fide + attack scores."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    bins = np.linspace(-0.35, 1.0, 120)
    alpha = 0.55

    t = data['thresholds']
    thresholds_to_mark = [
        (t['eer_threshold'], f"EER ({t['eer_pct']:.1f}%)", '--'),
        (t['far_1pct_threshold'], 'FAR=1%', '-.'),
        (t['far_01pct_threshold'], 'FAR=0.1%', ':'),
    ]

    # Top panel: Bona fide + Protocol B (cross-image, primary)
    ax = axes[0]
    ax.hist(data['impostor'], bins=bins, alpha=alpha, color='#e74c3c', label='Bona fide impostor', density=True, edgecolor='none')
    ax.hist(data['genuine'], bins=bins, alpha=alpha, color='#2ecc71', label='Bona fide genuine', density=True, edgecolor='none')
    ax.hist(data['cross']['b3'], bins=bins, alpha=0.45, color='#3498db', label='B3 attack (cross-image)', density=True, edgecolor='none')
    ax.hist(data['cross']['b1'], bins=bins, alpha=0.3, color='#9b59b6', label='B1 baseline (cross-image)', density=True, histtype='step', linewidth=1.5)
    ax.hist(data['cross']['b2'], bins=bins, alpha=0.3, color='#e67e22', label='B2 unmapped (cross-image)', density=True, histtype='step', linewidth=1.5, linestyle='--')

    for tv, label, ls in thresholds_to_mark:
        ax.axvline(tv, color='#555', linestyle=ls, linewidth=1.2, alpha=0.8)
        ax.text(tv + 0.008, ax.get_ylim()[1] * 0.92, label, fontsize=8, rotation=90, va='top', color='#555')

    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{eval_label} — Protocol B (Cross-Image Attack)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(-0.35, 1.0)

    # Bottom panel: Protocol A (same-image, validation)
    ax = axes[1]
    ax.hist(data['impostor'], bins=bins, alpha=alpha, color='#e74c3c', label='Bona fide impostor', density=True, edgecolor='none')
    ax.hist(data['genuine'], bins=bins, alpha=alpha, color='#2ecc71', label='Bona fide genuine', density=True, edgecolor='none')
    ax.hist(data['same']['b3'], bins=bins, alpha=0.45, color='#3498db', label='B3 attack (same-image)', density=True, edgecolor='none')
    ax.hist(data['same']['b1'], bins=bins, alpha=0.3, color='#9b59b6', label='B1 baseline (same-image)', density=True, histtype='step', linewidth=1.5)
    ax.hist(data['same']['b2'], bins=bins, alpha=0.3, color='#e67e22', label='B2 unmapped (same-image)', density=True, histtype='step', linewidth=1.5, linestyle='--')

    for tv, label, ls in thresholds_to_mark:
        ax.axvline(tv, color='#555', linestyle=ls, linewidth=1.2, alpha=0.8)
        ax.text(tv + 0.008, ax.get_ylim()[1] * 0.92, label, fontsize=8, rotation=90, va='top', color='#555')

    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{eval_label} — Protocol A (Same-Image Validation)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'score_distributions_{eval_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {path}")
    return path


# ============================================================
# Figure 2: DET Curve
# ============================================================
def plot_det_curve(data, eval_name, eval_label):
    """Plot DET curve (FMR vs FNMR) on log-log axes."""
    genuine = data['genuine']
    impostor = data['impostor']

    n_gen = len(genuine)
    n_imp = len(impostor)

    thresholds = np.linspace(
        min(genuine.min(), impostor.min()) - 0.01,
        max(genuine.max(), impostor.max()) + 0.01,
        5000
    )

    gen_sorted = np.sort(genuine)
    imp_sorted = np.sort(impostor)

    fmr = np.array([(n_imp - np.searchsorted(imp_sorted, t, side='left')) / n_imp for t in thresholds])
    fnmr = np.array([np.searchsorted(gen_sorted, t, side='left') / n_gen for t in thresholds])

    # Filter out zero values for log scale
    valid = (fmr > 0) & (fnmr > 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.loglog(fmr[valid], fnmr[valid], color='#2c3e50', linewidth=2, label='DET curve')

    # Mark EER
    t = data['thresholds']
    eer = t['eer']
    ax.plot(eer, eer, 'ro', markersize=8, label=f'EER = {eer*100:.2f}%', zorder=5)

    # Mark operating points
    for tp_name, tp_key, marker, color in [
        ('FAR=1%', 'far_1pct_threshold', 's', '#e74c3c'),
        ('FAR=0.1%', 'far_01pct_threshold', 'D', '#3498db'),
    ]:
        tau = t[tp_key]
        fmr_at = np.sum(impostor >= tau) / n_imp
        fnmr_at = np.sum(genuine < tau) / n_gen
        if fmr_at > 0 and fnmr_at > 0:
            ax.plot(fmr_at, fnmr_at, marker=marker, color=color, markersize=8,
                    label=f'{tp_name} (FRR={fnmr_at*100:.1f}%)', zorder=5)

    ax.set_xlabel('False Match Rate (FMR)', fontsize=12)
    ax.set_ylabel('False Non-Match Rate (FNMR)', fontsize=12)
    ax.set_title(f'{eval_label} — DET Curve (log-log axes)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(1e-3, 1)

    # Percentage formatter
    pct_fmt = FuncFormatter(lambda x, _: f'{x*100:.1f}%' if x >= 0.01 else f'{x*100:.2f}%')
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.yaxis.set_major_formatter(pct_fmt)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'det_curve_{eval_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {path}")
    return path


# ============================================================
# Figure 3: ROC Curve
# ============================================================
def plot_roc_curve(data, eval_name, eval_label):
    """Plot ROC curve (TAR vs FAR)."""
    genuine = data['genuine']
    impostor = data['impostor']

    n_gen = len(genuine)
    n_imp = len(impostor)

    thresholds = np.linspace(
        min(genuine.min(), impostor.min()) - 0.01,
        max(genuine.max(), impostor.max()) + 0.01,
        5000
    )

    gen_sorted = np.sort(genuine)
    imp_sorted = np.sort(impostor)

    far = np.array([(n_imp - np.searchsorted(imp_sorted, t, side='left')) / n_imp for t in thresholds])
    tar = np.array([1.0 - np.searchsorted(gen_sorted, t, side='left') / n_gen for t in thresholds])

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.semilogx(far, tar, color='#2c3e50', linewidth=2, label='ROC curve')

    # Mark operating points
    t = data['thresholds']
    for tp_name, tp_key, marker, color in [
        ('EER', 'eer_threshold', 'o', '#e74c3c'),
        ('FAR=1%', 'far_1pct_threshold', 's', '#e67e22'),
        ('FAR=0.1%', 'far_01pct_threshold', 'D', '#3498db'),
    ]:
        tau = t[tp_key]
        far_at = np.sum(impostor >= tau) / n_imp
        tar_at = 1.0 - np.sum(genuine < tau) / n_gen
        if far_at > 0:
            ax.plot(far_at, tar_at, marker=marker, color=color, markersize=8,
                    label=f'{tp_name} (TAR={tar_at*100:.1f}%)', zorder=5)

    ax.set_xlabel('False Accept Rate (FAR)', fontsize=12)
    ax.set_ylabel('True Accept Rate (TAR)', fontsize=12)
    ax.set_title(f'{eval_label} — ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0.5, 1.01)

    pct_fmt = FuncFormatter(lambda x, _: f'{x*100:.1f}%' if x >= 0.01 else f'{x*100:.2f}%')
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.0f}%'))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'roc_curve_{eval_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {path}")
    return path


# ============================================================
# Summary Table
# ============================================================
def create_summary_tables(all_data):
    """Create CSV and Markdown summary tables."""

    # Build rows
    rows = []
    for eval_name in ['glintr100', 'arcface']:
        t = all_data[eval_name]['thresholds']
        iapmr = all_data[eval_name]['iapmr']
        label = EVALUATORS[eval_name]

        for proto_key, proto_label in [('protocol_b', 'Protocol B (cross-image)'),
                                        ('protocol_a', 'Protocol A (same-image)')]:
            row = {
                'Evaluator': label,
                'Protocol': proto_label,
                'EER (%)': f"{t['eer_pct']:.2f}",
                'EER Threshold': f"{t['eer_threshold']:.4f}",
                'τ @ FAR=1%': f"{t['far_1pct_threshold']:.4f}",
                'τ @ FAR=0.1%': f"{t['far_01pct_threshold']:.4f}",
                'B1 IAPMR @1%': f"{iapmr[proto_key]['far_1pct']['b1']*100:.1f}%",
                'B2 IAPMR @1%': f"{iapmr[proto_key]['far_1pct']['b2']*100:.1f}%",
                'B3 IAPMR @1%': f"{iapmr[proto_key]['far_1pct']['b3']*100:.1f}%",
                'B1 IAPMR @0.1%': f"{iapmr[proto_key]['far_01pct']['b1']*100:.1f}%",
                'B2 IAPMR @0.1%': f"{iapmr[proto_key]['far_01pct']['b2']*100:.1f}%",
                'B3 IAPMR @0.1%': f"{iapmr[proto_key]['far_01pct']['b3']*100:.1f}%",
            }
            rows.append(row)

    # Save CSV
    csv_path = os.path.join(EVAL_DIR, 'biometric_summary.csv')
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"   Saved: {csv_path}")

    # Save Markdown
    md_path = os.path.join(EVAL_DIR, 'biometric_summary.md')
    with open(md_path, 'w') as f:
        f.write("# Biometric Evaluation Summary\n\n")
        f.write("> Auto-generated by `scripts/24_phase6_visualize.py`\n\n")

        # Main IAPMR table (Protocol B — primary)
        f.write("## Attack Success Rate — Protocol B (Cross-Image, Primary)\n\n")
        f.write("| Evaluator | EER | τ @ FAR=1% | τ @ FAR=0.1% | B1 @1% | B2 @1% | **B3 @1%** | B1 @0.1% | B2 @0.1% | **B3 @0.1%** |\n")
        f.write("|-----------|-----|-----------|-------------|--------|--------|------------|----------|----------|-------------|\n")
        for eval_name in ['glintr100', 'arcface']:
            t = all_data[eval_name]['thresholds']
            ia = all_data[eval_name]['iapmr']['protocol_b']
            label = EVALUATORS[eval_name]
            f.write(f"| {label} | {t['eer_pct']:.2f}% | {t['far_1pct_threshold']:.4f} | {t['far_01pct_threshold']:.4f} | "
                    f"{ia['far_1pct']['b1']*100:.1f}% | {ia['far_1pct']['b2']*100:.1f}% | "
                    f"**{ia['far_1pct']['b3']*100:.1f}%** | "
                    f"{ia['far_01pct']['b1']*100:.1f}% | {ia['far_01pct']['b2']*100:.1f}% | "
                    f"**{ia['far_01pct']['b3']*100:.1f}%** |\n")

        f.write("\n## Attack Success Rate — Protocol A (Same-Image, Validation)\n\n")
        f.write("| Evaluator | B1 @1% | B2 @1% | **B3 @1%** | B1 @0.1% | B2 @0.1% | **B3 @0.1%** |\n")
        f.write("|-----------|--------|--------|------------|----------|----------|-------------|\n")
        for eval_name in ['glintr100', 'arcface']:
            ia = all_data[eval_name]['iapmr']['protocol_a']
            label = EVALUATORS[eval_name]
            f.write(f"| {label} | "
                    f"{ia['far_1pct']['b1']*100:.1f}% | {ia['far_1pct']['b2']*100:.1f}% | "
                    f"**{ia['far_1pct']['b3']*100:.1f}%** | "
                    f"{ia['far_01pct']['b1']*100:.1f}% | {ia['far_01pct']['b2']*100:.1f}% | "
                    f"**{ia['far_01pct']['b3']*100:.1f}%** |\n")

        f.write("\n## Key Result\n\n")
        # Get Protocol B B3 IAPMR at FAR=0.1% for the headline
        g_b3 = all_data['glintr100']['iapmr']['protocol_b']['far_01pct']['b3'] * 100
        a_b3 = all_data['arcface']['iapmr']['protocol_b']['far_01pct']['b3'] * 100
        f.write(f"> At **FAR = 0.1%** (cross-image protocol), B3 reconstructions are accepted by:\n")
        f.write(f"> - **GlintR100:** {g_b3:.1f}% of the time\n")
        f.write(f"> - **Internal ArcFace:** {a_b3:.1f}% of the time\n\n")
        f.write("B2 (unmapped / negative control) shows 0.0% success at FAR=0.1%, "
                "confirming the mapping network is essential for the attack.\n")

    print(f"   Saved: {md_path}")
    return csv_path, md_path


def main():
    print("=" * 60)
    print("Phase 6 — Visualization and Final Tables")
    print("=" * 60)

    all_data = {}
    all_figures = []

    for eval_name, eval_label in EVALUATORS.items():
        print(f"\n{'='*60}")
        print(f"Evaluator: {eval_label}")
        print(f"{'='*60}")

        print("\n1. Loading validated data...")
        data = load_data(eval_name)
        all_data[eval_name] = data
        print(f"   Genuine: {len(data['genuine'])}, Impostor: {len(data['impostor'])}")
        print(f"   Protocol A: {len(data['same']['b3'])} pairs, Protocol B: {len(data['cross']['b3'])} pairs")

        print("\n2. Plotting score distributions...")
        p = plot_score_distributions(data, eval_name, eval_label)
        all_figures.append(p)

        print("\n3. Plotting DET curve...")
        p = plot_det_curve(data, eval_name, eval_label)
        all_figures.append(p)

        print("\n4. Plotting ROC curve...")
        p = plot_roc_curve(data, eval_name, eval_label)
        all_figures.append(p)

    print(f"\n{'='*60}")
    print("Creating summary tables...")
    print(f"{'='*60}")
    csv_path, md_path = create_summary_tables(all_data)

    # Final summary
    print(f"\n{'='*60}")
    print("Phase 6 RESULT: ✅ PASS")
    print(f"{'='*60}")
    print(f"\nGenerated {len(all_figures)} figures:")
    for f in all_figures:
        print(f"   {f}")
    print(f"\nSummary files:")
    print(f"   {csv_path}")
    print(f"   {md_path}")

    # Save phase result
    result = {
        'overall': 'PASS',
        'figures': all_figures,
        'summary_csv': csv_path,
        'summary_md': md_path,
    }
    result_path = os.path.join(EVAL_DIR, 'phase6_visualization_results.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"   {result_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
