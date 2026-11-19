import os
import sys
import json
import numpy as np
from itertools import combinations

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)

EVAL_DIR = 'outputs/evaluation'

EVALUATORS = {
    'glintr100': os.path.join(EVAL_DIR, 'biometric_embeddings_glintr100.npz'),
    'arcface': os.path.join(EVAL_DIR, 'biometric_embeddings_arcface_internal.npz'),
}


def load_original_embeddings(npz_path):
    """Load only original-image embeddings, grouped by identity."""
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data['embeddings']
    names = data['names']
    identities = data['identities']
    methods = data['methods']

    # Filter to originals only
    mask = (methods == 'original')
    orig_embs = embeddings[mask]
    orig_names = names[mask]
    orig_ids = identities[mask]

    # Group by identity
    id_to_indices = {}
    for i, identity in enumerate(orig_ids):
        id_to_indices.setdefault(str(identity), []).append(i)

    return orig_embs, orig_names, orig_ids, id_to_indices


def compute_genuine_scores(embeddings, id_to_indices):
    """Compute genuine scores: cosine between different images of the same identity."""
    scores = []
    for identity, indices in sorted(id_to_indices.items()):
        for i, j in combinations(indices, 2):
            cos = float(np.dot(embeddings[i], embeddings[j]))
            scores.append(cos)
    return np.array(scores, dtype=np.float64)


def compute_impostor_scores_primary(embeddings, id_to_indices):
    """Compute impostor scores: 1 representative image (img0) per identity.

    Uses the first image of each identity.
    Pairs: C(100, 2) = 4,950.
    """
    # Get first image per identity
    rep_indices = []
    rep_ids = []
    for identity in sorted(id_to_indices.keys()):
        rep_indices.append(id_to_indices[identity][0])  # img0
        rep_ids.append(identity)

    scores = []
    for i, j in combinations(range(len(rep_indices)), 2):
        cos = float(np.dot(embeddings[rep_indices[i]], embeddings[rep_indices[j]]))
        scores.append(cos)
    return np.array(scores, dtype=np.float64)


def threshold_sweep(genuine_scores, impostor_scores, n_steps=2000):
    """Sweep cosine threshold and compute FAR/FRR at each point."""
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    t_min = float(all_scores.min()) - 0.01
    t_max = float(all_scores.max()) + 0.01
    thresholds = np.linspace(t_min, t_max, n_steps)

    n_gen = len(genuine_scores)
    n_imp = len(impostor_scores)

    # Sort scores for efficient counting
    gen_sorted = np.sort(genuine_scores)
    imp_sorted = np.sort(impostor_scores)

    far_arr = np.zeros(n_steps)
    frr_arr = np.zeros(n_steps)

    for k, t in enumerate(thresholds):
        # FAR = fraction of impostor scores >= threshold (falsely accepted)
        n_fa = n_imp - np.searchsorted(imp_sorted, t, side='left')
        far_arr[k] = n_fa / n_imp

        # FRR = fraction of genuine scores < threshold (falsely rejected)
        n_fr = np.searchsorted(gen_sorted, t, side='left')
        frr_arr[k] = n_fr / n_gen

    return thresholds, far_arr, frr_arr


def compute_eer(thresholds, far_arr, frr_arr):
    """Find EER: threshold where FAR ≈ FRR (linear interpolation)."""
    diff = far_arr - frr_arr

    # Find the crossing point where diff changes sign
    for i in range(len(diff) - 1):
        if diff[i] >= 0 and diff[i + 1] < 0:
            # Linear interpolation
            alpha = diff[i] / (diff[i] - diff[i + 1])
            eer_threshold = thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i])
            eer_value = far_arr[i] + alpha * (far_arr[i + 1] - far_arr[i])
            return float(eer_value), float(eer_threshold)

    # Fallback: find minimum absolute difference
    idx = np.argmin(np.abs(diff))
    return float((far_arr[idx] + frr_arr[idx]) / 2), float(thresholds[idx])


def find_threshold_at_far(thresholds, far_arr, target_far):
    """Find the highest threshold where FAR <= target_far."""
    # FAR decreases as threshold increases
    valid = np.where(far_arr <= target_far)[0]
    if len(valid) == 0:
        return float(thresholds[-1])  # strictest threshold
    return float(thresholds[valid[0]])


def process_evaluator(eval_name, npz_path):
    """Run full Phase 3 pipeline for one evaluator."""
    print(f"\n{'='*60}")
    print(f"Evaluator: {eval_name}")
    print(f"{'='*60}")

    # Load original embeddings
    print("\n1. Loading original embeddings...")
    embeddings, names, ids, id_to_indices = load_original_embeddings(npz_path)
    n_ids = len(id_to_indices)
    n_imgs = len(embeddings)
    print(f"   {n_imgs} original embeddings, {n_ids} identities")
    for identity, indices in sorted(id_to_indices.items())[:3]:
        print(f"   {identity}: {len(indices)} images")
    print(f"   ...")

    # Genuine scores
    print("\n2. Computing genuine scores...")
    genuine = compute_genuine_scores(embeddings, id_to_indices)
    print(f"   {len(genuine)} genuine pairs (expected 1,000)")
    print(f"   Mean: {genuine.mean():.4f}, Std: {genuine.std():.4f}")
    print(f"   Min:  {genuine.min():.4f}, Max: {genuine.max():.4f}")

    # Impostor scores (primary: 1 per identity)
    print("\n3. Computing impostor scores (primary: 1 image per identity)...")
    impostor = compute_impostor_scores_primary(embeddings, id_to_indices)
    print(f"   {len(impostor)} impostor pairs (expected 4,950)")
    print(f"   Mean: {impostor.mean():.4f}, Std: {impostor.std():.4f}")
    print(f"   Min:  {impostor.min():.4f}, Max: {impostor.max():.4f}")

    # Sanity check: genuine mean >> impostor mean
    gap = genuine.mean() - impostor.mean()
    print(f"\n   Genuine-impostor gap: {gap:.4f}")
    if gap < 0.1:
        print(f"   WARNING: Gap is small — possible data issue.")
    else:
        print(f"   OK — Clear separation between genuine and impostor.")

    # Threshold sweep
    print("\n4. Threshold sweep...")
    thresholds, far_arr, frr_arr = threshold_sweep(genuine, impostor)
    print(f"   Swept {len(thresholds)} thresholds from {thresholds[0]:.3f} to {thresholds[-1]:.3f}")

    # EER
    print("\n5. Computing EER...")
    eer, eer_threshold = compute_eer(thresholds, far_arr, frr_arr)
    print(f"   EER:           {eer*100:.2f}%")
    print(f"   EER threshold: {eer_threshold:.4f}")

    # Operating thresholds
    print("\n6. Computing operating thresholds...")
    t_far_1pct = find_threshold_at_far(thresholds, far_arr, 0.01)
    t_far_01pct = find_threshold_at_far(thresholds, far_arr, 0.001)

    # Compute actual FAR/FRR at these thresholds
    def far_at_t(t):
        return float(np.sum(impostor >= t) / len(impostor))
    def frr_at_t(t):
        return float(np.sum(genuine < t) / len(genuine))

    print(f"   FAR=1%  threshold: {t_far_1pct:.4f}  (actual FAR={far_at_t(t_far_1pct)*100:.2f}%, FRR={frr_at_t(t_far_1pct)*100:.2f}%)")
    print(f"   FAR=0.1% threshold: {t_far_01pct:.4f}  (actual FAR={far_at_t(t_far_01pct)*100:.2f}%, FRR={frr_at_t(t_far_01pct)*100:.2f}%)")

    # Save outputs
    print("\n7. Saving outputs...")
    suffix = eval_name

    gen_path = os.path.join(EVAL_DIR, f'biometric_genuine_scores_{suffix}.npy')
    imp_path = os.path.join(EVAL_DIR, f'biometric_impostor_scores_{suffix}.npy')
    thresh_path = os.path.join(EVAL_DIR, f'biometric_thresholds_{suffix}.json')

    np.save(gen_path, genuine)
    print(f"   Saved: {gen_path} ({len(genuine)} scores)")

    np.save(imp_path, impostor)
    print(f"   Saved: {imp_path} ({len(impostor)} scores)")

    thresh_data = {
        'eer': round(eer, 6),
        'eer_pct': round(eer * 100, 2),
        'eer_threshold': round(eer_threshold, 6),
        'far_1pct_threshold': round(t_far_1pct, 6),
        'far_1pct_actual_far': round(far_at_t(t_far_1pct), 6),
        'far_1pct_actual_frr': round(frr_at_t(t_far_1pct), 6),
        'far_01pct_threshold': round(t_far_01pct, 6),
        'far_01pct_actual_far': round(far_at_t(t_far_01pct), 6),
        'far_01pct_actual_frr': round(frr_at_t(t_far_01pct), 6),
        'genuine_count': len(genuine),
        'impostor_count': len(impostor),
        'genuine_mean': round(float(genuine.mean()), 6),
        'genuine_std': round(float(genuine.std()), 6),
        'genuine_min': round(float(genuine.min()), 6),
        'genuine_max': round(float(genuine.max()), 6),
        'impostor_mean': round(float(impostor.mean()), 6),
        'impostor_std': round(float(impostor.std()), 6),
        'impostor_min': round(float(impostor.min()), 6),
        'impostor_max': round(float(impostor.max()), 6),
    }

    with open(thresh_path, 'w') as f:
        json.dump(thresh_data, f, indent=2)
    print(f"   Saved: {thresh_path}")

    return thresh_data


def main():
    print("=" * 60)
    print("Phase 3 — Bona Fide Score Distributions")
    print("=" * 60)

    all_results = {}
    for eval_name, npz_path in EVALUATORS.items():
        result = process_evaluator(eval_name, npz_path)
        all_results[eval_name] = result

    # Final summary
    print(f"\n{'='*60}")
    print("Phase 3 Summary")
    print(f"{'='*60}")
    print(f"\n{'Metric':<30} {'GlintR100':>15} {'ArcFace':>15}")
    print("-" * 60)

    g = all_results['glintr100']
    a = all_results['arcface']

    print(f"{'Genuine pairs':<30} {g['genuine_count']:>15} {a['genuine_count']:>15}")
    print(f"{'Impostor pairs':<30} {g['impostor_count']:>15} {a['impostor_count']:>15}")
    print(f"{'Genuine mean':<30} {g['genuine_mean']:>15.4f} {a['genuine_mean']:>15.4f}")
    print(f"{'Impostor mean':<30} {g['impostor_mean']:>15.4f} {a['impostor_mean']:>15.4f}")
    print(f"{'EER':<30} {g['eer_pct']:>14.2f}% {a['eer_pct']:>14.2f}%")
    print(f"{'EER threshold':<30} {g['eer_threshold']:>15.4f} {a['eer_threshold']:>15.4f}")
    print(f"{'FAR=1% threshold':<30} {g['far_1pct_threshold']:>15.4f} {a['far_1pct_threshold']:>15.4f}")
    print(f"{'FAR=0.1% threshold':<30} {g['far_01pct_threshold']:>15.4f} {a['far_01pct_threshold']:>15.4f}")

    # Overall sanity check
    ok = True
    for name, r in all_results.items():
        if r['genuine_mean'] <= r['impostor_mean']:
            print(f"\nFAIL: {name} genuine mean <= impostor mean")
            ok = False
        if r['eer'] > 0.15:
            print(f"\nWARN: {name} EER > 15% — unusually high")

    print(f"\n{'='*60}")
    if ok:
        print("Phase 3 RESULT: ✅ PASS")
        print("Safe to proceed to Phase 4 (Attack Score Computation).")
    else:
        print("Phase 3 RESULT: ❌ FAIL — Issues found above.")
    print(f"{'='*60}")

    # Save combined results
    all_results['overall'] = 'PASS' if ok else 'FAIL'
    combined_path = os.path.join(EVAL_DIR, 'phase3_bonafide_results.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to: {combined_path}")

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
