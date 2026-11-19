import os
import sys
import json
import csv
import numpy as np

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)

EVAL_DIR = 'outputs/evaluation'

EVALUATORS = {
    'glintr100': {
        'embeddings': os.path.join(EVAL_DIR, 'biometric_embeddings_glintr100.npz'),
        'thresholds': os.path.join(EVAL_DIR, 'biometric_thresholds_glintr100.json'),
        'csv_ref': os.path.join(EVAL_DIR, 'full_official_eval_aligned.csv'),
        'csv_cosine_col': 'cosine_naive_resize',
    },
    'arcface': {
        'embeddings': os.path.join(EVAL_DIR, 'biometric_embeddings_arcface_internal.npz'),
        'thresholds': os.path.join(EVAL_DIR, 'biometric_thresholds_arcface.json'),
        'csv_ref': os.path.join(EVAL_DIR, 'full_official_eval_arcface_internal.csv'),
        'csv_cosine_col': 'cosine_internal_naive',
    },
}

METHOD_LABELS = {
    'baseline': 'b1',
    'unmapped': 'b2',
    'mapped':   'b3',
}


def load_embeddings_indexed(npz_path):
    """Load embeddings and build lookup indices.

    Returns:
        embeddings: (N, 512) array
        lookup: dict mapping (method, filename) -> index
        id_to_images: dict mapping identity -> list of (filename, index) for originals
    """
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data['embeddings']
    names = data['names']
    identities = data['identities']
    methods = data['methods']

    lookup = {}
    id_to_images = {}

    for i in range(len(names)):
        method = str(methods[i])
        name = str(names[i])
        identity = str(identities[i])
        lookup[(method, name)] = i

        if method == 'original':
            if identity not in id_to_images:
                id_to_images[identity] = []
            id_to_images[identity].append((name, i))

    # Sort each identity's images by filename for deterministic img0/img1/... ordering
    for identity in id_to_images:
        id_to_images[identity].sort(key=lambda x: x[0])

    return embeddings, lookup, id_to_images


def compute_same_image_scores(embeddings, lookup):
    """Protocol A: same-image attack. original_i vs B1/B2/B3(original_i)."""
    # Get all original filenames
    orig_names = sorted(set(
        name for (method, name) in lookup if method == 'original'
    ))

    rows = []
    for fname in orig_names:
        orig_idx = lookup[('original', fname)]
        identity = fname.rsplit('_img', 1)[0]

        b1_idx = lookup.get(('baseline', fname))
        b2_idx = lookup.get(('unmapped', fname))
        b3_idx = lookup.get(('mapped', fname))

        b1_score = float(np.dot(embeddings[orig_idx], embeddings[b1_idx])) if b1_idx is not None else None
        b2_score = float(np.dot(embeddings[orig_idx], embeddings[b2_idx])) if b2_idx is not None else None
        b3_score = float(np.dot(embeddings[orig_idx], embeddings[b3_idx])) if b3_idx is not None else None

        rows.append({
            'image_name': fname,
            'identity': identity,
            'b1_score': b1_score,
            'b2_score': b2_score,
            'b3_score': b3_score,
        })

    return rows


def compute_cross_image_scores(embeddings, lookup, id_to_images):
    """Protocol B: cross-image attack. enrollment=img0, attack=img1-4."""
    rows = []

    for identity in sorted(id_to_images.keys()):
        images = id_to_images[identity]  # sorted list of (filename, index)
        if len(images) < 2:
            continue

        # img0 = enrollment
        enrollment_name, enrollment_idx = images[0]

        # img1-4 = attack sources
        for attack_name, _ in images[1:]:
            b1_idx = lookup.get(('baseline', attack_name))
            b2_idx = lookup.get(('unmapped', attack_name))
            b3_idx = lookup.get(('mapped', attack_name))

            b1_score = float(np.dot(embeddings[enrollment_idx], embeddings[b1_idx])) if b1_idx is not None else None
            b2_score = float(np.dot(embeddings[enrollment_idx], embeddings[b2_idx])) if b2_idx is not None else None
            b3_score = float(np.dot(embeddings[enrollment_idx], embeddings[b3_idx])) if b3_idx is not None else None

            rows.append({
                'enrollment_image': enrollment_name,
                'attack_source_image': attack_name,
                'identity': identity,
                'b1_score': b1_score,
                'b2_score': b2_score,
                'b3_score': b3_score,
            })

    return rows


def save_csv(rows, path, fieldnames):
    """Save rows to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for k, v in row.items():
                out[k] = f'{v:.6f}' if isinstance(v, float) else v
            writer.writerow(out)


def compute_iapmr(scores, threshold):
    """Compute IAPMR: fraction of scores >= threshold."""
    arr = np.array(scores, dtype=np.float64)
    return float(np.sum(arr >= threshold) / len(arr))


def score_stats(scores):
    """Compute basic statistics for a score array."""
    arr = np.array(scores, dtype=np.float64)
    return {
        'mean': round(float(arr.mean()), 4),
        'std': round(float(arr.std()), 4),
        'min': round(float(arr.min()), 4),
        'max': round(float(arr.max()), 4),
        'count': len(arr),
    }


def csv_spot_check(same_image_rows, csv_path, cosine_col, method_name, score_key):
    """Spot-check Protocol A scores against existing CSV."""
    if not os.path.isfile(csv_path):
        return None

    # Build reference from CSV
    csv_ref = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['method'].strip() == method_name:
                img = row['image_name'].strip()
                val = row[cosine_col].strip()
                if val and val != 'N/A':
                    csv_ref[img] = float(val)

    if not csv_ref:
        return None

    diffs = []
    for r in same_image_rows:
        fname = r['image_name']
        computed = r[score_key]
        if fname in csv_ref and computed is not None:
            diffs.append(abs(computed - csv_ref[fname]))

    if not diffs:
        return None

    max_diff = max(diffs)
    return {
        'checked': len(diffs),
        'max_diff': round(max_diff, 8),
        'match': max_diff < 0.001,
    }


def process_evaluator(eval_name, cfg):
    """Run full Phase 4 for one evaluator."""
    print(f"\n{'='*60}")
    print(f"Evaluator: {eval_name}")
    print(f"{'='*60}")

    # Load embeddings
    print("\n1. Loading embeddings...")
    embeddings, lookup, id_to_images = load_embeddings_indexed(cfg['embeddings'])
    n_ids = len(id_to_images)
    n_orig = sum(len(v) for v in id_to_images.values())
    print(f"   {len(embeddings)} total embeddings, {n_orig} originals, {n_ids} identities")

    # Load thresholds
    print("\n2. Loading thresholds from Phase 3...")
    with open(cfg['thresholds']) as f:
        thresholds = json.load(f)
    print(f"   EER threshold:     {thresholds['eer_threshold']:.4f}")
    print(f"   FAR=1% threshold:  {thresholds['far_1pct_threshold']:.4f}")
    print(f"   FAR=0.1% threshold:{thresholds['far_01pct_threshold']:.4f}")

    # Protocol A: same-image
    print("\n3. Computing Protocol A (same-image) attack scores...")
    same_rows = compute_same_image_scores(embeddings, lookup)
    print(f"   {len(same_rows)} same-image attack pairs")

    same_path = os.path.join(EVAL_DIR, f'biometric_attack_same_image_{eval_name}.csv')
    save_csv(same_rows, same_path, ['image_name', 'identity', 'b1_score', 'b2_score', 'b3_score'])
    print(f"   Saved: {same_path}")

    # Protocol B: cross-image
    print("\n4. Computing Protocol B (cross-image) attack scores...")
    cross_rows = compute_cross_image_scores(embeddings, lookup, id_to_images)
    print(f"   {len(cross_rows)} cross-image attack pairs")

    cross_path = os.path.join(EVAL_DIR, f'biometric_attack_cross_image_{eval_name}.csv')
    save_csv(cross_rows, cross_path,
             ['enrollment_image', 'attack_source_image', 'identity', 'b1_score', 'b2_score', 'b3_score'])
    print(f"   Saved: {cross_path}")

    # Score statistics
    print("\n5. Score statistics...")
    results = {'protocol_a': {}, 'protocol_b': {}}

    for label, score_key in [('b1', 'b1_score'), ('b2', 'b2_score'), ('b3', 'b3_score')]:
        same_scores = [r[score_key] for r in same_rows if r[score_key] is not None]
        cross_scores = [r[score_key] for r in cross_rows if r[score_key] is not None]

        s_a = score_stats(same_scores)
        s_b = score_stats(cross_scores)

        results['protocol_a'][label] = s_a
        results['protocol_b'][label] = s_b

        print(f"   {label.upper()} Protocol A: mean={s_a['mean']:.4f}, std={s_a['std']:.4f}, "
              f"min={s_a['min']:.4f}, max={s_a['max']:.4f} (n={s_a['count']})")
        print(f"   {label.upper()} Protocol B: mean={s_b['mean']:.4f}, std={s_b['std']:.4f}, "
              f"min={s_b['min']:.4f}, max={s_b['max']:.4f} (n={s_b['count']})")

    # IAPMR computation
    print("\n6. Computing IAPMR (attack success rate)...")
    threshold_points = {
        'eer': thresholds['eer_threshold'],
        'far_1pct': thresholds['far_1pct_threshold'],
        'far_01pct': thresholds['far_01pct_threshold'],
    }

    iapmr = {'protocol_a': {}, 'protocol_b': {}}

    for tp_name, tp_val in threshold_points.items():
        iapmr['protocol_a'][tp_name] = {}
        iapmr['protocol_b'][tp_name] = {}

        for label, score_key in [('b1', 'b1_score'), ('b2', 'b2_score'), ('b3', 'b3_score')]:
            same_scores = [r[score_key] for r in same_rows if r[score_key] is not None]
            cross_scores = [r[score_key] for r in cross_rows if r[score_key] is not None]

            ia_a = compute_iapmr(same_scores, tp_val)
            ia_b = compute_iapmr(cross_scores, tp_val)

            iapmr['protocol_a'][tp_name][label] = round(ia_a, 6)
            iapmr['protocol_b'][tp_name][label] = round(ia_b, 6)

    # Print IAPMR tables
    for proto_label, proto_key in [('Protocol A (same-image)', 'protocol_a'),
                                    ('Protocol B (cross-image)', 'protocol_b')]:
        print(f"\n   {proto_label}:")
        print(f"   {'Threshold':<20} {'B1 IAPMR':>10} {'B2 IAPMR':>10} {'B3 IAPMR':>10}")
        print(f"   {'-'*50}")
        for tp_name, tp_val in threshold_points.items():
            b1 = iapmr[proto_key][tp_name]['b1'] * 100
            b2 = iapmr[proto_key][tp_name]['b2'] * 100
            b3 = iapmr[proto_key][tp_name]['b3'] * 100
            print(f"   {tp_name:<20} {b1:>9.1f}% {b2:>9.1f}% {b3:>9.1f}%")

    # CSV spot-check for Protocol A
    print("\n7. CSV spot-check (Protocol A vs existing evaluation CSVs)...")
    csv_checks = {}
    for method_csv, score_key in [('baseline', 'b1_score'), ('unmapped', 'b2_score'), ('mapped', 'b3_score')]:
        check = csv_spot_check(same_rows, cfg['csv_ref'], cfg['csv_cosine_col'], method_csv, score_key)
        if check:
            status = "OK" if check['match'] else "DIFF"
            print(f"   {status}  {method_csv}: checked {check['checked']}, max_diff={check['max_diff']:.8f}")
            csv_checks[method_csv] = check
        else:
            print(f"   SKIP  {method_csv}: no reference data")

    # Save IAPMR JSON
    iapmr_path = os.path.join(EVAL_DIR, f'biometric_iapmr_{eval_name}.json')
    iapmr['thresholds_used'] = threshold_points
    iapmr['score_stats'] = results
    iapmr['csv_spot_checks'] = csv_checks
    with open(iapmr_path, 'w') as f:
        json.dump(iapmr, f, indent=2)
    print(f"\n   Saved: {iapmr_path}")

    return iapmr, results


def main():
    print("=" * 60)
    print("Phase 4 — Attack Score Computation")
    print("=" * 60)

    all_results = {}
    for eval_name, cfg in EVALUATORS.items():
        iapmr, stats = process_evaluator(eval_name, cfg)
        all_results[eval_name] = {'iapmr': iapmr, 'stats': stats}

    # Sanity checks
    print(f"\n{'='*60}")
    print("Sanity Checks")
    print(f"{'='*60}")
    ok = True

    for eval_name, data in all_results.items():
        for proto in ['protocol_a', 'protocol_b']:
            stats = data['stats'][proto]
            # B1 mean > B3 mean > B2 mean
            if stats['b1']['mean'] < stats['b3']['mean']:
                print(f"  WARN  {eval_name} {proto}: B1 mean ({stats['b1']['mean']}) < B3 mean ({stats['b3']['mean']})")
            if stats['b3']['mean'] < stats['b2']['mean']:
                print(f"  FAIL  {eval_name} {proto}: B3 mean ({stats['b3']['mean']}) < B2 mean ({stats['b2']['mean']})")
                ok = False

            # B2 should be near zero
            if abs(stats['b2']['mean']) > 0.15:
                print(f"  WARN  {eval_name} {proto}: B2 mean ({stats['b2']['mean']}) is not near zero")

        # Protocol B scores should generally be <= Protocol A
        for label in ['b1', 'b3']:
            a_mean = data['stats']['protocol_a'][label]['mean']
            b_mean = data['stats']['protocol_b'][label]['mean']
            if b_mean > a_mean + 0.05:
                print(f"  WARN  {eval_name}: Protocol B {label} mean ({b_mean}) > Protocol A ({a_mean})")

    if ok:
        print("  All sanity checks passed.")

    # Save combined results
    print(f"\n{'='*60}")
    if ok:
        print("Phase 4 RESULT: ✅ PASS")
        print("Safe to proceed to Phase 5 (Result Validation).")
    else:
        print("Phase 4 RESULT: ❌ FAIL — Issues found above.")
    print(f"{'='*60}")

    combined = {eval_name: data['iapmr'] for eval_name, data in all_results.items()}
    combined['overall'] = 'PASS' if ok else 'FAIL'
    combined_path = os.path.join(EVAL_DIR, 'phase4_attack_results.json')
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results saved to: {combined_path}")

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
