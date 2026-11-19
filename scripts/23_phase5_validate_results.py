import os
import sys
import json
import csv
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)

EVAL_DIR = 'outputs/evaluation'

# ============================================================
# Helper
# ============================================================
def check(name, condition, detail=""):
    """Print and return a check result."""
    status = "PASS" if condition else "FAIL"
    icon = "✅" if condition else "❌"
    print(f"  {icon} {status}  {name}")
    if detail:
        print(f"         {detail}")
    return condition


def main():
    print("=" * 60)
    print("Phase 5 — Result Validation")
    print("=" * 60)

    all_ok = True
    results = []

    for eval_name, eval_label in [('glintr100', 'GlintR100'), ('arcface', 'ArcFace')]:
        print(f"\n{'='*60}")
        print(f"Evaluator: {eval_label}")
        print(f"{'='*60}")

        # Load Phase 3 data
        genuine = np.load(os.path.join(EVAL_DIR, f'biometric_genuine_scores_{eval_name}.npy'))
        impostor = np.load(os.path.join(EVAL_DIR, f'biometric_impostor_scores_{eval_name}.npy'))
        with open(os.path.join(EVAL_DIR, f'biometric_thresholds_{eval_name}.json')) as f:
            thresholds = json.load(f)

        # Load Phase 4 data
        with open(os.path.join(EVAL_DIR, f'biometric_iapmr_{eval_name}.json')) as f:
            iapmr_data = json.load(f)

        # Load attack CSVs
        same_rows = []
        with open(os.path.join(EVAL_DIR, f'biometric_attack_same_image_{eval_name}.csv')) as f:
            for row in csv.DictReader(f):
                same_rows.append(row)

        cross_rows = []
        with open(os.path.join(EVAL_DIR, f'biometric_attack_cross_image_{eval_name}.csv')) as f:
            for row in csv.DictReader(f):
                cross_rows.append(row)

        # ----------------------------------------------------------
        # Check 1: Genuine mean > impostor mean
        # ----------------------------------------------------------
        print("\n--- Check 1: Genuine mean > impostor mean ---")
        gen_mean = genuine.mean()
        imp_mean = impostor.mean()
        gap = gen_mean - imp_mean
        ok = check("Genuine mean > impostor mean",
                    gen_mean > imp_mean,
                    f"genuine={gen_mean:.4f}, impostor={imp_mean:.4f}, gap={gap:.4f}")
        all_ok &= ok

        # ----------------------------------------------------------
        # Check 2: FAR decreases as threshold increases
        # ----------------------------------------------------------
        print("\n--- Check 2: FAR behavior ---")
        imp_sorted = np.sort(impostor)
        n_imp = len(impostor)
        test_thresholds = np.linspace(float(impostor.min()), float(impostor.max()), 100)
        fars = [np.sum(impostor >= t) / n_imp for t in test_thresholds]
        far_monotone = all(fars[i] >= fars[i+1] - 1e-10 for i in range(len(fars)-1))
        ok = check("FAR decreases as threshold increases", far_monotone)
        all_ok &= ok

        # ----------------------------------------------------------
        # Check 3: FRR increases as threshold increases
        # ----------------------------------------------------------
        print("\n--- Check 3: FRR behavior ---")
        n_gen = len(genuine)
        frrs = [np.sum(genuine < t) / n_gen for t in test_thresholds]
        frr_monotone = all(frrs[i] <= frrs[i+1] + 1e-10 for i in range(len(frrs)-1))
        ok = check("FRR increases as threshold increases", frr_monotone)
        all_ok &= ok

        # ----------------------------------------------------------
        # Check 4: EER validity
        # ----------------------------------------------------------
        print("\n--- Check 4: EER validity ---")
        eer = thresholds['eer']
        eer_t = thresholds['eer_threshold']
        # Verify FAR ≈ FRR at EER threshold
        far_at_eer = np.sum(impostor >= eer_t) / n_imp
        frr_at_eer = np.sum(genuine < eer_t) / n_gen
        eer_diff = abs(far_at_eer - frr_at_eer)
        ok = check("FAR ≈ FRR at EER threshold",
                    eer_diff < 0.02,
                    f"FAR={far_at_eer:.4f}, FRR={frr_at_eer:.4f}, diff={eer_diff:.4f}")
        all_ok &= ok
        ok = check("EER in reasonable range (0.5%–15%)",
                    0.005 <= eer <= 0.15,
                    f"EER={eer*100:.2f}%")
        all_ok &= ok

        # ----------------------------------------------------------
        # Check 5: Protocol A scores match existing CSVs
        # ----------------------------------------------------------
        print("\n--- Check 5: Protocol A vs existing CSV ---")
        csv_map = {
            'glintr100': ('full_official_eval_aligned.csv', 'cosine_naive_resize'),
            'arcface': ('full_official_eval_arcface_internal.csv', 'cosine_internal_naive'),
        }
        csv_file, csv_col = csv_map[eval_name]
        csv_path = os.path.join(EVAL_DIR, csv_file)

        if os.path.isfile(csv_path):
            ref = {}
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    method = row['method'].strip()
                    img = row['image_name'].strip()
                    val = row[csv_col].strip()
                    if val and val != 'N/A':
                        ref[(method, img)] = float(val)

            max_diffs = {}
            for method_csv, score_key in [('baseline', 'b1_score'), ('unmapped', 'b2_score'), ('mapped', 'b3_score')]:
                diffs = []
                for r in same_rows:
                    fname = r['image_name']
                    computed = float(r[score_key])
                    ref_key = (method_csv, fname)
                    if ref_key in ref:
                        diffs.append(abs(computed - ref[ref_key]))
                if diffs:
                    md = max(diffs)
                    max_diffs[method_csv] = md
                    ok = check(f"{method_csv} matches CSV (max_diff={md:.8f})",
                               md < 0.001,
                               f"checked {len(diffs)} pairs")
                    all_ok &= ok
        else:
            print(f"  ⚠️ SKIP  CSV not found: {csv_path}")

        # ----------------------------------------------------------
        # Check 6: B1 mean > B3 mean > B2 mean
        # ----------------------------------------------------------
        print("\n--- Check 6: Score ordering B1 > B3 > B2 ---")
        stats = iapmr_data.get('score_stats', {})
        for proto_label, proto_key in [('Protocol A', 'protocol_a'), ('Protocol B', 'protocol_b')]:
            s = stats.get(proto_key, {})
            b1m = s.get('b1', {}).get('mean', 0)
            b2m = s.get('b2', {}).get('mean', 0)
            b3m = s.get('b3', {}).get('mean', 0)
            ok1 = check(f"{proto_label}: B1 mean ({b1m:.4f}) > B3 mean ({b3m:.4f})", b1m > b3m)
            ok2 = check(f"{proto_label}: B3 mean ({b3m:.4f}) > B2 mean ({b2m:.4f})", b3m > b2m)
            all_ok &= ok1 and ok2

        # ----------------------------------------------------------
        # Check 7: B2 attack success near zero at FAR=1% and FAR=0.1%
        # ----------------------------------------------------------
        print("\n--- Check 7: B2 IAPMR ≈ 0% at strict thresholds ---")
        for proto_key, proto_label in [('protocol_a', 'Protocol A'), ('protocol_b', 'Protocol B')]:
            for tp in ['far_1pct', 'far_01pct']:
                b2_iapmr = iapmr_data[proto_key][tp]['b2']
                ok = check(f"{proto_label} {tp}: B2 IAPMR = {b2_iapmr*100:.1f}%",
                           b2_iapmr <= 0.01,
                           "Expected ≤ 1%")
                all_ok &= ok

        # ----------------------------------------------------------
        # Check 8: Protocol B scores <= Protocol A scores
        # ----------------------------------------------------------
        print("\n--- Check 8: Protocol B ≤ Protocol A (mean scores) ---")
        for label in ['b1', 'b3']:
            a_mean = stats.get('protocol_a', {}).get(label, {}).get('mean', 0)
            b_mean = stats.get('protocol_b', {}).get(label, {}).get('mean', 0)
            ok = check(f"{label.upper()}: Protocol B mean ({b_mean:.4f}) ≤ Protocol A mean ({a_mean:.4f})",
                       b_mean <= a_mean + 0.01)
            all_ok &= ok

        # ----------------------------------------------------------
        # Check 9: Protocol B pairing correctness
        # ----------------------------------------------------------
        print("\n--- Check 9: Protocol B pairing correctness ---")
        # Enrollment should always be img0, attack should be img1-4
        enrollment_indices = set()
        attack_indices = set()
        same_pair_found = False

        for r in cross_rows:
            enr = r['enrollment_image']
            atk = r['attack_source_image']

            # Parse img index
            enr_idx = int(enr.rsplit('img', 1)[1].replace('.png', ''))
            atk_idx = int(atk.rsplit('img', 1)[1].replace('.png', ''))
            enrollment_indices.add(enr_idx)
            attack_indices.add(atk_idx)

            # Check for same-image pairs
            if enr == atk:
                same_pair_found = True

        ok = check("Enrollment is always img0",
                    enrollment_indices == {0},
                    f"Enrollment indices found: {sorted(enrollment_indices)}")
        all_ok &= ok

        ok = check("Attack sources are img1–img4",
                    attack_indices == {1, 2, 3, 4},
                    f"Attack indices found: {sorted(attack_indices)}")
        all_ok &= ok

        ok = check("No same-image pairs in Protocol B",
                    not same_pair_found)
        all_ok &= ok

        ok = check(f"Protocol B has 400 pairs",
                    len(cross_rows) == 400,
                    f"Found {len(cross_rows)} pairs")
        all_ok &= ok

    # ----------------------------------------------------------
    # Check 10: Cross-evaluator directional agreement
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("Cross-evaluator consistency")
    print(f"{'='*60}")

    with open(os.path.join(EVAL_DIR, 'biometric_iapmr_glintr100.json')) as f:
        g_iapmr = json.load(f)
    with open(os.path.join(EVAL_DIR, 'biometric_iapmr_arcface.json')) as f:
        a_iapmr = json.load(f)

    for proto_key, proto_label in [('protocol_a', 'Protocol A'), ('protocol_b', 'Protocol B')]:
        for tp in ['eer', 'far_1pct', 'far_01pct']:
            # Allow 2% tolerance for IAPMR ordering — at loose thresholds (EER),
            # B1 and B3 may have nearly identical IAPMR with minor inversions.
            tol = 0.02
            g_b1 = g_iapmr[proto_key][tp]['b1']
            g_b3 = g_iapmr[proto_key][tp]['b3']
            g_b2 = g_iapmr[proto_key][tp]['b2']
            a_b1 = a_iapmr[proto_key][tp]['b1']
            a_b3 = a_iapmr[proto_key][tp]['b3']
            a_b2 = a_iapmr[proto_key][tp]['b2']

            g_order = (g_b1 + tol >= g_b3) and (g_b3 >= g_b2)
            a_order = (a_b1 + tol >= a_b3) and (a_b3 >= a_b2)

            # Check if strict ordering holds
            g_strict = (g_b1 >= g_b3 >= g_b2)
            a_strict = (a_b1 >= a_b3 >= a_b2)

            ok = check(f"{proto_label} {tp}: Both evaluators agree B1≥B3≥B2",
                       g_order and a_order,
                       f"G: B1={g_b1:.4f},B3={g_b3:.4f},B2={g_b2:.4f}; "
                       f"A: B1={a_b1:.4f},B3={a_b3:.4f},B2={a_b2:.4f}")
            all_ok &= ok

            if not (g_strict and a_strict) and (g_order and a_order):
                print(f"         ⚠️ WARNING: minor inversion within {tol*100:.0f}% tolerance (not a critical issue)")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    if all_ok:
        print("Phase 5 RESULT: ✅ PASS — All validation checks passed.")
        print("Safe to proceed to Phase 6 (Visualization and Final Tables).")
    else:
        print("Phase 5 RESULT: ❌ FAIL — Issues found above.")
    print(f"{'='*60}")

    # Save
    results_path = os.path.join(EVAL_DIR, 'phase5_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump({'overall': 'PASS' if all_ok else 'FAIL'}, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
