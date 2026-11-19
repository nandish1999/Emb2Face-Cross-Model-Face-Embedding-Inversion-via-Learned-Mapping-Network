import os
import sys
import csv
import time
import argparse
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from scripts import preprocessing_utils as pu


def parse_args():
    parser = argparse.ArgumentParser(description='Aligned re-ID evaluation')
    parser.add_argument('--output-prefix', type=str, default='tiny_reconstruction',
                        help='Output filename prefix (default: tiny_reconstruction)')
    parser.add_argument('--recon-dir', type=str, default='outputs/reconstructed',
                        help='Reconstruction output directory (default: outputs/reconstructed)')
    return parser.parse_args()


def main():
    args = parse_args()
    prefix = args.output_prefix

    print("=" * 70)
    print(f"Aligned Re-Identification Evaluation (prefix={prefix})")
    print("=" * 70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    recon_dir = args.recon_dir
    methods = {
        'baseline': os.path.join(recon_dir, 'baseline'),
        'unmapped': os.path.join(recon_dir, 'unmapped'),
        'mapped': os.path.join(recon_dir, 'mapped'),
    }
    originals_dir = os.path.join(recon_dir, 'originals')

    # ============================================================
    # 1. Find images
    # ============================================================
    print("1. Finding generated images...")
    original_files = sorted([f for f in os.listdir(originals_dir) if f.endswith('.png')])
    print(f"   Found {len(original_files)} original images")
    print()

    # ============================================================
    # 2. Load models
    # ============================================================
    print("2. Loading models...")

    # Face detector (SCRFD with keypoints)
    from insightface.model_zoo import get_model
    from insightface.utils.face_align import norm_crop

    print("   Loading SCRFD face detector...")
    det = get_model(
        'models/antelopev2_eval/scrfd_10g_bnkps.onnx',
        providers=['CPUExecutionProvider']
    )
    det.prepare(ctx_id=-1, input_size=(640, 640))

    # Recognition model (GlintR100 — independent eval judge)
    print("   Loading GlintR100 eval recognizer...")
    eval_session = pu.load_eval_arcface_session()

    print("   Models loaded.")
    print()

    # ============================================================
    # 3. Process each image
    # ============================================================
    print("3. Computing aligned re-identification cosines...")
    print()

    os.makedirs('outputs/evaluation', exist_ok=True)

    detail_path = f'outputs/evaluation/{prefix}_eval_aligned.csv'
    detail_file = open(detail_path, 'w', newline='')
    detail_writer = csv.writer(detail_file)
    detail_writer.writerow([
        'identity', 'image_name', 'method',
        'cosine_aligned', 'cosine_naive_resize',
        'face_detected', 'det_score',
        'original_image_path', 'generated_image_path'
    ])

    results = {'baseline': [], 'unmapped': [], 'mapped': []}
    results_naive = {'baseline': [], 'unmapped': [], 'mapped': []}
    detection_stats = {'success': 0, 'fail': 0}

    for fname in original_files:
        identity = fname.replace('.png', '')
        orig_path = os.path.join(originals_dir, fname)

        # Original embedding (112x112 already aligned)
        orig_bgr = cv2.imread(orig_path)
        orig_emb = pu.get_eval_embedding_direct(eval_session, orig_bgr)

        print(f"   {identity}:")

        for method_name, method_dir in methods.items():
            gen_path = os.path.join(method_dir, fname)
            if not os.path.exists(gen_path):
                print(f"     {method_name:>10}: MISSING")
                continue

            gen_img = cv2.imread(gen_path)

            # --- Method A: Aligned (detect + crop) ---
            bboxes, kpss = det.detect(gen_img, max_num=1)

            if len(bboxes) > 0:
                det_score = float(bboxes[0][4])
                kps = kpss[0]
                aligned = norm_crop(gen_img, kps, image_size=112)
                aligned_emb = pu.get_eval_embedding_direct(eval_session, aligned)
                cos_aligned = float(np.dot(orig_emb, aligned_emb))
                face_detected = True
                detection_stats['success'] += 1
            else:
                cos_aligned = None
                det_score = 0.0
                face_detected = False
                detection_stats['fail'] += 1

            # --- Method B: Naive resize (for comparison) ---
            naive_resized = cv2.resize(gen_img, (112, 112), interpolation=cv2.INTER_AREA)
            naive_emb = pu.get_eval_embedding_direct(eval_session, naive_resized)
            cos_naive = float(np.dot(orig_emb, naive_emb))

            if face_detected:
                results[method_name].append(cos_aligned)
            results_naive[method_name].append(cos_naive)

            detail_writer.writerow([
                identity, fname, method_name,
                f'{cos_aligned:.4f}' if cos_aligned is not None else 'N/A',
                f'{cos_naive:.4f}',
                'Y' if face_detected else 'N',
                f'{det_score:.4f}',
                orig_path, gen_path
            ])

            aligned_str = f'{cos_aligned:.4f}' if cos_aligned is not None else 'N/A'
            print(f"     {method_name:>10}: aligned={aligned_str}  naive={cos_naive:.4f}  det={'Y' if face_detected else 'N'}({det_score:.2f})")

        print()

    detail_file.close()

    # ============================================================
    # 4. Summary
    # ============================================================
    summary_path = f'outputs/evaluation/{prefix}_eval_aligned_summary.csv'
    summary_file = open(summary_path, 'w', newline='')
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(['method', 'eval_type', 'mean', 'std', 'min', 'max', 'n'])

    print("=" * 70)
    print("RE-IDENTIFICATION EVALUATION SUMMARY (ALIGNED vs NAIVE)")
    print("=" * 70)
    print()
    print(f"  Face detection success: {detection_stats['success']}/{detection_stats['success'] + detection_stats['fail']}")
    print()

    print(f"  {'Method':<14} {'Aligned Mean':>13} {'Naive Mean':>11} {'Diff':>8} {'N':>4}")
    print(f"  {'-'*14} {'-'*13} {'-'*11} {'-'*8} {'-'*4}")

    for method_name in ['baseline', 'unmapped', 'mapped']:
        label = {'baseline': 'B1 baseline', 'unmapped': 'B2 unmapped', 'mapped': 'B3 mapped'}[method_name]

        vals_a = results[method_name]
        vals_n = results_naive[method_name]

        if vals_a:
            mean_a = np.mean(vals_a)
            std_a = np.std(vals_a)
            min_a = np.min(vals_a)
            max_a = np.max(vals_a)
        else:
            mean_a = std_a = min_a = max_a = 0.0

        if vals_n:
            mean_n = np.mean(vals_n)
            std_n = np.std(vals_n)
            min_n = np.min(vals_n)
            max_n = np.max(vals_n)
        else:
            mean_n = std_n = min_n = max_n = 0.0

        diff = mean_a - mean_n
        print(f"  {label:<14} {mean_a:>13.4f} {mean_n:>11.4f} {diff:>+8.4f} {len(vals_a):>4}")

        summary_writer.writerow([method_name, 'aligned', f'{mean_a:.4f}', f'{std_a:.4f}', f'{min_a:.4f}', f'{max_a:.4f}', len(vals_a)])
        summary_writer.writerow([method_name, 'naive', f'{mean_n:.4f}', f'{std_n:.4f}', f'{min_n:.4f}', f'{max_n:.4f}', len(vals_n)])

    summary_file.close()

    # ============================================================
    # 5. Interpretation
    # ============================================================
    b1_a = np.mean(results['baseline']) if results['baseline'] else 0
    b2_a = np.mean(results['unmapped']) if results['unmapped'] else 0
    b3_a = np.mean(results['mapped']) if results['mapped'] else 0

    b1_n = np.mean(results_naive['baseline']) if results_naive['baseline'] else 0
    b2_n = np.mean(results_naive['unmapped']) if results_naive['unmapped'] else 0
    b3_n = np.mean(results_naive['mapped']) if results_naive['mapped'] else 0

    print()
    print("INTERPRETATION:")
    print()
    print(f"  Aligned evaluation:")
    print(f"    B1 (baseline):  {b1_a:.4f}")
    print(f"    B2 (unmapped):  {b2_a:.4f}")
    print(f"    B3 (mapped):    {b3_a:.4f}")
    print()

    if b3_a > b2_a:
        print(f"  ✓ B3 BEATS B2 (aligned): +{b3_a - b2_a:.4f}")
    else:
        print(f"  ✗ B3 does NOT beat B2 (aligned)")

    if b1_a > 0:
        print(f"  B3 is {b3_a/b1_a*100:.1f}% of B1 (aligned)")
    print()

    print(f"  Naive resize evaluation:")
    print(f"    B1 (baseline):  {b1_n:.4f}")
    print(f"    B2 (unmapped):  {b2_n:.4f}")
    print(f"    B3 (mapped):    {b3_n:.4f}")
    print()

    print(f"Output files:")
    print(f"  Detail CSV:  {detail_path}")
    print(f"  Summary CSV: {summary_path}")
    print()
    print(f"Done at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
