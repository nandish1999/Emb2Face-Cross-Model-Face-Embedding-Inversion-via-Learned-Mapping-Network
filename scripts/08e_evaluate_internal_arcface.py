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
    parser = argparse.ArgumentParser(description='Internal ArcFace re-ID evaluation')
    parser.add_argument('--output-prefix', type=str, default='reconstruction_10',
                        help='Output filename prefix')
    parser.add_argument('--recon-dir', type=str, default='outputs/reconstructed',
                        help='Reconstruction output directory (default: outputs/reconstructed)')
    return parser.parse_args()


def main():
    args = parse_args()
    prefix = args.output_prefix

    print("=" * 70)
    print(f"Step 8E: Internal ArcFace Evaluation (prefix={prefix})")
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

    # Face detector (from arc2face directory)
    from insightface.model_zoo import get_model
    from insightface.utils.face_align import norm_crop

    print("   Loading SCRFD face detector (arc2face side)...")
    det = get_model(
        'models/antelopev2_arc2face/scrfd_10g_bnkps.onnx',
        providers=['CPUExecutionProvider']
    )
    det.prepare(ctx_id=-1, input_size=(640, 640))

    # Internal ArcFace model (the one Arc2Face was trained with)
    print("   Loading Arc2Face's internal ArcFace model...")
    arcface_session = pu.load_arcface_session(
        model_path='models/antelopev2_arc2face/arcface.onnx'
    )

    print("   Models loaded.")
    print()

    # ============================================================
    # 3. Process each image
    # ============================================================
    print("3. Computing internal ArcFace cosine similarities...")
    print()

    os.makedirs('outputs/evaluation', exist_ok=True)

    detail_path = f'outputs/evaluation/{prefix}_eval_arcface_internal.csv'
    detail_file = open(detail_path, 'w', newline='')
    detail_writer = csv.writer(detail_file)
    detail_writer.writerow([
        'identity', 'image_name', 'method',
        'cosine_internal_aligned', 'cosine_internal_naive',
        'face_detected', 'det_score',
        'original_image_path', 'generated_image_path'
    ])

    results_aligned = {'baseline': [], 'unmapped': [], 'mapped': []}
    results_naive = {'baseline': [], 'unmapped': [], 'mapped': []}
    detection_stats = {'success': 0, 'fail': 0}

    for fname in original_files:
        identity = fname.replace('.png', '')
        orig_path = os.path.join(originals_dir, fname)

        # Original embedding (112x112 already aligned — direct inference)
        orig_bgr = cv2.imread(orig_path)
        orig_emb = pu.get_arcface_embedding_direct(arcface_session, orig_bgr)

        print(f"   {identity}:")

        for method_name, method_dir in methods.items():
            gen_path = os.path.join(method_dir, fname)
            if not os.path.exists(gen_path):
                print(f"     {method_name:>10}: MISSING")
                continue

            gen_img = cv2.imread(gen_path)

            # --- Aligned: detect + crop + internal ArcFace ---
            bboxes, kpss = det.detect(gen_img, max_num=1)

            if len(bboxes) > 0:
                det_score = float(bboxes[0][4])
                kps = kpss[0]
                aligned = norm_crop(gen_img, kps, image_size=112)
                aligned_emb = pu.get_arcface_embedding_direct(arcface_session, aligned)
                cos_aligned = float(np.dot(orig_emb, aligned_emb))
                face_detected = True
                detection_stats['success'] += 1
            else:
                cos_aligned = None
                det_score = 0.0
                face_detected = False
                detection_stats['fail'] += 1

            # --- Naive resize: 512->112 + internal ArcFace ---
            naive_resized = cv2.resize(gen_img, (112, 112), interpolation=cv2.INTER_AREA)
            naive_emb = pu.get_arcface_embedding_direct(arcface_session, naive_resized)
            cos_naive = float(np.dot(orig_emb, naive_emb))

            if face_detected:
                results_aligned[method_name].append(cos_aligned)
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
    summary_path = f'outputs/evaluation/{prefix}_eval_arcface_internal_summary.csv'
    summary_file = open(summary_path, 'w', newline='')
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(['method', 'eval_type', 'mean', 'std', 'min', 'max', 'n'])

    print("=" * 70)
    print("INTERNAL ARCFACE EVALUATION SUMMARY")
    print("=" * 70)
    print()
    total = detection_stats['success'] + detection_stats['fail']
    print(f"  Face detection success: {detection_stats['success']}/{total}")
    print()

    print(f"  {'Method':<14} {'Aligned Mean':>13} {'Naive Mean':>11} {'N_aligned':>10} {'N_naive':>8}")
    print(f"  {'-'*14} {'-'*13} {'-'*11} {'-'*10} {'-'*8}")

    for method_name in ['baseline', 'unmapped', 'mapped']:
        label = {'baseline': 'B1 baseline', 'unmapped': 'B2 unmapped', 'mapped': 'B3 mapped'}[method_name]

        vals_a = results_aligned[method_name]
        vals_n = results_naive[method_name]

        mean_a = np.mean(vals_a) if vals_a else 0.0
        std_a = np.std(vals_a) if vals_a else 0.0
        min_a = np.min(vals_a) if vals_a else 0.0
        max_a = np.max(vals_a) if vals_a else 0.0

        mean_n = np.mean(vals_n) if vals_n else 0.0
        std_n = np.std(vals_n) if vals_n else 0.0
        min_n = np.min(vals_n) if vals_n else 0.0
        max_n = np.max(vals_n) if vals_n else 0.0

        print(f"  {label:<14} {mean_a:>13.4f} {mean_n:>11.4f} {len(vals_a):>10} {len(vals_n):>8}")

        summary_writer.writerow([method_name, 'aligned', f'{mean_a:.4f}', f'{std_a:.4f}', f'{min_a:.4f}', f'{max_a:.4f}', len(vals_a)])
        summary_writer.writerow([method_name, 'naive', f'{mean_n:.4f}', f'{std_n:.4f}', f'{min_n:.4f}', f'{max_n:.4f}', len(vals_n)])

    summary_file.close()

    # ============================================================
    # 5. Interpretation
    # ============================================================
    b1_a = np.mean(results_aligned['baseline']) if results_aligned['baseline'] else 0
    b2_a = np.mean(results_aligned['unmapped']) if results_aligned['unmapped'] else 0
    b3_a = np.mean(results_aligned['mapped']) if results_aligned['mapped'] else 0

    b1_n = np.mean(results_naive['baseline']) if results_naive['baseline'] else 0
    b2_n = np.mean(results_naive['unmapped']) if results_naive['unmapped'] else 0
    b3_n = np.mean(results_naive['mapped']) if results_naive['mapped'] else 0

    print()
    print("COMPARISON WITH GLINTR100:")
    print()
    print(f"  {'Metric':<20} {'Internal ArcFace':>18} {'GlintR100':>12}")
    print(f"  {'-'*20} {'-'*18} {'-'*12}")
    print(f"  {'B1 aligned':<20} {b1_a:>18.4f} {'0.0295':>12}")
    print(f"  {'B2 aligned':<20} {b2_a:>18.4f} {'0.0006':>12}")
    print(f"  {'B3 aligned':<20} {b3_a:>18.4f} {'-0.0113':>12}")
    print(f"  {'B1 naive':<20} {b1_n:>18.4f} {'0.0199':>12}")
    print(f"  {'B2 naive':<20} {b2_n:>18.4f} {'-0.0058':>12}")
    print(f"  {'B3 naive':<20} {b3_n:>18.4f} {'0.0189':>12}")
    print()

    if b3_a > b2_a:
        print(f"  ALIGNED: B3 BEATS B2 by {b3_a - b2_a:.4f}")
    else:
        print(f"  ALIGNED: B3 does NOT beat B2 (diff={b3_a - b2_a:.4f})")

    if b3_n > b2_n:
        print(f"  NAIVE:   B3 BEATS B2 by {b3_n - b2_n:.4f}")
    else:
        print(f"  NAIVE:   B3 does NOT beat B2 (diff={b3_n - b2_n:.4f})")

    print()
    if b1_a > 0.1:
        print(f"  Internal ArcFace B1 is MUCH HIGHER than GlintR100 B1")
        print(f"  -> Arc2Face preserves identity in its own space!")
        print(f"  -> The issue was cross-model evaluator mismatch.")
    elif b1_a > 0.05:
        print(f"  Internal ArcFace B1 is slightly higher than GlintR100 B1")
        print(f"  -> Partial identity preservation detected.")
    else:
        print(f"  Internal ArcFace B1 is also low (< 0.05)")
        print(f"  -> Arc2Face does NOT preserve identity even in its own space.")

    print()
    print(f"Output files:")
    print(f"  Detail CSV:  {detail_path}")
    print(f"  Summary CSV: {summary_path}")
    print()
    print(f"Done at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
