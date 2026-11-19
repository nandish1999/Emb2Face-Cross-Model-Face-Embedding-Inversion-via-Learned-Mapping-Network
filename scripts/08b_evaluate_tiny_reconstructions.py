import os
import sys
import csv
import time
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from scripts import preprocessing_utils as pu


def resize_to_112(image_path):
    """Load any-size image and resize to 112x112 BGR for the eval model.
    
    Arc2Face generates 512x512 images. The eval recognizer expects 112x112.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
    return img


def main():
    print("=" * 70)
    print("Step 8B: Re-Identification Evaluation of Tiny Reconstructions")
    print("=" * 70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    recon_dir = 'outputs/reconstructed'
    methods = {
        'baseline': os.path.join(recon_dir, 'baseline'),
        'unmapped': os.path.join(recon_dir, 'unmapped'),
        'mapped': os.path.join(recon_dir, 'mapped'),
    }
    originals_dir = os.path.join(recon_dir, 'originals')

    # ============================================================
    # 1. Find all original images
    # ============================================================
    print("1. Finding generated images...")
    original_files = sorted([f for f in os.listdir(originals_dir) if f.endswith('.png')])
    print(f"   Found {len(original_files)} original images")
    for f in original_files:
        print(f"     {f}")
    print()

    # ============================================================
    # 2. Load evaluation judge model (GlintR100 — independent)
    # ============================================================
    print("2. Loading evaluation judge model (GlintR100)...")
    print("   IMPORTANT: This is NOT the same model used for training.")
    print("   Using an independent model prevents circular evaluation.")
    t0 = time.time()
    eval_session = pu.load_eval_arcface_session()
    print(f"   Loaded in {time.time() - t0:.1f}s")
    print()

    # ============================================================
    # 3. Extract embeddings and compute cosine similarities
    # ============================================================
    print("3. Computing re-identification cosine similarities...")
    print()

    os.makedirs('outputs/evaluation', exist_ok=True)

    # Detailed CSV
    detail_path = 'outputs/evaluation/tiny_reconstruction_eval.csv'
    detail_file = open(detail_path, 'w', newline='')
    detail_writer = csv.writer(detail_file)
    detail_writer.writerow([
        'identity', 'image_name', 'method',
        'cosine_similarity_original_vs_generated',
        'original_image_path', 'generated_image_path'
    ])

    results = {'baseline': [], 'unmapped': [], 'mapped': []}

    for fname in original_files:
        identity = fname.replace('.png', '')
        orig_path = os.path.join(originals_dir, fname)

        # Extract original embedding
        orig_bgr = resize_to_112(orig_path)  # Already 112x112 but consistent
        orig_emb = pu.get_eval_embedding_direct(eval_session, orig_bgr)

        print(f"   {identity}:")

        for method_name, method_dir in methods.items():
            gen_path = os.path.join(method_dir, fname)
            if not os.path.exists(gen_path):
                print(f"     {method_name}: MISSING")
                continue

            # Extract generated image embedding (resize 512->112)
            gen_bgr = resize_to_112(gen_path)
            gen_emb = pu.get_eval_embedding_direct(eval_session, gen_bgr)

            # Cosine similarity
            cos_sim = float(np.dot(orig_emb, gen_emb))

            results[method_name].append(cos_sim)

            detail_writer.writerow([
                identity, fname, method_name,
                f'{cos_sim:.4f}', orig_path, gen_path
            ])

            print(f"     {method_name:>10}: cosine = {cos_sim:.4f}")

        print()

    detail_file.close()

    # ============================================================
    # 4. Summary statistics
    # ============================================================
    summary_path = 'outputs/evaluation/tiny_reconstruction_eval_summary.csv'
    summary_file = open(summary_path, 'w', newline='')
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(['method', 'mean', 'std', 'min', 'max', 'n'])

    print("=" * 70)
    print("RE-IDENTIFICATION EVALUATION SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Method':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'N':>4}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")

    for method_name in ['baseline', 'unmapped', 'mapped']:
        vals = results[method_name]
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            vmin = np.min(vals)
            vmax = np.max(vals)
            n = len(vals)
        else:
            mean = std = vmin = vmax = 0.0
            n = 0

        label = {'baseline': 'B1 baseline', 'unmapped': 'B2 unmapped', 'mapped': 'B3 mapped'}[method_name]
        print(f"  {label:<12} {mean:>8.4f} {std:>8.4f} {vmin:>8.4f} {vmax:>8.4f} {n:>4}")

        summary_writer.writerow([method_name, f'{mean:.4f}', f'{std:.4f}', f'{vmin:.4f}', f'{vmax:.4f}', n])

    summary_file.close()

    # ============================================================
    # 5. Interpretation
    # ============================================================
    b1_mean = np.mean(results['baseline']) if results['baseline'] else 0
    b2_mean = np.mean(results['unmapped']) if results['unmapped'] else 0
    b3_mean = np.mean(results['mapped']) if results['mapped'] else 0

    print()
    print("INTERPRETATION:")
    print(f"  B1 (baseline):  {b1_mean:.4f} — gold standard (real ArcFace → Arc2Face)")
    print(f"  B2 (unmapped):  {b2_mean:.4f} — negative control (raw AdaFace → Arc2Face)")
    print(f"  B3 (mapped):    {b3_mean:.4f} — our method (mapped AdaFace → Arc2Face)")
    print()

    if b3_mean > b2_mean:
        improvement = b3_mean - b2_mean
        print(f"  ✓ B3 BEATS B2 by {improvement:.4f}")
        print(f"    → Mapping network adds value over raw cross-space input.")
    else:
        print(f"  ✗ B3 does NOT beat B2")
        print(f"    → Mapping network is not improving re-identification.")

    if b3_mean > 0.5 * b1_mean:
        print(f"  ✓ B3 achieves {b3_mean/b1_mean*100:.1f}% of B1 performance")
    else:
        print(f"  ! B3 is only {b3_mean/b1_mean*100:.1f}% of B1 performance")

    print()
    print(f"Output files:")
    print(f"  Detail CSV:  {detail_path}")
    print(f"  Summary CSV: {summary_path}")
    print()
    print(f"Done at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
