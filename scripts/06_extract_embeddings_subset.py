import os
import sys
import csv
import time
import argparse
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from scripts import preprocessing_utils as pu


def parse_args():
    parser = argparse.ArgumentParser(description='Extract embedding subset')
    parser.add_argument('--n', type=int, default=100,
                        help='Number of images to process (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for subset selection (default: 42)')
    parser.add_argument('--output-dir', type=str, default='embeddings/train',
                        help='Output directory (default: embeddings/train)')
    return parser.parse_args()


def main():
    args = parse_args()
    N = args.n

    print("=" * 60)
    print(f"Step 6A: Extract Paired Embeddings (N={N})")
    print("=" * 60)

    # ============================================================
    # 1. Load metadata
    # ============================================================
    print("\n1. Loading train_metadata.csv...")
    with open('data/train_metadata.csv') as f:
        all_rows = list(csv.DictReader(f))
    print(f"   Total training images: {len(all_rows)}")

    # Deterministic subset selection
    np.random.seed(args.seed)
    indices = np.random.choice(len(all_rows), size=min(N, len(all_rows)), replace=False)
    indices.sort()
    subset = [all_rows[i] for i in indices]
    print(f"   Selected subset: {len(subset)} images (seed={args.seed})")

    unique_ids = set(row['identity_id'] for row in subset)
    print(f"   Unique identities in subset: {len(unique_ids)}")

    # ============================================================
    # 2. Load models
    # ============================================================
    print("\n2. Loading models...")
    t0 = time.time()

    print("   Loading AdaFace IR-50...")
    adaface_model = pu.load_adaface_model()
    t_ada_load = time.time() - t0
    print(f"   AdaFace loaded in {t_ada_load:.1f}s")

    print("   Loading ArcFace (Arc2Face arcface.onnx)...")
    arcface_session = pu.load_arcface_session()
    t_arc_load = time.time() - t0
    print(f"   ArcFace loaded in {t_arc_load - t_ada_load:.1f}s")

    print(f"   Total model loading: {time.time() - t0:.1f}s")

    # ============================================================
    # 3. Extract embeddings
    # ============================================================
    print(f"\n3. Extracting embeddings for {len(subset)} images...")

    adaface_embeddings = []
    arcface_embeddings = []
    identity_ids = []
    image_paths = []
    errors = []

    t_start = time.time()
    for i, row in enumerate(subset):
        path = row['image_path']
        identity = row['identity_id']

        try:
            # Load image in both formats
            rgb, bgr = pu.load_image_for_both(path)

            # AdaFace embedding
            ada_emb = pu.get_adaface_embedding(adaface_model, rgb)

            # ArcFace embedding (direct ONNX, bypassing detection)
            arc_emb = pu.get_arcface_embedding_direct(arcface_session, bgr)

            adaface_embeddings.append(ada_emb)
            arcface_embeddings.append(arc_emb)
            identity_ids.append(identity)
            image_paths.append(path)

        except Exception as e:
            errors.append({'path': path, 'error': str(e)})

        # Progress reporting
        if (i + 1) % max(1, len(subset) // 10) == 0 or (i + 1) == len(subset):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(subset) - i - 1) / rate if rate > 0 else 0
            print(f"   [{i+1:>{len(str(len(subset)))}}/{len(subset)}] "
                  f"{elapsed:.1f}s elapsed, {rate:.1f} img/s, ETA {eta:.1f}s, "
                  f"errors={len(errors)}")

    t_extract = time.time() - t_start

    # ============================================================
    # 4. Convert to arrays and save
    # ============================================================
    print("\n4. Saving embeddings...")

    adaface_arr = np.array(adaface_embeddings, dtype=np.float32)
    arcface_arr = np.array(arcface_embeddings, dtype=np.float32)
    ids_arr = np.array(identity_ids)
    paths_arr = np.array(image_paths)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'train_subset_{len(adaface_arr)}.npz')
    np.savez_compressed(
        output_path,
        adaface=adaface_arr,
        arcface=arcface_arr,
        ids=ids_arr,
        paths=paths_arr,
    )
    file_size = os.path.getsize(output_path) / 1e6
    print(f"   Saved: {output_path}")
    print(f"   File size: {file_size:.2f} MB")

    # ============================================================
    # 5. Sanity checks and statistics
    # ============================================================
    print("\n5. Sanity checks...")
    print(f"   adaface shape: {adaface_arr.shape}")
    print(f"   arcface shape: {arcface_arr.shape}")
    print(f"   ids shape:     {ids_arr.shape}")

    # Check norms (should be ~1.0)
    ada_norms = np.linalg.norm(adaface_arr, axis=1)
    arc_norms = np.linalg.norm(arcface_arr, axis=1)
    print(f"   AdaFace norms: mean={ada_norms.mean():.4f}, std={ada_norms.std():.6f}")
    print(f"   ArcFace norms: mean={arc_norms.mean():.4f}, std={arc_norms.std():.6f}")

    # Cross-space cosine similarity (should be near 0 — different spaces)
    cross_sims = np.sum(adaface_arr * arcface_arr, axis=1)
    print(f"\n   Cross-space cosine similarity (AdaFace ↔ ArcFace):")
    print(f"     mean={cross_sims.mean():.4f}, std={cross_sims.std():.4f}")
    print(f"     min={cross_sims.min():.4f}, max={cross_sims.max():.4f}")
    print(f"     (Expected: near 0 — confirms different embedding spaces)")

    # Within-space same-identity similarity (if we have enough same-ID pairs)
    print(f"\n   Within-space same-identity check:")
    id_groups = {}
    for i, identity in enumerate(identity_ids):
        id_groups.setdefault(identity, []).append(i)
    multi_groups = {k: v for k, v in id_groups.items() if len(v) >= 2}

    if multi_groups:
        ada_same_sims = []
        arc_same_sims = []
        for identity, idxs in list(multi_groups.items())[:20]:
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    ada_same_sims.append(np.dot(adaface_arr[idxs[a]], adaface_arr[idxs[b]]))
                    arc_same_sims.append(np.dot(arcface_arr[idxs[a]], arcface_arr[idxs[b]]))
        ada_same_sims = np.array(ada_same_sims)
        arc_same_sims = np.array(arc_same_sims)
        print(f"     Found {len(multi_groups)} identities with 2+ images in subset")
        print(f"     AdaFace same-ID cosine: mean={ada_same_sims.mean():.4f} (should be >0.5)")
        print(f"     ArcFace same-ID cosine: mean={arc_same_sims.mean():.4f} (should be >0.5)")
    else:
        print(f"     No identities with 2+ images in subset (need larger N)")

    # ============================================================
    # 6. Summary
    # ============================================================
    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print("STEP 6A SUMMARY")
    print("=" * 60)
    print(f"Images processed: {len(adaface_arr)}")
    print(f"Errors:           {len(errors)}")
    print(f"Unique identities:{len(set(identity_ids))}")
    print(f"Extraction time:  {t_extract:.1f}s ({len(adaface_arr)/t_extract:.1f} img/s)")
    print(f"Total time:       {total_time:.1f}s (including model loading)")
    print(f"Output file:      {output_path} ({file_size:.2f} MB)")
    print(f"AdaFace shape:    {adaface_arr.shape}")
    print(f"ArcFace shape:    {arcface_arr.shape}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e['path']}: {e['error']}")

    # Estimate full dataset time
    if len(adaface_arr) > 0:
        rate = len(adaface_arr) / t_extract
        total_train = 485936
        est_hours = total_train / rate / 3600
        print(f"\nEstimated full dataset time: {est_hours:.1f} hours ({total_train} images at {rate:.1f} img/s)")

    print("\nDone.")


if __name__ == '__main__':
    main()
