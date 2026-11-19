import os
import sys
import csv
import time
import argparse
import glob
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from scripts import preprocessing_utils as pu


def parse_args():
    parser = argparse.ArgumentParser(description='Full embedding extraction')
    parser.add_argument('--metadata', type=str, default='data/train_metadata.csv',
                        help='Path to metadata CSV')
    parser.add_argument('--chunk-size', type=int, default=10000,
                        help='Save a chunk every N images (default: 10000)')
    parser.add_argument('--chunk-dir', type=str, default='embeddings/train/chunks',
                        help='Directory for chunk files')
    parser.add_argument('--output', type=str, default='embeddings/train/train_full.npz',
                        help='Final merged output path')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last completed chunk')
    return parser.parse_args()


def get_completed_chunks(chunk_dir):
    """Find already-completed chunk files and return the total images processed."""
    if not os.path.exists(chunk_dir):
        return 0, []
    pattern = os.path.join(chunk_dir, 'chunk_*.npz')
    chunk_files = sorted(glob.glob(pattern))
    total_done = 0
    for cf in chunk_files:
        data = np.load(cf)
        total_done += len(data['adaface'])
    return total_done, chunk_files


def save_chunk(chunk_dir, chunk_idx, ada_list, arc_list, ids_list, paths_list):
    """Save a chunk of embeddings to disk."""
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_path = os.path.join(chunk_dir, f'chunk_{chunk_idx:04d}.npz')
    np.savez_compressed(
        chunk_path,
        adaface=np.array(ada_list, dtype=np.float32),
        arcface=np.array(arc_list, dtype=np.float32),
        ids=np.array(ids_list),
        paths=np.array(paths_list),
    )
    size_mb = os.path.getsize(chunk_path) / 1e6
    return chunk_path, size_mb


def merge_chunks(chunk_dir, output_path):
    """Merge all chunk files into a single .npz file."""
    pattern = os.path.join(chunk_dir, 'chunk_*.npz')
    chunk_files = sorted(glob.glob(pattern))

    if not chunk_files:
        print("ERROR: No chunk files found to merge.")
        return False

    print(f"\nMerging {len(chunk_files)} chunks into {output_path}...")

    all_ada = []
    all_arc = []
    all_ids = []
    all_paths = []

    for cf in chunk_files:
        data = np.load(cf, allow_pickle=True)
        all_ada.append(data['adaface'])
        all_arc.append(data['arcface'])
        all_ids.append(data['ids'])
        all_paths.append(data['paths'])

    adaface = np.concatenate(all_ada, axis=0)
    arcface = np.concatenate(all_arc, axis=0)
    ids = np.concatenate(all_ids, axis=0)
    paths = np.concatenate(all_paths, axis=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        adaface=adaface,
        arcface=arcface,
        ids=ids,
        paths=paths,
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Merged: {output_path} ({len(adaface)} pairs, {size_mb:.1f} MB)")
    return True


def main():
    args = parse_args()

    print("=" * 70)
    print("Step 6 FULL: Extract All Paired Embeddings")
    print("=" * 70)
    print(f"Metadata:   {args.metadata}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk dir:  {args.chunk_dir}")
    print(f"Output:     {args.output}")
    print(f"Resume:     {args.resume}")
    print(f"Time:       {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ============================================================
    # 1. Load metadata
    # ============================================================
    print("1. Loading metadata...")
    with open(args.metadata) as f:
        all_rows = list(csv.DictReader(f))
    total = len(all_rows)
    print(f"   Total images: {total}")

    # ============================================================
    # 2. Check resume state
    # ============================================================
    start_idx = 0
    chunk_idx = 0
    if args.resume:
        done, existing = get_completed_chunks(args.chunk_dir)
        if done > 0:
            start_idx = done
            chunk_idx = len(existing)
            print(f"   RESUME: Found {len(existing)} chunks, {done} images already done.")
            print(f"   Starting from image {start_idx}.")
        else:
            print("   RESUME: No existing chunks found, starting from scratch.")
    else:
        # Clean start: warn if chunks exist
        done, existing = get_completed_chunks(args.chunk_dir)
        if done > 0:
            print(f"   WARNING: {len(existing)} existing chunks found ({done} images).")
            print(f"   Use --resume to continue, or delete {args.chunk_dir}/ to restart.")
            print(f"   Proceeding with fresh extraction (overwriting)...")
            # Remove old chunks
            for cf in existing:
                os.remove(cf)
            print(f"   Removed {len(existing)} old chunk files.")

    remaining = total - start_idx
    print(f"   Remaining: {remaining} images")

    if remaining <= 0:
        print("   All images already processed!")
        merge_chunks(args.chunk_dir, args.output)
        return

    # ============================================================
    # 3. Load models
    # ============================================================
    print("\n2. Loading models...")
    t0 = time.time()

    print("   Loading AdaFace IR-50...")
    adaface_model = pu.load_adaface_model()
    print(f"   AdaFace loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    print("   Loading ArcFace (arcface.onnx)...")
    arcface_session = pu.load_arcface_session()
    print(f"   ArcFace loaded in {time.time() - t1:.1f}s")

    print(f"   Total model loading: {time.time() - t0:.1f}s")

    # ============================================================
    # 4. Extract embeddings with chunk saving
    # ============================================================
    est_hours = remaining / 5.4 / 3600  # based on measured 5.4 img/s
    print(f"\n3. Extracting embeddings ({remaining} images, est. {est_hours:.1f} hours)...")
    print(f"   Saving chunks every {args.chunk_size} images to {args.chunk_dir}/")
    print()

    ada_buf = []
    arc_buf = []
    ids_buf = []
    paths_buf = []
    errors = []
    total_processed = 0

    t_start = time.time()
    log_interval = max(1, args.chunk_size // 10)  # Log ~10 times per chunk

    for i in range(start_idx, total):
        row = all_rows[i]
        path = row['image_path']
        identity = row['identity_id']

        try:
            rgb, bgr = pu.load_image_for_both(path)
            ada_emb = pu.get_adaface_embedding(adaface_model, rgb)
            arc_emb = pu.get_arcface_embedding_direct(arcface_session, bgr)

            ada_buf.append(ada_emb)
            arc_buf.append(arc_emb)
            ids_buf.append(identity)
            paths_buf.append(path)
        except Exception as e:
            errors.append({'index': i, 'path': path, 'error': str(e)})

        total_processed += 1

        # Progress logging
        if total_processed % log_interval == 0 or i == total - 1:
            elapsed = time.time() - t_start
            rate = total_processed / elapsed if elapsed > 0 else 0
            done_total = start_idx + total_processed
            eta_s = (total - done_total) / rate if rate > 0 else 0
            eta_h = eta_s / 3600
            pct = 100 * done_total / total
            print(f"   [{done_total:>6d}/{total}] ({pct:5.1f}%) "
                  f"{elapsed/60:.1f}min elapsed, {rate:.1f} img/s, "
                  f"ETA {eta_h:.1f}h, errors={len(errors)} "
                  f"[{time.strftime('%H:%M:%S')}]")

        # Save chunk
        if len(ada_buf) >= args.chunk_size or i == total - 1:
            if ada_buf:
                chunk_path, chunk_mb = save_chunk(
                    args.chunk_dir, chunk_idx, ada_buf, arc_buf, ids_buf, paths_buf
                )
                done_total = start_idx + total_processed
                print(f"   >>> CHUNK {chunk_idx} saved: {chunk_path} "
                      f"({len(ada_buf)} pairs, {chunk_mb:.1f} MB) "
                      f"[total: {done_total}/{total}]")
                chunk_idx += 1
                ada_buf = []
                arc_buf = []
                ids_buf = []
                paths_buf = []

    t_extract = time.time() - t_start

    # ============================================================
    # 5. Save error log
    # ============================================================
    if errors:
        os.makedirs('logs', exist_ok=True)
        error_path = 'logs/extraction_errors.csv'
        with open(error_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['index', 'path', 'error'])
            writer.writeheader()
            writer.writerows(errors)
        print(f"\n   Errors saved to {error_path} ({len(errors)} errors)")

    # ============================================================
    # 6. Merge chunks
    # ============================================================
    merge_ok = merge_chunks(args.chunk_dir, args.output)

    if not merge_ok:
        print("ERROR: Merge failed.")
        return

    # Verify merged file
    merged = np.load(args.output)
    n_merged = len(merged['adaface'])

    # ============================================================
    # 7. Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("FULL EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total images:     {total}")
    print(f"Processed:        {total_processed}")
    print(f"Errors:           {len(errors)}")
    print(f"Merged pairs:     {n_merged}")
    print(f"Extraction time:  {t_extract/3600:.1f} hours ({t_extract:.0f}s)")
    print(f"Rate:             {total_processed/t_extract:.1f} img/s")
    print(f"Output:           {args.output}")
    print(f"Output size:      {os.path.getsize(args.output)/1e6:.1f} MB")
    print(f"AdaFace shape:    {merged['adaface'].shape}")
    print(f"ArcFace shape:    {merged['arcface'].shape}")
    print(f"Finished at:      {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Next step:")
    print("  python scripts/07_train_mapping_network_subset.py --data embeddings/train/train_full.npz")
    print()
    print("Done.")


if __name__ == '__main__':
    main()
