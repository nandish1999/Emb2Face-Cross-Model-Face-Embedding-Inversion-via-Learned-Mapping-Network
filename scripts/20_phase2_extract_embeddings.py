import os
import sys
import json
import time
import csv
import numpy as np
import cv2

# ============================================================
# Setup
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from scripts import preprocessing_utils as pu

RECON_DIR = 'outputs/reconstructed_official_full'
EVAL_DIR = 'outputs/evaluation'

METHOD_DIRS = {
    'original': os.path.join(RECON_DIR, 'originals'),
    'baseline': os.path.join(RECON_DIR, 'baseline'),
    'unmapped': os.path.join(RECON_DIR, 'unmapped'),
    'mapped':   os.path.join(RECON_DIR, 'mapped'),
}

EVALUATORS = {
    'glintr100': {
        'model_path': 'models/antelopev2_eval/glintr100.onnx',
        'output_path': os.path.join(EVAL_DIR, 'biometric_embeddings_glintr100.npz'),
    },
    'arcface_internal': {
        'model_path': 'models/antelopev2_arc2face/arcface.onnx',
        'output_path': os.path.join(EVAL_DIR, 'biometric_embeddings_arcface_internal.npz'),
    },
}

CSV_FILES = {
    'glintr100': os.path.join(EVAL_DIR, 'full_official_eval_aligned.csv'),
    'arcface_internal': os.path.join(EVAL_DIR, 'full_official_eval_arcface_internal.csv'),
}

CSV_COSINE_COLS = {
    'glintr100': 'cosine_naive_resize',
    'arcface_internal': 'cosine_internal_naive',
}


def load_manifest():
    """Load Phase 1 image manifest."""
    manifest_path = os.path.join(EVAL_DIR, 'phase1_image_manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"   Loaded manifest: {len(manifest)} images")
    return manifest


def extract_all_embeddings(session, manifest):
    """Extract embeddings for all 2,000 images using one evaluator session.

    Returns:
        embeddings: np.ndarray (2000, 512)
        names:      list of str (2000)
        identities: list of str (2000)
        methods:    list of str (2000)
    """
    all_embeddings = []
    all_names = []
    all_identities = []
    all_methods = []

    for method_label, dirpath in METHOD_DIRS.items():
        print(f"   Extracting {method_label} ({dirpath})...")
        is_original = (method_label == 'original')

        for entry in manifest:
            fname = entry['filename']
            identity = entry['identity']
            fpath = os.path.join(dirpath, fname)

            # Load image as BGR
            bgr = cv2.imread(fpath)

            # Resize generated images (512×512 → 112×112)
            if not is_original:
                bgr = cv2.resize(bgr, (112, 112), interpolation=cv2.INTER_AREA)

            # Extract L2-normalized embedding
            emb = pu.get_arcface_embedding_direct(session, bgr)

            all_embeddings.append(emb)
            all_names.append(fname)
            all_identities.append(identity)
            all_methods.append(method_label)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    return embeddings, all_names, all_identities, all_methods


def validate_embeddings(embeddings, names, identities, methods, evaluator_name):
    """Run Phase 2 validation checks on extracted embeddings."""
    print(f"\n   Validation for {evaluator_name}:")
    all_ok = True

    # Check 1: Shape
    n, d = embeddings.shape
    if n != 2000:
        print(f"   FAIL  Expected 2000 embeddings, got {n}")
        all_ok = False
    else:
        print(f"   OK    {n} embeddings extracted")

    if d != 512:
        print(f"   FAIL  Expected dimension 512, got {d}")
        all_ok = False
    else:
        print(f"   OK    Dimension: {d}")

    # Check 2: Finite values
    n_nonfinite = np.sum(~np.isfinite(embeddings))
    if n_nonfinite > 0:
        print(f"   FAIL  {n_nonfinite} non-finite values found")
        all_ok = False
    else:
        print(f"   OK    All values are finite")

    # Check 3: L2 norm ≈ 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    norm_min = norms.min()
    norm_max = norms.max()
    norm_mean = norms.mean()
    norm_ok = np.allclose(norms, 1.0, atol=1e-5)
    if not norm_ok:
        print(f"   FAIL  Norms not unit: min={norm_min:.6f}, max={norm_max:.6f}, mean={norm_mean:.6f}")
        all_ok = False
    else:
        print(f"   OK    All norms ≈ 1.0 (min={norm_min:.6f}, max={norm_max:.6f})")

    # Check 4: Method counts
    methods_arr = np.array(methods)
    for m in ['original', 'baseline', 'unmapped', 'mapped']:
        cnt = np.sum(methods_arr == m)
        if cnt != 500:
            print(f"   FAIL  Method '{m}': {cnt} (expected 500)")
            all_ok = False
        else:
            print(f"   OK    Method '{m}': {cnt} embeddings")

    return all_ok


def spot_check_csv(embeddings, names, methods, evaluator_name):
    """Spot-check: recompute a few original→B3 cosines and compare to CSV."""
    csv_path = CSV_FILES.get(evaluator_name)
    cosine_col = CSV_COSINE_COLS.get(evaluator_name)

    if not csv_path or not os.path.isfile(csv_path):
        print(f"   SKIP  CSV not found for {evaluator_name}")
        return True

    # Build lookup: image_name → embedding, indexed by method
    name_to_idx = {}
    for i, (name, method) in enumerate(zip(names, methods)):
        name_to_idx[(method, name)] = i

    # Load CSV reference values for 'mapped' method (B3)
    csv_ref = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['method'].strip() == 'mapped':
                img_name = row['image_name'].strip()
                val_str = row[cosine_col].strip()
                if val_str != 'N/A' and val_str != '':
                    csv_ref[img_name] = float(val_str)

    if not csv_ref:
        print(f"   SKIP  No mapped cosine values found in CSV")
        return True

    # Spot-check first 10 images that have CSV values
    check_names = sorted(csv_ref.keys())[:10]
    max_diff = 0.0
    mismatches = []

    print(f"\n   CSV spot-check ({evaluator_name}, {cosine_col}):")
    for img_name in check_names:
        orig_idx = name_to_idx.get(('original', img_name))
        mapped_idx = name_to_idx.get(('mapped', img_name))

        if orig_idx is None or mapped_idx is None:
            continue

        computed = float(np.dot(embeddings[orig_idx], embeddings[mapped_idx]))
        expected = csv_ref[img_name]
        diff = abs(computed - expected)
        max_diff = max(max_diff, diff)

        status = "OK" if diff < 0.001 else "DIFF"
        print(f"   {status}  {img_name}: computed={computed:.4f}, CSV={expected:.4f}, diff={diff:.6f}")

        if diff >= 0.001:
            mismatches.append((img_name, computed, expected, diff))

    if mismatches:
        print(f"   WARN  {len(mismatches)} values differ by ≥0.001 from CSV")
    else:
        print(f"   OK    All spot-checked values match CSV (max diff: {max_diff:.6f})")

    return len(mismatches) == 0


def main():
    print("=" * 60)
    print("Phase 2 — Embedding Extraction")
    print("=" * 60)

    # Load manifest
    print("\n1. Loading Phase 1 manifest...")
    manifest = load_manifest()

    os.makedirs(EVAL_DIR, exist_ok=True)
    overall_ok = True
    results = {}

    for eval_name, eval_cfg in EVALUATORS.items():
        print(f"\n{'='*60}")
        print(f"Evaluator: {eval_name}")
        print(f"Model: {eval_cfg['model_path']}")
        print(f"{'='*60}")

        # Load ONNX session
        print(f"\n2. Loading ONNX model...")
        t0 = time.time()
        session = pu.load_arcface_session(model_path=eval_cfg['model_path'])
        print(f"   Loaded in {time.time()-t0:.1f}s")

        # Extract embeddings
        print(f"\n3. Extracting embeddings (2,000 images)...")
        t0 = time.time()
        embeddings, names, identities, methods = extract_all_embeddings(session, manifest)
        elapsed = time.time() - t0
        print(f"   Done in {elapsed:.1f}s ({elapsed/2000*1000:.1f}ms per image)")

        # Validate
        print(f"\n4. Validating embeddings...")
        valid = validate_embeddings(embeddings, names, identities, methods, eval_name)
        if not valid:
            overall_ok = False

        # Spot-check against CSV
        print(f"\n5. CSV spot-check...")
        csv_ok = spot_check_csv(embeddings, names, methods, eval_name)
        if not csv_ok:
            print(f"   NOTE  CSV spot-check found differences. This may indicate "
                  f"preprocessing inconsistency. Investigate before Phase 3.")

        # Save NPZ
        print(f"\n6. Saving NPZ...")
        np.savez_compressed(
            eval_cfg['output_path'],
            embeddings=embeddings,
            names=np.array(names),
            identities=np.array(identities),
            methods=np.array(methods),
        )
        fsize_mb = os.path.getsize(eval_cfg['output_path']) / (1024*1024)
        print(f"   Saved: {eval_cfg['output_path']} ({fsize_mb:.1f} MB)")

        results[eval_name] = {
            'embeddings_shape': list(embeddings.shape),
            'norm_min': float(np.linalg.norm(embeddings, axis=1).min()),
            'norm_max': float(np.linalg.norm(embeddings, axis=1).max()),
            'file_size_mb': round(fsize_mb, 1),
            'extraction_time_s': round(elapsed, 1),
            'validation_pass': valid,
            'csv_spot_check_pass': csv_ok,
        }

    # Summary
    print(f"\n{'='*60}")
    if overall_ok:
        print("Phase 2 RESULT: ✅ PASS — All embeddings extracted and validated.")
        print("Safe to proceed to Phase 3 (Bona Fide Score Distributions).")
    else:
        print("Phase 2 RESULT: ❌ FAIL — Issues found. Check output above.")
    print(f"{'='*60}")

    # Save results JSON
    results['overall'] = 'PASS' if overall_ok else 'FAIL'
    results_path = os.path.join(EVAL_DIR, 'phase2_extraction_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return 0 if overall_ok else 1


if __name__ == '__main__':
    sys.exit(main())
