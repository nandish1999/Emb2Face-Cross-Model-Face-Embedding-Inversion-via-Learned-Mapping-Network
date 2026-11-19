import os
import sys
import re
import json
from collections import defaultdict

# PIL for image reading / size check
from PIL import Image

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)

RECON_DIR = 'outputs/reconstructed_official_full'

DIRS = {
    'originals': os.path.join(RECON_DIR, 'originals'),
    'baseline':  os.path.join(RECON_DIR, 'baseline'),
    'unmapped':  os.path.join(RECON_DIR, 'unmapped'),
    'mapped':    os.path.join(RECON_DIR, 'mapped'),
}

EXPECTED_COUNT = 500
EXPECTED_IDENTITIES = 100
EXPECTED_IMAGES_PER_ID = 5
FILENAME_PATTERN = re.compile(r'^(id_\d+)_img(\d+)\.png$')

EXPECTED_RES = {
    'originals': (112, 112),
    'baseline':  (512, 512),
    'unmapped':  (512, 512),
    'mapped':    (512, 512),
}


def main():
    print("=" * 60)
    print("Phase 1 — Input Verification")
    print("=" * 60)

    all_ok = True
    results = {}

    # ----------------------------------------------------------
    # Check 1: Directory existence and file counts
    # ----------------------------------------------------------
    print("\n1. Checking directory existence and file counts...")
    dir_files = {}
    for label, dirpath in DIRS.items():
        if not os.path.isdir(dirpath):
            print(f"   FAIL  {label}: directory does not exist: {dirpath}")
            all_ok = False
            dir_files[label] = []
            continue
        files = sorted([f for f in os.listdir(dirpath) if f.endswith('.png')])
        dir_files[label] = files
        count = len(files)
        status = "OK" if count == EXPECTED_COUNT else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"   {status:4s}  {label}: {count} PNG files (expected {EXPECTED_COUNT})")

    results['file_counts'] = {k: len(v) for k, v in dir_files.items()}

    # ----------------------------------------------------------
    # Check 2: All images readable and not zero-byte
    # ----------------------------------------------------------
    print("\n2. Checking all images are readable (not corrupt / zero-byte)...")
    corrupt_files = []
    zero_byte_files = []
    for label, dirpath in DIRS.items():
        for fname in dir_files[label]:
            fpath = os.path.join(dirpath, fname)
            fsize = os.path.getsize(fpath)
            if fsize == 0:
                zero_byte_files.append(fpath)
                continue
            try:
                img = Image.open(fpath)
                img.verify()  # verify integrity without fully loading
            except Exception as e:
                corrupt_files.append((fpath, str(e)))

    if zero_byte_files:
        all_ok = False
        print(f"   FAIL  {len(zero_byte_files)} zero-byte files found:")
        for f in zero_byte_files[:5]:
            print(f"         {f}")
    else:
        print(f"   OK    No zero-byte files.")

    if corrupt_files:
        all_ok = False
        print(f"   FAIL  {len(corrupt_files)} corrupt files found:")
        for f, e in corrupt_files[:5]:
            print(f"         {f}: {e}")
    else:
        print(f"   OK    All {sum(len(v) for v in dir_files.values())} images are readable.")

    results['zero_byte'] = len(zero_byte_files)
    results['corrupt'] = len(corrupt_files)

    # ----------------------------------------------------------
    # Check 3: Resolution verification
    # ----------------------------------------------------------
    print("\n3. Checking image resolutions...")
    resolution_errors = []
    for label, dirpath in DIRS.items():
        expected_w, expected_h = EXPECTED_RES[label]
        for fname in dir_files[label]:
            fpath = os.path.join(dirpath, fname)
            try:
                img = Image.open(fpath)
                w, h = img.size
                if w != expected_w or h != expected_h:
                    resolution_errors.append((fpath, (w, h), (expected_w, expected_h)))
            except Exception:
                pass  # already caught in check 2

    if resolution_errors:
        all_ok = False
        print(f"   FAIL  {len(resolution_errors)} resolution mismatches:")
        for f, actual, expected in resolution_errors[:5]:
            print(f"         {f}: got {actual}, expected {expected}")
    else:
        print(f"   OK    originals: all 112×112")
        print(f"   OK    baseline:  all 512×512")
        print(f"   OK    unmapped:  all 512×512")
        print(f"   OK    mapped:    all 512×512")

    results['resolution_errors'] = len(resolution_errors)

    # ----------------------------------------------------------
    # Check 4: Filename matching across directories
    # ----------------------------------------------------------
    print("\n4. Checking filename matching across directories...")
    orig_set = set(dir_files['originals'])
    missing = {}
    for label in ['baseline', 'unmapped', 'mapped']:
        other_set = set(dir_files[label])
        in_orig_not_other = orig_set - other_set
        in_other_not_orig = other_set - orig_set
        if in_orig_not_other or in_other_not_orig:
            all_ok = False
            missing[label] = {
                'missing_from_dir': sorted(in_orig_not_other),
                'extra_in_dir': sorted(in_other_not_orig),
            }
            print(f"   FAIL  {label}: {len(in_orig_not_other)} missing, {len(in_other_not_orig)} extra")
        else:
            print(f"   OK    {label}: all 500 filenames match originals")

    results['filename_mismatches'] = {k: len(v.get('missing_from_dir', []))
                                       for k, v in missing.items()} if missing else {}

    # ----------------------------------------------------------
    # Check 5: Filename pattern parsing
    # ----------------------------------------------------------
    print("\n5. Parsing identity labels from filenames...")
    parsed = []
    malformed = []
    for fname in dir_files['originals']:
        m = FILENAME_PATTERN.match(fname)
        if m:
            identity = m.group(1)
            img_idx = int(m.group(2))
            parsed.append((fname, identity, img_idx))
        else:
            malformed.append(fname)

    if malformed:
        all_ok = False
        print(f"   FAIL  {len(malformed)} filenames do not match pattern id_XXXXX_imgN.png:")
        for f in malformed[:5]:
            print(f"         {f}")
    else:
        print(f"   OK    All {len(parsed)} filenames match pattern id_XXXXX_imgN.png")

    results['malformed_filenames'] = len(malformed)

    # ----------------------------------------------------------
    # Check 6: Identity grouping
    # ----------------------------------------------------------
    print("\n6. Verifying identity grouping...")
    id_to_images = defaultdict(list)
    for fname, identity, img_idx in parsed:
        id_to_images[identity].append((fname, img_idx))

    n_ids = len(id_to_images)
    wrong_count_ids = {k: len(v) for k, v in id_to_images.items()
                       if len(v) != EXPECTED_IMAGES_PER_ID}

    if n_ids != EXPECTED_IDENTITIES:
        all_ok = False
        print(f"   FAIL  Found {n_ids} unique identities (expected {EXPECTED_IDENTITIES})")
    else:
        print(f"   OK    {n_ids} unique identities")

    if wrong_count_ids:
        all_ok = False
        print(f"   FAIL  {len(wrong_count_ids)} identities have wrong image count:")
        for k, v in list(wrong_count_ids.items())[:5]:
            print(f"         {k}: {v} images (expected {EXPECTED_IMAGES_PER_ID})")
    else:
        print(f"   OK    All identities have exactly {EXPECTED_IMAGES_PER_ID} images")

    # Check for duplicate img indices within an identity
    dup_idx = {}
    for identity, items in id_to_images.items():
        indices = [idx for _, idx in items]
        if len(indices) != len(set(indices)):
            dup_idx[identity] = indices

    if dup_idx:
        all_ok = False
        print(f"   FAIL  {len(dup_idx)} identities have duplicate image indices")
    else:
        print(f"   OK    No duplicate image indices within any identity")

    results['identities'] = n_ids
    results['wrong_image_count_ids'] = len(wrong_count_ids)
    results['duplicate_indices'] = len(dup_idx)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    if all_ok:
        print("Phase 1 RESULT: ✅ PASS — All checks passed.")
        print("Safe to proceed to Phase 2 (Embedding Extraction).")
    else:
        print("Phase 1 RESULT: ❌ FAIL — Issues found above.")
        print("Fix all issues before proceeding to Phase 2.")
    print("=" * 60)

    # Save results JSON for downstream use
    results['overall'] = 'PASS' if all_ok else 'FAIL'
    results_path = 'outputs/evaluation/phase1_verification.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save parsed image list for Phase 2
    if all_ok:
        manifest_path = 'outputs/evaluation/phase1_image_manifest.json'
        manifest = []
        for fname, identity, img_idx in sorted(parsed):
            manifest.append({
                'filename': fname,
                'identity': identity,
                'img_index': img_idx,
            })
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Image manifest saved to: {manifest_path} ({len(manifest)} entries)")

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
