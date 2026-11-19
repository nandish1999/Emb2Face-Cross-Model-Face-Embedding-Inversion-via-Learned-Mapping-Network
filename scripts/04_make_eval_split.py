import os
import csv
import random
import shutil

# ============================================================
# Configuration
# ============================================================
DATASET_DIR = "data/raw/casia_webface/webface_112x112"
EVAL_DIR = "data/eval_ppt_500"
EVAL_META = "data/eval_metadata.csv"
TRAIN_META = "data/train_metadata.csv"
N_EVAL_IDS = 100
N_EVAL_IMGS_PER_ID = 5
RANDOM_SEED = 42
IMAGE_EXTENSIONS = ('.jpg', '.png', '.jpeg')

# ============================================================
# 1. Scan dataset
# ============================================================
print("Scanning dataset...")
identity_counts = {}
for identity_folder in sorted(os.listdir(DATASET_DIR)):
    folder_path = os.path.join(DATASET_DIR, identity_folder)
    if os.path.isdir(folder_path):
        n_images = len([f for f in os.listdir(folder_path)
                        if f.lower().endswith(IMAGE_EXTENSIONS)])
        identity_counts[identity_folder] = n_images

print(f"Total identities: {len(identity_counts)}")
print(f"Total images: {sum(identity_counts.values())}")
print(f"Images per identity: min={min(identity_counts.values())}, "
      f"max={max(identity_counts.values())}, "
      f"avg={sum(identity_counts.values()) / len(identity_counts):.1f}")

# ============================================================
# 2. Select eval identities (seeded random sample)
# ============================================================
eligible = {k: v for k, v in identity_counts.items() if v >= N_EVAL_IMGS_PER_ID}
print(f"Identities with >= {N_EVAL_IMGS_PER_ID} images: {len(eligible)}")

random.seed(RANDOM_SEED)
sorted_eligible = sorted(eligible.keys())
eval_ids_list = random.sample(sorted_eligible, N_EVAL_IDS)
eval_ids = set(eval_ids_list)
train_ids = set(identity_counts.keys()) - eval_ids

print(f"Eval identities: {len(eval_ids)}")
print(f"Train identities: {len(train_ids)}")

# ============================================================
# 3. Verify no overlap
# ============================================================
assert len(eval_ids & train_ids) == 0, "OVERLAP DETECTED!"
print("No overlap between train and eval. OK.")

# ============================================================
# 4. Create eval_ppt_500/
# ============================================================
os.makedirs(EVAL_DIR, exist_ok=True)
eval_records = []
for identity in sorted(eval_ids):
    src_dir = os.path.join(DATASET_DIR, identity)
    images = sorted([f for f in os.listdir(src_dir)
                     if f.lower().endswith(IMAGE_EXTENSIONS)])[:N_EVAL_IMGS_PER_ID]
    dst_dir = os.path.join(EVAL_DIR, identity)
    os.makedirs(dst_dir, exist_ok=True)
    for idx, img in enumerate(images):
        shutil.copy2(os.path.join(src_dir, img), os.path.join(dst_dir, img))
        eval_records.append({
            'identity_id': identity,
            'image_index': idx,
            'image_path': os.path.join(dst_dir, img),
            'source_path': os.path.join(src_dir, img)
        })

print(f"Eval images copied: {len(eval_records)}")

# ============================================================
# 5. Save eval_metadata.csv
# ============================================================
with open(EVAL_META, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=['identity_id', 'image_index', 'image_path', 'source_path'])
    writer.writeheader()
    writer.writerows(eval_records)
print(f"Saved {EVAL_META} ({len(eval_records)} rows)")

# ============================================================
# 6. Save train_metadata.csv
# ============================================================
train_records = []
for identity in sorted(train_ids):
    src_dir = os.path.join(DATASET_DIR, identity)
    for img in sorted(os.listdir(src_dir)):
        if img.lower().endswith(IMAGE_EXTENSIONS):
            train_records.append({
                'identity_id': identity,
                'image_path': os.path.join(src_dir, img)
            })
with open(TRAIN_META, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=['identity_id', 'image_path'])
    writer.writeheader()
    writer.writerows(train_records)
print(f"Saved {TRAIN_META} ({len(train_records)} rows)")

# ============================================================
# 7. Summary
# ============================================================
print()
print("=" * 60)
print("STEP 4 COMPLETE")
print("=" * 60)
print(f"Train: {len(train_ids)} identities, {len(train_records)} images")
print(f"Eval:  {len(eval_ids)} identities, {len(eval_records)} images")
print(f"Total: {len(train_ids) + len(eval_ids)} identities, "
      f"{len(train_records) + len(eval_records)} images")
print(f"Overlap: 0")
print("Done.")
