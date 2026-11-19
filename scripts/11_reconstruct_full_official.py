import os
import sys
import csv
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Set HF cache to project-local
os.environ['HF_HOME'] = os.path.join(PROJECT_ROOT, '.cache', 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(PROJECT_ROOT, '.cache', 'huggingface')

from scripts import preprocessing_utils as pu


# ============================================================
# Config
# ============================================================
OUTPUT_DIR = 'outputs/reconstructed_official_full'
CHECKPOINT = 'models/mapping_network/checkpoints/mapping_full_official_safe.pt'
EVAL_CSV = 'data/eval_metadata.csv'
SEED = 42
NUM_STEPS = 25
GUIDANCE_SCALE = 3.0


# ============================================================
# Mapping Network (must match training architecture)
# ============================================================
class MappingNetwork(nn.Module):
    """2-Layer MLP: AdaFace -> ArcFace embedding space."""
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        out = self.net(x)
        out = F.normalize(out, p=2, dim=1)
        return out


# ============================================================
# Arc2Face Pipeline
# ============================================================
def load_arc2face_pipeline():
    """Load Arc2Face pipeline with float32 for CPU."""
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler

    arc2face_path = os.path.join(PROJECT_ROOT, 'models', 'arc2face_repo')
    sys.path.insert(0, arc2face_path)
    from arc2face import CLIPTextModelWrapper, project_face_embs

    weights_dir = os.path.join(arc2face_path, 'arc2face_weights')
    base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'

    print("   Loading Arc2Face encoder (float32)...")
    encoder = CLIPTextModelWrapper.from_pretrained(
        weights_dir, subfolder="encoder", torch_dtype=torch.float32
    )

    print("   Loading Arc2Face UNet (float32)...")
    unet = UNet2DConditionModel.from_pretrained(
        weights_dir, subfolder="arc2face", torch_dtype=torch.float32
    )

    print("   Loading SD v1.5 base pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=torch.float32,
        safety_checker=None
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    return pipeline, project_face_embs


def reconstruct_face(pipeline, project_fn, embedding_512, seed=42):
    """Generate a face image from a 512-dim embedding with fixed seed."""
    id_emb = torch.tensor(embedding_512, dtype=torch.float32).unsqueeze(0)
    id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)
    id_emb = project_fn(pipeline, id_emb)

    generator = torch.Generator(device='cpu').manual_seed(seed)

    images = pipeline(
        prompt_embeds=id_emb,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        num_images_per_prompt=1,
        generator=generator
    ).images

    return images[0]


# ============================================================
# Main
# ============================================================
def main():
    import transformers, diffusers

    print("=" * 70)
    print("Step 11: Full Official Reconstruction (500 images, resume-safe)")
    print("=" * 70)
    print(f"Time:          {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"torch:         {torch.__version__}")
    print(f"transformers:  {transformers.__version__}")
    print(f"diffusers:     {diffusers.__version__}")
    print(f"Output:        {OUTPUT_DIR}")
    print(f"Checkpoint:    {CHECKPOINT}")
    print(f"Seed:          {SEED}")
    print(f"Steps:         {NUM_STEPS}")
    print()

    # Version check
    assert transformers.__version__.startswith('4.'), \
        f"WRONG transformers: {transformers.__version__}. Use: conda activate arc2face_official"

    # ============================================================
    # 1. Load evaluation metadata
    # ============================================================
    print("1. Loading evaluation metadata...")
    with open(EVAL_CSV) as f:
        all_rows = list(csv.DictReader(f))
    total_images = len(all_rows)
    print(f"   Total eval images: {total_images}")

    # ============================================================
    # 2. Create output directories
    # ============================================================
    dirs = {
        'originals': os.path.join(OUTPUT_DIR, 'originals'),
        'baseline': os.path.join(OUTPUT_DIR, 'baseline'),
        'unmapped': os.path.join(OUTPUT_DIR, 'unmapped'),
        'mapped': os.path.join(OUTPUT_DIR, 'mapped'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # ============================================================
    # 3. Check which images already exist (resume support)
    # ============================================================
    print("\n2. Checking existing outputs for resume...")
    to_process = []
    already_done = 0

    for row in all_rows:
        identity = row['identity_id']
        img_idx = row['image_index']
        fname = f"{identity}_img{img_idx}.png"

        # Check if ALL 3 outputs exist
        b1_exists = os.path.exists(os.path.join(dirs['baseline'], fname))
        b2_exists = os.path.exists(os.path.join(dirs['unmapped'], fname))
        b3_exists = os.path.exists(os.path.join(dirs['mapped'], fname))

        if b1_exists and b2_exists and b3_exists:
            already_done += 1
        else:
            to_process.append(row)

    remaining = len(to_process)
    print(f"   Already completed: {already_done}/{total_images}")
    print(f"   Remaining:         {remaining}/{total_images}")

    if remaining == 0:
        print("\n   ALL IMAGES ALREADY PROCESSED. Nothing to do.")
        print("   Run evaluation scripts to generate reports.")
        return

    est_min = remaining * 3 * 1.5  # ~1.5 min per reconstruction on CPU
    est_hours = est_min / 60
    print(f"   Estimated time:    {est_hours:.1f} hours ({est_min:.0f} min)")
    print()

    # ============================================================
    # 4. Load models
    # ============================================================
    print("3. Loading embedding models...")
    t0 = time.time()

    print("   Loading AdaFace IR-50...")
    adaface_model = pu.load_adaface_model()

    print("   Loading ArcFace (arcface.onnx)...")
    arcface_session = pu.load_arcface_session()

    print(f"   Embedding models loaded in {time.time() - t0:.1f}s")

    # ============================================================
    # 5. Load mapping network
    # ============================================================
    print("\n4. Loading mapping network...")
    mapping_net = MappingNetwork()
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    mapping_net.load_state_dict(ckpt['model_state_dict'])
    mapping_net.eval()
    print(f"   Loaded from epoch {ckpt['epoch']}, val_cos={ckpt['val_cos_mean']:.4f}")

    # ============================================================
    # 6. Load Arc2Face pipeline
    # ============================================================
    print("\n5. Loading Arc2Face pipeline...")
    t1 = time.time()
    pipeline, project_face_embs = load_arc2face_pipeline()
    print(f"   Pipeline loaded in {time.time() - t1:.1f}s")

    # ============================================================
    # 7. Metrics CSV (append mode for resume)
    # ============================================================
    metrics_path = os.path.join(OUTPUT_DIR, 'reconstruction_metrics.csv')
    metrics_header = [
        'identity', 'image_idx', 'image_path',
        'cos_mapped_vs_real', 'b1_time_s', 'b2_time_s', 'b3_time_s',
        'timestamp'
    ]

    # Load existing metrics to avoid duplicates
    existing_metrics = set()
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_metrics.add(f"{row['identity']}_{row['image_idx']}")

    write_header = not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0
    metrics_file = open(metrics_path, 'a', newline='')
    metrics_writer = csv.writer(metrics_file)
    if write_header:
        metrics_writer.writerow(metrics_header)
        metrics_file.flush()

    # ============================================================
    # 8. Process images
    # ============================================================
    from PIL import Image

    print(f"\n6. Processing {remaining} images (B1 + B2 + B3 each)...")
    print(f"   Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    completed = 0
    errors = 0
    start_time = time.time()

    for i, row in enumerate(to_process):
        identity = row['identity_id']
        img_idx = row['image_index']
        img_path = row['image_path']
        fname = f"{identity}_img{img_idx}"
        progress = already_done + completed + 1

        print(f"   [{progress}/{total_images}] {fname} (remaining: {remaining - completed})")

        try:
            # --- Extract embeddings ---
            rgb, bgr = pu.load_image_for_both(img_path)
            adaface_emb = pu.get_adaface_embedding(adaface_model, rgb)
            arcface_emb = pu.get_arcface_embedding_direct(arcface_session, bgr)

            # Save original
            orig_path = os.path.join(dirs['originals'], f"{fname}.png")
            if not os.path.exists(orig_path):
                orig = Image.open(img_path).convert('RGB')
                orig.save(orig_path)

            # --- B1: Real ArcFace -> Arc2Face ---
            b1_path = os.path.join(dirs['baseline'], f"{fname}.png")
            if os.path.exists(b1_path):
                t_b1 = 0.0
                print(f"       B1: already exists, skipping")
            else:
                t_b1_start = time.time()
                b1_img = reconstruct_face(pipeline, project_face_embs, arcface_emb, seed=SEED)
                t_b1 = time.time() - t_b1_start
                b1_img.save(b1_path)
                print(f"       B1 (ArcFace->Arc2Face): {t_b1:.1f}s")

            # --- B2: Raw AdaFace -> Arc2Face ---
            b2_path = os.path.join(dirs['unmapped'], f"{fname}.png")
            if os.path.exists(b2_path):
                t_b2 = 0.0
                print(f"       B2: already exists, skipping")
            else:
                t_b2_start = time.time()
                b2_img = reconstruct_face(pipeline, project_face_embs, adaface_emb, seed=SEED)
                t_b2 = time.time() - t_b2_start
                b2_img.save(b2_path)
                print(f"       B2 (AdaFace->Arc2Face): {t_b2:.1f}s")

            # --- B3: AdaFace -> Mapping -> Arc2Face ---
            b3_path = os.path.join(dirs['mapped'], f"{fname}.png")
            if os.path.exists(b3_path):
                t_b3 = 0.0
                print(f"       B3: already exists, skipping")
            else:
                t_b3_start = time.time()
                with torch.no_grad():
                    ada_tensor = torch.tensor(adaface_emb, dtype=torch.float32).unsqueeze(0)
                    mapped_emb = mapping_net(ada_tensor).squeeze(0).numpy()
                b3_img = reconstruct_face(pipeline, project_face_embs, mapped_emb, seed=SEED)
                t_b3 = time.time() - t_b3_start
                b3_img.save(b3_path)
                print(f"       B3 (Mapped->Arc2Face):  {t_b3:.1f}s")

            # --- Compute mapping cosine ---
            with torch.no_grad():
                ada_tensor = torch.tensor(adaface_emb, dtype=torch.float32).unsqueeze(0)
                mapped_emb = mapping_net(ada_tensor).squeeze(0).numpy()
            cos_mapped = float(F.cosine_similarity(
                torch.tensor(mapped_emb).unsqueeze(0),
                torch.tensor(arcface_emb).unsqueeze(0)
            ))

            # --- Write metrics ---
            metric_key = f"{identity}_{img_idx}"
            if metric_key not in existing_metrics:
                metrics_writer.writerow([
                    identity, img_idx, img_path,
                    f'{cos_mapped:.4f}',
                    f'{t_b1:.1f}', f'{t_b2:.1f}', f'{t_b3:.1f}',
                    time.strftime('%Y-%m-%d %H:%M:%S')
                ])
                metrics_file.flush()
                existing_metrics.add(metric_key)

            completed += 1

            # --- Progress estimate ---
            elapsed = time.time() - start_time
            rate = elapsed / completed if completed > 0 else 0
            eta_s = rate * (remaining - completed)
            eta_h = eta_s / 3600

            print(f"       cos_mapped={cos_mapped:.4f}  "
                  f"ETA: {eta_h:.1f}h  "
                  f"({completed}/{remaining} done)")
            print()

        except Exception as e:
            errors += 1
            print(f"       ERROR: {e}")
            traceback.print_exc()
            print(f"       Continuing to next image...")
            print()
            continue

    metrics_file.close()

    # ============================================================
    # 9. Summary
    # ============================================================
    total_elapsed = time.time() - start_time

    print("=" * 70)
    print("FULL RECONSTRUCTION SUMMARY")
    print("=" * 70)
    print(f"  Total eval images:     {total_images}")
    print(f"  Previously completed:  {already_done}")
    print(f"  Newly completed:       {completed}")
    print(f"  Errors:                {errors}")
    print(f"  Total completed:       {already_done + completed}/{total_images}")
    print(f"  Elapsed time:          {total_elapsed/3600:.1f}h ({total_elapsed:.0f}s)")
    print()
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Metrics: {metrics_path}")
    print()

    if already_done + completed == total_images:
        print("  ★★★ ALL 500 IMAGES COMPLETED ★★★")
        print("  Ready for evaluation.")
    elif errors > 0:
        print(f"  ⚠ {errors} errors occurred. Re-run to retry failed images.")

    print()
    print(f"  Done at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
