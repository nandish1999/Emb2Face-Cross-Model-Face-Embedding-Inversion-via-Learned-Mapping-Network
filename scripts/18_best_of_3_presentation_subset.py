import os
import sys
import csv
import time
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

os.environ['HF_HOME'] = os.path.join(PROJECT_ROOT, '.cache', 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(PROJECT_ROOT, '.cache', 'huggingface')

from scripts import preprocessing_utils as pu


# ============================================================
# Config defaults
# ============================================================
CHECKPOINT = 'models/mapping_network/checkpoints/mapping_full_official_safe.pt'
EVAL_CSV = 'data/eval_metadata.csv'
OUTPUT_DIR = 'outputs/reconstructed_bestof3_presentation'
METRICS_CSV = 'outputs/evaluation/bestof3_presentation_metrics.csv'
NUM_STEPS = 25
GUIDANCE_SCALE = 3.0


# ============================================================
# Mapping Network (must match original training architecture)
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
# Arc2Face Pipeline (same as 11_ script)
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
        base_model, text_encoder=encoder, unet=unet,
        torch_dtype=torch.float32, safety_checker=None
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    return pipeline, project_face_embs


def reconstruct_face(pipeline, project_fn, embedding_512, seed=42,
                     num_steps=25, guidance_scale=3.0):
    """Generate a face image from a 512-dim embedding."""
    id_emb = torch.tensor(embedding_512, dtype=torch.float32).unsqueeze(0)
    id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)
    id_emb = project_fn(pipeline, id_emb)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    images = pipeline(
        prompt_embeds=id_emb,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        generator=generator
    ).images
    return images[0]


def compute_arcface_cosine(arcface_session, original_path, generated_path):
    """Compute internal ArcFace cosine between original and generated.

    Handles both 112x112 eval images and 512x512 generated images.
    Generated images are resized to 112x112 for ArcFace.
    """
    try:
        import cv2
        # Original: 112x112 eval image
        bgr_orig = pu.load_image_bgr(original_path)
        emb_orig = pu.get_arcface_embedding_direct(arcface_session, bgr_orig)

        # Generated: 512x512 -> resize to 112x112 for ArcFace
        bgr_gen = cv2.imread(generated_path)
        if bgr_gen is None:
            return None
        if bgr_gen.shape[:2] != (112, 112):
            bgr_gen = cv2.resize(bgr_gen, (112, 112), interpolation=cv2.INTER_AREA)
        emb_gen = pu.get_arcface_embedding_direct(arcface_session, bgr_gen)

        cos = float(np.dot(emb_orig, emb_gen) /
                     (np.linalg.norm(emb_orig) * np.linalg.norm(emb_gen)))
        return cos
    except Exception as e:
        print("    WARNING: ArcFace cosine failed: " + str(e))
        return None


# ============================================================
# Argument Parser
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description='Best-of-N seed generation for presentation')
    p.add_argument('--images', type=str, required=True,
                   help='Comma-separated image names, e.g. id_1122_img0,id_1158_img2')
    p.add_argument('--seeds', type=str, default='23,42,100',
                   help='Comma-separated seeds (default: 23,42,100)')
    p.add_argument('--mapper', type=str, default=CHECKPOINT,
                   help='Mapper checkpoint path')
    p.add_argument('--output', type=str, default=OUTPUT_DIR,
                   help='Output directory')
    p.add_argument('--metrics', type=str, default=METRICS_CSV,
                   help='Metrics CSV path')
    p.add_argument('--steps', type=int, default=NUM_STEPS,
                   help='Diffusion steps (default: 25)')
    p.add_argument('--guidance-scale', type=float, default=GUIDANCE_SCALE,
                   help='Guidance scale (default: 3.0)')
    p.add_argument('--force', action='store_true',
                   help='Force overwrite existing selected outputs')
    return p.parse_args()


def main():
    args = parse_args()
    image_names = [s.strip() for s in args.images.split(',') if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print("=" * 70)
    print("  BEST-OF-N SEED GENERATION — Presentation Subset")
    print("=" * 70)
    print("  Images:     " + str(len(image_names)))
    print("  Seeds:      " + str(seeds))
    print("  Mapper:     " + args.mapper)
    print("  Output:     " + args.output)
    print("  Steps:      " + str(args.steps))
    print("  Guidance:   " + str(args.guidance_scale))
    print("  Force:      " + str(args.force))
    print()

    # Version check
    import transformers
    assert transformers.__version__.startswith('4.'), \
        "WRONG transformers: " + transformers.__version__ + ". Use: conda activate arc2face_official"

    # Create output dirs
    dirs = {
        'candidates': os.path.join(args.output, 'candidates'),
        'selected': os.path.join(args.output, 'selected'),
        'originals': os.path.join(args.output, 'originals'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Load eval metadata to find image paths
    print("1. Loading evaluation metadata...")
    with open(EVAL_CSV) as f:
        all_rows = list(csv.DictReader(f))
    # Build lookup: "id_XXXX_imgN" -> row
    row_lookup = {}
    for row in all_rows:
        key = row['identity_id'] + "_img" + row['image_index']
        row_lookup[key] = row

    # Validate requested images
    valid_images = []
    for name in image_names:
        if name in row_lookup:
            valid_images.append(name)
        else:
            print("   WARNING: " + name + " not found in eval metadata, skipping.")
    if not valid_images:
        print("ERROR: No valid images found.")
        sys.exit(1)
    print("   Valid images: " + str(len(valid_images)) + "/" + str(len(image_names)))

    # Load models
    print("\n2. Loading embedding models...")
    adaface_model = pu.load_adaface_model()
    arcface_session = pu.load_arcface_session()

    print("\n3. Loading mapping network...")
    mapping_net = MappingNetwork()
    ckpt = torch.load(args.mapper, map_location='cpu', weights_only=False)
    mapping_net.load_state_dict(ckpt['model_state_dict'])
    mapping_net.eval()
    print("   Loaded: epoch " + str(ckpt['epoch']) + ", val_cos=" + str(round(ckpt['val_cos_mean'], 4)))

    print("\n4. Loading Arc2Face pipeline...")
    t0 = time.time()
    pipeline, project_face_embs = load_arc2face_pipeline()
    print("   Pipeline loaded in " + str(round(time.time() - t0, 1)) + "s")

    # Metrics CSV
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    csv_file = open(args.metrics, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'image_name', 'identity', 'image_index', 'seed',
        'arcface_cosine', 'selected', 'selected_reason', 'candidate_path'
    ])

    # Process each image
    print("\n5. Generating candidates...\n")
    results = []  # list of {name, selected_seed, selected_cos, seed42_cos}
    total_generated = 0
    total_skipped = 0

    for img_i, img_name in enumerate(valid_images):
        row = row_lookup[img_name]
        identity = row['identity_id']
        img_idx = row['image_index']
        img_path = row['image_path']

        print("-" * 60)
        print("Image " + str(img_i + 1) + "/" + str(len(valid_images)) + ": " + img_name)

        # Copy original
        orig_dst = os.path.join(dirs['originals'], img_name + ".png")
        if not os.path.exists(orig_dst):
            from PIL import Image
            orig_img = Image.open(img_path).convert('RGB')
            orig_img.save(orig_dst)
            print("  Original saved.")

        # Extract AdaFace embedding
        rgb_img = pu.load_image_rgb(img_path)
        adaface_emb = pu.get_adaface_embedding(adaface_model, rgb_img)
        if adaface_emb is None:
            print("  ERROR: Could not extract AdaFace embedding. Skipping.")
            continue

        # Map through network
        with torch.no_grad():
            ada_t = torch.tensor(adaface_emb, dtype=torch.float32).unsqueeze(0)
            mapped_emb = mapping_net(ada_t).squeeze(0).numpy()

        # Generate candidates for each seed
        candidates = []  # (seed, path, arcface_cos)
        for seed in seeds:
            cand_path = os.path.join(dirs['candidates'],
                                     img_name + "_seed" + str(seed) + ".png")

            # Resume: skip if candidate already exists
            if os.path.exists(cand_path):
                print("  Seed " + str(seed) + ": EXISTS (reusing)")
                total_skipped += 1
            else:
                print("  Seed " + str(seed) + ": generating...", end="", flush=True)
                t1 = time.time()
                face_img = reconstruct_face(
                    pipeline, project_face_embs, mapped_emb,
                    seed=seed, num_steps=args.steps,
                    guidance_scale=args.guidance_scale
                )
                face_img.save(cand_path)
                dt = time.time() - t1
                print(" done (" + str(round(dt, 1)) + "s)")
                total_generated += 1

            # Compute ArcFace cosine
            arc_cos = compute_arcface_cosine(arcface_session, img_path, cand_path)
            cos_str = str(round(arc_cos, 4)) if arc_cos is not None else "N/A"
            print("    ArcFace cosine: " + cos_str)

            candidates.append({
                'seed': seed, 'path': cand_path, 'arcface_cos': arc_cos
            })

        # Select best candidate
        valid_cands = [c for c in candidates if c['arcface_cos'] is not None]
        if valid_cands:
            best = max(valid_cands, key=lambda c: c['arcface_cos'])
            reason = "highest_arcface_cosine"
        elif candidates:
            best = candidates[0]
            reason = "fallback_first_seed"
        else:
            print("  ERROR: No candidates generated.")
            continue

        # Save selected
        sel_path = os.path.join(dirs['selected'], img_name + ".png")
        if os.path.exists(sel_path) and not args.force:
            print("  Selected already exists (use --force to overwrite).")
        else:
            shutil.copy2(best['path'], sel_path)

        print("  SELECTED: seed " + str(best['seed']) +
              " (cosine=" + str(round(best['arcface_cos'], 4) if best['arcface_cos'] else 'N/A') + ")")

        # Find seed-42 cosine for comparison
        seed42_cos = None
        for c in candidates:
            if c['seed'] == 42:
                seed42_cos = c['arcface_cos']

        results.append({
            'name': img_name,
            'selected_seed': best['seed'],
            'selected_cos': best['arcface_cos'],
            'seed42_cos': seed42_cos,
        })

        # Write all candidates to CSV
        for c in candidates:
            is_selected = 'Y' if c['seed'] == best['seed'] else 'N'
            csv_writer.writerow([
                img_name, identity, img_idx, c['seed'],
                round(c['arcface_cos'], 6) if c['arcface_cos'] is not None else 'N/A',
                is_selected,
                reason if is_selected == 'Y' else '',
                c['path']
            ])

        csv_file.flush()

    csv_file.close()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  BEST-OF-" + str(len(seeds)) + " GENERATION COMPLETE")
    print("=" * 70)
    print("  Images processed:     " + str(len(results)))
    print("  Candidates generated: " + str(total_generated))
    print("  Candidates reused:    " + str(total_skipped))
    print("  Total candidates:     " + str(total_generated + total_skipped))
    print()

    # Seed distribution
    seed_counts = {}
    for r in results:
        s = r['selected_seed']
        seed_counts[s] = seed_counts.get(s, 0) + 1
    print("  Selected seed distribution:")
    for s in sorted(seed_counts):
        print("    Seed " + str(s) + ": " + str(seed_counts[s]) + " images")
    print()

    # Average selected cosine
    sel_cosines = [r['selected_cos'] for r in results if r['selected_cos'] is not None]
    s42_cosines = [r['seed42_cos'] for r in results if r['seed42_cos'] is not None]
    if sel_cosines:
        print("  Avg selected ArcFace cosine: " + str(round(np.mean(sel_cosines), 4)))
    if s42_cosines:
        print("  Avg seed-42 ArcFace cosine:  " + str(round(np.mean(s42_cosines), 4)))
        if sel_cosines:
            diff = np.mean(sel_cosines) - np.mean(s42_cosines)
            sign = "+" if diff >= 0 else ""
            print("  Best-of-" + str(len(seeds)) + " improvement:     " +
                  sign + str(round(diff, 4)))
    print()
    print("  Metrics:  " + args.metrics)
    print("  Selected: " + dirs['selected'])
    print("=" * 70)


if __name__ == "__main__":
    main()
