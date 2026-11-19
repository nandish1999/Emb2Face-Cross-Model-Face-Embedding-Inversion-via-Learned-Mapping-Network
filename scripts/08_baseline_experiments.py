import os
import sys
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Set HF cache to project-local directory
os.environ['HF_HOME'] = os.path.join(PROJECT_ROOT, '.cache', 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(PROJECT_ROOT, '.cache', 'huggingface')

from scripts import preprocessing_utils as pu


# ============================================================
# Mapping Network (must match training architecture)
# ============================================================

import torch.nn as nn

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
    """Load the Arc2Face Stable Diffusion pipeline. Downloads SD-1.5 on first run."""
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler

    # Add arc2face module to path
    arc2face_path = os.path.join(PROJECT_ROOT, 'models', 'arc2face_repo')
    sys.path.insert(0, arc2face_path)
    from arc2face import CLIPTextModelWrapper, project_face_embs

    weights_dir = os.path.join(arc2face_path, 'arc2face_weights')
    base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'

    print("   Loading Arc2Face encoder...")
    encoder = CLIPTextModelWrapper.from_pretrained(
        weights_dir, subfolder="encoder", torch_dtype=torch.float32
    )

    print("   Loading Arc2Face UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        weights_dir, subfolder="arc2face", torch_dtype=torch.float32
    )

    print("   Loading Stable Diffusion v1.5 base pipeline...")
    print("   (First run will download ~4 GB to .cache/huggingface/)")
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
    # CPU mode - no .to("cuda")

    return pipeline, project_face_embs


def reconstruct_face(pipeline, project_fn, embedding_512, num_steps=25, guidance_scale=3.0):
    """Generate a face image from a 512-dim ArcFace-compatible embedding.

    Args:
        pipeline: StableDiffusionPipeline with Arc2Face encoder+unet
        project_fn: project_face_embs function from arc2face
        embedding_512: numpy array (512,) or torch tensor
        num_steps: diffusion steps (25 default, lower = faster but worse)
        guidance_scale: classifier-free guidance (3.0 per Arc2Face paper)

    Returns:
        PIL.Image
    """
    if isinstance(embedding_512, np.ndarray):
        id_emb = torch.tensor(embedding_512, dtype=torch.float32).unsqueeze(0)
    else:
        id_emb = embedding_512.unsqueeze(0).float()

    id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)
    id_emb = project_fn(pipeline, id_emb)

    images = pipeline(
        prompt_embeds=id_emb,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1
    ).images

    return images[0]


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Baseline experiments B1/B2/B3')
    parser.add_argument('--n-ids', type=int, default=3,
                        help='Number of eval identities to process (default: 3)')
    parser.add_argument('--n-imgs', type=int, default=1,
                        help='Number of images per identity (default: 1)')
    parser.add_argument('--steps', type=int, default=25,
                        help='Diffusion inference steps (default: 25)')
    parser.add_argument('--checkpoint', type=str,
                        default='models/mapping_network/checkpoints/mapping_full.pt',
                        help='Mapping network checkpoint')
    parser.add_argument('--eval-csv', type=str,
                        default='data/eval_metadata.csv',
                        help='Evaluation metadata CSV')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/reconstructed',
                        help='Output directory root')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Step 8: Baseline Experiments (B1, B2, B3)")
    print("=" * 70)
    print(f"Identities:  {args.n_ids}")
    print(f"Images/id:   {args.n_imgs}")
    print(f"Total images: {args.n_ids * args.n_imgs}")
    print(f"Steps:        {args.steps}")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Time:         {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    torch.manual_seed(args.seed)

    # ============================================================
    # 1. Load evaluation metadata
    # ============================================================
    print("1. Loading evaluation metadata...")
    with open(args.eval_csv) as f:
        all_rows = list(csv.DictReader(f))

    # Group by identity
    id_map = {}
    for row in all_rows:
        identity = row['identity_id']
        if identity not in id_map:
            id_map[identity] = []
        id_map[identity].append(row)

    # Select identities
    selected_ids = sorted(id_map.keys())[:args.n_ids]
    print(f"   Selected {len(selected_ids)} identities: {selected_ids}")

    # Build image list
    images_to_process = []
    for identity in selected_ids:
        imgs = id_map[identity][:args.n_imgs]
        images_to_process.extend(imgs)

    total_images = len(images_to_process)
    print(f"   Total images to reconstruct: {total_images}")
    print(f"   Total reconstructions: {total_images * 3} (B1 + B2 + B3)")

    # ============================================================
    # 2. Load embedding models
    # ============================================================
    print("\n2. Loading embedding models...")
    t0 = time.time()

    print("   Loading AdaFace IR-50...")
    adaface_model = pu.load_adaface_model()

    print("   Loading ArcFace (arcface.onnx)...")
    arcface_session = pu.load_arcface_session()

    print(f"   Embedding models loaded in {time.time() - t0:.1f}s")

    # ============================================================
    # 3. Load mapping network
    # ============================================================
    print("\n3. Loading mapping network...")
    mapping_net = MappingNetwork()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    mapping_net.load_state_dict(ckpt['model_state_dict'])
    mapping_net.eval()
    print(f"   Loaded from epoch {ckpt['epoch']}, val_cos={ckpt['val_cos_mean']:.4f}")

    # ============================================================
    # 4. Load Arc2Face pipeline
    # ============================================================
    print("\n4. Loading Arc2Face pipeline...")
    t1 = time.time()
    pipeline, project_face_embs = load_arc2face_pipeline()
    print(f"   Arc2Face pipeline loaded in {time.time() - t1:.1f}s")

    # ============================================================
    # 5. Create output directories
    # ============================================================
    dirs = {
        'baseline': os.path.join(args.output_dir, 'baseline'),
        'unmapped': os.path.join(args.output_dir, 'unmapped'),
        'mapped': os.path.join(args.output_dir, 'mapped'),
        'originals': os.path.join(args.output_dir, 'originals'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Metrics CSV
    metrics_path = os.path.join(args.output_dir, 'baseline_metrics.csv')
    metrics_file = open(metrics_path, 'w', newline='')
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow([
        'identity', 'image_idx', 'image_path',
        'cos_arcface_real', 'cos_adaface_mapped',
        'b1_time_s', 'b2_time_s', 'b3_time_s'
    ])

    # ============================================================
    # 6. Run experiments
    # ============================================================
    print(f"\n5. Running experiments ({total_images} images x 3 baselines)...")
    print(f"   Estimated time per image: 2-10 min on CPU")
    print(f"   Estimated total: {total_images * 3 * 5 / 60:.0f}-{total_images * 3 * 10 / 60:.0f} min")
    print()

    all_b1_cos = []
    all_b2_cos = []
    all_b3_cos = []

    for i, row in enumerate(images_to_process):
        identity = row['identity_id']
        img_idx = row['image_index']
        img_path = row['image_path']
        fname = f"{identity}_img{img_idx}"

        print(f"   [{i+1}/{total_images}] {fname}")

        # --- Extract embeddings ---
        rgb, bgr = pu.load_image_for_both(img_path)
        adaface_emb = pu.get_adaface_embedding(adaface_model, rgb)
        arcface_emb = pu.get_arcface_embedding_direct(arcface_session, bgr)

        # Save original image copy for visual comparison
        from PIL import Image
        orig = Image.open(img_path).convert('RGB')
        orig.save(os.path.join(dirs['originals'], f"{fname}.png"))

        # --- B1: Real ArcFace -> Arc2Face ---
        t_b1 = time.time()
        b1_img = reconstruct_face(pipeline, project_face_embs, arcface_emb, num_steps=args.steps)
        t_b1 = time.time() - t_b1
        b1_img.save(os.path.join(dirs['baseline'], f"{fname}.png"))
        print(f"       B1 (ArcFace->Arc2Face): {t_b1:.1f}s")

        # --- B2: Raw AdaFace -> Arc2Face (no mapping) ---
        t_b2 = time.time()
        b2_img = reconstruct_face(pipeline, project_face_embs, adaface_emb, num_steps=args.steps)
        t_b2 = time.time() - t_b2
        b2_img.save(os.path.join(dirs['unmapped'], f"{fname}.png"))
        print(f"       B2 (AdaFace->Arc2Face): {t_b2:.1f}s")

        # --- B3: AdaFace -> Mapping -> Arc2Face ---
        t_b3 = time.time()
        with torch.no_grad():
            ada_tensor = torch.tensor(adaface_emb, dtype=torch.float32).unsqueeze(0)
            mapped_emb = mapping_net(ada_tensor).squeeze(0).numpy()
        b3_img = reconstruct_face(pipeline, project_face_embs, mapped_emb, num_steps=args.steps)
        t_b3 = time.time() - t_b3
        b3_img.save(os.path.join(dirs['mapped'], f"{fname}.png"))
        print(f"       B3 (Mapped->Arc2Face):  {t_b3:.1f}s")

        # --- Compute cosine similarity between mapped embedding and real ArcFace ---
        cos_real = float(F.cosine_similarity(
            torch.tensor(arcface_emb).unsqueeze(0),
            torch.tensor(arcface_emb).unsqueeze(0)
        ))  # Always 1.0, for reference
        cos_mapped = float(F.cosine_similarity(
            torch.tensor(mapped_emb).unsqueeze(0),
            torch.tensor(arcface_emb).unsqueeze(0)
        ))

        all_b1_cos.append(cos_real)
        all_b2_cos.append(0.0)  # AdaFace != ArcFace space, not comparable
        all_b3_cos.append(cos_mapped)

        metrics_writer.writerow([
            identity, img_idx, img_path,
            f'{cos_real:.4f}', f'{cos_mapped:.4f}',
            f'{t_b1:.1f}', f'{t_b2:.1f}', f'{t_b3:.1f}'
        ])
        metrics_file.flush()

        print(f"       Mapped-vs-Real cos: {cos_mapped:.4f}")
        print()

    metrics_file.close()

    # ============================================================
    # 7. Summary
    # ============================================================
    total_time_b1 = sum(float(1) for _ in all_b1_cos)  # placeholder
    avg_b3_cos = np.mean(all_b3_cos) if all_b3_cos else 0

    print("=" * 70)
    print("STEP 8 SUMMARY")
    print("=" * 70)
    print(f"Images processed:    {total_images}")
    print(f"Identities:          {len(selected_ids)}")
    print(f"")
    print(f"B3 mapped-vs-real cosine: {avg_b3_cos:.4f} (mean)")
    print(f"")
    print(f"Output directories:")
    print(f"  Originals: {dirs['originals']}")
    print(f"  B1 (baseline):  {dirs['baseline']}")
    print(f"  B2 (unmapped):  {dirs['unmapped']}")
    print(f"  B3 (mapped):    {dirs['mapped']}")
    print(f"  Metrics CSV:    {metrics_path}")
    print(f"")
    print(f"Visual comparison:")
    print(f"  Compare originals/ vs baseline/ vs unmapped/ vs mapped/")
    print(f"  B1 should look like the person (proves Arc2Face works)")
    print(f"  B2 should NOT look like the person (proves mapping needed)")
    print(f"  B3 should partially resemble the person (our method)")
    print(f"")
    print(f"Done at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
