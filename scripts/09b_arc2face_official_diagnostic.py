import os
import sys
import csv
import time
import numpy as np
import cv2
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Set HF cache to project-local
os.environ['HF_HOME'] = os.path.join(PROJECT_ROOT, '.cache', 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(PROJECT_ROOT, '.cache', 'huggingface')

from scripts import preprocessing_utils as pu


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


def reconstruct_b1(pipeline, project_fn, embedding_512, seed, num_steps=25, guidance_scale=3.0):
    """Generate a face image from ArcFace embedding with fixed seed."""
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


def compute_arcface_cosine(arcface_session, orig_bgr_112, gen_pil_image):
    """Compute ArcFace cosine between original 112x112 and generated 512x512."""
    orig_emb = pu.get_arcface_embedding_direct(arcface_session, orig_bgr_112)
    gen_np = np.array(gen_pil_image)[:, :, ::-1]  # RGB → BGR
    gen_112 = cv2.resize(gen_np, (112, 112), interpolation=cv2.INTER_AREA)
    gen_emb = pu.get_arcface_embedding_direct(arcface_session, gen_112)
    return float(np.dot(orig_emb, gen_emb))


def main():
    import transformers, diffusers
    print("=" * 70)
    print("B1 Diagnostic — Official Arc2Face Environment")
    print("=" * 70)
    print(f"Time:         {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"torch:        {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"diffusers:    {diffusers.__version__}")
    print(f"Device:       CPU (float32)")
    print()

    # Verify correct versions
    assert transformers.__version__.startswith('4.'), \
        f"WRONG transformers version: {transformers.__version__}. Expected 4.36.0. Use: conda activate arc2face_official"
    assert diffusers.__version__.startswith('0.29'), \
        f"WRONG diffusers version: {diffusers.__version__}. Expected 0.29.2."

    # ============================================================
    # 1. Load test image
    # ============================================================
    test_image_path = 'data/eval_ppt_500/id_10092/10092_477517.jpg'
    print(f"1. Loading test image: {test_image_path}")
    orig_bgr = cv2.imread(test_image_path)
    print(f"   Shape: {orig_bgr.shape}")

    # ============================================================
    # 2. Load ArcFace recognizer
    # ============================================================
    print("\n2. Loading ArcFace (arcface.onnx) for embedding extraction...")
    arcface_session = pu.load_arcface_session()
    print("   Loaded.")

    # ============================================================
    # 3. Extract ground truth ArcFace embedding
    # ============================================================
    print("\n3. Extracting original ArcFace embedding...")
    orig_emb = pu.get_arcface_embedding_direct(arcface_session, orig_bgr)
    print(f"   Shape: {orig_emb.shape}, norm: {np.linalg.norm(orig_emb):.4f}")

    # ============================================================
    # 4. Load Arc2Face pipeline
    # ============================================================
    print("\n4. Loading Arc2Face pipeline...")
    t0 = time.time()
    pipeline, project_fn = load_arc2face_pipeline()
    print(f"   Pipeline loaded in {time.time() - t0:.1f}s")

    # ============================================================
    # 5. Generate B1 for multiple seeds (fp32 only)
    # ============================================================
    os.makedirs('outputs/diagnostic', exist_ok=True)
    os.makedirs('outputs/evaluation', exist_ok=True)

    seeds = [23, 42, 100, 200]
    results = []

    print(f"\n5. Generating B1 images for {len(seeds)} seeds...")
    print(f"   Estimated time: {len(seeds) * 2:.0f} min\n")

    for seed in seeds:
        label = f'official_fp32_seed{seed}'
        print(f"   [{label}] Generating B1 with seed={seed}...")

        t1 = time.time()
        gen_img = reconstruct_b1(pipeline, project_fn, orig_emb, seed)
        gen_time = time.time() - t1

        img_path = f'outputs/diagnostic/b1_{label}.png'
        gen_img.save(img_path)

        cosine = compute_arcface_cosine(arcface_session, orig_bgr, gen_img)

        print(f"   [{label}] cosine={cosine:.4f}  time={gen_time:.1f}s  saved={img_path}")

        results.append({
            'label': label,
            'env': 'arc2face_official',
            'transformers': transformers.__version__,
            'diffusers': diffusers.__version__,
            'dtype': 'fp32',
            'seed': seed,
            'cosine': f'{cosine:.4f}',
            'time_s': f'{gen_time:.1f}',
            'image_path': img_path
        })

    # ============================================================
    # 6. Save CSV
    # ============================================================
    csv_path = 'outputs/evaluation/arc2face_b1_official_env_diagnostic.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # ============================================================
    # 7. Summary
    # ============================================================
    print()
    print("=" * 70)
    print("DIAGNOSTIC RESULTS — OFFICIAL ENV")
    print("=" * 70)
    print()
    print(f"  {'Label':<30} {'Seed':>6} {'Cosine':>10} {'Time':>8}")
    print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*8}")
    for r in results:
        print(f"  {r['label']:<30} {r['seed']:>6} {r['cosine']:>10} {r['time_s']:>8}s")

    cosines = [float(r['cosine']) for r in results]
    mean_cos = np.mean(cosines)
    max_cos = np.max(cosines)
    min_cos = np.min(cosines)

    print()
    print(f"  Mean cosine: {mean_cos:.4f}")
    print(f"  Max cosine:  {max_cos:.4f}")
    print(f"  Min cosine:  {min_cos:.4f}")
    print()

    # Compare with previous broken results
    print("  COMPARISON WITH BROKEN ENV (transformers 5.8.0):")
    print(f"  Previous fp32_seed42 cosine: 0.0211  (BROKEN)")
    broken_seed42 = [r for r in results if r['seed'] == 42]
    if broken_seed42:
        new_cos = float(broken_seed42[0]['cosine'])
        print(f"  New official_fp32_seed42:    {new_cos:.4f}")
        if new_cos > 0.3:
            print(f"  → IMPROVEMENT: +{new_cos - 0.0211:.4f}")
        elif new_cos > 0.05:
            print(f"  → PARTIAL IMPROVEMENT: +{new_cos - 0.0211:.4f}")
        else:
            print(f"  → NO IMPROVEMENT: still near zero")

    print()
    if mean_cos > 0.3:
        print("★★★ SUCCESS: Official env produces identity-preserving B1!")
        print("→ Root cause confirmed: transformers 5.x broke the encoder forward pass.")
        print("→ Next: re-run 10-identity reconstruction with official env.")
    elif mean_cos > 0.05:
        print("⚠ PARTIAL: Some improvement but below expected threshold.")
        print("→ May need GPU or high-res source images.")
    else:
        print("✗ FAILURE: Official env also produces near-zero cosine.")
        print("→ Issue is not transformers version. Check source image domain.")

    print()
    print(f"CSV saved: {csv_path}")
    print(f"Images saved: outputs/diagnostic/")
    print(f"Done at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
