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


def load_arc2face_pipeline(dtype):
    """Load Arc2Face pipeline with specified dtype."""
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler

    arc2face_path = os.path.join(PROJECT_ROOT, 'models', 'arc2face_repo')
    sys.path.insert(0, arc2face_path)
    from arc2face import CLIPTextModelWrapper, project_face_embs

    weights_dir = os.path.join(arc2face_path, 'arc2face_weights')
    base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'

    print(f"   Loading encoder (dtype={dtype})...")
    encoder = CLIPTextModelWrapper.from_pretrained(
        weights_dir, subfolder="encoder", torch_dtype=dtype
    )

    print(f"   Loading UNet (dtype={dtype})...")
    unet = UNet2DConditionModel.from_pretrained(
        weights_dir, subfolder="arc2face", torch_dtype=dtype
    )

    print("   Loading SD v1.5 base pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=dtype,
        safety_checker=None
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    return pipeline, project_face_embs


def reconstruct_b1(pipeline, project_fn, embedding_512, seed, dtype, num_steps=25, guidance_scale=3.0):
    """Generate a face image from ArcFace embedding with fixed seed and dtype."""
    if isinstance(embedding_512, np.ndarray):
        id_emb = torch.tensor(embedding_512, dtype=dtype).unsqueeze(0)
    else:
        id_emb = embedding_512.unsqueeze(0).to(dtype)

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
    # Original embedding
    orig_emb = pu.get_arcface_embedding_direct(arcface_session, orig_bgr_112)

    # Generated: convert PIL RGB → BGR numpy, resize 512→112
    gen_np = np.array(gen_pil_image)[:, :, ::-1]  # RGB → BGR
    gen_112 = cv2.resize(gen_np, (112, 112), interpolation=cv2.INTER_AREA)
    gen_emb = pu.get_arcface_embedding_direct(arcface_session, gen_112)

    return float(np.dot(orig_emb, gen_emb))


def main():
    print("=" * 70)
    print("Phase 1: Arc2Face B1 Dtype + Seed Diagnostic")
    print("=" * 70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ============================================================
    # 1. Load test image
    # ============================================================
    test_image_path = 'data/eval_ppt_500/id_10092/10092_477517.jpg'
    print(f"1. Loading test image: {test_image_path}")
    orig_bgr = cv2.imread(test_image_path)
    print(f"   Shape: {orig_bgr.shape}")
    print()

    # ============================================================
    # 2. Load ArcFace recognizer
    # ============================================================
    print("2. Loading ArcFace (arcface.onnx) for embedding extraction...")
    arcface_session = pu.load_arcface_session()
    print("   Loaded.")
    print()

    # ============================================================
    # 3. Extract ground truth ArcFace embedding
    # ============================================================
    print("3. Extracting original ArcFace embedding...")
    orig_emb = pu.get_arcface_embedding_direct(arcface_session, orig_bgr)
    print(f"   Shape: {orig_emb.shape}, norm: {np.linalg.norm(orig_emb):.4f}")
    print()

    # ============================================================
    # 4. Define test configurations
    # ============================================================
    os.makedirs('outputs/diagnostic', exist_ok=True)
    os.makedirs('outputs/evaluation', exist_ok=True)

    configs = [
        # (label, dtype, seed)
        ('fp32_seed42', torch.float32, 42),
        ('fp16_seed23', torch.float16, 23),
        ('fp16_seed42', torch.float16, 42),
        ('fp16_seed100', torch.float16, 100),
        ('fp16_seed200', torch.float16, 200),
    ]

    # ============================================================
    # 5. Run diagnostics
    # ============================================================
    results = []

    # Group by dtype to avoid reloading pipeline unnecessarily
    for dtype_label, dtype_val in [('fp32', torch.float32), ('fp16', torch.float16)]:
        dtype_configs = [(l, d, s) for l, d, s in configs if d == dtype_val]
        if not dtype_configs:
            continue

        print(f"{'='*70}")
        print(f"Loading Arc2Face pipeline ({dtype_label})...")
        print(f"{'='*70}")
        t0 = time.time()
        try:
            pipeline, project_fn = load_arc2face_pipeline(dtype_val)
            load_time = time.time() - t0
            print(f"   Pipeline loaded in {load_time:.1f}s")
        except Exception as e:
            print(f"   FAILED to load {dtype_label} pipeline: {e}")
            for label, _, seed in dtype_configs:
                results.append({
                    'label': label, 'dtype': dtype_label, 'seed': seed,
                    'cosine': 'LOAD_FAILED', 'time_s': 0,
                    'image_path': 'N/A'
                })
            continue

        for label, dtype, seed in dtype_configs:
            print(f"\n   [{label}] Generating B1 with seed={seed}...")
            t1 = time.time()
            try:
                gen_img = reconstruct_b1(pipeline, project_fn, orig_emb, seed, dtype)
                gen_time = time.time() - t1

                # Save image
                img_path = f'outputs/diagnostic/b1_{label}.png'
                gen_img.save(img_path)

                # Compute cosine
                cosine = compute_arcface_cosine(arcface_session, orig_bgr, gen_img)

                print(f"   [{label}] cosine={cosine:.4f}  time={gen_time:.1f}s  saved={img_path}")

                results.append({
                    'label': label, 'dtype': dtype_label, 'seed': seed,
                    'cosine': f'{cosine:.4f}', 'time_s': f'{gen_time:.1f}',
                    'image_path': img_path
                })
            except Exception as e:
                gen_time = time.time() - t1
                print(f"   [{label}] FAILED: {e}")
                results.append({
                    'label': label, 'dtype': dtype_label, 'seed': seed,
                    'cosine': f'ERROR: {e}', 'time_s': f'{gen_time:.1f}',
                    'image_path': 'N/A'
                })

        # Free memory
        del pipeline
        import gc
        gc.collect()

    # ============================================================
    # 6. Save CSV
    # ============================================================
    csv_path = 'outputs/evaluation/arc2face_b1_dtype_seed_diagnostic.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['label', 'dtype', 'seed', 'cosine', 'time_s', 'image_path'])
        writer.writeheader()
        writer.writerows(results)

    # ============================================================
    # 7. Summary
    # ============================================================
    print()
    print("=" * 70)
    print("DIAGNOSTIC RESULTS")
    print("=" * 70)
    print()
    print(f"  {'Label':<18} {'Dtype':>6} {'Seed':>6} {'Cosine':>10} {'Time':>8}")
    print(f"  {'-'*18} {'-'*6} {'-'*6} {'-'*10} {'-'*8}")
    for r in results:
        print(f"  {r['label']:<18} {r['dtype']:>6} {r['seed']:>6} {str(r['cosine']):>10} {str(r['time_s']):>8}s")

    print()
    print(f"CSV saved: {csv_path}")
    print(f"Images saved: outputs/diagnostic/")
    print()

    # Decision logic
    fp16_cosines = [float(r['cosine']) for r in results
                    if r['dtype'] == 'fp16' and 'ERROR' not in str(r['cosine']) and 'LOAD' not in str(r['cosine'])]
    fp32_cosines = [float(r['cosine']) for r in results
                    if r['dtype'] == 'fp32' and 'ERROR' not in str(r['cosine']) and 'LOAD' not in str(r['cosine'])]

    if fp16_cosines:
        fp16_mean = np.mean(fp16_cosines)
        fp32_mean = np.mean(fp32_cosines) if fp32_cosines else 0

        print(f"Float32 mean cosine: {fp32_mean:.4f}")
        print(f"Float16 mean cosine: {fp16_mean:.4f}")
        print()

        if fp16_mean > 0.3:
            print("★★★ RESULT: Float16 FIXES the issue! B1 cosine > 0.3")
            print("→ Proceed to Phase 2: fix pipeline permanently.")
        elif fp16_mean > 0.05:
            print("⚠ RESULT: Float16 partially improves. Need further investigation.")
            print("→ Consider Phase 3 (transformers 4.x test).")
        else:
            print("✗ RESULT: Float16 does NOT fix the issue.")
            print("→ Proceed to Phase 3 (transformers 4.x test).")
    else:
        print("Float16 pipeline failed to load. CPU may not support float16 inference.")
        print("→ Proceed to Phase 3 (transformers 4.x test).")

    print()
    print(f"Done at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
