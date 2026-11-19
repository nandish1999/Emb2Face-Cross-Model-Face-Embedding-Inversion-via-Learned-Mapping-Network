import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import onnxruntime as ort


# ============================================================
# Image loading
# ============================================================

def load_image_bgr(path: str) -> np.ndarray:
    """Load image as BGR uint8 numpy array using OpenCV.
    
    Returns:
        np.ndarray: shape (112, 112, 3), dtype uint8, BGR channel order
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if img.shape != (112, 112, 3):
        raise ValueError(f"Unexpected shape {img.shape} for {path}, expected (112, 112, 3)")
    return img


def load_image_rgb(path: str) -> np.ndarray:
    """Load image as RGB uint8 numpy array using PIL.
    
    Returns:
        np.ndarray: shape (112, 112, 3), dtype uint8, RGB channel order
    """
    pil_img = Image.open(path).convert('RGB')
    np_img = np.array(pil_img)
    if np_img.shape != (112, 112, 3):
        raise ValueError(f"Unexpected shape {np_img.shape} for {path}, expected (112, 112, 3)")
    return np_img


# ============================================================
# AdaFace
# ============================================================

def load_adaface_model(
    architecture: str = 'ir_50',
    checkpoint_path: str = 'models/adaface/adaface_ir50_webface4m.ckpt',
    repo_path: str = 'models/adaface_repo',
):
    """Load AdaFace model from checkpoint.
    
    Args:
        architecture: Model architecture (default: 'ir_50')
        checkpoint_path: Path to .ckpt file
        repo_path: Path to AdaFace repository (for net.py import)
    
    Returns:
        torch.nn.Module: AdaFace model in eval mode
    """
    # Add repo to path for net.py import
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    import net

    model = net.build_model(architecture)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_statedict = {
        key[6:]: val
        for key, val in ckpt['state_dict'].items()
        if key.startswith('model.')
    }
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def preprocess_adaface(rgb_image: np.ndarray) -> torch.Tensor:
    """Convert RGB uint8 image to AdaFace input tensor.
    
    AdaFace expects:
      1. BGR channel order (flip from RGB)
      2. Normalize: (pixel / 255 - 0.5) / 0.5  →  range [-1, 1]
      3. Transpose: HWC → CHW
      4. Batch dimension added
    
    Args:
        rgb_image: np.ndarray shape (112, 112, 3), dtype uint8, RGB
    
    Returns:
        torch.Tensor: shape (1, 3, 112, 112), dtype float32
    """
    bgr = rgb_image[:, :, ::-1].copy()  # RGB → BGR
    normalized = ((bgr / 255.0) - 0.5) / 0.5  # → [-1, 1]
    chw = normalized.transpose(2, 0, 1)  # HWC → CHW
    tensor = torch.from_numpy(np.ascontiguousarray(chw[np.newaxis])).float()
    return tensor


def get_adaface_embedding(model, rgb_image: np.ndarray) -> np.ndarray:
    """Extract L2-normalized 512-dim AdaFace embedding.
    
    Args:
        model: Loaded AdaFace model
        rgb_image: np.ndarray shape (112, 112, 3), dtype uint8, RGB
    
    Returns:
        np.ndarray: shape (512,), L2-normalized embedding
    """
    tensor = preprocess_adaface(rgb_image)
    with torch.no_grad():
        feature, norm = model(tensor)
        embedding = F.normalize(feature, dim=1)
    return embedding.squeeze().numpy()


def get_adaface_embedding_with_norm(model, rgb_image: np.ndarray):
    """Extract AdaFace embedding and quality norm.
    
    Args:
        model: Loaded AdaFace model
        rgb_image: np.ndarray shape (112, 112, 3), dtype uint8, RGB
    
    Returns:
        tuple: (embedding: np.ndarray shape (512,), quality_norm: float)
    """
    tensor = preprocess_adaface(rgb_image)
    with torch.no_grad():
        feature, norm = model(tensor)
        embedding = F.normalize(feature, dim=1)
    return embedding.squeeze().numpy(), norm.item()


# ============================================================
# ArcFace (direct ONNX, no face detection)
# ============================================================

def load_arcface_session(
    model_path: str = 'models/antelopev2_arc2face/arcface.onnx',
    providers: list = None,
):
    """Load ArcFace ONNX session for direct inference.
    
    Args:
        model_path: Path to arcface.onnx
        providers: ONNX execution providers (default: CPU only)
    
    Returns:
        ort.InferenceSession
    """
    if providers is None:
        providers = ['CPUExecutionProvider']
    sess = ort.InferenceSession(model_path, providers=providers)
    return sess


def preprocess_arcface(bgr_image: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 image to ArcFace ONNX input blob.
    
    ArcFace (InsightFace) expects:
      1. BGR channel order (as from cv2.imread)
      2. Normalize: (pixel - 127.5) / 127.5  →  range [-1, 1]
      3. Transpose: HWC → CHW
      4. Batch dimension added
    
    Args:
        bgr_image: np.ndarray shape (112, 112, 3), dtype uint8, BGR
    
    Returns:
        np.ndarray: shape (1, 3, 112, 112), dtype float32
    """
    img = bgr_image.astype(np.float32)
    img = (img - 127.5) / 127.5  # → [-1, 1]
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, 0)  # batch dim
    return img


def get_arcface_embedding_direct(session, bgr_image: np.ndarray) -> np.ndarray:
    """Extract L2-normalized 512-dim ArcFace embedding using direct ONNX inference.
    
    Bypasses InsightFace face detection (which fails on 112×112 images).
    
    Args:
        session: ONNX InferenceSession for arcface.onnx
        bgr_image: np.ndarray shape (112, 112, 3), dtype uint8, BGR
    
    Returns:
        np.ndarray: shape (512,), L2-normalized embedding
    """
    blob = preprocess_arcface(bgr_image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    embedding = session.run([output_name], {input_name: blob})[0][0]
    # L2 normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


# ============================================================
# Evaluation Judge ArcFace (glintr100 — independent model)
# ============================================================

def load_eval_arcface_session(
    model_path: str = 'models/antelopev2_eval/glintr100.onnx',
    providers: list = None,
):
    """Load the evaluation judge ArcFace (GlintR100) ONNX session.
    
    This model is DIFFERENT from Arc2Face's arcface.onnx.
    It must be used ONLY for evaluation, never for training data.
    
    Args:
        model_path: Path to glintr100.onnx
        providers: ONNX execution providers
    
    Returns:
        ort.InferenceSession
    """
    if providers is None:
        providers = ['CPUExecutionProvider']
    return ort.InferenceSession(model_path, providers=providers)


def get_eval_embedding_direct(session, bgr_image: np.ndarray) -> np.ndarray:
    """Extract embedding from evaluation judge model.
    
    Same preprocessing as ArcFace. Uses glintr100 model.
    
    Args:
        session: ONNX InferenceSession for glintr100.onnx
        bgr_image: np.ndarray shape (112, 112, 3), dtype uint8, BGR
    
    Returns:
        np.ndarray: shape (512,), L2-normalized embedding
    """
    return get_arcface_embedding_direct(session, bgr_image)


# ============================================================
# Convenience: unified loading
# ============================================================

def load_image_for_both(path: str):
    """Load image and return both formats needed by the models.
    
    Args:
        path: Path to 112×112 aligned face image
    
    Returns:
        tuple: (rgb_image, bgr_image) both as np.ndarray uint8
    """
    rgb = load_image_rgb(path)
    bgr = rgb[:, :, ::-1].copy()  # RGB → BGR
    return rgb, bgr


# ============================================================
# Smoke test
# ============================================================

if __name__ == '__main__':
    import glob

    print("=" * 60)
    print("Preprocessing Utils — Smoke Test")
    print("=" * 60)

    # Find a test image
    test_images = glob.glob('data/eval_ppt_500/**/*.jpg', recursive=True)[:3]
    if not test_images:
        print("No test images found in data/eval_ppt_500/")
        exit(1)

    # Load models
    print("\n1. Loading AdaFace...")
    adaface_model = load_adaface_model()
    print("   OK")

    print("\n2. Loading ArcFace (Arc2Face)...")
    arcface_session = load_arcface_session()
    print("   OK")

    print("\n3. Loading ArcFace (Eval Judge)...")
    eval_session = load_eval_arcface_session()
    print("   OK")

    # Test on sample images
    print("\n4. Testing embeddings on sample images...")
    for img_path in test_images:
        rgb, bgr = load_image_for_both(img_path)

        ada_emb = get_adaface_embedding(adaface_model, rgb)
        arc_emb = get_arcface_embedding_direct(arcface_session, bgr)
        eval_emb = get_eval_embedding_direct(eval_session, bgr)

        print(f"\n   {os.path.basename(img_path)}:")
        print(f"     AdaFace:  shape={ada_emb.shape}, norm={np.linalg.norm(ada_emb):.4f}")
        print(f"     ArcFace:  shape={arc_emb.shape}, norm={np.linalg.norm(arc_emb):.4f}")
        print(f"     EvalJudge: shape={eval_emb.shape}, norm={np.linalg.norm(eval_emb):.4f}")

        # Cross-similarity (should be different since different models)
        ada_arc_sim = np.dot(ada_emb, arc_emb)
        arc_eval_sim = np.dot(arc_emb, eval_emb)
        print(f"     AdaFace↔ArcFace cosine: {ada_arc_sim:.4f}")
        print(f"     ArcFace↔EvalJudge cosine: {arc_eval_sim:.4f}")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
