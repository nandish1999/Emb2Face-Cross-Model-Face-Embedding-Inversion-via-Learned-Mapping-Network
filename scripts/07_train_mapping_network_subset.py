import os
import sys
import time
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)


# ============================================================
# Model Architecture
# ============================================================

class MappingNetwork(nn.Module):
    """2-Layer MLP that maps AdaFace → ArcFace embedding space.
    
    Output is L2-normalized to unit sphere.
    """
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
        out = F.normalize(out, p=2, dim=1)  # L2 normalize to unit sphere
        return out


# ============================================================
# Loss Function
# ============================================================

def mapping_loss(pred, target, alpha=0.5):
    """Combined cosine + MSE loss.
    
    Args:
        pred: predicted ArcFace embeddings (N, 512), L2-normalized
        target: true ArcFace embeddings (N, 512), L2-normalized
        alpha: weight for cosine loss (1-alpha for MSE)
    
    Returns:
        total_loss, cos_loss, mse_loss
    """
    cos_loss = (1 - F.cosine_similarity(pred, target, dim=1)).mean()
    mse_loss = F.mse_loss(pred, target)
    total = alpha * cos_loss + (1 - alpha) * mse_loss
    return total, cos_loss, mse_loss


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader):
    """Evaluate model on a DataLoader.
    
    Returns dict with loss, cosine similarity stats.
    """
    model.eval()
    all_cos = []
    total_loss = 0
    total_cos_loss = 0
    total_mse_loss = 0
    n_batches = 0
    
    for X_batch, Y_batch in loader:
        pred = model(X_batch)
        loss, cos_l, mse_l = mapping_loss(pred, Y_batch)
        
        total_loss += loss.item()
        total_cos_loss += cos_l.item()
        total_mse_loss += mse_l.item()
        n_batches += 1
        
        cos_sim = F.cosine_similarity(pred, Y_batch, dim=1)
        all_cos.append(cos_sim.numpy())
    
    all_cos = np.concatenate(all_cos)
    
    return {
        'loss': total_loss / n_batches,
        'cos_loss': total_cos_loss / n_batches,
        'mse_loss': total_mse_loss / n_batches,
        'cos_mean': all_cos.mean(),
        'cos_std': all_cos.std(),
        'cos_min': all_cos.min(),
        'cos_max': all_cos.max(),
        'cos_median': np.median(all_cos),
    }


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train mapping network')
    parser.add_argument('--data', type=str, default='embeddings/train/train_subset_1000.npz',
                        help='Path to .npz with adaface/arcface arrays')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Max training epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation fraction (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Loss weight: alpha*cosine + (1-alpha)*MSE (default: 0.5)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Step 7A: Train Prototype Mapping Network")
    print("=" * 60)
    
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # ============================================================
    # 1. Load data
    # ============================================================
    print(f"\n1. Loading data from {args.data}...")
    data = np.load(args.data)
    X_all = torch.tensor(data['adaface'], dtype=torch.float32)
    Y_all = torch.tensor(data['arcface'], dtype=torch.float32)
    ids_all = data['ids']
    
    print(f"   Total pairs: {len(X_all)}")
    print(f"   AdaFace shape: {X_all.shape}")
    print(f"   ArcFace shape: {Y_all.shape}")
    
    # ============================================================
    # 2. Split by identity (not by random sample)
    # ============================================================
    print(f"\n2. Splitting by identity ({1-args.val_split:.0%} train / {args.val_split:.0%} val)...")
    
    unique_ids = sorted(set(ids_all))
    np.random.shuffle(unique_ids)
    n_val_ids = max(1, int(len(unique_ids) * args.val_split))
    val_ids = set(unique_ids[:n_val_ids])
    train_ids = set(unique_ids[n_val_ids:])
    
    train_mask = np.array([id_ not in val_ids for id_ in ids_all])
    val_mask = ~train_mask
    
    X_train, Y_train = X_all[train_mask], Y_all[train_mask]
    X_val, Y_val = X_all[val_mask], Y_all[val_mask]
    
    print(f"   Train: {len(X_train)} pairs ({len(train_ids)} identities)")
    print(f"   Val:   {len(X_val)} pairs ({len(val_ids)} identities)")
    
    # ============================================================
    # 3. Create DataLoaders
    # ============================================================
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # ============================================================
    # 4. Initialize model
    # ============================================================
    print("\n3. Initializing model...")
    model = MappingNetwork(input_dim=512, hidden_dim=1024, output_dim=512, dropout=0.1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Architecture: 512 → 1024 → 1024 → 512 (L2-normalized)")
    print(f"   Parameters: {n_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # ============================================================
    # 5. Pre-training baseline
    # ============================================================
    print("\n4. Pre-training baseline (random weights)...")
    baseline = evaluate(model, val_loader)
    print(f"   Val cosine similarity: {baseline['cos_mean']:.4f} ± {baseline['cos_std']:.4f}")
    print(f"   (Random baseline, should be near 0)")
    
    # ============================================================
    # 6. Training loop
    # ============================================================
    print(f"\n5. Training (max {args.epochs} epochs, patience={args.patience})...")
    
    checkpoint_dir = 'models/mapping_network/checkpoints'
    metrics_dir = 'outputs/evaluation'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Derive output tag from input filename (e.g. train_subset_10000 -> mapping_subset_10000)
    data_basename = os.path.splitext(os.path.basename(args.data))[0]  # e.g. 'train_subset_10000'
    output_tag = data_basename.replace('train_', 'mapping_')           # e.g. 'mapping_subset_10000'
    
    # Metrics CSV
    metrics_path = os.path.join(metrics_dir, f'{output_tag}_metrics.csv')
    metrics_file = open(metrics_path, 'w', newline='')
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow([
        'epoch', 'train_loss', 'train_cos_loss', 'train_mse_loss',
        'val_loss', 'val_cos_mean', 'val_cos_std', 'val_cos_min', 'val_cos_max',
        'lr', 'is_best'
    ])
    
    best_val_cos = -1
    patience_counter = 0
    best_epoch = 0
    t_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0
        epoch_cos_loss = 0
        epoch_mse_loss = 0
        n_batches = 0
        
        for X_batch, Y_batch in train_loader:
            pred = model(X_batch)
            loss, cos_l, mse_l = mapping_loss(pred, Y_batch, alpha=args.alpha)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_cos_loss += cos_l.item()
            epoch_mse_loss += mse_l.item()
            n_batches += 1
        
        train_loss = epoch_loss / n_batches
        train_cos = epoch_cos_loss / n_batches
        train_mse = epoch_mse_loss / n_batches
        
        # --- Validate ---
        val_metrics = evaluate(model, val_loader)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # --- Best model tracking ---
        is_best = val_metrics['cos_mean'] > best_val_cos
        if is_best:
            best_val_cos = val_metrics['cos_mean']
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = os.path.join(checkpoint_dir, f'{output_tag}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cos_mean': best_val_cos,
                'val_cos_std': val_metrics['cos_std'],
                'train_loss': train_loss,
                'args': vars(args),
            }, checkpoint_path)
        else:
            patience_counter += 1
        
        # --- Log ---
        metrics_writer.writerow([
            epoch, f'{train_loss:.6f}', f'{train_cos:.6f}', f'{train_mse:.6f}',
            f'{val_metrics["loss"]:.6f}', f'{val_metrics["cos_mean"]:.4f}',
            f'{val_metrics["cos_std"]:.4f}', f'{val_metrics["cos_min"]:.4f}',
            f'{val_metrics["cos_max"]:.4f}', f'{current_lr:.6f}',
            'Y' if is_best else 'N'
        ])
        metrics_file.flush()
        
        # --- Print progress ---
        if epoch <= 5 or epoch % 10 == 0 or is_best or patience_counter >= args.patience:
            marker = " ★ BEST" if is_best else ""
            print(f"   Epoch {epoch:>3d} | loss={train_loss:.4f} | "
                  f"val_cos={val_metrics['cos_mean']:.4f}±{val_metrics['cos_std']:.4f} | "
                  f"lr={current_lr:.6f}{marker}")
        
        # --- Early stopping ---
        if patience_counter >= args.patience:
            print(f"\n   Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break
    
    metrics_file.close()
    elapsed = time.time() - t_start
    
    # ============================================================
    # 7. Final evaluation with best model
    # ============================================================
    print(f"\n6. Final evaluation (best model from epoch {best_epoch})...")
    best_state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(best_state['model_state_dict'])
    
    final_train = evaluate(model, train_loader)
    final_val = evaluate(model, val_loader)
    
    print(f"   Train cosine: {final_train['cos_mean']:.4f} ± {final_train['cos_std']:.4f}")
    print(f"   Val cosine:   {final_val['cos_mean']:.4f} ± {final_val['cos_std']:.4f}")
    print(f"   Val cos range: [{final_val['cos_min']:.4f}, {final_val['cos_max']:.4f}]")
    
    # Quality assessment
    cos = final_val['cos_mean']
    if cos > 0.85:
        quality = "EXCELLENT — proceed to Arc2Face reconstruction"
    elif cos > 0.70:
        quality = "GOOD — may partially work with Arc2Face"
    elif cos > 0.50:
        quality = "MODERATE — mapping is learning, but needs more data"
    else:
        quality = "POOR — needs more data, deeper network, or different approach"
    
    # ============================================================
    # 8. Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 7A SUMMARY")
    print("=" * 60)
    print(f"Training data:     {len(X_train)} pairs ({len(train_ids)} identities)")
    print(f"Validation data:   {len(X_val)} pairs ({len(val_ids)} identities)")
    print(f"Parameters:        {n_params:,}")
    print(f"Best epoch:        {best_epoch}")
    print(f"Training time:     {elapsed:.1f}s ({elapsed/epoch:.2f}s/epoch)")
    print(f"")
    print(f"Random baseline:   cosine = {baseline['cos_mean']:.4f}")
    print(f"Best val cosine:   {best_val_cos:.4f} ± {final_val['cos_std']:.4f}")
    print(f"Improvement:       {best_val_cos - baseline['cos_mean']:+.4f}")
    print(f"")
    print(f"Quality:           {quality}")
    print(f"")
    print(f"Checkpoint:        {checkpoint_path}")
    print(f"Metrics CSV:       {metrics_path}")
    print(f"\nDone.")


if __name__ == '__main__':
    main()
