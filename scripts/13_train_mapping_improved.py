import os
import sys
import time
import math
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

class ImprovedMappingNetwork(nn.Module):
    """N-Layer MLP with residual connection for AdaFace -> ArcFace mapping.

    Output is L2-normalized to unit sphere.
    Residual: out = MLP(x) + residual_alpha * x, then L2-normalize.
    """
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512,
                 dropout=0.1, residual_alpha=0.1, num_hidden_layers=3):
        super().__init__()
        self.residual_alpha = residual_alpha

        layers = []
        # First hidden layer: input_dim -> hidden_dim
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        # Additional hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        # Output layer: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        if self.residual_alpha > 0:
            out = out + self.residual_alpha * x
        out = F.normalize(out, p=2, dim=1)
        return out


# ============================================================
# Loss Function
# ============================================================

def mapping_loss(pred, target, alpha=0.7):
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
def evaluate(model, loader, alpha=0.7):
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
        loss, cos_l, mse_l = mapping_loss(pred, Y_batch, alpha=alpha)

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
        'cos_median': float(np.median(all_cos)),
    }


# ============================================================
# Argument Parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train improved mapping network')
    parser.add_argument('--data', type=str, default='embeddings/train/train_full.npz',
                        help='Path to .npz with adaface/arcface arrays')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Max training epochs (default: 500)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for AdamW (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation fraction (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Loss weight: alpha*cosine + (1-alpha)*MSE (default: 0.7)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='LR warmup epochs (default: 10)')
    parser.add_argument('--residual-alpha', type=float, default=0.1,
                        help='Residual skip connection weight (default: 0.1)')
    parser.add_argument('--hidden-layers', type=int, default=3,
                        help='Number of hidden layers (default: 3)')
    parser.add_argument('--hidden-dim', type=int, default=1024,
                        help='Hidden layer dimension (default: 1024)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--checkpoint', type=str,
                        default='models/mapping_network/checkpoints/mapping_improved_v1.pt',
                        help='Checkpoint save path')
    parser.add_argument('--metrics', type=str,
                        default='outputs/evaluation/mapping_improved_v1_metrics.csv',
                        help='Metrics CSV save path')
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    print("=" * 70)
    print("  IMPROVED MAPPING NETWORK TRAINING")
    print("  3-layer MLP + Residual + Cosine-weighted loss")
    print("=" * 70)

    # Safety check: do not overwrite official checkpoint
    official = 'models/mapping_network/checkpoints/mapping_full_official_safe.pt'
    if os.path.abspath(args.checkpoint) == os.path.abspath(official):
        print("ERROR: Cannot overwrite official checkpoint!")
        print("  Use a different --checkpoint path.")
        sys.exit(1)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ============================================================
    # 1. Load data
    # ============================================================
    print("\n1. Loading data from " + args.data + "...")
    data = np.load(args.data)
    X_all = torch.tensor(data['adaface'], dtype=torch.float32)
    Y_all = torch.tensor(data['arcface'], dtype=torch.float32)
    ids_all = data['ids']

    print("   Total pairs: " + str(len(X_all)))
    print("   AdaFace shape: " + str(tuple(X_all.shape)))
    print("   ArcFace shape: " + str(tuple(Y_all.shape)))

    # ============================================================
    # 2. Split by identity (same method as 07_ script)
    # ============================================================
    split_pct = args.val_split
    print("\n2. Splitting by identity (" + str(int((1-split_pct)*100)) + "% train / " + str(int(split_pct*100)) + "% val)...")

    unique_ids = sorted(set(ids_all))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(unique_ids)
    n_val_ids = max(1, int(len(unique_ids) * split_pct))
    val_ids = set(unique_ids[:n_val_ids])
    train_ids = set(unique_ids[n_val_ids:])

    train_mask = np.array([id_ not in val_ids for id_ in ids_all])
    val_mask = ~train_mask

    X_train, Y_train = X_all[train_mask], Y_all[train_mask]
    X_val, Y_val = X_all[val_mask], Y_all[val_mask]

    print("   Train: " + str(len(X_train)) + " pairs (" + str(len(train_ids)) + " identities)")
    print("   Val:   " + str(len(X_val)) + " pairs (" + str(len(val_ids)) + " identities)")

    # ============================================================
    # 3. Create DataLoaders
    # ============================================================
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ============================================================
    # 4. Create model
    # ============================================================
    print("\n3. Creating improved model...")
    model = ImprovedMappingNetwork(
        input_dim=512, hidden_dim=args.hidden_dim, output_dim=512,
        dropout=args.dropout, residual_alpha=args.residual_alpha,
        num_hidden_layers=args.hidden_layers,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print("   Architecture: " + str(args.hidden_layers) + "-layer MLP + residual(alpha=" + str(args.residual_alpha) + ")")
    print("   Parameters: " + str(n_params))
    print("   Loss alpha: " + str(args.alpha) + " (cosine) + " + str(1-args.alpha) + " (MSE)")
    print("   Optimizer: AdamW, lr=" + str(args.lr) + ", weight_decay=" + str(args.weight_decay))
    print("   Warmup: " + str(args.warmup) + " epochs")
    print("   Scheduler: CosineAnnealingLR (after warmup)")
    print("   Max epochs: " + str(args.epochs) + ", patience: " + str(args.patience))

    # ============================================================
    # 5. Optimizer and scheduler
    # ============================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine annealing for post-warmup epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup, eta_min=1e-6
    )

    # ============================================================
    # 6. Setup metrics CSV
    # ============================================================
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    csv_file = open(args.metrics, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'epoch', 'train_loss', 'train_cos_loss', 'train_mse_loss',
        'val_loss', 'val_cos_loss', 'val_mse_loss',
        'val_cos_mean', 'val_cos_std', 'val_cos_min', 'val_cos_max', 'val_cos_median',
        'lr', 'is_best', 'epoch_time_s'
    ])

    # ============================================================
    # 7. Training loop
    # ============================================================
    print("\n4. Starting training...\n")
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    best_val_cos = -1.0
    best_epoch = 0
    patience_counter = 0
    baseline_cos = 0.8120  # Current baseline for comparison

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # --- Learning rate: warmup or scheduler ---
        if epoch <= args.warmup:
            warmup_lr = args.lr * epoch / args.warmup
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # --- Train ---
        model.train()
        train_loss_sum = 0
        train_cos_sum = 0
        train_mse_sum = 0
        n_train = 0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss, cos_l, mse_l = mapping_loss(pred, Y_batch, alpha=args.alpha)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_cos_sum += cos_l.item()
            train_mse_sum += mse_l.item()
            n_train += 1

        train_loss = train_loss_sum / n_train
        train_cos = train_cos_sum / n_train
        train_mse = train_mse_sum / n_train

        # --- Validate ---
        val_metrics = evaluate(model, val_loader, alpha=args.alpha)
        epoch_time = time.time() - epoch_start

        # --- Check best ---
        is_best = val_metrics['cos_mean'] > best_val_cos
        if is_best:
            best_val_cos = val_metrics['cos_mean']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cos_mean': best_val_cos,
                'val_cos_std': val_metrics['cos_std'],
                'val_cos_min': val_metrics['cos_min'],
                'val_cos_max': val_metrics['cos_max'],
                'args': vars(args),
            }, args.checkpoint)
        else:
            patience_counter += 1

        # --- Log ---
        csv_writer.writerow([
            epoch, train_loss, train_cos, train_mse,
            val_metrics['loss'], val_metrics['cos_loss'], val_metrics['mse_loss'],
            val_metrics['cos_mean'], val_metrics['cos_std'],
            val_metrics['cos_min'], val_metrics['cos_max'], val_metrics['cos_median'],
            current_lr, 'Y' if is_best else 'N', round(epoch_time, 1)
        ])
        csv_file.flush()

        # --- Print progress ---
        marker = " ***BEST***" if is_best else ""
        vs_baseline = val_metrics['cos_mean'] - baseline_cos
        sign = "+" if vs_baseline >= 0 else ""
        print(
            "Epoch " + str(epoch).rjust(3) + "/" + str(args.epochs) +
            " | lr=" + ("%.6f" % current_lr) +
            " | train_loss=" + ("%.4f" % train_loss) +
            " | val_cos=" + ("%.4f" % val_metrics['cos_mean']) +
            " (std=" + ("%.4f" % val_metrics['cos_std']) +
            " min=" + ("%.4f" % val_metrics['cos_min']) +
            " max=" + ("%.4f" % val_metrics['cos_max']) + ")" +
            " | vs_baseline=" + sign + ("%.4f" % vs_baseline) +
            " | " + ("%.0f" % epoch_time) + "s" + marker
        )

        # --- Early stopping ---
        if patience_counter >= args.patience:
            print("\nEarly stopping at epoch " + str(epoch) + " (no improvement for " + str(args.patience) + " epochs)")
            break

    csv_file.close()

    # ============================================================
    # 8. Final summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print("  Best epoch:        " + str(best_epoch))
    print("  Best val cosine:   " + ("%.4f" % best_val_cos))
    print("  Baseline cosine:   " + ("%.4f" % baseline_cos))
    diff = best_val_cos - baseline_cos
    sign = "+" if diff >= 0 else ""
    print("  Improvement:       " + sign + ("%.4f" % diff))
    print("")
    if best_val_cos > baseline_cos:
        print("  RESULT: IMPROVED over baseline!")
        print("  Next step: Run 25-image B3 reconstruction test.")
    elif best_val_cos > baseline_cos - 0.005:
        print("  RESULT: Similar to baseline (within noise).")
        print("  Consider checking per-identity metrics.")
    else:
        print("  RESULT: No improvement over baseline.")
        print("  Keep using mapping_full_official_safe.pt.")
    print("")
    print("  Checkpoint: " + args.checkpoint)
    print("  Metrics:    " + args.metrics)
    print("=" * 70)


if __name__ == "__main__":
    main()
