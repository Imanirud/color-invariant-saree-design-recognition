"""
Training and Validation Pipeline for Color-Invariant Saree Retrieval.
Phase 5 & Phase 6: End-to-end training loop with mixed precision,
differential learning rate, and validation checkpointing.
"""

import os
import time
import json
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models import ProposedColorInvariantModel
from src.losses import SupConLoss, BatchHardTripletLoss
from src.dataset import SareeDataset, BalancedBatchSampler, build_design_level_splits
from src.transforms import get_color_invariant_transforms
from src.metrics import compute_retrieval_metrics, compute_verification_metrics

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str = "cuda"
) -> float:
    """Execute one training epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for step, (images, labels, _) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=torch.cuda.is_available()):
            embeddings = model(images)
            loss = criterion(embeddings, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(num_batches, 1)

@torch.no_grad()
def evaluate_retrieval_and_verification(
    model: nn.Module,
    val_dataset: SareeDataset,
    device: str = "cuda",
    batch_size: int = 32
) -> Dict[str, Any]:
    """Extract embeddings on validation set and evaluate ranking and verification."""
    model.eval()
    dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_embeddings = []
    all_labels = []

    for images, labels, _ in dataloader:
        images = images.to(device)
        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=torch.cuda.is_available()):
            embs = model(images)
        all_embeddings.append(embs.cpu())
        all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Self-retrieval / Leave-one-out ranking on validation designs
    retrieval_results = compute_retrieval_metrics(
        query_embeddings=all_embeddings,
        gallery_embeddings=all_embeddings,
        query_labels=all_labels,
        gallery_labels=all_labels,
        top_k=[1, 3, 5]
    )

    # Verification simulation on val pairs
    n = len(all_labels)
    num_pairs = min(500, n * (n - 1) // 2)
    import random
    rng = random.Random(42)
    
    idx_a, idx_b, is_same = [], [], []
    for _ in range(num_pairs // 2):
        # Sample positive pair
        same_cls = rng.choice(list(val_dataset.design_to_indices.keys()))
        indices = val_dataset.design_to_indices[same_cls]
        if len(indices) >= 2:
            i, j = rng.sample(indices, 2)
            idx_a.append(i)
            idx_b.append(j)
            is_same.append(1)

    for _ in range(num_pairs - len(is_same)):
        # Sample negative pair
        i, j = rng.sample(range(n), 2)
        idx_a.append(i)
        idx_b.append(j)
        is_same.append(1 if all_labels[i] == all_labels[j] else 0)

    import numpy as np
    verif_results = compute_verification_metrics(
        embeddings_a=all_embeddings[idx_a],
        embeddings_b=all_embeddings[idx_b],
        is_same_design=np.array(is_same)
    )

    combined = {}
    combined.update(retrieval_results)
    combined.update({f"Verif_{k}": v for k, v in verif_results.items()})
    return combined

def run_training(
    records: List[Dict[str, Any]],
    output_dir: str = "./output",
    epochs: int = 15,
    p_classes: int = 8,
    k_instances: int = 4,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """Execute complete training and validation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "best_saree_model.pth")

    # Split dataset
    train_recs, val_recs, gallery_recs, query_recs = build_design_level_splits(records)
    
    transforms = get_color_invariant_transforms()
    train_dataset = SareeDataset(train_recs, transform=transforms["train"])
    val_dataset = SareeDataset(val_recs, transform=transforms["eval"])

    # Balanced Sampler
    sampler = BalancedBatchSampler(train_dataset, p_classes=p_classes, k_instances=k_instances, iterations_per_epoch=100)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=2, pin_memory=True)

    # Model & Metric Loss
    model = ProposedColorInvariantModel(backbone_name="resnet50", pretrained=True, embed_dim=256).to(device)
    criterion = SupConLoss(temperature=0.07).to(device)

    # Differential Learning Rate: Backbone fine-tunes slower than newly initialized projection head
    optimizer = AdamW([
        {"params": model.trunk.parameters(), "lr": lr * 0.1},
        {"params": model.pool.parameters(), "lr": lr},
        {"params": model.head.parameters(), "lr": lr}
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_r1 = -1.0
    history = []

    print(f"[Trainer] Starting training for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate_retrieval_and_verification(model, val_dataset, device)
        scheduler.step()

        elapsed = time.time() - t0
        r1 = val_metrics.get("Recall@1", 0.0)
        auc = val_metrics.get("Verif_ROC_AUC", 0.0)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - Loss: {train_loss:.4f} | Val R@1: {r1*100:.1f}% | Val AUC: {auc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_metrics": val_metrics
        })

        if r1 > best_r1:
            best_r1 = r1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_recall1": r1,
                "val_auc": auc
            }, checkpoint_path)
            print(f"  -> Saved new best checkpoint to: {checkpoint_path}")

    return {"best_recall1": best_r1, "history": history, "checkpoint_path": checkpoint_path}
