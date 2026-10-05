"""
Metric Learning Loss Functions for Saree Design Retrieval.
Phase 5 & Phase 10: Supervised Contrastive Loss (SupCon), Batch-Hard Triplet Loss,
and Hyperspherical ArcFace Loss.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Extends InfoNCE to handle multiple positive views/instances per anchor.
    Pulls all instances of the same design together in hyperspherical space
    while repelling all instances of differing designs.
    """
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
          features: L2-normalized embeddings of shape (batch_size, embed_dim)
          labels: ground truth design labels of shape (batch_size,)
        """
        device = features.device
        batch_size = features.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute cosine similarity matrix
        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)

        # For numerical stability, subtract max
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Mask-out self-contrast cases (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log-probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

        # Compute mean log-likelihood over positive pairs
        mask_pos_pairs = mask.sum(1)
        # Avoid division by zero for singletons
        mask_pos_pairs = torch.where(mask_pos_pairs > 0, mask_pos_pairs, torch.ones_like(mask_pos_pairs))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()

class BatchHardTripletLoss(nn.Module):
    """
    Batch-Hard Triplet Loss (Hermans et al., 2017).
    For each anchor in the batch, selects:
      - The hardest positive: p_hard = argmax_{p in P(a)} D(a, p)
      - The hardest negative: n_hard = argmin_{n in N(a)} D(a, n)
    Computes: L = mean( relu( D(a, p_hard) - D(a, n_hard) + margin ) )
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
          embeddings: L2-normalized vectors (B, D)
          labels: design targets (B,)
        """
        # Pairwise Euclidean distance on unit sphere: D = sqrt(2 - 2 * cos_sim)
        dot_product = torch.matmul(embeddings, embeddings.T)
        # Clamp dot product to [-1, 1] to avoid negative sqrt
        dist_sq = 2.0 - 2.0 * dot_product.clamp(-1.0, 1.0)
        dist = torch.sqrt(dist_sq.clamp(min=1e-12))

        # Binary masks
        labels = labels.unsqueeze(0)
        is_pos = torch.eq(labels, labels.T)
        is_neg = ~is_pos

        # Hardest positive: maximum distance among positives (exclude self)
        dist_pos = dist * is_pos.float()
        hardest_pos, _ = torch.max(dist_pos, dim=1)

        # Hardest negative: minimum distance among negatives
        # Add high distance to positives so they aren't selected
        dist_neg = dist + 1e5 * is_pos.float()
        hardest_neg, _ = torch.min(dist_neg, dim=1)

        # Triplet margin loss
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss (Deng et al., CVPR 2019).
    Enforces angular margin m in geodesic hypersphere space:
      cos(theta_y + m)
    """
    def __init__(self, num_classes: int, embed_dim: int = 256, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # L2-normalize class weights
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return F.cross_entropy(output, labels)
