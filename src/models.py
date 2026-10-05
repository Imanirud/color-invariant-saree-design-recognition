"""
Model Architectures for Saree Design Recognition.
Phase 3 & Phase 9: Baseline Pretrained Feature Extractor and Proposed
Metric-Learning Architecture with Generalized Mean (GeM) Pooling.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GeM(nn.Module):
    """
    Generalized Mean (GeM) Pooling layer.
    Computes: f(x) = [ (1/N) * sum_i (x_i^p) ]^(1/p)
    When p = 1: Average Pooling.
    When p -> infinity: Max Pooling.
    Learned p (~3.0) sharpens salient motif activations while suppressing uniform background.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if learnable else torch.tensor(p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp to avoid NaN in pow gradient
        p_val = self.p.clamp(min=1.0, max=10.0)
        x_clamped = x.clamp(min=self.eps)
        pooled = F.adaptive_avg_pool2d(x_clamped.pow(p_val), (1, 1)).pow(1.0 / p_val)
        return pooled.flatten(1)

    def __repr__(self):
        return f"GeM(p={self.p.data.tolist()[0]:.2f}, eps={self.eps})"

class BaselineRetrievalModel(nn.Module):
    """
    Phase 3 Baseline: Standard Pretrained ImageNet Backbone + Global Average Pooling (GAP)
    + Optional Projection + L2 Normalization.
    Represents the default off-the-shelf feature extractor.
    """
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        embed_dim: Optional[int] = 256,
        normalize: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.normalize = normalize
        self.embed_dim = embed_dim

        # Instantiate torchvision backbone
        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            # Extract convolutional trunk (all layers except avgpool and fc)
            self.trunk = nn.Sequential(*list(base.children())[:-2])
        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            self.trunk = base.features
        else:
            raise ValueError(f"Unsupported baseline backbone: {backbone_name}")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if embed_dim is not None and embed_dim != in_features:
            self.projection = nn.Linear(in_features, embed_dim, bias=False)
        else:
            self.projection = nn.Identity()
            self.embed_dim = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        pooled = self.pool(features).flatten(1)
        embeddings = self.projection(pooled)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

class ProposedColorInvariantModel(nn.Module):
    """
    Phase 5 & 9 Proposed Model:
    Pretrained Visual Backbone + GeM Pooling + Multi-Layer Projection Head
    + L2 Normalization onto the unit hypersphere S^(d-1).
    """
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        embed_dim: int = 256,
        gem_p: float = 3.0,
        normalize: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embed_dim = embed_dim
        self.normalize = normalize

        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            self.trunk = nn.Sequential(*list(base.children())[:-2])
        elif backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = models.convnext_tiny(weights=weights)
            in_features = base.classifier[2].in_features
            self.trunk = base.features
        elif backbone_name == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b2(weights=weights)
            in_features = base.classifier[1].in_features
            self.trunk = base.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Salient motif pooling
        self.pool = GeM(p=gem_p, learnable=True)

        # Non-linear projection head (MLP) for metric space alignment
        self.head = nn.Sequential(
            nn.Linear(in_features, in_features // 2, bias=False),
            nn.BatchNorm1d(in_features // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, embed_dim, bias=False)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map prior to pooling."""
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.trunk(x)
        pooled = self.pool(feat_map)
        embeddings = self.head(pooled)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Calculate total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
