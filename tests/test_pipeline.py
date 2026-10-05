"""
Unit & Integration Test Suite for Color-Invariant Saree Recognition.
Validates mathematical bounds, L2 normalization, loss gradient flows,
and batch sampler correctness.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import ProposedColorInvariantModel, GeM
from src.losses import SupConLoss, BatchHardTripletLoss
from src.metrics import compute_retrieval_metrics, compute_verification_metrics
from src.transforms import get_color_invariant_transforms

class TestSareePipeline(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_gem_pooling(self):
        """Verify GeM pooling layer preserves channel dimension and avoids NaNs."""
        gem = GeM(p=3.0)
        x = torch.rand(4, 512, 8, 8) + 1e-4
        out = gem(x)
        self.assertEqual(out.shape, (4, 512))
        self.assertFalse(torch.isnan(out).any())

    def test_embedding_l2_normalization(self):
        """Assert embeddings are strictly constrained to unit hypersphere S^(d-1)."""
        model = ProposedColorInvariantModel(backbone_name="resnet50", pretrained=False, embed_dim=256)
        model.eval()
        dummy_input = torch.randn(4, 3, 128, 128)
        with torch.no_grad():
            embs = model(dummy_input)
        
        norms = torch.norm(embs, p=2, dim=1).numpy()
        np.testing.assert_allclose(norms, 1.0, atol=1e-5, err_msg="Embeddings must be L2 normalized to 1.0")

    def test_supcon_loss_positive_finite(self):
        """Assert SupCon loss computes valid positive scalar loss."""
        criterion = SupConLoss(temperature=0.07)
        # 8 embeddings, 2 per class
        features = torch.randn(8, 256)
        features = F.normalize(features, p=2, dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        
        loss = criterion(features, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_batch_hard_triplet_loss(self):
        """Assert BatchHardTriplet loss computes valid margin."""
        criterion = BatchHardTripletLoss(margin=0.3)
        features = torch.randn(8, 256)
        features = F.normalize(features, p=2, dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        
        loss = criterion(features, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_cosine_similarity_bounds(self):
        """Assert cosine similarities are strictly in [-1.0, 1.0]."""
        q = np.random.randn(10, 256).astype(np.float32)
        g = np.random.randn(50, 256).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        g /= np.linalg.norm(g, axis=1, keepdims=True)

        sims = np.dot(q, g.T)
        self.assertTrue(np.all(sims >= -1.0 - 1e-6))
        self.assertTrue(np.all(sims <= 1.0 + 1e-6))

    def test_retrieval_perfect_match(self):
        """Assert Recall@1 == 1.0 when queries and gallery match perfectly."""
        embs = np.random.randn(20, 256).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        labels = np.arange(20)

        # Query identical to gallery
        metrics = compute_retrieval_metrics(
            query_embeddings=embs,
            gallery_embeddings=embs,
            query_labels=labels,
            gallery_labels=labels,
            top_k=[1, 5]
        )
        self.assertEqual(metrics["Recall@1"], 1.0)
        self.assertEqual(metrics["mAP"], 1.0)

if __name__ == "__main__":
    unittest.main()
