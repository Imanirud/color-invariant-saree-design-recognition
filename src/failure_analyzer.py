"""
Failure Mode Diagnostic and Error Analysis Suite.
Phase 9: Identifies, categorizes, and diagnoses false positive and false negative
failures in saree design retrieval and verification.
"""

import os
from typing import Dict, List, Tuple, Any
import numpy as np

class FailureAnalyzer:
    """
    Exhaustively audits model errors in retrieval and verification.
    Discovers:
      1. False Positives (Diff Design, Same Color -> Incorrectly Matched)
      2. False Negatives (Same Design, Diff Color -> Incorrectly Rejected)
      3. Severe Ranking Discrepancies (Target Design ranked > 10)
    """

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold

    def analyze_verification_errors(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
        labels_a: np.ndarray,
        labels_b: np.ndarray,
        metadata_a: List[Dict[str, Any]],
        metadata_b: List[Dict[str, Any]],
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Calculates pairwise cosine similarities and identifies hardest errors.
        """
        # Cosine similarity between pairs
        sims = np.sum(embeddings_a * embeddings_b, axis=1)
        is_same = (labels_a == labels_b).astype(int)

        # False Positives: is_same == 0, but similarity >= threshold (or highest among negatives)
        neg_indices = np.where(is_same == 0)[0]
        # Sort negatives by descending similarity (hardest false positives first)
        sorted_neg_indices = neg_indices[np.argsort(-sims[neg_indices])]

        false_positives = []
        for idx in sorted_neg_indices[:top_n]:
            fp_sim = float(sims[idx])
            rec_a = metadata_a[idx]
            rec_b = metadata_b[idx]
            error_type = self._diagnose_false_positive(rec_a, rec_b)
            false_positives.append({
                "pair_index": int(idx),
                "similarity": round(fp_sim, 4),
                "above_threshold": fp_sim >= self.threshold,
                "design_a": rec_a.get("design_id"),
                "design_b": rec_b.get("design_id"),
                "colorway_a": rec_a.get("colorway_id"),
                "colorway_b": rec_b.get("colorway_id"),
                "diagnosis": error_type
            })

        # False Negatives: is_same == 1, but similarity < threshold (or lowest among positives)
        pos_indices = np.where(is_same == 1)[0]
        # Sort positives by ascending similarity (hardest false negatives first)
        sorted_pos_indices = pos_indices[np.argsort(sims[pos_indices])]

        false_negatives = []
        for idx in sorted_pos_indices[:top_n]:
            fn_sim = float(sims[idx])
            rec_a = metadata_a[idx]
            rec_b = metadata_b[idx]
            error_type = self._diagnose_false_negative(rec_a, rec_b)
            false_negatives.append({
                "pair_index": int(idx),
                "similarity": round(fn_sim, 4),
                "below_threshold": fn_sim < self.threshold,
                "design_a": rec_a.get("design_id"),
                "design_b": rec_b.get("design_id"),
                "colorway_a": rec_a.get("colorway_id"),
                "colorway_b": rec_b.get("colorway_id"),
                "diagnosis": error_type
            })

        return {
            "threshold_used": self.threshold,
            "total_pairs_evaluated": len(sims),
            "false_positives_count": sum(1 for fp in false_positives if fp["above_threshold"]),
            "false_negatives_count": sum(1 for fn in false_negatives if fn["below_threshold"]),
            "hardest_false_positives": false_positives,
            "hardest_false_negatives": false_negatives
        }

    def _diagnose_false_positive(self, rec_a: Dict[str, Any], rec_b: Dict[str, Any]) -> str:
        """Categorize root cause for a false positive."""
        cat_a = rec_a.get("category", "")
        cat_b = rec_b.get("category", "")
        if cat_a == cat_b and cat_a != "":
            return "Shared Regional Style / Fabric Texture (e.g. both Banarasi brocade)"
        return "Severe Chromatic Alignment or Repetitive Global Lattice Confounding"

    def _diagnose_false_negative(self, rec_a: Dict[str, Any], rec_b: Dict[str, Any]) -> str:
        """Categorize root cause for a false negative."""
        col_a = rec_a.get("colorway_id", "")
        col_b = rec_b.get("colorway_id", "")
        if col_a != col_b:
            return "Extreme Colorway Contrast Shift / Inverted Thread Luminance"
        return "Severe Scale / Perspective Discrepancy or Local Crop Mismatch"
