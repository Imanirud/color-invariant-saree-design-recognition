"""
Evaluation Metrics for Saree Retrieval, Verification, and Color Invariance.
Phase 3, 6, & 8: Precision@K, Recall@K, mAP, MRR, ROC-AUC, EER, and Color-Invariance Subsets.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np

def compute_retrieval_metrics(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Computes standard image retrieval metrics:
      - Recall@K: Proportion of queries where at least one correct gallery match is within top K.
      - Mean Reciprocal Rank (MRR): Average of 1 / rank of first correct match.
      - Mean Average Precision (mAP): Mean AP across all queries.
      
    Assumes embeddings are L2-normalized; cosine similarity = dot product.
    """
    num_queries = query_embeddings.shape[0]
    num_gallery = gallery_embeddings.shape[0]
    
    # Compute full pairwise similarity matrix: (num_queries x num_gallery)
    sim_matrix = np.dot(query_embeddings, gallery_embeddings.T)

    recalls = {k: 0 for k in top_k}
    mrr_sum = 0.0
    ap_sum = 0.0

    max_k = max(top_k)

    for i in range(num_queries):
        q_label = query_labels[i]
        sims = sim_matrix[i]
        
        # Sort gallery indices descending by similarity
        sorted_indices = np.argsort(-sims)
        sorted_gallery_labels = gallery_labels[sorted_indices]

        # Binary indicator of matching design
        matches = (sorted_gallery_labels == q_label)
        total_relevant = np.sum(matches)

        if total_relevant == 0:
            continue

        # Recall@K
        for k in top_k:
            if np.any(matches[:k]):
                recalls[k] += 1

        # MRR
        first_match_rank = np.where(matches)[0]
        if len(first_match_rank) > 0:
            mrr_sum += 1.0 / (first_match_rank[0] + 1)

        # Average Precision (AP)
        cum_matches = np.cumsum(matches)
        ranks = np.arange(1, len(matches) + 1)
        precisions = cum_matches / ranks
        ap = np.sum(precisions * matches) / total_relevant
        ap_sum += ap

    results = {
        f"Recall@{k}": round(recalls[k] / float(num_queries), 4) for k in top_k
    }
    results["mAP"] = round(ap_sum / float(num_queries), 4)
    results["MRR"] = round(mrr_sum / float(num_queries), 4)
    return results

def calibrate_verification_threshold(val_sims: np.ndarray, val_targets: np.ndarray) -> Tuple[float, float]:
    """
    Calibrates the optimal verification decision threshold tau*
    STRICTLY on the Validation Set using Youden's J statistic (TPR - FPR).
    Returns (optimal_threshold, best_f1).
    """
    thresholds = np.linspace(-1.0, 1.0, 501)
    best_j = -1.0
    best_thresh = 0.5
    best_f1 = 0.0

    for t in thresholds:
        preds = (val_sims >= t).astype(int)
        tp = np.sum((preds == 1) & (val_targets == 1))
        fp = np.sum((preds == 1) & (val_targets == 0))
        fn = np.sum((preds == 0) & (val_targets == 1))
        tn = np.sum((preds == 0) & (val_targets == 0))

        tpr = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / float(fp + tn) if (fp + tn) > 0 else 0.0
        j_stat = tpr - fpr

        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if j_stat > best_j:
            best_j = j_stat
            best_thresh = t
            best_f1 = f1

    return float(best_thresh), float(best_f1)

def compute_verification_metrics(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    is_same_design: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluates pairwise design verification:
      - Given Image A and Image B, determine if they carry the same design.
      - Calculates Cosine Similarity, ROC-AUC, EER, and Accuracy/F1 at threshold.
    """
    # Pairwise cosine similarity: row-wise dot product
    sims = np.sum(embeddings_a * embeddings_b, axis=1)
    targets = is_same_design.astype(int)

    # Compute ROC curve and AUC manually without heavy sklearn dependencies
    sorted_order = np.argsort(-sims)
    sorted_targets = targets[sorted_order]
    
    n_pos = np.sum(targets == 1)
    n_neg = np.sum(targets == 0)

    if n_pos == 0 or n_neg == 0:
        return {"AUC": 0.0, "Accuracy": 0.0, "Optimal_Threshold": 0.5}

    tpr_list, fpr_list = [0.0], [0.0]
    cum_tp, cum_fp = 0, 0

    for t in sorted_targets:
        if t == 1:
            cum_tp += 1
        else:
            cum_fp += 1
        tpr_list.append(cum_tp / float(n_pos))
        fpr_list.append(cum_fp / float(n_neg))

    # Trapezoidal integration for ROC-AUC (compatible with NumPy 1.x and 2.x)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = trapz_fn(tpr_list, fpr_list)

    # Find Equal Error Rate (EER): where FPR == (1 - TPR)
    fnr = 1.0 - np.array(tpr_list)
    eer_idx = np.nanargmin(np.abs(np.array(fpr_list) - fnr))
    eer = float((fpr_list[eer_idx] + fnr[eer_idx]) / 2.0)

    # Use provided threshold or calibrate
    if threshold is None:
        threshold, _ = calibrate_verification_threshold(sims, targets)

    preds = (sims >= threshold).astype(int)
    tp = np.sum((preds == 1) & (targets == 1))
    fp = np.sum((preds == 1) & (targets == 0))
    fn = np.sum((preds == 0) & (targets == 1))
    tn = np.sum((preds == 0) & (targets == 0))

    accuracy = (tp + tn) / float(len(targets))
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "ROC_AUC": round(float(auc), 4),
        "EER": round(eer, 4),
        "Threshold": round(threshold, 4),
        "Accuracy": round(float(accuracy), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1": round(float(f1), 4),
        "Mean_Positive_Sim": round(float(np.mean(sims[targets == 1])), 4),
        "Mean_Negative_Sim": round(float(np.mean(sims[targets == 0])), 4)
    }

def evaluate_color_invariance_breakdown(
    subsets_sims: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Evaluates the 4 core color-invariance subsets:
      TEST A: Same Design + Same/Similar Color
      TEST B: Same Design + Different Color (CRITICAL OBJECTIVE)
      TEST C: Different Design + Same Color (CHIEF FAILURE MODE)
      TEST D: Different Design + Different Color
      
    Computes Color Bias Ratio (CBR) and Invariance Gap.
    """
    sim_a = float(np.mean(subsets_sims.get("TEST_A", [0.0])))
    sim_b = float(np.mean(subsets_sims.get("TEST_B", [0.0])))
    sim_c = float(np.mean(subsets_sims.get("TEST_C", [0.0])))
    sim_d = float(np.mean(subsets_sims.get("TEST_D", [0.0])))

    # Color Bias Ratio: ratio of false similarity (Test C) to cross-color true similarity (Test B)
    # A naive model has CBR > 1.0 (color dominates design).
    # A color-invariant model has CBR < 0.60.
    cbr = sim_c / max(sim_b, 1e-4)

    # Invariance Gap: Drop in similarity when colorway changes for the same design
    invariance_gap = sim_a - sim_b

    # Design Discrimination Margin: Margin between true match in diff color vs false match in same color
    discrimination_margin = sim_b - sim_c

    return {
        "Test_A_Mean_Sim (SameDesign_SameColor)": round(sim_a, 4),
        "Test_B_Mean_Sim (SameDesign_DiffColor)": round(sim_b, 4),
        "Test_C_Mean_Sim (DiffDesign_SameColor)": round(sim_c, 4),
        "Test_D_Mean_Sim (DiffDesign_DiffColor)": round(sim_d, 4),
        "Color_Bias_Ratio (Test_C / Test_B)": round(cbr, 4),
        "Invariance_Gap (Test_A - Test_B)": round(invariance_gap, 4),
        "Discrimination_Margin (Test_B - Test_C)": round(discrimination_margin, 4)
    }
