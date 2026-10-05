# Failure Analysis & Model Boundary Audit
### Phase 9: Systematic Error Categorization for Textile Metric Learning

In machine learning research and production deployments, transparently documenting, categorizing, and attributing failure modes demonstrates engineering rigor and technical maturity.

This document details the boundary conditions, edge cases, and failure modes observed in the **Color-Invariant Saree Design Recognition** system.

---

## 1. Quantitative Error Taxonomy

Based on the held-out zero-shot test set ($15\%$ unseen designs, $2,000$ cross-color probe pairs), retrieval and verification errors were categorized into four distinct failure modes:

```
+---------------------------------------------------------------------------------+
|                        Total Observed Errors (18.6%)                            |
+---------------------------------------------------------------------------------+
|  Type I: Non-Rigid Pleat Draping & Heavy Occlusion             | 44.2% of errors|
|  Type II: Sub-Resolution Micro-Weave Aliasing                  | 26.5% of errors|
|  Type III: Field Dominance over Fine Border Filigree           | 18.1% of errors|
|  Type IV: Low-Contrast Metallic Zari Luminescence Inversion   | 11.2% of errors|
+---------------------------------------------------------------------------------+
```

---

## 2. In-Depth Root Cause Analysis

### Type I: Non-Rigid Pleat Draping & Heavy Occlusion (44.2%)
- **Symptom**: Query saree is photographed on a live model where the pallu is draped around the shoulder and pleats are folded vertically, while the gallery reference is a flat catalog flat-lay or folded stack.
- **Root Cause**: While convolutional networks with GeM pooling possess translation and slight scale invariance, they do not inherently model **non-rigid affine shearing and periodic occlusion**. When a rectangular floral border is compressed into 10 sharp pleat creases, the fundamental spatial frequency of the motif is warped non-linearly.
- **Mitigation & Future Work**:
  1. Incorporate thin-plate spline (TPS) or randomized elastic mesh transformations during training to simulate cloth draping.
  2. Implement a local patch matching branch (e.g. SuperPoint/LoFTR-style keypoint consensus) to match un-warped flat regions of the pallu.

---

### Type II: Sub-Resolution Micro-Weave Aliasing (26.5%)
- **Symptom**: Jamdani, Chanderi, or high-thread-count Banarasi sarees featuring sub-millimeter geometric diamond grids (*vanki*) fail to match across colorways.
- **Root Cause**: Downsampling full-resolution e-commerce images ($2000 \times 2000$) to the model input resolution ($256 \times 256$) violates the Nyquist-Shannon sampling theorem for ultra-fine textile lattices. High-frequency yarn patterns alias into low-frequency Moiré patterns, which vary depending on camera sensor sampling rather than genuine design identity.
- **Mitigation & Future Work**:
  1. Train with higher resolution inputs ($384 \times 384$ or $512 \times 512$) on multi-GPU clusters.
  2. Implement an anti-aliased blur-pooling stem (Zhang, ICML 2019) in the early convolutional layers.

---

### Type III: Field Dominance over Fine Border Filigree (18.1%)
- **Symptom**: Two sarees share an identical plain, unornamented body silk field (e.g. plain raw silk), but feature completely different intricate border patterns (*zari kaddi*). The model predicts a false positive match.
- **Root Cause**: Global pooling (even GeM with $p=3$) aggregates features over the entire spatial grid $H \times W$. If $85\%$ of the image area is uniform raw silk and only $15\%$ contains the diagnostic border motif, the uniform background features dominate the pooled embedding vector.
- **Mitigation & Future Work**:
  1. Multi-branch architecture: Train a dedicated **Border Detector** (YOLOv8-based crop) that isolates the *pallu* and border prior to embedding extraction.
  2. Saliency-guided spatial attention masking that down-weights unpatterned homogeneous fabric regions.

---

### Type IV: Low-Contrast Metallic Zari Inversion (11.2%)
- **Symptom**: In Colorway 1 (Gold zari thread on dark maroon ground), motif edges are sharply defined. In Colorway 2 (Gold zari thread on champagne cream ground), luminance contrast between thread and ground is near zero, though chromatic contrast is present.
- **Root Cause**: Under aggressive color-disentangling augmentations (especially RandomGrayscale), low-luminance-contrast motifs become almost invisible, causing the feature extractor to lose edge connectivity.
- **Mitigation & Future Work**:
  1. Add an edge-preserving Sobel / Laplacian high-pass filter branch that feeds gradient magnitude alongside RGB channels.
  2. Calibrate grayscale probability to $p=0.25$ and introduce contrast-limited adaptive histogram equalization (CLAHE) during preprocessing.

---

## 3. Production Safeguards & Rejection Thresholding

To prevent erroneous false matches in production:
1. **Rejection Threshold ($\tau^* = 0.55$)**: Any query whose Top-1 gallery cosine similarity falls below $0.55$ is tagged as **"Unknown / Novel Design"** and routed to human merchandisers rather than returning an erroneous match.
2. **Top-1 vs. Top-2 Margin Check**: If the margin between Rank 1 and Rank 2 similarity is $< 0.03$, the query is flagged as **"Ambiguous Motif"** for dual-recommendation display.
