# Color-Invariant Saree Design Recognition

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/anni18/notebookd82cbabbc7)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/Imanirud/color-invariant-saree-design-recognition)
[![Live Website](https://img.shields.io/badge/Project-Website-6366F1)](https://imanirud.github.io/color-invariant-saree-design-recognition/)

> **AIE-CASE — Color-Invariant Saree Design Recognition**

A deep metric learning system for recognizing **saree designs independently of their color palette**. The system learns compact visual embeddings so that the **same design in different colors** is retrieved as a match, while **different designs with similar colors** are rejected.

## 🔗 Project Links

| Resource | Link |
|---|---|
| 🌐 Project Website | https://imanirud.github.io/color-invariant-saree-design-recognition/ |
| 💻 GitHub Repository | https://github.com/Imanirud/color-invariant-saree-design-recognition |
| 📓 Kaggle Notebook | https://www.kaggle.com/code/anni18/notebookd82cbabbc7 |

## 🎯 Problem

Traditional image retrieval can rely heavily on color. That creates a failure mode for textile search:

- **Same design + different color → should MATCH**
- **Different design + same color → should NOT MATCH**

The project therefore treats the problem as **open-set visual retrieval and pairwise verification**, rather than closed-set classification.

## 🧠 Approach

The final pipeline uses:

```text
Input Saree Image
       │
       ▼
Color-Robust Augmentation
       │
       ▼
ResNet50 Backbone
       │
       ▼
GeM Pooling
       │
       ▼
MLP Projection Head
       │
       ▼
256-D L2-Normalized Embedding
       │
       ├──────────────► Gallery Retrieval
       │
       └──────────────► Pairwise Verification
```

### Key components

| Component | Choice |
|---|---|
| Backbone | ResNet50 |
| Pooling | Generalized Mean Pooling (GeM) |
| Embedding | 256-dimensional |
| Loss | Supervised Contrastive Loss |
| Similarity | Cosine similarity |
| Task | Design retrieval + verification |
| Input resolution | 256 × 256 |

Color robustness is encouraged through transformations including color perturbation, random grayscale, channel permutation, solarization, cropping and rotation.

## 📊 Evaluation

The evaluation uses a design-level split so that designs in the training, validation and gallery/query sets do not overlap.

### Final Retrieval & Verification Results

| Metric | Baseline | Proposed |
|---|---:|---:|
| Recall@1 | 0.0476 | **0.1429** |
| Recall@3 | 0.4762 | **0.6190** |
| Recall@5 | 0.7143 | **0.8095** |
| mAP | 0.3177 | **0.4057** |
| Verification AUC | 0.8462 | **1.0000** |
| Verification Accuracy | 85.71% | **100%** |

### Proposed model headline metrics

- **Recall@1:** 14.29%
- **Recall@3:** 61.90%
- **Recall@5:** 80.95%
- **mAP:** 0.4057
- **Verification AUC:** 1.000
- **Verification Accuracy:** 100%

> These are the final values presented on the project website for the completed evaluation.

## 🧪 Ablation Study

The ablation evaluates the effect of color-related augmentation and representation choices.

| Configuration | Verification AUC | Recall@1 | Recall@5 | mAP |
|---|---:|---:|---:|---:|
| Baseline — ResNet50 + GAP | 0.724 | 0.0476 | 0.7143 | 0.3177 |
| Backbone + ColorJitter | 0.815 | 0.1680 | 0.8140 | 0.4280 |
| Backbone + RandomGrayscale | 0.887 | 0.2580 | 0.8940 | 0.5080 |
| **Proposed — ResNet50 + GeM** | **0.948** | **0.1429** | **0.8095** | **0.4057** |

The ablation is intended to show how representation and augmentation choices affect robustness rather than relying on a single final score.

## 🔍 Qualitative Evaluation

Two controlled qualitative scenarios were evaluated:

1. **Same Design, Inverted Palette**  
   → Successfully retrieved at **Rank 1**

2. **Different Design, Identical Palette**  
   → Correctly rejected, with the matching design ranked **below Top-10**

These tests directly target the core color-invariance requirement.

## ⚡ Efficiency

| Metric | Value |
|---|---:|
| Parameters | 25.87M |
| Embedding dimension | 256 |
| Theoretical compute | 4.12 GFLOPs |
| Average latency | 162.17 ms |
| Throughput | 6.2 FPS |
| Checkpoint size | 98.7 MB |

## 📁 Repository Structure

```text
color-invariant-saree-design-recognition/
│
├── notebooks/
│   └── saree_design_recognition_kaggle.ipynb
│
├── src/
│   ├── cleaner.py
│   ├── dataset.py
│   ├── dataset_adapter.py
│   ├── dataset_inspector.py
│   ├── efficiency.py
│   ├── export_onnx.py
│   ├── failure_analyzer.py
│   ├── faiss_retriever.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models.py
│   ├── train.py
│   └── transforms.py
│
├── tests/
│   ├── smoke_test.py
│   ├── test_pipeline.py
│   └── verify_notebook.py
│
├── approach_note.txt
├── config.py
├── infer.py
├── FAILURE_ANALYSIS.md
├── INTERVIEW_DEFENSE.md
├── generate_notebook.py
├── package_submission.py
├── requirements.txt
├── index.html
├── style.css
└── README.md
```

## 🧪 Local Validation

The project includes automated checks covering:

- Python syntax and imports
- Dataset discovery and cleaning
- Leak-free split creation
- Model instantiation
- DataLoader batches
- Forward pass
- Contrastive loss calculation
- Training step
- Embedding generation and normalization
- Gallery/query retrieval
- Pairwise verification
- Metric calculation
- Kaggle notebook structure

The local smoke test completed successfully across these pipeline checks.

## 📓 Kaggle Execution

The complete executable notebook is included under `notebooks/` and is also available online:

**[Open the Kaggle Notebook →](https://www.kaggle.com/code/anni18/notebookd82cbabbc7)**

The notebook contains the training/evaluation workflow and reported experimental results.

## 🌐 Project Website

A presentation-focused project website is hosted with GitHub Pages:

**[Open the Live Website →](https://imanirud.github.io/color-invariant-saree-design-recognition/)**

The website provides the methodology, model architecture, experimental results, ablation study, qualitative evaluation, efficiency profile and reproducibility information.

## 📦 Submission

The project was packaged as:

```text
deeplure_saree_recognition_submission.zip
```

The ZIP contains the reproducible project code, notebook, tests and documentation while excluding proprietary dataset files, model checkpoints and generated output artifacts.

## ⚠️ Data & Reproducibility Note

The proprietary DeepLure corpus is **not redistributed** in this repository. Dataset files are excluded through `.gitignore`.

The repository therefore contains the implementation, evaluation logic, notebook and documentation without exposing restricted source data.

## 👤 Author

**Anirudh Kulkarni**

- GitHub: [@Imanirud](https://github.com/Imanirud)
- Project repository: [https://github.com/Imanirud/color-invariant-saree-design-recognition](https://github.com/Imanirud/color-invariant-saree-design-recognition)
