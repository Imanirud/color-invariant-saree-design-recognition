"""
Central Configuration for Color-Invariant Saree Design Recognition.
Reproducible, portable, and Kaggle-ready.
"""
import os
import random
import numpy as np
import torch

class Config:
    # Environment & Reproducibility
    SEED = 42
    NUM_WORKERS = 2  # Kaggle free tier default is optimal with 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True   # Mixed precision (torch.amp) for faster training & lower VRAM
    
    # Dataset & Paths
    # Can be overridden via env vars or Kaggle input paths
    DATA_DIR = os.environ.get("DATA_DIR", "./data")
    KAGGLE_DATASET_DIR = "/kaggle/input/indian-saree-patterns"
    DEEPLURE_DATASET_DIR = "./data/deeplure_corpus"
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
    
    # Model Architecture
    BACKBONE = "convnext_small.fb_in22k_ft_in1k_384" # or 'resnet50', 'efficientnet_b2', 'dinov2_vits14'
    DEFAULT_BACKBONE = "resnet50" # Clean, widely available baseline
    EMBED_DIM = 256              # 256-D provides optimal balance between expressiveness & retrieval speed
    PRETRAINED = True
    POOLING = "gem"              # Generalized Mean Pooling (GeM) or 'avg'
    GEM_P = 3.0                  # GeM initial power
    NORMALIZE_EMBEDDINGS = True  # L2 normalization for cosine geometry
    
    # Preprocessing & Resolution
    IMG_SIZE = (256, 256)        # Standard resolution preserving fine textile weaves
    CROP_SIZE = (224, 224)
    
    # Optimization & Metric Learning
    BATCH_SIZE = 32              # PK sampler: P classes x K instances = e.g., 8 x 4 = 32
    P_CLASSES = 8                # Number of distinct designs per batch
    K_INSTANCES = 4              # Number of colorways/images per design in batch
    EPOCHS = 15
    LR = 1e-4
    BACKBONE_LR_RATIO = 0.1      # Differential learning rate: 1e-5 for pretrained backbone
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 2
    
    # Metric Learning Loss Parameters
    LOSS_TYPE = "SupCon"         # 'SupCon', 'BatchHardTriplet', 'ArcFace'
    TEMPERATURE = 0.07           # SupCon temperature parameter
    MARGIN = 0.3                 # Triplet margin or ArcFace margin
    SCALE = 30.0                 # ArcFace scale (s)
    
    # Verification & Evaluation
    VERIFICATION_VAL_PAIRS = 2000
    VERIFICATION_TEST_PAIRS = 2000
    TOP_K_RETRIEVAL = [1, 3, 5, 10]
    
    @classmethod
    def set_seed(cls, seed=None):
        if seed is None:
            seed = cls.SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

# Global default config dictionary
CONFIG = {
    "SEED": Config.SEED,
    "IMG_SIZE": Config.IMG_SIZE,
    "BATCH_SIZE": Config.BATCH_SIZE,
    "P_CLASSES": Config.P_CLASSES,
    "K_INSTANCES": Config.K_INSTANCES,
    "EPOCHS": Config.EPOCHS,
    "LR": Config.LR,
    "WEIGHT_DECAY": Config.WEIGHT_DECAY,
    "EMBED_DIM": Config.EMBED_DIM,
    "TEMPERATURE": Config.TEMPERATURE,
    "MARGIN": Config.MARGIN,
    "BACKBONE": Config.DEFAULT_BACKBONE,
    "NUM_WORKERS": Config.NUM_WORKERS,
    "USE_AMP": Config.USE_AMP,
    "NORMALIZE_EMBEDDINGS": Config.NORMALIZE_EMBEDDINGS
}
