"""
Universal Dataset Adapter for Saree Design Recognition.
Phase 1 & Phase 2: Handles DeepLure corpus, Kaggle datasets, flat folders,
or custom directory layouts without hardcoding folder names.
"""

import os
import glob
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter

class SareeDatasetAdapter:
    """
    Standardizes dataset access across diverse directory conventions.
    Extracts (image_path, design_id, colorway_id, category).
    """

    SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.samples: List[Dict[str, str]] = []
        self._discover_and_parse()

    def _discover_and_parse(self):
        """Recursively scan and normalize image metadata."""
        if not os.path.exists(self.root_dir):
            return

        all_files = []
        for root, _, files in os.walk(self.root_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in self.SUPPORTED_EXTS:
                    all_files.append(os.path.join(root, f))

        for fpath in all_files:
            rel = os.path.relpath(fpath, self.root_dir)
            parts = rel.split(os.sep)
            filename = os.path.splitext(os.path.basename(fpath))[0]

            # Case A: Standard Class Folder Structure (e.g. data/Banarasi/img01.jpg)
            if len(parts) >= 2:
                category = parts[0]
            else:
                category = "unknown"

            # Case B: Filename encodes design and colorway (e.g. design12_red.jpg or D12_C01.jpg)
            if "_" in filename:
                tokens = filename.split("_")
                # Heuristic: if first token is design identifier
                design_id = f"{category}_{tokens[0]}"
                colorway_id = "_".join(tokens[1:])
            else:
                design_id = f"{category}_{filename}"
                colorway_id = "default"

            self.samples.append({
                "path": fpath,
                "relative_path": rel,
                "category": category,
                "design_id": design_id,
                "colorway_id": colorway_id,
                "filename": os.path.basename(fpath)
            })

    def get_summary(self) -> Dict[str, Any]:
        """Return summary of parsed records."""
        categories = Counter(s["category"] for s in self.samples)
        designs = Counter(s["design_id"] for s in self.samples)
        return {
            "total_images": len(self.samples),
            "num_categories": len(categories),
            "category_distribution": dict(categories),
            "num_unique_design_ids": len(designs),
            "avg_images_per_design": round(len(self.samples) / max(len(designs), 1), 2)
        }
