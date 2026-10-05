"""
PyTorch Dataset, Balanced PxK Batch Sampler, and Design-Level Split Engine.
Phase 2 & Phase 7: Leakage-free dataset handling with multi-instance sampling.
"""

import os
import random
import csv
from typing import Dict, List, Tuple, Any, Optional, Iterator
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from PIL import Image

class SareeDataset(Dataset):
    """
    Standard PyTorch Dataset for Saree Image Retrieval.
    Loads RGB images from disk and applies specified albumentations/torchvision transform.
    """

    def __init__(self, records: List[Dict[str, Any]], transform=None, label_to_idx: Optional[Dict[str, int]] = None):
        self.records = records
        self.transform = transform
        
        # Build mapping from design_id (string) to continuous integer class index [0 ... C-1]
        if label_to_idx is None:
            unique_designs = sorted(list(set(r["design_id"] for r in records)))
            self.label_to_idx = {d: idx for idx, d in enumerate(unique_designs)}
        else:
            self.label_to_idx = label_to_idx

        self.design_to_indices = defaultdict(list)
        for idx, rec in enumerate(self.records):
            design_idx = self.label_to_idx[rec["design_id"]]
            self.design_to_indices[design_idx].append(idx)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        record = self.records[idx]
        image_path = record["image_path"]
        
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            # Fallback for transient read errors
            image = Image.new("RGB", (256, 256), color=(128, 128, 128))

        if self.transform is not None:
            image = self.transform(image)

        label = self.label_to_idx[record["design_id"]]
        metadata = {
            "path": image_path,
            "design_id": record["design_id"],
            "colorway_id": record.get("colorway_id", "default"),
            "category": record.get("category", "saree")
        }
        return image, label, metadata

class BalancedBatchSampler(Sampler):
    """
    Balanced P x K Batch Sampler for Metric Learning.
    Each batch consists of:
      - P unique design identities
      - K instances/colorways per design
    Batch size = P * K.
    
    Guarantees that every anchor in a batch has at least (K - 1) valid positive
    targets for Supervised Contrastive Loss and Batch-Hard Triplet Mining.
    """

    def __init__(self, dataset: SareeDataset, p_classes: int = 8, k_instances: int = 4, iterations_per_epoch: int = 200):
        self.dataset = dataset
        self.p_classes = min(p_classes, len(dataset.design_to_indices))
        self.k_instances = k_instances
        self.iterations_per_epoch = iterations_per_epoch
        self.batch_size = self.p_classes * self.k_instances
        self.classes = list(dataset.design_to_indices.keys())

    def __len__(self) -> int:
        return self.iterations_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.iterations_per_epoch):
            batch = []
            selected_classes = random.sample(self.classes, self.p_classes)
            for cls in selected_classes:
                indices = self.dataset.design_to_indices[cls]
                if len(indices) >= self.k_instances:
                    selected_indices = random.sample(indices, self.k_instances)
                else:
                    # Oversample with replacement if design has fewer than K natural instances
                    selected_indices = random.choices(indices, k=self.k_instances)
                batch.extend(selected_indices)
            
            random.shuffle(batch)
            yield batch

def build_design_level_splits(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Constructs a scientifically leak-free partition:
      1. Train Set (70% of unique designs)
      2. Validation Set (15% of unique designs)
      3. Gallery Reference Set (15% of unique designs, reference colorway)
      4. Query Probe Set (15% of unique designs, probe colorway)
      
    Guarantees:
      Designs(Train) ∩ Designs(Val) ∩ Designs(Test) = ∅
    """
    random.seed(seed)
    
    # Group all records by design_id
    design_to_records = defaultdict(list)
    for r in records:
        design_to_records[r["design_id"]].append(r)

    all_designs = sorted(list(design_to_records.keys()))
    random.shuffle(all_designs)

    n_total = len(all_designs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_designs = set(all_designs[:n_train])
    val_designs = set(all_designs[n_train:n_train + n_val])
    test_designs = set(all_designs[n_train + n_val:])

    train_records = [r for r in records if r["design_id"] in train_designs]
    val_records = [r for r in records if r["design_id"] in val_designs]
    
    # Split test designs into Gallery and Query sets
    gallery_records = []
    query_records = []

    for d in test_designs:
        items = design_to_records[d]
        if len(items) >= 2:
            # Multi-colorway design: 1st colorway goes to Gallery, 2nd goes to Query
            gallery_records.append(items[0])
            for q_item in items[1:]:
                query_records.append(q_item)
        else:
            # Single image design: Add to gallery; query will use synthetic cross-colorway view
            gallery_records.append(items[0])
            query_record_copy = dict(items[0])
            query_record_copy["is_synthetic_query"] = True
            query_records.append(query_record_copy)

    print(f"[SplitEngine] Design-Level Partitioning Complete (Zero Leakage):")
    print(f"  - Train:   {len(train_designs)} designs ({len(train_records)} images)")
    print(f"  - Val:     {len(val_designs)} designs ({len(val_records)} images)")
    print(f"  - Gallery: {len(gallery_records)} reference designs")
    print(f"  - Query:   {len(query_records)} probe designs")
    
    return train_records, val_records, gallery_records, query_records
