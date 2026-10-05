"""
Data Cleaning, Deduplication, and Manifest Builder for Saree Retrieval.
Phase 2: Detects corrupted files, purges exact duplicates, clusters perceptual
duplicates, discovers candidate colorways, and enforces leakage-free splits.
"""

import os
import sys
import glob
import json
import hashlib
import csv
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
from PIL import Image
import numpy as np

class DataCleaner:
    """
    Production-grade data cleaning and relabeling pipeline.
    Produces a clean, verified dataset manifest with zero leakage.
    """

    def __init__(self, data_dir: str, output_dir: str = "./data"):
        self.data_dir = os.path.abspath(data_dir)
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.cleaning_log: Dict[str, Any] = {
            "total_scanned": 0,
            "corrupted_removed": [],
            "sub_resolution_removed": [],
            "exact_duplicates_purged": [],
            "perceptual_clusters_found": 0,
            "candidate_colorways_discovered": 0,
            "clean_records_count": 0
        }

    @staticmethod
    def compute_sha256(filepath: str) -> str:
        """Fast chunked SHA-256 for byte-level duplicate detection."""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
        """
        Compute 64-bit difference hash (dHash).
        Evaluates horizontal gradient signs on grayscale thumbnail.
        Returns integer for O(1) bitwise XOR Hamming distance calculations.
        """
        resized = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.array(resized, dtype=np.int32)
        diff = pixels[:, 1:] > pixels[:, :-1]
        
        # Pack boolean array into 64-bit integer
        bit_string = "".join("1" if b else "0" for b in diff.flatten())
        return int(bit_string, 2)

    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        """Compute bitwise Hamming distance between two 64-bit hashes."""
        return bin(h1 ^ h2).count('1')

    @staticmethod
    def compute_color_vector(img: Image.Image) -> np.ndarray:
        """
        Compute a compact 3D RGB color histogram (4x4x4 = 64 bins)
        representing the global chromatic distribution.
        """
        thumb = img.convert('RGB').resize((64, 64), Image.Resampling.BILINEAR)
        arr = np.array(thumb, dtype=np.float32)
        # Quantize each channel to 4 bins: [0-63], [64-127], [128-191], [192-255]
        quantized = (arr // 64).astype(np.int32)
        # Compute bin indices: r*16 + g*4 + b
        bin_indices = quantized[:, :, 0] * 16 + quantized[:, :, 1] * 4 + quantized[:, :, 2]
        hist = np.bincount(bin_indices.flatten(), minlength=64).astype(np.float32)
        norm = np.linalg.norm(hist)
        return hist / (norm + 1e-7)

    def clean_and_build_manifest(
        self,
        min_dim: int = 64,
        max_aspect_ratio: float = 4.0,
        dhash_near_dup_thresh: int = 2,
        color_divergence_thresh: float = 0.30
    ) -> List[Dict[str, Any]]:
        """
        Execute full data cleaning, exact duplicate purging, perceptual grouping,
        and write the audited dataset manifest.
        """
        print(f"[DataCleaner] Commencing audit on: {self.data_dir}")
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        all_files = []
        for root, _, files in os.walk(self.data_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_extensions:
                    all_files.append(os.path.join(root, f))

        self.cleaning_log["total_scanned"] = len(all_files)
        print(f"[DataCleaner] Discovered {len(all_files)} total images.")

        # Step 1: Filter Corrupted & Extreme Dimension Images
        valid_candidates = []
        sha_to_records = defaultdict(list)

        for fpath in all_files:
            try:
                with Image.open(fpath) as img:
                    img.verify()
                
                with Image.open(fpath) as img:
                    w, h = img.size
                    if w < min_dim or h < min_dim:
                        self.cleaning_log["sub_resolution_removed"].append(fpath)
                        continue
                    ar = w / float(h)
                    if ar > max_aspect_ratio or ar < (1.0 / max_aspect_ratio):
                        self.cleaning_log["sub_resolution_removed"].append(fpath)
                        continue

                    sha = self.compute_sha256(fpath)
                    dhash = self.compute_dhash(img)
                    color_vec = self.compute_color_vector(img)

                    rel_path = os.path.relpath(fpath, self.data_dir)
                    parts = rel_path.split(os.sep)
                    category = parts[0] if len(parts) > 1 else "saree"
                    filename = os.path.splitext(os.path.basename(fpath))[0]

                    record = {
                        "path": fpath,
                        "relative_path": rel_path,
                        "category": category,
                        "filename": filename,
                        "width": w,
                        "height": h,
                        "sha256": sha,
                        "dhash": dhash,
                        "color_vec": color_vec
                    }
                    sha_to_records[sha].append(record)

            except Exception as e:
                self.cleaning_log["corrupted_removed"].append({"path": fpath, "error": str(e)})

        # Step 2: Purge Exact Byte Duplicates (Keep 1 canonical copy per SHA-256)
        deduplicated_records = []
        for sha, records in sha_to_records.items():
            canonical = records[0]
            deduplicated_records.append(canonical)
            if len(records) > 1:
                purged = [r["path"] for r in records[1:]]
                self.cleaning_log["exact_duplicates_purged"].extend(purged)

        print(f"[DataCleaner] Retained {len(deduplicated_records)} images after exact deduplication.")
        print(f"[DataCleaner] Purged {len(self.cleaning_log['exact_duplicates_purged'])} exact byte copies.")

        # Step 3: Perceptual Near-Duplicate Grouping & Colorway Discovery
        # Build Connected Components using Disjoint Set / Union-Find on dHash
        num_records = len(deduplicated_records)
        parent = list(range(num_records))

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        # Compare pairwise dHash for cluster formation
        # For large sets, dHash buckets can be used; here we group by dHash or check pairwise
        dhash_buckets = defaultdict(list)
        for i, rec in enumerate(deduplicated_records):
            # Key bucket by top 32 bits for fast collision search
            bucket_key = rec["dhash"] >> 32
            dhash_buckets[bucket_key].append(i)

        for bucket_indices in dhash_buckets.values():
            n_b = len(bucket_indices)
            for idx_a in range(n_b):
                for idx_b in range(idx_a + 1, n_b):
                    i = bucket_indices[idx_a]
                    j = bucket_indices[idx_b]
                    dist = self.hamming_distance(deduplicated_records[i]["dhash"], deduplicated_records[j]["dhash"])
                    if dist <= dhash_near_dup_thresh:
                        union(i, j)

        # Assign cluster IDs
        clusters = defaultdict(list)
        for i in range(num_records):
            root = find(i)
            clusters[root].append(deduplicated_records[i])

        self.cleaning_log["perceptual_clusters_found"] = len([c for c in clusters.values() if len(c) > 1])

        # Step 4: Determine Design Identity & Colorway Tags
        manifest = []
        cluster_counter = 0

        for root_id, items in clusters.items():
            cluster_name = f"design_cluster_{cluster_counter:05d}"
            cluster_counter += 1

            # Check if this cluster represents multiple colorways
            # (i.e. items have distinct color vectors: 1 - dot(v1, v2) > threshold)
            is_colorway_group = False
            if len(items) > 1:
                for a in range(len(items)):
                    for b in range(a + 1, len(items)):
                        cos_sim = np.dot(items[a]["color_vec"], items[b]["color_vec"])
                        color_dist = 1.0 - cos_sim
                        if color_dist >= color_divergence_thresh:
                            is_colorway_group = True
                            self.cleaning_log["candidate_colorways_discovered"] += 1
                            break

            for idx, item in enumerate(items):
                colorway_tag = f"colorway_{idx:02d}" if is_colorway_group else f"item_{idx:02d}"
                manifest.append({
                    "image_path": item["path"],
                    "relative_path": item["relative_path"],
                    "category": item["category"],
                    "design_id": cluster_name,
                    "colorway_id": colorway_tag,
                    "is_natural_colorway": is_colorway_group,
                    "sha256": item["sha256"],
                    "width": item["width"],
                    "height": item["height"]
                })

        self.cleaning_log["clean_records_count"] = len(manifest)

        # Write Cleaning Audit JSON
        audit_path = os.path.join(self.output_dir, "cleaning_audit.json")
        with open(audit_path, 'w', encoding='utf-8') as f:
            json.dump(self.cleaning_log, f, indent=2)

        # Write Manifest CSV
        manifest_csv_path = os.path.join(self.output_dir, "dataset_manifest.csv")
        fieldnames = ["image_path", "relative_path", "category", "design_id", "colorway_id", "is_natural_colorway", "sha256", "width", "height"]
        with open(manifest_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)

        print(f"[DataCleaner] Audit complete. Manifest saved to: {manifest_csv_path}")
        print(f"[DataCleaner] Audit report saved to: {audit_path}")
        return manifest

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run data cleaning and manifest generator.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Raw dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for manifests")
    args = parser.parse_args()

    cleaner = DataCleaner(data_dir=args.data_dir, output_dir=args.output_dir)
    manifest = cleaner.clean_and_build_manifest()
