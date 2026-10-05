"""
Dataset Inspector & Integrity Auditor for Saree Design Recognition.
Phase 1 & Phase 2: Autonomous dataset audit, anomaly detection, duplicate
identification, and colorway grouping.
"""

import os
import sys
import glob
import json
import hashlib
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image, ImageStat
import numpy as np

class DatasetInspector:
    """
    Exhaustive dataset diagnostic tool.
    Audits directory structure, checks image integrity, discovers exact and perceptual
    duplicates, computes dimension statistics, checks class distributions, and
    identifies candidate colorway groups.
    """
    
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    def __init__(self, root_dir: str, sample_size: Optional[int] = None):
        self.root_dir = os.path.abspath(root_dir)
        self.sample_size = sample_size
        self.image_records: List[Dict[str, Any]] = []
        self.corrupted_files: List[str] = []
        self.hash_to_files: Dict[str, List[str]] = defaultdict(list)
        self.phash_to_files: Dict[str, List[str]] = defaultdict(list)
        self.class_counts: Dict[str, int] = Counter()
        self.dimension_stats: Dict[str, Any] = {}
        
    def _compute_sha256(self, filepath: str) -> str:
        """Compute SHA-256 hash for exact duplicate detection."""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _compute_dhash(self, img: Image.Image, hash_size: int = 8) -> str:
        """
        Compute difference hash (dHash) for perceptual similarity.
        Robust against color variations and slight compression artifacts.
        Converts to grayscale and evaluates horizontal gradient differences.
        """
        # Resize to (hash_size + 1, hash_size) in grayscale
        resized = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.array(resized, dtype=np.float32)
        # Compare adjacent pixels horizontally: gradient > 0
        diff = pixels[:, 1:] > pixels[:, :-1]
        # Convert boolean array to hex string
        decimal_val = 0
        hex_str = []
        for index, val in enumerate(diff.flatten()):
            if val:
                decimal_val += 2 ** (index % 8)
            if (index % 8) == 7:
                hex_str.append(hex(decimal_val)[2:].rjust(2, '0'))
                decimal_val = 0
        return "".join(hex_str)

    def _compute_color_histogram(self, img: Image.Image, bins: int = 8) -> np.ndarray:
        """Compute normalized RGB color histogram for color distribution analysis."""
        img_rgb = img.convert('RGB')
        arr = np.array(img_rgb)
        hist, _ = np.histogramdd(arr.reshape(-1, 3), bins=(bins, bins, bins), range=[(0, 256), (0, 256), (0, 256)])
        hist = hist.astype(np.float32)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
        return hist.flatten()

    def scan(self) -> Dict[str, Any]:
        """Execute full dataset discovery and audit scan."""
        print(f"[DatasetInspector] Initiating audit on: {self.root_dir}")
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root directory does not exist: {self.root_dir}")

        all_candidate_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.VALID_EXTENSIONS:
                    all_candidate_files.append(os.path.join(root, file))

        total_discovered = len(all_candidate_files)
        print(f"[DatasetInspector] Discovered {total_discovered} total image files.")

        if self.sample_size and self.sample_size < total_discovered:
            import random
            random.seed(42)
            candidate_files = random.sample(all_candidate_files, self.sample_size)
            print(f"[DatasetInspector] Subsampled {self.sample_size} images for rapid audit.")
        else:
            candidate_files = all_candidate_files

        widths, heights, aspect_ratios, file_sizes = [], [], [], []
        colorway_candidates = defaultdict(list)

        for idx, fpath in enumerate(candidate_files):
            rel_path = os.path.relpath(fpath, self.root_dir)
            parts = rel_path.split(os.sep)
            
            # Infer class/category from immediate parent directory
            label = parts[0] if len(parts) > 1 else "root"
            self.class_counts[label] += 1

            file_size_kb = os.path.getsize(fpath) / 1024.0
            file_sizes.append(file_size_kb)

            # Integrity & metadata extraction
            try:
                with Image.open(fpath) as img:
                    img.verify() # Fast check for byte corruption
                
                # Reopen to read pixel data (verify closes the file handle)
                with Image.open(fpath) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    ar = round(w / float(h), 3) if h > 0 else 0
                    aspect_ratios.append(ar)
                    img_format = img.format
                    mode = img.mode

                    # Check for perceptual hash and exact hash
                    sha256 = self._compute_sha256(fpath)
                    dhash = self._compute_dhash(img)
                    
                    self.hash_to_files[sha256].append(fpath)
                    self.phash_to_files[dhash].append(fpath)

                    record = {
                        "path": fpath,
                        "relative_path": rel_path,
                        "label": label,
                        "filename": os.path.basename(fpath),
                        "width": w,
                        "height": h,
                        "aspect_ratio": ar,
                        "format": img_format,
                        "mode": mode,
                        "size_kb": round(file_size_kb, 2),
                        "sha256": sha256,
                        "dhash": dhash
                    }
                    self.image_records.append(record)

            except Exception as e:
                self.corrupted_files.append({"path": fpath, "error": str(e)})

            if (idx + 1) % 500 == 0 or (idx + 1) == len(candidate_files):
                print(f"[DatasetInspector] Processed {idx + 1}/{len(candidate_files)} images...")

        # Aggregate Duplicate Statistics
        exact_duplicate_groups = {k: v for k, v in self.hash_to_files.items() if len(v) > 1}
        perceptual_duplicate_groups = {k: v for k, v in self.phash_to_files.items() if len(v) > 1}

        # Candidate Colorways: images sharing identical or very close dHash (structural pattern)
        # but residing in separate files (potential multiple colorways of same weave motif)
        colorway_clusters = [v for k, v in perceptual_duplicate_groups.items() if len(v) > 1]

        # Aggregate Dimension Statistics
        if widths:
            self.dimension_stats = {
                "count": len(widths),
                "width": {
                    "min": int(np.min(widths)),
                    "max": int(np.max(widths)),
                    "mean": round(float(np.mean(widths)), 1),
                    "median": int(np.median(widths))
                },
                "height": {
                    "min": int(np.min(heights)),
                    "max": int(np.max(heights)),
                    "mean": round(float(np.mean(heights)), 1),
                    "median": int(np.median(heights))
                },
                "aspect_ratio": {
                    "min": float(np.min(aspect_ratios)),
                    "max": float(np.max(aspect_ratios)),
                    "mean": round(float(np.mean(aspect_ratios)), 3),
                    "median": round(float(np.median(aspect_ratios)), 3)
                },
                "size_kb": {
                    "min": round(float(np.min(file_sizes)), 1),
                    "max": round(float(np.max(file_sizes)), 1),
                    "mean": round(float(np.mean(file_sizes)), 1)
                },
                "tiny_images_count": sum(1 for w, h in zip(widths, heights) if w < 64 or h < 64),
                "oversized_images_count": sum(1 for w, h in zip(widths, heights) if w > 4000 or h > 4000)
            }

        report = {
            "root_directory": self.root_dir,
            "total_discovered": total_discovered,
            "total_valid_scanned": len(self.image_records),
            "corrupted_count": len(self.corrupted_files),
            "corrupted_files": self.corrupted_files,
            "classes": dict(self.class_counts),
            "num_classes": len(self.class_counts),
            "exact_duplicate_groups_count": len(exact_duplicate_groups),
            "perceptual_duplicate_groups_count": len(perceptual_duplicate_groups),
            "dimension_stats": self.dimension_stats,
            "sample_records": self.image_records[:5]
        }
        
        return report

    def generate_markdown_report(self, report: Dict[str, Any], output_path: str):
        """Generate human-readable engineering audit markdown."""
        md = []
        md.append("# Saree Dataset Inspection & Integrity Audit Report")
        md.append(f"**Analyzed Directory:** `{report['root_directory']}`\n")
        
        md.append("## 1. Executive Summary")
        md.append(f"- **Total Discovered Files:** {report['total_discovered']}")
        md.append(f"- **Successfully Audited:** {report['total_valid_scanned']}")
        md.append(f"- **Corrupted / Unreadable Files:** {report['corrupted_count']}")
        md.append(f"- **Identified Classes / Partitions:** {report['num_classes']}")
        md.append(f"- **Exact Duplicate Groups (Identical SHA-256):** {report['exact_duplicate_groups_count']}")
        md.append(f"- **Perceptual Duplicate Groups (dHash Structural Clusters):** {report['perceptual_duplicate_groups_count']}\n")

        md.append("## 2. Class Distribution")
        md.append("| Directory / Class Label | Image Count | Proportion |")
        md.append("|---|---|---|")
        total_valid = report['total_valid_scanned']
        for cls_name, count in sorted(report['classes'].items(), key=lambda x: -x[1]):
            prop = (count / total_valid * 100) if total_valid > 0 else 0
            md.append(f"| `{cls_name}` | {count} | {prop:.2f}% |")
        md.append("")

        if report.get('dimension_stats'):
            ds = report['dimension_stats']
            md.append("## 3. Dimensionality & Resolution Profile")
            md.append("| Metric | Min | Median | Mean | Max |")
            md.append("|---|---|---|---|---|")
            md.append(f"| **Width (px)** | {ds['width']['min']} | {ds['width']['median']} | {ds['width']['mean']} | {ds['width']['max']} |")
            md.append(f"| **Height (px)** | {ds['height']['min']} | {ds['height']['median']} | {ds['height']['mean']} | {ds['height']['max']} |")
            md.append(f"| **Aspect Ratio (W/H)** | {ds['aspect_ratio']['min']} | {ds['aspect_ratio']['median']} | {ds['aspect_ratio']['mean']} | {ds['aspect_ratio']['max']} |")
            md.append(f"| **File Size (KB)** | {ds['size_kb']['min']} | - | {ds['size_kb']['mean']} | {ds['size_kb']['max']} |")
            md.append(f"\n- **Sub-Resolution Outliers (< 64x64):** {ds['tiny_images_count']}")
            md.append(f"- **Excessive High-Res Outliers (> 4000x4000):** {ds['oversized_images_count']}\n")

        md.append("## 4. Architectural & Split Strategy Implications")
        md.append("1. **Data Leakage Defense**: Any exact or perceptual duplicates must be purged or grouped into the same split (train vs validation vs gallery). Under no circumstances can duplicate designs straddle train and test.")
        md.append("2. **Colorway Presence Verification**: If the dataset only contains high-level categorical folders (e.g. `handloom_sarees`, `normal_sarees`, `banarasi`), labels represent *categories*, NOT individual *design identities*.")
        md.append("3. **Scientific Gallery/Query Split**: To test genuine color invariance, evaluation must either use verified cross-colorway pairs or synthetic palette-shifted query sets where ground truth design identity is preserved while RGB color distribution is intentionally perturbed.")
        
        content = "\n".join(md)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[DatasetInspector] Audit report saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect and audit saree image dataset.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset root")
    parser.add_argument("--output_report", type=str, default="dataset_inspection_report.md", help="Output report path")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional sample limit")
    args = parser.parse_args()

    inspector = DatasetInspector(root_dir=args.data_dir, sample_size=args.sample_size)
    report = inspector.scan()
    inspector.generate_markdown_report(report, args.output_report)
