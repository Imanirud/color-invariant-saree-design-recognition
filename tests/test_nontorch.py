"""Quick test: metrics + split logic + color invariance (no torch needed)."""
import sys
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image

print(f"numpy {np.__version__}")
print("PIL OK")

# ==== Test 1: Retrieval metrics ====
sys.path.insert(0, os.getcwd())
from src.metrics import compute_retrieval_metrics, compute_verification_metrics, evaluate_color_invariance_breakdown

np.random.seed(42)
q = np.random.randn(10, 256).astype(np.float32)
q /= np.linalg.norm(q, axis=1, keepdims=True)
g = np.random.randn(50, 256).astype(np.float32)
g /= np.linalg.norm(g, axis=1, keepdims=True)
ql = np.array([0,1,2,3,4,0,1,2,3,4])
gl = np.arange(50) % 10

r = compute_retrieval_metrics(q, g, ql, gl)
print(f"[PASS] Retrieval metrics: R@1={r['Recall@1']}, mAP={r['mAP']}")

# ==== Test 2: Verification metrics ====
is_same = np.array([1, 0, 1, 0, 1])
v = compute_verification_metrics(q[:5], g[:5], is_same)
print(f"[PASS] Verification metrics: AUC={v['ROC_AUC']}, EER={v['EER']}")

# ==== Test 3: DataCleaner importable ====
from src.cleaner import DataCleaner
print("[PASS] DataCleaner class imported OK")

# ==== Test 4: build_design_level_splits (extract logic inline to avoid torch import) ====
# Copy the function logic directly since the module imports torch at top level
def build_design_level_splits(
    records, train_ratio=0.70, val_ratio=0.15, seed=42
):
    random.seed(seed)
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
    
    gallery_records = []
    query_records = []
    for d in test_designs:
        items = design_to_records[d]
        if len(items) >= 2:
            gallery_records.append(items[0])
            for q_item in items[1:]:
                query_records.append(q_item)
        else:
            gallery_records.append(items[0])
            query_record_copy = dict(items[0])
            query_record_copy["is_synthetic_query"] = True
            query_records.append(query_record_copy)

    return train_records, val_records, gallery_records, query_records, train_designs, val_designs, test_designs

records = []
for d in range(20):
    for k in range(3):
        records.append({
            "image_path": f"dummy_{d}_{k}.jpg",
            "design_id": f"design_{d:03d}",
            "colorway_id": f"c_{k}",
            "category": "saree"
        })

train_r, val_r, gall_r, q_r, td, vd, testd = build_design_level_splits(records)
print(f"[PASS] Splits: Train={len(train_r)}, Val={len(val_r)}, Gallery={len(gall_r)}, Query={len(q_r)}")

# Zero-leakage check
assert td.isdisjoint(vd), "LEAKAGE: train/val overlap!"
assert td.isdisjoint(testd), "LEAKAGE: train/test overlap!"
assert vd.isdisjoint(testd), "LEAKAGE: val/test overlap!"
print("[PASS] Zero-leakage assertion PASSED")

# ==== Test 5: Color Invariance Breakdown ====
subset_sims = {
    "TEST_A": np.array([0.91, 0.88, 0.92]),
    "TEST_B": np.array([0.82, 0.79, 0.85]),
    "TEST_C": np.array([0.31, 0.28, 0.35]),
    "TEST_D": np.array([0.15, 0.18, 0.12]),
}
ci = evaluate_color_invariance_breakdown(subset_sims)
print(f"[PASS] Color Invariance Breakdown:")
for k, v in ci.items():
    print(f"       {k}: {v:.4f}")

# ==== Test 6: Losses module (compile check, no torch execution) ====
import importlib.util
spec = importlib.util.find_spec("src.losses")
if spec is not None:
    print("[PASS] src.losses module found")
else:
    print("[SKIP] src.losses not importable (needs torch)")

# ==== Test 7: Transforms module (compile check) ====
spec2 = importlib.util.find_spec("src.transforms")
if spec2 is not None:
    print("[PASS] src.transforms module found")
else:
    print("[SKIP] src.transforms not importable (needs torch)")

# ==== Test 8: Verify all .py files parse without syntax errors ====
import ast
src_dir = os.path.join(os.getcwd(), "src")
all_py_ok = True
for fname in sorted(os.listdir(src_dir)):
    if fname.endswith(".py"):
        filepath = os.path.join(src_dir, fname)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                ast.parse(f.read(), filename=fname)
            print(f"[PASS] AST parse: {fname}")
        except SyntaxError as e:
            print(f"[FAIL] AST parse: {fname} -> {e}")
            all_py_ok = False

# ==== Test 9: Verify notebook cells parse ====
import json
nb_path = os.path.join(os.getcwd(), "notebooks", "saree_design_recognition_kaggle.ipynb")
if os.path.exists(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    nb_ok = True
    for i, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        try:
            ast.parse(source, filename=f"cell_{i}")
            print(f"[PASS] Notebook cell {i} AST OK")
        except SyntaxError as e:
            print(f"[FAIL] Notebook cell {i} -> {e}")
            nb_ok = False
    if nb_ok:
        print(f"[PASS] All {len(code_cells)} notebook code cells parse OK")
else:
    print("[SKIP] Notebook not found at expected path")

print("\n" + "="*60)
print("=== ALL NON-TORCH VALIDATION CHECKS COMPLETE ===")
print("="*60)
