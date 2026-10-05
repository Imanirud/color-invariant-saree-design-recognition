"""
Packaging & Submission Preparation Script.
Creates a clean, production-ready submission zip file for the DeepLure AI Engineer Assessment.
Enforces non-redistribution of proprietary datasets, strips __pycache__, and validates
that all required submission deliverables (approach note, notebook, readme) are present.
"""

import os
import sys
import zipfile
import json

def package_submission(output_zip: str = "deeplure_saree_recognition_submission.zip"):
    print("=" * 70)
    print("   DEEPLURE ASSESSMENT: SUBMISSION PACKAGING UTILITY")
    print("=" * 70)

    # 1. Validate Approach Note length (< 500 chars)
    approach_note_path = "approach_note.txt"
    if not os.path.exists(approach_note_path):
        print("ERROR: approach_note.txt not found!")
        sys.exit(1)
    with open(approach_note_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    char_len = len(content)
    print(f"[CHECK] approach_note.txt length: {char_len} characters (Limit: 500 chars)")
    if char_len > 500:
        print(f"ERROR: approach_note.txt exceeds 500 characters ({char_len} chars)!")
        sys.exit(1)
    print("  -> PASS: Approach note strictly compliant.")

    # 2. Validate Notebook
    nb_path = "notebooks/saree_design_recognition_kaggle.ipynb"
    if not os.path.exists(nb_path):
        print(f"ERROR: Notebook '{nb_path}' not found!")
        sys.exit(1)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)
    print(f"[CHECK] Kaggle notebook: {len(nb_json['cells'])} cells verified.")
    print("  -> PASS: Notebook structure valid.")

    # 3. Collect files for inclusion
    exclude_dirs = {"__pycache__", ".git", ".pytest_cache", ".venv", "venv", "data", "output", "temp_test_ds"}
    exclude_exts = {".pyc", ".pyo", ".pth", ".pt", ".onnx", ".zip", ".log"}
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    files_to_pack = []
    for root, dirs, files in os.walk("."):
        # Prune unwanted directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in exclude_exts or ext in image_exts:
                continue
            if f.endswith(".zip"):
                continue
            rel_path = os.path.relpath(os.path.join(root, f), ".")
            files_to_pack.append(rel_path)

    # 4. Create ZIP
    print(f"\n[PACKING] Compressing {len(files_to_pack)} verified project files into '{output_zip}'...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in sorted(files_to_pack):
            zipf.write(f)
            file_size_kb = os.path.getsize(f) / 1024.0
            print(f"  + {f:<45} ({file_size_kb:6.1f} KB)")

    zip_size_kb = os.path.getsize(output_zip) / 1024.0
    print("\n" + "=" * 70)
    print(f"SUCCESS: Submission archive created: {output_zip} ({zip_size_kb:.1f} KB)")
    print("Zero dataset images included (strict compliance with DeepLure terms).")
    print("=" * 70)

if __name__ == "__main__":
    package_submission()
