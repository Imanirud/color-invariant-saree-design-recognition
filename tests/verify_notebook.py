"""
Validate all code cells in saree_design_recognition_kaggle.ipynb for Python syntax.
"""

import json
import ast
import sys

def main():
    nb_path = "notebooks/saree_design_recognition_kaggle.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"Total notebook cells: {len(nb['cells'])} ({len(code_cells)} code cells)")

    all_code = []
    has_error = False

    for idx, cell in enumerate(code_cells, 1):
        source = "".join(cell["source"])
        try:
            ast.parse(source)
            print(f"  [PASS] Code Cell {idx:02d} AST syntax verified.")
            all_code.append(source)
        except SyntaxError as e:
            print(f"  [FAIL] Code Cell {idx:02d} SyntaxError: {e}")
            has_error = True

    if has_error:
        print("\n[RESULT] Notebook has syntax errors!")
        sys.exit(1)

    # Check concatenated script
    full_script = "\n\n".join(all_code)
    try:
        ast.parse(full_script)
        print("\n[RESULT] Complete Notebook concatenated script passed AST validation cleanly.")
    except SyntaxError as e:
        print(f"\n[RESULT] Concatenated script error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
