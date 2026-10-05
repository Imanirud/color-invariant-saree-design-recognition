"""
ONNX Export and Production Deployment Validator.
Phase 10: Exports PyTorch model to ONNX runtime format with dynamic batching
and verifies numerical fidelity on unit hypersphere.
"""

import os
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np

from src.models import ProposedColorInvariantModel

def export_to_onnx(
    model: nn.Module,
    output_path: str = "./output/saree_design_model.onnx",
    input_resolution: Tuple[int, int] = (256, 256),
    embed_dim: int = 256
) -> Dict[str, Any]:
    """
    Exports PyTorch model to ONNX format with dynamic batch sizing.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    dummy_input = torch.randn(1, 3, input_resolution[0], input_resolution[1], requires_grad=False)

    dynamic_axes = {
        "input_image": {0: "batch_size"},
        "embedding": {0: "batch_size"}
    }

    print(f"[ONNX Exporter] Exporting model to: {output_path}")
    try:
        # Prefer legacy TorchScript exporter for universal backward compatibility
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["embedding"],
            dynamic_axes=dynamic_axes
        )
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[ONNX Exporter] Successfully exported ONNX model ({file_size_mb:.2f} MB)")
    except Exception as e:
        print(f"[ONNX Exporter] ONNX export skipped (requires onnxscript / onnx package): {e}")
        return {
            "onnx_path": output_path,
            "size_mb": 0.0,
            "parity_verified": False,
            "max_diff": 0.0,
            "note": str(e)
        }

    # Validate numerical parity if onnxruntime is available
    parity_verified = False
    max_abs_diff = 0.0
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        
        with torch.no_grad():
            pytorch_out = model(dummy_input).numpy()
        
        onnx_inputs = {"input_image": dummy_input.numpy()}
        onnx_out = session.run(["embedding"], onnx_inputs)[0]

        max_abs_diff = float(np.max(np.abs(pytorch_out - onnx_out)))
        parity_verified = max_abs_diff < 1e-4
        print(f"[ONNX Exporter] Numerical Parity Check: Max Diff = {max_abs_diff:.6e} | Verified: {parity_verified}")
    except ImportError:
        print("[ONNX Exporter] onnxruntime not installed; skipped runtime verification.")

    return {
        "onnx_path": output_path,
        "size_mb": round(file_size_mb, 2),
        "parity_verified": parity_verified,
        "max_diff": max_abs_diff
    }

if __name__ == "__main__":
    model = ProposedColorInvariantModel(backbone_name="resnet50", pretrained=False, embed_dim=256)
    export_to_onnx(model)
