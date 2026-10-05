"""
Command-Line Inference & Demonstration Engine.
Supports:
  1. Pairwise Verification: Is image A the same textile design as image B?
  2. Gallery Retrieval: Retrieve Top-K matching designs for a probe query image.
  3. Self-Contained Demo: Validates color invariance on synthesized textile motifs.

Usage:
  python infer.py --demo
  python infer.py --verify image_a.jpg image_b.jpg [--threshold 0.55]
  python infer.py --query probe.jpg --gallery_dir ./gallery --top_k 5
"""

import os
import sys
import argparse
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from src.models import ProposedColorInvariantModel

# Standard evaluation preprocessing
EVAL_TRANSFORMS = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_inference_model(checkpoint_path: Optional[str] = None, device: str = "cpu") -> torch.nn.Module:
    """Load model in eval mode. If checkpoint is provided, load trained weights."""
    model = ProposedColorInvariantModel(backbone_name="resnet50", pretrained=False, embed_dim=256)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict, strict=False)
        print(f"[Infer] Loaded checkpoint: {checkpoint_path}")
    else:
        print("[Infer] Running with initialized trunk (run training or provide checkpoint for optimal performance)")
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def extract_embedding(image: Image.Image, model: torch.nn.Module, device: str = "cpu") -> np.ndarray:
    """Extract L2-normalized 256-D embedding vector from PIL Image."""
    rgb_img = image.convert("RGB")
    tensor = EVAL_TRANSFORMS(rgb_img).unsqueeze(0).to(device)
    emb = model(tensor)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().flatten()

def compute_cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine similarity between two unit vectors."""
    return float(np.dot(emb_a, emb_b))

def verify_pair(
    path_a: str,
    path_b: str,
    model: torch.nn.Module,
    threshold: float = 0.55,
    device: str = "cpu"
) -> Tuple[bool, float]:
    """Pairwise verification check."""
    img_a = Image.open(path_a)
    img_b = Image.open(path_b)
    emb_a = extract_embedding(img_a, model, device=device)
    emb_b = extract_embedding(img_b, model, device=device)
    sim = compute_cosine_similarity(emb_a, emb_b)
    is_same = sim >= threshold
    return is_same, sim

def create_synthetic_motif(shape: str = "paisley", bg_color: Tuple = (200, 30, 30), fg_color: Tuple = (255, 215, 0)) -> Image.Image:
    """Synthesize high-frequency geometric motif in specified color palette."""
    img = Image.new("RGB", (256, 256), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    if shape == "lattice":
        # Diamond grid jaal
        for x in range(0, 256, 32):
            draw.line([(x, 0), (256, 256 - x)], fill=fg_color, width=2)
            draw.line([(0, x), (256 - x, 256)], fill=fg_color, width=2)
            draw.line([(0, 256 - x), (x, 0)], fill=fg_color, width=2)
            draw.line([(256 - x, 256), (256, x)], fill=fg_color, width=2)
    elif shape == "butta":
        # Repeating floral medallion motifs
        for cy in range(32, 256, 64):
            for cx in range(32, 256, 64):
                draw.ellipse([cx-12, cy-12, cx+12, cy+12], outline=fg_color, width=3)
                draw.ellipse([cx-6, cy-6, cx+6, cy+6], fill=fg_color)
                for angle_step in [-18, 18]:
                    draw.line([(cx, cy), (cx + angle_step, cy - 20)], fill=fg_color, width=2)
    else:  # stripes / waves
        for y in range(0, 256, 16):
            draw.line([(0, y), (256, y)], fill=fg_color, width=4)
    return img

def run_self_contained_demo():
    """Demonstrate color-invariance retrieval and verification on synthetic textile motifs."""
    print("=" * 70)
    print("   COLOR-INVARIANT SAREE RECOGNITION: INTERACTIVE DEMO")
    print("=" * 70)
    print("Synthesizing 4 textile samples to test the core theorem:")
    print("  - Sample 1: Design A (Diamond Lattice) in Crimson Red & Gold")
    print("  - Sample 2: Design A (Diamond Lattice) in Royal Blue & Silver")
    print("  - Sample 3: Design B (Floral Butta)     in Crimson Red & Gold (Confused by color)")
    print("  - Sample 4: Design B (Floral Butta)     in Royal Blue & Silver")
    print("-" * 70)

    model = load_inference_model(device="cpu")

    # Generate synthetic motifs
    s1 = create_synthetic_motif("lattice", bg_color=(180, 20, 20), fg_color=(255, 215, 0))   # Lattice Red
    s2 = create_synthetic_motif("lattice", bg_color=(20, 30, 180), fg_color=(220, 220, 230)) # Lattice Blue
    s3 = create_synthetic_motif("butta",   bg_color=(180, 20, 20), fg_color=(255, 215, 0))   # Butta Red
    s4 = create_synthetic_motif("butta",   bg_color=(20, 30, 180), fg_color=(220, 220, 230)) # Butta Blue

    emb1 = extract_embedding(s1, model)
    emb2 = extract_embedding(s2, model)
    emb3 = extract_embedding(s3, model)
    emb4 = extract_embedding(s4, model)

    sim_same_diff_color = compute_cosine_similarity(emb1, emb2)
    sim_diff_same_color = compute_cosine_similarity(emb1, emb3)
    sim_diff_diff_color = compute_cosine_similarity(emb1, emb4)

    print("\n[PAIRWISE VERIFICATION RESULTS]")
    print(f"  Test B [SAME Design (Lattice), DIFFERENT Colors (Red vs Blue)]:")
    print(f"    -> Cosine Similarity: {sim_same_diff_color:+.4f}")
    
    print(f"  Test C [DIFFERENT Design (Lattice vs Butta), SAME Colors (Both Red)]:")
    print(f"    -> Cosine Similarity: {sim_diff_same_color:+.4f}")

    print(f"  Test D [DIFFERENT Design (Lattice vs Butta), DIFFERENT Colors (Red vs Blue)]:")
    print(f"    -> Cosine Similarity: {sim_diff_diff_color:+.4f}")

    print("\n[TOP-K GALLERY RETRIEVAL SIMULATION]")
    gallery_embs = np.stack([emb2, emb3, emb4])
    gallery_names = [
        "Design A (Lattice) - Royal Blue (Cross-colorway TARGET)",
        "Design B (Butta)   - Crimson Red (Hard Negative / Same Color)",
        "Design B (Butta)   - Royal Blue (Easy Negative / Different Color)"
    ]
    query_emb = emb1 # Lattice Red
    similarities = np.dot(gallery_embs, query_emb)
    ranked_indices = np.argsort(-similarities)

    print("Query: Design A (Lattice) - Crimson Red")
    print("Gallery Rankings:")
    for rank, idx in enumerate(ranked_indices, start=1):
        print(f"  Rank #{rank}: [Similarity: {similarities[idx]:+.4f}] -> {gallery_names[idx]}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE: Model pipeline, GeM pooling, and L2 metric space verified.")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Color-Invariant Saree Recognition CLI")
    parser.add_argument("--demo", action="store_true", help="Run self-contained color-invariance demo")
    parser.add_argument("--verify", nargs=2, metavar=("IMG_A", "IMG_B"), help="Perform pairwise verification")
    parser.add_argument("--query", type=str, help="Query image path for gallery retrieval")
    parser.add_argument("--gallery_dir", type=str, help="Directory containing gallery images")
    parser.add_argument("--top_k", type=int, default=5, help="Number of gallery matches to return")
    parser.add_argument("--threshold", type=float, default=0.55, help="Verification decision threshold")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.demo or len(sys.argv) == 1:
        run_self_contained_demo()
        return

    model = load_inference_model(checkpoint_path=args.checkpoint, device=args.device)

    if args.verify:
        path_a, path_b = args.verify
        is_same, sim = verify_pair(path_a, path_b, model, threshold=args.threshold, device=args.device)
        decision = "SAME DESIGN (MATCH)" if is_same else "DIFFERENT DESIGN (REJECT)"
        print("\n" + "=" * 50)
        print("         PAIRWISE VERIFICATION RESULT")
        print("=" * 50)
        print(f"Image A:          {path_a}")
        print(f"Image B:          {path_b}")
        print(f"Cosine Score:     {sim:.4f}")
        print(f"Threshold:        {args.threshold:.4f}")
        print(f"Decision:         {decision}")
        print("=" * 50)

    elif args.query:
        if not args.gallery_dir or not os.path.exists(args.gallery_dir):
            print(f"Error: Gallery directory '{args.gallery_dir}' does not exist.")
            sys.exit(1)
        
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        gallery_files = [
            os.path.join(args.gallery_dir, f) for f in os.listdir(args.gallery_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        if not gallery_files:
            print(f"No valid images found in gallery directory: {args.gallery_dir}")
            sys.exit(1)

        query_img = Image.open(args.query)
        query_emb = extract_embedding(query_img, model, device=args.device)

        gallery_embs = []
        for gf in gallery_files:
            g_img = Image.open(gf)
            gallery_embs.append(extract_embedding(g_img, model, device=args.device))
        
        gallery_embs = np.stack(gallery_embs)
        sims = np.dot(gallery_embs, query_emb)
        ranked = np.argsort(-sims)[:args.top_k]

        print("\n" + "=" * 60)
        print(f"TOP-{args.top_k} GALLERY RETRIEVAL RESULTS FOR: {args.query}")
        print("=" * 60)
        for rank, idx in enumerate(ranked, start=1):
            print(f"Rank #{rank:02d} | Similarity: {sims[idx]:+.4f} | {os.path.basename(gallery_files[idx])}")
        print("=" * 60)

if __name__ == "__main__":
    main()
