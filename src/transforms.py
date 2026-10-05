"""
Augmentation Pipelines and Color-Invariance Transformations.
Phase 4 & Phase 6: Principled textile data augmentation designed to enforce
color invariance while preserving fine geometric weave motifs.
"""

import random
from typing import Tuple, Dict, Any, Callable
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageEnhance

class RandomColorwayShift:
    """
    Simulates genuine textile dye / palette swaps.
    Applies severe hue rotation (0.0 to 1.0) and saturation re-scaling
    while strictly preserving edge gradients and spatial contrast.
    """
    def __init__(self, p: float = 0.8, hue_range: float = 0.5):
        self.p = p
        self.hue_range = hue_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        
        # Severe random hue perturbation in HSV space
        hue_factor = random.uniform(-self.hue_range, self.hue_range)
        sat_factor = random.uniform(0.6, 1.4)
        val_factor = random.uniform(0.8, 1.2)
        
        img = TF.adjust_hue(img, hue_factor)
        img = TF.adjust_saturation(img, sat_factor)
        img = TF.adjust_brightness(img, val_factor)
        return img

class RandomChannelPermutation:
    """
    Permutes RGB color channels randomly (e.g. RGB -> BGR, BRG, GBR).
    Completely breaks any residual color shortcut without altering
    spatial geometry or high-frequency weave textures.
    """
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        channels = list(img.split())
        random.shuffle(channels)
        return Image.merge("RGB", channels)

class RandomSolarization:
    """
    Selectively inverts pixels above a threshold.
    Simulates high-reflectance metallic zari threads under studio lighting.
    """
    def __init__(self, p: float = 0.2, threshold: int = 192):
        self.p = p
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return ImageOps.solarize(img, threshold=self.threshold)

def get_baseline_transforms(img_size: Tuple[int, int] = (256, 256)) -> Dict[str, Callable]:
    """
    Standard ImageNet transforms (No color invariance).
    Used for Phase 3 Baseline to establish the color-bias benchmark.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        T.Resize(img_size),
        T.RandomResizedCrop(img_size[0], scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    eval_transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size[0]),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return {"train": train_transform, "eval": eval_transform}

def get_color_invariant_transforms(img_size: Tuple[int, int] = (256, 256)) -> Dict[str, Callable]:
    """
    Phase 4 & 6: Principled Color-Invariance Augmentation Pipeline.
    
    Design Principles:
      1. Spatial Augmentation (Motif preserving):
         - RandomResizedCrop(scale=(0.7, 1.0)): Simulates varying viewing distances to motifs.
         - RandomHorizontalFlip(p=0.5): Sarees are symmetric along warp/weft.
         - RandomRotation(degrees=10): Tolerates minor draping tilt without distorting borders.
         - NO extreme perspective or heavy elastic warps (preserves geometric motif ratios).
      2. Chromatic Disentanglement (Color destroying):
         - RandomColorwayShift(p=0.8): Hue shifts simulate different dyed colorways.
         - RandomGrayscale(p=0.3): Eliminates chromatic dependence, forces luminance structure.
         - ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.4): Studio lighting invariance.
         - RandomChannelPermutation(p=0.25): Total decorrelation of RGB channels.
         - RandomSolarization(p=0.15): Zari reflectance invariance.
         - GaussianBlur(kernel_size=3, sigma=(0.1, 1.0), p=0.2): Robustness to slight camera blur.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        T.Resize(img_size),
        T.RandomResizedCrop(img_size[0], scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.BILINEAR),
        
        # Color-Invariance Operations
        RandomColorwayShift(p=0.8, hue_range=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.4),
        RandomChannelPermutation(p=0.25),
        RandomSolarization(p=0.15, threshold=200),
        T.RandomGrayscale(p=0.30),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    eval_transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size[0]),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return {"train": train_transform, "eval": eval_transform}

def generate_test_colorway_probe(img: Image.Image, mode: str = "cross_color") -> Image.Image:
    """
    Generates controlled probe queries for Phase 8 Color Invariance Testing:
      - 'same_color': Slight jitter simulating repeat photo in same palette (Test A).
      - 'cross_color': 180-degree inverted hue shift simulating opposite colorway (Test B).
    """
    if mode == "same_color":
        return TF.adjust_brightness(img, random.uniform(0.95, 1.05))
    elif mode == "cross_color":
        # 180 degree hue shift swaps reds <-> cyans, blues <-> yellows
        return TF.adjust_hue(img, 0.5)
    else:
        return img
