"""
Illumination-aware preprocessing stage:
1. Multi-format ingestion and grayscale conversion.
2. Aspect-ratio-preserving resizing.
3. Selectable illumination normalization (AUTO, CLAHE, GRADIENT, NONE).
4. Preprocessing preview generation.
"""

import os
from typing import Tuple, Optional, Union, Dict, Any
import cv2
import numpy as np
from .config import LunaAlignConfig


def load_image(image_input: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load an image from file path or validate an existing numpy array.
    Supports PNG, JPG, JPEG, TIFF, TIF, BMP and converts to uint8 grayscale.
    """
    if isinstance(image_input, np.ndarray):
        img = image_input
    elif isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        img = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read/decode image: {image_input}")
    else:
        raise TypeError(f"Expected str or np.ndarray, got {type(image_input)}")
        
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, 0]
            
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
        
    return img


def resize_preserving_aspect(
    image: np.ndarray, 
    max_dimension: int
) -> Tuple[np.ndarray, float]:
    """Resize image if its max dimension exceeds max_dimension, preserving aspect ratio."""
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    if max_dim <= max_dimension or max_dimension <= 0:
        return image, 1.0
        
    scale = max_dimension / float(max_dim)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def apply_clahe(
    image: np.ndarray, 
    clip_limit: float = 3.0, 
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def apply_gradient_emphasis(image: np.ndarray) -> np.ndarray:
    """
    Generate normalized gradient magnitude image emphasizing crater rims, slopes, and ridges.
    Uses Scharr operators for isotropic gradient response.
    """
    img_blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0.8)
    gx = cv2.Scharr(img_blurred, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_blurred, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return norm_mag


def apply_denoising(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply light Gaussian blur denoising to suppress lunar camera/sensor grain."""
    if kernel_size <= 1:
        return image
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (k, k), sigmaX=0.8)


def preprocess_image(
    image_input: Union[str, np.ndarray], 
    config: Optional[LunaAlignConfig] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Backward-compatible preprocessing function.
    Returns (raw_gray_image, preprocessed_image, scale_factor).
    """
    raw, proc, _, _, _, scale = preprocess_image_tier2(image_input, config)
    return raw, proc, scale


def preprocess_image_tier2(
    image_input: Union[str, np.ndarray],
    config: Optional[LunaAlignConfig] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Tier 2 Illumination-Aware Preprocessing Pipeline.
    Supports selectable illumination modes:
      - 'AUTO': CLAHE contrast normalization for SIFT/ORB + gradient edge map for illumination consistency.
      - 'CLAHE': CLAHE contrast normalization.
      - 'GRADIENT': Gradient/edge emphasis using isotropic Scharr magnitude.
      - 'NONE': Raw normalized grayscale (no contrast modification).
      
    Args:
        image_input: File path or numpy image array.
        config: LunaAlignConfig instance.
        
    Returns:
        Tuple of:
        - raw_gray: Raw resized grayscale image (before contrast preprocessing).
        - processed_for_matching: Image passed into feature detector.
        - gradient_edge_image: Gradient magnitude map for illumination checks.
        - preview_before: Thumbnail preview before preprocessing.
        - preview_after: Thumbnail preview after preprocessing.
        - scale_factor: Scaling factor applied.
    """
    if config is None:
        config = LunaAlignConfig()
        
    raw_gray_full = load_image(image_input)
    raw_gray, scale = resize_preserving_aspect(raw_gray_full, config.max_dimension)
    
    if config.apply_denoise:
        denoised = apply_denoising(raw_gray, config.denoise_kernel_size)
    else:
        denoised = raw_gray
        
    mode = getattr(config, "illumination_mode", "AUTO").upper()
    gradient_img = apply_gradient_emphasis(denoised)
    
    if mode in ("AUTO", "CLAHE_GRADIENT"):
        clahe_img = apply_clahe(denoised, config.clahe_clip_limit, config.clahe_tile_grid_size)
        processed_for_matching = clahe_img
    elif mode == "CLAHE":
        processed_for_matching = apply_clahe(denoised, config.clahe_clip_limit, config.clahe_tile_grid_size)
    elif mode in ("GRADIENT", "EDGE"):
        processed_for_matching = gradient_img
    else:  # NONE
        processed_for_matching = denoised
        
    # Generate thumbnail previews (max 200px)
    preview_before, _ = resize_preserving_aspect(raw_gray, 200)
    preview_after, _ = resize_preserving_aspect(processed_for_matching, 200)
    
    return raw_gray, processed_for_matching, gradient_img, preview_before, preview_after, scale
