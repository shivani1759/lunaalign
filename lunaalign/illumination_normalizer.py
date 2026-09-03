"""
Illumination-invariant preprocessing & structural consistency module.
Computes:
1. Directional Phase Congruency features.
2. Illumination structural consistency gate (Normalized Gradient Correlation across inliers).
"""

from typing import Optional
import cv2
import numpy as np


def compute_phase_congruency_features(
    image: np.ndarray,
    num_orientations: int = 4
) -> np.ndarray:
    """
    Compute shadow-invariant local energy features across multiple orientations.
    Lunar crater boundaries remain stable in frequency phase even when shadows invert.
    """
    img_f = image.astype(np.float32) / 255.0
    energy_sum = np.zeros_like(img_f)
    
    blurred = cv2.GaussianBlur(img_f, (5, 5), sigmaX=1.2)
    
    for angle_deg in np.linspace(0, 180, num_orientations, endpoint=False):
        theta = np.deg2rad(angle_deg)
        kx = np.cos(theta)
        ky = np.sin(theta)
        
        dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        grad_dir = np.abs(dx * kx + dy * ky)
        energy_sum += grad_dir ** 2
        
    energy_map = np.sqrt(energy_sum)
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    norm_energy = cv2.normalize(energy_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enhanced = clahe.apply(norm_energy)
    
    orig_clahe = clahe.apply(image)
    blended = cv2.addWeighted(orig_clahe, 0.55, enhanced, 0.45, 0)
    return blended


def compute_illumination_consistency(
    image_a: np.ndarray,
    image_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    patch_size: int = 15
) -> Optional[float]:
    """
    Fix 5: Compute illumination-consistency metric comparing local terrain structure
    around RANSAC inlier matches using gradient magnitude cross-correlation.
    
    1. Uses CLAHE-normalized grayscale images.
    2. Computes Scharr/Sobel gradient magnitude images.
    3. Extracts small local patches around each inlier correspondence.
    4. Calculates Normalized Cross-Correlation (NCC) / Pearson correlation.
    5. Averages valid patch scores.
    
    Returns:
        Float similarity score in [0.0, 1.0], or None if unavailable/insufficient inliers.
    """
    if len(pts_a) < 3 or len(pts_b) < 3:
        return None
        
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_a_norm = clahe.apply(image_a) if image_a.ndim == 2 else clahe.apply(cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY))
    img_b_norm = clahe.apply(image_b) if image_b.ndim == 2 else clahe.apply(cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY))
    
    # Compute isotropic gradient magnitude maps
    gx_a = cv2.Scharr(img_a_norm, cv2.CV_32F, 1, 0)
    gy_a = cv2.Scharr(img_a_norm, cv2.CV_32F, 0, 1)
    mag_a = cv2.magnitude(gx_a, gy_a)
    
    gx_b = cv2.Scharr(img_b_norm, cv2.CV_32F, 1, 0)
    gy_b = cv2.Scharr(img_b_norm, cv2.CV_32F, 0, 1)
    mag_b = cv2.magnitude(gx_b, gy_b)
    
    half = patch_size // 2
    h_a, w_a = mag_a.shape[:2]
    h_b, w_b = mag_b.shape[:2]
    
    correlations = []
    
    for i in range(len(pts_a)):
        xa, ya = int(round(pts_a[i, 0])), int(round(pts_a[i, 1]))
        xb, yb = int(round(pts_b[i, 0])), int(round(pts_b[i, 1]))
        
        if (xa - half < 0 or xa + half + 1 >= w_a or ya - half < 0 or ya + half + 1 >= h_a or
            xb - half < 0 or xb + half + 1 >= w_b or yb - half < 0 or yb + half + 1 >= h_b):
            continue
            
        p_a = mag_a[ya - half : ya + half + 1, xa - half : xa + half + 1].flatten()
        p_b = mag_b[yb - half : yb + half + 1, xb - half : xb + half + 1].flatten()
        
        std_a = np.std(p_a)
        std_b = np.std(p_b)
        
        if std_a < 1e-4 or std_b < 1e-4:
            continue
            
        r = float(np.corrcoef(p_a, p_b)[0, 1])
        if not np.isnan(r):
            correlations.append(max(0.0, r))
            
    if not correlations:
        return None
        
    avg_score = float(np.mean(correlations))
    return round(float(np.clip(avg_score, 0.0, 1.0)), 2)
