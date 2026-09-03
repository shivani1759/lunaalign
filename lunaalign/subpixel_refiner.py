"""
Tier 2 Sub-pixel refinement module.
Refines RANSAC inlier correspondences to fractional-pixel precision
using 2D quadratic parabolic peak interpolation over local patch cross-correlation.
Implements Fix 6:
- If RMSE_after < RMSE_before by at least 0.01 px: Applied — improved registration
- If RMSE_after is within 0.01 px: Applied — no measurable improvement
- If RMSE_after > RMSE_before by > 0.01 px: Rejected — refinement worsened registration (revert to RANSAC)
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


def refine_matches_subpixel(
    image_a: np.ndarray,
    image_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    patch_size: int = 15
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper."""
    refined_a, refined_b, _ = refine_inliers_subpixel(
        image_a=image_a,
        image_b=image_b,
        pts_a=pts_a,
        pts_b=pts_b,
        patch_size=patch_size
    )
    return refined_a, refined_b


def refine_inliers_subpixel(
    image_a: np.ndarray,
    image_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    transform_matrix: Optional[np.ndarray] = None,
    transform_type: str = "homography",
    patch_size: int = 15
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Sub-pixel refinement with Fix 6 evaluation logic.
    """
    from .geometric_verifier import compute_rmse
    
    if len(pts_a) == 0 or len(pts_b) == 0:
        return pts_a, pts_b, {
            "rmse_before_px": None,
            "rmse_after_px": None,
            "effective_rmse_px": None,
            "avg_subpixel_correction_px": 0.0,
            "max_subpixel_correction_px": 0.0,
            "subpixel_status": "not_applicable",
            "status_label": "Not Applicable",
            "conclusion": "No inliers available for sub-pixel refinement.",
            "percentage_improvement": 0.0,
            "refined_points_count": 0
        }
        
    rmse_before = compute_rmse(pts_a, pts_b, transform_matrix, transform_type) if transform_matrix is not None else float('inf')
    
    refined_b = pts_b.copy().astype(np.float32)
    half = patch_size // 2
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]
    
    corrections = []
    
    for i in range(len(pts_a)):
        xa, ya = int(round(pts_a[i, 0])), int(round(pts_a[i, 1]))
        xb, yb = int(round(pts_b[i, 0])), int(round(pts_b[i, 1]))
        
        search_radius = 2
        if (xa - half < 0 or xa + half + 1 >= w_a or ya - half < 0 or ya + half + 1 >= h_a or
            xb - half - search_radius < 0 or xb + half + search_radius + 1 >= w_b or 
            yb - half - search_radius < 0 or yb + half + search_radius + 1 >= h_b):
            corrections.append(0.0)
            continue
            
        patch_a = image_a[ya - half : ya + half + 1, xa - half : xa + half + 1].astype(np.float32)
        search_b = image_b[
            yb - half - search_radius : yb + half + search_radius + 1,
            xb - half - search_radius : xb + half + search_radius + 1
        ].astype(np.float32)
        
        res = cv2.matchTemplate(search_b, patch_a, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        max_x, max_y = max_loc
        
        if 0 < max_x < res.shape[1] - 1 and 0 < max_y < res.shape[0] - 1:
            c = res[max_y, max_x]
            cl = res[max_y, max_x - 1]
            cr = res[max_y, max_x + 1]
            ct = res[max_y - 1, max_x]
            cb = res[max_y + 1, max_x]
            
            denom_x = 2.0 * (2.0 * c - cl - cr) + 1e-7
            denom_y = 2.0 * (2.0 * c - ct - cb) + 1e-7
            
            dx = (cr - cl) / denom_x
            dy = (cb - ct) / denom_y
            
            dx = float(np.clip(dx, -0.5, 0.5))
            dy = float(np.clip(dy, -0.5, 0.5))
            
            offset_x = (max_x - search_radius) + dx
            offset_y = (max_y - search_radius) + dy
            
            refined_b[i, 0] = xb + offset_x
            refined_b[i, 1] = yb + offset_y
            
            disp = float(np.sqrt((offset_x - (max_x - search_radius))**2 + (offset_y - (max_y - search_radius))**2))
            corrections.append(disp if disp > 0 else float(np.sqrt(dx**2 + dy**2)))
        else:
            corrections.append(0.0)
            
    rmse_after = compute_rmse(pts_a, refined_b, transform_matrix, transform_type) if transform_matrix is not None else float('inf')
    
    # Fix 6 Decision Logic:
    diff = rmse_before - rmse_after
    
    if diff > 0.01:
        subpixel_status = "applied_improved_registration"
        status_label = "Applied — improved registration"
        pct_imp = round((diff / rmse_before) * 100.0, 1)
        conclusion = f"Sub-pixel refinement reduced reprojection RMSE by {pct_imp}%."
        effective_b = refined_b
        effective_rmse = rmse_after
    elif abs(diff) <= 0.01:
        subpixel_status = "applied_no_measurable_improvement"
        status_label = "Applied — no measurable improvement"
        pct_imp = 0.0
        conclusion = "Geometric alignment was already stable at pixel-level precision."
        effective_b = refined_b
        effective_rmse = rmse_after
    else:  # rmse_after > rmse_before + 0.01
        subpixel_status = "rejected_worsened_registration"
        status_label = "Rejected — refinement worsened registration"
        pct_imp = round((diff / rmse_before) * 100.0, 1)
        conclusion = "Refinement worsened reprojection error; reverted to original RANSAC integer coordinates."
        effective_b = pts_b.copy()
        effective_rmse = rmse_before
        
    avg_correction = float(np.mean(corrections)) if corrections else 0.0
    max_correction = float(np.max(corrections)) if corrections else 0.0
    
    metrics = {
        "rmse_before_px": round(rmse_before, 2) if rmse_before != float('inf') else None,
        "rmse_after_px": round(rmse_after, 2) if rmse_after != float('inf') else None,
        "effective_rmse_px": round(effective_rmse, 2) if effective_rmse != float('inf') else None,
        "avg_subpixel_correction_px": round(avg_correction, 3),
        "max_subpixel_correction_px": round(max_correction, 3),
        "subpixel_status": subpixel_status,
        "status_label": status_label,
        "percentage_improvement": pct_imp,
        "conclusion": conclusion,
        "refined_points_count": len(pts_a)
    }
    
    return pts_a, effective_b, metrics
