"""
Geometric verification stage using RANSAC.
Estimates transformation (homography or affine), classifies inliers vs outliers,
and calculates reprojection RMSE.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
from .config import LunaAlignConfig


def compute_rmse(
    pts_a_inliers: np.ndarray,
    pts_b_inliers: np.ndarray,
    transform_matrix: np.ndarray,
    transform_type: str = "homography"
) -> float:
    """
    Compute Root Mean Square Error (RMSE) in pixels between points in Image A
    and projected points from Image B using the estimated transformation matrix.
    
    Formula: RMSE = sqrt( 1/N * sum(||pts_a - transform(pts_b)||^2) )
    
    Args:
        pts_a_inliers: (N, 2) inlier coordinates in Image A.
        pts_b_inliers: (N, 2) inlier coordinates in Image B.
        transform_matrix: 3x3 Homography or 2x3 Affine matrix.
        transform_type: 'homography' or 'affine'.
        
    Returns:
        Reprojection RMSE in pixels. Returns float('inf') if invalid.
    """
    if len(pts_a_inliers) == 0 or transform_matrix is None:
        return float('inf')
        
    N = len(pts_a_inliers)
    
    if transform_type.lower() == "homography" and transform_matrix.shape == (3, 3):
        pts_b_reshaped = pts_b_inliers.reshape(-1, 1, 2).astype(np.float32)
        pts_b_proj = cv2.perspectiveTransform(pts_b_reshaped, transform_matrix).reshape(-1, 2)
    elif transform_matrix.shape == (2, 3):
        ones = np.ones((N, 1), dtype=np.float32)
        pts_b_homo = np.hstack([pts_b_inliers, ones])  # (N, 3)
        pts_b_proj = (transform_matrix @ pts_b_homo.T).T  # (N, 2)
    elif transform_matrix.shape == (3, 3):
        pts_b_reshaped = pts_b_inliers.reshape(-1, 1, 2).astype(np.float32)
        pts_b_proj = cv2.perspectiveTransform(pts_b_reshaped, transform_matrix).reshape(-1, 2)
    else:
        return float('inf')
        
    errors = np.linalg.norm(pts_a_inliers - pts_b_proj, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return round(rmse, 2)


def verify_geometry_ransac(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    config: Optional[LunaAlignConfig] = None
) -> Tuple[int, float, float, Optional[np.ndarray], np.ndarray, str]:
    """
    Perform RANSAC geometric verification.
    Estimates transformation mapping Image B into Image A coordinate frame.
    
    Args:
        pts_a: (N, 2) Candidate matched point coordinates in Image A.
        pts_b: (N, 2) Candidate matched point coordinates in Image B.
        config: LunaAlignConfig instance.
        
    Returns:
        Tuple of:
        - inliers_count: Number of RANSAC inliers.
        - inlier_ratio: Ratio of inliers / candidate matches.
        - rmse_px: Reprojection RMSE over inliers in pixels.
        - transform_matrix: Estimated 3x3 Homography or 2x3 Affine matrix (or None).
        - inlier_mask: 1D boolean numpy array indicating inliers (True for inlier).
        - transform_type: 'homography' or 'affine'.
    """
    if config is None:
        config = LunaAlignConfig()
        
    candidate_count = len(pts_a)
    inlier_mask = np.zeros(candidate_count, dtype=bool)
    
    min_points = 4 if config.transform_type.lower() == "homography" else 3
    
    if candidate_count < min_points:
        return 0, 0.0, float('inf'), None, inlier_mask, config.transform_type
        
    transform_matrix = None
    mask_arr = None
    actual_transform_type = config.transform_type.lower()
    
    if actual_transform_type == "homography":
        matrix, raw_mask = cv2.findHomography(
            pts_b,
            pts_a,
            method=cv2.RANSAC,
            ransacReprojThreshold=config.ransac_reproj_threshold,
            maxIters=config.ransac_max_iters,
            confidence=config.ransac_confidence
        )
        if matrix is not None and raw_mask is not None:
            transform_matrix = matrix
            mask_arr = raw_mask.ravel().astype(bool)
        else:
            matrix, raw_mask = cv2.estimateAffinePartial2D(
                pts_b,
                pts_a,
                method=cv2.RANSAC,
                ransacReprojThreshold=config.ransac_reproj_threshold,
                maxIters=config.ransac_max_iters,
                confidence=config.ransac_confidence
            )
            if matrix is not None and raw_mask is not None:
                transform_matrix = matrix
                mask_arr = raw_mask.ravel().astype(bool)
                actual_transform_type = "affine"
    else:
        matrix, raw_mask = cv2.estimateAffinePartial2D(
            pts_b,
            pts_a,
            method=cv2.RANSAC,
            ransacReprojThreshold=config.ransac_reproj_threshold,
            maxIters=config.ransac_max_iters,
            confidence=config.ransac_confidence
        )
        if matrix is not None and raw_mask is not None:
            transform_matrix = matrix
            mask_arr = raw_mask.ravel().astype(bool)
            actual_transform_type = "affine"
            
    if mask_arr is not None and np.any(mask_arr):
        inlier_mask = mask_arr
        inliers_count = int(np.sum(inlier_mask))
        inlier_ratio = round(inliers_count / float(candidate_count), 4)
        
        pts_a_inliers = pts_a[inlier_mask]
        pts_b_inliers = pts_b[inlier_mask]
        
        rmse_px = compute_rmse(pts_a_inliers, pts_b_inliers, transform_matrix, actual_transform_type)
    else:
        inliers_count = 0
        inlier_ratio = 0.0
        rmse_px = float('inf')
        transform_matrix = None
        
    return inliers_count, inlier_ratio, rmse_px, transform_matrix, inlier_mask, actual_transform_type
