"""
Configuration dataclass and default parameters for LunaAlign.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class LunaAlignConfig:
    """Configuration parameters for LunaAlign registration pipeline."""
    
    # Preprocessing & Illumination Handling
    max_dimension: int = 1200
    clahe_clip_limit: float = 3.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    apply_denoise: bool = True
    denoise_kernel_size: int = 3
    illumination_mode: str = "AUTO"  # 'AUTO', 'CLAHE', 'GRADIENT', 'NONE'
    min_illumination_consistency: float = 0.40  # Threshold for gradient similarity gate
    
    # Feature Extraction & Matching
    feature_type: str = "AUTO"  # 'SIFT', 'ORB', or 'AUTO' (prefers SIFT)
    sift_nfeatures: int = 5000
    orb_nfeatures: int = 5000
    ratio_threshold: float = 0.75  # Lowe's ratio test threshold
    matcher_type: str = "FLANN"   # 'FLANN' or 'BF'
    min_candidate_matches: int = 30  # Candidate match gate threshold
    
    # Geometric Verification (RANSAC)
    ransac_reproj_threshold: float = 3.5  # Reprojection threshold in pixels
    ransac_confidence: float = 0.995
    ransac_max_iters: int = 2000
    transform_type: str = "homography"  # 'homography' or 'affine'
    
    # Tier 2: Sub-Pixel Refinement
    enable_subpixel: bool = True
    subpixel_patch_size: int = 15
    
    # Spatial Distribution Check
    grid_rows: int = 4
    grid_cols: int = 4
    
    # Decision Gate Thresholds
    min_inliers: int = 20
    min_inlier_ratio: float = 0.25
    max_rmse_px: float = 3.5  # Reprojection error maximum threshold
    min_spatial_coverage_cells: int = 4  # out of 16 (for 4x4 grid)
    
    # Visualizations
    draw_rejected_matches: bool = True
    output_dir: str = "results"
