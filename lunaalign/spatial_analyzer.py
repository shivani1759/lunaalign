"""
Spatial distribution analysis stage.
Partitions the reference image (Image A) into a grid (e.g. 4x4 = 16 cells)
and computes cell occupancy to reject false positives from localized feature clusters.
"""

from typing import Tuple, Dict
import numpy as np


def check_spatial_distribution(
    pts_a_inliers: np.ndarray,
    image_shape: Tuple[int, int],
    grid_rows: int = 4,
    grid_cols: int = 4
) -> Tuple[int, int, float, Dict[str, int]]:
    """
    Divide Image A into a regular grid and count how many cells contain at least
    one inlier match point.
    
    Args:
        pts_a_inliers: (N, 2) inlier point coordinates (x, y) in Image A.
        image_shape: (height, width) of Image A.
        grid_rows: Number of grid rows (default 4).
        grid_cols: Number of grid columns (default 4).
        
    Returns:
        Tuple of:
        - occupied_cells: Number of grid cells containing >= 1 inlier point.
        - total_cells: Total cells in the grid (rows * cols).
        - coverage_ratio: occupied_cells / total_cells.
        - cell_counts: Dictionary mapping "row_col" -> count of inliers in that cell.
    """
    total_cells = grid_rows * grid_cols
    
    if len(pts_a_inliers) == 0:
        return 0, total_cells, 0.0, {}
        
    h, w = image_shape[:2]
    cell_h = max(1.0, h / float(grid_rows))
    cell_w = max(1.0, w / float(grid_cols))
    
    cell_counts: Dict[str, int] = {}
    occupied_set = set()
    
    for pt in pts_a_inliers:
        x, y = float(pt[0]), float(pt[1])
        col_idx = min(grid_cols - 1, max(0, int(x // cell_w)))
        row_idx = min(grid_rows - 1, max(0, int(y // cell_h)))
        
        cell_key = f"{row_idx}_{col_idx}"
        cell_counts[cell_key] = cell_counts.get(cell_key, 0) + 1
        occupied_set.add((row_idx, col_idx))
        
    occupied_cells = len(occupied_set)
    coverage_ratio = round(occupied_cells / float(total_cells), 4)
    
    return occupied_cells, total_cells, coverage_ratio, cell_counts
