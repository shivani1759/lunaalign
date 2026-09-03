"""
Visualization and reporting stage.
Generates inlier match side-by-side plot, warped registered image overlay,
formatted terminal card report, and machine-readable JSON.
"""

import os
import json
from typing import Dict, Any, List, Optional
import cv2
import numpy as np


def generate_match_visualization(
    img_a: np.ndarray,
    img_b: np.ndarray,
    kps_a: List[cv2.KeyPoint],
    kps_b: List[cv2.KeyPoint],
    candidate_matches: List[cv2.DMatch],
    inlier_mask: np.ndarray,
    draw_rejected: bool = False
) -> np.ndarray:
    """
    Generate side-by-side match visualization image.
    - Image A (left) and Image B (right).
    - RANSAC inliers drawn in bright green.
    - Rejected matches optionally drawn in subtle red.
    """
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    
    canvas_h = max(h_a, h_b)
    canvas_w = w_a + w_b
    
    color_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR) if img_a.ndim == 2 else img_a
    color_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR) if img_b.ndim == 2 else img_b
    
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h_a, :w_a] = color_a
    canvas[:h_b, w_a:w_a + w_b] = color_b
    
    # Draw rejected matches if enabled
    if draw_rejected and len(candidate_matches) > 0 and len(inlier_mask) == len(candidate_matches):
        for idx, match in enumerate(candidate_matches):
            if not inlier_mask[idx]:
                pt_a = (int(round(kps_a[match.queryIdx].pt[0])), int(round(kps_a[match.queryIdx].pt[1])))
                pt_b = (int(round(kps_b[match.trainIdx].pt[0])) + w_a, int(round(kps_b[match.trainIdx].pt[1])))
                cv2.line(canvas, pt_a, pt_b, (40, 40, 200), 1, cv2.LINE_AA)
                
    # Draw inlier matches in green
    inlier_count = 0
    if len(candidate_matches) > 0 and len(inlier_mask) == len(candidate_matches):
        for idx, match in enumerate(candidate_matches):
            if inlier_mask[idx]:
                inlier_count += 1
                pt_a = (int(round(kps_a[match.queryIdx].pt[0])), int(round(kps_a[match.queryIdx].pt[1])))
                pt_b = (int(round(kps_b[match.trainIdx].pt[0])) + w_a, int(round(kps_b[match.trainIdx].pt[1])))
                
                cv2.circle(canvas, pt_a, 4, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(canvas, pt_b, 4, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.line(canvas, pt_a, pt_b, (0, 230, 0), 2, cv2.LINE_AA)
                
    cv2.putText(canvas, "Image A (Reference)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Image B (Target)", (w_a + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Inlier Matches: {inlier_count}", (canvas_w - 260, canvas_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    return canvas


def generate_registered_image(
    img_a: np.ndarray,
    img_b: np.ndarray,
    transform_matrix: Optional[np.ndarray],
    transform_type: str = "homography"
) -> np.ndarray:
    """
    Warp Image B into Image A's coordinate frame using the estimated transformation matrix.
    """
    h_a, w_a = img_a.shape[:2]
    
    if transform_matrix is None:
        return cv2.resize(img_b, (w_a, h_a))
        
    if transform_type.lower() == "homography" and transform_matrix.shape == (3, 3):
        warped_b = cv2.warpPerspective(img_b, transform_matrix, (w_a, h_a), flags=cv2.INTER_LINEAR)
    elif transform_matrix.shape == (2, 3):
        warped_b = cv2.warpAffine(img_b, transform_matrix, (w_a, h_a), flags=cv2.INTER_LINEAR)
    elif transform_matrix.shape == (3, 3):
        warped_b = cv2.warpPerspective(img_b, transform_matrix, (w_a, h_a), flags=cv2.INTER_LINEAR)
    else:
        warped_b = cv2.resize(img_b, (w_a, h_a))
        
    return warped_b


def format_terminal_card(report: Dict[str, Any]) -> str:
    """
    Produce the clean terminal report card matching LunaAlign specification.
    """
    lines = []
    lines.append("LunaAlign — Baseline Registration Report\n")
    lines.append(f"Decision: {report['decision']}")
    lines.append(f"Confidence: {report['confidence'].capitalize()}\n")
    lines.append(f"Image A keypoints: {report['image_a_keypoints']:,}")
    lines.append(f"Image B keypoints: {report['image_b_keypoints']:,}")
    lines.append(f"Candidate matches: {report['candidate_matches']:,}")
    lines.append(f"RANSAC inliers: {report['inliers']:,}")
    lines.append(f"Inlier ratio: {report['inlier_ratio'] * 100:.1f}%")
    rmse_str = f"{report['rmse_px']:.2f} px" if report.get('rmse_px') is not None else "N/A"
    lines.append(f"Reprojection RMSE: {rmse_str}")
    lines.append(f"Spatial coverage: {report['spatial_coverage_cells']} / {report['grid_total_cells']} grid cells\n")
    lines.append(f"Transformation: {report['transform_type'].capitalize()}")
    lines.append(f"Registration quality: {report.get('quality', 'Inconclusive')}")
    
    return "\n".join(lines)


def save_visualizations(
    output_dir: str,
    img_a: np.ndarray,
    img_b: np.ndarray,
    kps_a: List[cv2.KeyPoint],
    kps_b: List[cv2.KeyPoint],
    candidate_matches: List[cv2.DMatch],
    inlier_mask: np.ndarray,
    transform_matrix: Optional[np.ndarray],
    transform_type: str,
    report: Dict[str, Any],
    draw_rejected: bool = True
) -> Dict[str, str]:
    """
    Save all visualization artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}
    
    # 1. inlier_matches.png
    match_vis = generate_match_visualization(
        img_a, img_b, kps_a, kps_b, candidate_matches, inlier_mask, draw_rejected
    )
    matches_path = os.path.join(output_dir, "inlier_matches.png")
    cv2.imwrite(matches_path, match_vis)
    saved_paths["inlier_matches"] = matches_path
    
    # 2. registered_image_b.png
    reg_img_b = generate_registered_image(img_a, img_b, transform_matrix, transform_type)
    reg_path = os.path.join(output_dir, "registered_image_b.png")
    cv2.imwrite(reg_path, reg_img_b)
    saved_paths["registered_image_b"] = reg_path
    
    # 3. report.json
    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    saved_paths["report_json"] = json_path
    
    return saved_paths
