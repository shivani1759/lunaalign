"""
LunaAlign core pipeline orchestrator with completed Tier 2 Reliability Layer:
1. Illumination-aware preprocessing.
2. Multi-scale feature matching.
3. RANSAC geometric verification.
4. Sub-pixel refinement of RANSAC inliers.
5. Illumination consistency gradient structure assessment.
6. Multi-gate confidence scoring (0-100 score + explanation).
7. Registered image + completed telemetry JSON.
"""

from typing import Dict, Any, Optional, Union
import numpy as np
from .config import LunaAlignConfig
from .preprocessor import preprocess_image_tier2
from .feature_matcher import detect_and_match_features
from .multi_scale_matcher import multi_scale_detect_and_match
from .geometric_verifier import verify_geometry_ransac
from .spatial_analyzer import check_spatial_distribution
from .subpixel_refiner import refine_inliers_subpixel
from .illumination_normalizer import compute_illumination_consistency
from .decision_engine import make_decision
from .visualizer import save_visualizations, format_terminal_card


class LunaAlignPipeline:
    """LunaAlign Pipeline with completed Tier 2 Reliability Layer."""
    
    def __init__(self, config: Optional[LunaAlignConfig] = None):
        self.config = config or LunaAlignConfig()
        
    def run(
        self,
        image_a_input: Union[str, np.ndarray],
        image_b_input: Union[str, np.ndarray],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. Illumination-aware preprocessing
        raw_a, proc_a, grad_a, prev_before_a, prev_after_a, scale_a = preprocess_image_tier2(image_a_input, self.config)
        raw_b, proc_b, grad_b, prev_before_b, prev_after_b, scale_b = preprocess_image_tier2(image_b_input, self.config)
        
        # 2. Existing multi-scale feature matching
        kps_a, kps_b, candidate_matches, pts_a, pts_b, detected_scale, detector_name = multi_scale_detect_and_match(
            proc_a, proc_b, self.config
        )
        
        # 3. Existing RANSAC geometric verification
        inliers_count, inlier_ratio, rmse_raw, transform_matrix, inlier_mask, transform_type = verify_geometry_ransac(
            pts_a, pts_b, self.config
        )
        
        # 4. Tier 2: Sub-pixel refinement of RANSAC inliers
        pts_a_inliers = pts_a[inlier_mask] if np.any(inlier_mask) else np.empty((0, 2), dtype=np.float32)
        pts_b_inliers = pts_b[inlier_mask] if np.any(inlier_mask) else np.empty((0, 2), dtype=np.float32)
        
        if self.config.enable_subpixel and inliers_count >= 4 and transform_matrix is not None:
            pts_a_ref, pts_b_ref, subpixel_metrics = refine_inliers_subpixel(
                image_a=proc_a,
                image_b=proc_b,
                pts_a=pts_a_inliers,
                pts_b=pts_b_inliers,
                transform_matrix=transform_matrix,
                transform_type=transform_type,
                patch_size=self.config.subpixel_patch_size
            )
            effective_rmse = subpixel_metrics.get("effective_rmse_px") or rmse_raw
        else:
            subpixel_metrics = {
                "rmse_before_px": round(rmse_raw, 2) if rmse_raw != float('inf') else None,
                "rmse_after_px": round(rmse_raw, 2) if rmse_raw != float('inf') else None,
                "effective_rmse_px": round(rmse_raw, 2) if rmse_raw != float('inf') else None,
                "avg_subpixel_correction_px": 0.0,
                "max_subpixel_correction_px": 0.0,
                "subpixel_status": "not_applicable",
                "status_label": "Not Applicable",
                "percentage_improvement": 0.0,
                "conclusion": "No inliers available for sub-pixel refinement.",
                "refined_points_count": inliers_count
            }
            effective_rmse = rmse_raw
            
        # 5. Spatial distribution check
        spatial_cells, total_cells, coverage_ratio, cell_counts = check_spatial_distribution(
            pts_a_inliers,
            proc_a.shape,
            self.config.grid_rows,
            self.config.grid_cols
        )
        
        # 6. Illumination Consistency Gate (Fix 5)
        illum_consistency = None
        if inliers_count >= 4:
            illum_consistency = compute_illumination_consistency(
                image_a=proc_a,
                image_b=proc_b,
                pts_a=pts_a_inliers,
                pts_b=pts_b_inliers,
                patch_size=15
            )
            
        # 7. Multi-gate confidence assessment (Fixes 1, 2, 3, 4, 7)
        decision_result = make_decision(
            candidate_matches=len(candidate_matches),
            inliers_count=inliers_count,
            inlier_ratio=inlier_ratio,
            rmse_px=effective_rmse,
            spatial_coverage_cells=spatial_cells,
            grid_total_cells=total_cells,
            config=self.config,
            subpixel_metrics=subpixel_metrics,
            illumination_consistency=illum_consistency
        )
        
        matrix_list = transform_matrix.tolist() if transform_matrix is not None else []
        
        # Required Telemetry JSON structure (Fix 8)
        report_data = {
            "decision": decision_result["decision"],
            "confidence_score": decision_result["confidence_score"],
            "confidence_label": decision_result["confidence_label"],
            "confidence": decision_result["confidence"],
            "image_a_keypoints": len(kps_a),
            "image_b_keypoints": len(kps_b),
            "candidate_matches": len(candidate_matches),
            "candidate_match_threshold": self.config.min_candidate_matches,
            "ransac_inliers": inliers_count,
            "minimum_inlier_threshold": self.config.min_inliers,
            "inlier_ratio": round(inlier_ratio, 3),
            "minimum_inlier_ratio": self.config.min_inlier_ratio,
            "rmse_before_subpixel_px": subpixel_metrics.get("rmse_before_px"),
            "rmse_after_subpixel_px": subpixel_metrics.get("rmse_after_px"),
            "maximum_rmse_px": self.config.max_rmse_px,
            "subpixel_status": subpixel_metrics.get("subpixel_status"),
            "subpixel_status_label": subpixel_metrics.get("status_label"),
            "subpixel_conclusion": subpixel_metrics.get("conclusion"),
            "subpixel_improvement_pct": subpixel_metrics.get("percentage_improvement"),
            "average_subpixel_shift_px": subpixel_metrics.get("avg_subpixel_correction_px"),
            "spatial_coverage_cells": spatial_cells,
            "minimum_spatial_coverage_cells": self.config.min_spatial_coverage_cells,
            "illumination_consistency": illum_consistency,
            "minimum_illumination_consistency": self.config.min_illumination_consistency,
            "gates": decision_result["gates"],
            "gates_detail": decision_result["gates_detail"],
            "explanation": decision_result["explanation"],
            "product_message": decision_result["product_message"],
            "disclaimer": decision_result["disclaimer"],
            
            # Legacy/Additional fields for UI compatibility
            "inliers": inliers_count,
            "rmse_px": effective_rmse if effective_rmse != float('inf') else None,
            "grid_total_cells": total_cells,
            "transform_type": transform_type,
            "transform_matrix": matrix_list,
            "detected_scale_ratio": round(detected_scale, 3),
            "detector": detector_name,
            "illumination_mode": self.config.illumination_mode,
            "quality": decision_result["quality"],
            "spatial_coverage_ratio": coverage_ratio
        }
        
        saved_paths = {}
        target_output_dir = output_dir or self.config.output_dir
        if target_output_dir:
            saved_paths = save_visualizations(
                output_dir=target_output_dir,
                img_a=proc_a,
                img_b=proc_b,
                kps_a=kps_a,
                kps_b=kps_b,
                candidate_matches=candidate_matches,
                inlier_mask=inlier_mask,
                transform_matrix=transform_matrix,
                transform_type=transform_type,
                report=report_data,
                draw_rejected=self.config.draw_rejected_matches
            )
            
        terminal_report = format_terminal_card(report_data)
        
        return {
            "report": report_data,
            "terminal_report": terminal_report,
            "saved_paths": saved_paths,
            "preprocessed_a": proc_a,
            "preprocessed_b": proc_b,
            "raw_a": raw_a,
            "raw_b": raw_b,
            "preview_before_a": prev_before_a,
            "preview_after_a": prev_after_a,
            "preview_before_b": prev_before_b,
            "preview_after_b": prev_after_b,
            "subpixel_metrics": subpixel_metrics,
            "illumination_consistency": illum_consistency,
            "transform_matrix": transform_matrix
        }
