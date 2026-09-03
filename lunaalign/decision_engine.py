"""
Tier 2 Multi-Gate Confidence Assessment Engine for LunaAlign.
Implements:
- Fix 1: Correct Inlier Count Gate label with dynamic thresholds.
- Fix 2: Candidate Match Gate.
- Fix 3: Show every threshold clearly (Measured / configured threshold).
- Fix 4: 100-point Numeric Confidence Score & Critical Gate override rule.
- Fix 5: Illumination Consistency Gate.
- Fix 7: Plain-language explanations.
"""

from typing import Dict, Any, Optional
from .config import LunaAlignConfig


PRODUCT_MESSAGE = (
    "LunaAlign does not simply find matches. It evaluates whether the matches "
    "are trustworthy enough to support lunar terrain correspondence and registration."
)


def calculate_confidence_score(
    candidate_matches: int,
    inliers_count: int,
    inlier_ratio: float,
    rmse_px: Optional[float],
    spatial_coverage_cells: int,
    illumination_consistency: Optional[float],
    subpixel_status: str,
    config: LunaAlignConfig
) -> int:
    """
    Fix 4: 100-Point Scoring Model
    Candidate match strength:        10 points
    RANSAC inlier count:             25 points
    Inlier ratio:                    20 points
    Refined RMSE:                    20 points
    Spatial distribution:            15 points
    Illumination consistency:        5 points
    Sub-pixel refinement quality:    5 points
    Total:                          100 points
    """
    # 1. Candidate Matches (10 pts)
    min_cand = config.min_candidate_matches
    if candidate_matches >= min_cand * 2:
        cand_score = 10.0
    elif candidate_matches >= min_cand:
        cand_score = 5.0 + 5.0 * ((candidate_matches - min_cand) / max(1, min_cand))
    else:
        cand_score = 5.0 * (candidate_matches / max(1, min_cand))
        
    # 2. Inliers Count (25 pts)
    min_inl = config.min_inliers
    if inliers_count >= 100:
        inl_score = 25.0
    elif inliers_count >= 50:
        inl_score = 20.0
    elif inliers_count >= min_inl:
        inl_score = 10.0 + 10.0 * ((inliers_count - min_inl) / max(1, 50 - min_inl))
    else:
        inl_score = 10.0 * (inliers_count / max(1, min_inl))
        
    # 3. Inlier Ratio (20 pts)
    min_rat = config.min_inlier_ratio
    if inlier_ratio >= 0.75:
        rat_score = 20.0
    elif inlier_ratio >= 0.50:
        rat_score = 16.0
    elif inlier_ratio >= min_rat:
        rat_score = 8.0 + 8.0 * ((inlier_ratio - min_rat) / max(1e-4, 0.50 - min_rat))
    else:
        rat_score = 8.0 * (inlier_ratio / max(1e-4, min_rat))
        
    # 4. Refined RMSE (20 pts)
    if rmse_px is None or rmse_px == float('inf'):
        rmse_score = 0.0
    elif rmse_px <= 1.0:
        rmse_score = 20.0
    elif rmse_px <= 2.0:
        rmse_score = 16.0
    elif rmse_px <= config.max_rmse_px:
        rmse_score = 8.0 + 8.0 * ((config.max_rmse_px - rmse_px) / max(1e-4, config.max_rmse_px - 2.0))
    else:
        rmse_score = 0.0
        
    # 5. Spatial Distribution (15 pts)
    min_spat = config.min_spatial_coverage_cells
    if spatial_coverage_cells >= 12:
        spat_score = 15.0
    elif spatial_coverage_cells >= 8:
        spat_score = 11.0
    elif spatial_coverage_cells >= min_spat:
        spat_score = 5.0 + 6.0 * ((spatial_coverage_cells - min_spat) / max(1, 8 - min_spat))
    else:
        spat_score = 5.0 * (spatial_coverage_cells / max(1, min_spat))
        
    # 6. Illumination Consistency (5 pts)
    if illumination_consistency is not None:
        if illumination_consistency >= 0.70:
            illum_score = 5.0
        elif illumination_consistency >= config.min_illumination_consistency:
            illum_score = 3.5
        else:
            illum_score = 1.0
    else:
        illum_score = 2.5  # Neutral when not available
        
    # 7. Sub-Pixel Quality (5 pts)
    if subpixel_status in ("applied_improved_registration", "applied_no_measurable_improvement"):
        subpix_score = 5.0
    else:
        subpix_score = 0.0
        
    total = int(round(cand_score + inl_score + rat_score + rmse_score + spat_score + illum_score + subpix_score))
    total = max(0, min(100, total))
    return total


def make_decision(
    candidate_matches: int,
    inliers_count: int,
    inlier_ratio: float,
    rmse_px: Optional[float],
    spatial_coverage_cells: int,
    grid_total_cells: int,
    config: Optional[LunaAlignConfig] = None,
    subpixel_metrics: Optional[Dict[str, Any]] = None,
    illumination_consistency: Optional[float] = None
) -> Dict[str, Any]:
    """
    Tier 2 Multi-Gate Decision Logic with full fixes.
    """
    if config is None:
        config = LunaAlignConfig()
        
    subpix_status = subpixel_metrics.get("subpixel_status", "not_applicable") if subpixel_metrics else "not_applicable"
    
    # 1. Check Individual Gates (Pass / Fail)
    gate_cand = candidate_matches >= config.min_candidate_matches
    gate_inliers = inliers_count >= config.min_inliers
    gate_ratio = inlier_ratio >= config.min_inlier_ratio
    gate_rmse = (rmse_px is not None) and (rmse_px != float('inf')) and (rmse_px <= config.max_rmse_px)
    gate_spatial = spatial_coverage_cells >= config.min_spatial_coverage_cells
    
    if illumination_consistency is not None:
        gate_illum = illumination_consistency >= config.min_illumination_consistency
    else:
        gate_illum = True  # Not failing if unavailable
        
    gate_subpix = subpix_status in ("applied_improved_registration", "applied_no_measurable_improvement", "not_applicable")
    
    critical_gates_passed = gate_cand and gate_inliers and gate_ratio and gate_rmse and gate_spatial
    
    # 2. Compute Numeric Confidence Score (0 - 100)
    raw_score = calculate_confidence_score(
        candidate_matches=candidate_matches,
        inliers_count=inliers_count,
        inlier_ratio=inlier_ratio,
        rmse_px=rmse_px,
        spatial_coverage_cells=spatial_coverage_cells,
        illumination_consistency=illumination_consistency,
        subpixel_status=subpix_status,
        config=config
    )
    
    # Fix 4 Critical Rule: A high score must never override a failed critical gate
    if critical_gates_passed:
        decision = "SAME REGION"
        confidence_score = max(60, raw_score)
        if confidence_score >= 80:
            confidence_label = "HIGH CONFIDENCE"
            confidence_level = "High"
            quality = "Good"
        else:
            confidence_label = "MEDIUM CONFIDENCE"
            confidence_level = "Medium"
            quality = "Good" if confidence_score >= 70 else "Moderate"
            
        explanation = (
            "Trustworthiness conclusion: This pair passes all critical geometric gates. "
            "The matches are numerous, spatially distributed, and accurately aligned. "
            + ("Gradient structure is consistent despite illumination differences. " if (illumination_consistency and illumination_consistency >= 0.40) else "")
            + "The terrain correspondence is trustworthy."
        )
    else:
        decision = "DIFFERENT REGION / INSUFFICIENT EVIDENCE"
        confidence_score = min(55, raw_score) if (inliers_count < 10 or not gate_inliers or not gate_cand) else min(59, raw_score)
        if confidence_score >= 40:
            confidence_label = "LOW CONFIDENCE / REVIEW NEEDED"
            confidence_level = "Low"
            quality = "Inconclusive"
        else:
            confidence_label = "INSUFFICIENT EVIDENCE"
            confidence_level = "Low"
            quality = "Poor"
            
        reasons = []
        if not gate_cand:
            reasons.append(f"insufficient candidate matches ({candidate_matches} < {config.min_candidate_matches})")
        if not gate_inliers:
            reasons.append(f"low inlier count ({inliers_count} < {config.min_inliers})")
        if not gate_ratio:
            reasons.append(f"low inlier ratio ({round(inlier_ratio*100, 1)}% < {round(config.min_inlier_ratio*100, 1)}%)")
        if not gate_rmse:
            reasons.append(f"high reprojection RMSE ({rmse_px}px > {config.max_rmse_px}px)")
        if not gate_spatial:
            reasons.append(f"clustered spatial coverage ({spatial_coverage_cells}/16 cells < {config.min_spatial_coverage_cells})")
            
        failed_str = ", ".join(reasons) if reasons else "insufficient geometric consistency"
        explanation = (
            f"Why this is not SAME REGION: Although some feature matches were detected, "
            f"this pair failed critical evidence gates ({failed_str}). "
            f"The system correctly returns insufficient evidence instead of a false positive."
        )
        
    subpix_gate_val = "improved" if subpix_status == "applied_improved_registration" else (
        "no_measurable_improvement" if subpix_status == "applied_no_measurable_improvement" else "rejected"
    )
    
    return {
        "decision": decision,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "confidence": confidence_level.lower(),
        "quality": quality,
        "explanation": explanation,
        "gates": {
            "candidate_matches": "pass" if gate_cand else "fail",
            "inlier_count": "pass" if gate_inliers else "fail",
            "inlier_ratio": "pass" if gate_ratio else "fail",
            "rmse": "pass" if gate_rmse else "fail",
            "spatial_coverage": "pass" if gate_spatial else "fail",
            "illumination_consistency": "pass" if gate_illum else ("unavailable" if illumination_consistency is None else "fail"),
            "subpixel_refinement": subpix_gate_val
        },
        "gates_detail": {
            "candidate_matches": {
                "passed": gate_cand,
                "measured": candidate_matches,
                "threshold": config.min_candidate_matches,
                "text": f"{candidate_matches} matches / minimum {config.min_candidate_matches}"
            },
            "inlier_count": {
                "passed": gate_inliers,
                "measured": inliers_count,
                "threshold": config.min_inliers,
                "text": f"{inliers_count} inliers / minimum {config.min_inliers}"
            },
            "inlier_ratio": {
                "passed": gate_ratio,
                "measured": inlier_ratio,
                "threshold": config.min_inlier_ratio,
                "text": f"{round(inlier_ratio*100, 1)}% / minimum {round(config.min_inlier_ratio*100, 1)}%"
            },
            "rmse": {
                "passed": gate_rmse,
                "measured": rmse_px,
                "threshold": config.max_rmse_px,
                "text": f"{rmse_px if rmse_px is not None else 'N/A'} px / maximum {config.max_rmse_px} px"
            },
            "spatial_coverage": {
                "passed": gate_spatial,
                "measured": spatial_coverage_cells,
                "threshold": config.min_spatial_coverage_cells,
                "total_cells": grid_total_cells,
                "text": f"{spatial_coverage_cells} / {grid_total_cells} cells / minimum {config.min_spatial_coverage_cells} cells"
            },
            "illumination_consistency": {
                "passed": gate_illum,
                "measured": illumination_consistency,
                "threshold": config.min_illumination_consistency,
                "available": illumination_consistency is not None,
                "text": f"Gradient similarity: {illumination_consistency if illumination_consistency is not None else 'N/A'} / minimum {config.min_illumination_consistency}"
            },
            "subpixel_refinement": {
                "passed": gate_subpix,
                "status": subpix_status,
                "status_label": subpixel_metrics.get("status_label", "N/A") if subpixel_metrics else "N/A",
                "text": subpixel_metrics.get("conclusion", "N/A") if subpixel_metrics else "N/A"
            }
        },
        "product_message": PRODUCT_MESSAGE,
        "disclaimer": (
            "Note: LunaAlign baseline decision is a feature-based geometric confidence assessment, "
            "not absolute geographic proof. Future versions incorporate multi-sensor ephemeris, "
            "DEM raycasting, and deep feature representations."
        )
    }
