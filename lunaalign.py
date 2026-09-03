#!/usr/bin/env python3
"""
LunaAlign CLI: Lunar Image Registration & Overlap Verification Prototype.

Usage:
    python lunaalign.py image_a.png image_b.png --output results/
"""

import argparse
import sys
import os
import json
from lunaalign.config import LunaAlignConfig
from lunaalign.pipeline import LunaAlignPipeline


def main():
    parser = argparse.ArgumentParser(
        description="LunaAlign: Determine whether two lunar images depict the same terrain region and produce registration reports."
    )
    parser.add_argument("image_a", help="Path to reference lunar image (Image A)")
    parser.add_argument("image_b", help="Path to target lunar image (Image B)")
    parser.add_argument(
        "--output", "-o", 
        default="results", 
        help="Directory to save registration artifacts (default: results/)"
    )
    parser.add_argument(
        "--detector", 
        choices=["AUTO", "SIFT", "ORB"], 
        default="AUTO", 
        help="Feature detector to use (default: AUTO)"
    )
    parser.add_argument(
        "--matcher", 
        choices=["FLANN", "BF"], 
        default="FLANN", 
        help="Descriptor matcher (default: FLANN)"
    )
    parser.add_argument(
        "--ratio-thresh", 
        type=float, 
        default=0.75, 
        help="Lowe's ratio test threshold (default: 0.75)"
    )
    parser.add_argument(
        "--ransac-thresh", 
        type=float, 
        default=3.5, 
        help="RANSAC reprojection error threshold in pixels (default: 3.5)"
    )
    parser.add_argument(
        "--min-inliers", 
        type=int, 
        default=20, 
        help="Minimum inliers required for SAME REGION decision (default: 20)"
    )
    parser.add_argument(
        "--min-ratio", 
        type=float, 
        default=0.25, 
        help="Minimum inlier ratio required for SAME REGION decision (default: 0.25)"
    )
    parser.add_argument(
        "--max-rmse", 
        type=float, 
        default=4.5, 
        help="Maximum allowable RMSE in pixels (default: 4.5)"
    )
    parser.add_argument(
        "--min-cells", 
        type=int, 
        default=4, 
        help="Minimum occupied spatial grid cells (default: 4 out of 16)"
    )
    parser.add_argument(
        "--transform", 
        choices=["homography", "affine"], 
        default="homography", 
        help="Transformation model to estimate (default: homography)"
    )
    parser.add_argument(
        "--json-only", 
        action="store_true", 
        help="Output raw JSON to stdout instead of terminal card"
    )
    
    args = parser.parse_args()
    
    # Configure pipeline
    config = LunaAlignConfig(
        feature_type=args.detector,
        matcher_type=args.matcher,
        ratio_threshold=args.ratio_thresh,
        ransac_reproj_threshold=args.ransac_thresh,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_ratio,
        max_rmse_px=args.max_rmse,
        min_spatial_coverage_cells=args.min_cells,
        transform_type=args.transform,
        output_dir=args.output
    )
    
    pipeline = LunaAlignPipeline(config)
    
    try:
        results = pipeline.run(args.image_a, args.image_b, output_dir=args.output)
    except Exception as e:
        sys.stderr.write(f"Error executing LunaAlign pipeline: {e}\n")
        sys.exit(1)
        
    if args.json_only:
        print(json.dumps(results["report"], indent=2))
    else:
        print(results["terminal_report"])
        print()
        if results["saved_paths"]:
            print(f"Artifacts saved to {os.path.abspath(args.output)}:")
            for name, path in results["saved_paths"].items():
                print(f"  - {os.path.basename(path)} ({path})")


if __name__ == "__main__":
    main()
