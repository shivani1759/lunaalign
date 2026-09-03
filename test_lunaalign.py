"""
Automated unit and integration test suite for LunaAlign.
Tests:
  1. Preprocessing and feature extraction
  2. Same-region registration (expect SAME REGION, High/Medium confidence)
  3. Different-region rejection (expect DIFFERENT REGION / INSUFFICIENT EVIDENCE)
  4. Visualizations and JSON report structure
"""

import os
import shutil
import unittest
import numpy as np
from lunaalign.config import LunaAlignConfig
from lunaalign.pipeline import LunaAlignPipeline
from lunaalign.preprocessor import preprocess_image, load_image
from lunaalign.feature_matcher import detect_and_match_features
from lunaalign.spatial_analyzer import check_spatial_distribution
from generate_test_data import create_test_dataset


class TestLunaAlign(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(os.path.dirname(__file__), "test_data")
        cls.results_dir = os.path.join(os.path.dirname(__file__), "test_results")
        create_test_dataset(cls.test_dir)
        
        cls.img_1a_path = os.path.join(cls.test_dir, "region1_a.png")
        cls.img_1b_path = os.path.join(cls.test_dir, "region1_b.png")
        cls.img_2_path = os.path.join(cls.test_dir, "region2.png")
        
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.results_dir):
            shutil.rmtree(cls.results_dir)
            
    def test_preprocessing(self):
        config = LunaAlignConfig(max_dimension=800)
        raw, proc, scale = preprocess_image(self.img_1a_path, config)
        
        self.assertEqual(proc.dtype, np.uint8)
        self.assertLessEqual(max(proc.shape), 800)
        self.assertGreater(scale, 0.0)
        self.assertLessEqual(scale, 1.0)
        
    def test_feature_extraction_and_matching(self):
        config = LunaAlignConfig()
        _, proc_a, _ = preprocess_image(self.img_1a_path, config)
        _, proc_b, _ = preprocess_image(self.img_1b_path, config)
        
        kps_a, kps_b, matches, pts_a, pts_b, detector = detect_and_match_features(proc_a, proc_b, config)
        
        self.assertGreater(len(kps_a), 100)
        self.assertGreater(len(kps_b), 100)
        self.assertGreater(len(matches), 30)
        self.assertEqual(len(pts_a), len(matches))
        self.assertEqual(len(pts_b), len(matches))
        
    def test_spatial_distribution(self):
        # Synthetic grid test
        pts = np.array([
            [10, 10], [10, 300], [300, 10], [300, 300],
            [100, 100], [200, 200]
        ], dtype=np.float32)
        
        occupied, total, ratio, _ = check_spatial_distribution(pts, (400, 400), grid_rows=4, grid_cols=4)
        self.assertEqual(total, 16)
        self.assertGreaterEqual(occupied, 4)
        self.assertGreaterEqual(ratio, 0.25)
        
    def test_same_region_registration(self):
        """Verify that same-region overlapping pair evaluates to SAME REGION."""
        out_dir = os.path.join(self.results_dir, "same_region")
        pipeline = LunaAlignPipeline()
        result = pipeline.run(self.img_1a_path, self.img_1b_path, output_dir=out_dir)
        
        report = result["report"]
        print("\n--- Same Region Test Report ---")
        print(result["terminal_report"])
        
        self.assertEqual(report["decision"], "SAME REGION")
        self.assertIn(report["confidence"], ["high", "medium"])
        self.assertGreaterEqual(report["inliers"], 20)
        self.assertGreaterEqual(report["inlier_ratio"], 0.25)
        self.assertLessEqual(report["rmse_px"], 4.5)
        self.assertGreaterEqual(report["spatial_coverage_cells"], 4)
        
        # Verify artifact files exist
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "inlier_matches.png")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "registered_image_b.png")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "report.json")))
        
    def test_different_region_rejection(self):
        """Verify that different lunar terrain evaluates to DIFFERENT REGION / INSUFFICIENT EVIDENCE."""
        out_dir = os.path.join(self.results_dir, "diff_region")
        pipeline = LunaAlignPipeline()
        result = pipeline.run(self.img_1a_path, self.img_2_path, output_dir=out_dir)
        
        report = result["report"]
        print("\n--- Different Region Test Report ---")
        print(result["terminal_report"])
        
        self.assertEqual(report["decision"], "DIFFERENT REGION / INSUFFICIENT EVIDENCE")
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "report.json")))


if __name__ == "__main__":
    unittest.main()
