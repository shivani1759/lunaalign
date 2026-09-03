"""
Multi-scale pyramid and tile-based cross-resolution matching module.
Enables matching between high-resolution swaths (e.g. Chandrayaan-2 OHRC at ~0.25m/px)
and lower-resolution context maps (e.g. TMC-2 at ~5m/px or global basemaps).
"""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
from .config import LunaAlignConfig
from .feature_matcher import get_feature_detector, get_matcher


def multi_scale_detect_and_match(
    image_a: np.ndarray,
    image_b: np.ndarray,
    config: LunaAlignConfig,
    scales: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 2.0)
) -> Tuple[
    List[cv2.KeyPoint],
    List[cv2.KeyPoint],
    List[cv2.DMatch],
    np.ndarray,
    np.ndarray,
    float,
    str
]:
    """
    Perform multi-scale pyramid matching across multiple scale factors.
    Tests downsampled/upsampled versions of Image B against Image A
    to automatically identify and bridge the Ground Sampling Distance (GSD) gap.
    
    Args:
        image_a: Reference Image A.
        image_b: Target Image B.
        config: LunaAlignConfig instance.
        scales: Scale factors to test for Image B.
        
    Returns:
        Tuple of:
        - best_kps_a: KeyPoints in Image A.
        - best_kps_b: KeyPoints in Image B (scaled back to original Image B coordinates).
        - best_matches: Filtered DMatches.
        - best_pts_a: Matched points in Image A.
        - best_pts_b: Matched points in Image B (original scale).
        - best_scale_factor: The detected scale ratio between Image B and Image A.
        - detector_name: 'SIFT' or 'ORB'.
    """
    detector, detector_name = get_feature_detector(config)
    matcher = get_matcher(detector_name, config.matcher_type)
    
    kps_a, desc_a = detector.detectAndCompute(image_a, None)
    if desc_a is None or len(kps_a) == 0:
        return [], [], [], np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), 1.0, detector_name
        
    if detector_name == "SIFT" and desc_a.dtype != np.float32:
        desc_a = desc_a.astype(np.float32)
        
    best_candidate_count = -1
    best_result = ([], [], [], np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), 1.0)
    
    for s in scales:
        if s == 1.0:
            scaled_b = image_b
        else:
            new_w = max(32, int(round(image_b.shape[1] * s)))
            new_h = max(32, int(round(image_b.shape[0] * s)))
            scaled_b = cv2.resize(image_b, (new_w, new_h), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
            
        kps_b_s, desc_b_s = detector.detectAndCompute(scaled_b, None)
        if desc_b_s is None or len(kps_b_s) < 2 or len(desc_b_s) < 2:
            continue
            
        if detector_name == "SIFT" and desc_b_s.dtype != np.float32:
            desc_b_s = desc_b_s.astype(np.float32)
            
        try:
            raw_matches = matcher.knnMatch(desc_a, desc_b_s, k=2)
        except Exception:
            bf = cv2.BFMatcher(cv2.NORM_L2 if detector_name == "SIFT" else cv2.NORM_HAMMING, crossCheck=False)
            raw_matches = bf.knnMatch(desc_a, desc_b_s, k=2)
            
        # Lowe's ratio test
        good_matches = []
        pts_a_list = []
        pts_b_list = []
        
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < config.ratio_threshold * n.distance:
                    good_matches.append(m)
                    pts_a_list.append(kps_a[m.queryIdx].pt)
                    # Rescale keypoint coordinate back to original Image B size
                    pt_b_orig = (kps_b_s[m.trainIdx].pt[0] / s, kps_b_s[m.trainIdx].pt[1] / s)
                    pts_b_list.append(pt_b_orig)
                    
        # Check if this scale gives significantly more matches
        if len(good_matches) > best_candidate_count:
            best_candidate_count = len(good_matches)
            
            # Rescale all kps_b back to original image coordinates
            kps_b_orig = [
                cv2.KeyPoint(x=kp.pt[0] / s, y=kp.pt[1] / s, size=kp.size / s, angle=kp.angle, response=kp.response, octave=kp.octave, class_id=kp.class_id)
                for kp in kps_b_s
            ]
            
            pts_a_arr = np.array(pts_a_list, dtype=np.float32) if pts_a_list else np.empty((0, 2), dtype=np.float32)
            pts_b_arr = np.array(pts_b_list, dtype=np.float32) if pts_b_list else np.empty((0, 2), dtype=np.float32)
            
            best_result = (kps_a, kps_b_orig, good_matches, pts_a_arr, pts_b_arr, s)
            
    best_kps_a, best_kps_b, best_matches, best_pts_a, best_pts_b, best_scale = best_result
    return best_kps_a, best_kps_b, best_matches, best_pts_a, best_pts_b, best_scale, detector_name
