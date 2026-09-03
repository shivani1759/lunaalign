"""
Feature extraction and descriptor matching stage.
Uses SIFT (with ORB fallback), FLANN / BF matching, and Lowe's ratio test.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np
from .config import LunaAlignConfig


def get_feature_detector(
    config: LunaAlignConfig
) -> Tuple[cv2.Feature2D, str]:
    """
    Initialize feature detector based on config preferences and OpenCV capabilities.
    Prefers SIFT, falls back to ORB if SIFT is unavailable or explicitly requested.
    
    Args:
        config: LunaAlignConfig instance.
        
    Returns:
        Tuple of (detector_instance, detector_name).
    """
    requested = config.feature_type.upper()
    
    if requested in ("SIFT", "AUTO"):
        try:
            detector = cv2.SIFT_create(nfeatures=config.sift_nfeatures)
            return detector, "SIFT"
        except Exception:
            if requested == "SIFT":
                pass
                
    # Fallback or explicit ORB
    detector = cv2.ORB_create(
        nfeatures=config.orb_nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20
    )
    return detector, "ORB"


def get_matcher(
    detector_name: str, 
    matcher_type: str = "FLANN"
) -> cv2.DescriptorMatcher:
    """
    Configure descriptor matcher based on detector type and requested matcher.
    
    - SIFT (float descriptors): FLANN KD-Tree index or BFMatcher with cv2.NORM_L2
    - ORB (binary descriptors): FLANN LSH index or BFMatcher with cv2.NORM_HAMMING
    
    Args:
        detector_name: 'SIFT' or 'ORB'.
        matcher_type: 'FLANN' or 'BF'.
        
    Returns:
        cv2.DescriptorMatcher instance.
    """
    matcher_type = matcher_type.upper()
    
    if detector_name == "SIFT":
        if matcher_type == "FLANN":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:  # ORB / Binary
        if matcher_type == "FLANN":
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,
                key_size=12,
                multi_probe_level=1
            )
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def detect_and_match_features(
    image_a: np.ndarray,
    image_b: np.ndarray,
    config: Optional[LunaAlignConfig] = None
) -> Tuple[
    List[cv2.KeyPoint],
    List[cv2.KeyPoint],
    List[cv2.DMatch],
    np.ndarray,
    np.ndarray,
    str
]:
    """
    Detect keypoints and descriptors in both images, match using k-NN (k=2),
    and filter ambiguous matches with Lowe's ratio test.
    
    Args:
        image_a: Preprocessed grayscale Image A.
        image_b: Preprocessed grayscale Image B.
        config: LunaAlignConfig instance.
        
    Returns:
        Tuple of:
        - keypoints_a: KeyPoints detected in Image A.
        - keypoints_b: KeyPoints detected in Image B.
        - candidate_matches: List of cv2.DMatch passing Lowe's ratio test.
        - matched_pts_a: (N, 2) float32 coordinates in Image A.
        - matched_pts_b: (N, 2) float32 coordinates in Image B.
        - detector_name: Name of detector used ('SIFT' or 'ORB').
    """
    if config is None:
        config = LunaAlignConfig()
        
    detector, detector_name = get_feature_detector(config)
    
    # Detect and compute descriptors
    kps_a, desc_a = detector.detectAndCompute(image_a, None)
    kps_b, desc_b = detector.detectAndCompute(image_b, None)
    
    # Handle empty descriptor edge cases
    if desc_a is None or desc_b is None or len(kps_a) == 0 or len(kps_b) == 0:
        return (
            kps_a or [],
            kps_b or [],
            [],
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            detector_name
        )
        
    # Ensure float32 for SIFT FLANN matcher compatibility
    if detector_name == "SIFT":
        if desc_a.dtype != np.float32:
            desc_a = desc_a.astype(np.float32)
        if desc_b.dtype != np.float32:
            desc_b = desc_b.astype(np.float32)
            
    # Need at least 2 descriptors in image B for knnMatch(k=2)
    if len(desc_b) < 2 or len(desc_a) < 2:
        return (
            kps_a,
            kps_b,
            [],
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            detector_name
        )
        
    matcher = get_matcher(detector_name, config.matcher_type)
    
    try:
        raw_matches = matcher.knnMatch(desc_a, desc_b, k=2)
    except Exception:
        # Fallback to BFMatcher if FLANN fails on edge data
        bf = cv2.BFMatcher(
            cv2.NORM_L2 if detector_name == "SIFT" else cv2.NORM_HAMMING,
            crossCheck=False
        )
        raw_matches = bf.knnMatch(desc_a, desc_b, k=2)
        
    # Apply Lowe's ratio test: match is accepted if d1 < ratio * d2
    candidate_matches: List[cv2.DMatch] = []
    pts_a_list = []
    pts_b_list = []
    
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < config.ratio_threshold * n.distance:
                candidate_matches.append(m)
                pts_a_list.append(kps_a[m.queryIdx].pt)
                pts_b_list.append(kps_b[m.trainIdx].pt)
                
    pts_a = np.array(pts_a_list, dtype=np.float32) if pts_a_list else np.empty((0, 2), dtype=np.float32)
    pts_b = np.array(pts_b_list, dtype=np.float32) if pts_b_list else np.empty((0, 2), dtype=np.float32)
    
    return kps_a, kps_b, candidate_matches, pts_a, pts_b, detector_name
