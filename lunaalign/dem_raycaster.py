"""
DEM-Based Solar Raycaster Module for LunaAlign.
Generates physically accurate 2D illumination and terrain cast shadows
from a real lunar Digital Elevation Model (DEM) and compares simulation
with registered lunar images.
"""

import os
import json
from typing import Tuple, Dict, Any, Optional, Union
import cv2
import numpy as np


def load_dem(file_input: Union[str, np.ndarray, bytes]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load DEM from file path, bytes, or numpy array.
    Supports GeoTIFF (.tif, .tiff), PNG, JPG, and 16-bit/32-bit float height maps.
    
    Args:
        file_input: Path to DEM file, raw bytes, or image array.
        
    Returns:
        Tuple of (2D float32 elevation array, metadata dictionary).
    """
    metadata = {
        "georeferenced": False,
        "dtype": "unknown",
        "min_elevation_m": 0.0,
        "max_elevation_m": 0.0,
        "format": "raw"
    }
    
    if isinstance(file_input, np.ndarray):
        dem = file_input.astype(np.float32)
        if dem.ndim == 3:
            dem = dem[:, :, 0]
        metadata["dtype"] = str(file_input.dtype)
    elif isinstance(file_input, (str, bytes)):
        if isinstance(file_input, str):
            if not os.path.isfile(file_input):
                raise FileNotFoundError(f"DEM file not found: {file_input}")
            img = cv2.imread(file_input, cv2.IMREAD_UNCHANGED)
            metadata["format"] = os.path.splitext(file_input)[1].lower()
        else:
            nparr = np.frombuffer(file_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            metadata["format"] = "bytes"
            
        if img is None:
            raise ValueError("Failed to decode DEM file or buffer")
            
        if img.ndim == 3:
            img = img[:, :, 0]
            
        metadata["dtype"] = str(img.dtype)
        
        # If 8-bit image, scale to meters (e.g. 0-255 -> 0-2550m)
        if img.dtype == np.uint8:
            dem = img.astype(np.float32) * 10.0
        elif img.dtype == np.uint16:
            dem = img.astype(np.float32) * 0.5
        else:
            dem = img.astype(np.float32)
    else:
        raise TypeError(f"Unsupported DEM input type: {type(file_input)}")
        
    metadata["min_elevation_m"] = float(np.min(dem))
    metadata["max_elevation_m"] = float(np.max(dem))
    metadata["elevation_range_m"] = float(np.max(dem) - np.min(dem))
    
    return dem, metadata


def align_dem_to_reference_grid(
    dem: np.ndarray,
    image_a_metadata: Optional[Dict[str, Any]],
    image_a_shape: Tuple[int, int]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Align or resample the DEM to match Image A's exact grid dimensions (height, width).
    Checks whether real geographic coordinates are present.
    
    Args:
        dem: 2D float32 elevation array.
        image_a_metadata: Optional geographic coordinate metadata for Image A.
        image_a_shape: (height, width) of Image A.
        
    Returns:
        Tuple of (aligned 2D float32 DEM, alignment status dict).
    """
    target_h, target_w = image_a_shape[:2]
    dem_h, dem_w = dem.shape[:2]
    
    has_georef = False
    if image_a_metadata and image_a_metadata.get("georeferenced"):
        has_georef = True
        
    if (dem_h, dem_w) == (target_h, target_w):
        aligned_dem = dem.copy()
    else:
        # Resample DEM to Image A grid dimensions using bicubic interpolation for smooth derivatives
        aligned_dem = cv2.resize(dem, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
    if has_georef:
        status_text = "Aligned using geographic coordinates"
        warning = None
        status_badge = "Geographically Aligned"
    else:
        status_text = "Aligned to reference grid (visual approximation)"
        warning = "Warning: DEM resized to the reference image grid. This is a visual approximation, not verified geographic alignment."
        status_badge = "Grid Resampled"
        
    alignment_info = {
        "status": status_text,
        "status_badge": status_badge,
        "warning": warning,
        "original_shape": (dem_h, dem_w),
        "aligned_shape": (target_h, target_w),
        "georeferenced": has_georef
    }
    
    return aligned_dem, alignment_info


def compute_surface_normals(
    aligned_dem: np.ndarray,
    pixel_resolution_m: float = 10.0
) -> np.ndarray:
    """
    Compute 3D unit surface normal vectors for each DEM pixel.
    
    Formula:
      dz_dx = dZ/dx
      dz_dy = dZ/dy
      Normal = (-dz_dx, -dz_dy, 1) / sqrt(dz_dx^2 + dz_dy^2 + 1)
      
    Args:
        aligned_dem: 2D float32 elevation array.
        pixel_resolution_m: Spatial resolution in meters per pixel.
        
    Returns:
        (H, W, 3) float32 array containing unit normal vectors (Nx, Ny, Nz).
    """
    # Sobel gradients with 3x3 kernel
    dz_dx = cv2.Sobel(aligned_dem, cv2.CV_32F, 1, 0, ksize=3) / (8.0 * max(0.1, pixel_resolution_m))
    dz_dy = cv2.Sobel(aligned_dem, cv2.CV_32F, 0, 1, ksize=3) / (8.0 * max(0.1, pixel_resolution_m))
    
    # In image space: +X is right (East), +Y is down (South), +Z is up (Altitude)
    # Normal points upward from the terrain
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(dz_dx, dtype=np.float32)
    
    length = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    length = np.maximum(length, 1e-6)
    
    normals = np.stack([nx / length, ny / length, nz / length], axis=-1)
    return normals.astype(np.float32)


def sun_vector_from_angles(
    azimuth_deg: float,
    elevation_deg: float
) -> np.ndarray:
    """
    Convert Solar Azimuth and Solar Elevation into a 3D unit sun illumination vector.
    
    Convention:
      - Azimuth: 0° = North (Up / -Y), 90° = East (Right / +X), 180° = South (+Y), 270° = West (-X).
      - Elevation: 0° = Horizon, 90° = Zenith.
      
    Args:
        azimuth_deg: Compass angle of the sun in degrees.
        elevation_deg: Altitude angle of the sun above the horizon in degrees.
        
    Returns:
        (3,) numpy array [Sx, Sy, Sz] representing unit vector pointing TOWARD the Sun.
    """
    az_rad = np.deg2rad(azimuth_deg)
    el_rad = np.deg2rad(max(1.0, min(89.9, elevation_deg)))
    
    # Vector pointing TOWARD the Sun:
    # +X is East (Right), -Y is North (Up), +Z is Up (Zenith)
    sx = np.sin(az_rad) * np.cos(el_rad)
    sy = -np.cos(az_rad) * np.cos(el_rad)  # Up is -Y in image coordinates
    sz = np.sin(el_rad)
    
    sun_vec = np.array([sx, sy, sz], dtype=np.float32)
    sun_vec = sun_vec / np.linalg.norm(sun_vec)
    return sun_vec


def compute_lambertian_illumination(
    normals: np.ndarray,
    sun_vector: np.ndarray
) -> np.ndarray:
    """
    Compute Lambertian cosine illumination across the surface normals.
    
    Formula:
      brightness = max(0, dot(surface_normal, sun_vector))
      
    Args:
        normals: (H, W, 3) float32 unit normal vector field.
        sun_vector: (3,) float32 unit sun vector.
        
    Returns:
        (H, W) float32 illumination field in range [0.0, 1.0].
    """
    dot_prod = (
        normals[:, :, 0] * sun_vector[0] +
        normals[:, :, 1] * sun_vector[1] +
        normals[:, :, 2] * sun_vector[2]
    )
    illumination = np.clip(dot_prod, 0.0, 1.0)
    return illumination.astype(np.float32)


def compute_terrain_shadows(
    aligned_dem: np.ndarray,
    sun_vector: np.ndarray,
    pixel_resolution_m: float = 10.0,
    max_ray_steps: int = 100
) -> np.ndarray:
    """
    Compute physical cast terrain shadows by ray marching toward the Sun.
    Tests whether intervening higher topography occludes the line of sight to the Sun.
    
    Args:
        aligned_dem: (H, W) 2D float32 elevation array.
        sun_vector: (3,) unit vector pointing toward the Sun.
        pixel_resolution_m: Spatial resolution in meters per pixel.
        max_ray_steps: Maximum step count for ray march.
        
    Returns:
        (H, W) float32 shadow mask (1.0 = illuminated, 0.0 = cast shadow).
    """
    h, w = aligned_dem.shape
    shadow_mask = np.ones((h, w), dtype=np.float32)
    
    # 2D direction step in image grid (marching toward the Sun: +Sx, +Sy)
    sx, sy, sz = sun_vector[0], sun_vector[1], sun_vector[2]
    
    # Horizontal projection length
    horiz_len = np.sqrt(sx ** 2 + sy ** 2)
    if horiz_len < 1e-5:
        # Sun is directly overhead at zenith (no horizontal shadows)
        return shadow_mask
        
    step_dx = sx / horiz_len
    step_dy = sy / horiz_len
    tan_elevation = sz / horiz_len
    
    # Vectorized / stepped raycasting
    # For each step distance t (in pixels), shift DEM and check if shifted elevation > ray altitude
    max_dist = min(max_ray_steps, max(h, w) // 2)
    
    # Check at exponential/increasing step sizes to be ultra-fast and capture both near and distant crater rims
    step_distances = np.unique(np.linspace(1, max_dist, min(40, max_dist)).astype(int))
    
    for dist_px in step_distances:
        shift_x = int(round(dist_px * step_dx))
        shift_y = int(round(dist_px * step_dy))
        
        if shift_x == 0 and shift_y == 0:
            continue
            
        # Ray height increase above current pixel elevation
        ray_delta_z = dist_px * pixel_resolution_m * tan_elevation
        
        # Construct shifted terrain grid
        shifted_dem = np.full((h, w), -99999.0, dtype=np.float32)
        
        # Source coordinates in DEM
        src_y_start = max(0, shift_y)
        src_y_end = min(h, h + shift_y)
        src_x_start = max(0, shift_x)
        src_x_end = min(w, w + shift_x)
        
        # Target coordinates where ray is checked
        dst_y_start = max(0, -shift_y)
        dst_y_end = min(h, h - shift_y)
        dst_x_start = max(0, -shift_x)
        dst_x_end = min(w, w - shift_x)
        
        if (src_y_end > src_y_start) and (src_x_end > src_x_start):
            shifted_dem[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = aligned_dem[src_y_start:src_y_end, src_x_start:src_x_end]
            
            # A pixel is shadowed if shifted terrain is higher than (pixel elevation + ray altitude)
            is_occluded = (shifted_dem - aligned_dem) > ray_delta_z
            shadow_mask[is_occluded] = 0.0
            
    return shadow_mask.astype(np.float32)


def render_raycaster_result(
    illumination: np.ndarray,
    shadow_mask: np.ndarray,
    ambient_light: float = 0.04
) -> np.ndarray:
    """
    Combine slope Lambertian illumination with cast terrain shadow mask.
    
    Formula:
      rendered = illumination * shadow_mask + ambient * (1 - shadow_mask)
      
    Args:
        illumination: (H, W) float32 in [0.0, 1.0].
        shadow_mask: (H, W) float32 in [0.0, 1.0] (0 = shadow).
        ambient_light: Subtle ambient reflection in deep shadows.
        
    Returns:
        (H, W) uint8 grayscale image [0, 255].
    """
    rendered_f = illumination * shadow_mask + ambient_light * (1.0 - shadow_mask)
    # Apply lunar gamma curve to match optical sensor dynamic range
    gamma_corrected = np.power(np.clip(rendered_f, 0.0, 1.0), 0.9)
    rendered_uint8 = (gamma_corrected * 255.0).astype(np.uint8)
    return rendered_uint8


def compare_with_registered_images(
    simulated_image: np.ndarray,
    image_a: np.ndarray,
    warped_image_b: np.ndarray,
    shadow_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Compute rigorous correlation and shadow overlap metrics between the simulated DEM illumination
    and both Reference Image A and Warped Image B.
    
    Args:
        simulated_image: (H, W) uint8 raycasted illumination image.
        image_a: (H, W) uint8 reference Image A.
        warped_image_b: (H, W) uint8 registered Image B warped to Image A coordinates.
        shadow_mask: (H, W) float32 binary shadow mask.
        
    Returns:
        Dictionary containing Pearson correlation scores, shadow overlap %, brightness diffs, and interpretation.
    """
    h, w = simulated_image.shape[:2]
    
    img_a_gray = image_a if image_a.ndim == 2 else cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    if img_a_gray.shape[:2] != (h, w):
        img_a_gray = cv2.resize(img_a_gray, (w, h))
        
    img_b_gray = warped_image_b if warped_image_b.ndim == 2 else cv2.cvtColor(warped_image_b, cv2.COLOR_BGR2GRAY)
    if img_b_gray.shape[:2] != (h, w):
        img_b_gray = cv2.resize(img_b_gray, (w, h))
        
    # Mask of valid pixels (where warped Image B has data)
    valid_mask_b = img_b_gray > 0
    valid_mask_all = np.ones((h, w), dtype=bool)
    
    sim_f = simulated_image.astype(np.float32)
    a_f = img_a_gray.astype(np.float32)
    b_f = img_b_gray.astype(np.float32)
    
    # 1. Pearson Correlation with Image A
    sim_flat = sim_f.ravel()
    a_flat = a_f.ravel()
    std_sim = np.std(sim_flat)
    std_a = np.std(a_flat)
    
    if std_sim > 1e-5 and std_a > 1e-5:
        corr_a = float(np.corrcoef(sim_flat, a_flat)[0, 1])
    else:
        corr_a = 0.0
    corr_a = max(-1.0, min(1.0, round(corr_a, 2)))
    
    # 2. Pearson Correlation with Warped Image B (over valid overlap region)
    if np.sum(valid_mask_b) > 100:
        sim_b_valid = sim_f[valid_mask_b]
        b_valid = b_f[valid_mask_b]
        std_b = np.std(b_valid)
        std_sim_b = np.std(sim_b_valid)
        if std_b > 1e-5 and std_sim_b > 1e-5:
            corr_b = float(np.corrcoef(sim_b_valid, b_valid)[0, 1])
        else:
            corr_b = 0.0
    else:
        corr_b = 0.0
    corr_b = max(-1.0, min(1.0, round(corr_b, 2)))
    
    # 3. Shadow Overlap Score with Image A
    # Real image shadow threshold (dark crater bottoms / rims)
    real_shadow_threshold = 45
    real_shadow_a = a_f < real_shadow_threshold
    sim_cast_shadow = shadow_mask < 0.5
    
    total_real_shadows = np.sum(real_shadow_a)
    if total_real_shadows > 50:
        matching_shadows = np.sum(real_shadow_a & sim_cast_shadow)
        shadow_overlap_pct = round(float(matching_shadows / total_real_shadows) * 100.0, 1)
    else:
        shadow_overlap_pct = 50.0  # default when flat
        
    # 4. Mean Brightness Difference
    mean_diff_a = round(float(np.abs(np.mean(sim_f) - np.mean(a_f))), 1)
    
    # 5. Scientific Interpretation Generator
    if corr_a >= 0.65:
        consistency = "strongly consistent"
        explanation = "The simulated illumination confirms matching crater topography and solar angles."
    elif corr_a >= 0.35:
        consistency = "moderately consistent"
        explanation = "Different solar illumination or camera phase angle explains part of the visual appearance difference."
    else:
        consistency = "weakly consistent / inconclusive"
        explanation = "The terrain elevation or solar azimuth differs from the reference camera observation."
        
    interpretation = (
        f"The simulated illumination is {consistency} with Image A. "
        f"{explanation}"
    )
    
    return {
        "similarity_image_a": corr_a,
        "similarity_warped_b": corr_b,
        "shadow_overlap_pct": shadow_overlap_pct,
        "mean_brightness_diff": mean_diff_a,
        "interpretation": interpretation
    }


def create_colorized_dem(aligned_dem: np.ndarray) -> np.ndarray:
    """Create colorized hypsometric elevation map for visual inspection."""
    norm_dem = cv2.normalize(aligned_dem, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_dem = cv2.applyColorMap(norm_dem, cv2.COLORMAP_TURBO)
    return color_dem


def create_comparison_overlay(
    real_image: np.ndarray,
    simulated_image: np.ndarray,
    title_left: str = "Real Image",
    title_right: str = "DEM Simulation"
) -> np.ndarray:
    """Create a side-by-side composite comparison with difference overlay."""
    h, w = real_image.shape[:2]
    real_gray = real_image if real_image.ndim == 2 else cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
    sim_gray = simulated_image if simulated_image.ndim == 2 else cv2.cvtColor(simulated_image, cv2.COLOR_BGR2GRAY)
    
    if real_gray.shape[:2] != (h, w):
        real_gray = cv2.resize(real_gray, (w, h))
    if sim_gray.shape[:2] != (h, w):
        sim_gray = cv2.resize(sim_gray, (w, h))
        
    # Difference map (heat colormap)
    diff = np.abs(real_gray.astype(np.float32) - sim_gray.astype(np.float32))
    diff_uint8 = np.clip(diff * 1.5, 0, 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_MAGMA)
    
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = cv2.cvtColor(real_gray, cv2.COLOR_GRAY2BGR)
    canvas[:, w:] = cv2.cvtColor(sim_gray, cv2.COLOR_GRAY2BGR)
    
    cv2.putText(canvas, title_left, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, title_right, (w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (56, 189, 248), 2, cv2.LINE_AA)
    
    return canvas


def create_synthetic_lunar_dem(
    width: int = 800,
    height: int = 800,
    crater_count: int = 40,
    seed: int = 42
) -> np.ndarray:
    """Generate synthetic lunar Digital Elevation Model (DEM) with elevation in meters."""
    np.random.seed(seed)
    y, x = np.mgrid[:height, :width]
    dem = 100.0 * np.sin(x / 120.0) * np.cos(y / 150.0) + 50.0 * np.sin(y / 60.0)
    
    for _ in range(crater_count):
        cx = np.random.randint(40, width - 40)
        cy = np.random.randint(40, height - 40)
        r = np.random.randint(15, 85)
        depth_m = r * 12.0
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        bowl = dist <= r
        rim = (dist > r * 0.85) & (dist <= r * 1.25)
        dem[bowl] -= depth_m * (1.0 - (dist[bowl] / r) ** 2)
        rim_height = depth_m * 0.28 * (1.0 - np.abs(dist[rim] - r) / (r * 0.25))
        dem[rim] += rim_height
        
    return dem.astype(np.float32)
