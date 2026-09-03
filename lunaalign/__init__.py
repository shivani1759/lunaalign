"""
LunaAlign: Lunar Image Registration & Overlap Verification with DEM Solar Raycaster.
"""

__version__ = "0.2.0"

from .config import LunaAlignConfig
from .pipeline import LunaAlignPipeline
from .preprocessor import preprocess_image, load_image
from .feature_matcher import detect_and_match_features
from .multi_scale_matcher import multi_scale_detect_and_match
from .illumination_normalizer import compute_phase_congruency_features
from .dem_raycaster import (
    load_dem,
    align_dem_to_reference_grid,
    compute_surface_normals,
    sun_vector_from_angles,
    compute_lambertian_illumination,
    compute_terrain_shadows,
    render_raycaster_result,
    compare_with_registered_images,
    create_colorized_dem,
    create_comparison_overlay,
    create_synthetic_lunar_dem
)
from .subpixel_refiner import refine_matches_subpixel
from .geometric_verifier import verify_geometry_ransac, compute_rmse
from .spatial_analyzer import check_spatial_distribution
from .decision_engine import make_decision
from .visualizer import save_visualizations, format_terminal_card

# Convenience wrapper
def generate_solar_hillshade(dem, sun_azimuth_deg=45.0, sun_elevation_deg=30.0, cell_size_m=10.0):
    normals = compute_surface_normals(dem, cell_size_m)
    sun_vec = sun_vector_from_angles(sun_azimuth_deg, sun_elevation_deg)
    illum = compute_lambertian_illumination(normals, sun_vec)
    shadows = compute_terrain_shadows(dem, sun_vec, cell_size_m)
    return render_raycaster_result(illum, shadows)

__all__ = [
    "LunaAlignConfig",
    "LunaAlignPipeline",
    "preprocess_image",
    "load_image",
    "detect_and_match_features",
    "multi_scale_detect_and_match",
    "compute_phase_congruency_features",
    "load_dem",
    "align_dem_to_reference_grid",
    "compute_surface_normals",
    "sun_vector_from_angles",
    "compute_lambertian_illumination",
    "compute_terrain_shadows",
    "render_raycaster_result",
    "compare_with_registered_images",
    "create_colorized_dem",
    "create_comparison_overlay",
    "create_synthetic_lunar_dem",
    "generate_solar_hillshade",
    "refine_matches_subpixel",
    "verify_geometry_ransac",
    "compute_rmse",
    "check_spatial_distribution",
    "make_decision",
    "save_visualizations",
    "format_terminal_card",
]
