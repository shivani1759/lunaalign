"""
Procedural Lunar Surface Generator for testing LunaAlign.
Generates realistic lunar surface textures containing craters, mare terrain,
ejecta blankets, and sensor noise.
Creates:
  1. Same-region pair: Image A and transformed Image B (rotation, scale, lighting, noise).
  2. Different-region pair: Distinct crater patterns and distributions.
"""

import os
import cv2
import numpy as np


def generate_lunar_terrain(seed: int, width: int = 1000, height: int = 1000, crater_density: int = 50) -> np.ndarray:
    """
    Generate synthetic lunar terrain texture with craters and micro-topography.
    """
    np.random.seed(seed)
    
    # 1. Base lunar regolith / mare texture (Perlin-like multi-scale noise)
    base = np.zeros((height, width), dtype=np.float32)
    for scale in [128, 64, 32, 16, 8]:
        noise = np.random.normal(loc=120, scale=15, size=(height // scale + 2, width // scale + 2)).astype(np.float32)
        resized_noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        base += resized_noise * (scale / 128.0)
        
    # Normalize base
    base = cv2.normalize(base, None, 80, 160, cv2.NORM_MINMAX)
    
    # 2. Add craters of varying sizes
    for _ in range(crater_density):
        cx = np.random.randint(50, width - 50)
        cy = np.random.randint(50, height - 50)
        r = np.random.randint(12, 90)
        
        # Draw crater floor (darker) and bright sunlit rim
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        
        # Rim and bowl mask
        bowl_mask = dist <= r
        rim_mask = (dist > r * 0.85) & (dist <= r * 1.2)
        
        # Sun angle shadow/highlight simulation (sun from top-left)
        angle = np.arctan2(y - cy, x - cx)
        sun_shading = np.cos(angle - np.pi * 0.75)
        
        # Subtract from bowl (depression)
        depth = (1.0 - (dist / r)) * 40.0
        base[bowl_mask] = np.clip(base[bowl_mask] - depth[bowl_mask], 20, 255)
        
        # Add bright rim on sun-facing side, shadow on opposite
        rim_effect = sun_shading * 50.0 * (1.0 - np.abs(dist - r) / (r * 0.35))
        base[rim_mask] = np.clip(base[rim_mask] + rim_effect[rim_mask], 20, 255)
        
        # Add small central peak for large craters
        if r > 45 and np.random.rand() > 0.4:
            peak_mask = dist <= r * 0.2
            base[peak_mask] = np.clip(base[peak_mask] + 35, 0, 255)
            
    # 3. Add fine sensor noise / regolith grain
    fine_noise = np.random.normal(0, 4, (height, width)).astype(np.float32)
    lunar_img = np.clip(base + fine_noise, 0, 255).astype(np.uint8)
    
    return lunar_img


def create_test_dataset(output_dir: str = "test_data"):
    """
    Create synthetic test pairs:
    1. Same-region pair:
       - region1_a.png: Reference Image A
       - region1_b.png: Target Image B (Image A rotated 12 deg, scale 0.95, brightness shift, noise)
    2. Different-region pair:
       - region2.png: Distinct lunar area with different crater distribution
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating synthetic lunar test images...")
    
    # Generate Reference Lunar Region 1
    region1_a = generate_lunar_terrain(seed=42, width=1024, height=1024, crater_density=45)
    path_1a = os.path.join(output_dir, "region1_a.png")
    cv2.imwrite(path_1a, region1_a)
    print(f"  Created {path_1a}")
    
    # Generate transformed version of Region 1 (Image B)
    # Rotation 12 degrees, scale 0.92, translation (20, -15)
    h, w = region1_a.shape
    center = (w // 2, h // 2)
    angle = 12.0
    scale = 0.92
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += 20.0
    M[1, 2] -= 15.0
    
    region1_b = cv2.warpAffine(region1_a, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    
    # Apply brightness/contrast shift and additional sensor noise
    region1_b = cv2.convertScaleAbs(region1_b, alpha=1.1, beta=-10)
    noise = np.random.normal(0, 3, region1_b.shape).astype(np.float32)
    region1_b = np.clip(region1_b.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    path_1b = os.path.join(output_dir, "region1_b.png")
    cv2.imwrite(path_1b, region1_b)
    print(f"  Created {path_1b} (transformed overlapping region)")
    
    # Generate completely different Lunar Region 2
    region2 = generate_lunar_terrain(seed=999, width=1024, height=1024, crater_density=60)
    path_2 = os.path.join(output_dir, "region2.png")
    cv2.imwrite(path_2, region2)
    print(f"  Created {path_2} (different lunar region)")
    
    print("Test dataset created successfully.")


if __name__ == "__main__":
    create_test_dataset()
