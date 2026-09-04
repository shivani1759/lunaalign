LunaAlign

"Same Moon, Bigger Insights.

"Adaptive Cross-Sensor Lunar Image Correspondence & Registration

LunaAlign is a software framework for finding and validating
correspondences between lunar images captured by different sensors, with
a focus on Chandrayaan-2 OHRC, TMC-2 and IIRS imagery.

The system addresses the major challenges of cross-sensor lunar image
registration:

Large differences in spatial scale and resolution

Changes in illumination caused by Sun angle and shadows

Differences in sensor appearance and image characteristics

Repetitive lunar terrain that can produce false matches

The need for reliable, measurable registration quality

Sub-pixel-accurate alignment as a future refinement stage

Problem Statement

SIH26166 --- Multi-modal, Sun angle and scale invariant image
correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)

Image registration aligns a source/moving image with a
reference/fixed image so that the same physical lunar terrain is
represented in a common spatial coordinate system.

LunaAlign aims to make this process more reliable across different
Chandrayaan-2 sensors and observation conditions.

Our Approach

LunaAlign follows a modular, reliability-driven pipeline:

Lunar Images
     │
     ▼
Preprocessing
(Normalization / Enhancement)
     │
     ▼
Multi-Scale Feature Extraction
(SIFT / ORB)
     │
     ▼
Feature Matching
(Lowe's Ratio Test)
     │
     ▼
Geometric Verification
(RANSAC)
     │
     ▼
Registration
(Homography / Affine)
     │
     ▼
Reliability Analysis
(Inlier Ratio + RMSE + Spatial Coverage)
     │
     ▼
Composite Correspondence Score
     │
     ▼
Aligned Image + Quality Metrics

Key Features

1. Adaptive Cross-Sensor Registration

Designed to handle differences between OHRC, TMC-2 and IIRS
observations, including scale, illumination and sensor characteristics.

2. Multi-Scale Feature Matching

Uses feature-based methods such as SIFT/ORB to establish correspondences
despite large differences in image scale.

3. Robust Geometric Verification

RANSAC removes geometrically inconsistent matches and estimates the
transformation required for registration.

4. Spatial Reliability Gating

Checks whether reliable matches are distributed across the scene rather
than being concentrated in one small region.

5. Composite Correspondence Score

Combines multiple indicators such as:

Inlier ratio

RMSE

Spatial coverage

Match quality

This produces an interpretable registration/correspondence score for an
image pair.

The score is a confidence/quality measure, not a calibrated
probability unless separately calibrated.

6. Precision-Aware Registration

The framework can incorporate local refinement for improved coordinate
precision and evaluation using registration error.

7. Optional Physical Validation

A future/advanced extension can use DEM and Sun-geometry information to
independently check whether terrain illumination and shadow patterns are
physically consistent.

Data Sources

The project is designed around publicly available lunar mission data,
particularly:

OHRC --- high-resolution optical imagery

TMC-2 --- wider-context, lower-resolution optical imagery

IIRS --- hyperspectral/spectral observations

Other compatible lunar reference datasets where available

Primary Chandrayaan-2 data can be obtained through ISRO / IISDC
PRADAN.

Evaluation Metrics

LunaAlign evaluates registration using measurable indicators:

Metric                  Purpose                     Better Result

Number of Matches       Measures detected           Higher, if reliable
correspondences

Inlier Count            Matches consistent with the Higher
estimated geometry

Inlier Ratio            Fraction of geometrically   Higher
valid matches

RMSE                    Registration/reprojection   Lower
error

Spatial Coverage        Distribution of matches     Wider / more uniform
across the scene

Correspondence Score    Combined quality assessment Higher

RMSE

RMSE measures the average distance between the observed matched points
and their transformed locations.

A lower RMSE generally indicates better geometric alignment.

Input Definition

Reference / Fixed Image

The target image whose coordinate system is retained.

Source / Moving Image

The image that is geometrically transformed to align with the reference
image.

Example:

Reference Image  → Fixed
Source Image     → Moving
                      │
                      ▼
                 Transformation
                      │
                      ▼
               Registered Image

Technology Stack

Core Development

Python

OpenCV

NumPy

SciPy

Computer Vision

SIFT / ORB

Feature matching

Lowe's ratio test

RANSAC

Homography / Affine transformation

Geospatial / Scientific Tools

Rasterio

QGIS

Matplotlib

Development & Testing

Jupyter Notebook

Future / Extension

PyTorch

Learned feature matching methods such as LoFTR/SuperGlue





Getting Started

1. Clone the repository

git clone <YOUR-REPOSITORY-URL>
cd LunaAlign

2. Create a virtual environment

python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

Typical dependencies include:

opencv-python
numpy
scipy
matplotlib
rasterio
scikit-learn
jupyter

4. Add lunar images

Place compatible reference and source images inside the appropriate data
directory.

Example:

data/
├── ohrc/
│   └── reference.tif
└── tmc2/
    └── source.tif

5. Run the registration pipeline

The exact command depends on the implementation. A typical workflow is:

python main.py

Expected Output

For each image pair, LunaAlign can provide:

Feature correspondence visualization

RANSAC inlier matches

Estimated transformation

Registered/aligned image

Inlier ratio

RMSE

Spatial coverage

Composite correspondence score

Same-region / different-region decision

Example conceptual output:

Input:
OHRC + TMC-2

Matches        : 142
Inlier Ratio   : 0.76
RMSE           : 2.31 px
Spatial Cover  : Good
Score          : 0.87

Result:
SAME REGION

The numerical values above are illustrative; actual values depend on the
image pair and implementation.

Future Enhancements

Improved illumination-aware preprocessing

Learned correspondence models such as LoFTR

Coarse-to-fine registration

Sub-pixel local refinement

Stronger spatial reliability models

IIRS-specific spectral/band handling

DEM + Sun-geometry validation

GPU acceleration for large datasets

Support for additional lunar missions and reference datasets

Current Prototype Status

The current prototype demonstrates OHRC ↔ TMC-2 image registration.

The architecture is designed to be modular so that additional sensor
combinations, including IIRS, can be incorporated and evaluated
without replacing the complete system.

Applications

LunaAlign can support:

Lunar surface mapping

Multi-sensor data fusion

Terrain and geological analysis

Change/comparison studies

Scientific image analysis

Mission-data interpretation

Future lunar exploration research

References

Indian Space Research Organisation (ISRO) --- Chandrayaan-2 mission
and science data

Indian Space Science Data Centre (ISSDC) / PRADAN --- Chandrayaan-2
datasets

MoonMetaSync --- Lunar Image Registration Analysis

SIFT-based registration studies for Chandrayaan-2 IIRS imagery

LoFTR --- Detector-Free Local Feature Matching with Transformers

SuperGlue --- Learning Feature Matching with Graph Neural Networks

High-precision lunar image registration research using dense
matching, RANSAC and local geometric correction

Team

B Sanjai & Sanjay B (PPT Design)
R Ashwanth & Vishaak K K (Built Prototype)
S Shivani Sree & Rithaaesh H (Research Work)
