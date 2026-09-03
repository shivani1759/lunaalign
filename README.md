# LunaAlign — Lunar Terrain Registration & Correspondence Analysis

An aerospace-grade lunar surface image registration and overlap verification system. 
LunaAlign determines whether two lunar surface images depict the same terrain region across varying illumination angles, shadow orientations, and sensor geometries.

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.10+
- `pip` package manager

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py --port 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in any modern web browser.

---

## ☁️ Public Cloud Deployment (Render)

This project is pre-configured for 1-click or standard Web Service deployment on **Render.com**.

### Option A: Standard Render Web Service
1. Push this repository to GitHub / GitLab.
2. Log in to [Render Dashboard](https://dashboard.render.com/) and click **New +** → **Web Service**.
3. Connect your repository.
4. Set the following build settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
   - **Plan:** Free
5. Click **Create Web Service**. Render will automatically supply the `PORT` environment variable, bind to `0.0.0.0`, and issue a public HTTPS URL:
   `https://lunaalign-xxxx.onrender.com`

### Option B: Blueprint Deployment (`render.yaml`)
1. Connect your repository to Render Blueprints.
2. Render reads `render.yaml` and auto-provisions the web service.

---

## 🔬 Core Algorithm Pipeline

```text
Image A + Image B
        ↓
Illumination-aware Preprocessing (CLAHE + Sobel Gradient Normalization)
        ↓
Multi-Scale Feature Matching (SIFT / ORB + FLANN Ratio Test)
        ↓
RANSAC Geometric Verification (Homography Estimation)
        ↓
Sub-Pixel Refinement (Quadratic Peak Interpolation on Normalized Cross-Correlation)
        ↓
Multi-Gate Confidence Assessment (7 Scientific Verification Gates)
        ↓
SAME REGION / DIFFERENT REGION / INSUFFICIENT EVIDENCE (0–100 Trust Score)
        ↓
Warped Registered Image + Interactive Comparison + DEM Solar Raycaster
```

---

## 🛠️ CLI Usage
Run registration from the command line:
```bash
python lunaalign.py path/to/image_a.png path/to/image_b.png --output results/
```
