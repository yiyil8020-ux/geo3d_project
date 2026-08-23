# geo2model: Intelligent 3D Geological Modeling from Planar Maps

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![GemPy 2026](https://img.shields.io/badge/GemPy-2026.0.3-emerald.svg)](https://www.gempy.org/)
[![PyVista](https://img.shields.io/badge/PyVista-3D_Interactive-orange.svg)](https://docs.pyvista.org/)
[![Open Source](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**From 2D planar geological maps to open, reproducible, and interactive 3D implicit geological models.**

> **Core Philosophy:** Transform scanned 2D raster maps (JPG/PNG) into full-fledged 3D subsurface models using open-source scientific tools (OpenCV, scikit-image, GemPy, PyVista), featuring human-in-the-loop validation, cross-section registration, and practical engineering analysis.

---

## 🌟 Key Highlights & Capabilities

- 🗺️ **Raster Map Vectorization**: Non-local means denoising, residual illumination correction, CIELAB over-clustering ($K=22$) with adaptive $\Delta E < 5.0$ merge, dark line skeletonization, and contour line tracing.
- 📐 **Geological Rules & Spatial Registration**:
  - Sliding-window 3-point attitude estimation with strict 3-tier quality control.
  - Pixel-ratio cross-section spatial registration (`sectionreg.py`) mapping 2D section picks to 3D world coordinates (round-trip error $< 10^{-6}\text{m}$).
- 🧊 **Implicit 3D Modeling (GemPy 2026)**:
  - Co-Kriging potential field implicit interpolation.
  - Multi-series fault cutting, erosional unconformities, and topography mesh integration.
- 📊 **Quantitative Benchmark (6 Geological Scenarios)**:
  - Pixel-level analytical ground truth benchmark (`mapgen.py`).
  - **99.5%–99.7%** stratum classification accuracy across all scenarios.
  - **85.7%** 3D voxel consistency (micro) with cross-section deep constraints.
- 🛠️ **Engineering Analysis Applications**:
  - **Virtual Boreholes**: Subsurface layer stratigraphic columns with CSV interval tables and automatic surface trimming.
  - **Arbitrary Cross-Sections**: 2D cross-sections between any two coordinates with terrain profile and lithology coloring.
  - **Horizontal Slices (Flat Cuts)**: Horizontal lithological slices at arbitrary depths.
- 🖥️ **Interactive Web UI & Multi-format Export**:
  - One-click launcher (`启动界面.command` / Gradio app) for review and modeling.
  - Exports interactive standalone HTML (360° rotate/zoom in browser), Wavefront OBJ, glTF, and VTK (`.vti`/`.vtm`).

---

## 🏗️ System Architecture

```
                    Input: Raster Geological Map (JPG/PNG) + Case JSON
                                          │
   ┌─────────────────────── Stage 1: Vectorization ───────────────────────┐
   │  [segment.py]    Preprocessing (NL-Means + Illumination) → CIELAB Clustering  │
   │  [vectorize.py]  Contact Line Tracing & Fault Candidate Extraction          │
   │  [terrain.py]    Contour Tracing (Dashed / Dotted) → 2D DEM Interpolation    │
   └──────────────────────────────────┬───────────────────────────────────┘
                                      │
   ┌──────────────────── Stage 2: Database & Registration ────────────────┐
   │  [geodatabase.py] Human-in-the-loop Review Checkpoints (Units / Faults / DEM)│
   │  [constraints.py] 3-Point Dip/Azimuth Estimation + Dip Symbol Readings       │
   │  [sectionreg.py]  Cross-Section Pixel-to-World Spatial Registration          │
   └──────────────────────────────────┬───────────────────────────────────┘
                                      │
   ┌───────────────────── Stage 3: Implicit 3D Modeling ──────────────────┐
   │  [model3d.py]     GemPy 2026 Co-Kriging Interpolation (Faults + Unconformity)│
   │                   Exports: Interactive HTML / OBJ / glTF / VTK / Slices      │
   └──────────────────────────────────┬───────────────────────────────────┘
                                      │
   ┌───────────────────── Stage 4: Engineering Applications ───────────────┐
   │  [apps.py]        Virtual Borehole Log / Arbitrary Cross-Section / Slices    │
   └──────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/yiyi019/geo3d_project.git
cd geo3d_project

python3 -m venv .venv-gempy
source .venv-gempy/bin/activate   # Windows: .venv-gempy\Scripts\activate
pip install -r requirements-geo2model.txt
```

### 2. Launch Interactive Web GUI

Double-click **`启动界面.command`** on macOS, or run:

```bash
python app/geo2model_app.py
```
Open `http://127.0.0.1:7860` in your web browser to upload maps, review checkpoints, build 3D models, and query virtual boreholes.

---

### 3. Command-Line End-to-End Pipeline

Run a synthetic benchmark case (Folds + Fault + DEM + Cross-sections):

```bash
# 1. Generate synthetic benchmark dataset (6 scenarios)
python scripts/10_gen_synthetic_map.py

# 2. Run end-to-end pipeline on base scenario with deep section constraints
python scripts/11_run_pipeline.py configs/synth_base_deep.json

# 3. View interactive 3D model in your browser
open data/output/geo2model/synth_base_deep/model/model_3d.html  # macOS
```

---

## 📈 Benchmark & Evaluation Results

Quantitative evaluation across 6 synthetic geological scenarios:

| Scenario | Geological Features | Stratum Accuracy | Macro IoU | Boundary F1 (3px) | DEM MAE | 3D Voxel Agreement (micro / excl. basement) |
|---|---|---|---|---|---|---|
| **base** | Folds + Normal Fault + Terrain | **99.67%** | 96.78% | 99.42% | 5.0 m | **85.7% / 76.2%** |
| **gentle_nofault** | Gentle Folds (No Fault) | **99.72%** | 99.14% | 99.99% | 5.2 m | **94.1% / 89.2%** |
| **steep_narrow** | Steep Tight Folds + 190m Throw | **99.63%** | 82.36% | 99.80% | 7.1 m | 57.9% / 35.3% |
| **similar_colors** | Adjacent strata with $\Delta E \approx 6.1$ | **99.69%** | 96.82% | 99.77% | 5.0 m | **85.7% / 76.3%** |
| **high_relief** | High Relief Terrain ($\sim 400\text{m}$) | **99.53%** | 97.62% | 99.67% | 4.4 m | **86.9%** / 77.9% |
| **flat_dem** | Flat Topography | **99.72%** | 99.61% | 96.18% | 0.0 m | **83.6% / 73.0%** |

### 🔬 Key Finding: Cross-Section Constraints are Essential
* **Surface data only**: 3D voxel consistency is **54.7%** (excl. basement 21.3%).
* **+ 2 Cross-section registrations**: 3D voxel consistency jumps to **85.7%** (excl. basement **76.2%**, $+54.9\%$).

---

## 📁 Repository Structure

```text
geo3d_project/
├── app/
│   └── geo2model_app.py            # Gradio Web GUI application
├── geo2model/                      # Core Python algorithm package
│   ├── segment.py                  # Stratum segmentation & pre-processing
│   ├── vectorize.py                # Boundary skeletonization & fault extraction
│   ├── terrain.py                  # Contour chaining & DEM interpolation
│   ├── geodatabase.py              # Review table & database builder
│   ├── constraints.py              # 3-point dip estimator & virtual points
│   ├── sectionreg.py               # Cross-section spatial registration
│   ├── model3d.py                  # GemPy 2026 implicit modeling & export
│   ├── apps.py                     # Virtual borehole, section & slice apps
│   ├── metrics.py                  # Quantitative evaluation metrics
│   ├── degrade.py                  # 7 scan degradation models
│   └── mapgen.py                   # 6-scene analytical benchmark generator
├── configs/                        # JSON presets for synthetic and real maps
├── scripts/                        # Pipeline execution and evaluation runners
│   ├── 10_gen_synthetic_map.py     # Generate synthetic benchmark
│   ├── 11_run_pipeline.py          # End-to-end pipeline runner
│   ├── 12_evaluate_all.py          # Batch benchmark evaluator
│   ├── 13_robustness.py            # 7 degradations x 4 levels evaluation
│   ├── 14_surface_agreement.py     # Real-map map-fit evaluator
│   └── 15_ablation_sweep.py        # Ablation study parameter sweep
├── docs/                           # Documentation and user manuals
├── requirements-geo2model.txt      # Python dependencies
└── 启动界面.command                 # macOS one-click launcher
```

---

## 🛠️ Tech Stack

* **Language**: Python 3.14
* **Computer Vision**: OpenCV (`cv2`), scikit-image (`skimage`)
* **Scientific Computing**: NumPy, SciPy, pandas, scikit-learn
* **3D Implicit Modeling**: GemPy 2026 (`gempy`, `gempy_engine`, `gempy_viewer`)
* **3D Visualization & Export**: PyVista, VTK, Trame
* **UI**: Gradio

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
