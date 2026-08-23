# Project Story: 3D Geological Modeling from Planar Maps

## Inspiration

Geological maps are everywhere, but **3D geological models are hard to get**. Paper and scanned maps capture decades of field work, yet turning them into subsurface 3D models still often requires days of manual digitizing and expensive commercial software.

The `geo2model` project started from a simple question:

> Can we go from a **2D planar raster geological map** to an **open, usable 3D geological model** using modern open-source scientific tools?

We wanted to build an accessible, automated bridge so that students, field engineers, and researchers can unlock the 3D value stored in 2D geological map archives.

---

## What We Built

We designed and implemented the **`geo2model` end-to-end prototype pipeline**:

```text
2D Raster Map (PNG/JPG)
    ↓
Image Vectorization (Segmentation, Skeletonization, Contours to DEM)
    ↓
Database & Constraints (3-Point Dip Estimation, Cross-Section Registration)
    ↓
Implicit 3D Modeling (GemPy 2026 Co-Kriging, Faults & Unconformities)
    ↓
Engineering Applications (Virtual Boreholes, Arbitrary Sections, Horizontal Slices)
```

### Highlights of the System:
1. **Robust Vectorization**: Overcomes scanning noise and illumination gradients via dual-stage filtering (Bilateral + NL-Means) and CIELAB over-clustering with adaptive $\Delta E$ color merging.
2. **Deep Constraints from Sections**: Introduces pixel-ratio spatial registration (`sectionreg.py`) to map 2D cross-section picks into 3D world coordinates, reducing subsurface ambiguity.
3. **Rigorous Benchmarking**: Developed `mapgen.py` to generate 6 synthetic geological scenarios with analytical ground truth, enabling rigorous pixel-level and 3D voxel-level accuracy measurement.
4. **Engineering Analytics**: Built direct query tools for virtual borehole logging with CSV interval tables, arbitrary 2D cross-sections with topography, and depth slices.

---

## Key Results & Discoveries

* **Stratum Segmentation Accuracy**: Reached **99.53%–99.72%** pixel accuracy across 6 benchmark scenarios.
* **Deep Constraint Impact**: Quantitatively demonstrated that adding cross-section registration constraints boosts 3D voxel agreement from **54.7%** to **85.7%** (excluding basement: 21.7% to **76.2%**).
* **Scan Degradation Robustness**: Evaluated 28 combinations (7 degradation types $\times$ 4 intensity levels), proving $>92.4\%$ accuracy under typical scanning conditions.

---

## Tech Stack & Open Science

* **Core Language**: Python 3.14
* **Computer Vision**: OpenCV, scikit-image, scikit-learn
* **3D Implicit Modeling**: GemPy 2026 (`gempy_engine`, `gempy_viewer`)
* **3D Visualization**: PyVista, VTK, Trame
* **UI**: Gradio

Our goal is to make 3D geological modeling open, reproducible, and accessible to everyone without proprietary licenses.
