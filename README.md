# 3D Geological Modeling from Planar Maps

**Intelligent 3D Geological Modeling from Planar Maps** · `geo3d_project`

From planar geological maps toward open, reproducible **3D geological models** (GemPy).

> **Tagline:** From a flat geological map to a 3D earth model — open, automatic, and ready for GemPy.

---

## Status (2026-07)

| Module | Status | Notes |
|--------|--------|--------|
| Map color clustering (KMeans / MeanShift) | Prototype | Gradio UI; limited on noisy / B&W maps |
| Synthetic GemPy inputs (folds + fault) | Done | `scripts/01_make_synthetic_data.py` |
| GemPy 3D modeling + topography | Done | `scripts/02_gempy_model.py` |
| Interactive 3D (browser) | Done | `data/output/synthetic/model_3d.html` |
| Real map → GemPy CSV | In progress | Legend-driven / line-vector paths planned |

**Modeling is built with GemPy.** PyVista is used for visualization and interactive HTML export only.

---

## Quick start: fold + fault + topography (modeling)

```bash
git clone https://github.com/yiyi019/geo3d_project.git
cd geo3d_project

python3 -m venv .venv-gempy
source .venv-gempy/bin/activate   # Windows: .venv-gempy\Scripts\activate
pip install -r requirements-modeling.txt

python scripts/01_make_synthetic_data.py
python scripts/02_gempy_model.py
# macOS:
open data/output/synthetic/model_3d.html
```

### Outputs

| File | Description |
|------|-------------|
| **`data/output/synthetic/model_3d.html`** | **Interactive 3D** (drag to rotate, scroll to zoom) |
| `section_y_mid.png` | X–Z section (best view of folds + fault) |
| `section_y_topo.png` | Section with topography |
| `section_x_mid.png` | Y–Z section |
| `model_3d.png` | Static preview |

Synthetic scenario: cylindrical **folds**, east-dipping **normal fault** (`Main_Fault`), multi-scale **terrain**.

---

## Quick start: map clustering UI (extraction prototype)

```bash
cd scripts/geomap_demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# MeanShift version (recommended)
python script/app_meanshift.py   # http://localhost:7861

# KMeans version
python script/app.py             # http://localhost:7860
```

---

## Data contract (map extractors → GemPy)

Any future extractor should output:

```csv
# surface_points.csv
X,Y,Z,formation

# orientations.csv
X,Y,Z,azimuth,dip,polarity,formation
```

Then reuse `scripts/02_gempy_model.py` (adjust stack / fault mapping as needed).

---

## Repository layout

```text
geo3d_project/
├── README.md
├── AGENTS.md                      # Project context for developers / AI
├── requirements-modeling.txt      # GemPy environment
├── data/
│   ├── csv/synthetic/             # Generated surface points & orientations
│   └── output/synthetic/          # Figures + interactive HTML
├── docs/
│   ├── 使用说明.md
│   ├── 技术路线.md
│   └── Project_Story.md
└── scripts/
    ├── 01_make_synthetic_data.py  # Folds + fault synthetic data
    ├── 02_gempy_model.py          # GemPy model + PyVista export
    └── geomap_demo/               # Gradio color-clustering prototype
```

---

## Tech stack

**Python** · **GemPy** · **PyVista** · **OpenCV** · **Gradio** · **NumPy** · **SciPy** · **scikit-learn** · **Matplotlib** · **pandas** · **scikit-image**

---

## Docs

- [使用说明](docs/使用说明.md) — runbook  
- [技术路线](docs/技术路线.md) — dual path (extract vs model)  
- [Project Story](docs/Project_Story.md) — narrative for submissions  
- [AGENTS.md](AGENTS.md) — full project context  

---

## License / note

University innovation project (校创). Research code; synthetic demo data are **not** real field measurements.
