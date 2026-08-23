# 3D Geological Modeling from Planar Maps

**Intelligent 3D Geological Modeling from Planar Maps** · `geo3d_project`

From planar geological maps toward open, reproducible **3D geological models** (GemPy).

> **Tagline:** From a flat geological map to a 3D earth model — open, automatic, and ready for GemPy.

---

## Status (2026-07)

**★ 2026-07-26 起，项目主体为 `geo2model/` 端到端原型**（地质图 → 矢量化 → 数据库 →
GemPy 三维模型 → 虚拟钻孔/剖面/平切应用 → 定量评估），覆盖申报书全部功能点。
文档：`docs/使用说明_geo2model.md`；技术报告见 `校创/技术报告/`。

| Module | Status | Notes |
|--------|--------|--------|
| **geo2model 端到端流水线** | **Done** | `scripts/10-13_*.py` + `configs/*.json` |
| ├ 合成地质图基准（含像素真值） | Done | `geo2model/mapgen.py`，6 内置场景 |
| ├ 分区分割（LAB 聚类+预处理） | Done | 清洁图像素准确率 99.7% |
| ├ 界线/断层/等高线矢量化 | Done | `vectorize.py` / `terrain.py`（虚线+点线） |
| ├ 数据库构建+人机交互审核 | Done | `geodatabase.py` / `constraints.py`（不整合系列） |
| ├ GemPy 建模+多格式导出 | Done | `model3d.py`（HTML/VTK/OBJ/glTF） |
| ├ 虚拟钻孔/任意剖面/平切图 | Done | `apps.py` |
| ├ 剖面深部约束配准 | Done | `sectionreg.py`（体素一致率 55%→86%，剔基底 22%→76%） |
| └ 鲁棒性测试（7 退化×4 档） | Done | `scripts/13_robustness.py` |
| Map color clustering (KMeans / MeanShift) | Legacy | 旧 Gradio 原型，`scripts/geomap_demo/` |
| Synthetic GemPy inputs (folds + fault) | Done | `scripts/01_make_synthetic_data.py` |
| GemPy 3D modeling + topography | Done | `scripts/02_gempy_model.py` |

**Modeling is built with GemPy.** PyVista is used for visualization and interactive HTML export only.

---

## How Codex & GPT-5.6 were used (AI-assisted development)

This project was built with **AI pair programming**. We used **OpenAI Codex** and **GPT-5.6** as coding / research copilots throughout the pipeline (with a human always reviewing geology logic and running experiments). Highlights:

### 1. Architecture & research synthesis
- Distilled literature (map digitization, DIGMAPPER-style modular flows, map2loop data contracts, interactive boundary extraction) into a **dual-path design**:
  - Path A: raster map → structured CSV (extraction prototype)
  - Path B: CSV → **GemPy** 3D model (modeling backend first)
- Codex/GPT-5.6 helped turn “color clustering alone is brittle” into a concrete plan: synthetic constraints first, then harder map extraction later.

### 2. Implementation (agentic coding)
- **Codex-style agent workflows** for multi-file edits: synthetic data generator, GemPy runner, Gradio apps, docs, and repo hygiene.
- **GPT-5.6** for:
  - Designing fold / fault / topography math and GemPy `ImporterHelper` / `map_stack_to_surfaces` / `set_is_fault` wiring
  - Debugging dependency & API issues (GemPy 2026, PyVista `export_html`, trame)
  - Producing bilingual docs, Project Story, and submission media (gallery 3:2 frames, demo slideshow video script)
- Human role: domain decisions (stratigraphic order, fault throw, evaluation criteria), running scripts, and verifying interactive `model_3d.html`.

### 3. What AI did *not* replace
- Geological validity checks and competition honesty (synthetic demo ≠ real map-derived model)
- Choosing open stack: **GemPy + PyVista + OpenCV + Gradio** (no commercial modeler required)

### 4. Reproducible AI-era artifacts in-repo
| Artifact | Role |
|----------|------|
| `AGENTS.md` | Project context file so coding agents stay aligned |
| `scripts/01_make_synthetic_data.py` | AI-assisted synthetic folds + fault |
| `scripts/02_gempy_model.py` | AI-assisted GemPy + interactive HTML export |
| `docs/Project_Story.md` | Narrative for judges |
| `data/output/synthetic/model_3d.html` | Judge-facing interactive 3D result |

> **For judges:** clone the repo, run the Quick start below (~few minutes on CPU). No cloud credentials required.

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
