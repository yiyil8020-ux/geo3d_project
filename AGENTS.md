# AGENTS.md — 项目开发与技术上下文

> 本文件供 AI 编程助手与开发者快速了解项目全貌。
> **系统版本：geo2model 端到端原型**

---

## 1. 项目核心概述

**项目名称**：`geo3d_project` (geo2model) — 基于平面地质图的智能化三维地质建模系统

**核心目标**：从扫描的二维 JPG/PNG 平面地质图中提取结构化约束（地层边界、断层迹线、等高线 DEM、图面/分窗三点法产状、剖面图深部约束），对接 GemPy 2026 隐式协同克里金标量场算法构建三维地质体，并扩展虚拟钻孔、任意剖面及平切图等工程分析功能。

**运行环境**：macOS / Linux，Python 3.14（统一虚拟环境 `.venv-gempy`，见 `requirements-geo2model.txt`）。

---

## 2. 核心架构与模块分工

```
地质图 PNG → segment.py(分区分割) → vectorize.py(接触线/断层) → terrain.py(等高线→DEM)
           → geodatabase.py(交互审核) → constraints.py(层位点/产状/断层约束) + sectionreg.py(剖面配准)
           → model3d.py(GemPy 隐式建模与多格式导出) → apps.py(虚拟钻孔/任意剖面/平切图)
           → metrics.py(定量评测, mapgen.py 合成基准 + degrade.py 扫描退化)
```

| 模块文件 | 主要职能与关键算法 |
|---|---|
| `geo2model/segment.py` | 图像预处理（双边滤波 + 非局部均值去噪 + 残差式两遍光照校正）；暗线与水系剥离；CIELAB 空间 KMeans(K=22) 过聚类 + $\Delta E < 5.0$ 层次合并；距离变换回填。 |
| `geo2model/vectorize.py` | 单元对共享边界提取与 Skeleton 骨架追踪折线化（接触线）；粗线距离变换核心提取 + 几何过滤（断层候选）。 |
| `geo2model/terrain.py` | 虚线（短划线方向兼容）与点线（圆点近邻）等高线图元追踪成链；高程赋值与 `scipy.interpolate.griddata` 二维连续 DEM 重建。 |
| `geo2model/geodatabase.py` | 人机交互三级检查点（`units.csv` 审核表、`faults.json` 断层确认、`contours.json` 等高线赋值）；生成标准数据表。 |
| `geo2model/constraints.py` | 接触线重采样层位点；接触线分窗三点法局部产状拟合（三关质控）；图面产状读数通道；断层深部投影面；系列兜底机制。 |
| `geo2model/sectionreg.py` | 剖面图像素比例映射空间配准（横轴线性插值 + 高程轴变换），将剖面像素拾取点精确转为世界坐标（往返误差 $< 10^{-6}\text{m}$）。 |
| `geo2model/model3d.py` | GemPy 2026 协同克里金隐式建模；断层 Series 优先切割；不整合多 Series 堆叠；挂接 DEM 地形（Topography）；导出交互 HTML、OBJ、glTF、VTK 及重建地质图。 |
| `geo2model/apps.py` | 虚拟钻孔（任意点地表裁剪柱状图与 CSV 层段表）；任意两点地质剖面（带地形轮廓填色）；任意标高水平切片。 |
| `geo2model/mapgen.py` | 6 场景解析合成地质图基准（像素级解析真值、剖面图及全套制图要素）。 |
| `geo2model/metrics.py` | 像素分类准确率（匈牙利匹配）、边界 F1（3px 容差）、DEM MAE、三维体素一致率（micro/macro/剔基底）。 |
| `geo2model/degrade.py` | 7 种扫描退化（高斯噪声、高斯模糊、JPEG 压缩、褪色、光照不均、断线、污渍）× 4 档强度生成。 |

---

## 3. 运行指南与常用命令

### 3.1 图形界面运行
```bash
# 双击 启动界面.command 或在终端运行：
.venv-gempy/bin/python app/geo2model_app.py
```
在浏览器打开 `http://127.0.0.1:7860`，支持案例成果演示、新图交互式全流程建模与工程分析查询。

### 3.2 命令行脚本流
```bash
PY=.venv-gempy/bin/python

# 1. 生成 6 个合成基准场景
$PY scripts/10_gen_synthetic_map.py

# 2. 单案例端到端运行 (base场景 + 剖面深部约束)
$PY scripts/11_run_pipeline.py configs/synth_base_deep.json

# 3. 批量评估 6 场景指标
$PY scripts/12_evaluate_all.py

# 4. 运行 28 组扫描退化鲁棒性扫描
$PY scripts/13_robustness.py

# 5. 真实教学图幅全流程运行
$PY scripts/11_run_pipeline.py configs/real_map1.json
$PY scripts/11_run_pipeline.py configs/real_map2.json
```

---

## 4. 关键数据契约与接口规则

详见 `geo2model/CONTRACTS.md`：
1. **GemPy surface_points.csv**：列必须为 `X, Y, Z, formation`（代表单元底界面出露点）；
2. **GemPy orientations.csv**：列必须为 `X, Y, Z, azimuth, dip, polarity, formation`；
3. **GemPy 2026 规则**：每个 Series 必须至少包含 1 条产状，否则隐式求解器会报错；系统已内置系列级近水平兜底机制。
4. **代码规范**：所有新增修改需保持清晰的中文注释与类型安全。
