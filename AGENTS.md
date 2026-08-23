# AGENTS.md — 项目上下文文件

> 本文件供 AI 编程助手（如 Claude、Gemini 等）快速了解项目全貌。
> 人类开发者同样可以阅读本文件来快速上手。
> **最后更新：2026-07-26**

---

## ★ 2026-07-26 重大更新：geo2model 端到端原型

项目主体已从"分离的聚类原型 + 合成建模脚本"升级为统一的
**geo2model 包**（`geo2model/`），实现申报书全部功能闭环：

```
地质图 PNG → segment(分区) → vectorize(界线/断层) → terrain(等高线→DEM)
          → geodatabase(人机审核) → constraints(层位点/产状/断层约束)
          → model3d(GemPy+导出) → apps(虚拟钻孔/任意剖面/平切图)
          → metrics(定量评估, 合成真值 mapgen + 退化 degrade)
```

- 统一环境：`.venv-gempy`（见 `requirements-geo2model.txt`），旧 geomap_demo/.venv 已废弃
- 数据契约：`geo2model/CONTRACTS.md`（改接口先改它）
- 运行入口：`scripts/10~13_*.py` + `configs/*.json`；手册 `docs/使用说明_geo2model.md`
- 关键实测：合成基准分区准确率 99.5%+（6 场景）、三维体素一致率 base 场景
  micro 85.7%（macro 79.4%、剔基底 76.2%；剖面深部约束是决定因素
  55%→86%）；真实教学图幅 2 例全流程演示
- 注意事项：cv2 5.0 的 filter2D 需 float32 核；pandas 3.0 严格 dtype（字符串列须 object）；
  GemPy 每个 series 至少需 1 条产状；lith id = 结构框架元素顺序 1..n，基底=n+1

---

## 项目概述

**项目名称**：geo3d_project — 基于地质图的三维地质建模工具

**核心目标**：从扫描的二维地质图中提取结构化约束，最终用 GemPy 生成三维地质模型。

**当前策略**：图像提取尚不鲁棒 → **先用合成 CSV 跑通建模端**，锁定 `surface_points` / `orientations` 接口；提取端以后对接同一接口。

**用户背景**：Python 初学者（"小白"）。所有代码修改必须附带详细的、逐行级别的中文注释，解释算法原理、函数用途和库的使用方法。

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.14 |
| 图像处理 | OpenCV (`cv2`)、scikit-image (`skimage`) |
| 数值计算 | NumPy、SciPy |
| 机器学习 | scikit-learn (`KMeans`) |
| Web UI | Gradio |
| 可视化 | Matplotlib；3D 可选 pyvista |
| 三维建模 | GemPy 2026 + gempy_viewer（已集成 MVP） |
| 虚拟环境 | 图像：`scripts/geomap_demo/.venv`；建模：`.venv-gempy`（项目根） |

---

## 目录结构

```
geo3d_project/
├── AGENTS.md              ← 本文件：项目上下文（供 AI 助手阅读）
├── README.md              ← 项目简介（待补充）
├── .gitignore             ← Git 忽略规则
├── .claude/               ← Claude Code 配置
│
├── requirements-modeling.txt  ← GemPy 建模依赖
├── .venv-gempy/           ← 建模虚拟环境（勿提交）
│
├── data/
│   ├── csv/synthetic/     ← ★ 合成 GemPy 输入 CSV（已生成）
│   ├── output/synthetic/  ← ★ 建模剖面/3D 输出图
│   └── raw_maps/          ← 原始地质图（待用）
│
├── docs/
│   ├── 使用说明.md         ← 合成数据+建模运行说明
│   └── 技术路线.md         ← 提取 / 建模双路径说明
│
└── scripts/
    ├── 01_make_synthetic_data.py  ← 生成合成 surface_points / orientations
    ├── 02_gempy_model.py          ← GemPy 建模 + 出图（已跑通）
    │
    └── geomap_demo/       ← 图像聚类原型工作目录
        ├── .venv/              ← Python 虚拟环境（已安装所有依赖）
        ├── requirements.txt    ← 依赖清单
        ├── source/             ← 待处理的地质图原图（地质图1.png, 地质图2.png）
        ├── outputs/            ← 运行产物输出
        │   ├── contours/           ← KMeans 版轮廓 CSV
        │   ├── contours_meanshift/ ← MeanShift 版轮廓 CSV
        │   ├── debug/              ← 调试图像（黑线mask、聚类预览等）
        │   └── masks_clean/        ← 各地层二值 mask 图
        ├── archive_v1/         ← 归档的旧版代码（命令行交互版）
        ├── others/             ← 临时测试图片（可清理）
        ├── test/               ← 早期实验脚本
        │
        └── script/             ← ★ 当前活跃代码 ★
            ├── app.py               ← KMeans 版 Gradio Web UI（端口 7860）
            ├── app_meanshift.py     ← MeanShift 版 Gradio Web UI（端口 7861）
            ├── blacklinemiss.py     ← 命令行版聚类脚本（已被 app.py 集成替代）
            ├── show_results.py      ← 命令行版结果可视化
            ├── test_estimate_k.py   ← Auto-K 自动估算聚类数验证
            ├── test_meanshift.py    ← MeanShift 合成数据实验
            └── test_meanshift_real.py ← MeanShift 真实数据实验
```

---

## 核心算法流水线

```
                        ┌─────────────────────┐
                        │   输入：扫描地质图    │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                              │
            【KMeans 版】                  【MeanShift 版】
            app.py :7860               app_meanshift.py :7861
                    │                              │
                    │                    ┌─────────┴─────────┐
                    │                    │ pyrMeanShiftFilter │
                    │                    │ (空间+颜色平滑)    │
                    │                    └─────────┬─────────┘
                    │                              │
                    └──────────────┬───────────────┘
                                   │
                         ┌─────────┴─────────┐
                         │   高斯模糊降噪     │
                         └─────────┬─────────┘
                                   │
                         ┌─────────┴─────────┐
                         │  灰度阈值提取黑线   │
                         │  + 膨胀扩展        │
                         └─────────┬─────────┘
                                   │
                         ┌─────────┴─────────┐
                         │  转 LAB 颜色空间    │
                         │  排除黑线做 KMeans  │
                         └─────────┬─────────┘
                                   │
                         ┌─────────┴─────────┐
                         │  距离变换填补黑线   │
                         │  中值滤波平滑边界   │
                         └─────────┬─────────┘
                                   │
                         ┌─────────┴─────────┐
                         │  生成各地层 mask    │
                         │  闭运算 + 高斯平滑  │
                         └─────────┬─────────┘
                                   │
                         ┌─────────┴─────────┐
                         │  轮廓提取 + 简化    │
                         │  (Douglas-Peucker) │
                         └─────────┬─────────┘
                                   │
                         ┌─────────┴─────────┐
                         │  输出 CSV 坐标文件  │
                         │  (cluster_id,      │
                         │   contour_id,      │
                         │   point_index,     │
                         │   x, y)            │
                         └────────────────────┘
```

---

## 运行方式

### 启动 Web UI

```bash
# 进入工作目录
cd /Users/yiyi/lzu/校创/geo3d_project/scripts/geomap_demo

# KMeans 版（端口 7860）
.venv/bin/python script/app.py

# MeanShift 版（端口 7861，推荐）
.venv/bin/python script/app_meanshift.py
```

两个版本可同时运行，在不同终端窗口分别启动。

### 浏览器访问

- KMeans 版：http://localhost:7860
- MeanShift 版：http://localhost:7861

### 安装依赖（如需重建环境）

```bash
cd /Users/yiyi/lzu/校创/geo3d_project/scripts/geomap_demo
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

---

## 关键文件说明

### `script/app.py`（KMeans 版，573 行）

- **`run_clustering()`**：颜色聚类核心函数，封装了黑线排除、LAB 空间聚类、距离变换填补、中值滤波、mask 生成的完整流程
- **`run_contour_extract()`**：轮廓提取函数，使用 `cv2.findContours` + `cv2.approxPolyDP`（Douglas-Peucker 简化）提取并导出 CSV
- **Auto-K 功能**：当勾选"自动识别聚类数量"时，使用 K=25 预聚类 + LAB 距离阈值（12.0）合并相似中心的方法自动估算地层数量

### `script/app_meanshift.py`（MeanShift 版，604 行）

- 在 KMeans 版基础上增加 `cv2.pyrMeanShiftFiltering` 预处理步骤
- 三个核心参数：sp（空间窗口半径）、sr（颜色窗口半径）、maxLevel（金字塔层数）
- 额外输出一张"MeanShift 滤波后图像"，方便用户判断滤波效果

### `script/blacklinemiss.py`（命令行版，609 行）

- 最早的独立脚本版本，通过 `input()` 交互输入参数
- 已被 `app.py` 的 Gradio 版替代，保留作为算法参考和学习材料
- 注释极其详尽，每个函数/参数/算法原理均有中文解释

---

## CSV 输出格式

### 当前输出（轮廓坐标）

文件路径：`outputs/contours/all_contours.csv` 或 `outputs/contours_meanshift/all_contours_meanshift.csv`

```csv
cluster_id,contour_id,point_index,x,y
0,0,0,123,456
0,0,1,130,460
...
```

- `cluster_id`：地层编号（0, 1, 2, ...）
- `contour_id`：该地层内的第几条轮廓
- `point_index`：轮廓上的第几个点
- `x, y`：像素坐标

### GemPy 输入格式（已用合成数据验证）

路径：`data/csv/synthetic/`

**层位控制点** (`surface_points.csv`)：
```csv
X,Y,Z,formation
500.0,1200.0,50.0,Shale
```

**产状控制点** (`orientations.csv`)：
```csv
X,Y,Z,azimuth,dip,polarity,formation
550.0,1250.0,52.0,90,20,1,Shale
```

注意：GemPy 默认 surface 列名为 `formation`（不是 `surface`）。产状可用 azimuth/dip/polarity，库内会转为 G_x/G_y/G_z。

---

## 开发状态

| 模块 | 状态 | 文件 |
|------|------|------|
| 颜色聚类（KMeans） | ✅ 完成 | `script/app.py` |
| 颜色聚类（MeanShift） | ✅ 完成 | `script/app_meanshift.py` |
| 黑线排除与填补 | ✅ 完成 | 集成在上述两个文件中 |
| 边界平滑（中值滤波+高斯） | ✅ 完成 | 集成在上述两个文件中 |
| 自动估算聚类数量（Auto-K） | ✅ 完成 | 集成在上述两个文件中 |
| 轮廓提取与 CSV 导出 | ✅ 完成 | 集成在上述两个文件中 |
| Gradio Web UI | ✅ 完成 | 两个版本分别运行在 7860/7861 |
| 合成数据生成 | ✅ 完成 | `scripts/01_make_synthetic_data.py` |
| GemPy 三维建模 MVP | ✅ 完成 | `scripts/02_gempy_model.py` |
| 建模文档 | ✅ 完成 | `docs/使用说明.md`、`docs/技术路线.md` |
| 像素坐标 → 地理坐标转换 | ❌ 未开始 | — |
| Z 轴高程匹配（DEM） | ❌ 未开始 | — |
| 图例驱动/黑白线划提取 | ❌ 未开始 | 聚类路线待升级 |
| 产状从图中自动识别 | ❌ 未开始 | — |

---

## 已知问题

1. **代码重复**：`app.py` 和 `app_meanshift.py` 的 `run_contour_extract()` 函数及轮廓提取 UI 几乎完全相同，应提取到公共模块
2. **路径硬编码**：CSV 输出路径使用相对路径 `"outputs/contours"`，依赖运行时的工作目录
3. **数据目录空置**：`data/raw_maps/`、`data/csv/` 为空，实际地质图放在 `scripts/geomap_demo/source/`
4. **端口冲突**：如果上次运行未正常关闭，端口 7860/7861 可能被占用，需要 `lsof -i :7860` 检查并 `kill` 残留进程
5. **MeanShift 性能**：对大图像（>3000px）`pyrMeanShiftFiltering` 可能需要 10-30 秒，UI 上没有加载提示

---

## 编码规范

- **注释语言**：全部使用中文注释
- **注释粒度**：每个函数、每个重要参数、每个算法步骤都需要注释说明原理
- **变量命名**：使用英文蛇形命名法（`snake_case`），但注释中用中文解释含义
- **核大小**：高斯模糊、中值滤波、形态学运算的核大小必须为奇数，代码中已有自动 +1 修正逻辑
- **颜色空间**：Gradio 传入 RGB 格式，OpenCV 内部使用 BGR，聚类使用 LAB；注意转换
- **Gradio 版本**：当前使用 Gradio v6+，`theme=gr.themes.Soft()` 须在 `app.launch()` 中传入

---

## 运行建模（合成数据）

```bash
cd /Users/yiyi/lzu/校创/geo3d_project
.venv-gempy/bin/python scripts/01_make_synthetic_data.py
.venv-gempy/bin/python scripts/02_gempy_model.py
# 主结果：data/output/synthetic/model_3d.html（浏览器可交互旋转/缩放）
# 可选：--interactive 弹出桌面 3D 窗口；--no-browser 不自动开浏览器
# open data/output/synthetic/model_3d.html
```

## 下一步开发路线

```
已完成:
  (A) 地质图 → 颜色聚类 → 轮廓 CSV(x,y)     # 原型，鲁棒性不足
  (B) 合成数据 → surface_points + orientations → GemPy → 剖面/3D

下一阶段（优先）:
  锁定接口契约，加强信息提取：
    彩色：图例驱动 + 超像素 + 交互校正
    黑白：线划骨架矢量化
  输出对齐 formation 的 surface_points / orientations
  （可选）像素→工程坐标、DEM Z、断层 series
```
