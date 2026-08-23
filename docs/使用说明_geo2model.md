# geo2model 原型使用说明

> 基于平面地质图的智能化三维构建 · 原型程序运行手册
> 环境：macOS / Linux，Python 3.14（见 `requirements-geo2model.txt`）

---

## 1. 环境准备

```bash
cd geo3d_project
python3 -m venv .venv-gempy                       # 已存在可跳过
.venv-gempy/bin/pip install -r requirements-geo2model.txt
```

以下命令均在 `geo3d_project/` 目录下执行，`PY=.venv-gempy/bin/python`。

## 1.5 图形界面（推荐新手）

双击项目根目录的 **`启动界面.command`**（或运行
`.venv-gempy/bin/python app/geo2model_app.py`），浏览器会自动打开
`http://127.0.0.1:7860`，三个页签：

- **① 快速演示**：浏览任意已完成案例的全部成果 + 可交互 3D；
- **② 新地图全流程**：上传地图 → 设参数 → 点「运行提取」→ 在三张表格里
  完成人工审核（单元命名/层序、断层确认、等高线高程）→ 点「审核入库并
  建模」→ 直接看重建地质图/剖面/3D；
- **③ 模型查询**：任选案例做虚拟钻孔、任意剖面、平切图。

界面调用的函数与命令行完全相同，产物写在
`data/output/geo2model/ui_session/`（每次「运行提取」会覆盖重建）。

## 2. 五分钟跑通（合成基准）

```bash
PY=.venv-gempy/bin/python
$PY scripts/10_gen_synthetic_map.py base        # 生成合成地质图+真值
$PY scripts/11_run_pipeline.py configs/synth_base_deep.json   # 端到端
open data/output/geo2model/synth_base_deep/model/model_3d.html  # 交互3D
```

运行结束会打印分区识别准确率、三维体素一致率、DEM 误差。

## 3. 脚本总览

| 脚本 | 作用 |
|---|---|
| `scripts/10_gen_synthetic_map.py [场景名...]` | 生成 6 个内置合成场景（图+像素级真值+剖面图） |
| `scripts/11_run_pipeline.py <配置.json> [--interactive]` | 端到端流水线（提取→数据库→建模→应用→评估） |
| `scripts/12_evaluate_all.py` | 六场景批量评估，汇总指标表 |
| `scripts/13_robustness.py` | 7 种扫描退化 × 4 档强度的鲁棒性扫描 |
| `scripts/14_surface_agreement.py <案例目录>` | 真实图幅闭环指标：地表重现一致率 |
| `scripts/15_ablation_sweep.py <案例目录> <range>` | 消融辅助：指定插值 range 重算体素一致率 |

## 4. 案例配置文件（configs/*.json）

关键字段（完整示例见 `configs/synth_base_deep.json`、`configs/real_map2.json`）：

```jsonc
{
  "case_dir": "data/output/geo2model/我的案例",  // 案例目录（input/map.png 为输入图）
  "georef":  {"world": [x0, x1, y0, y1]},        // 图幅四角世界坐标（米）
  "z_range": [z0, z1],                            // 模型深度范围
  "resolution": [nx, ny, nz],                     // 体素网格
  "segment":  { "dark_gray_max": 110, ... },      // 分割参数（SegmentParams 字段）
  "terrain":  { "mode": "contours" | "flat",      // 等高线DEM / 平坦DEM
                "elevations": "from_truth" | {"0": 650},
                "elevations_file": "configs/xx_contours.json",
                "dot_area_max": 25 },             // 点线等高线开启圆点模式
  "review":   "from_truth" | { "units": [...] },  // 人机交互审核（见下）
  "manual_orientations": [                        // 图面产状符号人工读数
      {"u": 383, "v": 949, "dip": 48, "azimuth": 315} ],
  "constraints": { "use_section_picks": true },   // 剖面图深部约束
  "model":    { "also_html": true },
  "apps":     "auto" | {"boreholes": [[x,y]], "sections": [[[x1,y1],[x2,y2]]], "slices": [z]},
  "stop_after": "extract",                        // 只跑提取（交互工作流第一步）
  "evaluate": true                                // 有真值时计算指标
}
```

## 5. 真实地图的人机交互工作流（半自动）

**第一遍：只跑提取，看结果**

```bash
$PY scripts/11_run_pipeline.py configs/我的图_extract.json   # 配置里 stop_after=extract
```

查看 `case_dir/extract/`：
- `units_auto.csv` + `preview_clusters.png`：自动分区结果（人机交互①）
- `preview_vectors.png` + `faults.json`：接触线与断层候选（人机交互②）
- `preview_contours.png` + `contours.json`：等高线链（人机交互③）

**第二遍：写审核信息，跑全流程**

在配置的 `review.units` 里给每个分区命名、定层序（order：新→老 = 0,1,2,...）、
定系列（series：不整合盖层单独成组）、标水体/忽略簇；多个簇可以同名（自动合并）。
断层在 `review.fault_confirm` 里给要确认的 fault_id（或 "all"）。
等高线高程写成 `{"elevations": {"链id": 高程}}` 存 json，用 `elevations_file` 指向它。
图面产状读数填 `manual_orientations`（u,v 像素位置 + dip/azimuth）。

```bash
$PY scripts/11_run_pipeline.py configs/我的图.json
```

也可以加 `--interactive`，流水线会在三个交互点暂停等你编辑文件。

## 6. 输出产物（case_dir 下）

| 目录 | 内容 |
|---|---|
| `extract/` | labels.npy、units_auto.csv、contacts/faults/contours.json、dem.npy、预览图 |
| `db/` | units.csv、surface_points.csv、orientations.csv、model_config.json（GemPy 直读） |
| `model/` | model_3d.html（交互3D）、剖面 PNG、重建地质图、lith_block.npy、export/（VTK/OBJ/glTF） |
| `apps/` | borehole_N.png/csv（虚拟钻孔）、section_N.png（任意剖面）、slice_N.png（平切图） |
| `eval/` | metrics.json（准确率/IoU/边界F1/体素一致率）、pipeline_report.json |

`model_3d.html` 用浏览器打开即可拖拽旋转；`export/model_lith.vti`、
`model_surfaces.vtm` 可用 ParaView 打开；OBJ/glTF 可导入 Blender 等。

## 7. 常见问题

- **深色地层被当成线划**：调低 `segment.dark_gray_max`（如 55），保证它低于最深
  填色的灰度、高于黑线灰度。
- **相邻地层颜色太近被合并**：调低 `segment.merge_delta_e`（合成图默认 5.0）。
- **点线等高线断链**：设 `terrain.dot_area_max`（圆点面积上限）并把
  `link_dist` 调到略大于点间距。
- **模型深部失真**：只有图面约束时深部多解，务必提供剖面深部拾取
  （`use_section_picks`）或产状读数（`manual_orientations`）；插值参数
  `model.kriging_range` 默认 10（参数扫描结论，见技术报告）。
- **某系列报"无产状"警告**：近水平盖层会自动插入 dip=5° 兜底，可在
  `review.units` 对应单元填 dip/azimuth 人工指定。
