# geo2model 数据契约（模块间接口规范）

> 所有模块必须严格遵守本文件定义的数据格式。修改契约必须先改本文件。
> 坐标约定：图像像素 (u,v)，u=列（向右），v=行（向下），原点在左上角。
> 世界坐标 (X,Y,Z)：X 向东，Y 向北，Z 向上，单位米。

## 运行目录结构（每个案例 case 一个目录）

```
data/output/geo2model/<case>/
├── input/map.png              # 输入地质图（RGB）
├── truth/                     # 仅合成案例：真值
│   ├── labels_gt.npy          # int16 HxW，每像素地层单元 id（0..n-1，含基底）
│   ├── boundary_gt.npy        # uint8 HxW，1=真值地质界线像素（接触线，不含等高线）
│   ├── dem_gt.npy             # float32 HxW，每像素地表高程（米）
│   ├── truth_meta.json        # 单元表、几何参数、georef、断层参数等
│   └── section_gt.png         # 合成剖面图（深部约束配准演示用）
├── extract/                   # 矢量化输出
│   ├── labels.npy             # int16 HxW，提取的分区标签（0..k-1）
│   ├── units_auto.csv         # 自动单元表（供人工校正）
│   ├── contacts.json          # 接触线折线
│   ├── faults.json            # 断层折线
│   ├── contours.json          # 等高线折线（含人工赋高程接口）
│   ├── dem.npy                # float32 HxW，插值 DEM（无等高线时为常值）
│   └── preview_*.png          # 各步骤预览图（人机交互检查点）
├── db/                        # 数据库构建输出（GemPy 直接可读）
│   ├── units.csv              # ★人工校正后的单元表（人机交互主入口）
│   ├── surface_points.csv     # X,Y,Z,formation
│   ├── orientations.csv       # X,Y,Z,azimuth,dip,polarity,formation
│   └── model_config.json      # 建模配置（extent/resolution/series/faults）
├── model/                     # 三维建模输出
│   ├── model_3d.html          # 交互 3D
│   ├── section_*.png          # 剖面
│   ├── lith_block.npy         # float64 (nx,ny,nz) 岩性体素解
│   ├── lith_meta.json         # 见下
│   └── export/model.vtk|.obj|.gltf
├── apps/                      # 应用输出：borehole_*.png / section_*.png / slice_*.png
└── eval/                      # metrics.json、鲁棒性曲线
```

## 关键文件格式

### units.csv（单元表；人机交互校正的核心文件）
```csv
unit_id,name,role,order,color_r,color_g,color_b,dip,azimuth
0,Q,strata,0,245,240,214,,
5,Main_Fault,fault,-1,30,30,30,65,90
7,Lake,water,-1,150,200,230,,
```
- `unit_id`：与 labels.npy 中的值对应
- `role`：strata（地层）| fault（断层）| water（水体）| ignore（忽略）
- `order`：地层新→老为 0,1,2,...；非地层为 -1
- `dip`,`azimuth`：可选，人工从图面读取的代表产状（半自动通道）；空则由算法估算

### surface_points.csv / orientations.csv（GemPy 契约，不可更改列名）
```csv
X,Y,Z,formation
```
```csv
X,Y,Z,azimuth,dip,polarity,formation
```
- 接触线上的点属于**较年轻**一侧地层（GemPy surface point = 单元底界面）
- azimuth=倾向方位角（0=北,90=东），dip=倾角度，polarity=1

### contacts.json
```json
[{"contact_id": 0, "unit_a": 1, "unit_b": 2, "polyline_px": [[u,v], ...]}]
```
- `unit_a`/`unit_b`：接触两侧的分区 id（数值升序，无新老语义——提取阶段
  还不知道层序）。新老关系与 formation 归属由 constraints 按 units.csv 的
  order/series 推断：同系列相邻层 → 年轻方底界；跨系列（不整合）→
  年轻系列最老成员的底界（归属对象可能不是接触两侧任一单元）。

### faults.json
```json
[{"fault_id": 0, "name": "Fault_1", "polyline_px": [[u,v],...],
  "dip": 65.0, "azimuth": 90.0, "confirmed": false}]
```

### contours.json（等高线；半自动赋高程）
```json
[{"contour_id": 0, "elevation": null, "polyline_px": [[u,v],...]}]
```
`elevation` 为 null 时表示待人工赋值（预览图上标注 contour_id 供对照）。

### model_config.json
```json
{"extent": [x0,x1,y0,y1,z0,z1], "resolution": [nx,ny,nz],
 "series": {"Fault_Series_1": ["Fault_1"], "Strat_Series": ["K1","P3","P1"]},
 "fault_series": ["Fault_Series_1"],
 "georef": {"img_w":1200,"img_h":1200,"world":[x0,x1,y0,y1]}}
```
`series` 键顺序 = 从新到老；地层列表内部同样新→老。

### lith_meta.json（apps 模块的唯一依赖）
```json
{"extent": [...], "resolution": [nx,ny,nz],
 "id_to_name": {"1": "Fault_1", "2": "K1", ...},
 "name_to_color": {"K1": [r,g,b], ...},
 "surface_z": null}
```
- lith_block.npy：float (nx,ny,nz)，C 序，axis0=X（西→东），axis1=Y（南→北），
  axis2=Z（下→上）；值为 id_to_name 的键（可能为浮点，四舍五入取整）
- 体素中心坐标：x_i = x0 + (i+0.5)*(x1-x0)/nx，其余同理

### metrics.json（eval 输出）
```json
{"pixel_accuracy": 0.97, "pixel_accuracy_excl_boundary": 0.99,
 "macro_iou": 0.95, "per_class_iou": {"K1": 0.96},
 "boundary_f1_tol3px": 0.93, "voxel_agreement_3d": 0.90,
 "n_units_gt": 6, "n_units_pred": 6}
```

## GeoRef（像素↔世界，georef.py 已实现，直接 import 使用）
- `GeoRef(img_w, img_h, world=(x0,x1,y0,y1))`
- `px_to_world(u,v) -> (x,y)`：像素中心 → 世界；v 轴翻转（图像向下 = 世界向南）
- `world_to_px(x,y) -> (u,v)`
