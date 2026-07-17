# 合成地质数据（synthetic）

用于绕过地质图信息提取，直接验证 GemPy 三维建模。当前场景包含：

- **褶皱**：沿 X 的圆柱状褶皱 + Y 向弱起伏
- **正断层**：`Main_Fault`，近南北走向、向东倾约 65°，上盘下落约 140 m
- **地形起伏**：在 `02_gempy_model.py` 中程序生成 DEM（丘陵 + 断层谷）

## 文件

| 文件 | 说明 |
|------|------|
| `surface_points.csv` | 层位 + 断层面控制点：`X,Y,Z,formation` |
| `orientations.csv` | 产状：`X,Y,Z,azimuth,dip,polarity,formation` |

## 重新生成 + 建模

```bash
cd geo3d_project
.venv-gempy/bin/python scripts/01_make_synthetic_data.py
.venv-gempy/bin/python scripts/02_gempy_model.py
open data/output/synthetic/model_3d.html
```

建议先看 `data/output/synthetic/section_y_mid.png`（X–Z 剖面最能显示褶皱与断层）。
