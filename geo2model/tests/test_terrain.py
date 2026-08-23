# -*- coding: utf-8 -*-
"""
test_terrain.py — 虚线等高线提取与 DEM 重建测试
=================================================

直接运行，不依赖 pytest：
    .venv-gempy/bin/python -m geo2model.tests.test_terrain

验证指标（对合成真值）：
- synth_base：赋值率 ≥70%，DEM MAE < 15 米（相对高差 ~370m 的 4%）
- synth_high_relief（等高线更密）：赋值率 ≥70%，DEM MAE < 20 米
"""

import json
import os
import sys

import cv2
import numpy as np

# 项目根目录加入 sys.path，保证 geo2model 包可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from geo2model.terrain import (  # noqa: E402
    TerrainParams,
    apply_elevations,
    assign_elevations_from_truth,
    contours_preview,
    contours_to_dem,
    extract_contours,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "output", "geo2model")


def run_case(case: str, mae_limit: float):
    """单案例全流程：提取 → 预览 → 真值赋高程 → DEM 重建 → 对比真值。"""
    case_dir = os.path.join(DATA, case)
    print(f"\n===== 案例 {case} =====")

    img_bgr = cv2.imread(os.path.join(case_dir, "input", "map.png"))
    assert img_bgr is not None, f"找不到输入图: {case_dir}/input/map.png"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    params = TerrainParams()

    # ---- 提取 ----
    contours = extract_contours(img_rgb, params)
    with open(os.path.join(case_dir, "truth", "contours_gt.json"), encoding="utf-8") as f:
        contours_gt = json.load(f)
    print(f"提取等高线链: {len(contours)} 条（真值 {len(contours_gt)} 条，"
          f"交叉截断会使提取段数偏多）")
    assert len(contours) >= len(contours_gt) * 0.7, "提取条数过少"

    # ---- 预览图 ----
    extract_dir = os.path.join(case_dir, "extract")
    os.makedirs(extract_dir, exist_ok=True)
    preview_png = os.path.join(extract_dir, "preview_contours.png")
    contours_preview(img_rgb, contours, preview_png)
    assert os.path.exists(preview_png) and os.path.getsize(preview_png) > 0, \
        "预览图未生成"
    print(f"预览图: {preview_png}")

    # ---- 真值模拟赋高程 ----
    mapping = assign_elevations_from_truth(contours, contours_gt)
    rate = len(mapping) / max(len(contours), 1)
    print(f"赋值成功: {len(mapping)}/{len(contours)} 条（{rate:.1%}）")
    assert rate >= 0.70, f"赋值率 {rate:.1%} < 70%"
    apply_elevations(contours, mapping)

    # ---- DEM 重建与真值对比 ----
    dem_gt = np.load(os.path.join(case_dir, "truth", "dem_gt.npy"))
    dem = contours_to_dem(contours, dem_gt.shape, params)
    assert dem.shape == dem_gt.shape and dem.dtype == np.float32
    mae = float(np.mean(np.abs(dem - dem_gt)))
    relief = float(dem_gt.max() - dem_gt.min())
    print(f"DEM MAE = {mae:.2f} 米（真值高差 {relief:.0f} 米，上限 {mae_limit} 米）")
    assert mae < mae_limit, f"DEM MAE {mae:.2f} ≥ {mae_limit}"


if __name__ == "__main__":
    run_case("synth_base", mae_limit=15.0)
    run_case("synth_high_relief", mae_limit=20.0)
    print("\nALL TESTS PASSED")
