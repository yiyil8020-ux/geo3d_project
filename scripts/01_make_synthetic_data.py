#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_make_synthetic_data.py
=========================
生成带「褶皱 + 正断层」的合成地质数据，供 GemPy 三维建模使用。

几何场景（教学用假数据，非真实填图）：
--------------------------------------
- 范围：X/Y ∈ [0, 2000] m，Z ∈ [0, 1000] m
- 地层（新→老）：Shale → Sandstone → Claystone → Limestone
- 褶皱：沿 X 方向的圆柱状褶皱 + Y 方向弱起伏（近似倾伏/干涉）
- 断层：Main_Fault，近南北走向、向东倾的正断层，上盘（东盘）下落
- 产状：在褶皱翼部按界面解析梯度计算局部 dip/azimuth；断层单独给产状

输出：
------
- data/csv/synthetic/surface_points.csv  → X,Y,Z,formation
- data/csv/synthetic/orientations.csv    → X,Y,Z,azimuth,dip,polarity,formation

运行：
------
    .venv-gempy/bin/python scripts/01_make_synthetic_data.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "csv" / "synthetic"

# ---------------------------------------------------------------------------
# 模型范围与地层
# ---------------------------------------------------------------------------
EXTENT = [0.0, 2000.0, 0.0, 2000.0, 0.0, 1000.0]

# 各地层在「未褶皱、未断错」参考状态下的基准底界面高程（米）
# GemPy：surface points 表示该单元底界面
LAYER_Z0 = {
    "Shale": 720.0,
    "Sandstone": 580.0,
    "Claystone": 440.0,
    "Limestone": 300.0,
}
LAYER_ORDER = ["Shale", "Sandstone", "Claystone", "Limestone"]
FAULT_NAME = "Main_Fault"

# ---------------------------------------------------------------------------
# 褶皱参数
# ---------------------------------------------------------------------------
# 区域缓倾（向东略降），再叠加褶皱起伏
REGIONAL_DIP_DEG = 6.0
FOLD_AMP = 95.0  # 主褶皱振幅（米）——越大褶皱越明显
FOLD_WAVELENGTH = 1100.0  # 主褶皱波长（米），沿 X
FOLD_PHASE_X = 350.0  # 相位：控制背斜/向斜位置
CROSS_AMP = 35.0  # 沿 Y 的次级起伏振幅
CROSS_WAVELENGTH = 1600.0

# ---------------------------------------------------------------------------
# 断层参数（正断层：上盘下落）
# ---------------------------------------------------------------------------
# 在 Z = FAULT_Z_REF 时，断层面 X = FAULT_X_REF
FAULT_X_REF = 1050.0
FAULT_Z_REF = 500.0
FAULT_DIP = 65.0  # 倾角（度），倾向东 → azimuth=90
FAULT_AZIMUTH = 90.0
FAULT_THROW = 140.0  # 垂直断距（米），东盘（上盘）下落
FAULT_BUFFER = 90.0  # 采样时避开断层面附近，避免两侧点混在一起


def _regional_slope() -> float:
    """区域倾角对应的 dZ/dX（向东为负）。"""
    return -math.tan(math.radians(REGIONAL_DIP_DEG))


def fold_undulation(x: float, y: float) -> float:
    """
    褶皱起伏量（不含区域倾角、不含断距）。

    Z_fold = A * cos(2π (x - phase) / λ)
             + B * cos(2π (y - phase_y) / λy)
    """
    main = FOLD_AMP * math.cos(2.0 * math.pi * (x - FOLD_PHASE_X) / FOLD_WAVELENGTH)
    cross = CROSS_AMP * math.cos(2.0 * math.pi * (y - 250.0) / CROSS_WAVELENGTH)
    return main + cross


def fold_gradients(x: float, y: float) -> tuple[float, float]:
    """
    褶皱面 Z 对 x、y 的偏导数 dZ/dx、dZ/dy（解析式）。

    用于把「翼部产状」算出来，而不是全图同一个 dip。
    """
    # d/dx [A cos(2π(x-p)/λ)] = -A (2π/λ) sin(...)
    dzdx = _regional_slope() + (
        -FOLD_AMP
        * (2.0 * math.pi / FOLD_WAVELENGTH)
        * math.sin(2.0 * math.pi * (x - FOLD_PHASE_X) / FOLD_WAVELENGTH)
    )
    dzdy = (
        -CROSS_AMP
        * (2.0 * math.pi / CROSS_WAVELENGTH)
        * math.sin(2.0 * math.pi * (y - 250.0) / CROSS_WAVELENGTH)
    )
    return dzdx, dzdy


def gradients_to_dip_azimuth(dzdx: float, dzdy: float) -> tuple[float, float]:
    """
    由界面坡度求产状（倾向方位角、倾角）。

    约定：
    - dip_direction（azimuth）：下倾方向的方位角，0=北、90=东
    - dip：与水平面夹角 0–90°
    """
    # 坡度向量在水平面上的「下坡」方向
    # 高度随位移变化：沿 (dx,dy) 移动时 dZ = dzdx*dx + dzdy*dy
    # 最陡下坡方向 = -grad(Z) 在 xy 的投影
    gx, gy = -dzdx, -dzdy
    slope = math.hypot(dzdx, dzdy)
    dip = math.degrees(math.atan(slope))
    # atan2(东分量, 北分量)：这里 x=东、y=北
    azimuth = math.degrees(math.atan2(gx, gy)) % 360.0
    # 近水平时 azimuth 不稳定，给一个默认东倾
    if slope < 1e-6:
        return 90.0, 0.0
    return round(azimuth, 2), round(dip, 2)


def fault_x_at_z(z: float) -> float:
    """
    断层面在给定高程处的 X 坐标。

    倾向东、倾角 FAULT_DIP：
    越往深处（Z 减小），断层面越往东移：
        X(z) = X_ref + (Z_ref - z) / tan(dip)
    """
    return FAULT_X_REF + (FAULT_Z_REF - z) / math.tan(math.radians(FAULT_DIP))


def is_hanging_wall(x: float, z: float) -> bool:
    """上盘（东盘）判定：点在断层面以东。"""
    return x > fault_x_at_z(z)


def layer_surface_z(x: float, y: float, z0: float) -> float:
    """
    某地层底界面在 (x,y) 的高程：褶皱 + 区域倾角 + 断距。

    简化断距：先按未断错褶皱求 Z，再用该 Z 判断盘别并施加垂直断距。
    （真实运动学更复杂；对合成演示足够。）
    """
    z = z0 + _regional_slope() * (x - 1000.0) + fold_undulation(x, y)
    if is_hanging_wall(x, z):
        z -= FAULT_THROW
    return z


def make_strat_surface_points() -> list[dict]:
    """
    地层界面控制点：在断层两侧分别布点，避开断层面缓冲带。
    """
    # 比简单模型更密，才能托住褶皱曲率
    x_vals = [150.0, 350.0, 550.0, 750.0, 900.0, 1200.0, 1400.0, 1600.0, 1800.0]
    y_vals = [200.0, 500.0, 800.0, 1100.0, 1400.0, 1700.0]
    z_min, z_max = EXTENT[4], EXTENT[5]

    rows: list[dict] = []
    for formation, z0 in LAYER_Z0.items():
        for x in x_vals:
            for y in y_vals:
                z = layer_surface_z(x, y, z0)
                # 模型盒内留边
                if z <= z_min + 30 or z >= z_max - 30:
                    continue
                # 远离断层面
                if abs(x - fault_x_at_z(z)) < FAULT_BUFFER:
                    continue
                rows.append(
                    {
                        "X": round(x, 3),
                        "Y": round(y, 3),
                        "Z": round(z, 3),
                        "formation": formation,
                    }
                )
    return rows


def make_fault_surface_points() -> list[dict]:
    """
    断层面控制点：沿走向（Y）与倾向方向在断层面上取样。
    """
    y_vals = [200.0, 500.0, 800.0, 1100.0, 1400.0, 1700.0]
    z_vals = [150.0, 300.0, 450.0, 600.0, 750.0, 900.0]
    rows: list[dict] = []
    for y in y_vals:
        for z in z_vals:
            x = fault_x_at_z(z)
            if not (EXTENT[0] + 50 < x < EXTENT[1] - 50):
                continue
            if not (EXTENT[4] + 20 < z < EXTENT[5] - 20):
                continue
            rows.append(
                {
                    "X": round(x, 3),
                    "Y": round(y, 3),
                    "Z": round(z, 3),
                    "formation": FAULT_NAME,
                }
            )
    return rows


def make_strat_orientations() -> list[dict]:
    """
    地层产状：在褶皱两翼、断层两侧各取若干点，用解析梯度算 dip/azimuth。
    """
    # 选在背斜/向斜翼部，产状差异更直观
    sample_xy = [
        (300.0, 600.0),
        (300.0, 1400.0),
        (600.0, 1000.0),
        (800.0, 1000.0),
        (1300.0, 1000.0),
        (1500.0, 600.0),
        (1500.0, 1400.0),
        (1750.0, 1000.0),
    ]
    rows: list[dict] = []
    for formation, z0 in LAYER_Z0.items():
        for x, y in sample_xy:
            z = layer_surface_z(x, y, z0)
            if abs(x - fault_x_at_z(z)) < FAULT_BUFFER:
                continue
            if not (EXTENT[4] + 40 < z < EXTENT[5] - 40):
                continue
            # 产状由「未施加断距」的褶皱几何决定（刚性断块内产状连续）
            # 对演示：仍用褶皱梯度；断距只平移，不改倾向
            dzdx, dzdy = fold_gradients(x, y)
            azimuth, dip = gradients_to_dip_azimuth(dzdx, dzdy)
            # 过滤几乎水平的点，减少无信息产状
            if dip < 2.0:
                dip = max(dip, 5.0)
                azimuth = 90.0
            rows.append(
                {
                    "X": round(x, 3),
                    "Y": round(y, 3),
                    "Z": round(z, 3),
                    "azimuth": azimuth,
                    "dip": dip,
                    "polarity": 1,
                    "formation": formation,
                }
            )
    return rows


def make_fault_orientations() -> list[dict]:
    """断层产状：倾向东、倾角 FAULT_DIP。"""
    rows: list[dict] = []
    for y, z in [(500.0, 400.0), (1000.0, 500.0), (1500.0, 600.0)]:
        x = fault_x_at_z(z)
        rows.append(
            {
                "X": round(x, 3),
                "Y": round(y, 3),
                "Z": round(z, 3),
                "azimuth": FAULT_AZIMUTH,
                "dip": FAULT_DIP,
                "polarity": 1,
                "formation": FAULT_NAME,
            }
        )
    return rows


def make_surface_points() -> pd.DataFrame:
    rows = make_strat_surface_points() + make_fault_surface_points()
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("未生成任何 surface points")

    counts = df["formation"].value_counts()
    for name in LAYER_ORDER + [FAULT_NAME]:
        n = int(counts.get(name, 0))
        if n < 6:
            raise RuntimeError(f"{name} 控制点过少（{n}），请调整褶皱/断层/网格参数")
    return df


def make_orientations() -> pd.DataFrame:
    rows = make_strat_orientations() + make_fault_orientations()
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("未生成任何 orientations")
    # 每层至少 1 个产状
    for name in LAYER_ORDER + [FAULT_NAME]:
        if name not in set(df["formation"]):
            raise RuntimeError(f"{name} 缺少产状点")
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sp = make_surface_points()
    ori = make_orientations()

    sp_path = OUTPUT_DIR / "surface_points.csv"
    ori_path = OUTPUT_DIR / "orientations.csv"
    sp.to_csv(sp_path, index=False)
    ori.to_csv(ori_path, index=False)

    print("=" * 60)
    print("合成地质数据已生成（褶皱 + 正断层）")
    print("=" * 60)
    print(f"surface_points : {sp_path}")
    print(f"  行数={len(sp)}")
    print(sp["formation"].value_counts().to_string())
    print()
    print(f"orientations   : {ori_path}")
    print(f"  行数={len(ori)}")
    print(ori.groupby("formation")[["dip", "azimuth"]].mean().round(1).to_string())
    print()
    print("模型 extent   :", EXTENT)
    print(
        f"褶皱: 振幅={FOLD_AMP} m, 波长={FOLD_WAVELENGTH} m; "
        f"区域倾角={REGIONAL_DIP_DEG}°"
    )
    print(
        f"断层: {FAULT_NAME}, 倾角={FAULT_DIP}°→E, "
        f"断距={FAULT_THROW} m, 迹线 X≈{FAULT_X_REF} @ Z={FAULT_Z_REF}"
    )
    print("下一步：.venv-gempy/bin/python scripts/02_gempy_model.py")


if __name__ == "__main__":
    main()
