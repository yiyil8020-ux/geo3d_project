# -*- coding: utf-8 -*-
"""
mapgen.py — 合成地质图生成器（含像素级真值）
==============================================

作用（定量验证的地基）：
    先用解析函数定义一个"真实"三维地质体（区域倾斜 + 褶皱 + 可选正断层 + 起伏地形），
    再做正向渲染：把地表出露单元画成彩色地质图（黑色地质界线、加粗断层线、
    虚线等高线、河流、单元代号标注），同时保存像素级真值标签 / 真值 DEM /
    真值边界 / 真值剖面图。

    这样「地质图 → 三维模型」流水线的每一步都能和真值对比，
    从而定量回答申报书的硬指标：地层分区识别准确率是否 ≥ 90%、
    重建三维模型与真值的体素一致率是多少。

几何约定（与 scripts/01_make_synthetic_data.py 一脉相承，但全部向量化）：
    - 世界坐标：X 向东、Y 向北、Z 向上，单位米
    - 每个地层由"底界面"高程场 b_i(x,y) 定义（新→老，底界面依次降低）
    - 底界面 = 基准高程 z0 + 区域倾斜 + 褶皱起伏 - 断层落差（上盘）
    - 地表某点出露单元 = 满足 z_dem >= b_i 的最年轻地层；都不满足则为基底
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from .georef import GeoRef
from .sectionreg import SectionFrame

# ---------------------------------------------------------------------------
# 场景参数
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """一个合成地质图场景的全部参数（可复现）。"""

    name: str = "base"
    # ---- 图像与世界范围 ----
    img_w: int = 1100
    img_h: int = 1100
    world: tuple = (0.0, 2000.0, 0.0, 2000.0)  # (x0, x1, y0, y1)
    z_range: tuple = (0.0, 1000.0)

    # ---- 地层（新→老）：(名称, 底界面基准高程, RGB 颜色) ----
    layers: tuple = (
        ("N1", 780.0, (250, 240, 190)),
        ("K2", 680.0, (205, 225, 145)),
        ("K1", 570.0, (150, 200, 120)),
        ("P2", 450.0, (240, 165, 120)),
        ("P1", 330.0, (215, 125, 95)),
    )
    basement_name: str = "S1"
    basement_color: tuple = (125, 185, 175)

    # ---- 构造：区域倾斜 + 褶皱 ----
    regional_dip_deg: float = 6.0     # 区域缓倾角（向东降低）
    x_center: float = 1000.0          # 区域倾斜的枢轴 X
    fold_amp: float = 90.0            # 主褶皱振幅（米）
    fold_wavelength: float = 1150.0   # 主褶皱波长（米，沿 X）
    fold_phase_x: float = 350.0
    cross_amp: float = 35.0           # 沿 Y 的次级起伏
    cross_wavelength: float = 1600.0
    cross_phase_y: float = 250.0

    # ---- 断层（None 表示无断层）----
    fault: dict | None = field(
        default_factory=lambda: {
            "name": "F1",
            "x_ref": 1150.0,   # 在 z_ref 高程处断层面的 X
            "z_ref": 500.0,
            "dip": 65.0,       # 倾角（度），倾向东
            "azimuth": 90.0,
            "throw": 140.0,    # 垂直断距（米），东盘（上盘）下落
        }
    )

    # ---- 地形 DEM ----
    dem_base: float = 700.0
    dem_amp1: float = 110.0
    dem_wl1: float = 1700.0
    dem_amp2: float = 75.0
    dem_wl2: float = 1300.0
    dem_amp3: float = 45.0
    dem_wl3: float = 2100.0
    dem_flat: bool = False            # True → 平坦地形（退化情形）

    # ---- 制图样式 ----
    contact_w: int = 3                # 地质界线宽（像素）
    fault_w: int = 7                  # 断层线宽（像素，比界线粗，作为识别依据之一）
    contour_interval: float = 50.0    # 等高线间距（米）
    contour_w: int = 2
    dash_on: int = 9                  # 等高线虚线：实段长（像素）
    dash_off: int = 7                 # 空段长
    n_rivers: int = 2                 # 河流条数（蓝色干扰要素）
    draw_labels: bool = True          # 是否标注单元代号文字（干扰要素）
    seed: int = 42

    # ------------------------------------------------------------------
    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def basement_id(self) -> int:
        return self.n_layers  # 基底 id 排在所有地层之后

    def unit_table(self) -> list[dict]:
        """单元表（含基底），与 CONTRACTS.md 的 units.csv 对应。"""
        rows = []
        for i, (name, _z0, color) in enumerate(self.layers):
            rows.append(
                {
                    "unit_id": i,
                    "name": name,
                    "role": "strata",
                    "order": i,
                    "color_r": color[0],
                    "color_g": color[1],
                    "color_b": color[2],
                }
            )
        rows.append(
            {
                "unit_id": self.basement_id,
                "name": self.basement_name,
                "role": "strata",
                "order": self.n_layers,
                "color_r": self.basement_color[0],
                "color_g": self.basement_color[1],
                "color_b": self.basement_color[2],
            }
        )
        return rows


# ---------------------------------------------------------------------------
# 解析几何场（全部支持 numpy 数组输入）
# ---------------------------------------------------------------------------


def _regional_slope(scn: Scenario) -> float:
    """区域倾斜的 dZ/dX（向东为负）。"""
    return -math.tan(math.radians(scn.regional_dip_deg))


def fold_z(scn: Scenario, x, y):
    """褶皱起伏量（米）。"""
    main = scn.fold_amp * np.cos(
        2.0 * np.pi * (np.asarray(x) - scn.fold_phase_x) / scn.fold_wavelength
    )
    cross = scn.cross_amp * np.cos(
        2.0 * np.pi * (np.asarray(y) - scn.cross_phase_y) / scn.cross_wavelength
    )
    return main + cross


def fault_x_at_z(scn: Scenario, z):
    """断层面在高程 z 处的 X 坐标（无断层时返回 +inf，即所有点都在下盘）。"""
    if scn.fault is None:
        return np.full_like(np.asarray(z, dtype=float), np.inf)
    f = scn.fault
    return f["x_ref"] + (f["z_ref"] - np.asarray(z, dtype=float)) / math.tan(
        math.radians(f["dip"])
    )


def dem_at(scn: Scenario, x, y):
    """地表高程 DEM（米）。多尺度正弦丘陵 + 断层谷；可选平坦模式。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if scn.dem_flat:
        return np.full(np.broadcast(x, y).shape, scn.dem_base)
    z = (
        scn.dem_base
        + scn.dem_amp1 * np.sin(2.0 * np.pi * x / scn.dem_wl1)
        + scn.dem_amp2 * np.cos(2.0 * np.pi * y / scn.dem_wl2)
        + scn.dem_amp3 * np.sin(2.0 * np.pi * (x + 0.6 * y) / scn.dem_wl3)
        + 25.0 * np.cos(2.0 * np.pi * x / 600.0) * np.sin(2.0 * np.pi * y / 800.0)
    )
    if scn.fault is not None:
        # 断层谷：沿断层地表迹线附近轻微下凹（用 x_ref 近似即可）
        z = z - 45.0 * np.exp(-((x - scn.fault["x_ref"]) ** 2) / (2.0 * 180.0**2))
    z_min, z_max = scn.z_range
    return np.clip(z, z_min + 80.0, z_max - 40.0)


def unit_ids_at(scn: Scenario, x, y, z):
    """真值核心：任意三维点 (x,y,z) 处的地层单元 id（向量化）。

    判定：从老到新依次检查 z 是否高于该层底界面，最后命中的（最年轻的）
    即为所在单元；都不满足则是基底。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    slope = _regional_slope(scn)
    fold = fold_z(scn, x, y)

    # 上盘判定：点位于断层面以东（正断层，上盘下落 throw 米）
    if scn.fault is not None:
        hanging = x > fault_x_at_z(scn, z)
        drop = np.where(hanging, scn.fault["throw"], 0.0)
    else:
        drop = 0.0

    out = np.full(np.broadcast(x, y, z).shape, scn.basement_id, dtype=np.int16)
    # 从老到新覆盖赋值：最终每个点得到「z >= 底界面」的最年轻层
    for i in range(scn.n_layers - 1, -1, -1):
        z0 = scn.layers[i][1]
        bottom = z0 + slope * (x - scn.x_center) + fold - drop
        out = np.where(z >= bottom, np.int16(i), out)
    return out


def layer_bottom_z(scn: Scenario, layer_idx: int, x, y, hanging_side=None):
    """第 layer_idx 层底界面高程场（用于剖面真值点、深部拾取）。

    hanging_side: None → 自动按「底界面自身位置」判断盘别（与 unit_ids_at
    的查询点判定略有差别，仅用于生成真值拾取点，足够精确）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z0 = scn.layers[layer_idx][1]
    z_pre = z0 + _regional_slope(scn) * (x - scn.x_center) + fold_z(scn, x, y)
    if scn.fault is None:
        return z_pre
    if hanging_side is None:
        hanging_side = x > fault_x_at_z(scn, z_pre)
    return np.where(hanging_side, z_pre - scn.fault["throw"], z_pre)


def truth_voxel_units(scn: Scenario, extent, resolution):
    """在规则网格上生成三维真值单元体素（用于与 GemPy 重建模型对比）。

    返回 (units, below_ground)：
    - units: int16 (nx,ny,nz)，与 CONTRACTS.md 的 lith_block 同轴序
    - below_ground: bool (nx,ny,nz)，True=位于地表以下（参与对比的体素）
    """
    x0, x1, y0, y1, z0, z1 = extent
    nx, ny, nz = resolution
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    zs = z0 + (np.arange(nz) + 0.5) * (z1 - z0) / nz
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    units = unit_ids_at(scn, X, Y, Z)
    dem = dem_at(scn, X[:, :, 0], Y[:, :, 0])  # (nx,ny)
    below = Z <= dem[:, :, None]
    return units, below


# ---------------------------------------------------------------------------
# 渲染辅助
# ---------------------------------------------------------------------------


def _boundary_mask(labels: np.ndarray) -> np.ndarray:
    """4-邻域标签不一致的像素 = 地质界线真值（1 像素宽）。"""
    b = np.zeros(labels.shape, dtype=np.uint8)
    b[:-1, :] |= (labels[:-1, :] != labels[1:, :]).astype(np.uint8)
    b[:, :-1] |= (labels[:, :-1] != labels[:, 1:]).astype(np.uint8)
    return b


def _draw_dashed_polyline(img, pts, color, thickness, dash_on, dash_off):
    """手工画虚线折线：沿折线累计长度，按 on/off 交替画小段。"""
    if len(pts) < 2:
        return
    pos = 0.0
    pen_down = True
    seg_start = pts[0]
    for i in range(1, len(pts)):
        p0 = np.asarray(pts[i - 1], dtype=float)
        p1 = np.asarray(pts[i], dtype=float)
        seg_len = float(np.hypot(*(p1 - p0)))
        if seg_len < 1e-9:
            continue
        t = 0.0
        while t < seg_len:
            budget = (dash_on if pen_down else dash_off) - pos
            step = min(budget, seg_len - t)
            a = p0 + (p1 - p0) * (t / seg_len)
            b = p0 + (p1 - p0) * ((t + step) / seg_len)
            if pen_down:
                cv2.line(
                    img,
                    (int(round(a[0])), int(round(a[1]))),
                    (int(round(b[0])), int(round(b[1]))),
                    color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )
            t += step
            pos += step
            if pos >= (dash_on if pen_down else dash_off) - 1e-9:
                pen_down = not pen_down
                pos = 0.0
        seg_start = pts[i]
    _ = seg_start  # 仅为可读性保留


def _fault_trace_px(scn: Scenario, ref: GeoRef, dem_px: np.ndarray):
    """断层地表迹线（像素折线）：逐行找 x - fault_x(z_dem) 的符号翻转点。"""
    if scn.fault is None:
        return []
    h, w = dem_px.shape
    us = np.arange(w)
    pts = []
    for v in range(0, h, 3):  # 每 3 行取一点，足够平滑
        x_row, _ = ref.px_to_world(us, np.full(w, v))
        fx = fault_x_at_z(scn, dem_px[v, :])
        f = x_row - fx
        sign_change = np.where(np.diff(np.signbit(f)))[0]
        if len(sign_change) > 0:
            u = int(sign_change[0])
            pts.append([u, v])
    return pts


def _make_rivers(scn: Scenario, dem_px: np.ndarray, rng: np.random.Generator):
    """生成沿地形最陡下降方向蜿蜒的河流折线（蓝色干扰要素）。"""
    h, w = dem_px.shape
    gy, gx = np.gradient(dem_px.astype(float))
    rivers = []
    for _ in range(scn.n_rivers):
        # 从图幅上部 1/3 的较高处出发
        u = float(rng.integers(w // 6, w - w // 6))
        v = float(rng.integers(h // 8, h // 3))
        pts = []
        for _step in range(2500):
            ui, vi = int(round(u)), int(round(v))
            if not (2 <= ui < w - 2 and 2 <= vi < h - 2):
                break
            pts.append([ui, vi])
            # 最陡下降方向 = -梯度；加一点噪声让河道自然弯曲
            dx, dy = -gx[vi, ui], -gy[vi, ui]
            norm = math.hypot(dx, dy) + 1e-9
            u += 1.6 * dx / norm + rng.normal(0, 0.55)
            v += 1.6 * dy / norm + rng.normal(0, 0.55)
        if len(pts) > 60:
            rivers.append(pts)
    return rivers


def _label_positions(labels: np.ndarray, unit_id: int, k: int = 2):
    """找单元内部离边界最远的 k 个点（放置文字标注，模拟真实图面注记）。"""
    mask = (labels == unit_id).astype(np.uint8)
    if mask.sum() < 400:
        return []
    n, comp = cv2.connectedComponents(mask)
    out = []
    sizes = [(comp == i).sum() for i in range(1, n)]
    order = np.argsort(sizes)[::-1][:k]
    for oi in order:
        cid = int(oi) + 1
        m = (comp == cid).astype(np.uint8)
        if m.sum() < 400:
            continue
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        v, u = np.unravel_index(np.argmax(dist), dist.shape)
        if dist[v, u] > 14:  # 离界线至少 14 px，避免文字压线太狠
            out.append((int(u), int(v)))
    return out


def _draw_dip_symbols(
    scn: Scenario,
    img: np.ndarray,
    labels: np.ndarray,
    dem: np.ndarray,
    rng: np.random.Generator,
    n_sites: int = 14,
) -> list[dict]:
    """在地层内部撒产状符号（真值由褶皱解析梯度给出）并画在图上。

    真实地质图上产状符号 = 走向短线 + 倾向小刻度 + 倾角数字；
    流水线端由"用户点选符号位置并键入读数"完成半自动录入，
    合成评估中该人工步骤由 truth/dip_symbols_gt.csv 模拟。
    """
    ref = GeoRef(scn.img_w, scn.img_h, scn.world)
    h, w = labels.shape
    sites: list[tuple[int, int]] = []
    # 每个出露单元取内部点（离界线最远处），全图凑 ~n_sites 个
    uids = [int(t) for t in np.unique(labels)]
    per_unit = max(2, int(np.ceil(n_sites / max(len(uids), 1))))
    for uid in uids:
        for (u, v) in _label_positions(labels, uid, k=per_unit + 1)[1:]:
            # 与单元代号文字错开（_label_positions 第一个点留给文字）
            sites.append((u, v))
    rng.shuffle(sites)
    sites = sites[:n_sites]

    out = []
    for u, v in sites:
        x, y = ref.px_to_world(u, v)
        dzdx, dzdy = fold_gradients_np(scn, float(x), float(y))
        # 下倾方向 = -grad；azimuth: 北=0 顺时针
        dip = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
        if dip < 1.0:
            continue
        az = math.degrees(math.atan2(-dzdx, -dzdy)) % 360.0
        out.append({"u": u, "v": v, "azimuth": round(az, 1), "dip": round(dip, 1)})

        # 图面绘制：世界北 = 图像上方（v 减小）
        az_r = math.radians(az)
        d_dip = np.array([math.sin(az_r), -math.cos(az_r)])   # 倾向 (du,dv)
        d_strike = np.array([-d_dip[1], d_dip[0]])            # 走向（垂直倾向）
        p = np.array([u, v], dtype=float)
        p1 = (p - 11 * d_strike).astype(int)
        p2 = (p + 11 * d_strike).astype(int)
        p3 = (p + 8 * d_dip).astype(int)
        cv2.line(img, tuple(p1), tuple(p2), (40, 36, 34), 2, cv2.LINE_AA)
        cv2.line(img, tuple(p.astype(int)), tuple(p3), (40, 36, 34), 2, cv2.LINE_AA)
        cv2.putText(
            img, f"{dip:.0f}", tuple((p + 14 * d_dip + [2, 6]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 36, 34), 1, cv2.LINE_AA,
        )
    return out


def fold_gradients_np(scn: Scenario, x: float, y: float) -> tuple[float, float]:
    """底界面高程场对 x/y 的解析偏导（含区域倾斜项）。"""
    dzdx = _regional_slope(scn) + (
        -scn.fold_amp
        * (2.0 * math.pi / scn.fold_wavelength)
        * math.sin(2.0 * math.pi * (x - scn.fold_phase_x) / scn.fold_wavelength)
    )
    dzdy = (
        -scn.cross_amp
        * (2.0 * math.pi / scn.cross_wavelength)
        * math.sin(2.0 * math.pi * (y - scn.cross_phase_y) / scn.cross_wavelength)
    )
    return dzdx, dzdy


# ---------------------------------------------------------------------------
# 主入口：生成一个案例
# ---------------------------------------------------------------------------


def generate_case(scn: Scenario, case_dir: str | Path) -> dict:
    """生成完整合成案例：地质图 + 全套真值。返回关键路径与元信息。"""
    case_dir = Path(case_dir)
    (case_dir / "input").mkdir(parents=True, exist_ok=True)
    (case_dir / "truth").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(scn.seed)

    ref = GeoRef(scn.img_w, scn.img_h, scn.world)
    U, V = np.meshgrid(np.arange(scn.img_w), np.arange(scn.img_h))
    X, Y = ref.px_to_world(U, V)

    # ---- 真值：DEM 与地表出露单元 ----
    dem = dem_at(scn, X, Y).astype(np.float32)                # (H,W)
    labels = unit_ids_at(scn, X, Y, dem)                      # (H,W) int16
    boundary = _boundary_mask(labels)                         # 1px 界线真值

    # ---- 渲染底图：单元填色 ----
    palette = np.array(
        [c for _n, _z, c in scn.layers] + [scn.basement_color], dtype=np.uint8
    )
    img = palette[labels]                                     # (H,W,3) RGB

    # ---- 等高线（虚线、深灰）----
    from skimage import measure

    z_min, z_max = float(dem.min()), float(dem.max())
    levels = np.arange(
        math.ceil(z_min / scn.contour_interval) * scn.contour_interval,
        z_max,
        scn.contour_interval,
    )
    contours_gt = []
    img_bgrlike = img  # cv2 画线不区分通道语义，保持 RGB 即可
    for lev in levels:
        for arr in measure.find_contours(dem, lev):
            # find_contours 返回 (row, col)=(v,u)；抽稀减小体积
            pts = [[float(p[1]), float(p[0])] for p in arr[::4]]
            if len(pts) < 5:
                continue
            contours_gt.append({"elevation": float(lev), "polyline_px": pts})
            _draw_dashed_polyline(
                img_bgrlike,
                pts,
                (80, 78, 74),
                scn.contour_w,
                scn.dash_on,
                scn.dash_off,
            )

    # ---- 河流（蓝色实线，干扰要素）----
    rivers = _make_rivers(scn, dem, rng)
    for pts in rivers:
        cv2.polylines(
            img_bgrlike,
            [np.asarray(pts, dtype=np.int32)],
            False,
            (105, 165, 225),
            2,
            lineType=cv2.LINE_AA,
        )

    # ---- 地质界线（黑色实线）----
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (scn.contact_w, scn.contact_w)
    )
    contact_mask = cv2.dilate(boundary, kernel)
    img[contact_mask > 0] = (35, 32, 30)

    # ---- 断层线（更粗的黑线，覆盖在界线之上）----
    fault_pts = _fault_trace_px(scn, ref, dem)
    if fault_pts:
        cv2.polylines(
            img,
            [np.asarray(fault_pts, dtype=np.int32)],
            False,
            (20, 18, 18),
            scn.fault_w,
            lineType=cv2.LINE_AA,
        )

    # ---- 产状符号（真实地质图的标配：走向短线+倾向刻度+倾角数字）----
    dip_symbols = _draw_dip_symbols(scn, img, labels, dem, rng)
    import pandas as pd

    pd.DataFrame(dip_symbols).to_csv(
        case_dir / "truth" / "dip_symbols_gt.csv", index=False
    )

    # ---- 单元代号文字（深灰，干扰要素）----
    if scn.draw_labels:
        for row in scn.unit_table():
            for (u, v) in _label_positions(labels, row["unit_id"], k=2):
                cv2.putText(
                    img,
                    row["name"],
                    (u - 14, v + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (50, 46, 44),
                    2,
                    lineType=cv2.LINE_AA,
                )

    # ---- 落盘：输入图 + 真值 ----
    map_path = case_dir / "input" / "map.png"
    cv2.imwrite(str(map_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    np.save(case_dir / "truth" / "labels_gt.npy", labels.astype(np.int16))
    np.save(case_dir / "truth" / "boundary_gt.npy", boundary)
    np.save(case_dir / "truth" / "dem_gt.npy", dem)
    with open(case_dir / "truth" / "contours_gt.json", "w", encoding="utf-8") as f:
        json.dump(contours_gt, f, ensure_ascii=False)

    # ---- 剖面真值（深部约束配准演示）：两条交叉剖面线，贴近真实图幅 ----
    x0w, x1w, y0w, y1w = scn.world
    section = render_section(scn, case_dir / "truth")
    section2 = render_section(
        scn, case_dir / "truth",
        p1=(x0w + 0.62 * (x1w - x0w), y0w + 120.0),
        p2=(x0w + 0.22 * (x1w - x0w), y1w - 120.0),
        suffix="2",
    )

    meta = {
        "scenario": asdict(scn),
        "georef": ref.to_dict(),
        "units": scn.unit_table(),
        "fault_trace_px": fault_pts,
        "fault_params": scn.fault,
        "n_contours": len(contours_gt),
        "section_frame": section["frame"],
        "section2_frame": section2["frame"],
        "dem_range": [z_min, z_max],
    }
    with open(case_dir / "truth" / "truth_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print(f"[mapgen] 案例 {scn.name} 已生成: {case_dir}")
    print(f"  地质图: {map_path} ({scn.img_w}x{scn.img_h})")
    print(f"  单元数: {scn.n_layers}+基底, 等高线: {len(contours_gt)} 条, "
          f"断层: {'有' if fault_pts else '无'}")
    return {"map_path": str(map_path), "meta": meta}


# ---------------------------------------------------------------------------
# 剖面图渲染（供 sectionreg 深部约束演示）
# ---------------------------------------------------------------------------


def render_section(
    scn: Scenario,
    out_dir: str | Path,
    p1: tuple | None = None,
    p2: tuple | None = None,
    suffix: str = "",
) -> dict:
    """沿剖面线 A–B 渲染真值剖面图 PNG + 配准 frame + 真值拾取点 CSV。

    拾取点 = 各层底界面在剖面图上的像素位置（模拟用户在剖面图上点选
    深部界线点），供 sectionreg.picks_to_surface_points 转成三维约束。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x0, x1, y0, y1 = scn.world
    if p1 is None:
        p1 = (x0 + 150.0, y0 + 0.35 * (y1 - y0))
    if p2 is None:
        p2 = (x1 - 150.0, y0 + 0.75 * (y1 - y0))

    # 绘图区尺寸与边距（像素）
    plot_w, plot_h = 720, 380
    ml, mr, mt, mb = 80, 40, 50, 60
    img_w, img_h = plot_w + ml + mr, plot_h + mt + mb
    z_min, z_max = scn.z_range

    frame = SectionFrame(
        p1_world=tuple(p1),
        p2_world=tuple(p2),
        img_w=img_w,
        img_h=img_h,
        margin_left=ml,
        margin_right=mr,
        margin_top=mt,
        margin_bottom=mb,
        z_min=z_min,
        z_max=z_max,
    )

    # 绘图区每个像素 → 世界坐标 → 单元 id
    us = np.arange(plot_w) + ml
    vs = np.arange(plot_h) + mt
    UU, VV = np.meshgrid(us, vs)
    XX, YY, ZZ = frame.px_to_world(UU, VV)
    units = unit_ids_at(scn, XX, YY, ZZ)
    dem_line = dem_at(scn, XX, YY)
    air = ZZ > dem_line

    palette = np.array(
        [c for _n, _z, c in scn.layers] + [scn.basement_color], dtype=np.uint8
    )
    plot = palette[units]
    plot[air] = (252, 252, 252)  # 地表以上留白

    # 界线：垂向相邻单元不同且都在地下 → 黑线
    edge = np.zeros(units.shape, dtype=bool)
    edge[1:, :] |= (units[1:, :] != units[:-1, :]) & ~air[1:, :] & ~air[:-1, :]
    plot[edge] = (30, 28, 26)
    # 地形线：空气/地下交界 → 深棕
    surf = np.zeros_like(edge)
    surf[1:, :] |= air[:-1, :] & ~air[1:, :]
    plot[surf] = (90, 60, 35)

    # 拼装整幅剖面图（白底 + 黑框 + 刻度）
    img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
    img[mt : mt + plot_h, ml : ml + plot_w] = plot
    cv2.rectangle(img, (ml, mt), (ml + plot_w, mt + plot_h), (0, 0, 0), 1)
    for z_tick in np.linspace(z_min, z_max, 6):
        _u, v = frame.world_to_px(p1[0], p1[1], z_tick)
        v = int(round(float(v)))
        cv2.line(img, (ml - 6, v), (ml, v), (0, 0, 0), 1)
        cv2.putText(
            img, f"{z_tick:.0f}", (8, v + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
        )
    cv2.putText(img, "A", (ml - 20, mt - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "B", (ml + plot_w + 5, mt - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    section_png = out_dir / f"section{suffix}_gt.png"
    cv2.imwrite(str(section_png), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # ---- 真值拾取点：各层底界面每隔 ~45px 一个 ----
    # 断层面附近留缓冲：layer_bottom_z 的盘别近似在断层楔区与逐点真值
    # 存在最大 throw 的偏差，且真实用户在剖面图上也不会紧贴断层拾取
    picks = []
    for i, (name, _z0, _c) in enumerate(scn.layers):
        for u in range(ml + 25, ml + plot_w - 25, 45):
            # u → 沿线世界位置
            xw, yw, _ = frame.px_to_world(float(u), float(mt))
            zb = float(layer_bottom_z(scn, i, xw, yw))
            if not (z_min + 15 < zb < z_max - 15):
                continue
            if zb > float(dem_at(scn, xw, yw)):  # 已被剥蚀，不存在
                continue
            if scn.fault is not None:
                if abs(xw - float(fault_x_at_z(scn, zb))) < 80.0:
                    continue  # 断层缓冲带内不拾取
            _, v = frame.world_to_px(xw, yw, zb)
            picks.append({"u": float(u), "v": float(v), "formation": name})

    import pandas as pd

    picks_df = pd.DataFrame(picks)
    picks_csv = out_dir / f"section{suffix}_picks_gt.csv"
    picks_df.to_csv(picks_csv, index=False)
    with open(out_dir / f"section{suffix}_frame.json", "w", encoding="utf-8") as f:
        json.dump(frame.to_dict(), f, ensure_ascii=False, indent=1)

    return {
        "section_png": str(section_png),
        "picks_csv": str(picks_csv),
        "frame": frame.to_dict(),
        "n_picks": len(picks),
    }


# ---------------------------------------------------------------------------
# 内置场景（覆盖不同难度，供鲁棒性与泛化测试）
# ---------------------------------------------------------------------------


def builtin_scenarios() -> dict[str, Scenario]:
    """六个内置场景：由易到难。"""
    scns = {}

    scns["base"] = Scenario(name="base")

    # 无断层 + 缓褶皱：最简单
    scns["gentle_nofault"] = Scenario(
        name="gentle_nofault",
        fault=None,
        fold_amp=55.0,
        regional_dip_deg=4.0,
        seed=7,
    )

    # 紧闭褶皱 + 大断距：条带窄、界线密
    scns["steep_narrow"] = Scenario(
        name="steep_narrow",
        fold_amp=130.0,
        fold_wavelength=750.0,
        regional_dip_deg=8.0,
        fault=dict(name="F1", x_ref=1050.0, z_ref=500.0,
                   dip=60.0, azimuth=90.0, throw=190.0),
        seed=11,
    )

    # 相近颜色：相邻地层色差小，考验聚类合并
    scns["similar_colors"] = Scenario(
        name="similar_colors",
        layers=(
            ("N1", 780.0, (247, 239, 196)),
            ("K2", 680.0, (196, 219, 148)),
            ("K1", 570.0, (176, 208, 142)),   # 与 K2 接近
            ("P2", 450.0, (238, 168, 122)),
            ("P1", 330.0, (228, 148, 108)),   # 与 P2 接近
        ),
        seed=23,
    )

    # 高起伏地形：等高线密集、界线锯齿（V 字法则显著）
    scns["high_relief"] = Scenario(
        name="high_relief",
        dem_amp1=170.0,
        dem_amp2=110.0,
        dem_amp3=60.0,
        seed=31,
    )

    # 平坦地形：无等高线信息 → 检验三点法退化时的兜底逻辑
    scns["flat_dem"] = Scenario(
        name="flat_dem",
        dem_flat=True,
        seed=37,
    )
    return scns
