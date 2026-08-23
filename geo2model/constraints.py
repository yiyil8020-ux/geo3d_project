# -*- coding: utf-8 -*-
"""
constraints.py — 接触线/断层/DEM → GemPy 建模约束
===================================================

产出（见 CONTRACTS.md，GemPy 直接可读）：
- db/surface_points.csv  层位控制点  X,Y,Z,formation
- db/orientations.csv    产状        X,Y,Z,azimuth,dip,polarity,formation
- db/model_config.json   建模配置（extent/resolution/series/断层）

关键地质约定：
1. GemPy 的 surface point 表示某单元的**底界面**。地质图上年轻单元 A
   与较老单元 B 的接触线，就是 A 的底界出露线 → formation = A（较年轻者）。
2. 层序差 |Δorder| > 1 的接触（如跨层юxta置）多半是断层接触或不整合，
   不能当普通层位点使用 → 跳过并记为可疑（防止污染插值）。
3. 产状三通道（半自动设计）：
   a) 人工通道：units.csv 中该单元的 dip/azimuth（用户从图面产状符号读数）
   b) 自动通道：接触线三点法推广——对有地形起伏的接触线，
      用其世界坐标点 (x,y,z) 最小二乘拟合平面 z=ax+by+c，
      平面梯度即倾向/倾角（经典构造地质"三点问题"的最小二乘版）
   c) 兜底通道：配置给定的默认产状（平坦地形等退化情形）
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from .georef import GeoRef
from .lines import resample_polyline


@dataclass
class ConstraintParams:
    """约束生成参数。"""

    sample_step_px: float = 16.0        # 接触线取点间隔（像素）
    fault_buffer_px: float = 12.0       # 距断层线多近的层位点要丢弃
    orient_per_formation: int = 3       # 每个地层最多产状数
    plane_fit_min_pts: int = 10         # 平面拟合最少点数
    plane_fit_min_relief: float = 10.0  # 拟合点高程极差下限（米），太平不可信
    min_dip: float = 1.5                # 拟合产状的可信倾角范围
    max_dip: float = 88.0
    default_dip: float | None = None    # 兜底产状（None=没有）
    default_azimuth: float | None = None
    max_sp_per_formation: int = 260     # 层位点上限（防止 GemPy 求解过慢）
    fault_depth_offsets: tuple = (130.0, 280.0, 430.0)  # 断层深部虚拟点
    z_margin: float = 15.0              # 点位离模型顶底的安全边距
    thickness_priors: dict | None = None  # 隐伏单元厚度先验 {"P1": 400.0}
    # 沿倾向下延投影（构造地质的剖面构建法）：把接触线采样点按邻近产状
    # 向深部投影生成虚拟层位点，弥补"无剖面图"图幅的深部约束空缺。
    # 两翼各自下延，同一界面的投影点在褶皱轴部自然汇合成闭合形态。
    depth_projection: tuple | None = None   # 下延距离列表(米)，如 (280, 620)
    depth_projection_radius: float = 650.0  # 采样点找邻近产状的最大水平距离(米)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _dem_sample(dem: np.ndarray, u, v):
    """双线性采样 DEM（dem 与图像同尺寸同朝向）。"""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return map_coordinates(dem, [v.ravel(), u.ravel()], order=1, mode="nearest").reshape(u.shape)


def _gradient_to_dip_azimuth(a: float, b: float) -> tuple[float, float]:
    """平面 z = a·x + b·y + c 的梯度 → (azimuth, dip)。

    下倾方向 = 高程下降最快的水平方向 = -(a, b)；
    azimuth 以北为 0、顺时针（东=90）：atan2(东分量, 北分量)。
    """
    dip = math.degrees(math.atan(math.hypot(a, b)))
    az = math.degrees(math.atan2(-a, -b)) % 360.0
    return az, dip


def _fit_plane_orientation(w: np.ndarray, params: "ConstraintParams"):
    """对一段三维点 (n,3) 做最小二乘平面拟合 → (score, center, az, dip)。

    质控：
    - 高程极差不足（太平）不可信 → None
    - 平面点在水平面上的展布要足够二维（第二奇异值下限），
      否则共线点的拟合病态 → None
    - 倾角超出可信范围 → None
    """
    relief = float(w[:, 2].max() - w[:, 2].min())
    if relief < params.plane_fit_min_relief:
        return None
    xy = w[:, :2] - w[:, :2].mean(axis=0)
    sv = np.linalg.svd(xy, compute_uv=False)
    if len(sv) < 2 or sv[1] < 15.0:  # 水平展布第二主轴 < 15 m → 近共线
        return None
    A = np.stack([w[:, 0], w[:, 1], np.ones(len(w))], axis=1)
    coef, res, *_ = np.linalg.lstsq(A, w[:, 2], rcond=None)
    az, dip = _gradient_to_dip_azimuth(float(coef[0]), float(coef[1]))
    if not (params.min_dip <= dip <= params.max_dip):
        return None
    rms = float(np.sqrt(res[0] / len(w))) if len(res) else 0.0
    center = w.mean(axis=0)
    score = relief / (rms + 3.0)
    return score, center, az, dip


def _fault_dist_map(shape: tuple, faults: list[dict]) -> np.ndarray:
    """每个像素到最近确认断层线的距离（无断层时全为 +inf）。"""
    h, w = shape
    if not faults:
        return np.full((h, w), np.inf, dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    for fl in faults:
        pts = np.asarray(fl["polyline_px"], dtype=np.int32)
        cv2.polylines(mask, [pts], False, 1, 1)
    # distanceTransform 计算的是「到零像素」的距离 → 取反
    return cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def build_constraints(
    extract_dir: str | Path,
    db_dir: str | Path,
    georef: GeoRef,
    z_range: tuple[float, float],
    resolution: tuple[int, int, int] = (60, 60, 60),
    params: ConstraintParams | None = None,
    extra_surface_points: pd.DataFrame | None = None,
    extra_orientations: pd.DataFrame | None = None,
) -> dict:
    """接触线 + 断层 + DEM + 单元表 → GemPy 三件套。返回统计摘要。

    extra_surface_points: 额外深部层位点（剖面图配准/钻孔），列 X,Y,Z,formation。
    必须在此处合入（而不是事后追加 CSV），层序判定才能把
    「仅有深部点的地层」也正确纳入 series。

    extra_orientations: 人工录入的产状（图面产状符号读数），
    列 X,Y,Z,azimuth,dip,polarity,formation。人工读数与接触线拟合
    产状**并用**（实测二者互补：符号点覆盖单元内部、拟合点贴合界线）。
    """
    params = params or ConstraintParams()
    extract_dir, db_dir = Path(extract_dir), Path(db_dir)

    units = pd.read_csv(db_dir / "units.csv")
    with open(extract_dir / "contacts.json", encoding="utf-8") as f:
        contacts = json.load(f)
    faults_all_path = db_dir / "faults.json"
    faults = []
    if faults_all_path.is_file():
        with open(faults_all_path, encoding="utf-8") as f:
            faults = [fl for fl in json.load(f) if fl.get("confirmed")]
    dem = np.load(extract_dir / "dem.npy")

    strata = units[units["role"] == "strata"].copy()
    # order<0 是"非地层"哨兵值：地层单元必须由人工审核填好层序，
    # 否则会被当成全局最年轻层导致接触归属系统性错误 → 剔除并警告
    bad_order = strata["order"] < 0
    if bad_order.any():
        print(f"[constraints] 警告: 下列地层单元 order<0（未审核层序），"
              f"不参与建模: {strata.loc[bad_order, 'name'].tolist()}")
        strata = strata[~bad_order]
    strata = strata.sort_values("order")
    if "series" not in strata.columns:
        strata["series"] = "Strat_Series"
    strata["series"] = strata["series"].fillna("Strat_Series")
    id2name = dict(zip(strata["unit_id"].astype(int), strata["name"]))
    id2order = dict(zip(strata["unit_id"].astype(int), strata["order"].astype(int)))
    id2series = dict(zip(strata["unit_id"].astype(int), strata["series"]))
    # 同名单元（多簇合并）去重后的名称序列，新 → 老
    names_by_age = list(dict.fromkeys(strata["name"].tolist()))
    name2series = dict(zip(strata["name"], strata["series"]))
    name2order = dict(zip(strata["name"], strata["order"].astype(int)))
    # 每个系列最老的成员（不整合系列的"底面"归属对象）
    series_oldest: dict[str, str] = {}
    for nm in names_by_age:
        series_oldest[name2series[nm]] = nm  # 越靠后越老，最后写入的即最老

    fault_dist = _fault_dist_map(dem.shape, faults)

    # ---------------- 层位控制点 ----------------
    sp_rows: list[dict] = []
    suspect_contacts: list[int] = []
    # 记录每条接触线的世界坐标采样（供产状拟合复用）
    formation_lines: dict[str, list[np.ndarray]] = {}

    for c in contacts:
        a, b = c["unit_a"], c["unit_b"]
        if a not in id2order or b not in id2order:
            continue  # 非地层单元（水体/忽略簇）的边界
        d_order = abs(id2order[a] - id2order[b])
        if d_order == 0:
            continue
        if id2series[a] == id2series[b]:
            # 同系列（整合序列）：只有相邻层的接触是底界出露线；
            # 跨层接触 = 断层错断/绘图误差嫌疑，跳过防止污染插值
            if d_order > 1:
                suspect_contacts.append(c["contact_id"])
                continue
            formation = id2name[a] if id2order[a] < id2order[b] else id2name[b]
        else:
            # 跨系列（不整合）：年轻系列以侵蚀面盖在老系列之上，
            # 接触线是"年轻系列最老成员"的底面出露线（该处更年轻的
            # 成员已被剥蚀缺失时，底面位置不变）
            young_id = a if id2order[a] < id2order[b] else b
            formation = series_oldest[id2series[young_id]]

        pts = resample_polyline(c["polyline_px"], params.sample_step_px)
        pts = np.asarray(pts, dtype=float)
        if len(pts) == 0:
            continue
        # 丢弃断层缓冲带内的点（断层错断处的接触位置不代表原始层面）
        ui = np.clip(pts[:, 0].astype(int), 0, dem.shape[1] - 1)
        vi = np.clip(pts[:, 1].astype(int), 0, dem.shape[0] - 1)
        keep = fault_dist[vi, ui] > params.fault_buffer_px
        pts = pts[keep]
        if len(pts) == 0:
            continue
        x, y = georef.px_to_world(pts[:, 0], pts[:, 1])
        z = _dem_sample(dem, pts[:, 0], pts[:, 1])
        z = np.clip(z, z_range[0] + params.z_margin, z_range[1] - params.z_margin)

        world = np.stack([x, y, z], axis=1)
        formation_lines.setdefault(formation, []).append(world)
        for xi, yi, zi in world:
            sp_rows.append(
                {"X": round(float(xi), 2), "Y": round(float(yi), 2),
                 "Z": round(float(zi), 2), "formation": formation}
            )

    sp = pd.DataFrame(sp_rows)
    # 每层控制点太多时抽稀（对 GemPy 求解速度影响很大）
    # 步长用向上取整，保证抽稀后不超过 max_sp_per_formation
    if not sp.empty:
        sp = (
            sp.groupby("formation", group_keys=False)[sp.columns]
            .apply(lambda g: g.iloc[:: max(
                1, -(-len(g) // params.max_sp_per_formation))])
            .reset_index(drop=True)
        )

    # 深部约束点（剖面配准/钻孔）先合入，厚度先验后判断——
    # 保证"已有实测深部点的地层不再用先验"这条守卫能看到全部实测点
    if extra_surface_points is not None and len(extra_surface_points) > 0:
        known = set(strata["name"])
        extra = extra_surface_points[
            extra_surface_points["formation"].isin(known)
        ][["X", "Y", "Z", "formation"]].copy()
        unknown = set(extra_surface_points["formation"]) - known
        if unknown:
            print(f"[constraints] 警告: 深部点中未知地层 {unknown}，已忽略")
        sp = pd.concat([sp, extra], ignore_index=True)
        print(f"[constraints] 已合入深部层位点 {len(extra)} 个")

    # 厚度先验通道：
    # 底界从不出露的隐伏单元（无钻孔/剖面时）可由用户给出估计厚度，
    # 用其上覆相邻单元的底界采样点整体下移生成虚拟底界点。
    # 例：P1 底界隐伏，用户估计 P1 厚 400 m → P1 底界点 = P2 底界点 z-400。
    if params.thickness_priors:
        for f_name, thick in params.thickness_priors.items():
            if f_name not in name2order:
                print(f"[constraints] 警告: 厚度先验的地层 {f_name} 不存在，跳过")
                continue
            if not sp.empty and (sp["formation"] == f_name).any():
                continue  # 已有实测点则不用先验
            # 找同系列中紧邻的年轻单元（其底界即本单元顶界）
            upper = None
            for nm in names_by_age:
                if (name2series[nm] == name2series[f_name]
                        and name2order[nm] == name2order[f_name] - 1):
                    upper = nm
                    break
            if upper is None or sp.empty or not (sp["formation"] == upper).any():
                print(f"[constraints] 警告: {f_name} 厚度先验缺少上覆 {upper} 的底界点，跳过")
                continue
            base = sp[sp["formation"] == upper].copy()
            base["Z"] = (base["Z"] - float(thick)).clip(
                lower=z_range[0] + params.z_margin)
            base["formation"] = f_name
            base = base.iloc[::2]  # 先验点密度减半（它只是软约束）
            sp = pd.concat([sp, base], ignore_index=True)
            print(f"[constraints] 厚度先验: {f_name} = {upper} 底界下移 "
                  f"{thick} m（{len(base)} 点）")

    # ---------------- 产状 ----------------
    # 通道优先级：图面符号人工读数 > units.csv 单值 > 接触线分窗拟合 > 深部点拟合
    ori_rows: list[dict] = []
    manual_formations: set = set()
    if extra_orientations is not None and len(extra_orientations) > 0:
        known = set(strata["name"])
        for _, r in extra_orientations.iterrows():
            if r["formation"] not in known:
                continue
            ori_rows.append(
                {"X": round(float(r["X"]), 2), "Y": round(float(r["Y"]), 2),
                 "Z": round(float(r["Z"]), 2), "azimuth": float(r["azimuth"]),
                 "dip": float(r["dip"]), "polarity": int(r.get("polarity", 1)),
                 "formation": r["formation"]}
            )
            manual_formations.add(r["formation"])
        print(f"[constraints] 人工产状读数 {len(ori_rows)} 条 "
              f"(覆盖地层: {sorted(manual_formations)})")

    name2userdip = {
        r["name"]: (r["dip"], r["azimuth"])
        for _, r in strata.iterrows()
        if pd.notna(r["dip"]) and pd.notna(r["azimuth"])
    }

    # 说明：人工读数与接触线拟合产状**并用**（实测二者互补——
    # 符号点覆盖单元内部，拟合点贴合界线，同时保留时场约束最完整）
    for formation, lines_w in formation_lines.items():
        rows_f: list[dict] = []
        if formation in name2userdip:
            # (a) 人工读数通道：产状放在最长接触线的分位点上
            dip, az = name2userdip[formation]
            longest = max(lines_w, key=len)
            for frac in np.linspace(0.15, 0.85, params.orient_per_formation):
                p = longest[int(frac * (len(longest) - 1))]
                rows_f.append(
                    {"X": round(float(p[0]), 2), "Y": round(float(p[1]), 2),
                     "Z": round(float(p[2]), 2), "azimuth": float(az),
                     "dip": float(dip), "polarity": 1, "formation": formation}
                )
        else:
            # (b) 三点法（最小二乘平面拟合）通道 —— 分窗拟合
            # 整条接触线可能绕褶皱鼻端弯曲（非平面），整体拟合会退化；
            # 沿线取滑动窗口分段拟合，两翼各自得到可信的局部产状。
            cands = []
            win = max(params.plane_fit_min_pts, 12)
            for w in lines_w:
                if len(w) < params.plane_fit_min_pts:
                    continue
                step = max(win // 2, 1)
                for s in range(0, max(len(w) - win + 1, 1), step):
                    seg = w[s: s + win]
                    if len(seg) < params.plane_fit_min_pts:
                        continue
                    fit = _fit_plane_orientation(seg, params)
                    if fit is not None:
                        cands.append(fit)
            # 按质量分挑选，且彼此保持 250 m 以上间距（避免产状扎堆）
            cands.sort(key=lambda t: t[0], reverse=True)
            chosen: list = []
            for score, center, az, dip in cands:
                if len(chosen) >= params.orient_per_formation:
                    break
                if all(np.hypot(center[0] - c[1][0], center[1] - c[1][1]) > 250.0
                       for c in chosen):
                    chosen.append((score, center, az, dip))
            for _score, center, az, dip in chosen:
                rows_f.append(
                    {"X": round(float(center[0]), 2), "Y": round(float(center[1]), 2),
                     "Z": round(float(center[2]), 2), "azimuth": round(az, 2),
                     "dip": round(dip, 2), "polarity": 1, "formation": formation}
                )
        if not rows_f and params.default_dip is not None:
            # (c) 兜底通道
            longest = max(lines_w, key=len)
            p = longest[len(longest) // 2]
            rows_f.append(
                {"X": round(float(p[0]), 2), "Y": round(float(p[1]), 2),
                 "Z": round(float(p[2]), 2),
                 "azimuth": float(params.default_azimuth or 90.0),
                 "dip": float(params.default_dip), "polarity": 1,
                 "formation": formation}
            )
        ori_rows.extend(rows_f)

    # (b') 隐伏地层产状兜底：只有深部点（剖面拾取/钻孔）而无出露接触线的
    # 地层（典型：从不出露的最下部地层），用深部点的局部邻域拟合产状。
    # 两条相交剖面的拾取点在交叉区具备二维展布，可以稳定拟合。
    have_ori_names = {r["formation"] for r in ori_rows}
    if not sp.empty:
        from scipy.spatial import cKDTree

        for formation in set(sp["formation"]) & set(names_by_age):
            if formation in have_ori_names:
                continue
            pts_f = sp[sp["formation"] == formation][["X", "Y", "Z"]].to_numpy(float)
            if len(pts_f) < params.plane_fit_min_pts:
                continue
            tree = cKDTree(pts_f[:, :2])
            cands = []
            for p in pts_f:
                k = min(14, len(pts_f))
                _, idx = tree.query(p[:2], k=k)
                fit = _fit_plane_orientation(pts_f[idx], params)
                if fit is not None:
                    cands.append(fit)
            cands.sort(key=lambda t: t[0], reverse=True)
            chosen = []
            for score, center, az, dip in cands:
                if len(chosen) >= 2:
                    break
                if all(np.hypot(center[0] - c[1][0], center[1] - c[1][1]) > 250.0
                       for c in chosen):
                    chosen.append((score, center, az, dip))
            for _score, center, az, dip in chosen:
                ori_rows.append(
                    {"X": round(float(center[0]), 2), "Y": round(float(center[1]), 2),
                     "Z": round(float(center[2]), 2), "azimuth": round(az, 2),
                     "dip": round(dip, 2), "polarity": 1, "formation": formation}
                )
                print(f"[constraints] 隐伏地层 {formation} 由深部点拟合产状: "
                      f"dip={dip:.1f}°, az={az:.0f}°")

    ori = pd.DataFrame(ori_rows)

    # ---------------- 断层约束 ----------------
    fault_series: dict[str, list[str]] = {}
    for fl in faults:
        fname = fl["name"]
        dip, az = float(fl["dip"]), float(fl["azimuth"])
        pts = resample_polyline(fl["polyline_px"], params.sample_step_px * 2.5)
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            continue
        x, y = georef.px_to_world(pts[:, 0], pts[:, 1])
        z = _dem_sample(dem, pts[:, 0], pts[:, 1])
        # 地表迹线点
        for xi, yi, zi in zip(x, y, z):
            sp_rows_f = {"X": round(float(xi), 2), "Y": round(float(yi), 2),
                         "Z": round(float(zi), 2), "formation": fname}
            sp = pd.concat([sp, pd.DataFrame([sp_rows_f])], ignore_index=True)
        # 沿倾向的深部虚拟点：把面延到深处，避免断层面只有地表一排点
        sin_az, cos_az = math.sin(math.radians(az)), math.cos(math.radians(az))
        for dz in params.fault_depth_offsets:
            if dz > (z_range[1] - z_range[0]) * 0.8:
                continue
            dxy = dz / math.tan(math.radians(dip))
            for xi, yi, zi in zip(x[::3], y[::3], z[::3]):
                zi2 = float(zi - dz)
                if zi2 < z_range[0] + params.z_margin:
                    continue
                sp = pd.concat(
                    [sp, pd.DataFrame([{
                        "X": round(float(xi + dxy * sin_az), 2),
                        "Y": round(float(yi + dxy * cos_az), 2),
                        "Z": round(zi2, 2), "formation": fname}])],
                    ignore_index=True,
                )
        # 断层产状（沿迹线取两个分位点）
        for frac in (0.3, 0.7):
            k = int(frac * (len(x) - 1))
            ori = pd.concat(
                [ori, pd.DataFrame([{
                    "X": round(float(x[k]), 2), "Y": round(float(y[k]), 2),
                    "Z": round(float(max(z[k] - 150.0, z_range[0] + 40.0)), 2),
                    "azimuth": az, "dip": dip, "polarity": 1,
                    "formation": fname}])],
                ignore_index=True,
            )
        fault_series[f"Fault_{fname}"] = [fname]

    # ---------------- 层序与配置 ----------------
    # 只有有层位点的地层才能进 series；全局最老单元作为 GemPy 基底
    # （基底没有底界面，无需/不能有层位点）
    have_pts = set(sp["formation"].unique()) if not sp.empty else set()
    basement_name = names_by_age[-1] if names_by_age else "basement"
    modelable = [n for n in names_by_age[:-1] if n in have_pts]
    dropped = [n for n in names_by_age[:-1] if n not in have_pts]
    if dropped:
        print(f"[constraints] 警告: 下列地层无层位点，被移出层序: {dropped}")

    # 按 units.csv 的 series 列分组（不整合系列各自成组），组间按
    # 最年轻成员排序（新组在前=GemPy 侵蚀关系的上覆组）
    strat_series_map: dict[str, list[str]] = {}
    for n in modelable:  # names_by_age 有序 → 每组成员自然新→老
        strat_series_map.setdefault(name2series[n], []).append(n)
    series_group_order = sorted(
        strat_series_map.keys(),
        key=lambda s: min(name2order[n] for n in strat_series_map[s]),
    )
    series_strata = modelable  # 摘要用：全部可建模地层（新→老）

    # 一致性保障：surface_points / orientations 里绝不能出现
    # 「不在层序也不是断层」的地层，否则 GemPy 建模直接崩溃。
    # 注意 sp 与 ori 要**分别**过滤——例如人工产状符号落在基底出露区时，
    # 该产状的 formation（基底）没有任何层位点，只清 sp 会漏掉它。
    allowed = set(modelable) | set(fault_series and
                                   [n for v in fault_series.values() for n in v] or [])
    bad_sp = set(sp["formation"].unique()) - allowed if not sp.empty else set()
    if bad_sp:
        print(f"[constraints] 警告: 丢弃无法入序的层位点 {sorted(bad_sp)} "
              f"(共 {int(sp['formation'].isin(bad_sp).sum())} 个)")
        sp = sp[~sp["formation"].isin(bad_sp)].reset_index(drop=True)
    bad_ori = set(ori["formation"].unique()) - allowed if not ori.empty else set()
    if bad_ori:
        print(f"[constraints] 警告: 丢弃无法入序的产状 {sorted(bad_ori)} "
              f"(共 {int(ori['formation'].isin(bad_ori).sum())} 条)")
        ori = ori[~ori["formation"].isin(bad_ori)].reset_index(drop=True)

    # 产状兜底校验：GemPy 要求**每个 series** 至少有一个产状。
    # 缺产状的系列（典型：近水平的不整合盖层，如第四系 Q）插入一个
    # 近水平默认产状并警告——盖层近水平在地质上通常也是合理近似。
    have_ori = set(ori["formation"].unique()) if not ori.empty else set()
    for s_name, members in strat_series_map.items():
        if set(members) & have_ori:
            continue
        # 该系列没有任何产状 → 挑层位点最多的成员，在其一个层位点处兜底
        counts = sp["formation"].value_counts() if not sp.empty else {}
        member = max(members, key=lambda n: int(counts.get(n, 0)))
        rows_m = sp[sp["formation"] == member]
        if rows_m.empty:
            continue
        p = rows_m.iloc[len(rows_m) // 2]
        dip0 = float(params.default_dip) if params.default_dip is not None else 5.0
        az0 = float(params.default_azimuth) if params.default_azimuth is not None else 90.0
        print(f"[constraints] 警告: 系列 {s_name} 无产状，为 {member} 插入 "
              f"默认产状 dip={dip0}°（可在 units.csv 人工指定）")
        ori = pd.concat(
            [ori, pd.DataFrame([{
                "X": float(p["X"]), "Y": float(p["Y"]), "Z": float(p["Z"]),
                "azimuth": az0, "dip": dip0, "polarity": 1,
                "formation": member}])],
            ignore_index=True,
        )
        have_ori.add(member)

    # ---------------- 沿倾向下延投影（深部虚拟层位点） ----------------
    if params.depth_projection and not sp.empty and not ori.empty:
        from scipy.spatial import cKDTree

        x0g, x1g, y0g, y1g = georef.world
        ori_strata = ori[ori["formation"].isin(names_by_age)]
        if len(ori_strata) > 0:
            tree = cKDTree(ori_strata[["X", "Y"]].to_numpy(float))
            proj_rows = []
            sp_strata = sp[sp["formation"].isin(names_by_age)]
            for _, r in sp_strata.iloc[::3].iterrows():  # 每 3 点投 1 点，控制规模
                d, k = tree.query([r["X"], r["Y"]])
                if d > params.depth_projection_radius:
                    continue  # 附近没有产状，姿态未知，不投影
                o = ori_strata.iloc[int(k)]
                az = math.radians(float(o["azimuth"]))
                dip = math.radians(float(o["dip"]))
                for L in params.depth_projection:
                    # 沿倾向向下的单位向量 = (sinAz·cosDip, cosAz·cosDip, -sinDip)
                    xn = float(r["X"]) + math.sin(az) * math.cos(dip) * L
                    yn = float(r["Y"]) + math.cos(az) * math.cos(dip) * L
                    zn = float(r["Z"]) - math.sin(dip) * L
                    if not (x0g < xn < x1g and y0g < yn < y1g):
                        continue
                    if zn < z_range[0] + params.z_margin:
                        continue
                    proj_rows.append({"X": round(xn, 2), "Y": round(yn, 2),
                                      "Z": round(zn, 2),
                                      "formation": r["formation"]})
            if proj_rows:
                sp = pd.concat([sp, pd.DataFrame(proj_rows)], ignore_index=True)
                print(f"[constraints] 沿倾向下延投影: 追加深部虚拟层位点 "
                      f"{len(proj_rows)} 个 (距离档 {params.depth_projection})")

    x0, x1, y0, y1 = georef.world
    model_config = {
        "extent": [x0, x1, y0, y1, float(z_range[0]), float(z_range[1])],
        "resolution": list(resolution),
        "series": {**fault_series,
                   **{s: strat_series_map[s] for s in series_group_order}},
        "fault_series": list(fault_series.keys()),
        "basement_name": basement_name,
        "georef": georef.to_dict(),
        "suspect_contact_ids": suspect_contacts,
        "params": asdict(params),
    }

    db_dir.mkdir(parents=True, exist_ok=True)
    if sp.empty:
        sp = pd.DataFrame(columns=["X", "Y", "Z", "formation"])
    if ori.empty:
        ori = pd.DataFrame(columns=["X", "Y", "Z", "azimuth", "dip", "polarity", "formation"])
    sp.to_csv(db_dir / "surface_points.csv", index=False)
    ori.to_csv(db_dir / "orientations.csv", index=False)
    with open(db_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=1)

    if len(sp) == 0:
        raise ValueError("未能提取到有效的地层界面控制点。请检查单元审核表中的层序（order）设置是否正确。")

    summary = {
        "n_surface_points": len(sp),
        "n_orientations": len(ori),
        "formations": {k: int(v) for k, v in sp["formation"].value_counts().items()} if not sp.empty else {},
        "series_strata": series_strata,
        "basement": basement_name,
        "n_faults": len(fault_series),
        "n_suspect_contacts": len(suspect_contacts),
    }
    print(f"[constraints] 层位点 {summary['n_surface_points']}, "
          f"产状 {summary['n_orientations']}, 断层 {summary['n_faults']}, "
          f"可疑跨层接触 {summary['n_suspect_contacts']} 条")
    return summary
