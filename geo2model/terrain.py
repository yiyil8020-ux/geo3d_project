# -*- coding: utf-8 -*-
"""
terrain.py — 虚线等高线提取与 DEM 重建
========================================

地形信息是「地质图 → 三维模型」的 Z 轴来源。本模块解决两件事：

1. 等高线提取（extract_contours）
   地质图上等高线是**深灰色虚线**，而地质界线/断层是黑色**实线**、
   注记文字是紧凑的深灰笔画块。三者在暗色掩膜里混在一起，
   区分的关键是**形态学差异**：
       - 虚线由一串细长小段（dash）组成：单段面积小、长宽比大；
       - 实线是贯穿全图的大连通域（面积远超单个 dash）；
       - 文字是紧凑团块（长宽比小），且不会排成长链。
   因此流程为：暗掩膜 → 连通域筛 dash → 按"距离近 + 方向顺"把相邻
   dash 连接成链 → 链足够长才认定为等高线 → 链上质心序列即折线。

2. DEM 重建（contours_to_dem）
   等高线赋高程后，把折线点当作 (u,v,z) 散点，在粗网格上做
   线性插值（凸包外用最近邻补齐），高斯平滑后放大回全图尺寸。

坐标约定与 CONTRACTS.md 一致：像素 (u,v)，u=列（向右），v=行（向下）。
contours.json 契约：[{"contour_id":i,"elevation":None|float,
                     "polyline_px":[[u,v],...]}, ...]
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from . import lines


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------


@dataclass
class TerrainParams:
    dark_gray_max: int = 110
    dash_area_min: int = 4        # 虚线小段面积范围
    dash_area_max: int = 260
    dash_aspect_min: float = 1.5  # 小段长宽比下限（区分文字块）
    dot_area_max: int = 0         # ≤该面积的组件按"无方向圆点"接受（点线
                                  # 等高线用；0=关闭。圆点免长宽比/方向检查）
    link_dist: float = 18.0       # 相邻虚线段中心最大连接距离(px)
    link_angle_deg: float = 42.0  # 连接方向与段方向的最大夹角
    min_chain_dashes: int = 5     # 成链最少段数（滤掉文字笔画）
    grid_step: int = 4            # DEM 插值网格步长(px)
    smooth_sigma: float = 2.0     # DEM 平滑（在粗网格上的高斯σ）


# ---------------------------------------------------------------------------
# 第一步：暗掩膜 + 连通域 → 虚线小段（dash）候选
# ---------------------------------------------------------------------------


def _dash_segments(img_rgb: np.ndarray, params: TerrainParams):
    """从 RGB 图中提取虚线小段候选：返回 (质心数组 N×2, 单位主方向 N×2)。

    原理：
    - 等高线颜色约 (80,78,74)，灰度 ≈78 < dark_gray_max，会进暗掩膜；
      河流是蓝色（灰度 ≈154）不会混入。
    - 黑色实线（界线 3px / 断层 7px）在图上纵横贯通，形成**大面积**
      连通域（数千至数万像素），被 dash_area_max 拒掉；顺带地，凡是
      与实线相交的 dash 也会被并进大连通域而丢失——这正是等高线链
      在交叉处被截断、提取条数多于真值条数的原因。
    - 文字笔画粗（thickness=2）且横竖勾连，单字符连通域紧凑，
      minAreaRect 长宽比小，被 dash_aspect_min 拒掉；个别细长字符
      （如 "1"）漏进来也没关系，后面的"成链段数"过滤兜底。
    - 主方向用 PCA（协方差最大特征向量）估计：dash 是细长条，
      像素分布的第一主轴就是它的走向，对 2px 宽的小段比
      minAreaRect 的角度量化更稳。
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray < params.dark_gray_max).astype(np.uint8)

    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    centroids, directions = [], []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (params.dash_area_min <= area <= params.dash_area_max):
            continue
        # 只在组件包围盒内找像素（比全图 np.where 快得多）
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y0 = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        ys, xs = np.nonzero(lab[y0 : y0 + h, x0 : x0 + w] == i)
        pts = np.column_stack([xs + x0, ys + y0]).astype(np.float32)  # (u,v)

        # 点线模式：足够小的组件按"无方向圆点"接受（方向记 (0,0) 哨兵，
        # 连接阶段对它免方向检查）。真实图幅的等高线常用点线而非虚线。
        if params.dot_area_max > 0 and area <= params.dot_area_max:
            centroids.append(pts.mean(axis=0))
            directions.append(np.zeros(2))
            continue

        # 长宽比过滤：minAreaRect 给出最小外接旋转矩形的 (宽,高)
        (_c, (rw, rh), _a) = cv2.minAreaRect(pts)
        long_side, short_side = max(rw, rh), min(rw, rh)
        if long_side / max(short_side, 1.0) < params.dash_aspect_min:
            continue  # 紧凑团块 → 文字/杂点

        # PCA 主方向：中心化后协方差矩阵的最大特征向量
        c = pts.mean(axis=0)
        d = pts - c
        cov = d.T @ d / len(d)
        evals, evecs = np.linalg.eigh(cov)     # eigh 按特征值升序
        vec = evecs[:, -1]                     # 最大特征值对应的方向
        centroids.append(c)
        directions.append(vec / (np.linalg.norm(vec) + 1e-12))

    if not centroids:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.asarray(centroids, dtype=float), np.asarray(directions, dtype=float)


# ---------------------------------------------------------------------------
# 第二步：邻段连接 → 链
# ---------------------------------------------------------------------------


def _link_dashes(cents: np.ndarray, dirs: np.ndarray, params: TerrainParams):
    """把相邻且方向顺滑的 dash 两两连接，返回边集合 set[(i,j)] (i<j)。

    连接规则（三重约束，防止串线）：
    1. 距离：质心间距 ≤ link_dist。虚线周期 = dash_on+dash_off = 16px，
       相邻 dash 质心间距约 16px，默认 18px 刚好容纳（含弯曲拉伸）。
    2. 方向兼容：连接向量与**两个** dash 的主方向夹角都要 < link_angle_deg。
       主方向有 180° 对称性（PCA 给的向量正负号任意），所以用
       |cos| 取绝对夹角。相邻两条平行等高线即使距离够近，
       连接向量也近乎垂直于走向（夹角≈90°），会被拒绝——
       这是防止不同高程等高线互相串接的关键。
    3. 每段每侧最多连 1 个：把连接向量投影到该段主方向上，
       按正负号分成"前/后"两侧，每侧只保留距离最近的兼容邻居；
       且要求**双向互选**（i 选了 j 且 j 也选了 i 才成边）。
       互选保证每个节点度 ≤2，链必然是简单路径或闭环，
       无需再处理分叉。
    """
    n = len(cents)
    if n == 0:
        return set()
    tree = cKDTree(cents)
    pairs = tree.query_pairs(params.link_dist)

    cos_max = np.cos(np.radians(params.link_angle_deg))  # 夹角小 ⇔ |cos| 大
    best = np.full((n, 2), -1, dtype=int)      # best[i][side] = 最近兼容邻居
    best_d = np.full((n, 2), np.inf)
    iso = np.hypot(dirs[:, 0], dirs[:, 1]) < 0.5   # 无方向圆点（点线模式）
    # 圆点的候选邻居表：{i: [(dist, j), ...]}，稍后按"最近 + 反向侧次近"选边
    iso_cand: dict[int, list] = {}

    for i, j in pairs:
        vec = cents[j] - cents[i]
        dist = float(np.hypot(vec[0], vec[1]))
        if dist < 1e-9:
            continue
        u = vec / dist
        # 180° 对称：|cos| ≥ cos(link_angle) 即方向兼容；圆点免方向检查
        if not iso[i] and abs(float(u @ dirs[i])) < cos_max:
            continue
        if not iso[j] and abs(float(u @ dirs[j])) < cos_max:
            continue
        # 有方向的段：按主方向投影正负分侧，每侧留最近
        if not iso[i]:
            side_i = 0 if float(vec @ dirs[i]) >= 0 else 1   # j 在 i 的哪一侧
            if dist < best_d[i, side_i]:
                best_d[i, side_i], best[i, side_i] = dist, j
        else:
            iso_cand.setdefault(i, []).append((dist, j))
        if not iso[j]:
            side_j = 0 if float(-vec @ dirs[j]) >= 0 else 1  # i 在 j 的哪一侧
            if dist < best_d[j, side_j]:
                best_d[j, side_j], best[j, side_j] = dist, i
        else:
            iso_cand.setdefault(j, []).append((dist, i))

    # 圆点分侧（旋转不变，修复"固定斜轴在链走向垂直于轴时退化"的缺陷）：
    # 槽0 = 最近邻；槽1 = 与槽0方向夹角 > 100° 的最近邻（即"另一侧"的
    # 前进方向）。任意走向的直线点列与闭合点线圈都能得到前后各一邻居。
    for i, cand in iso_cand.items():
        cand.sort()
        d0, j0 = cand[0]
        best_d[i, 0], best[i, 0] = d0, j0
        u0 = (cents[j0] - cents[i]) / max(d0, 1e-9)
        for d1, j1 in cand[1:]:
            u1 = (cents[j1] - cents[i]) / max(d1, 1e-9)
            if float(u0 @ u1) < -0.17:  # 夹角 > 100° → 反向侧
                best_d[i, 1], best[i, 1] = d1, j1
                break

    edges = set()
    for i in range(n):
        for side in (0, 1):
            j = best[i, side]
            if j >= 0 and (best[j, 0] == i or best[j, 1] == i):  # 双向互选
                edges.add((min(i, j), max(i, j)))
    return edges


def _edges_to_chains(n: int, edges: set):
    """图遍历把边集合分解成有序链，返回 list[(节点顺序列表, 是否闭环)]。

    因为互选机制保证度 ≤2，图只有两种连通形态：
    - 简单路径：从度 1 的端点出发一路走到另一端点；
    - 闭环（闭合等高线）：所有节点度都是 2，从任意节点出发绕一圈。
    """
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    visited = [False] * n
    chains = []

    def walk(start: int):
        """从 start 沿邻接关系走到头，返回节点顺序。"""
        order = [start]
        visited[start] = True
        prev, cur = -1, start
        while True:
            nxt = -1
            for nb in adj[cur]:
                if nb != prev and not visited[nb]:
                    nxt = nb
                    break
            if nxt < 0:
                return order
            visited[nxt] = True
            order.append(nxt)
            prev, cur = cur, nxt

    # 先处理开链：从端点（度 1）出发
    for i in range(n):
        if not visited[i] and len(adj[i]) == 1:
            chains.append((walk(i), False))
    # 剩下未访问且有边的都是闭环
    for i in range(n):
        if not visited[i] and len(adj[i]) == 2:
            chains.append((walk(i), True))
    return chains


# ---------------------------------------------------------------------------
# 主入口：等高线提取
# ---------------------------------------------------------------------------


def extract_contours(img_rgb: np.ndarray, params: TerrainParams | None = None) -> list[dict]:
    """从 RGB 地质图提取等高线折线（CONTRACTS.md contours.json 格式）。

    返回 [{"contour_id": i, "elevation": None, "polyline_px": [[u,v],...]},...]
    按折线长度降序编号；elevation 留空待赋值（人工或真值模拟）。
    """
    if params is None:
        params = TerrainParams()

    # 1-3. 暗掩膜 → 连通域筛 dash → 质心 + 主方向
    cents, dirs = _dash_segments(img_rgb, params)
    # 4. 邻段连接（距离 + 双段方向兼容 + 每侧最多 1 个 + 互选）
    edges = _link_dashes(cents, dirs, params)
    # 5. 图遍历成链
    chains = _edges_to_chains(len(cents), edges)

    polylines = []
    for order, is_cycle in chains:
        # 段数太少的链是文字笔画/残段，丢弃
        if len(order) < params.min_chain_dashes:
            continue
        pts = [[float(cents[k][0]), float(cents[k][1])] for k in order]
        # 6. 闭合检测：闭环天然闭合；开链首尾足够近也视为闭合（补首点）
        p0, p1 = np.asarray(pts[0]), np.asarray(pts[-1])
        if is_cycle or float(np.hypot(*(p0 - p1))) < params.link_dist * 1.5:
            pts.append(list(pts[0]))
        # Douglas-Peucker 简化，去掉共线冗余点
        pts = lines.simplify_polyline(pts, eps=1.5)
        if len(pts) >= 2:
            polylines.append(pts)

    # 按折线长度降序编号（长的更可能是完整等高线，方便人工优先赋值）
    polylines.sort(key=lines.polyline_length, reverse=True)
    return [
        {"contour_id": i, "elevation": None, "polyline_px": pts}
        for i, pts in enumerate(polylines)
    ]


# ---------------------------------------------------------------------------
# 预览图（人工赋高程的对照界面）
# ---------------------------------------------------------------------------


def contours_preview(img_rgb: np.ndarray, contours: list[dict], out_png: str):
    """原图半透明淡化 + 随机色画等高线 + 中点标 contour_id。

    人工赋高程流程：对照原图上的高程注记，在 contours.json 里
    按预览图上的编号填 elevation。
    """
    canvas = (img_rgb.astype(np.float32) * 0.45 + 255.0 * 0.55).astype(np.uint8)
    rng = np.random.default_rng(0)  # 固定种子 → 每次预览颜色一致
    for c in contours:
        color = tuple(int(x) for x in rng.integers(0, 200, size=3))  # 偏深色可读
        pts = np.asarray(c["polyline_px"], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, color, 2, lineType=cv2.LINE_AA)
        mid = pts[len(pts) // 2]
        cv2.putText(
            canvas,
            str(c["contour_id"]),
            (int(mid[0]) + 3, int(mid[1]) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(str(out_png), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# 赋高程
# ---------------------------------------------------------------------------


def assign_elevations_from_truth(
    contours: list[dict],
    contours_gt: list[dict],
    max_mean_dist: float = 8.0,
) -> dict[int, float]:
    """模拟人工赋值（合成图定量评估用；真实图由人工提供映射）。

    对每条提取等高线沿弧长抽样点，与每条真值等高线（cKDTree）算
    平均最近距离，取距离最小且 < max_mean_dist 的真值高程。
    多条提取链匹配同一真值线是正常的（同一高程被交叉截断成多段）。

    返回 {contour_id: elevation}；匹配不上的链不出现在映射里。
    """
    # 真值线逐条建 KD 树（真值折线较稀，先重采样加密保证距离度量准确）
    gt_trees = []
    for g in contours_gt:
        pts = lines.resample_polyline(g["polyline_px"], step=3.0)
        gt_trees.append((cKDTree(np.asarray(pts)), float(g["elevation"])))

    mapping: dict[int, float] = {}
    for c in contours:
        sample = np.asarray(lines.resample_polyline(c["polyline_px"], step=5.0))
        best_dist, best_elev = np.inf, None
        for tree, elev in gt_trees:
            d, _ = tree.query(sample)
            md = float(np.mean(d))
            if md < best_dist:
                best_dist, best_elev = md, elev
        if best_elev is not None and best_dist < max_mean_dist:
            mapping[c["contour_id"]] = best_elev
    return mapping


def apply_elevations(contours: list[dict], mapping: dict[int, float]):
    """把 {contour_id: elevation} 映射就地写进等高线列表。"""
    for c in contours:
        if c["contour_id"] in mapping:
            c["elevation"] = float(mapping[c["contour_id"]])


# ---------------------------------------------------------------------------
# DEM 重建
# ---------------------------------------------------------------------------


def contours_to_dem(
    contours: list[dict],
    img_hw: tuple,
    params: TerrainParams | None = None,
    fallback_z: float = 500.0,
) -> np.ndarray:
    """已赋高程的等高线 → 逐像素 DEM（float32 HxW，单位米）。

    步骤：
    1. 所有已赋高程折线沿弧长重采样成 (u,v,z) 散点（加密保证插值支撑）；
    2. 在 grid_step 步长的粗网格上 griddata 线性插值——等高线之间
       高程线性过渡，正是等高线的几何含义；
    3. 散点凸包外（图边缘、山顶/谷底闭圈内侧之外）线性插值为 NaN，
       用最近邻补齐（相当于沿最外圈等高线高程外推平台）；
    4. 粗网格上高斯平滑：抹平三角剖分的棱脊、软化最外圈平台边缘；
    5. cv2.resize 双线性放大回全图尺寸（粗网格插值省 grid_step² 倍计算量）。

    退化保护：已赋高程的不同数值 < 2 个时无法插值出起伏，
    返回 fallback_z 常值面并打印中文警告。
    """
    if params is None:
        params = TerrainParams()
    h, w = int(img_hw[0]), int(img_hw[1])

    pts_uv, pts_z = [], []
    for c in contours:
        if c["elevation"] is None:
            continue
        rp = lines.resample_polyline(c["polyline_px"], step=float(params.grid_step))
        pts_uv.extend(rp)
        pts_z.extend([float(c["elevation"])] * len(rp))

    if len(set(pts_z)) < 2:
        print(f"[terrain] 警告：已赋高程的等高线不足两个不同高程值，"
              f"无法插值地形，返回常值面 z={fallback_z}")
        return np.full((h, w), fallback_z, dtype=np.float32)

    pts_uv = np.asarray(pts_uv, dtype=float)
    pts_z = np.asarray(pts_z, dtype=float)

    # 粗网格坐标（像素坐标系下每 grid_step 取一点）
    gu = np.arange(0, w, params.grid_step, dtype=float)
    gv = np.arange(0, h, params.grid_step, dtype=float)
    GU, GV = np.meshgrid(gu, gv)

    # 线性插值（凸包内）+ 最近邻补齐（凸包外）
    grid_lin = griddata(pts_uv, pts_z, (GU, GV), method="linear")
    nan = np.isnan(grid_lin)
    if nan.any():
        grid_nn = griddata(pts_uv, pts_z, (GU[nan], GV[nan]), method="nearest")
        grid_lin[nan] = grid_nn

    # 高斯平滑（σ 定义在粗网格上，等效全图 σ*grid_step 像素）
    grid_sm = gaussian_filter(grid_lin, sigma=params.smooth_sigma)

    # 放大回全尺寸；cv2.resize 的 dsize 是 (宽, 高)
    dem = cv2.resize(grid_sm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return dem
