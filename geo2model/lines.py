# -*- coding: utf-8 -*-
"""
lines.py — 二值线划 mask → 有序折线（骨架追踪矢量化）
======================================================

提供矢量化模块（vectorize / terrain）共用的折线工具：

1. mask_to_polylines：二值 mask → 骨架化 → 沿骨架像素追踪出有序折线
   （在交叉点断开），再用 Douglas-Peucker 简化。
2. 折线几何小工具：长度、等距重采样、简化。

骨架追踪原理：
    skimage.morphology.skeletonize 把线划细化成 1 像素宽；
    每个骨架像素统计 8-邻域内的骨架像素数 = 度 degree：
        度 1 → 线端点；度 2 → 线中间；度 ≥3 → 交叉点。
    从端点出发沿未访问像素一路走到端点/交叉点，得到一条有序折线；
    交叉点处断开，保证每条折线是简单路径（无分叉）。
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

# 8-邻域偏移（行, 列）
_NBR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def polyline_length(pts) -> float:
    """折线总长度（像素单位）。pts: [[u,v],...]"""
    p = np.asarray(pts, dtype=float)
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.hypot(*np.diff(p, axis=0).T)))


def simplify_polyline(pts, eps: float = 1.5):
    """Douglas-Peucker 折线简化（cv2.approxPolyDP，闭合=False）。"""
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    if len(p) < 3:
        return [list(map(float, q)) for q in np.asarray(pts, dtype=float)]
    ap = cv2.approxPolyDP(p, eps, False).reshape(-1, 2)
    return [[float(a), float(b)] for a, b in ap]


def resample_polyline(pts, step: float):
    """沿折线按弧长每 step 像素取一点（含首尾）。"""
    p = np.asarray(pts, dtype=float)
    if len(p) < 2:
        return [list(map(float, q)) for q in p]
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-9:
        return [list(map(float, p[0]))]
    targets = np.arange(0.0, total, step)
    targets = np.append(targets, total)  # 保证含终点
    out = []
    j = 0
    for t in targets:
        while j < len(seg) and s[j + 1] < t:
            j += 1
        if j >= len(seg):
            out.append(list(map(float, p[-1])))
            continue
        r = (t - s[j]) / max(seg[j], 1e-12)
        q = p[j] + (p[j + 1] - p[j]) * r
        out.append([float(q[0]), float(q[1])])
    return out


def mask_to_polylines(
    mask: np.ndarray,
    min_len: float = 15.0,
    eps: float = 1.5,
) -> list[list[list[float]]]:
    """二值 mask → 有序折线列表（像素坐标 [[u,v],...]）。

    参数
    ----
    mask    : uint8/bool HxW，非零 = 线划像素
    min_len : 丢弃总长小于该值的碎线（像素）
    eps     : Douglas-Peucker 简化容差（像素）；<=0 表示不简化
    """
    skel = skeletonize(mask.astype(bool))
    h, w = skel.shape
    # 每个骨架像素的 8-邻域骨架数（卷积一次算完，比逐像素快得多）
    # 注意：cv2 5.0 的 filter2D 对 uint8 卷积核结果异常，必须用 float32 核
    nbr_cnt = cv2.filter2D(
        skel.astype(np.uint8), -1, np.ones((3, 3), np.float32),
        borderType=cv2.BORDER_CONSTANT,
    ) - skel.astype(np.uint8)
    degree = np.where(skel, nbr_cnt, 0)

    visited = np.zeros_like(skel, dtype=bool)
    polylines: list = []

    def neighbors(r, c):
        for dr, dc in _NBR:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and skel[rr, cc]:
                yield rr, cc

    def walk(r, c):
        """从 (r,c) 出发追踪一条路径，返回 [(r,c),...]；在端点/交叉点停。"""
        path = [(r, c)]
        visited[r, c] = True
        cur = (r, c)
        prev = None
        while True:
            nxt = None
            for rr, cc in neighbors(*cur):
                if (rr, cc) == prev:
                    continue
                if degree[rr, cc] >= 3:
                    # 走到交叉点：把交叉点收进来后停（交叉点可被多条线共享）
                    path.append((rr, cc))
                    return path
                if not visited[rr, cc]:
                    nxt = (rr, cc)
                    break
            if nxt is None:
                return path
            visited[nxt] = True
            prev = cur
            cur = nxt
            path.append(cur)

    # 第一轮：从所有端点（度1）与交叉点旁出发
    starts = list(zip(*np.where((degree == 1) & ~visited)))
    for r, c in starts:
        if visited[r, c]:
            continue
        path = walk(r, c)
        if len(path) >= 2:
            polylines.append(path)

    # 交叉点的每条分支单独追踪
    junctions = list(zip(*np.where(degree >= 3)))
    for r, c in junctions:
        for rr, cc in neighbors(r, c):
            if not visited[rr, cc] and degree[rr, cc] < 3:
                path = walk(rr, cc)
                path.insert(0, (r, c))
                if len(path) >= 2:
                    polylines.append(path)

    # 第二轮：剩下未访问的是闭环（如闭合等高线/闭合界线），任取一点绕圈
    rest = list(zip(*np.where(skel & ~visited & (degree == 2))))
    for r, c in rest:
        if visited[r, c]:
            continue
        path = walk(r, c)
        # 闭环补上起点使首尾相接
        if len(path) >= 3:
            path.append(path[0])
            polylines.append(path)

    # (r,c) → (u,v)=(列,行)，过滤碎线 + 简化
    out = []
    for path in polylines:
        pts = [[float(c), float(r)] for r, c in path]
        if polyline_length(pts) < min_len:
            continue
        out.append(simplify_polyline(pts, eps) if eps > 0 else pts)
    return out
