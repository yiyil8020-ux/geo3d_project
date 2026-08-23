# -*- coding: utf-8 -*-
"""
vectorize.py — 分区标签图 → 地质界线折线 + 断层候选
=====================================================

输入：segment 模块产出的全像素标签图 labels + 原图
输出（见 CONTRACTS.md）：
- contacts.json ：每条「两个单元之间的共享边界」折线（地质接触线）
- faults.json   ：断层候选折线（待人工确认，confirmed=false）

核心思想：
1. 接触线：标签图上 4-邻域两侧标签不同的位置就是单元边界；
   按「单元对 (a,b)」分别取边界带 → 骨架化 → 追踪成有序折线。
   这样每条折线天然带着"它分隔哪两个单元"的属性，
   后续可直接按层序新老关系换算成 GemPy 的层位控制点。
2. 断层候选：制图规范中断层线比普通地质界线粗。
   对暗色线划做距离变换，半径明显大于普通界线的"粗线核心"
   即断层候选；再加上长度、跨单元数等属性供人工确认。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .lines import mask_to_polylines, polyline_length, simplify_polyline


@dataclass
class VectorizeParams:
    """矢量化参数。"""

    min_contact_len: float = 22.0   # 接触线最短长度（像素），滤碎屑
    simplify_eps: float = 1.6       # 折线简化容差（像素）
    dark_gray_max: int = 110        # 暗线划灰度阈值（与 segment 一致）
    fault_min_radius: float = 2.4   # 距离变换半径阈值：粗于它=断层候选核心
    fault_min_len: float = 90.0     # 断层候选最短长度（像素）
    fault_default_dip: float = 70.0     # 人工未填时断层默认倾角
    fault_default_azimuth: float = 90.0
    merge_gap: float = 28.0         # 断层碎段拼接：端点间隙上限
    merge_angle_deg: float = 32.0   # 拼接方向偏差上限


# ---------------------------------------------------------------------------
# 接触线提取
# ---------------------------------------------------------------------------


def extract_contacts(labels: np.ndarray, params: VectorizeParams) -> list[dict]:
    """按单元对提取共享边界折线。

    实现：对水平/垂直相邻像素比较标签，得到所有「相邻单元对」；
    对每一对生成两侧像素并集的细带 mask（约 2px 宽，保证骨架连续），
    骨架化后追踪成折线。
    """
    lab = labels.astype(np.int32)
    h, w = lab.shape

    # 找出全部相邻单元对（排序后去重）
    ph = np.stack([lab[:, :-1].ravel(), lab[:, 1:].ravel()], axis=1)
    pv = np.stack([lab[:-1, :].ravel(), lab[1:, :].ravel()], axis=1)
    pairs = np.concatenate([ph, pv], axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pairs.sort(axis=1)
    uniq_pairs, counts = np.unique(pairs, axis=0, return_counts=True)

    contacts: list[dict] = []
    cid = 0
    for (a, b), n_edge in zip(uniq_pairs, counts):
        if n_edge < params.min_contact_len:  # 边界太短的单元对直接跳过
            continue
        ma = lab == a
        mb = lab == b
        # 两侧像素并集：a 的紧邻 b 处 + b 的紧邻 a 处
        band = np.zeros((h, w), dtype=np.uint8)
        band[:, :-1] |= (ma[:, :-1] & mb[:, 1:]).astype(np.uint8)
        band[:, 1:] |= (mb[:, :-1] & ma[:, 1:]).astype(np.uint8)
        band[:, :-1] |= (mb[:, :-1] & ma[:, 1:]).astype(np.uint8)
        band[:, 1:] |= (ma[:, :-1] & mb[:, 1:]).astype(np.uint8)
        band[:-1, :] |= (ma[:-1, :] & mb[1:, :]).astype(np.uint8)
        band[1:, :] |= (mb[:-1, :] & ma[1:, :]).astype(np.uint8)
        band[:-1, :] |= (mb[:-1, :] & ma[1:, :]).astype(np.uint8)
        band[1:, :] |= (ma[:-1, :] & mb[1:, :]).astype(np.uint8)

        for poly in mask_to_polylines(
            band, min_len=params.min_contact_len, eps=params.simplify_eps
        ):
            contacts.append(
                {
                    "contact_id": cid,
                    "unit_a": int(a),
                    "unit_b": int(b),
                    "polyline_px": poly,
                }
            )
            cid += 1
    return contacts


# ---------------------------------------------------------------------------
# 断层候选提取
# ---------------------------------------------------------------------------


def _merge_broken_polylines(polys: list, gap: float, angle_deg: float) -> list:
    """把因交叉断开的近共线碎段拼回长线。

    规则：两条线的端点距离 < gap，且两条线在端点处的走向夹角 < angle_deg
    （考虑 180° 对称），则拼接。贪心迭代直到无可拼接。
    """

    def _end_dir(poly, at_start: bool):
        p = np.asarray(poly, dtype=float)
        if at_start:
            d = p[0] - p[min(3, len(p) - 1)]
        else:
            d = p[-1] - p[-min(4, len(p))]
        n = np.hypot(*d) + 1e-9
        return d / n

    def _angle(u, v):
        cosv = abs(float(np.dot(u, v)))  # 180° 对称
        return np.degrees(np.arccos(np.clip(cosv, -1, 1)))

    polys = [list(map(list, p)) for p in polys]
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(polys)
        for i in range(len(polys)):
            if used[i]:
                continue
            pi = polys[i]
            best = None
            for j in range(len(polys)):
                if j == i or used[j]:
                    continue
                pj = polys[j]
                # 四种端点组合
                for ei, ej, rev_i, rev_j in (
                    (pi[-1], pj[0], False, False),
                    (pi[-1], pj[-1], False, True),
                    (pi[0], pj[0], True, False),
                    (pi[0], pj[-1], True, True),
                ):
                    d = float(np.hypot(ei[0] - ej[0], ei[1] - ej[1]))
                    if d > gap:
                        continue
                    di = _end_dir(pi[::-1] if rev_i else pi, at_start=False)
                    dj = _end_dir(pj[::-1] if rev_j else pj, at_start=True)
                    if _angle(di, dj) > angle_deg:
                        continue
                    if best is None or d < best[0]:
                        best = (d, j, rev_i, rev_j)
            if best is not None:
                _, j, rev_i, rev_j = best
                pj = polys[j]
                merged = (pi[::-1] if rev_i else pi) + (pj if not rev_j else pj[::-1])
                used[i] = used[j] = True
                out.append(merged)
                changed = True
            else:
                used[i] = True
                out.append(pi)
        polys = out
    return polys


def extract_fault_candidates(
    img_rgb: np.ndarray,
    labels: np.ndarray,
    params: VectorizeParams,
) -> list[dict]:
    """从暗线划中找「明显粗于普通界线」的长线 = 断层候选。"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < params.dark_gray_max).astype(np.uint8)

    # 距离变换：每个暗像素到最近非暗像素的距离 ≈ 线的半宽
    dist = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    core = (dist > params.fault_min_radius).astype(np.uint8)
    if core.sum() < 30:
        return []
    # 把核心还原到接近原线宽后骨架化追踪（膨胀半径≈阈值）
    r = int(round(params.fault_min_radius))
    core = cv2.dilate(
        core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    )
    polys = mask_to_polylines(core, min_len=params.fault_min_len * 0.4, eps=2.0)
    polys = _merge_broken_polylines(polys, params.merge_gap, params.merge_angle_deg)
    polys = [p for p in polys if polyline_length(p) >= params.fault_min_len]

    faults = []
    for k, poly in enumerate(sorted(polys, key=polyline_length, reverse=True)):
        # 属性：跨过多少个不同单元（沿线采样标签）
        pts = np.asarray(poly, dtype=int)
        pts[:, 0] = np.clip(pts[:, 0], 0, labels.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, labels.shape[0] - 1)
        crossed = len(np.unique(labels[pts[:, 1], pts[:, 0]]))
        p0, p1 = np.asarray(poly[0]), np.asarray(poly[-1])
        chord = float(np.hypot(*(p1 - p0)))
        length = polyline_length(poly)
        faults.append(
            {
                "fault_id": k,
                "name": f"F{k + 1}",
                "polyline_px": simplify_polyline(poly, 2.0),
                "dip": params.fault_default_dip,
                "azimuth": params.fault_default_azimuth,
                "length_px": round(length, 1),
                "straightness": round(chord / max(length, 1e-9), 3),
                "n_units_crossed": int(crossed),
                "confirmed": False,
            }
        )
    return faults


# ---------------------------------------------------------------------------
# 汇总入口 + 预览
# ---------------------------------------------------------------------------


def vectorize_map(
    img_rgb: np.ndarray,
    labels: np.ndarray,
    extract_dir: str | Path,
    params: VectorizeParams | None = None,
) -> dict:
    """接触线 + 断层候选一次提取并落盘，输出预览图。"""
    params = params or VectorizeParams()
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    contacts = extract_contacts(labels, params)
    faults = extract_fault_candidates(img_rgb, labels, params)

    with open(extract_dir / "contacts.json", "w", encoding="utf-8") as f:
        json.dump(contacts, f, ensure_ascii=False)
    with open(extract_dir / "faults.json", "w", encoding="utf-8") as f:
        json.dump(faults, f, ensure_ascii=False, indent=1)

    # 预览：接触线红色、断层蓝色粗线
    prev = img_rgb.copy()
    for c in contacts:
        pts = np.asarray(c["polyline_px"], dtype=np.int32)
        cv2.polylines(prev, [pts], False, (230, 30, 30), 2)
    for fl in faults:
        pts = np.asarray(fl["polyline_px"], dtype=np.int32)
        cv2.polylines(prev, [pts], False, (30, 60, 230), 4)
        if len(pts) > 0:
            cv2.putText(
                prev, fl["name"], tuple(pts[len(pts) // 2]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 60, 230), 2, cv2.LINE_AA,
            )
    cv2.imwrite(
        str(extract_dir / "preview_vectors.png"),
        cv2.cvtColor(prev, cv2.COLOR_RGB2BGR),
    )

    print(f"[vectorize] 接触线 {len(contacts)} 条, 断层候选 {len(faults)} 条")
    return {"contacts": contacts, "faults": faults}
