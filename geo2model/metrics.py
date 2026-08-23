# -*- coding: utf-8 -*-
"""
metrics.py — 定量评估指标（分区 / 边界 / 三维体素一致率）
==========================================================

用途：把矢量化 / 建模结果与合成真值（truth/）对比，输出 eval/metrics.json
所需的各项指标（见 CONTRACTS.md 的 metrics.json 格式）。

核心问题：预测的标签编号（extract/labels.npy 里的 0..k-1）和真值的标签编号
（truth/labels_gt.npy 里的 0..n-1）是两套独立编号，同一个地层单元在两边的
编号通常不同。所以评估前必须先"对齐标签"：

1. 构建混淆矩阵：统计每对 (gt 标签, pred 标签) 的重叠像素数。
2. 匈牙利算法（scipy.optimize.linear_sum_assignment）：在混淆矩阵上求
   总重叠像素数最大的一一匹配。这等价于"给每个预测分区找它最像的真值分区，
   且不允许两个预测分区抢同一个真值分区"。
3. 按匹配结果把 pred 重映射到 gt 的标签空间，之后再算像素准确率 / IoU 等。

边界指标采用带容差的 precision / recall / F1（图像分割文献里常用的
Boundary F1，又叫 BF score）：预测边界点只要落在真值边界 tol_px 像素范围内
就算命中。距离查询用 scipy.ndimage.distance_transform_edt（欧氏距离变换）
一次算出"每个像素到最近边界像素的距离"，避免逐点搜索。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment


# ----------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------

def _valid_mask(shape, ignore_mask):
    """把 ignore_mask 转成"参与统计"的布尔掩码（True=统计该像素）。"""
    if ignore_mask is None:
        return np.ones(shape, dtype=bool)
    ignore_mask = np.asarray(ignore_mask, dtype=bool)
    if ignore_mask.shape != tuple(shape):
        raise ValueError("ignore_mask 形状必须与标签图一致")
    return ~ignore_mask


def _confusion_counts(pred, gt, ignore_mask=None):
    """构建混淆矩阵（重叠像素计数）。

    返回
    ----
    gt_labels   : 真值中出现过的标签值（升序）
    pred_labels : 预测中出现过的标签值（升序）
    counts      : int64 矩阵，counts[i, j] = gt==gt_labels[i] 且
                  pred==pred_labels[j] 的像素个数

    实现技巧：np.unique(return_inverse=True) 把任意标签值压缩成
    0..n-1 的下标，再用 np.bincount 一次性统计所有组合的出现次数，
    比双重 for 循环快得多。
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError("pred 与 gt 形状必须一致")
    valid = _valid_mask(gt.shape, ignore_mask)

    g_flat = gt[valid].ravel()
    p_flat = pred[valid].ravel()

    gt_labels, g_idx = np.unique(g_flat, return_inverse=True)
    pred_labels, p_idx = np.unique(p_flat, return_inverse=True)

    n_g, n_p = len(gt_labels), len(pred_labels)
    # 把 (gt下标, pred下标) 编码成一个整数：gt下标 * n_p + pred下标，
    # 然后 bincount 统计每种编码出现的次数，reshape 回矩阵形状。
    combo = g_idx.astype(np.int64) * n_p + p_idx
    counts = np.bincount(combo, minlength=n_g * n_p).reshape(n_g, n_p)
    return gt_labels, pred_labels, counts


def _label_boundary(label_img):
    """求标签图的边界像素（4-邻域中存在不同标签的像素记为边界）。

    做法：把图分别沿上下、左右方向错位一格比较。若某像素与它的
    下邻/右邻标签不同，则这两个像素都是边界像素。
    """
    lab = np.asarray(label_img)
    boundary = np.zeros(lab.shape, dtype=bool)
    # 垂直方向：比较每个像素与其正下方像素
    diff_v = lab[1:, :] != lab[:-1, :]
    boundary[1:, :] |= diff_v
    boundary[:-1, :] |= diff_v
    # 水平方向：比较每个像素与其右侧像素
    diff_h = lab[:, 1:] != lab[:, :-1]
    boundary[:, 1:] |= diff_h
    boundary[:, :-1] |= diff_h
    return boundary


def _boundary_prf(pred_boundary, gt_boundary, tol_px):
    """带容差的边界 precision / recall / F1。

    precision = 预测边界点中，距最近真值边界 ≤ tol_px 的比例
    recall    = 真值边界点中，距最近预测边界 ≤ tol_px 的比例

    distance_transform_edt(~mask) 给出每个像素到 mask 中最近 True
    像素的欧氏距离，一次变换即可查询所有点。
    """
    n_pred = int(pred_boundary.sum())
    n_gt = int(gt_boundary.sum())
    if n_pred == 0 and n_gt == 0:
        # 两边都没有边界（例如全图同一标签）：约定为完全一致
        return 1.0, 1.0, 1.0
    if n_pred == 0 or n_gt == 0:
        # 一边有边界另一边没有：完全不匹配
        return 0.0, 0.0, 0.0

    dist_to_gt = distance_transform_edt(~gt_boundary)
    dist_to_pred = distance_transform_edt(~pred_boundary)
    precision = float((dist_to_gt[pred_boundary] <= tol_px).mean())
    recall = float((dist_to_pred[gt_boundary] <= tol_px).mean())
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


# ----------------------------------------------------------------------
# 公开接口
# ----------------------------------------------------------------------

def match_labels(pred, gt, ignore_mask=None) -> dict[int, int]:
    """用匈牙利算法求 pred 标签 → gt 标签 的最优一一匹配。

    参数
    ----
    pred, gt    : HxW 整数标签图（标签值不必连续）
    ignore_mask : 可选 HxW 布尔图，True 处像素不参与统计

    返回
    ----
    {pred标签: gt标签} 字典。若 pred 的类别数多于 gt，多出来的
    pred 标签没有匹配对象，不会出现在字典里；重叠数为 0 的"凑数
    匹配"也会被剔除。

    原理：linear_sum_assignment 求解"指派问题"——给定代价矩阵，
    选出一组不同行不同列的元素使总代价最小。我们把代价取为
    "负的重叠像素数"，最小化负数即最大化总重叠。
    """
    gt_labels, pred_labels, counts = _confusion_counts(pred, gt, ignore_mask)
    # 行=gt，列=pred；取负号把"最大化重叠"转成"最小化代价"
    row_idx, col_idx = linear_sum_assignment(-counts)
    mapping: dict[int, int] = {}
    for gi, pj in zip(row_idx, col_idx):
        if counts[gi, pj] > 0:  # 零重叠的匹配没有意义，剔除
            mapping[int(pred_labels[pj])] = int(gt_labels[gi])
    return mapping


def segmentation_metrics(pred, gt, ignore_mask=None, tol_px=3,
                         mapping=None) -> dict:
    """分区结果的全套定量指标。

    参数
    ----
    pred, gt    : HxW 整数标签图
    ignore_mask : True 处像素不参与统计
    tol_px      : 边界 F1 的距离容差（像素）
    mapping     : 可选 {pred标签: gt标签}；为 None 时自动调用 match_labels

    返回 dict，键包括：
    pixel_accuracy, per_class_iou, macro_iou, per_class_recall,
    boundary_precision, boundary_recall, boundary_f1,
    n_units_gt, n_units_pred, mapping
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError("pred 与 gt 形状必须一致")
    valid = _valid_mask(gt.shape, ignore_mask)

    if mapping is None:
        mapping = match_labels(pred, gt, ignore_mask)

    # ---- 把 pred 重映射到 gt 标签空间 ----
    # 没有匹配的 pred 标签记为 -999：一个 gt 里不存在的哨兵值，
    # 保证这些像素在后面的比较中永远不命中。
    remapped = np.full(pred.shape, -999, dtype=np.int64)
    for p_lab, g_lab in mapping.items():
        remapped[pred == p_lab] = g_lab

    gt64 = gt.astype(np.int64)
    correct = (remapped == gt64) & valid
    n_valid = int(valid.sum())
    pixel_accuracy = float(correct.sum() / n_valid) if n_valid else 0.0

    # ---- 逐类 IoU 与 recall（只统计 gt 中实际出现的标签） ----
    gt_labels = np.unique(gt64[valid])
    per_class_iou: dict[int, float] = {}
    per_class_recall: dict[int, float] = {}
    for g_lab in gt_labels:
        gt_hit = (gt64 == g_lab) & valid
        pred_hit = (remapped == g_lab) & valid
        inter = int((gt_hit & pred_hit).sum())
        union = int((gt_hit | pred_hit).sum())
        n_gt_px = int(gt_hit.sum())
        per_class_iou[int(g_lab)] = float(inter / union) if union else 0.0
        per_class_recall[int(g_lab)] = float(inter / n_gt_px) if n_gt_px else 0.0
    macro_iou = float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0

    # ---- 边界指标 ----
    pred_boundary = _label_boundary(remapped) & valid
    gt_boundary = _label_boundary(gt64) & valid
    b_prec, b_rec, b_f1 = _boundary_prf(pred_boundary, gt_boundary, tol_px)

    n_units_pred = int(len(np.unique(pred[valid])))

    return {
        "pixel_accuracy": pixel_accuracy,
        "per_class_iou": per_class_iou,
        "macro_iou": macro_iou,
        "per_class_recall": per_class_recall,
        "boundary_precision": b_prec,
        "boundary_recall": b_rec,
        "boundary_f1": b_f1,
        "n_units_gt": int(len(gt_labels)),
        "n_units_pred": n_units_pred,
        "mapping": {int(k): int(v) for k, v in mapping.items()},
    }


def array_agreement(pred, gt, mask=None) -> dict:
    """两个同形状数组的一致比例（用于三维体素解对比等）。

    参数
    ----
    pred, gt : 同形状数组（任意维；元素可为整数 id 或字符串单元名）
    mask     : 可选布尔数组，True 处才参与统计

    返回
    ----
    {"agreement": 总体一致比例,
     "per_value": {gt值: 该值处的一致比例},
     "n_total": 参与统计的元素个数}
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError("pred 与 gt 形状必须一致")
    if mask is None:
        mask = np.ones(gt.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != gt.shape:
            raise ValueError("mask 形状必须与数组一致")

    p_flat = pred[mask].ravel()
    g_flat = gt[mask].ravel()
    n_total = int(g_flat.size)
    same = p_flat == g_flat
    agreement = float(same.mean()) if n_total else 0.0

    per_value: dict = {}
    for val in np.unique(g_flat):
        sel = g_flat == val
        # numpy 标量转成普通 Python 值（int/str），方便写入 json
        key = val.item() if isinstance(val, np.generic) else val
        per_value[key] = float(same[sel].mean())

    return {"agreement": agreement, "per_value": per_value, "n_total": n_total}


def confusion_table(pred, gt, ignore_mask=None) -> pd.DataFrame:
    """混淆矩阵表：行=gt 标签，列=pred 标签，值=重叠像素数（调试用）。"""
    gt_labels, pred_labels, counts = _confusion_counts(pred, gt, ignore_mask)
    table = pd.DataFrame(
        counts,
        index=[int(v) for v in gt_labels],
        columns=[int(v) for v in pred_labels],
    )
    table.index.name = "gt"
    table.columns.name = "pred"
    return table
