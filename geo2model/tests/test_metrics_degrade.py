# -*- coding: utf-8 -*-
"""
test_metrics_degrade.py — metrics.py 与 degrade.py 的自测脚本
=============================================================

直接用 python 运行（不依赖 pytest）：
    .venv-gempy/bin/python geo2model/tests/test_metrics_degrade.py

全部断言通过时打印 ALL TESTS PASSED。
"""

from __future__ import annotations

import os
import sys
import tempfile

import cv2
import numpy as np

# 把项目根目录加入 import 路径，保证脚本可以从任何目录直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from geo2model import degrade, metrics  # noqa: E402


# ----------------------------------------------------------------------
# 测试数据构造
# ----------------------------------------------------------------------

def make_quadrant_gt():
    """200x200 真值标签图：四个象限，标签值故意不连续（10/20/30/40）。"""
    gt = np.zeros((200, 200), dtype=np.int16)
    gt[:100, :100] = 10
    gt[:100, 100:] = 20
    gt[100:, :100] = 30
    gt[100:, 100:] = 40
    return gt


def relabel(gt, perm):
    """按 {旧标签: 新标签} 字典重编号，模拟 pred 用另一套编号。"""
    out = np.zeros_like(gt)
    for old, new in perm.items():
        out[gt == old] = new
    return out


# ----------------------------------------------------------------------
# metrics 测试
# ----------------------------------------------------------------------

def test_match_labels_and_accuracy():
    """打乱标签编号 + 少量噪声像素：映射应完全恢复，准确率符合预期。"""
    gt = make_quadrant_gt()
    perm = {10: 3, 20: 7, 30: 1, 40: 5}  # gt标签 → pred标签
    pred = relabel(gt, perm)

    # 撒 200 个噪声像素：设成 pred 中额外的标签 9（gt 里不存在的类，
    # 顺便测试 "pred 类多于 gt 时允许部分 pred 无匹配"）
    rng = np.random.default_rng(7)
    noise_pos = rng.choice(gt.size, size=200, replace=False)
    pred_noisy = pred.copy()
    pred_noisy.flat[noise_pos] = 9

    expect_mapping = {p: g for g, p in perm.items()}  # pred → gt
    got = metrics.match_labels(pred_noisy, gt)
    assert got == expect_mapping, f"match_labels 映射错误: {got}"
    assert 9 not in got, "多余的 pred 标签 9 不应出现在映射里"

    res = metrics.segmentation_metrics(pred_noisy, gt)
    # 恰好 200/40000 = 0.5% 的像素被破坏
    expect_acc = 1.0 - 200 / gt.size
    assert abs(res["pixel_accuracy"] - expect_acc) < 1e-9, res["pixel_accuracy"]
    assert 0.99 <= res["pixel_accuracy"] < 1.0
    assert res["macro_iou"] > 0.97
    assert res["n_units_gt"] == 4
    assert res["n_units_pred"] == 5  # 四个匹配类 + 噪声类 9
    assert res["mapping"] == expect_mapping
    for v in res["per_class_recall"].values():
        assert 0.98 <= v <= 1.0

    # ignore_mask：把噪声像素全部忽略后准确率应恢复为 1
    ignore = np.zeros(gt.shape, dtype=bool)
    ignore.flat[noise_pos] = True
    res_ig = metrics.segmentation_metrics(pred_noisy, gt, ignore_mask=ignore)
    assert res_ig["pixel_accuracy"] == 1.0

    # confusion_table：交叉计数应与直接统计一致
    table = metrics.confusion_table(pred_noisy, gt)
    assert int(table.to_numpy().sum()) == gt.size
    assert table.loc[10, 3] == int(((gt == 10) & (pred_noisy == 3)).sum())
    assert table.loc[40, 9] == int(((gt == 40) & (pred_noisy == 9)).sum())
    print("  [ok] match_labels + pixel_accuracy + ignore_mask + confusion_table")


def test_perfect_prediction():
    """完美预测（仅标签重编号）：所有指标应等于 1。"""
    gt = make_quadrant_gt()
    pred = relabel(gt, {10: 0, 20: 1, 30: 2, 40: 3})
    res = metrics.segmentation_metrics(pred, gt)
    assert res["pixel_accuracy"] == 1.0
    assert res["macro_iou"] == 1.0
    assert all(v == 1.0 for v in res["per_class_iou"].values())
    assert all(v == 1.0 for v in res["per_class_recall"].values())
    assert res["boundary_precision"] == 1.0
    assert res["boundary_recall"] == 1.0
    assert res["boundary_f1"] == 1.0
    print("  [ok] perfect prediction -> all metrics = 1")


def test_boundary_tolerance():
    """边界平移 2px：tol=3 时 F1 应接近 1，tol=0 时应明显下降。"""
    gt = np.zeros((60, 200), dtype=np.int32)
    gt[:, 100:] = 1          # 真值边界在第 100 列附近
    pred = np.zeros_like(gt)
    pred[:, 102:] = 1        # 预测边界整体右移 2 像素

    res3 = metrics.segmentation_metrics(pred, gt, tol_px=3)
    res0 = metrics.segmentation_metrics(pred, gt, tol_px=0)
    assert res3["boundary_f1"] > 0.99, res3["boundary_f1"]
    assert res0["boundary_f1"] < 0.5, res0["boundary_f1"]
    print("  [ok] boundary F1: shift 2px -> tol=3 ~1.0, tol=0 low")


def test_array_agreement():
    """array_agreement 手工算例。"""
    pred = np.array([1, 1, 2, 3])
    gt = np.array([1, 2, 2, 3])
    res = metrics.array_agreement(pred, gt)
    assert res["agreement"] == 0.75
    assert res["n_total"] == 4
    assert res["per_value"] == {1: 1.0, 2: 0.5, 3: 1.0}

    # 带 mask：屏蔽掉唯一不一致的元素后应完全一致
    mask = np.array([True, False, True, True])
    res_m = metrics.array_agreement(pred, gt, mask=mask)
    assert res_m["agreement"] == 1.0
    assert res_m["n_total"] == 3

    # 三维数组 + 字符串值（模拟体素单元名）
    gt3 = np.array([["K1", "K1"], ["P1", "P1"]]).reshape(2, 2, 1)
    pr3 = np.array([["K1", "P1"], ["P1", "P1"]]).reshape(2, 2, 1)
    res3 = metrics.array_agreement(pr3, gt3)
    assert res3["agreement"] == 0.75
    assert res3["per_value"] == {"K1": 0.5, "P1": 1.0}
    print("  [ok] array_agreement manual cases")


# ----------------------------------------------------------------------
# degrade 测试
# ----------------------------------------------------------------------

def make_test_map():
    """构造带渐变、色块和黑色图线的彩色测试图（模拟简化地质图）。"""
    h, w = 160, 200
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = np.linspace(60, 220, w, dtype=np.uint8)[None, :]
    img[..., 1] = np.linspace(200, 80, h, dtype=np.uint8)[:, None]
    img[..., 2] = 150
    img[30:90, 40:120] = (200, 120, 90)    # 一个色块
    img[100:150, 130:190] = (120, 190, 140)
    # 黑色图线（line_break 需要暗像素）；cv2 坐标是 (x, y)
    cv2.line(img, (10, 10), (190, 150), (10, 10, 10), 2)
    cv2.line(img, (100, 5), (100, 155), (0, 0, 0), 2)
    return img


def test_degrade_all_kinds():
    """逐 kind × severity：形状/dtype 不变、确有差异、同种子可复现。"""
    img = make_test_map()
    img_backup = img.copy()
    for kind in degrade.DEGRADATION_KINDS:
        for severity in (1, 2, 3, 4):
            out1 = degrade.apply_degradation(
                img, kind, severity, np.random.default_rng(42))
            out2 = degrade.apply_degradation(
                img, kind, severity, np.random.default_rng(42))
            assert out1.shape == img.shape, (kind, severity)
            assert out1.dtype == np.uint8, (kind, severity)
            assert np.any(out1 != img), f"{kind} s{severity} 输出与原图完全相同"
            assert np.array_equal(out1, out2), \
                f"{kind} s{severity} 同种子结果不可复现"
    assert np.array_equal(img, img_backup), "apply_degradation 不应修改输入图"
    print("  [ok] all kinds x severities: shape/dtype/diff/reproducible")


def test_apply_profile():
    """多退化叠加：输出合法且与单一退化不同。"""
    img = make_test_map()
    profile = [("fade", 2), ("noise", 2), ("jpeg", 3)]
    out1 = degrade.apply_profile(img, profile, np.random.default_rng(3))
    out2 = degrade.apply_profile(img, profile, np.random.default_rng(3))
    assert out1.shape == img.shape and out1.dtype == np.uint8
    assert np.any(out1 != img)
    assert np.array_equal(out1, out2)
    print("  [ok] apply_profile stacking + reproducible")


def test_grid_preview():
    """预览拼图能生成非空 PNG（写到临时目录，用完即删）。"""
    img = make_test_map()
    tmp_dir = tempfile.mkdtemp(prefix="degrade_preview_")
    out_png = os.path.join(tmp_dir, "grid.png")
    try:
        degrade.degradation_grid_preview(img, out_png)
        assert os.path.isfile(out_png) and os.path.getsize(out_png) > 0
    finally:
        if os.path.isfile(out_png):
            os.remove(out_png)
        os.rmdir(tmp_dir)
    print("  [ok] degradation_grid_preview writes a non-empty png")


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("== metrics tests ==")
    test_match_labels_and_accuracy()
    test_perfect_prediction()
    test_boundary_tolerance()
    test_array_agreement()
    print("== degrade tests ==")
    test_degrade_all_kinds()
    test_apply_profile()
    test_grid_preview()
    print("ALL TESTS PASSED")
