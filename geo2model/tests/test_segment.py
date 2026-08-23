# -*- coding: utf-8 -*-
"""
test_segment.py — segment.py 的自测脚本（真实合成数据）
========================================================

直接用 python 运行（不依赖 pytest）：
    .venv-gempy/bin/python -m geo2model.tests.test_segment

用 mapgen 生成的合成案例做端到端验证：
    synth_base           → 默认参数分割，与 truth/labels_gt.npy 对比全部指标
    synth_similar_colors → 相近色场景（考验 ΔE 合并阈值不吞掉真实单元）

全部断言通过时打印 ALL TESTS PASSED。
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

# 把项目根目录加入 import 路径，保证脚本可以从任何目录直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from geo2model import metrics  # noqa: E402
from geo2model.segment import SegmentParams, save_extract, segment_map  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASES_DIR = os.path.join(ROOT, "data", "output", "geo2model")


def load_case(case: str):
    """读取一个合成案例的输入图（RGB）与真值标签。"""
    case_dir = os.path.join(CASES_DIR, case)
    bgr = cv2.imread(os.path.join(case_dir, "input", "map.png"), cv2.IMREAD_COLOR)
    assert bgr is not None, f"读不到 {case} 的 map.png"
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gt = np.load(os.path.join(case_dir, "truth", "labels_gt.npy"))
    return case_dir, img, gt


def run_and_eval(case: str):
    """对一个案例跑默认参数分割 + 全指标评估，打印所有指标。"""
    case_dir, img, gt = load_case(case)
    result = segment_map(img, SegmentParams())
    res = metrics.segmentation_metrics(result["labels"], gt)
    print(f"-- {case} --")
    for key in ("pixel_accuracy", "macro_iou", "boundary_precision",
                "boundary_recall", "boundary_f1", "n_units_gt", "n_units_pred"):
        val = res[key]
        print(f"  {key:20s}: {val:.4f}" if isinstance(val, float)
              else f"  {key:20s}: {val}")
    print(f"  per_class_iou       : "
          f"{ {k: round(v, 4) for k, v in res['per_class_iou'].items()} }")
    print(f"  per_class_recall    : "
          f"{ {k: round(v, 4) for k, v in res['per_class_recall'].items()} }")
    print(f"  mapping (pred->gt)  : {res['mapping']}")
    print(f"  units: {[(u['unit_id'], u['role_guess'], u['n_px'])
                       for u in result['units']]}")
    return case_dir, img, result, res


def test_synth_base():
    """clean 基准图：像素准确率 / 宏 IoU / 单元数三条验收线。"""
    case_dir, img, result, res = run_and_eval("synth_base")
    assert res["pixel_accuracy"] >= 0.93, \
        f"pixel_accuracy 不达标: {res['pixel_accuracy']:.4f} < 0.93"
    assert res["macro_iou"] >= 0.85, \
        f"macro_iou 不达标: {res['macro_iou']:.4f} < 0.85"
    # 注意：场景虽然定义了 6 个单元（5 层 + 基底 S1），但该几何下基底
    # 从未出露，labels_gt 实际只有 5 个单元（n_units_gt=5）。因此单元数
    # 验收区间取 [5, 9]：不少于真值单元数，允许少量水体/干扰簇。
    assert res["n_units_pred"] >= res["n_units_gt"], \
        f"识别单元数 {res['n_units_pred']} < 真值单元数 {res['n_units_gt']}"
    assert 5 <= res["n_units_pred"] <= 9, \
        f"识别单元数超出区间: {res['n_units_pred']} not in [5, 9]"
    print("  [ok] synth_base: accuracy/macro_iou/n_units 达标")
    return case_dir, img, result


def test_synth_similar_colors():
    """相近色场景：相邻地层 ΔE 仅 6~8，合并阈值不能吞掉真实单元。"""
    _case_dir, _img, _result, res = run_and_eval("synth_similar_colors")
    assert res["pixel_accuracy"] >= 0.88, \
        f"pixel_accuracy 不达标: {res['pixel_accuracy']:.4f} < 0.88"
    print("  [ok] synth_similar_colors: accuracy 达标")


def test_save_extract(case_dir, img, result):
    """save_extract 跑通且四个产物文件都存在非空。"""
    extract_dir = os.path.join(case_dir, "extract")
    written = save_extract(result, extract_dir, img_rgb=img)
    expect = ["labels.npy", "units_auto.csv",
              "preview_clusters.png", "preview_boundaries.png"]
    for name in expect:
        path = written.get(name)
        assert path and os.path.isfile(path) and os.path.getsize(path) > 0, \
            f"产物缺失或为空: {name}"
    # labels.npy 回读校验：dtype/形状与内存结果一致
    back = np.load(os.path.join(extract_dir, "labels.npy"))
    assert back.dtype == np.int16 and back.shape == result["labels"].shape
    assert np.array_equal(back, result["labels"])
    print(f"  [ok] save_extract -> {extract_dir}")


def test_legend_mode():
    """legend 模式冒烟测试：用真值图例色驱动，标签应映射为图例序号。"""
    _case_dir, img, gt = load_case("synth_base")
    legend = [
        ("N1", (250, 240, 190)),
        ("K2", (205, 225, 145)),
        ("K1", (150, 200, 120)),
        ("P2", (240, 165, 120)),
        ("P1", (215, 125, 95)),
        ("S1", (125, 185, 175)),
    ]
    result = segment_map(img, SegmentParams(mode="legend", legend_colors=legend))
    # 图例序号恰与真值 unit_id 同序 → 无需匈牙利匹配，直接恒等映射评估
    ids = {u["unit_id"] for u in result["units"]}
    assert ids <= set(range(len(legend))), f"legend 单元号越界: {ids}"
    assert all("name" in u for u in result["units"]), "legend 模式 units 应带 name"
    res = metrics.segmentation_metrics(
        result["labels"], gt, mapping={i: i for i in range(len(legend))}
    )
    print(f"  legend 模式 pixel_accuracy: {res['pixel_accuracy']:.4f}")
    assert res["pixel_accuracy"] >= 0.93, \
        f"legend 模式 pixel_accuracy 不达标: {res['pixel_accuracy']:.4f}"
    print("  [ok] legend mode: 标签映射为图例序号且达标")


if __name__ == "__main__":
    print("== segment tests ==")
    case_dir, img, result = test_synth_base()
    test_synth_similar_colors()
    test_save_extract(case_dir, img, result)
    test_legend_mode()
    print("ALL TESTS PASSED")
