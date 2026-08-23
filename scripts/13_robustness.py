#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_robustness.py — 提取环节鲁棒性扫描（退化 × 强度）
======================================================

对 synth_base 地质图施加 7 种扫描退化 × 4 档强度（degrade.py），
逐一测试信息提取三模块的表现：
- 分割：像素准确率 / 宏 IoU / 边界 F1（申报书指标的鲁棒性曲线）
- 等高线→DEM：赋值成功链数 / DEM MAE
- 断层：候选是否命中真值迹线

输出：
- data/output/geo2model/robustness/robustness.csv
- data/output/geo2model/robustness/curves_*.png（准确率-强度曲线）
- data/output/geo2model/robustness/degradation_preview.png（退化拼图）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from geo2model.degrade import DEGRADATION_KINDS, apply_degradation  # noqa: E402
from geo2model.degrade import degradation_grid_preview  # noqa: E402
from geo2model.metrics import segmentation_metrics  # noqa: E402
from geo2model.segment import SegmentParams, segment_map  # noqa: E402
from geo2model.terrain import (  # noqa: E402
    TerrainParams, extract_contours, assign_elevations_from_truth,
    apply_elevations, contours_to_dem,
)
from geo2model.vectorize import VectorizeParams, extract_fault_candidates  # noqa: E402
from geo2model.lines import polyline_length  # noqa: E402

CASE = PROJECT_ROOT / "data" / "output" / "geo2model" / "synth_base"
OUT = PROJECT_ROOT / "data" / "output" / "geo2model" / "robustness"


def fault_hit(faults: list[dict], fault_trace_gt: np.ndarray) -> bool:
    """判断是否有断层候选命中真值迹线（平均距离 < 15px 且长度足够）。"""
    if len(fault_trace_gt) == 0:
        return False
    from scipy.spatial import cKDTree

    tree = cKDTree(fault_trace_gt)
    for fl in faults:
        poly = np.asarray(fl["polyline_px"], dtype=float)
        if polyline_length(poly) < 300:
            continue
        d, _ = tree.query(poly)
        if float(np.mean(d)) < 15.0:
            return True
    return False


def eval_one(img: np.ndarray, labels_gt, contours_gt, dem_gt, trace_gt) -> dict:
    """对一张（退化后）地质图跑提取三模块并打分。"""
    out: dict = {}
    t0 = time.time()
    seg = segment_map(img, SegmentParams())
    m = segmentation_metrics(seg["labels"], labels_gt, tol_px=3)
    out.update(
        pixel_accuracy=m["pixel_accuracy"],
        macro_iou=m["macro_iou"],
        boundary_f1=m["boundary_f1"],
        n_units_pred=m["n_units_pred"],
    )
    # 等高线 → DEM
    contours = extract_contours(img, TerrainParams())
    mapping = assign_elevations_from_truth(contours, contours_gt)
    apply_elevations(contours, mapping)
    dem = contours_to_dem(contours, img.shape[:2], TerrainParams(), fallback_z=700.0)
    out["n_contours_assigned"] = len(mapping)
    out["dem_mae_m"] = float(np.mean(np.abs(dem - dem_gt)))
    # 断层
    faults = extract_fault_candidates(img, seg["labels"], VectorizeParams())
    out["fault_detected"] = bool(fault_hit(faults, trace_gt))
    out["eval_s"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img = cv2.cvtColor(cv2.imread(str(CASE / "input" / "map.png")), cv2.COLOR_BGR2RGB)
    labels_gt = np.load(CASE / "truth" / "labels_gt.npy")
    dem_gt = np.load(CASE / "truth" / "dem_gt.npy")
    with open(CASE / "truth" / "contours_gt.json", encoding="utf-8") as f:
        contours_gt = json.load(f)
    with open(CASE / "truth" / "truth_meta.json", encoding="utf-8") as f:
        trace_gt = np.asarray(json.load(f)["fault_trace_px"], dtype=float)

    degradation_grid_preview(img, OUT / "degradation_preview.png")

    rows = []
    # 基线（无退化）
    base = eval_one(img, labels_gt, contours_gt, dem_gt, trace_gt)
    rows.append({"kind": "none", "severity": 0, **base})
    print(f"[基线] acc={base['pixel_accuracy']:.4f} f1={base['boundary_f1']:.4f} "
          f"dem={base['dem_mae_m']:.1f} fault={base['fault_detected']}")

    for ki, kind in enumerate(DEGRADATION_KINDS):
        for sev in (1, 2, 3, 4):
            rng = np.random.default_rng(1000 + ki * 10 + sev)
            dimg = apply_degradation(img, kind, sev, rng)
            r = eval_one(dimg, labels_gt, contours_gt, dem_gt, trace_gt)
            rows.append({"kind": kind, "severity": sev, **r})
            print(f"[{kind} s{sev}] acc={r['pixel_accuracy']:.4f} "
                  f"iou={r['macro_iou']:.3f} f1={r['boundary_f1']:.3f} "
                  f"dem={r['dem_mae_m']:.1f} fault={r['fault_detected']} "
                  f"({r['eval_s']}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "robustness.csv", index=False)

    # 曲线图：每种退化一条曲线
    for metric, ylabel, fname in (
        ("pixel_accuracy", "Pixel accuracy", "curves_accuracy.png"),
        ("boundary_f1", "Boundary F1 (tol=3px)", "curves_boundary_f1.png"),
        ("dem_mae_m", "DEM MAE (m)", "curves_dem_mae.png"),
    ):
        plt.figure(figsize=(8, 5))
        base_v = df[df["kind"] == "none"][metric].iloc[0]
        for kind in DEGRADATION_KINDS:
            sub = df[df["kind"] == kind].sort_values("severity")
            plt.plot([0] + sub["severity"].tolist(),
                     [base_v] + sub[metric].tolist(),
                     marker="o", label=kind)
        if metric == "pixel_accuracy":
            plt.axhline(0.9, color="red", ls="--", lw=1, label="90% target")
        plt.xlabel("Degradation severity (0 = clean)")
        plt.ylabel(ylabel)
        plt.title(f"Extraction robustness: {ylabel} vs degradation")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / fname, dpi=150)
        plt.close()
        print(f"已保存: {OUT / fname}")


if __name__ == "__main__":
    main()
