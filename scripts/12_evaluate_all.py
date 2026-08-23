#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_evaluate_all.py — 六个合成场景批量端到端评估
=================================================

对每个内置场景跑完整流水线（含剖面深部约束），汇总指标：
- 分区识别像素准确率 / 宏 IoU / 边界 F1（申报书硬指标 ≥90%）
- DEM MAE
- 三维体素一致率

输出：
- data/output/geo2model/summary_eval.json
- data/output/geo2model/summary_eval.md（报告可直接引用的表格）
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from geo2model.mapgen import builtin_scenarios  # noqa: E402
from geo2model.pipeline import run_case  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "data" / "output" / "geo2model"


def scenario_config(name: str) -> dict:
    """按场景生成流水线配置（flat_dem 用平坦地形模式验证退化兜底）。"""
    cfg = {
        "case_dir": str(OUT_ROOT / f"synth_{name}"),
        "georef": {"world": [0, 2000, 0, 2000]},
        "z_range": [0, 1000],
        "resolution": [50, 50, 50],
        "segment": {},
        "terrain": {"mode": "contours", "elevations": "from_truth"},
        "review": "from_truth",
        "constraints": {"use_section_picks": True},
        "model": {"also_html": True, "also_mesh_exports": False},
        "apps": "auto",
        "evaluate": True,
    }
    if name == "flat_dem":
        cfg["terrain"] = {"mode": "flat", "flat_z": 700.0}
    return cfg


def main() -> None:
    results = {}
    for name in builtin_scenarios().keys():
        print("=" * 70)
        print(f"场景: {name}")
        print("=" * 70)
        try:
            report = run_case(scenario_config(name))
            ev = report.get("eval", {})
            seg = ev.get("segmentation", {})
            results[name] = {
                "pixel_accuracy": seg.get("pixel_accuracy"),
                "macro_iou": seg.get("macro_iou"),
                "boundary_f1": seg.get("boundary_f1"),
                "dem_mae_m": ev.get("dem_mae_m"),
                "voxel_3d": ev.get("voxel_3d", {}).get("agreement"),
                "n_contacts": report.get("n_contacts"),
                "n_fault_candidates": report.get("n_fault_candidates"),
                "total_s": report.get("timings", {}).get("total"),
                "status": "ok",
            }
        except Exception as exc:
            traceback.print_exc()
            results[name] = {"status": f"FAILED: {type(exc).__name__}: {exc}"}

    with open(OUT_ROOT / "summary_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # Markdown 汇总表
    lines = [
        "| 场景 | 像素准确率 | 宏IoU | 边界F1 | DEM MAE(m) | 三维体素一致率 | 耗时(s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        if r.get("status") != "ok":
            lines.append(f"| {name} | {r['status']} | | | | | |")
            continue

        def fmt(v, pct=True):
            if v is None:
                return "—"
            return f"{v * 100:.2f}%" if pct else f"{v:.1f}"

        lines.append(
            f"| {name} | {fmt(r['pixel_accuracy'])} | {fmt(r['macro_iou'])} "
            f"| {fmt(r['boundary_f1'])} | {fmt(r['dem_mae_m'], pct=False)} "
            f"| {fmt(r['voxel_3d'])} | {r['total_s']} |"
        )
    md = "\n".join(lines)
    with open(OUT_ROOT / "summary_eval.md", "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
