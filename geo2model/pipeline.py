# -*- coding: utf-8 -*-
"""
pipeline.py — 端到端流程编排（地质图 → 三维模型 → 应用 → 评估）
=================================================================

按申报书四阶段串联各模块，每个 case 由一个 JSON 配置驱动：

{
  "case_dir": "data/output/geo2model/synth_base",
  "georef":   {"world": [0, 2000, 0, 2000]},
  "z_range":  [0, 1000],
  "resolution": [50, 50, 50],
  "segment":  {...SegmentParams 字段覆盖...},
  "terrain":  {"mode": "contours",           # contours | flat
               "elevations": "from_truth",    # 或 {"0": 650, ...} 人工映射
               "flat_z": 600},
  "review":   "from_truth",                   # 或 {"units": [...], "fault_confirm": "all"}
  "constraints": {..., "use_section_picks": false},
  "model":    {"also_html": true},
  "apps":     "auto",                         # 或 {"boreholes": [...], ...}
  "evaluate": true
}

人机交互检查点（半自动流程的暂停位）：
  ① segment 后 → extract/units_auto.csv + preview（命名/层序/角色）
  ② vectorize 后 → extract/faults.json（断层确认与产状）
  ③ terrain 后 → extract/preview_contours.png（等高线赋高程）
批量实验时这三处由配置(或真值模拟)自动完成，交互模式则逐步暂停。
"""

from __future__ import annotations

import json
import time
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

from .georef import GeoRef


def _override(dc_cls, overrides: dict | None):
    """用配置字典覆盖 dataclass 默认参数（只认识的字段）。"""
    kw = {}
    if overrides:
        valid = {f.name for f in fields(dc_cls)}
        kw = {k: v for k, v in overrides.items() if k in valid}
    return dc_cls(**kw)


def run_case(cfg: dict, interactive: bool = False) -> dict:
    """执行一个完整案例。返回各阶段摘要与产物路径。"""
    from . import apps as apps_mod
    from . import metrics as metrics_mod
    from .constraints import ConstraintParams, build_constraints
    from .geodatabase import apply_review
    from .model3d import ModelParams, build_model
    from .segment import SegmentParams, segment_map, save_extract
    from .terrain import (
        TerrainParams, extract_contours, contours_preview,
        assign_elevations_from_truth, apply_elevations, contours_to_dem,
    )
    from .vectorize import VectorizeParams, vectorize_map

    t_all = time.time()
    case_dir = Path(cfg["case_dir"])
    extract_dir = case_dir / "extract"
    db_dir = case_dir / "db"
    model_dir = case_dir / "model"
    apps_dir = case_dir / "apps"
    eval_dir = case_dir / "eval"
    for d in (extract_dir, db_dir, model_dir, apps_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    map_path = Path(cfg.get("map_path", case_dir / "input" / "map.png"))
    img_bgr = cv2.imread(str(map_path))
    if img_bgr is None:
        raise FileNotFoundError(f"读不到地质图: {map_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    georef = GeoRef(w, h, tuple(cfg["georef"]["world"]))
    z_range = tuple(cfg.get("z_range", (0.0, 1000.0)))
    timings: dict[str, float] = {}
    report: dict = {"case": str(case_dir), "timings": timings}

    truth_dir = case_dir / "truth"
    truth_meta_path = truth_dir / "truth_meta.json"
    has_truth = truth_meta_path.is_file()

    # ================= 阶段 1a：地层分区分割 =================
    t0 = time.time()
    seg_params = _override(SegmentParams, cfg.get("segment"))
    seg = segment_map(img, seg_params)
    save_extract(seg, extract_dir, img_rgb=img)
    timings["segment"] = round(time.time() - t0, 2)
    report["n_units_auto"] = len(seg["units"])
    print(f"[pipeline] 分割完成: {len(seg['units'])} 个分区 "
          f"({timings['segment']}s)")

    if interactive:
        input("→ 人机交互①: 请检查/编辑 extract/units_auto.csv 与预览图，回车继续 ")

    # ================= 阶段 1b：界线与断层矢量化 =================
    t0 = time.time()
    vec_params = _override(VectorizeParams, cfg.get("vectorize"))
    vec = vectorize_map(img, seg["labels"], extract_dir, vec_params)
    timings["vectorize"] = round(time.time() - t0, 2)
    report["n_contacts"] = len(vec["contacts"])
    report["n_fault_candidates"] = len(vec["faults"])

    if interactive:
        input("→ 人机交互②: 请确认 extract/faults.json 中的断层（confirmed 字段），回车继续 ")

    # ================= 阶段 1c：等高线 → DEM =================
    t0 = time.time()
    terr_cfg = cfg.get("terrain", {"mode": "flat"})
    terr_params = _override(TerrainParams, terr_cfg)
    contours: list = []
    if terr_cfg.get("mode", "contours") == "contours":
        contours = extract_contours(img, terr_params)
        contours_preview(img, contours, extract_dir / "preview_contours.png")
        elev = terr_cfg.get("elevations")
        if elev == "from_truth" and (truth_dir / "contours_gt.json").is_file():
            with open(truth_dir / "contours_gt.json", encoding="utf-8") as f:
                contours_gt = json.load(f)
            mapping = assign_elevations_from_truth(contours, contours_gt)
        elif isinstance(elev, dict):
            mapping = {int(k): float(v) for k, v in elev.items()}
        elif terr_cfg.get("elevations_file"):
            # 人工赋值文件（真实地图工作流）：{"elevations": {"链id": 高程}}
            ef = Path(terr_cfg["elevations_file"])
            if not ef.is_absolute():
                ef = Path(cfg.get("_project_root", ".")) / ef
            with open(ef, encoding="utf-8") as f:
                mapping = {int(k): float(v)
                           for k, v in json.load(f)["elevations"].items()}
        else:
            mapping = {}
        if interactive and not mapping:
            input("→ 人机交互③: 请按 preview_contours.png 编辑配置里的等高线高程映射，回车继续 ")
        apply_elevations(contours, mapping)
        dem = contours_to_dem(
            contours, (h, w), terr_params,
            fallback_z=float(terr_cfg.get("flat_z", np.mean(z_range))),
        )
        report["n_contours"] = len(contours)
        report["n_contours_with_z"] = sum(c["elevation"] is not None for c in contours)
    else:
        dem = np.full((h, w), float(terr_cfg.get("flat_z", np.mean(z_range))),
                      dtype=np.float32)
    np.save(extract_dir / "dem.npy", dem)
    with open(extract_dir / "contours.json", "w", encoding="utf-8") as f:
        json.dump(contours, f, ensure_ascii=False)
    timings["terrain"] = round(time.time() - t0, 2)

    # 只做提取阶段（人机交互工作流：先看提取结果，再写审核配置）
    if cfg.get("stop_after") == "extract":
        print("[pipeline] stop_after=extract，提取阶段完成。请查看 "
              f"{extract_dir} 下的 units_auto.csv 与 preview_*.png")
        return report

    # ================= 阶段 2：数据库构建（人机校正合入） =================
    t0 = time.time()
    review = cfg.get("review")
    apply_review(extract_dir, db_dir, review,
                 truth_meta_path=truth_meta_path if has_truth else None)

    cons_cfg = dict(cfg.get("constraints") or {})
    use_picks = cons_cfg.pop("use_section_picks", False)
    cons_params = _override(ConstraintParams, cons_cfg)

    # 可选：剖面图深部约束（像素比例映射配准，见 sectionreg.py）
    # 支持多条剖面：section / section2 / ...
    deep_points = None
    if use_picks:
        import pandas as pd
        from .sectionreg import SectionFrame, picks_to_surface_points

        deep_list = []
        for prefix in ("section", "section2", "section3"):
            picks_csv = truth_dir / f"{prefix}_picks_gt.csv"
            frame_json = truth_dir / f"{prefix}_frame.json"
            if not (picks_csv.is_file() and frame_json.is_file()):
                continue
            with open(frame_json, encoding="utf-8") as f:
                frame = SectionFrame.from_dict(json.load(f))
            picks = pd.read_csv(picks_csv).to_dict("records")
            deep_list.append(picks_to_surface_points(picks, frame))
        if deep_list:
            deep_points = pd.concat(deep_list, ignore_index=True)
            report["n_deep_picks"] = len(deep_points)

    # 通用深部点文件（真实图幅：人工读剖面/钻孔录入的世界坐标点，
    # 列 X,Y,Z,formation）——申报书预留的钻孔/物探数据输入窗口
    if cons_cfg.get("deep_points_files"):
        import pandas as pd
        dfs = [] if deep_points is None else [deep_points]
        for pth in cons_cfg.pop("deep_points_files"):
            p = Path(pth)
            if not p.is_absolute():
                p = Path(cfg.get("_project_root", ".")) / p
            if p.is_file():
                dfs.append(pd.read_csv(p))
                print(f"[pipeline] 读入深部点文件: {p.name} ({len(dfs[-1])} 点)")
            else:
                print(f"[pipeline] 警告: 深部点文件不存在: {p}")
        if dfs:
            deep_points = pd.concat(dfs, ignore_index=True)

    # 人工产状录入（半自动通道）：来源可以是
    #   a) 合成评估: truth/dip_symbols_gt.csv 模拟"用户点选图面产状符号并键入读数"
    #   b) 真实地图: 配置里的 manual_orientations 列表 [{"u","v","dip","azimuth"},...]
    manual_ori = None
    manual_src: list[dict] = []
    if review == "from_truth" and (truth_dir / "dip_symbols_gt.csv").is_file():
        import pandas as pd
        manual_src = pd.read_csv(truth_dir / "dip_symbols_gt.csv").to_dict("records")
    elif cfg.get("manual_orientations"):
        manual_src = list(cfg["manual_orientations"])
    if manual_src:
        import pandas as pd
        from .geodatabase import load_units

        units_df = load_units(db_dir)
        uid2name = dict(zip(units_df["unit_id"].astype(int), units_df["name"]))
        uid2role = dict(zip(units_df["unit_id"].astype(int), units_df["role"]))
        rows = []
        for s in manual_src:
            u_px, v_px = float(s["u"]), float(s["v"])
            ui, vi = int(np.clip(u_px, 0, w - 1)), int(np.clip(v_px, 0, h - 1))
            uid = int(seg["labels"][vi, ui])
            if uid2role.get(uid) != "strata":
                continue
            xw, yw = georef.px_to_world(u_px, v_px)
            zw = float(dem[vi, ui])
            rows.append({"X": float(xw), "Y": float(yw), "Z": zw,
                         "azimuth": float(s["azimuth"]), "dip": float(s["dip"]),
                         "polarity": 1, "formation": uid2name[uid]})
        if rows:
            manual_ori = pd.DataFrame(rows)

    summary = build_constraints(
        extract_dir, db_dir, georef, z_range,
        resolution=tuple(cfg.get("resolution", (50, 50, 50))),
        params=cons_params,
        extra_surface_points=deep_points,
        extra_orientations=manual_ori,
    )
    report["constraints"] = summary
    timings["database"] = round(time.time() - t0, 2)

    # ================= 阶段 3：GemPy 三维建模 =================
    t0 = time.time()
    model_params = _override(ModelParams, cfg.get("model"))
    model_out = build_model(db_dir, model_dir, dem_px=dem, params=model_params)
    timings["model3d"] = round(time.time() - t0, 2)
    report["model"] = model_out

    # ================= 阶段 4：应用扩展 =================
    t0 = time.time()
    apps_cfg = cfg.get("apps", "auto")
    x0, x1, y0, y1 = georef.world
    if apps_cfg == "auto":
        apps_cfg = {
            "boreholes": [
                [x0 + 0.3 * (x1 - x0), y0 + 0.4 * (y1 - y0)],
                [x0 + 0.62 * (x1 - x0), y0 + 0.55 * (y1 - y0)],
                [x0 + 0.8 * (x1 - x0), y0 + 0.75 * (y1 - y0)],
            ],
            "sections": [
                [[x0 + 50, y0 + 0.35 * (y1 - y0)], [x1 - 50, y0 + 0.75 * (y1 - y0)]],
            ],
            "slices": [
                z_range[0] + 0.45 * (z_range[1] - z_range[0]),
                z_range[0] + 0.65 * (z_range[1] - z_range[0]),
            ],
        }
    if apps_cfg:
        lith, meta = apps_mod.load_lith(model_dir)
        apps_out = apps_mod.run_apps(
            model_dir, apps_dir,
            boreholes=[tuple(b) for b in apps_cfg.get("boreholes", [])],
            sections=[(tuple(s[0]), tuple(s[1])) for s in apps_cfg.get("sections", [])],
            slices=list(apps_cfg.get("slices", [])),
        )
        report["apps"] = apps_out
    timings["apps"] = round(time.time() - t0, 2)

    # ================= 阶段 5：定量评估（有真值时） =================
    if cfg.get("evaluate", True) and has_truth:
        t0 = time.time()
        report["eval"] = evaluate_case(case_dir, cfg)
        timings["evaluate"] = round(time.time() - t0, 2)

    timings["total"] = round(time.time() - t_all, 2)
    with open(eval_dir / "pipeline_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print(f"[pipeline] 案例完成，总耗时 {timings['total']}s → {case_dir}")
    return report


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------


def evaluate_case(case_dir: str | Path, cfg: dict | None = None) -> dict:
    """与真值对比：分区精度 / 边界 F1 / DEM 误差 / 三维体素一致率。"""
    import pandas as pd

    from . import metrics as metrics_mod
    from .mapgen import Scenario, truth_voxel_units

    case_dir = Path(case_dir)
    extract_dir, db_dir, model_dir, eval_dir = (
        case_dir / "extract", case_dir / "db", case_dir / "model", case_dir / "eval"
    )
    out: dict = {}

    with open(case_dir / "truth" / "truth_meta.json", encoding="utf-8") as f:
        tmeta = json.load(f)

    # ---- 1. 分区识别精度（申报书核心指标） ----
    labels = np.load(extract_dir / "labels.npy")
    labels_gt = np.load(case_dir / "truth" / "labels_gt.npy")
    seg_m = metrics_mod.segmentation_metrics(labels, labels_gt, tol_px=3)
    out["segmentation"] = seg_m

    # ---- 2. DEM 误差 ----
    dem_path = extract_dir / "dem.npy"
    if dem_path.is_file():
        dem = np.load(dem_path)
        dem_gt = np.load(case_dir / "truth" / "dem_gt.npy")
        out["dem_mae_m"] = float(np.mean(np.abs(dem - dem_gt)))

    # ---- 3. 三维体素一致率（重建模型 vs 真值模型） ----
    lith_path = model_dir / "lith_block.npy"
    if lith_path.is_file():
        with open(model_dir / "lith_meta.json", encoding="utf-8") as f:
            lmeta = json.load(f)
        lith = np.load(lith_path)
        scn = Scenario(**tmeta["scenario"])
        gt_units, below = truth_voxel_units(
            scn, lmeta["extent"], lmeta["resolution"]
        )
        # 双方都转成「单元名称」再比（id 体系不同）
        id2name_gt = {u["unit_id"]: u["name"] for u in tmeta["units"]}
        gt_names = np.vectorize(lambda t: id2name_gt.get(int(t), "?"))(gt_units)
        id2name_pred = {int(k): v for k, v in lmeta["id_to_name"].items()}
        pred_names = np.vectorize(lambda t: id2name_pred.get(int(t), "?"))(lith)
        vox = metrics_mod.array_agreement(pred_names, gt_names, mask=below)
        # 诚实口径补充：micro 一致率会被"深部必是基底"的平凡正确体素
        # 抬高，故同时报告 宏平均(各单元等权)、剔除基底后的一致率、
        # 基底体积占比与"全猜基底"基线
        basement = tmeta["units"][-1]["name"]
        vox["macro_agreement"] = float(np.mean(list(vox["per_value"].values())))
        nb_mask = below & (gt_names != basement)
        if nb_mask.any():
            vox["agreement_excl_basement"] = float(
                (pred_names[nb_mask] == gt_names[nb_mask]).mean())
        vox["basement_share"] = float((gt_names[below] == basement).mean())
        out["voxel_3d"] = vox

    with open(eval_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)

    pa = seg_m.get("pixel_accuracy")
    va = out.get("voxel_3d", {}).get("agreement")
    print(f"[evaluate] 分区像素准确率={pa:.4f}"
          + (f", 三维体素一致率={va:.4f}" if va is not None else "")
          + (f", DEM MAE={out['dem_mae_m']:.1f} m" if "dem_mae_m" in out else ""))
    return out
