# -*- coding: utf-8 -*-
"""
geo2model_app.py — geo2model 图形界面（Gradio 本地网页应用）
=============================================================

启动方式（任选其一）：
    双击项目根目录的  启动界面.command
    或命令行：.venv-gempy/bin/python app/geo2model_app.py

三个页签：
    ① 快速演示   —— 浏览已完成案例的全部成果（含可交互 3D）
    ② 新地图全流程 —— 上传地质图 → 提取 → 在表格里人工审核 → 一键建模
    ③ 模型查询   —— 对任意已建模案例做虚拟钻孔 / 任意剖面 / 平切图

设计原则：界面只做"参数收集 + 表格审核 + 结果展示"，
所有计算都调用 geo2model 包里与命令行完全相同的函数——
界面跑出来的结果和脚本跑出来的结果逐字节一致。
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr  # noqa: E402

from geo2model import apps as apps_mod  # noqa: E402
from geo2model.pipeline import run_case  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data" / "output" / "geo2model"
UI_CASE = DATA_ROOT / "ui_session"  # 界面工作目录（每次"提取"覆盖重建）


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def _served(path: str | Path) -> str:
    """本地文件 → Gradio 静态服务 URL（用于在新标签页打开 3D HTML）。"""
    return f"/gradio_api/file={Path(path).resolve()}"


def _html_link(path: Path, text: str) -> str:
    if not path.is_file():
        return f"<span style='color:#999'>（{text}：尚未生成）</span>"
    return (f"<a href='{_served(path)}' target='_blank' "
            f"style='display:inline-block;padding:8px 18px;background:#2f6b4f;"
            f"color:#fff;border-radius:6px;text-decoration:none'>{text}</a>")


def _list_cases() -> list[str]:
    """扫描所有已完成三维建模的案例目录名。"""
    out = []
    if DATA_ROOT.is_dir():
        for d in sorted(DATA_ROOT.iterdir()):
            if (d / "model" / "lith_meta.json").is_file():
                out.append(d.name)
    return out


def _gallery_of(case: str) -> list[tuple[str, str]]:
    """收集一个案例的代表性成果图 [(路径, 标题), ...]。"""
    c = DATA_ROOT / case
    wanted = [
        (c / "input" / "map.png", "输入地质图"),
        (c / "extract" / "preview_clusters.png", "分区分割"),
        (c / "extract" / "preview_vectors.png", "接触线+断层候选"),
        (c / "extract" / "preview_contours.png", "等高线链"),
        (c / "model" / "geomap_rebuilt.png", "模型重建地质图"),
        (c / "model" / "section_y_mid.png", "重建剖面"),
        (c / "apps" / "borehole_2.png", "虚拟钻孔"),
        (c / "apps" / "section_1.png", "任意剖面"),
        (c / "apps" / "slice_1.png", "任意平切图"),
    ]
    return [(str(p), t) for p, t in wanted if p.is_file()]


def _metrics_md(case: str) -> str:
    """案例指标摘要（有 eval 就展示）。"""
    p = DATA_ROOT / case / "eval" / "metrics.json"
    if not p.is_file():
        return "*该案例无真值评估（真实图幅）*"
    m = json.loads(p.read_text(encoding="utf-8"))
    seg = m.get("segmentation", {})
    vox = m.get("voxel_3d", {})
    rows = [
        "| 指标 | 数值 |", "|---|---|",
        f"| 分区像素准确率 | {seg.get('pixel_accuracy', 0)*100:.2f}% |",
        f"| 宏 IoU | {seg.get('macro_iou', 0)*100:.2f}% |",
        f"| 边界 F1 (3px) | {seg.get('boundary_f1', 0)*100:.2f}% |",
    ]
    if "dem_mae_m" in m:
        rows.append(f"| DEM MAE | {m['dem_mae_m']:.1f} m |")
    if vox:
        rows.append(f"| 三维体素一致率 | {vox.get('agreement', 0)*100:.2f}% |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# 页签① 快速演示
# ---------------------------------------------------------------------------


def demo_load(case: str):
    if not case:
        return [], "请选择案例", "<span></span>"
    link = _html_link(DATA_ROOT / case / "model" / "model_3d.html",
                      "▶ 打开可交互 3D 模型")
    return _gallery_of(case), _metrics_md(case), link


# ---------------------------------------------------------------------------
# 页签② 新地图全流程
# ---------------------------------------------------------------------------


def _base_cfg(world_w, world_h, z_min, z_max, res, dark_max, merge_de,
              k_max, min_region, terr_mode, flat_z) -> dict:
    seg = {"dark_gray_max": int(dark_max), "merge_delta_e": float(merge_de),
           "k_max": int(k_max), "min_region_px": int(min_region)}
    cfg = {
        "case_dir": str(UI_CASE),
        "georef": {"world": [0.0, float(world_w), 0.0, float(world_h)]},
        "z_range": [float(z_min), float(z_max)],
        "resolution": [int(res), int(res), max(int(res) - 10, 20)],
        "segment": seg,
        "vectorize": {"dark_gray_max": int(dark_max)},
        "terrain": ({"mode": "contours"} if terr_mode == "等高线"
                    else {"mode": "flat", "flat_z": float(flat_z)}),
        "evaluate": False,
        "_project_root": str(PROJECT_ROOT),
    }
    return cfg


def ui_extract(img_path, world_w, world_h, z_min, z_max, res,
               dark_max, merge_de, k_max, min_region, terr_mode, flat_z,
               progress=gr.Progress()):
    """①提取：跑 stop_after=extract，产出预览图和三张待审核表。"""
    empty = (None, None, None,
             pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "")
    if not img_path:
        return (*empty[:-1], "⚠ 请先上传地质图")
    try:
        progress(0.05, desc="准备工作目录")
        if UI_CASE.exists():
            shutil.rmtree(UI_CASE)
        (UI_CASE / "input").mkdir(parents=True)
        shutil.copy(img_path, UI_CASE / "input" / "map.png")

        cfg = _base_cfg(world_w, world_h, z_min, z_max, res, dark_max,
                        merge_de, k_max, min_region, terr_mode, flat_z)
        cfg["stop_after"] = "extract"
        progress(0.2, desc="分割 / 矢量化 / 等高线提取中…")
        run_case(cfg)

        ext = UI_CASE / "extract"
        units = pd.read_csv(ext / "units_auto.csv")
        units["series"] = "Strat_Series"
        units = units[["unit_id", "name", "role", "order", "series",
                       "dip", "azimuth", "color_r", "color_g", "color_b"]]

        faults = json.loads((ext / "faults.json").read_text(encoding="utf-8"))
        fdf = pd.DataFrame(
            [{"fault_id": f["fault_id"], "name": f["name"],
              "confirmed": bool(f.get("confirmed")),
              "dip": f.get("dip"), "azimuth": f.get("azimuth"),
              "length_px": f.get("length_px"),
              "straightness": f.get("straightness")} for f in faults]
        )

        contours = json.loads((ext / "contours.json").read_text(encoding="utf-8"))
        from geo2model.lines import polyline_length
        cdf = pd.DataFrame(
            [{"contour_id": c["contour_id"],
              "length_px": round(polyline_length(c["polyline_px"]), 0),
              "elevation": c.get("elevation")} for c in contours]
        )

        def p(name):
            f = ext / name
            return str(f) if f.is_file() else None

        msg = (f"✅ 提取完成：{len(units)} 个分区、{len(fdf)} 条断层候选、"
               f"{len(cdf)} 条等高线链。\n"
               "请在下面三张表里完成人工审核（名称/层序 order 新→老=0,1,2…/"
               "role 改 water·ignore/断层 confirmed 打 ✓/等高线填 elevation），"
               "然后点『② 审核入库并建模』。")
        return (p("preview_clusters.png"), p("preview_vectors.png"),
                p("preview_contours.png"), units, fdf, cdf, msg)
    except Exception:
        return (*empty[:-1], "❌ 提取失败：\n```\n" + traceback.format_exc()[-1500:] + "\n```")


def ui_model(world_w, world_h, z_min, z_max, res, dark_max, merge_de,
             k_max, min_region, terr_mode, flat_z,
             units_df, faults_df, contours_df, manual_ori_text, export_mesh,
             progress=gr.Progress()):
    """②建模：把三张审核表变成 review 配置，跑完整流水线。"""
    fail = (None, None, "<span></span>", "")
    if not (UI_CASE / "extract" / "labels.npy").is_file():
        return (*fail[:-1], "⚠ 请先执行『① 运行提取』")
    try:
        # ---- 审核表 → review 字典 ----
        units_edits = []
        for _, r in pd.DataFrame(units_df).iterrows():
            e = {"unit_id": int(r["unit_id"]), "name": str(r["name"]),
                 "role": str(r["role"]), "order": int(r["order"]),
                 "series": str(r.get("series") or "Strat_Series")}
            for k in ("dip", "azimuth"):
                v = r.get(k)
                if v is not None and str(v) not in ("", "nan", "None"):
                    e[k] = float(v)
            units_edits.append(e)
        fdf = pd.DataFrame(faults_df)
        confirm = ([int(r["fault_id"]) for _, r in fdf.iterrows()
                    if bool(r.get("confirmed"))] if len(fdf) else [])
        edits = {str(int(r["fault_id"])): {"dip": float(r["dip"]),
                                           "azimuth": float(r["azimuth"])}
                 for _, r in fdf.iterrows()
                 if bool(r.get("confirmed"))} if len(fdf) else {}

        elevations = {}
        for _, r in pd.DataFrame(contours_df).iterrows():
            v = r.get("elevation")
            if v is not None and str(v) not in ("", "nan", "None"):
                elevations[str(int(r["contour_id"]))] = float(v)

        manual = []
        for line in (manual_ori_text or "").strip().splitlines():
            try:
                u, v, dip, az = (float(t) for t in line.replace("，", ",").split(",")[:4])
                manual.append({"u": u, "v": v, "dip": dip, "azimuth": az})
            except Exception:
                pass  # 忽略格式不对的行

        cfg = _base_cfg(world_w, world_h, z_min, z_max, res, dark_max,
                        merge_de, k_max, min_region, terr_mode, flat_z)
        cfg["review"] = {"units": units_edits, "fault_confirm": confirm,
                         "fault_edits": edits}
        if cfg["terrain"]["mode"] == "contours":
            cfg["terrain"]["elevations"] = elevations
        if manual:
            cfg["manual_orientations"] = manual
        cfg["model"] = {"also_html": True, "also_mesh_exports": bool(export_mesh)}
        cfg["apps"] = "auto"

        progress(0.15, desc="数据库构建 + GemPy 建模中（约 1 分钟）…")
        run_case(cfg)

        mdir = UI_CASE / "model"
        link = _html_link(mdir / "model_3d.html", "▶ 打开可交互 3D 模型")
        n_ori = len(manual)
        msg = (f"✅ 建模完成！人工产状 {n_ori} 条、确认断层 {len(confirm)} 条、"
               f"等高线赋值 {len(elevations)} 条。\n产物目录：`{UI_CASE}`；"
               "可去『③ 模型查询』页签做钻孔/剖面/平切。")
        return (str(mdir / "geomap_rebuilt.png") if (mdir / "geomap_rebuilt.png").is_file() else None,
                str(mdir / "section_y_mid.png") if (mdir / "section_y_mid.png").is_file() else None,
                link, msg)
    except Exception:
        return (*fail[:-1], "❌ 建模失败：\n```\n" + traceback.format_exc()[-1500:] + "\n```")


# ---------------------------------------------------------------------------
# 页签③ 模型查询
# ---------------------------------------------------------------------------


def q_extent(case: str) -> str:
    try:
        meta = json.loads((DATA_ROOT / case / "model" / "lith_meta.json")
                          .read_text(encoding="utf-8"))
        x0, x1, y0, y1, z0, z1 = meta["extent"]
        return (f"模型范围：X {x0:.0f}–{x1:.0f} m，Y {y0:.0f}–{y1:.0f} m，"
                f"Z {z0:.0f}–{z1:.0f} m")
    except Exception:
        return "（请选择案例）"


def _load(case):
    return apps_mod.load_lith(DATA_ROOT / case / "model")


def q_borehole(case, x, y):
    try:
        lith, meta = _load(case)
        out = DATA_ROOT / case / "apps" / "ui_borehole.png"
        out.parent.mkdir(exist_ok=True)
        apps_mod.virtual_borehole(lith, meta, float(x), float(y), out)
        return str(out), ""
    except Exception as e:
        return None, f"❌ {e}"


def q_section(case, x1, y1, x2, y2):
    try:
        lith, meta = _load(case)
        out = DATA_ROOT / case / "apps" / "ui_section.png"
        out.parent.mkdir(exist_ok=True)
        apps_mod.arbitrary_section(lith, meta, (float(x1), float(y1)),
                                   (float(x2), float(y2)), out)
        return str(out), ""
    except Exception as e:
        return None, f"❌ {e}"


def q_slice(case, z):
    try:
        lith, meta = _load(case)
        out = DATA_ROOT / case / "apps" / "ui_slice.png"
        out.parent.mkdir(exist_ok=True)
        apps_mod.level_slice(lith, meta, float(z), out)
        return str(out), ""
    except Exception as e:
        return None, f"❌ {e}"


# ---------------------------------------------------------------------------
# 界面布局
# ---------------------------------------------------------------------------

with gr.Blocks(title="geo2model · 平面地质图智能化三维构建") as demo:
    gr.Markdown(
        "# geo2model · 基于平面地质图的智能化三维构建\n"
        "输入一张 JPG/PNG 地质图 → 半自动矢量化 → 人工审核 → GemPy 三维模型 "
        "→ 虚拟钻孔 / 任意剖面 / 平切图。兰州大学大创项目原型。"
    )

    # ---------------- ① 快速演示 ----------------
    with gr.Tab("① 快速演示"):
        gr.Markdown("浏览已完成案例的全部成果（合成基准含定量评分）。")
        with gr.Row():
            demo_case = gr.Dropdown(choices=_list_cases(), label="选择案例",
                                    value=("synth_base_deep" if "synth_base_deep"
                                           in _list_cases() else None))
            demo_btn = gr.Button("加载案例", variant="primary")
            demo_refresh = gr.Button("刷新列表")
        demo_link = gr.HTML()
        with gr.Row():
            demo_gallery = gr.Gallery(label="成果一览", columns=3, height=520)
            demo_metrics = gr.Markdown()
        demo_btn.click(demo_load, [demo_case],
                       [demo_gallery, demo_metrics, demo_link])
        demo_refresh.click(lambda: gr.Dropdown(choices=_list_cases()),
                           None, demo_case)

    # ---------------- ② 新地图全流程 ----------------
    with gr.Tab("② 新地图全流程"):
        gr.Markdown(
            "**第一步** 上传地图并设置参数 → 点『① 运行提取』；"
            "**第二步** 在三张表里完成人工审核 → 点『② 审核入库并建模』。"
        )
        with gr.Row():
            up_img = gr.Image(type="filepath", label="上传地质图 (PNG/JPG)")
            with gr.Column():
                with gr.Row():
                    in_ww = gr.Number(value=2000, label="图幅宽度对应世界 X (米)")
                    in_wh = gr.Number(value=2000, label="图幅高度对应世界 Y (米)")
                with gr.Row():
                    in_z0 = gr.Number(value=0, label="模型底 Z (米)")
                    in_z1 = gr.Number(value=1000, label="模型顶 Z (米)")
                    in_res = gr.Number(value=50, precision=0, label="体素分辨率")
                with gr.Row():
                    in_dark = gr.Number(value=110, precision=0,
                                        label="暗线阈值 dark_gray_max")
                    in_de = gr.Number(value=5.0, label="合并阈值 ΔE")
                    in_k = gr.Number(value=22, precision=0, label="过聚类 K")
                    in_minpx = gr.Number(value=150, precision=0, label="最小区域(px)")
                with gr.Row():
                    in_terr = gr.Radio(["等高线", "平坦"], value="等高线",
                                       label="地形模式")
                    in_flatz = gr.Number(value=500, label="平坦地形高程 (米)")
        btn_extract = gr.Button("① 运行提取", variant="primary")
        ex_status = gr.Markdown()
        with gr.Row():
            ex_p1 = gr.Image(label="分区分割预览", interactive=False)
            ex_p2 = gr.Image(label="接触线+断层候选", interactive=False)
            ex_p3 = gr.Image(label="等高线链（按编号赋高程）", interactive=False)
        gr.Markdown("### 人工审核（直接在表格里改）")
        units_tbl = gr.Dataframe(label="单元表：name 命名 / order 新→老 0,1,2… / "
                                       "role= strata·water·ignore / series 不整合分组 / "
                                       "dip·azimuth 可选产状",
                                 interactive=True)
        with gr.Row():
            faults_tbl = gr.Dataframe(label="断层候选：confirmed 勾选确认，可改 dip/azimuth",
                                      interactive=True)
            contours_tbl = gr.Dataframe(label="等高线链：填 elevation（米），不确定的留空",
                                        interactive=True)
        manual_ori = gr.Textbox(label="产状符号读数（可选，每行：u,v,dip,azimuth 像素坐标+度）",
                                lines=3, placeholder="例：1831,383,25,135")
        export_mesh = gr.Checkbox(value=False, label="同时导出 OBJ/glTF/VTK（稍慢）")
        btn_model = gr.Button("② 审核入库并建模", variant="primary")
        md_status = gr.Markdown()
        md_link = gr.HTML()
        with gr.Row():
            md_geomap = gr.Image(label="模型重建地质图（与输入对比）", interactive=False)
            md_section = gr.Image(label="重建剖面", interactive=False)

        params = [in_ww, in_wh, in_z0, in_z1, in_res, in_dark, in_de,
                  in_k, in_minpx, in_terr, in_flatz]
        btn_extract.click(ui_extract, [up_img, *params],
                          [ex_p1, ex_p2, ex_p3, units_tbl, faults_tbl,
                           contours_tbl, ex_status])
        btn_model.click(ui_model,
                        [*params, units_tbl, faults_tbl, contours_tbl,
                         manual_ori, export_mesh],
                        [md_geomap, md_section, md_link, md_status])

    # ---------------- ③ 模型查询 ----------------
    with gr.Tab("③ 模型查询"):
        gr.Markdown("对任意已建模案例做工程查询（申报书三项应用扩展）。")
        with gr.Row():
            q_case = gr.Dropdown(choices=_list_cases(), label="选择案例",
                                 value=("synth_base_deep" if "synth_base_deep"
                                        in _list_cases() else None))
            q_refresh = gr.Button("刷新列表")
        q_info = gr.Markdown()
        q_case.change(q_extent, [q_case], [q_info])
        q_refresh.click(lambda: gr.Dropdown(choices=_list_cases()), None, q_case)
        with gr.Row():
            with gr.Column():
                gr.Markdown("**虚拟钻孔**")
                bh_x = gr.Number(value=1000, label="X (米)")
                bh_y = gr.Number(value=1000, label="Y (米)")
                bh_btn = gr.Button("打钻！", variant="primary")
                bh_err = gr.Markdown()
            bh_img = gr.Image(label="钻孔柱状图", interactive=False)
        bh_btn.click(q_borehole, [q_case, bh_x, bh_y], [bh_img, bh_err])
        with gr.Row():
            with gr.Column():
                gr.Markdown("**任意剖面**（两个端点）")
                with gr.Row():
                    sc_x1 = gr.Number(value=100, label="X1")
                    sc_y1 = gr.Number(value=500, label="Y1")
                with gr.Row():
                    sc_x2 = gr.Number(value=1900, label="X2")
                    sc_y2 = gr.Number(value=1500, label="Y2")
                sc_btn = gr.Button("切剖面！", variant="primary")
                sc_err = gr.Markdown()
            sc_img = gr.Image(label="剖面图", interactive=False)
        sc_btn.click(q_section, [q_case, sc_x1, sc_y1, sc_x2, sc_y2],
                     [sc_img, sc_err])
        with gr.Row():
            with gr.Column():
                gr.Markdown("**任意平切图**")
                sl_z = gr.Number(value=400, label="切片高程 Z (米)")
                sl_btn = gr.Button("平切！", variant="primary")
                sl_err = gr.Markdown()
            sl_img = gr.Image(label="平切图", interactive=False)
        sl_btn.click(q_slice, [q_case, sl_z], [sl_img, sl_err])

    gr.Markdown("---\n技术报告：`技术报告/` · 手册：`docs/使用说明_geo2model.md` · "
                "命令行与界面产物完全一致")


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        allowed_paths=[str(DATA_ROOT)],
        inbrowser=True,
        server_name="127.0.0.1",
        show_error=True,
    )
