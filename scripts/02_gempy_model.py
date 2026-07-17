#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_gempy_model.py
=================
用合成 CSV 数据构建 GemPy 三维地质模型（褶皱 + 断层 + 地形起伏），
并导出 2D 剖面图 / 可交互 3D。

输入（由 01_make_synthetic_data.py 生成）：
------------------------------------------
- data/csv/synthetic/surface_points.csv   # 含地层界面 + Main_Fault
- data/csv/synthetic/orientations.csv

输出：
------
- data/output/synthetic/section_y_mid.png   # 沿 Y 中间的 XZ 剖面（最能看见褶皱/断层）
- data/output/synthetic/section_x_mid.png   # 沿 X 中间的 YZ 剖面
- data/output/synthetic/input_data_2d.png   # 仅输入点/产状的 2D 视图
- data/output/synthetic/geomap_topo.png     # 地形地质图（有地形时）
- data/output/synthetic/model_3d.html       # ★ 可交互 3D（浏览器旋转/缩放）
- data/output/synthetic/model_3d.png        # 静态预览截图（可选）

依赖版本（本项目实测）：
------------------------
- gempy 2026.0.3
- gempy_viewer 2026.0.3
- gempy_engine 2026.0.3
- pyvista + trame（HTML 交互导出）

运行方式：
---------
    cd geo3d_project
    .venv-gempy/bin/python scripts/01_make_synthetic_data.py   # 若尚未生成数据
    .venv-gempy/bin/python scripts/02_gempy_model.py           # 导出 HTML 并尝试打开浏览器
    .venv-gempy/bin/python scripts/02_gempy_model.py --interactive  # 额外弹出桌面 3D 窗口
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

# 无界面保存图片：必须在 import pyplot / gempy_viewer 之前设置
# 这样服务器/无显示器环境也能出图
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import numpy as np

import gempy as gp
import gempy_viewer as gpv
import matplotlib.pyplot as plt
from gempy.API.grid_API import set_active_grid
from gempy.core.data.grid_modules.topography import Topography

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_ROOT / "data" / "csv" / "synthetic"
OUT_DIR = PROJECT_ROOT / "data" / "output" / "synthetic"

SURFACE_POINTS_CSV = CSV_DIR / "surface_points.csv"
ORIENTATIONS_CSV = CSV_DIR / "orientations.csv"

# 与 01_make_synthetic_data.py 保持一致
EXTENT = [0.0, 2000.0, 0.0, 2000.0, 0.0, 1000.0]

# 地层从新到老；断层单独 series（更年轻，会切穿地层）
LAYER_ORDER = ("Shale", "Sandstone", "Claystone", "Limestone")
FAULT_NAME = "Main_Fault"
FAULT_SERIES = "Fault_Series"
STRAT_SERIES = "Strat_Series"

# 网格略加密，更好表现褶皱曲率与断层面
RESOLUTION = [50, 50, 50]


def _require_input_files() -> None:
    """检查输入 CSV 是否存在，缺失则给出明确提示。"""
    missing = [p for p in (SURFACE_POINTS_CSV, ORIENTATIONS_CSV) if not p.is_file()]
    if missing:
        msg = (
            "缺少输入 CSV 文件：\n  - "
            + "\n  - ".join(str(p) for p in missing)
            + "\n\n请先运行：\n"
            "  .venv-gempy/bin/python scripts/01_make_synthetic_data.py"
        )
        raise FileNotFoundError(msg)


def build_model() -> gp.data.GeoModel:
    """
    从 CSV 创建 GeoModel：断层 series + 地层 series。

    步骤：
    1. ImporterHelper 读 surface points / orientations
    2. map_stack_to_surfaces：Fault_Series（年轻）在前，Strat_Series（老）在后
    3. set_is_fault：把 Fault_Series 标成真正的断层（可错断地层）
    """
    _require_input_files()

    print("[1/5] 读取 CSV 并创建 GeoModel ...")
    print(f"  surface_points : {SURFACE_POINTS_CSV}")
    print(f"  orientations   : {ORIENTATIONS_CSV}")
    print(f"  extent         : {EXTENT}")
    print(f"  resolution     : {RESOLUTION}")

    geo_model = gp.create_geomodel(
        project_name="synthetic_fold_fault_topo",
        extent=EXTENT,
        resolution=RESOLUTION,
        importer_helper=gp.data.ImporterHelper(
            path_to_surface_points=str(SURFACE_POINTS_CSV),
            path_to_orientations=str(ORIENTATIONS_CSV),
            coord_x_name="X",
            coord_y_name="Y",
            coord_z_name="Z",
            surface_name="formation",
        ),
    )

    print("[2/5] 映射断层 + 地层序列（断层更年轻，切穿地层）...")
    # 字典键顺序：先写断层组，再写地层组（从新到老）
    gp.map_stack_to_surfaces(
        gempy_model=geo_model,
        mapping_object={
            FAULT_SERIES: FAULT_NAME,
            STRAT_SERIES: LAYER_ORDER,
        },
    )
    # 声明 Fault_Series 为断层关系（而不是普通不整合/侵蚀）
    gp.set_is_fault(
        frame=geo_model,
        fault_groups=[FAULT_SERIES],
    )

    print(geo_model.structural_frame)
    return geo_model


def add_topography(geo_model: gp.data.GeoModel) -> None:
    """
    添加有起伏的合成 DEM 地形。

    设计意图（便于在 3D 里一眼看出地形）：
    - 大尺度丘陵：正弦/余弦叠加
    - 沿断层迹线附近略微下凹（模拟断层谷）
    - 高程裁剪在模型盒内，避免贴顶/贴底

    实现：构造 values_2d (nx, ny, 3) = (X, Y, Z)，挂到 grid.topography。
    """
    print("[3/5] 添加合成地形起伏 ...")
    # 用规则网格的分辨率；若没有 dense regular_grid 则退回 50×50
    try:
        nx, ny = int(geo_model.grid.regular_grid.resolution[0]), int(
            geo_model.grid.regular_grid.resolution[1]
        )
    except Exception:
        nx, ny = 50, 50

    xs = np.linspace(EXTENT[0], EXTENT[1], nx)
    ys = np.linspace(EXTENT[2], EXTENT[3], ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # 基础高程 + 多尺度起伏（单位：米）
    Z = (
        700.0
        + 110.0 * np.sin(2.0 * np.pi * X / 1700.0)
        + 75.0 * np.cos(2.0 * np.pi * Y / 1300.0)
        + 45.0 * np.sin(2.0 * np.pi * (X + 0.6 * Y) / 2100.0)
        + 25.0 * np.cos(2.0 * np.pi * X / 600.0) * np.sin(2.0 * np.pi * Y / 800.0)
    )
    # 断层谷：在 X≈1050 附近下凹
    Z = Z - 55.0 * np.exp(-((X - 1050.0) ** 2) / (2.0 * 180.0**2))

    z_min, z_max = EXTENT[4] + 80.0, EXTENT[5] - 40.0
    Z = np.clip(Z, z_min, z_max)

    values_2d = np.stack([X, Y, Z], axis=-1)
    geo_model.grid.topography = Topography(
        _regular_grid=geo_model.grid.regular_grid,
        values_2d=values_2d,
    )
    set_active_grid(geo_model.grid, [gp.data.Grid.GridTypes.TOPOGRAPHY])
    print(
        f"  地形高程范围: {float(Z.min()):.1f} – {float(Z.max()):.1f} m "
        f"(分辨率 {nx}×{ny})"
    )


def compute(geo_model: gp.data.GeoModel):
    """调用 GemPy 隐式插值求解器计算模型（含地形网格）。"""
    print("[4/5] 计算三维地质模型 (gp.compute_model) ...")
    sol = gp.compute_model(geo_model)
    print(f"  完成: {sol}")
    return sol


def _save_current_figure(path: Path, dpi: int = 150) -> None:
    """把当前 matplotlib 活动图保存到 path。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    print(f"  已保存: {path}")


def _get_pyvista_plotter(gempy_to_vista):
    """从 gempy_viewer 返回对象中取出底层 pyvista.Plotter。"""
    if hasattr(gempy_to_vista, "p") and gempy_to_vista.p is not None:
        return gempy_to_vista.p
    if hasattr(gempy_to_vista, "plotter") and gempy_to_vista.plotter is not None:
        return gempy_to_vista.plotter
    return None


def export_interactive_3d(
    geo_model: gp.data.GeoModel,
    *,
    open_browser: bool = True,
    open_desktop_window: bool = False,
    also_png: bool = True,
) -> Path | None:
    """
    导出可交互三维模型。

    主产物：model_3d.html
        - 用 PyVista + trame 把场景写成单文件 HTML
        - 用浏览器打开后可鼠标拖拽旋转、滚轮缩放、右键平移
        - 这才是「可交互 3D」，不是静态 PNG 截图

    可选：
        - model_3d.png：静态预览（方便快速扫一眼）
        - --interactive：再开一个桌面 PyVista 窗口（本机实时交互）
    """
    try:
        import pyvista as pv  # noqa: F401
    except ImportError:
        print("  提示: 未安装 pyvista，无法导出 3D。请执行：")
        print("    .venv-gempy/bin/pip install pyvista trame trame-vtk trame-vuetify nest_asyncio2")
        return None

    html_path = OUT_DIR / "model_3d.html"
    png_path = OUT_DIR / "model_3d.png"

    print("  构建 3D 场景 (gempy_viewer.plot_3d，含地形) ...")
    # show=False：不要立刻弹窗；off_screen=True：可在无显示器环境渲染/导出
    plot3d_kwargs = dict(
        show_data=True,
        show_lith=True,
        show_surfaces=True,
        show_topography=True,
        image=False,
        show=False,
        plotter_type="basic",
        kwargs_plotter={"off_screen": True},
    )
    try:
        gempy_vista = gpv.plot_3d(geo_model, **plot3d_kwargs)
    except TypeError:
        # 兼容旧版 viewer 无 show_topography 参数
        plot3d_kwargs.pop("show_topography", None)
        gempy_vista = gpv.plot_3d(geo_model, **plot3d_kwargs)

    plotter = _get_pyvista_plotter(gempy_vista)
    if plotter is None:
        print("  警告: 未能取得 PyVista Plotter，跳过交互 3D 导出")
        return None

    # ---- 交互 HTML（核心交付物）----
    try:
        plotter.export_html(str(html_path))
        print(f"  已保存可交互 3D: {html_path}")
        print("  用法: 用浏览器打开该 HTML，鼠标拖拽旋转 / 滚轮缩放")
    except Exception as exc:
        # 常见原因：未装 trame
        print(f"  警告: export_html 失败（{type(exc).__name__}: {exc}）")
        print("  请安装: .venv-gempy/bin/pip install trame trame-vtk trame-vuetify nest_asyncio2")
        html_path = None

    # ---- 静态 PNG 预览（次要）----
    if also_png:
        try:
            plotter.screenshot(str(png_path))
            print(f"  已保存静态预览: {png_path}")
        except Exception as exc:
            print(f"  警告: 截图失败（{type(exc).__name__}: {exc}）")

    # 关闭离屏 plotter，避免资源占用
    try:
        plotter.close()
    except Exception:
        pass

    # ---- 浏览器打开 HTML ----
    if open_browser and html_path is not None and html_path.is_file():
        url = html_path.resolve().as_uri()
        print(f"  正在打开浏览器: {url}")
        webbrowser.open(url)

    # ---- 桌面交互窗口（可选，阻塞直到关闭窗口）----
    if open_desktop_window:
        print("  打开桌面 PyVista 交互窗口（关闭窗口后脚本继续）...")
        try:
            gpv.plot_3d(
                geo_model,
                show_data=True,
                show_lith=True,
                show_surfaces=True,
                show_topography=True,
                image=False,
                show=True,
                plotter_type="basic",
                kwargs_plotter={"off_screen": False},
            )
        except TypeError:
            gpv.plot_3d(
                geo_model,
                show_data=True,
                show_lith=True,
                show_surfaces=True,
                image=False,
                show=True,
                plotter_type="basic",
                kwargs_plotter={"off_screen": False},
            )

    return html_path


def export_figures(
    geo_model: gp.data.GeoModel,
    *,
    open_browser: bool = True,
    open_desktop_window: bool = False,
) -> None:
    """导出 2D 剖面 + 地形地质图 + 可交互 3D HTML。"""
    print("[5/5] 导出可视化结果 ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 仅输入数据（Y 向：最能看出褶皱与断层两侧点位）----
    gpv.plot_2d(
        geo_model,
        show_data=True,
        show_lith=False,
        show_boundaries=False,
        direction="y",
        cell_number="mid",
    )
    _save_current_figure(OUT_DIR / "input_data_2d.png")

    # ---- Y 向中间剖面（X–Z：褶皱波状 + 断层错断）----
    gpv.plot_2d(
        geo_model,
        show_data=True,
        show_lith=True,
        show_boundaries=True,
        direction="y",
        cell_number="mid",
    )
    _save_current_figure(OUT_DIR / "section_y_mid.png")

    # ---- X 向中间剖面（Y–Z）----
    gpv.plot_2d(
        geo_model,
        show_data=True,
        show_lith=True,
        show_boundaries=True,
        direction="x",
        cell_number="mid",
    )
    _save_current_figure(OUT_DIR / "section_x_mid.png")

    # ---- 剖面叠加地形线（展示地表起伏）----
    try:
        gpv.plot_2d(
            geo_model,
            show_data=False,
            show_lith=True,
            show_boundaries=True,
            show_topography=True,
            direction="y",
            cell_number="mid",
        )
        _save_current_figure(OUT_DIR / "section_y_topo.png")
    except Exception as exc:
        print(f"  提示: 含地形剖面导出跳过（{type(exc).__name__}: {exc}）")

    # ---- 可交互 3D ----
    try:
        export_interactive_3d(
            geo_model,
            open_browser=open_browser,
            open_desktop_window=open_desktop_window,
            also_png=True,
        )
    except Exception as exc:  # 3D 失败不应阻断 2D 成功
        print(f"  警告: 3D 可视化失败（{type(exc).__name__}: {exc}），2D 结果仍可用")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(
        description="用合成 CSV 跑 GemPy 建模，并导出可交互 3D HTML",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="额外弹出桌面 PyVista 3D 窗口（可实时旋转；关闭窗口后脚本结束）",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="不自动用系统浏览器打开 model_3d.html",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 60)
    print("GemPy 合成数据三维建模")
    print(f"gempy 版本: {getattr(gp, '__version__', 'unknown')}")
    print("=" * 60)

    try:
        geo_model = build_model()
        add_topography(geo_model)
        compute(geo_model)
        export_figures(
            geo_model,
            open_browser=not args.no_browser,
            open_desktop_window=args.interactive,
        )
    except Exception as exc:
        print(f"\n建模失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    print()
    print("=" * 60)
    print("建模成功（褶皱 + 断层 + 地形）！输出目录:", OUT_DIR)
    print("可交互 3D 文件:", OUT_DIR / "model_3d.html")
    print("  → 浏览器打开后拖拽旋转 / 滚轮缩放")
    print("重点看: section_y_mid.png（褶皱波状 + 断层错断）")
    print("或加参数 --interactive 打开桌面 3D 窗口")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
