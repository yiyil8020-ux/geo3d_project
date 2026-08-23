# -*- coding: utf-8 -*-
"""
model3d.py — GemPy 隐式三维建模封装（数据库 → 模型 → 多格式导出）
==================================================================

输入：db/ 目录（surface_points.csv + orientations.csv + model_config.json + units.csv）
输出：model/ 目录
    - lith_block.npy + lith_meta.json   岩性体素解（应用模块 apps 的输入）
    - section_y_mid.png / section_x_mid.png   正交剖面
    - geomap_rebuilt.png                模型重建的地表地质图（可与输入图对比验证）
    - model_3d.html / model_3d.png      可交互 3D 与静态预览
    - export/model_lith.vti             体素模型（ParaView 可开）
    - export/model_surfaces.vtm         层面网格（VTK MultiBlock）
    - export/model.obj / model.gltf     通用三维格式

GemPy 2026 API 要点（经实测验证）：
- lith_block 值按「结构框架元素顺序」从 1 开始编号（断层元素占号但
  不出现在岩性体中），基底 = 元素数 + 1
- resolution=[nx,ny,nz] 的 dense 网格 → lith_block.reshape(nx,ny,nz)，
  axis2 为 Z 自下而上
- 挂接 Topography 后 DENSE 与 TOPOGRAPHY 网格默认同时激活，
  raw_arrays.geological_map 即地表地质图（nx*ny）
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from .georef import GeoRef


@dataclass
class ModelParams:
    """建模与导出参数。"""

    also_html: bool = True       # 导出可交互 3D HTML（略慢）
    also_mesh_exports: bool = True   # obj/gltf/vtm 导出
    dpi: int = 150
    # 克里金核参数（在 GemPy 内部归一化坐标系中）。
    # range 控制协方差衰减距离：默认 1.7 时深部外推塌缩严重；
    # 经参数扫描（见技术报告）在只有图面+剖面约束的场景下 range≈10 最优
    kriging_range: float = 10.0
    kriging_c_o: float = 10.0
    drift_degree: int = 1        # 趋势项阶数（1=线性）


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def build_model(
    db_dir: str | Path,
    model_dir: str | Path,
    dem_px: np.ndarray | None = None,
    params: ModelParams | None = None,
) -> dict:
    """从数据库目录构建 GemPy 模型并导出全部产物。

    dem_px: 与输入图同尺寸的逐像素 DEM（extract/dem.npy）；
            None 表示不挂地形（模型为满盒）。
    """
    import gempy as gp

    params = params or ModelParams()
    db_dir, model_dir = Path(db_dir), Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "export").mkdir(exist_ok=True)

    with open(db_dir / "model_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    units = pd.read_csv(db_dir / "units.csv")
    extent = cfg["extent"]
    resolution = cfg["resolution"]
    georef = GeoRef.from_dict(cfg["georef"])

    # ---- 1. 建模型骨架：读 CSV + 层序映射 + 断层声明 ----
    print("[model3d] 创建 GeoModel ...")
    geo_model = gp.create_geomodel(
        project_name="geo2model",
        extent=extent,
        resolution=resolution,
        importer_helper=gp.data.ImporterHelper(
            path_to_surface_points=str(db_dir / "surface_points.csv"),
            path_to_orientations=str(db_dir / "orientations.csv"),
            coord_x_name="X",
            coord_y_name="Y",
            coord_z_name="Z",
            surface_name="formation",
        ),
    )

    # series 键顺序 = 新→老（断层组在前）
    mapping = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg["series"].items()}
    # GemPy 对单元素序列也接受字符串
    mapping = {k: (v[0] if isinstance(v, tuple) and len(v) == 1 else v) for k, v in mapping.items()}
    gp.map_stack_to_surfaces(gempy_model=geo_model, mapping_object=mapping)
    if cfg.get("fault_series"):
        gp.set_is_fault(frame=geo_model, fault_groups=list(cfg["fault_series"]))

    # 克里金核参数（对深部外推形态影响极大，见 ModelParams 注释）
    ko = geo_model.interpolation_options.kernel_options
    ko.range = float(params.kriging_range)
    ko.c_o = float(params.kriging_c_o)
    ko.uni_degree = int(params.drift_degree)
    print(f"[model3d] 插值参数: range={ko.range}, c_o={ko.c_o}, "
          f"drift={ko.uni_degree}")

    # ---- 2. 单元颜色与地图一致（提高对比可读性）----
    name2color = {
        str(r["name"]): (int(r["color_r"]), int(r["color_g"]), int(r["color_b"]))
        for _, r in units.iterrows()
        if pd.notna(r.get("color_r"))
    }
    elements = [el for g in geo_model.structural_frame.structural_groups for el in g.elements]
    for el in elements:
        if el.name in name2color:
            r, g_, b = name2color[el.name]
            el.color = f"#{r:02x}{g_:02x}{b:02x}"

    # ---- 3. 地形 ----
    if dem_px is not None:
        from gempy.core.data.grid_modules.topography import Topography

        nx, ny = int(resolution[0]), int(resolution[1])
        xs = np.linspace(extent[0], extent[1], nx)
        ys = np.linspace(extent[2], extent[3], ny)
        Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
        # 世界 → 像素 → 双线性采样 DEM
        u, v = georef.world_to_px(Xg.ravel(), Yg.ravel())
        Zg = map_coordinates(
            dem_px.astype(float),
            [np.clip(v, 0, dem_px.shape[0] - 1), np.clip(u, 0, dem_px.shape[1] - 1)],
            order=1, mode="nearest",
        ).reshape(nx, ny)
        Zg = np.clip(Zg, extent[4] + 1.0, extent[5] - 1.0)
        geo_model.grid.topography = Topography(
            _regular_grid=geo_model.grid.regular_grid,
            values_2d=np.stack([Xg, Yg, Zg], axis=-1),
        )
        print(f"[model3d] 地形已挂接: {Zg.min():.0f}–{Zg.max():.0f} m")

    # ---- 4. 求解 ----
    print("[model3d] 计算隐式模型 (gp.compute_model) ...")
    sol = gp.compute_model(geo_model)

    # ---- 5. 岩性体素解 + id→名称映射 ----
    nx, ny, nz = (int(t) for t in resolution)
    lith = np.round(np.asarray(sol.raw_arrays.lith_block)).astype(np.int32)
    lith = lith.reshape(nx, ny, nz)

    # lith id 按元素顺序 1..n；断层元素占号但不出现；基底 = n+1
    id_to_name = {}
    fault_names = set()
    for grp in geo_model.structural_frame.structural_groups:
        is_fault_grp = getattr(grp, "is_fault", False)
        for el in grp.elements:
            if is_fault_grp:
                fault_names.add(el.name)
    for i, el in enumerate(elements):
        id_to_name[str(i + 1)] = el.name
    id_to_name[str(len(elements) + 1)] = cfg.get("basement_name", "basement")

    # 只保留 lith 中真实出现的 id（把断层占号剔除干净）
    present = {str(t) for t in np.unique(lith)}
    id_to_name = {k: v for k, v in id_to_name.items() if k in present or v not in fault_names}

    lith_meta = {
        "extent": extent,
        "resolution": resolution,
        "id_to_name": id_to_name,
        "name_to_color": {k: list(v) for k, v in name2color.items()},
        # 地表高程场 (nx,ny)：应用层用它把钻孔/剖面/平切裁剪到地表以下
        "surface_z": (np.asarray(Zg).round(2).tolist() if dem_px is not None else None),
    }
    np.save(model_dir / "lith_block.npy", lith)
    with open(model_dir / "lith_meta.json", "w", encoding="utf-8") as f:
        json.dump(lith_meta, f, ensure_ascii=False, indent=1)

    # ---- 6. 重建地表地质图（与输入图对照的直观验证）----
    geomap_path = None
    if dem_px is not None and sol.raw_arrays.geological_map is not None:
        gm = np.round(np.asarray(sol.raw_arrays.geological_map)).astype(int)
        gm = gm.reshape(nx, ny)  # [ix, iy]
        img = _ids_to_rgb(gm, id_to_name, name2color)
        # [ix,iy] → 图像行=北在上：转置后按行倒序
        img = np.transpose(img, (1, 0, 2))[::-1, :, :]
        geomap_path = model_dir / "geomap_rebuilt.png"
        plt.figure(figsize=(7, 7))
        plt.imshow(img, extent=[extent[0], extent[1], extent[2], extent[3]])
        plt.title("Rebuilt geological map (from 3D model + DEM)")
        plt.xlabel("X (m)"); plt.ylabel("Y (m)")
        plt.tight_layout()
        plt.savefig(geomap_path, dpi=params.dpi)
        plt.close()
        print(f"  已保存重建地质图: {geomap_path}")

    # ---- 7. 剖面图 ----
    import gempy_viewer as gpv

    for direction, fname in (("y", "section_y_mid.png"), ("x", "section_x_mid.png")):
        try:
            gpv.plot_2d(
                geo_model, show_data=True, show_lith=True,
                show_boundaries=True, direction=direction, cell_number="mid",
            )
            fig = plt.gcf()
            fig.savefig(model_dir / fname, dpi=params.dpi, bbox_inches="tight")
            plt.close("all")
            print(f"  已保存剖面: {model_dir / fname}")
        except Exception as exc:
            print(f"  警告: 剖面 {direction} 绘制失败 ({type(exc).__name__}: {exc})")

    # ---- 8. 3D 导出（HTML/OBJ/glTF/VTK）----
    html_path = None
    if params.also_html or params.also_mesh_exports:
        html_path = _export_3d(geo_model, model_dir, lith, lith_meta, params, db_dir=Path(db_dir))

    print("[model3d] 完成")
    return {
        "lith_block": str(model_dir / "lith_block.npy"),
        "lith_meta": str(model_dir / "lith_meta.json"),
        "html": str(html_path) if html_path else None,
        "geomap_rebuilt": str(geomap_path) if geomap_path else None,
    }


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _ids_to_rgb(ids: np.ndarray, id_to_name: dict, name2color: dict) -> np.ndarray:
    """id 数组 → RGB 图（未知单元灰色）。"""
    out = np.full(ids.shape + (3,), 160, dtype=np.uint8)
    for k, name in id_to_name.items():
        color = name2color.get(name)
        if color is None:
            continue
        out[ids == int(k)] = color
    return out


def _export_3d_in_subprocess(db_dir: Path, model_dir: Path, params: ModelParams) -> Path | None:
    """在独立主进程中执行 3D HTML 导出，彻底解决 aiohttp/trame 在 Gradio 子线程注册信号的限制。"""
    import subprocess
    import sys
    html_path = model_dir / "model_3d.html"
    code = f"""
import gempy as gp
import gempy_viewer as gpv
import pyvista as pv
from pathlib import Path
import json

db_dir = Path({repr(str(db_dir))})
model_dir = Path({repr(str(model_dir))})
with open(db_dir / 'model_config.json') as f:
    cfg = json.load(f)

geo_model = gp.create_geomodel(
    project_name='geo2model',
    extent=cfg['extent'],
    resolution=cfg['resolution'],
    importer_helper=gp.data.ImporterHelper(
        path_to_surface_points=str(db_dir / 'surface_points.csv'),
        path_to_orientations=str(db_dir / 'orientations.csv'),
        coord_x_name='X', coord_y_name='Y', coord_z_name='Z', surface_name='formation',
    ),
)
mapping = {{k: tuple(v) if isinstance(v, list) else v for k, v in cfg['series'].items()}}
mapping = {{k: (v[0] if isinstance(v, tuple) and len(v) == 1 else v) for k, v in mapping.items()}}
gp.map_stack_to_surfaces(gempy_model=geo_model, mapping_object=mapping)

sol = gp.compute_model(geo_model)
pv.global_theme.allow_empty_mesh = True
try:
    gempy_vista = gpv.plot_3d(
        geo_model, show_data=True, show_lith=True, show_surfaces=True,
        show_topography=True, image=False, show=False,
        plotter_type='basic', kwargs_plotter={{'off_screen': True}},
    )
except Exception:
    gempy_vista = gpv.plot_3d(
        geo_model, show_data=True, show_lith=True, show_surfaces=True,
        image=False, show=False, plotter_type='basic',
        kwargs_plotter={{'off_screen': True}},
    )
plotter = getattr(gempy_vista, 'p', None) or getattr(gempy_vista, 'plotter', None)
if plotter is not None:
    plotter.export_html(str(model_dir / 'model_3d.html'))
    try:
        plotter.screenshot(str(model_dir / 'model_3d.png'))
    except Exception:
        pass
"""
    try:
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        if html_path.is_file():
            print(f"  [主进程导出] 已保存可交互 3D: {html_path}")
            return html_path
    except Exception as exc:
        print(f"  [主进程导出] 警告: {exc}")
    return None


def _export_3d(geo_model, model_dir: Path, lith, lith_meta, params: ModelParams, db_dir: Path | None = None):
    """PyVista 三维场景：HTML + 截图 + OBJ/glTF/VTK。失败不阻断主流程。"""
    import threading
    in_subthread = threading.current_thread() != threading.main_thread()
    if in_subthread and db_dir is not None:
        sub_res = _export_3d_in_subprocess(db_dir, model_dir, params)
        if sub_res is not None:
            return sub_res

    try:
        import gempy_viewer as gpv
        import pyvista as pv
    except ImportError:
        print("  提示: 未安装 pyvista/gempy_viewer，跳过 3D 导出")
        return None

    # 个别界面可能在模型盒内完全没有几何（如某层被推到盒外/完全隐伏），
    # 生成的空网格默认会让 PyVista 抛异常——允许空网格，3D 导出不因此中断
    pv.global_theme.allow_empty_mesh = True

    html_path = model_dir / "model_3d.html"
    try:
        try:
            gempy_vista = gpv.plot_3d(
                geo_model, show_data=True, show_lith=True, show_surfaces=True,
                show_topography=True, image=False, show=False,
                plotter_type="basic", kwargs_plotter={"off_screen": True},
            )
        except TypeError:
            gempy_vista = gpv.plot_3d(
                geo_model, show_data=True, show_lith=True, show_surfaces=True,
                image=False, show=False, plotter_type="basic",
                kwargs_plotter={"off_screen": True},
            )
    except Exception as exc:
        print(f"  警告: 3D 场景构建失败（{type(exc).__name__}: {exc}），跳过 3D 导出")
        if db_dir is not None:
            return _export_3d_in_subprocess(db_dir, model_dir, params)
        return None
    plotter = getattr(gempy_vista, "p", None) or getattr(gempy_vista, "plotter", None)
    if plotter is None:
        print("  警告: 未取得 PyVista Plotter，跳过 3D 导出")
        if db_dir is not None:
            return _export_3d_in_subprocess(db_dir, model_dir, params)
        return None

    if params.also_html:
        try:
            plotter.export_html(str(html_path))
            print(f"  已保存可交互 3D: {html_path}")
        except Exception as exc:
            print(f"  警告: export_html 失败 ({exc})，尝试子进程导出...")
            if db_dir is not None:
                html_path = _export_3d_in_subprocess(db_dir, model_dir, params)
            else:
                html_path = None
        try:
            plotter.screenshot(str(model_dir / "model_3d.png"))
        except Exception:
            pass

    if params.also_mesh_exports:
        exp = model_dir / "export"
        # 通用网格格式
        for fn, call in (
            ("model.obj", lambda p: plotter.export_obj(str(exp / "model.obj"))),
            ("model.gltf", lambda p: plotter.export_gltf(str(exp / "model.gltf"))),
        ):
            try:
                call(plotter)
                print(f"  已导出: {exp / fn}")
            except Exception as exc:
                print(f"  警告: {fn} 导出失败 ({type(exc).__name__})")
        # 体素 VTK（ParaView 直接体绘制/切片）
        try:
            ext = lith_meta["extent"]
            res = lith_meta["resolution"]
            grid = pv.ImageData(
                dimensions=(res[0] + 1, res[1] + 1, res[2] + 1),
                spacing=(
                    (ext[1] - ext[0]) / res[0],
                    (ext[3] - ext[2]) / res[1],
                    (ext[5] - ext[4]) / res[2],
                ),
                origin=(ext[0], ext[2], ext[4]),
            )
            grid.cell_data["lith_id"] = np.asarray(lith).ravel(order="F")
            grid.save(str(exp / "model_lith.vti"))
            print(f"  已导出: {exp / 'model_lith.vti'}")
        except Exception as exc:
            print(f"  警告: vti 导出失败 ({type(exc).__name__}: {exc})")
        # 层面网格 MultiBlock
        try:
            mb = pv.MultiBlock()
            meshes = getattr(plotter, "meshes", None) or []
            for i, m in enumerate(meshes):
                if isinstance(m, pv.PolyData):
                    mb.append(m, f"mesh_{i}")
            if len(mb) > 0:
                mb.save(str(exp / "model_surfaces.vtm"))
                print(f"  已导出: {exp / 'model_surfaces.vtm'}")
        except Exception as exc:
            print(f"  警告: vtm 导出失败 ({type(exc).__name__})")

    try:
        plotter.close()
    except Exception:
        pass
    return html_path
