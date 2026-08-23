# -*- coding: utf-8 -*-
"""
apps.py — 应用扩展：虚拟钻孔 / 任意剖面 / 任意平切图
======================================================

本模块是流水线第四阶段（见系统技术路线与 CONTRACTS.md）。
它**只依赖** GemPy 建模输出的两个文件：

- ``lith_block.npy``：float (nx, ny, nz)，C 序的岩性体素解。
  axis0=X（西→东）、axis1=Y（南→北）、axis2=Z（下→上），
  每个体素的值 ≈ 岩性单元的整数 id（可能是浮点，需四舍五入）。
- ``lith_meta.json``：含 extent=[x0,x1,y0,y1,z0,z1]、resolution=[nx,ny,nz]、
  id_to_name、name_to_color（0-255 RGB）。

核心原理：
    三维隐式模型本质上是一个"体素网格"——把地下空间切成 nx*ny*nz 个
    小立方体，每个立方体记录它属于哪个岩性单元。有了这个网格，
    很多工程应用就变成简单的"数组切片/采样"问题：

    - 虚拟钻孔  = 固定 (x, y)，取出一整列 Z 方向的体素（lith[i, j, :]）；
    - 任意剖面  = 沿地表一条线段采样若干点，把每个点的整列拼起来；
    - 平切图    = 固定高程 z，取出一整层体素（lith[:, :, k]）。

    世界坐标 → 数组下标的换算：体素 i 的中心是
    x_i = x0 + (i + 0.5) * dx，其中 dx = (x1 - x0) / nx。
    反过来，坐标 x 落在第 floor((x - x0) / dx) 个体素里。
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无窗口后端：只写文件不弹窗，服务器/脚本环境必需

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

# 未知 id 的兜底颜色（灰色，0-1 归一化）
_GRAY = (0.6, 0.6, 0.6)


# ----------------------------------------------------------------------
# 数据读取
# ----------------------------------------------------------------------

def load_lith(model_dir):
    """读取模型目录下的岩性体素解与元数据。

    参数
    ----
    model_dir : str | Path
        含 lith_block.npy 与 lith_meta.json 的目录（通常是 model/）。

    返回
    ----
    (lith, meta)
        lith : int 型 ndarray，形状 (nx, ny, nz)。GemPy 输出的体素值
               可能是浮点（如 2.0000001），这里用四舍五入转成整数 id，
               后续所有查表（id → 名称/颜色）都要求整数键。
        meta : dict，lith_meta.json 的内容。
    """
    model_dir = Path(model_dir)
    lith_raw = np.load(model_dir / "lith_block.npy")
    # np.rint = 四舍五入到最近整数（仍是浮点），再 astype 转成整数类型
    lith = np.rint(lith_raw).astype(np.int32)
    with open(model_dir / "lith_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return lith, meta


# ----------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------

def _color_of(meta, lith_id):
    """内部工具：岩性 id → (单元名, 颜色)。

    查找链条：id → id_to_name 得到名称 → name_to_color 得到 RGB。
    JSON 的键只能是字符串，所以 id_to_name 的键形如 "2"，
    查表前必须把整数 id 转成 str。
    颜色从 0-255 归一化到 0-1（matplotlib 的颜色约定）。

    未知 id（不在 id_to_name 里，例如 GemPy 的背景值）返回
    ("Unknown", 灰色)；有名称但缺颜色时保留名称、用灰色兜底。
    """
    if int(lith_id) == AIR_ID:
        return "Air", (1.0, 1.0, 1.0)
    name = meta.get("id_to_name", {}).get(str(int(lith_id)))
    if name is None:
        return "Unknown", _GRAY
    rgb = meta.get("name_to_color", {}).get(name)
    if rgb is None:
        return name, _GRAY
    return name, tuple(float(c) / 255.0 for c in rgb[:3])


# 地表以上的"空气"哨兵 id（lith_meta.surface_z 提供地表高程场时启用）
AIR_ID = -9999


def _surface_grid(meta):
    """内部工具：取出地表高程场 (nx,ny) ndarray；没有则返回 None。"""
    sz = meta.get("surface_z")
    return None if sz is None else np.asarray(sz, dtype=float)


def _grid_info(meta):
    """内部工具：从 meta 解出网格几何（范围、体素数、体素尺寸）。"""
    x0, x1, y0, y1, z0, z1 = (float(v) for v in meta["extent"])
    nx, ny, nz = (int(v) for v in meta["resolution"])
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    dz = (z1 - z0) / nz
    return (x0, x1, y0, y1, z0, z1), (nx, ny, nz), (dx, dy, dz)


def _categorical(meta, id_array):
    """内部工具：把 id 数组变成可 imshow 的"类别索引图"。

    imshow 默认把数值当**连续量**上色，而岩性 id 是**类别**——
    id=4 并不比 id=2 "大两倍"。正确做法：
    1. 找出数组里实际出现的 id 并排序；
    2. 把每个 id 映射成 0,1,2,... 的连续索引；
    3. 用这些 id 的单元色构建 ListedColormap（离散色表），
       索引 0 用第 0 个颜色、索引 1 用第 1 个……一一对应。

    返回 (索引数组, 色表, 图例句柄列表)。
    """
    ids = np.unique(id_array)  # 已排序的、实际出现的 id
    # searchsorted：对每个元素查它在 ids 里的位置，即类别索引
    idx = np.searchsorted(ids, id_array)
    colors, handles = [], []
    for lid in ids:
        name, color = _color_of(meta, lid)
        colors.append(color)
        handles.append(Patch(facecolor=color, edgecolor="black", label=name))
    return idx, ListedColormap(colors), handles


def _save(fig, out_png):
    """内部工具：统一保存出图（dpi=150）并释放内存。"""
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)  # 及时关闭，否则批量出图时内存会持续增长
    print(f"已保存: {out_png}")


# ----------------------------------------------------------------------
# 应用一：虚拟钻孔
# ----------------------------------------------------------------------

def virtual_borehole(lith, meta, x, y, out_png, out_csv=None):
    """虚拟钻孔：在 (x, y) 处"打一口井"，输出自上而下的层段表与柱状图。

    原理：钻孔就是体素网格里的一根竖直列。先把世界坐标 (x, y)
    换算成体素下标 (i, j)，取出 lith[i, j, :]（长度 nz，自下而上），
    再从最上层往下扫描，把相邻的同 id 体素合并成"层段"。
    层段顶底取体素的上下边界（z0 + k*dz），厚度 = 体素数 * dz。

    参数
    ----
    lith, meta : load_lith 的返回值
    x, y       : 钻孔的世界坐标（米）
    out_png    : 柱状图输出路径
    out_csv    : 可选，层段表 CSV 输出路径

    返回
    ----
    list[dict]，自上而下的层段：
    [{"name": 单元名, "z_top": 顶高程, "z_bottom": 底高程, "thickness": 厚度}, ...]
    （z_top > z_bottom，单位米）

    异常
    ----
    ValueError : (x, y) 超出模型平面范围。
    """
    (x0, x1, y0, y1, z0, z1), (nx, ny, nz), (dx, dy, dz) = _grid_info(meta)

    if not (x0 <= x <= x1) or not (y0 <= y <= y1):
        raise ValueError(
            f"钻孔位置超出模型范围：(x={x}, y={y}) 不在 "
            f"[{x0}, {x1}] x [{y0}, {y1}] 内"
        )

    # 世界坐标 → 体素下标；min(..., n-1) 处理恰好落在最大边界上的情况
    i = min(int((x - x0) / dx), nx - 1)
    j = min(int((y - y0) / dy), ny - 1)
    column = lith[i, j, :]  # 一整列，下标 k 越大越靠上（Z 向上）

    # 钻孔从**地表**开钻，不是从模型盒顶：有地表高程场时裁掉地表以上体素
    surface = _surface_grid(meta)
    ground_z = None
    k_start = nz - 1
    if surface is not None:
        ground_z = float(surface[i, j])
        # 地表所在体素：中心高程 ≤ 地表的最高体素
        k_start = min(int((ground_z - z0) / dz), nz - 1)
        k_start = max(k_start, 0)

    # 自上而下扫描，合并相邻同 id 段
    segments = []   # 返回给调用者的层段表
    seg_ids = []    # 每段的 id（仅内部绘图取色用，不进返回值）
    k = k_start
    while k >= 0:
        cur_id = int(column[k])
        k_low = k
        # 只要下一个（更深的）体素还是同一个 id，就并入当前段
        while k_low - 1 >= 0 and int(column[k_low - 1]) == cur_id:
            k_low -= 1
        z_top = z0 + (k + 1) * dz      # 段顶 = 最上体素的上边界
        z_bottom = z0 + k_low * dz     # 段底 = 最下体素的下边界
        name, _ = _color_of(meta, cur_id)
        segments.append({
            "name": name,
            "z_top": z_top,
            "z_bottom": z_bottom,
            "thickness": z_top - z_bottom,
        })
        seg_ids.append(cur_id)
        k = k_low - 1  # 跳到下一段的最上体素

    # 首段顶界裁剪到真实地表高程（体素上边界可能略高于地表）
    if ground_z is not None and segments:
        segments[0]["z_top"] = min(segments[0]["z_top"], round(ground_z, 1))
        segments[0]["thickness"] = segments[0]["z_top"] - segments[0]["z_bottom"]

    # ---- 绘图：左=彩色柱状图，右=层段表格文字 ----
    fig, (ax_col, ax_tab) = plt.subplots(
        1, 2, figsize=(9, 10), width_ratios=[1.0, 1.3]
    )
    for seg, lid in zip(segments, seg_ids):
        _, color = _color_of(meta, lid)
        # Rectangle((左下角x, 左下角y), 宽, 高)
        ax_col.add_patch(Rectangle(
            (0.0, seg["z_bottom"]), 1.0, seg["thickness"],
            facecolor=color, edgecolor="black", linewidth=1.0,
        ))
        z_mid = (seg["z_top"] + seg["z_bottom"]) / 2.0
        ax_col.text(
            1.1, z_mid,
            f"{seg['name']}\n{seg['z_top']:.1f} / {seg['z_bottom']:.1f} m",
            va="center", fontsize=8,
        )
    pad = 0.02 * (z1 - z0)
    ax_col.set_xlim(-0.3, 3.0)
    ax_col.set_ylim(z0 - pad, z1 + pad)
    ax_col.set_xticks([])
    ax_col.set_ylabel("Elevation (m)")
    ax_col.set_title("Borehole Log")

    ax_tab.axis("off")
    lines = [f"{'No.':>3}  {'Unit':<12}{'Top(m)':>10}{'Bot(m)':>10}{'Thick(m)':>10}",
             "-" * 48]
    for n, seg in enumerate(segments, start=1):
        lines.append(
            f"{n:>3}  {seg['name']:<12}"
            f"{seg['z_top']:>10.1f}{seg['z_bottom']:>10.1f}{seg['thickness']:>10.1f}"
        )
    ax_tab.text(0.0, 0.98, "\n".join(lines),
                family="monospace", fontsize=9, va="top")
    ax_tab.set_title("Layer Table")

    fig.suptitle(f"Virtual Borehole at (x={x:.1f} m, y={y:.1f} m)")
    _save(fig, out_png)

    # ---- 可选：层段表 CSV ----
    if out_csv is not None:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["name", "z_top", "z_bottom", "thickness"])
            writer.writeheader()
            writer.writerows(segments)
        print(f"已保存: {out_csv}")

    return segments


# ----------------------------------------------------------------------
# 应用二：任意剖面
# ----------------------------------------------------------------------

def arbitrary_section(lith, meta, p1, p2, out_png, n_samples=500):
    """任意剖面：沿地表线段 p1→p2 竖直下切，输出剖面图。

    原理：在 p1、p2 之间取 n_samples 个等距采样点（含两端）。
    每个采样点相当于一口"微型钻孔"——用最近邻方式找到它所在的
    体素列 (i, j)，取整列 lith[i, j, :]。把所有列横向拼起来，
    就得到 (n_samples, nz) 的剖面 id 数组：
    行 = 沿线位置，列 = 从深到浅的 nz 个 z 层（体素中心高程）。

    参数
    ----
    p1, p2    : (x, y) 世界坐标，剖面线两端点
    out_png   : 剖面图输出路径
    n_samples : 沿线采样点数（越大剖面越平滑）

    返回
    ----
    (n_samples, nz) 的 int ndarray（岩性 id）。
    """
    (x0, x1, y0, y1, z0, z1), (nx, ny, nz), (dx, dy, dz) = _grid_info(meta)
    x_a, y_a = float(p1[0]), float(p1[1])
    x_b, y_b = float(p2[0]), float(p2[1])

    # 沿线等距采样点（linspace 含两端点）
    xs = np.linspace(x_a, x_b, n_samples)
    ys = np.linspace(y_a, y_b, n_samples)

    # 最近邻：坐标落在哪个体素就取哪个体素；floor 后 clip 防越界
    ii = np.clip(np.floor((xs - x0) / dx).astype(int), 0, nx - 1)
    jj = np.clip(np.floor((ys - y0) / dy).astype(int), 0, ny - 1)

    # numpy 花式索引：ii、jj 是等长下标数组，lith[ii, jj] 一次取出
    # 全部 n_samples 根列，结果形状 (n_samples, nz)
    section = lith[ii, jj].copy()

    # 地表以上标记为"空气"（白色留白），剖面才有真实的地形轮廓线
    surface = _surface_grid(meta)
    if surface is not None:
        ground = surface[ii, jj]                       # (n_samples,)
        z_centers = z0 + (np.arange(nz) + 0.5) * dz    # (nz,)
        section[z_centers[None, :] > ground[:, None]] = AIR_ID

    dist = float(np.hypot(x_b - x_a, y_b - y_a))  # 剖面线总长（米）

    # ---- 绘图 ----
    idx, cmap, handles = _categorical(meta, section)
    fig, ax = plt.subplots(figsize=(10, 5))
    # imshow 按"行=纵轴"画图，所以要转置让 z 成为纵轴；
    # origin='lower' 使 k=0（最深）画在最下方，即 z 向上。
    # extent 把像素坐标换成物理坐标：横轴 0~dist，纵轴 z0~z1。
    # vmin/vmax 设成 -0.5 ~ N-0.5，保证整数索引 0..N-1 恰好
    # 落在 N 个离散颜色的正中间。
    ax.imshow(
        idx.T, origin="lower", aspect="auto",
        extent=(0.0, dist, z0, z1),
        cmap=cmap, vmin=-0.5, vmax=len(cmap.colors) - 0.5,
        interpolation="nearest",
    )
    ax.set_xlabel("Distance along section (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(
        f"Section: ({x_a:.1f}, {y_a:.1f}) to ({x_b:.1f}, {y_b:.1f})")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=8, title="Units")
    _save(fig, out_png)

    return section


# ----------------------------------------------------------------------
# 应用三：任意平切图
# ----------------------------------------------------------------------

def level_slice(lith, meta, z, out_png):
    """任意平切图：给定高程 z，输出该高程的水平切片平面图。

    原理：找到离 z 最近的体素层 k，直接取 lith[:, :, k]，
    得到 (nx, ny) 的水平切片——相当于把模型在高程 z 处"锯开"
    俯视看到的地质图（可与地表地质图对比，或指导深部开采设计）。

    参数
    ----
    z       : 切片高程（米）；超出范围时取最近的顶/底层
    out_png : 平面图输出路径

    返回
    ----
    (nx, ny) 的 int ndarray（岩性 id），axis0=X、axis1=Y。
    """
    (x0, x1, y0, y1, z0, z1), (nx, ny, nz), (dx, dy, dz) = _grid_info(meta)

    # 高程 z 落在第 floor((z-z0)/dz) 层；clip 把范围外的 z 拉回边界层
    k = int(np.clip(np.floor((z - z0) / dz), 0, nz - 1))
    slab = lith[:, :, k].copy()  # (nx, ny)

    # 切片高程高于当地地表处 = 空气（露天），白色留白
    surface = _surface_grid(meta)
    if surface is not None:
        z_center = z0 + (k + 0.5) * dz
        slab[surface < z_center] = AIR_ID

    # ---- 绘图 ----
    idx, cmap, handles = _categorical(meta, slab)
    fig, ax = plt.subplots(figsize=(8, 7))
    # slab 是 (nx, ny)=(X, Y)，而 imshow 把 axis0 当行（纵轴），
    # 所以转置成 (ny, nx) 让 X 为横轴、Y 为纵轴；
    # origin='lower' 使 j=0（最南）在图底部 → 图上方为北。
    ax.imshow(
        idx.T, origin="lower",
        extent=(x0, x1, y0, y1),
        cmap=cmap, vmin=-0.5, vmax=len(cmap.colors) - 0.5,
        interpolation="nearest",
    )
    ax.set_xlabel("X - East (m)")
    ax.set_ylabel("Y - North (m)")
    ax.set_title(f"Level Slice at z = {z:.1f} m")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=8, title="Units")
    _save(fig, out_png)

    return slab


# ----------------------------------------------------------------------
# 便捷总入口
# ----------------------------------------------------------------------

def run_apps(model_dir, out_dir, boreholes=(), sections=(), slices=()):
    """一键跑完全部应用：批量生成钻孔、剖面、平切图。

    参数
    ----
    model_dir : 含 lith_block.npy / lith_meta.json 的目录
    out_dir   : 产物输出目录（自动创建）
    boreholes : [(x, y), ...] 钻孔位置列表
    sections  : [((x1, y1), (x2, y2)), ...] 剖面线端点列表
    slices    : [z, ...] 平切高程列表

    返回
    ----
    dict，产物路径清单：
    {"boreholes": [png...], "borehole_csvs": [csv...],
     "sections": [png...], "slices": [png...]}
    """
    lith, meta = load_lith(model_dir)
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    products = {"boreholes": [], "borehole_csvs": [], "sections": [], "slices": []}

    for n, (bx, by) in enumerate(boreholes, start=1):
        png = str(out_dir / f"borehole_{n}.png")
        csv_path = str(out_dir / f"borehole_{n}.csv")
        virtual_borehole(lith, meta, bx, by, png, out_csv=csv_path)
        products["boreholes"].append(png)
        products["borehole_csvs"].append(csv_path)

    for n, (p1, p2) in enumerate(sections, start=1):
        png = str(out_dir / f"section_{n}.png")
        arbitrary_section(lith, meta, p1, p2, png)
        products["sections"].append(png)

    for n, z in enumerate(slices, start=1):
        png = str(out_dir / f"slice_{n}.png")
        level_slice(lith, meta, z, png)
        products["slices"].append(png)

    return products
