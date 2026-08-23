# -*- coding: utf-8 -*-
"""
sectionreg.py — 剖面图深部约束配准（像素比例映射）
====================================================

背景
----
地质图幅常附有剖面图：沿主图上一条剖面线 A–B 绘制的垂直切面图。
要把剖面图上人工点选的深部地质界线点转成三维建模约束
（GemPy surface_points），需要建立「剖面图像素 (u,v) ↔ 全局三维
世界坐标 (X,Y,Z)」的映射。

算法原理（基于像素比例映射的空间配准）
--------------------------------------
剖面图的有效绘图区由图像四周边距界定：
    水平方向 u ∈ [margin_left, img_w - margin_right]
    垂直方向 v ∈ [margin_top,  img_h - margin_bottom]

1. 水平配准：绘图区左边缘对应剖面线起点 A，右边缘对应终点 B。
   将 u 按比例线性映射为沿线距离 s ∈ [0, L]（L 为 A→B 世界长度），
   再在 A→B 连线上线性插值得平面坐标 (X, Y)：
       t = (u - margin_left) / 绘图区宽
       (X, Y) = A + t * (B - A)

2. 垂直配准：绘图区上边缘对应高程轴最大值 z_max，下边缘对应
   z_min。注意图像 v 轴向下增大而世界 Z 向上增大，故映射需翻转：
       r = (v - margin_top) / 绘图区高
       Z = z_max - r * (z_max - z_min)

3. 逆映射（世界 → 像素）：把点 P=(X,Y) 向剖面线方向单位向量
   点乘投影求沿线距离 s，再按比例反算 u；Z 按高程轴比例反算 v。
   （剖面外的点也可投影，是否位于绘图区内由 inside() 判定。）

坐标约定见 CONTRACTS.md：像素 (u,v) 原点在左上角，u 向右、v 向下；
世界 (X,Y,Z)：X 向东、Y 向北、Z 向上，单位米。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SectionFrame:
    """剖面图配准框架：像素 (u,v) ↔ 世界 (X,Y,Z) 转换器。

    参数
    ----
    p1_world, p2_world : 剖面线起点 A / 终点 B 在主图上的世界坐标 (x, y)
    img_w, img_h       : 剖面图图像宽、高（像素）
    margin_left/right  : 绘图区左、右边距（像素）
    margin_top/bottom  : 绘图区上、下边距（像素）
    z_min, z_max       : 绘图区下、上边缘对应的世界高程（米）
    """

    p1_world: tuple[float, float]
    p2_world: tuple[float, float]
    img_w: int
    img_h: int
    margin_left: float
    margin_right: float
    margin_top: float
    margin_bottom: float
    z_min: float
    z_max: float

    def __post_init__(self):
        if self.plot_w <= 0 or self.plot_h <= 0:
            raise ValueError("边距过大：绘图区宽/高必须为正")
        if self.length <= 0:
            raise ValueError("剖面线两端点重合：世界长度必须为正")
        if self.z_max <= self.z_min:
            raise ValueError("高程轴范围非法：要求 z_max > z_min")

    # ---- 基本几何量 ----
    @property
    def length(self) -> float:
        """剖面线 A→B 的世界长度 L（米）。"""
        dx = self.p2_world[0] - self.p1_world[0]
        dy = self.p2_world[1] - self.p1_world[1]
        return float(np.hypot(dx, dy))

    @property
    def plot_w(self) -> float:
        """绘图区宽（像素）。"""
        return self.img_w - self.margin_left - self.margin_right

    @property
    def plot_h(self) -> float:
        """绘图区高（像素）。"""
        return self.img_h - self.margin_top - self.margin_bottom

    # ---- 像素 → 世界 ----
    def px_to_world(self, u, v):
        """剖面图像素 (u, v) → 全局三维坐标 (x, y, z)。

        支持标量与 numpy 数组（可广播）。绘图区外的输入按同一线性
        关系外插（不截断），是否在区内请用 inside() 判断。
        """
        u_arr = np.asarray(u, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        scalar = u_arr.ndim == 0 and v_arr.ndim == 0

        # 水平：u → 归一化比例 t → 在 A→B 连线上线性插值
        t = (u_arr - self.margin_left) / self.plot_w
        x1, y1 = self.p1_world
        x2, y2 = self.p2_world
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        # 垂直：v 向下增大、Z 向上增大 → 翻转比例映射
        r = (v_arr - self.margin_top) / self.plot_h
        z = self.z_max - r * (self.z_max - self.z_min)

        if scalar:
            return float(x), float(y), float(z)
        return x, y, z

    # ---- 世界 → 像素 ----
    def world_to_px(self, x, y, z):
        """全局三维坐标 (x, y, z) → 剖面图像素 (u, v)。

        (x, y) 通过点乘投影到剖面线方向求沿线距离 s，再按
        s/L 比例反算 u；z 按高程轴比例反算 v。支持标量与数组。
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        z_arr = np.asarray(z, dtype=float)
        scalar = x_arr.ndim == 0 and y_arr.ndim == 0 and z_arr.ndim == 0

        x1, y1 = self.p1_world
        x2, y2 = self.p2_world
        L = self.length
        # 沿线距离 = (P - A) · (B - A) / |B - A| （标量投影）
        s = ((x_arr - x1) * (x2 - x1) + (y_arr - y1) * (y2 - y1)) / L
        u = self.margin_left + (s / L) * self.plot_w
        v = self.margin_top + (self.z_max - z_arr) / (self.z_max - self.z_min) * self.plot_h

        if scalar:
            return float(u), float(v)
        return u, v

    def inside(self, u, v):
        """判断像素 (u, v) 是否落在绘图区内（含边界）。

        标量输入返回 bool；数组输入返回同形状布尔数组。
        """
        u_arr = np.asarray(u, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        ok = (
            (u_arr >= self.margin_left)
            & (u_arr <= self.img_w - self.margin_right)
            & (v_arr >= self.margin_top)
            & (v_arr <= self.img_h - self.margin_bottom)
        )
        if ok.ndim == 0:
            return bool(ok)
        return ok

    # ---- JSON 序列化 ----
    def to_dict(self) -> dict:
        return {
            "p1_world": list(self.p1_world),
            "p2_world": list(self.p2_world),
            "img_w": self.img_w,
            "img_h": self.img_h,
            "margin_left": self.margin_left,
            "margin_right": self.margin_right,
            "margin_top": self.margin_top,
            "margin_bottom": self.margin_bottom,
            "z_min": self.z_min,
            "z_max": self.z_max,
        }

    @staticmethod
    def from_dict(d: dict) -> "SectionFrame":
        return SectionFrame(
            p1_world=tuple(float(t) for t in d["p1_world"]),
            p2_world=tuple(float(t) for t in d["p2_world"]),
            img_w=int(d["img_w"]),
            img_h=int(d["img_h"]),
            margin_left=float(d["margin_left"]),
            margin_right=float(d["margin_right"]),
            margin_top=float(d["margin_top"]),
            margin_bottom=float(d["margin_bottom"]),
            z_min=float(d["z_min"]),
            z_max=float(d["z_max"]),
        )


def picks_to_surface_points(picks, frame: SectionFrame) -> pd.DataFrame:
    """人工点选的剖面界线点 → GemPy surface_points DataFrame。

    参数
    ----
    picks : [{"u": ..., "v": ..., "formation": "K1"}, ...]
            人工在剖面图上点选的地质界线点（像素坐标 + 所属地层）
    frame : SectionFrame 配准框架

    返回
    ----
    列为 X, Y, Z, formation 的 DataFrame（见 CONTRACTS.md GemPy 契约）。
    落在绘图区外的点选打印中文警告并跳过（点选误差通常应在区内，
    区外多为误操作）。
    """
    rows = []
    for p in picks:
        u, v, formation = p["u"], p["v"], p["formation"]
        if not frame.inside(u, v):
            print(f"警告：点选 (u={u:.1f}, v={v:.1f}, formation={formation}) "
                  f"位于剖面图绘图区外，已跳过")
            continue
        x, y, z = frame.px_to_world(u, v)
        rows.append({"X": x, "Y": y, "Z": z, "formation": formation})
    return pd.DataFrame(rows, columns=["X", "Y", "Z", "formation"])


def picks_csv_to_surface_points(picks_csv, frame_json, out_csv=None) -> pd.DataFrame:
    """文件版便捷入口：点选 CSV + 配准 JSON → surface_points DataFrame。

    参数
    ----
    picks_csv  : CSV 文件路径，列为 u, v, formation
    frame_json : SectionFrame.to_dict() 保存的 JSON 文件路径
    out_csv    : 可选；给出则把结果写为 CSV（列 X,Y,Z,formation，无索引）
    """
    with open(frame_json, "r", encoding="utf-8") as f:
        frame = SectionFrame.from_dict(json.load(f))
    picks = pd.read_csv(picks_csv).to_dict("records")
    df = picks_to_surface_points(picks, frame)
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
    return df


def draw_frame_overlay(section_img_path, frame: SectionFrame, out_png):
    """在剖面图上叠加配准框架，供人工校对 frame 参数。

    叠加内容：
    - 红色矩形：绘图区边框（由四周边距界定）
    - A / B 端点标注（附世界坐标）
    - 若干高程刻度横线及数值标签

    用户核对边框是否与剖面图坐标轴重合、高程刻度是否与图上
    标注一致，即可确认配准参数正确。
    """
    img = plt.imread(section_img_path)

    fig, ax = plt.subplots(figsize=(frame.img_w / 100.0, frame.img_h / 100.0), dpi=100)
    ax.imshow(img)

    # 绘图区边框
    u0 = frame.margin_left
    u1 = frame.img_w - frame.margin_right
    v0 = frame.margin_top
    v1 = frame.img_h - frame.margin_bottom
    ax.plot([u0, u1, u1, u0, u0], [v0, v0, v1, v1, v0],
            color="red", lw=1.5, label="绘图区边框")

    # A / B 端点标注（绘图区左右边缘）
    ax.annotate(f"A ({frame.p1_world[0]:.0f}, {frame.p1_world[1]:.0f})",
                xy=(u0, v1), xytext=(u0, v1 + 0.5 * frame.margin_bottom),
                color="blue", fontsize=9, ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="blue"))
    ax.annotate(f"B ({frame.p2_world[0]:.0f}, {frame.p2_world[1]:.0f})",
                xy=(u1, v1), xytext=(u1, v1 + 0.5 * frame.margin_bottom),
                color="blue", fontsize=9, ha="right", va="top",
                arrowprops=dict(arrowstyle="->", color="blue"))

    # 高程刻度横线（含上下边缘，等分 5 段）
    for z in np.linspace(frame.z_min, frame.z_max, 6):
        _, v = frame.world_to_px(frame.p1_world[0], frame.p1_world[1], z)
        ax.plot([u0, u1], [v, v], color="green", lw=0.8, ls="--", alpha=0.7)
        ax.text(u0 - 4, v, f"{z:.0f}", color="green", fontsize=8,
                ha="right", va="center")

    ax.set_xlim(0, frame.img_w)
    ax.set_ylim(frame.img_h, 0)  # 图像坐标 v 向下
    ax.set_title("SectionFrame overlay check")
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)
