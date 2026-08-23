# -*- coding: utf-8 -*-
"""
georef.py — 像素坐标 ↔ 世界坐标 仿射配准
==========================================

约定（见 CONTRACTS.md）：
- 图像坐标 (u, v)：u=列号（向右增大），v=行号（向下增大），原点在左上角
- 世界坐标 (X, Y)：X 向东，Y 向北（向上），单位米
- 图像上边缘对应世界 y_max，下边缘对应 y_min → v 轴需要翻转

原型阶段采用轴对齐仿射（无旋转）。真实图幅若有旋转/投影，
可在此扩展为完整仿射矩阵，接口不变。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeoRef:
    """像素 ↔ 世界 坐标转换器。

    参数
    ----
    img_w, img_h : 图像宽、高（像素）
    world        : (x_min, x_max, y_min, y_max) 图幅四角的世界坐标（米）
    """

    img_w: int
    img_h: int
    world: tuple[float, float, float, float]  # (x0, x1, y0, y1)

    # ---- 每像素代表的世界尺寸（米/像素） ----
    @property
    def px_size_x(self) -> float:
        x0, x1, _, _ = self.world
        return (x1 - x0) / self.img_w

    @property
    def px_size_y(self) -> float:
        _, _, y0, y1 = self.world
        return (y1 - y0) / self.img_h

    def px_to_world(self, u, v):
        """像素中心 → 世界坐标。u/v 可为标量或 numpy 数组。"""
        x0, x1, y0, y1 = self.world
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        x = x0 + (u + 0.5) * self.px_size_x
        # v 向下增大，世界 Y 向北增大 → 翻转
        y = y1 - (v + 0.5) * self.px_size_y
        return x, y

    def world_to_px(self, x, y):
        """世界坐标 → 像素（浮点；四舍五入后才是整数下标）。"""
        x0, x1, y0, y1 = self.world
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        u = (x - x0) / self.px_size_x - 0.5
        v = (y1 - y) / self.px_size_y - 0.5
        return u, v

    # ---- 序列化（写入 json 用） ----
    def to_dict(self) -> dict:
        return {
            "img_w": self.img_w,
            "img_h": self.img_h,
            "world": list(self.world),
        }

    @staticmethod
    def from_dict(d: dict) -> "GeoRef":
        return GeoRef(
            img_w=int(d["img_w"]),
            img_h=int(d["img_h"]),
            world=tuple(float(t) for t in d["world"]),
        )
