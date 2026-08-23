#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_surface_agreement.py — 真实图幅的"地表重现一致率"
======================================================

真实图幅没有三维真值，可用的闭环指标是：三维模型在地表处的出露单元
（lith_block 沿 surface_z 采样）与流水线提取的分区标签是否一致。
该指标衡量"模型是否忠实还原了它的输入地质图"（GemPy 文献中常用的
map fit 概念）。

用法：
    .venv-gempy/bin/python scripts/14_surface_agreement.py data/output/geo2model/real_map2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from geo2model.georef import GeoRef  # noqa: E402


def main(case_dir: str) -> None:
    case = Path(case_dir)
    if not case.is_absolute():
        case = PROJECT_ROOT / case

    lith = np.rint(np.load(case / "model" / "lith_block.npy")).astype(int)
    with open(case / "model" / "lith_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    labels = np.load(case / "extract" / "labels.npy")
    units = pd.read_csv(case / "db" / "units.csv")
    with open(case / "db" / "model_config.json", encoding="utf-8") as f:
        mc = json.load(f)
    georef = GeoRef.from_dict(mc["georef"])

    id2name_lith = {int(k): v for k, v in meta["id_to_name"].items()}
    uid2name = dict(zip(units["unit_id"].astype(int), units["name"]))
    uid2role = dict(zip(units["unit_id"].astype(int), units["role"]))

    surface_z = np.asarray(meta["surface_z"], dtype=float)  # (nx,ny)
    x0, x1, y0, y1, z0, z1 = (float(t) for t in meta["extent"])
    nx, ny, nz = (int(t) for t in meta["resolution"])
    dz = (z1 - z0) / nz

    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")

    # 模型地表出露单元：沿 surface_z 取最近体素
    kk = np.clip(((surface_z - z0) / dz - 0.5).round().astype(int), 0, nz - 1)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    surf_id = lith[ii, jj, kk]
    surf_name = np.vectorize(lambda t: id2name_lith.get(int(t), "?"))(surf_id)

    # 提取分区在同一网格点的单元名（水体/忽略簇不参与统计）
    u, v = georef.world_to_px(Xg.ravel(), Yg.ravel())
    ui = np.clip(np.rint(u).astype(int), 0, labels.shape[1] - 1)
    vi = np.clip(np.rint(v).astype(int), 0, labels.shape[0] - 1)
    ext_uid = labels[vi, ui].reshape(nx, ny)
    ext_name = np.vectorize(lambda t: uid2name.get(int(t), "?"))(ext_uid)
    ext_role = np.vectorize(lambda t: uid2role.get(int(t), "?"))(ext_uid)

    mask = ext_role == "strata"
    agree = (surf_name == ext_name) & mask
    rate = agree.sum() / mask.sum()
    print(f"案例: {case.name}")
    print(f"地表重现一致率: {rate:.4f}  (参与统计网格点 {int(mask.sum())})")
    # 分单元
    for name in sorted(set(ext_name[mask].tolist())):
        m = mask & (ext_name == name)
        print(f"  {name:6s}: {(surf_name[m] == name).mean():.3f}  (n={int(m.sum())})")

    out = {"surface_agreement": float(rate),
           "per_unit": {name: float((surf_name[mask & (ext_name == name)] == name).mean())
                        for name in sorted(set(ext_name[mask].tolist()))}}
    with open(case / "eval" / "surface_agreement.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/output/geo2model/real_map2")
