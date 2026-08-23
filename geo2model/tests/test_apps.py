# -*- coding: utf-8 -*-
"""
test_apps.py — apps 模块自测（不依赖 pytest，直接运行）
========================================================

    /path/to/.venv-gempy/bin/python geo2model/tests/test_apps.py

思路：不跑 GemPy，而是**手工构造**一个解析可控的假体素模型
（我们精确知道每个体素的岩性 id），写成契约规定的
lith_block.npy + lith_meta.json，再驱动三个应用逐一验证。

假模型设计（extent = [0,400, 0,400, -300,0]，40x40x30 体素，
体素尺寸 dx=dy=dz=10 米）：
- 三个水平地层，按体素中心高程 zc 划分：
    zc > -100        → id=2 (K1)
    -200 < zc <= -100 → id=3 (P3)
    zc <= -200       → id=4 (P1)
- 一个倾斜板状体（岩脉）id=1：满足
    |zc - (0.6*xc - 320)| <= 15 且 xc >= 200
  即一块沿 X 方向下倾的斜面块，只出现在模型东半部。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# 把项目根目录（geo2model 的上一级）加入 sys.path，
# 这样无论从哪个目录运行本脚本都能 import geo2model 包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geo2model.apps import (  # noqa: E402  （import 必须在改 sys.path 之后）
    arbitrary_section,
    level_slice,
    load_lith,
    run_apps,
    virtual_borehole,
)

# ---- 假模型的几何常量（与文件头注释一致） ----
EXTENT = [0.0, 400.0, 0.0, 400.0, -300.0, 0.0]
NX, NY, NZ = 40, 40, 30
DX = DY = DZ = 10.0  # 一个体素的边长（米）


def make_fake_model(model_dir: Path) -> np.ndarray:
    """在 model_dir 下生成契约格式的假数据，返回真值 int 数组。"""
    x0, x1, y0, y1, z0, z1 = EXTENT
    # 三个方向的体素中心坐标（契约：x_i = x0 + (i+0.5)*dx）
    xc = x0 + (np.arange(NX) + 0.5) * DX
    yc = y0 + (np.arange(NY) + 0.5) * DY
    zc = z0 + (np.arange(NZ) + 0.5) * DZ
    # meshgrid(indexing="ij")：生成三个 (NX,NY,NZ) 数组，
    # X[i,j,k]=xc[i]、Z[i,j,k]=zc[k]，便于按坐标批量赋 id
    X, _Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")

    # 三个水平地层
    truth = np.where(Z > -100.0, 2, np.where(Z > -200.0, 3, 4))
    # 斜面块（岩脉）覆盖在上面
    dike = (np.abs(Z - (0.6 * X - 320.0)) <= 15.0) & (X >= 200.0)
    truth[dike] = 1
    truth = truth.astype(np.int64)

    # 契约允许体素值为浮点（≈整数 id）：故意加 0.2 偏移，
    # 验证 load_lith 的四舍五入逻辑
    np.save(model_dir / "lith_block.npy", truth.astype(np.float64) + 0.2)

    meta = {
        "extent": EXTENT,
        "resolution": [NX, NY, NZ],
        "id_to_name": {"1": "Dike", "2": "K1", "3": "P3", "4": "P1"},
        "name_to_color": {
            "Dike": [200, 60, 60],
            "K1": [120, 200, 120],
            "P3": [230, 200, 120],
            "P1": [150, 150, 220],
        },
        "surface_z": None,
    }
    with open(model_dir / "lith_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return truth


def assert_png_ok(path: str) -> None:
    """断言 PNG 存在且大于 10KB（保证不是空图/损坏图）。"""
    assert os.path.exists(path), f"PNG 不存在: {path}"
    size = os.path.getsize(path)
    assert size > 10 * 1024, f"PNG 过小 ({size} B): {path}"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model_dir = tmp / "model"
        out_dir = tmp / "apps"
        model_dir.mkdir()
        out_dir.mkdir()

        truth = make_fake_model(model_dir)

        # ---------- load_lith：读取 + 四舍五入 ----------
        lith, meta = load_lith(model_dir)
        assert lith.shape == (NX, NY, NZ), f"形状错误: {lith.shape}"
        assert np.issubdtype(lith.dtype, np.integer), "load_lith 应返回整数数组"
        assert np.array_equal(lith, truth), "四舍五入后应恢复真值 id"
        print("load_lith OK")

        # ---------- 应用一：虚拟钻孔 ----------
        # (55, 105) 在模型西部（xc<200 无岩脉），真值层序自上而下：
        #   K1: 0 ~ -100，P3: -100 ~ -200，P1: -200 ~ -300
        bh_png = str(out_dir / "bh.png")
        bh_csv = str(out_dir / "bh.csv")
        segs = virtual_borehole(lith, meta, 55.0, 105.0, bh_png, out_csv=bh_csv)

        assert [s["name"] for s in segs] == ["K1", "P3", "P1"], \
            f"层序错误: {[s['name'] for s in segs]}"
        # z 分界与解析真值误差 < 一个体素高度 DZ
        true_bounds = [(0.0, -100.0), (-100.0, -200.0), (-200.0, -300.0)]
        for seg, (zt, zb) in zip(segs, true_bounds):
            assert seg["z_top"] > seg["z_bottom"], "应满足 z_top > z_bottom"
            assert abs(seg["z_top"] - zt) < DZ, f"顶界偏差过大: {seg}"
            assert abs(seg["z_bottom"] - zb) < DZ, f"底界偏差过大: {seg}"
            assert abs(seg["thickness"] - (seg["z_top"] - seg["z_bottom"])) < 1e-9
        # 层段应无缝覆盖整个 z 范围
        assert abs(sum(s["thickness"] for s in segs) - 300.0) < 1e-9
        assert_png_ok(bh_png)
        assert os.path.exists(bh_csv), "未生成层段 CSV"
        print("virtual_borehole OK")

        # ---------- 越界钻孔应抛 ValueError ----------
        try:
            virtual_borehole(lith, meta, -999.0, 0.0, str(out_dir / "bad.png"))
        except ValueError:
            print("越界 ValueError OK")
        else:
            raise AssertionError("越界钻孔未抛出 ValueError")

        # ---------- 应用二：任意剖面 ----------
        # 沿南缘的东西向剖面。真值：
        #   最浅层 zc=-5  → 全为 K1(2)（岩脉顶面在该高程之下）
        #   最深层 zc=-295 → 全为 P1(4)（岩脉限定 xc>=200，此深度不出露）
        #   岩脉(1) 应在剖面中部某处出现（如 xc≈300, zc≈-140）
        sec_png = str(out_dir / "sec.png")
        n_samples = 200
        sec = arbitrary_section(lith, meta, (5.0, 5.0), (395.0, 5.0),
                                sec_png, n_samples=n_samples)
        assert sec.shape == (n_samples, NZ), f"剖面形状错误: {sec.shape}"
        assert np.all(sec[:, -1] == 2), "浅部（最顶层）应全为 K1(2)"
        assert np.all(sec[:, 0] == 4), "深部（最底层）应全为 P1(4)"
        assert np.any(sec == 1), "剖面应切到岩脉(1)"
        assert set(np.unique(sec)) <= {1, 2, 3, 4}
        assert_png_ok(sec_png)
        print("arbitrary_section OK")

        # ---------- 应用三：任意平切图 ----------
        # z=-155 → 最近体素层中心 zc=-155，位于 P3(3) 层内；
        # 岩脉条件 |−155−(0.6xc−320)|<=15 → xc∈[250,300]（i=25..29）
        sl_png = str(out_dir / "sl.png")
        slab = level_slice(lith, meta, -155.0, sl_png)
        assert slab.shape == (NX, NY), f"切片形状错误: {slab.shape}"
        assert slab[10, 10] == 3, "xc=105 处应为 P3(3)"
        assert slab[27, 20] == 1, "xc=275 处应切到岩脉(1)"
        assert slab[35, 20] == 3, "xc=355 处应为 P3(3)"
        assert np.array_equal(slab, truth[:, :, 14]), "切片应等于真值第 14 层"
        assert_png_ok(sl_png)
        print("level_slice OK")

        # ---------- 总入口 run_apps ----------
        run_dir = tmp / "apps_batch"
        products = run_apps(
            model_dir, run_dir,
            boreholes=[(55.0, 105.0)],
            sections=[((5.0, 5.0), (395.0, 5.0))],
            slices=[-155.0],
        )
        assert products["boreholes"] == [str(run_dir / "borehole_1.png")]
        assert products["sections"] == [str(run_dir / "section_1.png")]
        assert products["slices"] == [str(run_dir / "slice_1.png")]
        for paths in products.values():
            for p in paths:
                assert os.path.exists(p), f"产物缺失: {p}"
        print("run_apps OK")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
