# -*- coding: utf-8 -*-
"""
test_sectionreg.py — 剖面图深部约束配准模块测试
================================================

直接运行，不依赖 pytest：
    .venv-gempy/bin/python geo2model/tests/test_sectionreg.py
"""

import json
import os
import sys
import tempfile

import numpy as np

# 项目根目录加入 sys.path，保证 geo2model 包可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from geo2model.sectionreg import (  # noqa: E402
    SectionFrame,
    draw_frame_overlay,
    picks_csv_to_surface_points,
    picks_to_surface_points,
)


def make_frame() -> SectionFrame:
    """构造测试用配准框架。"""
    return SectionFrame(
        p1_world=(100.0, 200.0),
        p2_world=(1900.0, 1700.0),
        img_w=900,
        img_h=600,
        margin_left=80.0,
        margin_right=40.0,
        margin_top=50.0,
        margin_bottom=60.0,
        z_min=0.0,
        z_max=1000.0,
    )


def test_roundtrip_array():
    """px_to_world / world_to_px 数组批量往返一致。"""
    frame = make_frame()
    rng = np.random.default_rng(42)
    # 绘图区内随机采样（含少量边界附近点）
    u = rng.uniform(80.0, 860.0, size=200)
    v = rng.uniform(50.0, 540.0, size=200)

    x, y, z = frame.px_to_world(u, v)
    u2, v2 = frame.world_to_px(x, y, z)
    assert np.allclose(u, u2, atol=1e-6), "u 往返不一致"
    assert np.allclose(v, v2, atol=1e-6), "v 往返不一致"

    # 标量输入返回 float
    x0, y0, z0 = frame.px_to_world(80.0, 50.0)
    assert isinstance(x0, float) and isinstance(y0, float) and isinstance(z0, float)
    u0, v0 = frame.world_to_px(x0, y0, z0)
    assert isinstance(u0, float) and isinstance(v0, float)
    print("往返一致性测试通过")


def test_corners():
    """四个角点映射正确。"""
    frame = make_frame()
    uL, uR = 80.0, 900.0 - 40.0   # 绘图区左右边缘
    vT, vB = 50.0, 600.0 - 60.0   # 绘图区上下边缘

    # 左上角：A 上方 z_max
    assert np.allclose(frame.px_to_world(uL, vT), (100.0, 200.0, 1000.0), atol=1e-9)
    # 左下角：A 处 z_min
    assert np.allclose(frame.px_to_world(uL, vB), (100.0, 200.0, 0.0), atol=1e-9)
    # 右上角：B 上方 z_max
    assert np.allclose(frame.px_to_world(uR, vT), (1900.0, 1700.0, 1000.0), atol=1e-9)
    # 右下角：B 处 z_min
    assert np.allclose(frame.px_to_world(uR, vB), (1900.0, 1700.0, 0.0), atol=1e-9)

    # 剖面线长度
    assert np.isclose(frame.length, np.hypot(1800.0, 1500.0))

    # inside 判定
    assert frame.inside(uL, vT) and frame.inside(uR, vB)
    assert not frame.inside(uL - 1.0, vT)
    assert not frame.inside(uL, vB + 1.0)
    print("角点映射测试通过")


def test_picks_to_surface_points():
    """3 个区内点选往返 + 1 个区外点被跳过。"""
    frame = make_frame()
    picks = [
        {"u": 200.0, "v": 100.0, "formation": "K1"},
        {"u": 450.0, "v": 300.0, "formation": "P3"},
        {"u": 800.0, "v": 500.0, "formation": "P1"},
        {"u": 10.0, "v": 100.0, "formation": "BAD"},  # 区外，应被跳过
    ]
    df = picks_to_surface_points(picks, frame)

    assert list(df.columns) == ["X", "Y", "Z", "formation"]
    assert len(df) == 3, f"应保留 3 个点，实际 {len(df)}"
    assert "BAD" not in set(df["formation"]), "区外点未被跳过"

    # 往返断言：DataFrame 中的世界坐标反算回像素应等于原点选
    for p, (_, row) in zip(picks[:3], df.iterrows()):
        u2, v2 = frame.world_to_px(row["X"], row["Y"], row["Z"])
        assert np.isclose(u2, p["u"], atol=1e-6)
        assert np.isclose(v2, p["v"], atol=1e-6)
        assert row["formation"] == p["formation"]
    print("picks_to_surface_points 测试通过")


def test_serialization(tmpdir):
    """to_dict/from_dict 往返 + 文件版便捷入口。"""
    frame = make_frame()

    # dict 往返
    d = frame.to_dict()
    frame2 = SectionFrame.from_dict(json.loads(json.dumps(d)))
    assert frame2 == frame, "to_dict/from_dict 往返不一致"

    # 文件版入口：CSV + JSON → DataFrame（+ 输出 CSV）
    frame_json = os.path.join(tmpdir, "frame.json")
    picks_csv = os.path.join(tmpdir, "picks.csv")
    out_csv = os.path.join(tmpdir, "surface_points.csv")
    with open(frame_json, "w", encoding="utf-8") as f:
        json.dump(d, f)
    with open(picks_csv, "w", encoding="utf-8") as f:
        f.write("u,v,formation\n200,100,K1\n450,300,P3\n10,100,BAD\n")

    df = picks_csv_to_surface_points(picks_csv, frame_json, out_csv=out_csv)
    assert len(df) == 2 and os.path.exists(out_csv)
    print("序列化与文件入口测试通过")


def test_draw_overlay(tmpdir):
    """生成假剖面图并叠加配准框架，断言输出文件存在。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = make_frame()
    fake_img = np.full((frame.img_h, frame.img_w, 3), 0.95)  # 近白底假剖面图
    img_path = os.path.join(tmpdir, "section_fake.png")
    out_png = os.path.join(tmpdir, "overlay.png")
    plt.imsave(img_path, fake_img)

    draw_frame_overlay(img_path, frame, out_png)
    assert os.path.exists(out_png), "overlay 输出文件不存在"
    assert os.path.getsize(out_png) > 0
    print(f"overlay 测试通过（输出 {out_png}）")


if __name__ == "__main__":
    tmpdir = tempfile.mkdtemp(prefix="test_sectionreg_")
    test_roundtrip_array()
    test_corners()
    test_picks_to_surface_points()
    test_serialization(tmpdir)
    test_draw_overlay(tmpdir)
    print("ALL TESTS PASSED")
