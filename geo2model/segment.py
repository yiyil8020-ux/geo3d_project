# -*- coding: utf-8 -*-
"""
segment.py — 地层分区颜色分割（地质图 → 全像素单元标签）
==========================================================

任务：把彩色地质图分割成"每个像素属于哪个填色单元"的标签图，
供后续矢量化 / 建库 / 三维建模使用（见 CONTRACTS.md 的 extract/）。

核心思路（针对地质图的特点设计）：
    地质图的本质是"少数几种平涂色块 + 各种线划/文字干扰"。因此：
    1. 先把干扰要素（黑色界线、粗断层线、深灰虚线等高线、单元代号文字、
       蓝色细河流）用颜色/形态学规则剥离成"排除掩膜"；
    2. 对剩下的干净色块像素在 LAB 空间做 KMeans 过聚类（K 取大于真实
       单元数），再按 ΔE76 色差做层次合并——过聚类+合并比直接猜 K 稳健；
    3. 被排除的像素用"最近非排除像素的标签"回填（欧氏距离变换），
       等价于让色块隔着线划自然生长闭合；
    4. 中值滤波 + 小连通域吸收，清掉抗锯齿残留的碎屑。

为什么在 LAB 空间聚类：LAB 是感知均匀色彩空间，两个颜色的欧氏距离
（即 ΔE76）近似等于人眼感受到的色差，比 RGB 距离更适合"颜色相近就该
合并"这类判断。

关于 merge_delta_e 的默认值：合成场景 similar_colors 中相邻地层的
实测色差 ΔE(K2,K1)=6.1、ΔE(P2,P1)=8.1，阈值必须低于 6 才能保住这些
真实单元；而 KMeans 重复中心（同一平涂色被拆成多簇）的 ΔE≈0，
远低于阈值。故默认取 5.0：既合并重复中心，又不吞掉相近色单元。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import pdist
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans

# 蓝色水系的 HSV 判定域（OpenCV 约定：H∈[0,180]）
# 合成图河流色 (105,165,225) 的 HSV≈(105,136,225)，落在域中心附近；
# 各地层填色的 H 均 <95（黄绿橙）或 S/V 不满足，不会误入。
_WATER_H_LO, _WATER_H_HI = 95, 135
_WATER_S_MIN, _WATER_V_MIN = 60, 120


@dataclass
class SegmentParams:
    """分割参数（默认值按合成场景 base / similar_colors 调通）。"""

    mode: str = "auto"            # auto=自动聚类 | legend=图例驱动
    k_max: int = 22               # 过聚类初始 K（应明显大于预期单元数）
    merge_delta_e: float = 5.0    # LAB ΔE76 中心合并阈值（见模块注释）
    dark_gray_max: int = 110      # 暗线划阈值（灰度）：界线/断层/等高线/文字
    dark_dilate: int = 2          # 暗掩膜膨胀半径（像素），吃掉抗锯齿过渡带
    smooth_median: int = 5        # 标签图中值滤波核（奇数；<=1 关闭）
    min_region_px: int = 150      # 小连通域吸收阈值（面积小于此值被邻区吞并）
    sample_max: int = 300_000     # 聚类抽样上限（控制 KMeans 耗时）
    seed: int = 0                 # 随机种子（抽样与 KMeans 可复现）
    legend_colors: list | None = None  # legend 模式: [(name,(r,g,b)),...]
    # ---- 扫描退化预处理（鲁棒性实测：噪声/JPEG/不均匀光照下决定成败）----
    illum_correct: bool = True    # 大尺度高斯背景扣除，校正不均匀光照
    denoise: bool = True          # 双边滤波去噪（保边），抑制噪声/JPEG 块效应


# ----------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------

def _is_blue_water(rgb) -> bool:
    """判断一个 RGB 代表色是否落在蓝色水系域（用于 role_guess）。"""
    hsv = cv2.cvtColor(np.uint8([[list(rgb)]]), cv2.COLOR_RGB2HSV)[0, 0]
    return (
        _WATER_H_LO <= int(hsv[0]) <= _WATER_H_HI
        and int(hsv[1]) > _WATER_S_MIN
        and int(hsv[2]) > _WATER_V_MIN
    )


def _lab_to_rgb255(lab_center) -> list[int]:
    """单个 LAB 中心 → [r,g,b]（0..255 整数）。"""
    rgb = lab2rgb(np.asarray(lab_center, dtype=float).reshape(1, 1, 3))
    return [int(round(float(v) * 255.0)) for v in np.clip(rgb.reshape(3), 0, 1)]


def _fill_from_nearest(labels: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    """把 hole_mask=True 的像素用"最近的非空洞像素"的标签回填。

    原理：distance_transform_edt(hole, return_indices=True) 对每个像素
    给出"最近的 False（非空洞）像素"的行列下标，一次变换即可完成全部
    回填查询——比逐像素搜索快几个数量级。非空洞像素的下标指向自身，
    所以直接花式索引即可。
    """
    if not hole_mask.any():
        return labels
    if hole_mask.all():  # 全图都是空洞（病态输入），无从回填
        return labels
    idx = distance_transform_edt(
        hole_mask, return_distances=False, return_indices=True
    )
    return labels[tuple(idx)]


def _assign_nearest_center(lab_flat, keep_idx, centers, chunk=200_000):
    """非排除像素按最近 LAB 中心赋标签（分块广播，防内存爆）。

    距离矩阵 (N, k) 若一次算完，N≈百万、k≈20 时要吃上百 MB；
    按 chunk 行分块后峰值内存只有 chunk*k*4 字节量级。
    """
    labels_flat = np.zeros(lab_flat.shape[0], dtype=np.int16)
    c = centers.astype(np.float32)[None, :, :]          # (1, k, 3)
    for s in range(0, keep_idx.size, chunk):
        sel = keep_idx[s : s + chunk]
        part = lab_flat[sel][:, None, :]                # (n, 1, 3)
        d2 = ((part - c) ** 2).sum(axis=2)              # ΔE76 的平方
        labels_flat[sel] = d2.argmin(axis=1).astype(np.int16)
    return labels_flat


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------

def segment_map(img_rgb: np.ndarray, params: SegmentParams | None = None) -> dict:
    """地质图颜色分割主入口。

    参数
    ----
    img_rgb : uint8 HxWx3，RGB 地质图
    params  : SegmentParams；None 时用默认值

    返回 dict
    ---------
    labels       : int16 HxW，全像素单元标签（auto 模式 0..k-1 按面积降序；
                   legend 模式为图例序号）
    units        : [{"unit_id","color":[r,g,b],"n_px","role_guess"(,"name")}]
    exclude_mask : uint8 HxW，1=曾被排除并回填的像素（线划/河流/碎屑）
    water_mask   : uint8 HxW，1=蓝色水系域像素（含河流与湖泊）
    """
    if params is None:
        params = SegmentParams()
    img = np.ascontiguousarray(np.asarray(img_rgb)[..., :3], dtype=np.uint8)
    h, w = img.shape[:2]
    rng = np.random.default_rng(params.seed)

    # ---- 步骤 0：扫描退化预处理 -------------------------------------
    # (a) 保边去噪：双边滤波平滑噪声/JPEG 块效应的同时保住线划边缘
    #     （高斯模糊会把黑线糊进色块，双边不会）。清洁图上几乎无副作用。
    #     若估计噪声很强（skimage estimate_sigma），先上非局部均值去噪
    #     （对强高斯噪声远强于双边滤波，代价是慢几秒）。
    if params.denoise:
        from skimage.restoration import estimate_sigma

        sigma_est = float(np.mean(estimate_sigma(img, channel_axis=-1)))
        if sigma_est > 8.0:
            img = cv2.fastNlMeansDenoisingColored(
                img, None, h=int(min(3 * sigma_est, 30)),
                hColor=10, templateWindowSize=7, searchWindowSize=21,
            )
        img = cv2.bilateralFilter(img, d=7, sigmaColor=35, sigmaSpace=5)

    # (b) 残差式不均匀光照校正（两遍聚类）：
    #     朴素的"大尺度高斯背景扣除"会把大片单色区的固有亮度差误当光照
    #     扣掉，反而破坏清洁图。改为：先按原图聚类一遍，用
    #     「像素实际亮度 − 所属簇中心亮度」的残差场估计光照（理想平涂图
    #     残差≈0，凡是大尺度非零残差都是光照造成的），高斯平滑后从 L
    #     通道扣除，再走正常流程重新聚类。清洁图上残差为零 → 无副作用。
    if params.illum_correct:
        from dataclasses import replace as _dc_replace

        # 估计光照的这一遍用放宽的合并阈值：同一单元被明暗拆开的簇
        # （ΔE 主要在 L 分量，约 10~25）必须并回同簇，残差才能暴露光照；
        # 正式分割仍用 params.merge_delta_e，不受影响。
        first = segment_map(
            img,
            _dc_replace(params, illum_correct=False, denoise=False,
                        merge_delta_e=max(params.merge_delta_e, 14.0)),
        )
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        lum = lab_img[:, :, 0]
        # 各簇中心亮度（OpenCV LAB 的 L 通道是 0..255 尺度）
        max_uid = max(u["unit_id"] for u in first["units"])
        center_l = np.zeros(max_uid + 1, dtype=np.float32)
        for u in first["units"]:
            c = np.uint8([[list(u["color"])]])
            center_l[u["unit_id"]] = float(
                cv2.cvtColor(c, cv2.COLOR_RGB2LAB)[0, 0, 0]
            )
        resid = lum - center_l[np.clip(first["labels"], 0, max_uid)]
        valid = (first["exclude_mask"] == 0).astype(np.float32)
        sig = min(h, w) / 8.0
        # 掩膜加权高斯平滑：排除线划像素对光照场的污染
        num = cv2.GaussianBlur(resid * valid, (0, 0), sigmaX=sig)
        den = cv2.GaussianBlur(valid, (0, 0), sigmaX=sig) + 1e-6
        bg = num / den
        if float(np.abs(bg).max()) > 2.0:  # 光照场显著才校正
            lab_img[:, :, 0] = np.clip(lum - bg, 0, 255)
            img = cv2.cvtColor(lab_img.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # ---- 步骤 1：暗线划掩膜 ----------------------------------------
    # 黑色界线(灰度~32)/断层(~18)/深灰等高线(~78)/代号文字(~46) 的灰度都
    # 远低于任何地层填色（最暗的也 >140），一个灰度阈值即可全部拿下。
    # 抗锯齿(LINE_AA)会在线划两侧留 1~2px 的"半黑"过渡带，灰度略高于
    # 阈值，故再膨胀 dark_dilate 像素把过渡带一并吃进掩膜。
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark = (gray < params.dark_gray_max).astype(np.uint8)
    if params.dark_dilate > 0:
        ksz = 2 * params.dark_dilate + 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        dark = cv2.dilate(dark, kern)

    # ---- 步骤 2：水系处理 ------------------------------------------
    # 蓝色域像素 = 水系。5x5 开运算（先腐蚀后膨胀）会抹掉宽度 <5px 的
    # 细长结构：开运算后仍存在的是"面状水体"（湖，保留参与聚类形成
    # 自己的簇）；被抹掉的是"线状河流"，与线划同样处理——排除后回填。
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    water_mask = (
        (hsv[..., 0] >= _WATER_H_LO)
        & (hsv[..., 0] <= _WATER_H_HI)
        & (hsv[..., 1] > _WATER_S_MIN)
        & (hsv[..., 2] > _WATER_V_MIN)
    ).astype(np.uint8)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lake = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, k5)
    river = cv2.subtract(water_mask, lake)
    if params.dark_dilate > 0:
        # 河流也是抗锯齿细线，同样膨胀吃掉蓝-底色过渡带
        river = cv2.dilate(river, kern)
        river = cv2.subtract(river, lake)  # 不侵蚀湖体
    exclude = (dark > 0) | (river > 0)

    # ---- 步骤 3：抽样 + LAB 空间 KMeans 过聚类 -----------------------
    # 全图转 LAB 一次（后面全图赋标签也要用）。抽样上限 sample_max
    # 控制耗时；rng 固定种子保证可复现。K 故意取大（过聚类），
    # 宁可把一种颜色拆成几簇，也不漏掉小单元——拆多了下一步会合并。
    lab_flat = rgb2lab(img.astype(np.float64) / 255.0).reshape(-1, 3)
    lab_flat = lab_flat.astype(np.float32)
    keep_idx = np.flatnonzero(~exclude.ravel())
    if keep_idx.size == 0:
        raise ValueError("全图像素都被排除，无法聚类（检查输入图/阈值参数）")
    if keep_idx.size > params.sample_max:
        sample_idx = rng.choice(keep_idx, size=params.sample_max, replace=False)
    else:
        sample_idx = keep_idx
    sample = lab_flat[sample_idx]
    k = int(min(params.k_max, sample.shape[0]))
    km = KMeans(n_clusters=k, n_init=4, random_state=params.seed)
    with warnings.catch_warnings():
        # 平涂色图上大量像素颜色完全相同，KMeans 找到的有效簇数常少于
        # k_max，sklearn 会发 ConvergenceWarning——这正是"过聚类"的预期
        # 行为（多余中心会重合，下一步按 ΔE 合并），故定向屏蔽该警告。
        from sklearn.exceptions import ConvergenceWarning
        warnings.simplefilter("ignore", ConvergenceWarning)
        km.fit(sample)
    centers = km.cluster_centers_.astype(np.float64)          # (k, 3) LAB
    center_n = np.bincount(km.labels_, minlength=k).astype(np.float64)

    # ---- 步骤 4：中心层次合并（ΔE76 < merge_delta_e 视为同色） -------
    # 平涂色图上 KMeans 会把同一种颜色拆成多个几乎重合的中心（ΔE≈0），
    # 用平均连接层次聚类按色差把它们并回去。合并后中心 = 成员中心按
    # 各自样本数加权平均（等价于把成员样本合在一起重新求均值）。
    if k > 1:
        z = linkage(pdist(centers), method="average")  # pdist 欧氏距离即 ΔE76
        groups = fcluster(z, t=params.merge_delta_e, criterion="distance") - 1
    else:
        groups = np.zeros(1, dtype=int)
    m = int(groups.max()) + 1
    merged = np.zeros((m, 3), dtype=np.float64)
    for g in range(m):
        sel = groups == g
        wsum = center_n[sel].sum()
        wts = center_n[sel] / wsum if wsum > 0 else None
        merged[g] = np.average(centers[sel], axis=0, weights=wts)

    # ---- 步骤 5：全图非排除像素按最近合并中心赋标签（分块） ----------
    labels_flat = _assign_nearest_center(lab_flat, keep_idx, merged)
    labels = labels_flat.reshape(h, w)

    # ---- 步骤 6：排除像素回填 ---------------------------------------
    # 线划/河流覆盖处的真实地层色被涂掉了，用"最近非排除像素的标签"
    # 填充：界线居中画在两单元交界上，两侧同时向内生长会在线划中线
    # 汇合，恢复出的边界与真值边界基本重合（误差 ~1px）。
    labels = _fill_from_nearest(labels, exclude)

    # ---- 步骤 7：标签图中值滤波 -------------------------------------
    # 抗锯齿残余像素会被就近误分成孤立的杂色点，中值滤波用邻域多数值
    # 覆盖它们。中值取的是窗口内实际出现的标签值，不会造出新标签。
    # 标签数 < 256，可安全转 uint8 使用 cv2.medianBlur（比通用中值快）。
    if params.smooth_median and params.smooth_median > 1:
        ksz = params.smooth_median | 1  # 保证奇数
        labels = cv2.medianBlur(labels.astype(np.uint8), ksz).astype(np.int16)

    # ---- 步骤 8：小连通域清理 ---------------------------------------
    # 面积 < min_region_px 的碎块（线划交叉处的残渣、文字回填残影等）
    # 在地质图上没有制图意义，视为排除区再回填一次，被周围大区吞并。
    small = np.zeros((h, w), dtype=bool)
    for lab_val in np.unique(labels):
        n_cc, comp, stats, _ = cv2.connectedComponentsWithStats(
            (labels == lab_val).astype(np.uint8), connectivity=4
        )
        areas = stats[1:, cv2.CC_STAT_AREA]
        bad = np.flatnonzero(areas < params.min_region_px) + 1
        if bad.size:
            small |= np.isin(comp, bad)
    if small.any() and not small.all():
        labels = _fill_from_nearest(labels, small)
    exclude_out = (exclude | small).astype(np.uint8)

    # ---- 步骤 9/10：单元表 + （可选）图例映射 ------------------------
    if params.mode == "legend" and params.legend_colors:
        # legend 模式：每个合并中心找 ΔE 最近的图例色，多簇可映射到
        # 同一图例单元；标签重映射为图例序号，units 带上图例名。
        legend_names = [str(nm) for nm, _c in params.legend_colors]
        legend_rgb = np.array(
            [list(c) for _nm, c in params.legend_colors], dtype=np.uint8
        )
        legend_lab = rgb2lab(
            legend_rgb.reshape(1, -1, 3).astype(np.float64) / 255.0
        ).reshape(-1, 3)
        # (m, n_legend) 色差矩阵 → 每个中心的最近图例序号
        de = np.linalg.norm(merged[:, None, :] - legend_lab[None, :, :], axis=2)
        to_legend = de.argmin(axis=1).astype(np.int16)
        labels = to_legend[labels]
        units = []
        counts = np.bincount(labels.ravel(), minlength=len(legend_names))
        for li, nm in enumerate(legend_names):
            if counts[li] == 0:
                continue
            color = [int(v) for v in legend_rgb[li]]
            units.append(
                {
                    "unit_id": int(li),
                    "name": nm,
                    "color": color,
                    "n_px": int(counts[li]),
                    "role_guess": "water" if _is_blue_water(color) else "strata",
                }
            )
    else:
        # auto 模式：按面积降序重排为 0..k-1（unit_0 = 最大单元），
        # 代表色 = 合并中心转回 RGB；蓝色域簇标 water，其余 strata。
        vals, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts)
        lut = np.full(m, -1, dtype=np.int16)
        units = []
        for new_id, oi in enumerate(order):
            old = int(vals[oi])
            lut[old] = new_id
            color = _lab_to_rgb255(merged[old])
            units.append(
                {
                    "unit_id": int(new_id),
                    "color": color,
                    "n_px": int(counts[oi]),
                    "role_guess": "water" if _is_blue_water(color) else "strata",
                }
            )
        labels = lut[labels]

    return {
        "labels": labels.astype(np.int16),
        "units": units,
        "exclude_mask": exclude_out,
        "water_mask": water_mask,
    }


# ----------------------------------------------------------------------
# 落盘：extract/ 产物
# ----------------------------------------------------------------------

def save_extract(result: dict, extract_dir: str | Path, img_rgb=None) -> dict:
    """把 segment_map 结果写入 extract/（见 CONTRACTS.md）。

    写出：
    - labels.npy            : int16 HxW 分区标签
    - units_auto.csv        : 自动单元表（供人工校正 name/order/role）
    - preview_clusters.png  : 每簇涂代表色的预览图
    - preview_boundaries.png: 原图上把标签边界描红 1px（需传 img_rgb）

    返回 {文件名: 绝对路径字符串}。
    """
    d = Path(extract_dir)
    d.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(result["labels"]).astype(np.int16)
    written: dict[str, str] = {}

    # ---- labels.npy ----
    np.save(d / "labels.npy", labels)
    written["labels.npy"] = str(d / "labels.npy")

    # ---- units_auto.csv（列与 CONTRACTS.md units.csv 一致）----
    # name 默认 unit_<id>；order=-1 表示新老次序待人工确定；
    # dip/azimuth 留空待人工从图面读取。
    rows = []
    for u in result["units"]:
        rows.append(
            {
                "unit_id": u["unit_id"],
                "name": u.get("name", f"unit_{u['unit_id']}"),
                "role": u["role_guess"],
                "order": -1,
                "color_r": u["color"][0],
                "color_g": u["color"][1],
                "color_b": u["color"][2],
                "dip": "",
                "azimuth": "",
            }
        )
    cols = ["unit_id", "name", "role", "order",
            "color_r", "color_g", "color_b", "dip", "azimuth"]
    pd.DataFrame(rows, columns=cols).to_csv(d / "units_auto.csv", index=False)
    written["units_auto.csv"] = str(d / "units_auto.csv")

    # ---- preview_clusters.png：标签 → 代表色 ----
    pal = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
    for u in result["units"]:
        pal[u["unit_id"]] = u["color"]
    clusters_img = pal[np.clip(labels, 0, pal.shape[0] - 1)]
    cv2.imwrite(
        str(d / "preview_clusters.png"),
        cv2.cvtColor(clusters_img, cv2.COLOR_RGB2BGR),
    )
    written["preview_clusters.png"] = str(d / "preview_clusters.png")

    # ---- preview_boundaries.png：原图 + 红色 1px 标签边界 ----
    if img_rgb is not None:
        over = np.ascontiguousarray(np.asarray(img_rgb)[..., :3], dtype=np.uint8).copy()
        # 4-邻域标签不一致 → 边界像素（与 mapgen 真值边界同一定义）
        b = np.zeros(labels.shape, dtype=bool)
        b[:-1, :] |= labels[:-1, :] != labels[1:, :]
        b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        over[b] = (255, 0, 0)
        cv2.imwrite(
            str(d / "preview_boundaries.png"),
            cv2.cvtColor(over, cv2.COLOR_RGB2BGR),
        )
        written["preview_boundaries.png"] = str(d / "preview_boundaries.png")

    return written
