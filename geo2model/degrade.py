# -*- coding: utf-8 -*-
"""
degrade.py — 扫描地质图退化模拟（鲁棒性测试）
==============================================

真实的纸质地质图扫描件会带有各种"退化"：扫描噪声、对焦模糊、JPEG 压缩
伪影、颜料褪色、扫描光照不均、图线磨损断裂、纸张污渍等。本模块把这些
退化逐一实现为可控强度（severity 1~4 档）的图像变换，用于测试矢量化
流水线在退化输入下的表现（跑出"指标 vs 退化强度"的鲁棒性曲线）。

约定：
- 输入输出均为 uint8 HxWx3 的 RGB 图像（与 input/map.png 一致）。
- 所有随机性都来自调用者传入的 np.random.Generator（rng），
  同一个种子跑两次结果完全一样，保证实验可复现。
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# 支持的退化类型（apply_degradation 的 kind 参数取值范围）
DEGRADATION_KINDS = [
    "noise",         # 高斯噪声（传感器噪声）
    "blur",          # 高斯模糊（对焦不实/扫描分辨率低）
    "jpeg",          # JPEG 压缩伪影（块效应、边缘振铃）
    "fade",          # 褪色（向白混合 + 降饱和）
    "illumination",  # 不均匀光照（扫描仪灯管/纸张弯曲造成的明暗场）
    "line_break",    # 黑色图线断裂（印刷磨损）
    "stain",         # 纸张污渍（黄褐色半透明水渍）
]

# 各退化在 severity=1..4 时的强度参数表（下标 = severity-1）
_NOISE_SIGMA = [4.0, 8.0, 16.0, 28.0]      # 高斯噪声标准差（灰度级）
_BLUR_SIGMA = [0.7, 1.2, 2.0, 3.2]         # 高斯模糊标准差（像素）
_JPEG_QUALITY = [70, 45, 28, 15]           # JPEG 质量（越低伪影越重）
_FADE_STRENGTH = [0.15, 0.30, 0.45, 0.60]  # 褪色混合比例
_ILLUM_AMP = [0.10, 0.18, 0.28, 0.40]      # 光照场相对幅度
_BREAK_N = [30, 60, 120, 240]              # 断线点个数
_BREAK_R = [3, 4, 6, 8]                    # 每个断点的修补半径（像素）
_STAIN_N_RANGE = [(2, 3), (3, 5), (4, 6), (6, 8)]  # 污渍个数 (min,max)
_STAIN_ALPHA = [0.18, 0.28, 0.38, 0.50]    # 污渍基础不透明度


def _check_input(img, severity):
    """输入合法性检查：uint8 RGB 图 + severity 1~4。"""
    img = np.asarray(img)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img 必须是 uint8 HxWx3 RGB 图像")
    if severity not in (1, 2, 3, 4):
        raise ValueError("severity 必须是 1~4 的整数")
    return img


def _to_uint8(img_float):
    """浮点图裁剪到 [0,255] 并转回 uint8。"""
    return np.clip(img_float, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# 各退化的具体实现（内部函数，统一从 apply_degradation 调用）
# ----------------------------------------------------------------------

def _degrade_noise(img, severity, rng):
    """高斯噪声：每个像素独立加一个 N(0, sigma^2) 的扰动。"""
    sigma = _NOISE_SIGMA[severity - 1]
    noise = rng.normal(0.0, sigma, size=img.shape)
    return _to_uint8(img.astype(np.float32) + noise)


def _degrade_blur(img, severity, rng):
    """高斯模糊：与高斯核卷积。ksize 传 (0,0) 让 OpenCV 按 sigma 自动定核大小。"""
    sigma = _BLUR_SIGMA[severity - 1]
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _degrade_jpeg(img, severity, rng):
    """JPEG 压缩伪影：编码成低质量 JPEG 字节流再解码回来。"""
    quality = _JPEG_QUALITY[severity - 1]
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV 编解码按 BGR 约定
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _degrade_fade(img, severity, rng):
    """褪色：先向白色混合（颜料变浅），再向灰度混合（饱和度下降）。"""
    s = _FADE_STRENGTH[severity - 1]
    out = img.astype(np.float32)
    # 1) 向白色混合：out = (1-s)*原色 + s*白
    out = out * (1.0 - s) + 255.0 * s
    # 2) 降饱和：向自身亮度（加权灰度）混合，混合比例取 s 的一半
    gray = out[..., 0] * 0.299 + out[..., 1] * 0.587 + out[..., 2] * 0.114
    desat = 0.5 * s
    out = out * (1.0 - desat) + gray[..., None] * desat
    return _to_uint8(out)


def _degrade_illumination(img, severity, rng):
    """不均匀光照：叠加平滑的随机低频明暗场。

    原理：先生成一张很小（4x4）的随机图——它天然只含低频信息；
    双三次插值放大到全图尺寸后再高斯平滑，得到一张缓慢变化的
    "明暗系数场"，逐像素乘到原图上（>1 变亮，<1 变暗）。
    """
    amp = _ILLUM_AMP[severity - 1]
    h, w = img.shape[:2]
    field_small = rng.uniform(-1.0, 1.0, size=(4, 4)).astype(np.float32)
    field = cv2.resize(field_small, (w, h), interpolation=cv2.INTER_CUBIC)
    # 大 sigma 高斯平滑进一步去掉插值残留的高频
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=min(h, w) / 16.0)
    gain = 1.0 + amp * field  # 逐像素亮度增益
    return _to_uint8(img.astype(np.float32) * gain[..., None])


def _degrade_line_break(img, severity, rng):
    """黑线断裂：随机挑一批暗像素，用图像修补(inpaint)把周围小圆抹掉。

    inpaint 会用圆外的背景颜色"补"进圆内，视觉效果正好是黑色图线
    在这些位置被擦断了。注意所有断点合并成一张 mask 只调一次
    cv2.inpaint（逐点调用会慢几十倍）。
    """
    n_break = _BREAK_N[severity - 1]
    radius = _BREAK_R[severity - 1]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark_rc = np.argwhere(gray < 110)  # (行, 列) 列表
    if len(dark_rc) == 0:
        return img.copy()  # 图上没有暗线，无事可做
    n_pick = min(n_break, len(dark_rc))
    picked = dark_rc[rng.choice(len(dark_rc), size=n_pick, replace=False)]

    mask = np.zeros(gray.shape, dtype=np.uint8)
    for r, c in picked:
        cv2.circle(mask, (int(c), int(r)), radius, 255, thickness=-1)
    return cv2.inpaint(img, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def _degrade_stain(img, severity, rng):
    """纸张污渍：若干个半透明黄褐色椭圆，边缘用高斯模糊做柔和过渡。"""
    n_min, n_max = _STAIN_N_RANGE[severity - 1]
    base_alpha = _STAIN_ALPHA[severity - 1]
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    n_stain = int(rng.integers(n_min, n_max + 1))
    for _ in range(n_stain):
        # 椭圆位置、大小、方向都随机
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        ax = int(rng.uniform(0.05, 0.20) * min(h, w)) + 2
        ay = int(rng.uniform(0.05, 0.20) * min(h, w)) + 2
        angle = float(rng.uniform(0.0, 180.0))
        # 黄褐色（茶渍/水渍色），带少量随机抖动
        color = np.array([
            160.0 + rng.uniform(-25, 25),   # R
            120.0 + rng.uniform(-20, 20),   # G
            60.0 + rng.uniform(-20, 20),    # B
        ], dtype=np.float32)
        alpha = float(np.clip(base_alpha * rng.uniform(0.7, 1.2), 0.05, 0.85))

        # 在单通道 mask 上画实心椭圆，再模糊 → 得到边缘柔和的 alpha 图
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 255, thickness=-1)
        blur_sigma = max(2.0, 0.15 * (ax + ay) / 2.0)
        mask_f = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma) / 255.0
        alpha_map = (mask_f * alpha)[..., None]
        # 标准 alpha 混合：out = (1-a)*底图 + a*污渍色
        out = out * (1.0 - alpha_map) + color[None, None, :] * alpha_map
    return _to_uint8(out)


_DEGRADE_FUNCS = {
    "noise": _degrade_noise,
    "blur": _degrade_blur,
    "jpeg": _degrade_jpeg,
    "fade": _degrade_fade,
    "illumination": _degrade_illumination,
    "line_break": _degrade_line_break,
    "stain": _degrade_stain,
}


# ----------------------------------------------------------------------
# 公开接口
# ----------------------------------------------------------------------

def apply_degradation(img, kind, severity, rng) -> np.ndarray:
    """对 RGB 图像施加一种退化。

    参数
    ----
    img      : uint8 HxWx3 RGB 图像（不会被原地修改）
    kind     : DEGRADATION_KINDS 之一
    severity : 强度档位，1（轻）~ 4（重）
    rng      : np.random.Generator，所有随机数都从它取，保证可复现

    返回
    ----
    同形状同 dtype 的新图像。
    """
    img = _check_input(img, severity)
    if kind not in _DEGRADE_FUNCS:
        raise ValueError(f"未知退化类型 {kind!r}，可选：{DEGRADATION_KINDS}")
    return _DEGRADE_FUNCS[kind](img, severity, rng)


def apply_profile(img, profile, rng) -> np.ndarray:
    """依次叠加多种退化。

    参数
    ----
    profile : [(kind, severity), ...]，按列表顺序逐个施加
              （顺序有影响，例如"先褪色再加噪"≠"先加噪再褪色"）
    """
    out = np.asarray(img)
    for kind, severity in profile:
        out = apply_degradation(out, kind, severity, rng)
    return out


def degradation_grid_preview(img, out_png):
    """生成所有 kind × severity 的缩略拼图（报告用）。

    行 = 退化类型，列 = severity 1~4。内部固定随机种子，
    保证每次生成的预览图一致。图内文字用英文（避免中文字体问题）。
    """
    import matplotlib
    matplotlib.use("Agg")  # 无界面后端，服务器/脚本环境均可用
    import matplotlib.pyplot as plt

    img = _check_input(img, 1)
    # 缩小到最长边 ~256 像素做缩略图，加快绘制
    h, w = img.shape[:2]
    scale = 256.0 / max(h, w)
    if scale < 1.0:
        thumb_src = cv2.resize(img, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    else:
        thumb_src = img

    n_rows = len(DEGRADATION_KINDS)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.4, n_rows * 2.0))
    for i, kind in enumerate(DEGRADATION_KINDS):
        for j in range(n_cols):
            severity = j + 1
            # 每格独立的固定种子 rng → 预览图可复现
            rng = np.random.default_rng(1000 + i * 10 + severity)
            degraded = apply_degradation(thumb_src, kind, severity, rng)
            ax = axes[i, j]
            ax.imshow(degraded)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"severity {severity}", fontsize=10)
            if j == 0:
                ax.set_ylabel(kind, fontsize=10)
    fig.suptitle("Degradation grid (kind x severity)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png
