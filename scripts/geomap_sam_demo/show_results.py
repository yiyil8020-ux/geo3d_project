# ============================================================
# 🎨 一键查看聚类结果
# 把原图、黑线 mask、聚类预览图、所有 cluster mask 拼在一张图里
# ============================================================

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- 路径配置（和 blacklinemiss.py 默认一致）----------
input_path = "data/cropped_map.png"
debug_dir = "outputs/debug"
mask_dir = "outputs/masks_clean"

# ---------- 收集所有要展示的图 ----------
panels = []  # 每项是 (标题, 图像数组)

# 原图
if os.path.exists(input_path):
    img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    panels.append(("原图 Original", img))

# 黑线 mask
line_mask_path = os.path.join(debug_dir, "line_mask.png")
if os.path.exists(line_mask_path):
    panels.append(("黑线 Line Mask", cv2.imread(line_mask_path, cv2.IMREAD_GRAYSCALE)))

# 聚类预览
preview_path = os.path.join(debug_dir, "cluster_no_lines_preview.png")
if os.path.exists(preview_path):
    panels.append(("聚类预览 Preview", cv2.cvtColor(cv2.imread(preview_path), cv2.COLOR_BGR2RGB)))

# 各个 cluster mask（按文件名排序）
mask_files = sorted(glob.glob(os.path.join(mask_dir, "cluster_*_clean.png")))
for mf in mask_files:
    name = os.path.basename(mf).replace("_clean.png", "")
    panels.append((name, cv2.imread(mf, cv2.IMREAD_GRAYSCALE)))

if not panels:
    print("⚠️  没找到任何图，请先运行 blacklinemiss.py")
    exit(0)

# ---------- 自动计算网格大小 ----------
n = len(panels)
cols = 4
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
axes = np.atleast_2d(axes).reshape(rows, cols)

for idx, (title, img) in enumerate(panels):
    r, c = idx // cols, idx % cols
    ax = axes[r, c]
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

# 关掉空的子图
for idx in range(n, rows * cols):
    r, c = idx // cols, idx % cols
    axes[r, c].axis("off")

plt.suptitle("🗺️  地质图聚类结果总览", fontsize=14, y=1.02)
plt.tight_layout()

out_path = os.path.join(debug_dir, "all_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"✨ 已保存总览图: {out_path}")
print(f"📊 共展示 {n} 张图")
