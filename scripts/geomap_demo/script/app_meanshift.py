# ============================================================
# 地质图智能边界提取系统 - MeanShift 版
# 基于 OpenCV 的 pyrMeanShiftFiltering 进行颜色分割
# 运行方式：python app_meanshift.py → 浏览器自动打开 http://localhost:7861
# 对比版本：与 app.py (KMeans 版) 使用相同的 UI 结构，方便对比效果
# ============================================================


# ==================== 0. 导入库 ====================

import os
import csv
import time
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage import distance_transform_edt, median_filter
import gradio as gr


# ==================== 1. 算法原理说明 ====================
#
# 【KMeans 版的问题】
# KMeans 把每个像素当作一个独立的颜色点来聚类，它完全不知道像素在图像中的
# 位置（空间信息）。所以：
#   - 同一块绿色地层，如果因为扫描光照导致左边偏亮、右边偏暗，KMeans 可能
#     把它们分成两个不同的类别。
#   - 边缘处的"混合色"像素（黑线和地层颜色的过渡）也会被单独归类。
#
# 【MeanShift 滤波的优势】
# cv2.pyrMeanShiftFiltering() 是 OpenCV 提供的一个图像分割预处理函数。
# 它的核心思想是：
#   1. 对于每个像素，在它的"邻域"内寻找颜色和位置都相似的像素。
#   2. 用这些相似像素的平均颜色来替换当前像素的颜色。
#   3. 这个过程会迭代进行，直到收敛。
#
# 结果：图像会变成一张"色块画"，每个区域内部颜色完全均匀，边缘却仍然锐利。
# 这正好适合地质图——每个地层变成一整块纯色，之后再做聚类就轻松多了。
#
# 参数说明：
#   sp (Spatial window radius)：空间窗口半径，单位是像素。
#       - 越大 → 平滑范围越广 → 相邻的小噪点更容易被吞并
#       - 推荐 10~30
#   sr (Color window radius)：颜色窗口半径，在 RGB 颜色空间中的距离。
#       - 越大 → 颜色差异容忍度越高 → 更多相近颜色会被合并
#       - 推荐 20~50
#   maxLevel：图像金字塔层数。
#       - 金字塔 = 先把图像缩小处理，再放大还原（加速 + 扩大感受野）
#       - 推荐 1~3
# ============================================================


# ==================== 2. MeanShift 颜色分割核心函数 ====================

def run_meanshift_clustering(
    img_rgb,            # 原始图像，numpy 数组，形状 (h, w, 3)，RGB 格式
    sp,                 # MeanShift 空间窗口半径
    sr,                 # MeanShift 颜色窗口半径
    max_level,          # 图像金字塔层数
    auto_k,             # 是否自动识别聚类数量
    n_clusters,         # 手动聚类数量（当 auto_k=False 时使用）
    line_threshold,     # 黑线灰度阈值
    min_area,           # 最小区域面积
    dilate_iter,        # 黑线膨胀次数
    morph_kernel_size,  # 闭运算核大小
    median_size,        # 中值滤波窗口
    smooth_size         # 边缘平滑核
):
    # --- 参数校验 ---
    # 形态学相关的核大小必须是奇数
    morph_kernel_size = int(morph_kernel_size)
    if morph_kernel_size % 2 == 0:
        morph_kernel_size += 1
    median_size = int(median_size)
    if median_size % 2 == 0:
        median_size += 1
    smooth_size = int(smooth_size)
    if smooth_size % 2 == 0:
        smooth_size += 1

    sp = int(sp)
    sr = int(sr)
    max_level = int(max_level)
    n_clusters = int(n_clusters)
    line_threshold = int(line_threshold)
    min_area = int(min_area)
    dilate_iter = int(dilate_iter)

    log_lines = []      # 收集日志信息
    log = lambda msg: log_lines.append(msg)

    h, w, _ = img_rgb.shape
    log(f"图像尺寸: {w} x {h} 像素，总像素数: {h * w:,}")

    # --- 第一步：MeanShift 滤波 ---
    # cv2.pyrMeanShiftFiltering() 需要 BGR 格式的输入
    # 注意：这是整个新方案的核心步骤！
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    log(f"开始 MeanShift 滤波 (sp={sp}, sr={sr}, maxLevel={max_level})...")
    t0 = time.time()

    # cv2.pyrMeanShiftFiltering() 的工作流程：
    # 1. 构建图像金字塔（从高分辨率到低分辨率的多层缩小版本）
    # 2. 在最低分辨率层上先做 MeanShift 聚类（速度快）
    # 3. 将结果逐层向上传播到高分辨率层
    # 4. 每个像素最终被替换为它所在"模态"（颜色峰值）的平均颜色
    #
    # 返回值：一张与原图同样大小的图像，但颜色被"均匀化"了
    filtered_bgr = cv2.pyrMeanShiftFiltering(
        img_bgr,        # 输入图像（BGR 格式）
        sp=sp,          # 空间窗口半径（像素）
        sr=sr,          # 颜色窗口半径（RGB 距离）
        maxLevel=max_level  # 金字塔层数
    )

    t1 = time.time()
    log(f"✓ MeanShift 滤波完成，耗时 {t1 - t0:.2f} 秒")

    # 将滤波后的图像转回 RGB，用于后续处理和展示
    filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)

    # --- 第二步：提取黑线 mask ---
    # 在滤波后的图像上提取黑线（因为 MeanShift 会让黑线更加清晰锐利）
    gray = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2GRAY)
    line_mask = gray < line_threshold

    # 膨胀黑线，确保完全覆盖
    kernel = np.ones((3, 3), np.uint8)
    line_mask = cv2.dilate(
        line_mask.astype(np.uint8), kernel, iterations=dilate_iter
    ).astype(bool)

    line_pixel_count = np.sum(line_mask)
    log(f"黑线像素: {line_pixel_count:,} ({line_pixel_count / (h * w) * 100:.1f}%)")

    # 生成黑线 mask 可视化图
    line_mask_vis = line_mask.astype(np.uint8) * 255

    # --- 第三步：在滤波后的图像上做轻量级 KMeans 聚类 ---
    # 为什么还要 KMeans？
    # MeanShift 滤波把图像变成了"色块画"，但它并没有给每个像素分配类别标签。
    # 经过滤波后，图像中的独立颜色数量已经大大减少（从数万种变成几十种），
    # 此时再用 KMeans 聚类就非常容易且准确了。
    #
    # 与原版的区别：
    # - 原版：在原始（仅高斯模糊过的）图像上 KMeans → 颜色噪声多，聚类困难
    # - 新版：在 MeanShift 滤波后的图像上 KMeans → 颜色已均匀，聚类轻松

    img_lab = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2LAB)
    # LAB 颜色空间更符合人眼感知，颜色距离更有意义

    pixels = img_lab.reshape(-1, 3)
    flat_line_mask = line_mask.reshape(-1)
    valid_pixels = pixels[~flat_line_mask]
    # 排除黑线区域的像素，只对"有颜色"的区域聚类

    # MeanShift 滤波后颜色已经很均匀，采样数可以少一些
    actual_sample = min(100000, valid_pixels.shape[0])
    sample_idx = np.random.choice(valid_pixels.shape[0], actual_sample, replace=False)
    sample_pixels = valid_pixels[sample_idx]

    log(f"有效像素: {valid_pixels.shape[0]:,}，采样: {actual_sample:,}")

    if auto_k:
        log("开始自动识别聚类数量...")
        # 滤波后的图像颜色更纯净，自动识别更准确
        # 【算法详解】
        # 1. 先用 K=25 做一次预聚类，得到 25 个颜色中心点
        # 2. 计算所有中心点之间的 LAB 颜色距离
        # 3. 如果两个中心点的距离 < 阈值（12.0），说明它们视觉上几乎一样，合并
        # 4. 最终剩下的独立中心点数量 = 自动识别的 K 值
        #
        # 为什么在 MeanShift 滤波后效果更好？
        # 因为滤波消除了渐变色和噪声，预聚类的中心点更加集中和纯净，
        # 合并步骤不容易出现"误合并"或"漏合并"。

        kmeans_est = KMeans(n_clusters=min(25, actual_sample), random_state=42, n_init=3)
        kmeans_est.fit(sample_pixels)
        centers = kmeans_est.cluster_centers_

        merge_threshold = 12.0
        merged_centers = []
        for c in centers:
            if not merged_centers:
                merged_centers.append(c)
            else:
                dists = np.linalg.norm(np.array(merged_centers) - c, axis=1)
                if np.min(dists) > merge_threshold:
                    merged_centers.append(c)

        n_clusters = len(merged_centers)
        log(f"自动识别结果: 图像中主要包含 {n_clusters} 种独立的地层颜色。")

    log(f"开始最终 KMeans 聚类 (k={n_clusters})...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(sample_pixels)
    log("KMeans 训练完成。")

    # --- 预测所有像素的类别 ---
    labels_all = kmeans.predict(pixels)
    label_img = labels_all.reshape(h, w).astype(np.int32)
    label_img[line_mask] = -1
    # 黑线区域标记为 -1，后面用距离变换填补

    # --- 距离变换填补黑线 ---
    # 和原版完全相同的逻辑：
    # 对于每个黑线像素，找到距离它最近的非黑线像素，用它的标签来填补
    invalid_mask = (label_img == -1)
    log(f"填补黑线像素: {np.sum(invalid_mask):,}")

    distances, nearest_coords = distance_transform_edt(
        invalid_mask, return_distances=True, return_indices=True
    )
    filled = label_img.copy()
    filled[invalid_mask] = label_img[
        nearest_coords[0][invalid_mask],
        nearest_coords[1][invalid_mask]
    ]
    label_img = filled
    log("✓ 黑线填补完成。")

    # --- 中值滤波平滑标签图 ---
    label_before = label_img.copy()
    label_img = median_filter(label_img, size=median_size)
    changed = np.sum(label_img != label_before)
    log(f"中值滤波修改: {changed:,} 像素 ({changed / (h * w) * 100:.1f}%)")

    # --- 生成聚类预览图 ---
    centers_lab = kmeans.cluster_centers_.astype(np.uint8)
    centers_rgb = cv2.cvtColor(
        centers_lab.reshape(1, n_clusters, 3), cv2.COLOR_LAB2RGB
    ).reshape(n_clusters, 3)
    preview = centers_rgb[label_img]

    # --- 生成各地层 mask ---
    mask_images = []

    for i in range(n_clusters):
        mask = (label_img == i)
        mask = remove_small_objects(mask, min_size=min_area)
        mask = remove_small_holes(mask, area_threshold=min_area)

        mask_uint8 = mask.astype(np.uint8) * 255
        k = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, k)

        mask_uint8 = cv2.GaussianBlur(mask_uint8, (smooth_size, smooth_size), 0)
        mask_uint8 = (mask_uint8 > 127).astype(np.uint8) * 255

        area = np.sum(mask_uint8 > 0)
        pct = area / (h * w) * 100
        log(f"  cluster_{i}: {area:,} px ({pct:.1f}%)")

        mask_color = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2RGB)
        cv2.putText(
            mask_color,
            f"cluster_{i} ({pct:.1f}%)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        mask_images.append(mask_color)

    log(f"\n✓ 完成！共 {n_clusters} 个地层 mask。")

    # 返回值：
    #   filtered_rgb   → MeanShift 滤波后的图像（新增！方便查看滤波效果）
    #   line_mask_vis  → 黑线 mask 可视化图
    #   preview        → 聚类预览图
    #   mask_images    → 各地层 mask 图列表
    #   label_img      → 标签数组（传给轮廓提取步骤）
    #   日志文本        → 拼接所有 log 行
    return filtered_rgb, line_mask_vis, preview, mask_images, label_img, "\n".join(log_lines)


# ==================== 3. 轮廓提取核心函数 ====================
# 与 app.py 完全相同，直接复用

def run_contour_extract(
    label_img,          # 标签数组（来自聚类步骤的 gr.State）
    img_rgb,            # 原始图像（来自上传组件的 gr.State）
    epsilon_ratio,      # Douglas-Peucker 简化程度
    min_contour_len,    # 最小轮廓周长
    contour_thickness   # 轮廓线粗细
):
    if label_img is None:
        return None, "⚠ 请先运行第一步（MeanShift 颜色分割）！", None

    epsilon_ratio = float(epsilon_ratio)
    min_contour_len = int(min_contour_len)
    contour_thickness = int(contour_thickness)

    h, w = label_img.shape
    n_clusters = label_img.max() + 1

    log_lines = []
    log = lambda msg: log_lines.append(msg)

    # --- 生成轮廓颜色 ---
    colors_bgr = []
    for i in range(n_clusters):
        hue = int(179 * i / n_clusters)
        c_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        c_bgr = cv2.cvtColor(c_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors_bgr.append(tuple(int(v) for v in c_bgr))

    # --- 准备叠加图 ---
    overlay_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # --- 逐层提取轮廓 ---
    all_data = []
    total_contours = 0
    total_pts_orig = 0
    total_pts_simp = 0

    for cid in range(n_clusters):
        mask = (label_img == cid).astype(np.uint8) * 255
        contours_raw, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cluster_count = 0
        for contour in contours_raw:
            perimeter = cv2.arcLength(contour, closed=True)
            if perimeter < min_contour_len:
                continue

            epsilon = epsilon_ratio * perimeter
            simplified = cv2.approxPolyDP(contour, epsilon, closed=True)

            total_pts_orig += len(contour)
            total_pts_simp += len(simplified)

            for pi, pt in enumerate(simplified):
                x, y = int(pt[0][0]), int(pt[0][1])
                all_data.append([cid, cluster_count, pi, x, y])

            cv2.drawContours(overlay_bgr, [simplified], -1, colors_bgr[cid], contour_thickness)
            cluster_count += 1

        total_contours += cluster_count
        log(f"  cluster_{cid}: {cluster_count} 个轮廓")

    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # --- 生成 CSV ---
    output_dir = "outputs/contours_meanshift"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "all_contours_meanshift.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "contour_id", "point_index", "x", "y"])
        writer.writerows(all_data)

    preview_lines = ["cluster_id, contour_id, point_index, x, y"]
    for row in all_data[:30]:
        preview_lines.append(", ".join(str(v) for v in row))
    if len(all_data) > 30:
        preview_lines.append(f"... 共 {len(all_data)} 行（仅显示前 30 行）")

    log(f"\n轮廓总数: {total_contours}")
    log(f"原始点数: {total_pts_orig:,}")
    log(f"简化后:   {total_pts_simp:,}")
    if total_pts_orig > 0:
        log(f"简化率:   {(1 - total_pts_simp / total_pts_orig) * 100:.1f}%")
    log(f"CSV 行数: {len(all_data):,}")
    log(f"CSV 路径: {csv_path}")

    csv_preview = "\n".join(preview_lines)
    full_log = "\n".join(log_lines)

    return overlay_rgb, csv_preview + "\n\n" + full_log, csv_path


# ==================== 4. 构建 Gradio 界面 ====================

with gr.Blocks(
    title="地质图边界提取 - MeanShift 版"
) as app:

    # ---- 标题 ----
    gr.Markdown("# 🗺️ 地质图智能边界提取系统 — MeanShift 版")
    gr.Markdown(
        '**算法流程**：上传地质图 → MeanShift 滤波（空间+颜色平滑）→ KMeans 聚类 → 轮廓提取 → 导出 CSV\n\n'
        '💡 与 KMeans 版的区别：增加了 **MeanShift 滤波** 预处理步骤，'
        '利用空间+颜色双重信息将图像变成"色块画"，消除渐变和噪点后再聚类，效果更好。'
    )

    # ---- 隐藏状态 ----
    state_label_img = gr.State(value=None)
    state_original_img = gr.State(value=None)

    # ============================================================
    # 第一步：MeanShift 颜色分割
    # ============================================================
    gr.Markdown("---")
    gr.Markdown("## ▶ 第一步：MeanShift 颜色分割")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                label="上传地质图",
                type="numpy",
                height=400
            )

        with gr.Column(scale=1):
            gr.Markdown("### MeanShift 滤波参数")
            gr.Markdown(
                "这三个参数控制 MeanShift 滤波的强度。\n"
                "- **空间半径 (sp)**：越大 → 平滑范围越广，小碎块被吞并\n"
                "- **颜色半径 (sr)**：越大 → 容忍更大的颜色差异，更多相近色合并\n"
                "- **金字塔层数**：越大 → 感受野更广 → 加速但可能损失细节"
            )

            sl_sp = gr.Slider(
                minimum=5, maximum=60, value=20, step=5,
                label="空间窗口半径 sp（推荐 10~30）"
            )
            # sp 控制"多大范围内的像素算'邻居'"
            # sp=20 表示以当前像素为中心，半径 20 像素内的区域

            sl_sr = gr.Slider(
                minimum=10, maximum=80, value=30, step=5,
                label="颜色窗口半径 sr（推荐 20~50）"
            )
            # sr 控制"多大的颜色差异算'相似'"
            # sr=30 表示 RGB 颜色距离在 30 以内的算同一类

            sl_maxlevel = gr.Slider(
                minimum=0, maximum=4, value=1, step=1,
                label="图像金字塔层数 maxLevel（推荐 1~2）"
            )
            # 金字塔 = 先把图像缩小若干倍，在缩小版上处理后再放大
            # 0 = 不使用金字塔（最慢但最精确）
            # 1~2 = 适中（推荐）
            # 3~4 = 最快但可能丢失小细节

            gr.Markdown("---")
            gr.Markdown("### 聚类参数")

            with gr.Row():
                cb_auto_k = gr.Checkbox(label="自动识别聚类数量", value=False)
                sl_clusters = gr.Slider(
                    minimum=2, maximum=25, value=12, step=1,
                    label="手动聚类数量（勾选自动识别后此项无效）"
                )

            sl_threshold = gr.Slider(
                minimum=40, maximum=180, value=90, step=5,
                label="黑线灰度阈值（推荐 80~120）"
            )
            sl_min_area = gr.Slider(
                minimum=100, maximum=5000, value=500, step=100,
                label="最小区域面积（推荐 300~2000）"
            )

            with gr.Accordion("高级参数", open=False):
                sl_dilate = gr.Slider(
                    minimum=1, maximum=5, value=1, step=1,
                    label="黑线膨胀次数（推荐 1~3）"
                )
                sl_morph = gr.Slider(
                    minimum=3, maximum=11, value=5, step=2,
                    label="闭运算核大小（推荐 3~7）"
                )
                sl_median = gr.Slider(
                    minimum=3, maximum=21, value=7, step=2,
                    label="中值滤波窗口（推荐 5~15）"
                )
                sl_smooth = gr.Slider(
                    minimum=3, maximum=21, value=9, step=2,
                    label="边缘平滑核（推荐 7~15）"
                )

    btn_cluster = gr.Button("🚀 运行 MeanShift 颜色分割", variant="primary")

    # ---- 分割结果展示区 ----
    gr.Markdown("### 分割结果")

    with gr.Row():
        out_filtered = gr.Image(label="MeanShift 滤波后的图像（色块画效果）", height=300)
        # 这是新增的输出！让用户直观看到 MeanShift 滤波把图像变成了什么样
        out_linemask = gr.Image(label="黑线 mask", height=300)

    with gr.Row():
        out_preview = gr.Image(label="聚类预览图", height=300)

    out_gallery = gr.Gallery(
        label="各地层 mask",
        columns=4,
        height=300
    )

    out_cluster_log = gr.Textbox(
        label="运行日志",
        lines=10,
        interactive=False
    )

    # ---- 绑定按钮点击事件 ----
    btn_cluster.click(
        fn=run_meanshift_clustering,
        inputs=[
            img_input,
            sl_sp, sl_sr, sl_maxlevel,
            cb_auto_k, sl_clusters, sl_threshold, sl_min_area,
            sl_dilate, sl_morph, sl_median, sl_smooth
        ],
        outputs=[
            out_filtered,       # 返回值 1 → MeanShift 滤波后图像（新增！）
            out_linemask,       # 返回值 2 → 黑线 mask 图
            out_preview,        # 返回值 3 → 聚类预览图
            out_gallery,        # 返回值 4 → mask 图列表 → Gallery
            state_label_img,    # 返回值 5 → 标签数组 → 隐藏状态
            out_cluster_log     # 返回值 6 → 日志文本
        ]
    )

    # 保存原始图像到 State
    img_input.change(
        fn=lambda img: img,
        inputs=[img_input],
        outputs=[state_original_img]
    )

    # ============================================================
    # 第二步：轮廓提取
    # ============================================================
    gr.Markdown("---")
    gr.Markdown("## ▶ 第二步：轮廓提取")

    with gr.Row():
        with gr.Column(scale=1):
            sl_epsilon = gr.Slider(
                minimum=0.0, maximum=0.01, value=0.002, step=0.0005,
                label="轮廓简化程度 epsilon_ratio（推荐 0.001~0.005）"
            )
            sl_min_len = gr.Slider(
                minimum=10, maximum=200, value=50, step=10,
                label="最小轮廓周长（推荐 30~100）"
            )
            sl_thickness = gr.Slider(
                minimum=1, maximum=5, value=2, step=1,
                label="轮廓线粗细（推荐 1~3）"
            )
            btn_contour = gr.Button("🚀 运行轮廓提取", variant="primary")

    gr.Markdown("### 提取结果")

    with gr.Row():
        out_overlay = gr.Image(label="轮廓叠加图", height=400)

    out_contour_log = gr.Textbox(
        label="CSV 预览 & 运行日志",
        lines=12,
        interactive=False
    )

    out_csv_file = gr.File(label="下载 CSV 文件")

    btn_contour.click(
        fn=run_contour_extract,
        inputs=[
            state_label_img,
            state_original_img,
            sl_epsilon, sl_min_len, sl_thickness
        ],
        outputs=[
            out_overlay,
            out_contour_log,
            out_csv_file
        ]
    )

    # ---- 底部说明 ----
    gr.Markdown("---")
    gr.Markdown(
        '💡 **MeanShift 版使用提示**：\n'
        '- **sp 和 sr 是最关键的两个参数**。如果地层边界模糊，试着减小 sr；如果噪点太多，增大 sp。\n'
        '- 可以先只调 MeanShift 参数，观察"滤波后的图像"效果，满意后再点击运行。\n'
        '- 建议和 KMeans 版 (app.py, 端口 7860) 同时运行，用相同的图片对比效果。'
    )


# ==================== 5. 启动应用 ====================

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,          # 使用 7861 端口，避免和 KMeans 版的 7860 冲突
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft()
    )
    # 注意端口号是 7861（不是 7860）！
    # 这样你可以同时运行两个版本：
    #   KMeans 版：http://localhost:7860
    #   MeanShift 版：http://localhost:7861
