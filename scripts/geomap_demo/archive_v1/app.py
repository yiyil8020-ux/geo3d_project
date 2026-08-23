# ============================================================
# 地质图智能边界提取系统 - 图形界面
# 基于 Gradio 构建，集成颜色聚类 + 轮廓提取完整流程
# 运行方式：python app.py → 浏览器自动打开 http://localhost:7860
# ============================================================


# ==================== 0. 导入库 ====================

import os
import csv
import io           # io = Input/Output，提供内存中的文件对象（这里用于 CSV 字符串拼接）
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage import distance_transform_edt, median_filter
import gradio as gr
# gradio = 一个用 Python 快速构建 Web UI 的库
# 只需要定义"输入组件"和"输出组件"，Gradio 自动生成网页界面。
# 用户在浏览器中操作，Python 函数在后台处理。
# 核心概念：
#   gr.Blocks()     → 自定义布局的容器（类似 HTML 的 <div>）
#   gr.Row()        → 让子组件水平排列
#   gr.Column()     → 让子组件垂直排列
#   gr.Slider()     → 滑块组件
#   gr.Image()      → 图像显示/上传组件
#   gr.Button()     → 按钮组件
#   gr.State()      → 隐藏状态，在前后端之间传递数据（用户看不到）
#   .click()        → 给按钮绑定点击事件（点击后执行某个 Python 函数）


# ==================== 1. 颜色聚类核心函数 ====================
# 这个函数封装了 blacklinemiss.py 的全部处理逻辑。
# 输入：原始图像 + 各种参数
# 输出：黑线 mask 图、聚类预览图、各地层 mask 图列表、标签数组、日志文本

def run_clustering(
    img_rgb,            # 原始图像，numpy 数组，形状 (h, w, 3)，RGB 格式
    n_clusters,         # 聚类数量
    line_threshold,     # 黑线灰度阈值
    blur_size,          # 高斯模糊核大小
    min_area,           # 最小区域面积
    sample_size_cfg,    # KMeans 采样数
    morph_kernel_size,  # 闭运算核大小
    dilate_iter,        # 黑线膨胀次数
    median_size,        # 中值滤波窗口
    smooth_size         # 边缘平滑核
):
    # --- 参数校验 ---
    # 高斯模糊和形态学核大小必须是奇数，如果用户输入了偶数则 +1 修正
    blur_size = int(blur_size)
    if blur_size % 2 == 0:
        blur_size += 1
    morph_kernel_size = int(morph_kernel_size)
    if morph_kernel_size % 2 == 0:
        morph_kernel_size += 1
    median_size = int(median_size)
    if median_size % 2 == 0:
        median_size += 1
    smooth_size = int(smooth_size)
    if smooth_size % 2 == 0:
        smooth_size += 1

    n_clusters = int(n_clusters)
    line_threshold = int(line_threshold)
    min_area = int(min_area)
    sample_size_cfg = int(sample_size_cfg)
    dilate_iter = int(dilate_iter)

    log_lines = []      # 收集日志信息，最后一起返回
    log = lambda msg: log_lines.append(msg)
    # lambda 是"匿名函数"的简写。
    # log("hello") 等价于 log_lines.append("hello")
    # 这样后续代码中写 log("xxx") 就能把信息收集到列表里。

    h, w, _ = img_rgb.shape
    log(f"图像尺寸: {w} x {h} 像素，总像素数: {h * w:,}")

    # --- 高斯模糊 ---
    img_blur = cv2.GaussianBlur(img_rgb, (blur_size, blur_size), 0)

    # --- 提取黑线 mask ---
    gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    line_mask = gray < line_threshold

    kernel = np.ones((3, 3), np.uint8)
    line_mask = cv2.dilate(
        line_mask.astype(np.uint8), kernel, iterations=dilate_iter
    ).astype(bool)

    line_pixel_count = np.sum(line_mask)
    log(f"黑线像素: {line_pixel_count:,} ({line_pixel_count / (h * w) * 100:.1f}%)")

    # 生成黑线 mask 可视化图（白色=黑线区域）
    line_mask_vis = line_mask.astype(np.uint8) * 255

    # --- 转 LAB + 聚类 ---
    img_lab = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)
    pixels = img_lab.reshape(-1, 3)
    flat_line_mask = line_mask.reshape(-1)
    valid_pixels = pixels[~flat_line_mask]

    actual_sample = min(sample_size_cfg, valid_pixels.shape[0])
    sample_idx = np.random.choice(valid_pixels.shape[0], actual_sample, replace=False)
    sample_pixels = valid_pixels[sample_idx]

    log(f"有效像素: {valid_pixels.shape[0]:,}，采样: {actual_sample:,}")
    log(f"开始 KMeans 聚类 (k={n_clusters})...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(sample_pixels)
    log("KMeans 训练完成。")

    # --- 预测所有像素 ---
    labels_all = kmeans.predict(pixels)
    label_img = labels_all.reshape(h, w).astype(np.int32)
    label_img[line_mask] = -1

    # --- 距离变换填补黑线 ---
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

    # --- 中值滤波 ---
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
    # preview 是一张 RGB 图像，每个像素被替换成它所属聚类中心的颜色

    # --- 生成各地层 mask ---
    mask_images = []
    # mask_images 列表，存放每个地层的 mask 可视化图
    # Gradio 的 Gallery 组件可以展示一组图片

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

        # 给 mask 图加上标题文字，方便在 Gallery 中识别
        mask_color = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2RGB)
        cv2.putText(
            mask_color,                       # 在这张图上写字
            f"cluster_{i} ({pct:.1f}%)",      # 文字内容
            (10, 30),                          # 文字左下角坐标 (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,          # 字体
            0.8,                               # 字号
            (0, 255, 0),                       # 颜色 (绿色，RGB)
            2                                  # 线宽
        )
        # cv2.putText() 在图像上绘制文字。
        # 注意：坐标 (10, 30) 是文字基线左端的位置，y=30 表示距离顶部 30 像素。

        mask_images.append(mask_color)

    log(f"\n✓ 完成！共 {n_clusters} 个地层 mask。")

    # 返回值说明：
    #   line_mask_vis  → 黑线 mask 可视化图（灰度图）
    #   preview        → 聚类预览图（RGB）
    #   mask_images    → 各地层 mask 图列表（用于 Gallery）
    #   label_img      → 标签数组（传给轮廓提取步骤，存在 gr.State 中）
    #   日志文本        → 拼接所有 log 行
    return line_mask_vis, preview, mask_images, label_img, "\n".join(log_lines)


# ==================== 2. 轮廓提取核心函数 ====================
# 封装 contour_extract.py 的逻辑。
# 输入：标签数组 + 原始图像 + 参数
# 输出：轮廓叠加图 + CSV 文本预览 + CSV 文件路径

def run_contour_extract(
    label_img,          # 标签数组（来自聚类步骤的 gr.State）
    img_rgb,            # 原始图像（来自上传组件的 gr.State）
    epsilon_ratio,      # Douglas-Peucker 简化程度
    min_contour_len,    # 最小轮廓周长
    contour_thickness   # 轮廓线粗细
):
    if label_img is None:
        return None, "⚠ 请先运行第一步（颜色聚类）！", None

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
    # Gradio 传入的图像是 RGB 格式，OpenCV 绘图需要 BGR
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

            # 绘制轮廓到叠加图上
            # 注意：颜色是 BGR 格式（OpenCV 的要求）
            cv2.drawContours(overlay_bgr, [simplified], -1, colors_bgr[cid], contour_thickness)

            cluster_count += 1

        total_contours += cluster_count
        log(f"  cluster_{cid}: {cluster_count} 个轮廓")

    # BGR → RGB 转回来给 Gradio 显示
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # --- 生成 CSV ---
    # 保存到文件
    output_dir = "outputs/contours"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "all_contours.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "contour_id", "point_index", "x", "y"])
        writer.writerows(all_data)

    # 生成预览文本（前 30 行）
    preview_lines = ["cluster_id, contour_id, point_index, x, y"]
    for row in all_data[:30]:
        preview_lines.append(", ".join(str(v) for v in row))
    if len(all_data) > 30:
        preview_lines.append(f"... 共 {len(all_data)} 行（仅显示前 30 行）")

    # --- 统计 ---
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


# ==================== 3. 构建 Gradio 界面 ====================

# gr.Blocks() 是 Gradio 的自定义布局模式，让你可以精确控制组件的排列。
# 用 with 语句嵌套表示包含关系，类似 HTML 的 <div> 嵌套。
# 【Gradio 中文主题】
# 在启动时设置 css 自定义样式

with gr.Blocks(
    title="地质图智能边界提取"     # 浏览器标签页标题
) as app:
    # app 是一个 Gradio 应用对象，后续用 app.launch() 启动

    # ---- 标题 ----
    gr.Markdown("# 🗺️ 地质图智能边界提取系统")
    # gr.Markdown() 显示 Markdown 格式的文字。
    # # 开头表示一级标题，🗺️ 是地图 emoji。

    gr.Markdown("流程：上传地质图 → 调参 → 颜色聚类 → 查看效果 → 轮廓提取 → 导出 CSV")

    # ---- 隐藏状态 ----
    # gr.State() 是一个"看不见的存储空间"，用于在不同按钮点击事件之间传递数据。
    # 用户在界面上看不到它，但后台 Python 函数可以读写它。
    state_label_img = gr.State(value=None)
    # 存储聚类结果（label_img 标签数组），供轮廓提取步骤使用
    state_original_img = gr.State(value=None)
    # 存储用户上传的原始图像（numpy 数组），供轮廓提取步骤叠加轮廓线

    # ============================================================
    # 第一步：颜色聚类
    # ============================================================
    gr.Markdown("---")
    gr.Markdown("## ▶ 第一步：颜色聚类")

    with gr.Row():
        # gr.Row() 让内部组件水平排列（左右并排）

        with gr.Column(scale=1):
            # gr.Column() 让内部组件垂直排列（上下堆叠）
            # scale=1 表示这一列占可用宽度的 1 份

            img_input = gr.Image(
                label="上传地质图",      # 组件标签
                type="numpy",           # 返回值类型：numpy 数组（而非文件路径）
                height=400              # 组件高度（像素）
            )
            # gr.Image() 是一个图像组件。
            # 设置 type="numpy" 后，用户上传图片会自动转成 numpy 数组（RGB 格式）。
            # 上传方式：点击组件区域选择文件，或直接把图片拖进去。

        with gr.Column(scale=1):
            gr.Markdown("### 参数设置")

            # gr.Slider() 创建一个滑块组件。
            # 用户可以拖动滑块来调整数值。
            # 参数说明：
            #   minimum / maximum → 滑块的最小/最大值
            #   value             → 默认值（初始位置）
            #   step              → 步进（每次拖动变化的最小单位）
            #   label             → 显示在滑块上方的文字说明

            sl_clusters = gr.Slider(
                minimum=2, maximum=25, value=12, step=1,
                label="聚类数量（地层种类数，推荐 10~15）"
            )
            sl_threshold = gr.Slider(
                minimum=40, maximum=180, value=90, step=5,
                label="黑线灰度阈值（推荐 80~120，越大排除越多）"
            )
            sl_blur = gr.Slider(
                minimum=1, maximum=11, value=3, step=2,
                label="高斯模糊核（必须奇数，推荐 3 或 5）"
            )
            sl_min_area = gr.Slider(
                minimum=100, maximum=5000, value=500, step=100,
                label="最小区域面积（推荐 300~2000）"
            )
            sl_sample = gr.Slider(
                minimum=50000, maximum=500000, value=200000, step=50000,
                label="采样像素数（推荐 100000~300000）"
            )

            # 使用 Accordion 折叠不常用的参数，减少界面拥挤
            with gr.Accordion("高级参数", open=False):
                # gr.Accordion() 是一个可折叠的面板。
                # open=False 表示默认折叠（用户可以点击展开）。
                # 把不常调整的参数放在这里，让界面更简洁。

                sl_morph = gr.Slider(
                    minimum=3, maximum=11, value=5, step=2,
                    label="闭运算核大小（推荐 3~7）"
                )
                sl_dilate = gr.Slider(
                    minimum=1, maximum=5, value=1, step=1,
                    label="黑线膨胀次数（推荐 1~3）"
                )
                sl_median = gr.Slider(
                    minimum=3, maximum=21, value=7, step=2,
                    label="中值滤波窗口（推荐 5~15，越大边界越平滑）"
                )
                sl_smooth = gr.Slider(
                    minimum=3, maximum=21, value=9, step=2,
                    label="边缘平滑核（推荐 7~15）"
                )

    btn_cluster = gr.Button("🚀 运行颜色聚类", variant="primary")
    # gr.Button() 创建一个按钮。
    # variant="primary" 让按钮显示为强调色（通常是蓝色），更醒目。

    # ---- 聚类结果展示区 ----
    gr.Markdown("### 聚类结果")

    with gr.Row():
        out_linemask = gr.Image(label="黑线 mask（白色=被排除的区域）", height=300)
        out_preview = gr.Image(label="聚类预览图", height=300)

    out_gallery = gr.Gallery(
        label="各地层 mask",
        columns=4,          # 每行显示 4 张图
        height=300
    )
    # gr.Gallery() 是一个图片画廊组件，可以展示多张图片。
    # 用户可以点击某张图放大查看。

    out_cluster_log = gr.Textbox(
        label="运行日志",
        lines=8,            # 显示 8 行高
        interactive=False   # 用户不能编辑（只读）
    )
    # gr.Textbox() 是一个文本框组件。
    # interactive=False 让它变成只读的"输出框"。

    # ---- 绑定按钮点击事件 ----
    # btn_cluster.click() 的意思是：
    # 当用户点击"运行颜色聚类"按钮时，执行 run_clustering 函数。
    #
    # fn     → 要执行的 Python 函数
    # inputs → 函数的输入参数，从这些 UI 组件中读取值
    # outputs → 函数的返回值，写入这些 UI 组件中显示

    btn_cluster.click(
        fn=run_clustering,
        inputs=[
            img_input,
            sl_clusters, sl_threshold, sl_blur, sl_min_area,
            sl_sample, sl_morph, sl_dilate, sl_median, sl_smooth
        ],
        outputs=[
            out_linemask,       # 返回值 1 → 黑线 mask 图
            out_preview,        # 返回值 2 → 聚类预览图
            out_gallery,        # 返回值 3 → mask 图列表 → Gallery
            state_label_img,    # 返回值 4 → 标签数组 → 隐藏状态
            out_cluster_log     # 返回值 5 → 日志文本 → Textbox
        ]
    )

    # 同时保存原始图像到 State（供轮廓提取使用）
    # 当用户上传图像时，自动把图像存入 state_original_img
    img_input.change(
        fn=lambda img: img,     # 原样传递
        inputs=[img_input],
        outputs=[state_original_img]
    )
    # .change() 事件：当组件的值发生变化时触发。
    # 这里的意思是：用户上传新图片后，自动把图片存到 state_original_img。

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
    # gr.File() 是一个文件下载组件。
    # 后台返回文件路径后，用户可以在界面上点击下载。

    # ---- 绑定按钮事件 ----
    btn_contour.click(
        fn=run_contour_extract,
        inputs=[
            state_label_img,    # 从隐藏状态读取标签数组
            state_original_img, # 从隐藏状态读取原始图像
            sl_epsilon, sl_min_len, sl_thickness
        ],
        outputs=[
            out_overlay,        # 轮廓叠加图
            out_contour_log,    # CSV 预览 + 日志
            out_csv_file        # CSV 文件下载
        ]
    )

    # ---- 底部说明 ----
    gr.Markdown("---")
    gr.Markdown(
        "💡 **使用提示**：先运行第一步调整参数直到地层分区满意，"
        "再运行第二步提取轮廓坐标。每次可以反复调参重跑。"
    )


# ==================== 4. 启动应用 ====================

if __name__ == "__main__":
    # __name__ == "__main__" 的含义：
    # 只有直接运行这个文件时（python app.py）才会执行下面的代码。
    # 如果这个文件被其他文件 import，则不会执行。

    app.launch(
        server_name="0.0.0.0",     # 允许局域网内其他设备访问
        server_port=7860,          # 端口号
        share=False,               # 不生成公网链接
        inbrowser=True,            # 自动在浏览器中打开
        theme=gr.themes.Soft()     # 使用 Soft 主题，外观更柔和
    )
    # app.launch() 启动 Gradio 的 Web 服务器。
    # 启动后会在终端显示类似：
    #   Running on local URL: http://0.0.0.0:7860
    # 用浏览器打开这个地址就能看到界面。
    #
    # inbrowser=True：自动打开默认浏览器。
    # share=False：不创建 Gradio 的公网共享链接（设 True 可以生成临时公网链接分享给别人）。
