# ============================================================
# 地层边界轮廓坐标提取
# 流程：加载标签图 → 逐层提取轮廓 → 简化轮廓 → 输出 CSV + 验证图
# 前置条件：需要先运行 blacklinemiss.py 生成 label_img_median.npy
# ============================================================


# ==================== 0. 导入第三方库 ====================

import os
# os = Operating System（操作系统）模块
# 这里用到：
#   os.makedirs()   → 创建文件夹
#   os.path.join()  → 拼接文件路径

import csv
# csv = Python 内置的 CSV 文件读写模块
# CSV（Comma-Separated Values）= 逗号分隔值文件，是一种最简单的表格格式。
# 用 Excel 或记事本都能打开。
# 这里用到：
#   csv.writer()    → 创建一个 CSV 写入器
#   writer.writerow()  → 写入一行数据

import cv2
# OpenCV 图像处理库
# 这里用到：
#   cv2.findContours()   → 从二值图中提取轮廓（边界点序列）
#   cv2.approxPolyDP()   → 用 Douglas-Peucker 算法简化轮廓
#   cv2.arcLength()      → 计算轮廓的周长
#   cv2.drawContours()   → 在图像上绘制轮廓线
#   cv2.imread()         → 读取图像
#   cv2.imwrite()        → 保存图像
#   cv2.cvtColor()       → 颜色空间转换

import numpy as np
# numpy 数值数组处理库
# 这里用到：
#   np.load()    → 加载 .npy 文件（上一步保存的标签图）
#   np.zeros()   → 创建全 0 数组
#   np.uint8     → 8 位无符号整数类型


# ==================== 1. 用户输入参数 ====================
print("===== 地层边界轮廓提取配置 =====")
print("（直接回车使用默认值）\n")

# --- 路径类参数 ---
label_path = input("请输入标签图路径（默认 outputs/debug/label_img_median.npy）：").strip() \
             or "outputs/debug/label_img_median.npy"
# 这是上一步 blacklinemiss.py 中值滤波后保存的标签图。
# .npy 是 numpy 专用的二进制文件格式，用 np.load() 加载后直接得到数组。

original_img_path = input("请输入原始图像路径（用于叠加验证，默认 data/cropped_map.png）：").strip() \
                    or "data/cropped_map.png"
# 原始地质图，用来在上面叠加绘制轮廓线，方便验证提取结果是否正确。

output_dir = input("请输入输出目录（默认 outputs/contours）：").strip() or "outputs/contours"

# --- 数值类参数 ---
epsilon_ratio_str = input(
    "请输入轮廓简化程度 epsilon_ratio（推荐 0.001~0.005，默认 0.002，越大越简化）："
).strip()
epsilon_ratio = float(epsilon_ratio_str) if epsilon_ratio_str else 0.002
# 【什么是 epsilon_ratio？】
# 这是 Douglas-Peucker 轮廓简化算法的核心参数。
# 实际的 epsilon 值 = 轮廓周长 × epsilon_ratio
#
# epsilon 的含义：允许简化后的轮廓偏离原始轮廓的最大距离（像素）。
# 例如一条轮廓周长 1000 像素，epsilon_ratio=0.002 → epsilon=2 像素
# 也就是说，简化后的轮廓和原始轮廓的偏差不超过 2 个像素。
#
# 用比例（ratio）而不是固定像素数的好处：
# 不管轮廓是大是小，简化程度都是相对一致的。
# 大轮廓（周长大）允许稍大的偏差，小轮廓（周长小）保持精细。
#
# epsilon_ratio 越大 → 简化越多 → 点越少 → CSV 文件越小 → 但边界越粗糙
# epsilon_ratio 越小 → 简化越少 → 点越多 → CSV 文件越大 → 但边界越精确

min_contour_len_str = input(
    "请输入最小轮廓周长/像素（短于此值的轮廓丢弃，推荐 30~100，默认 50）："
).strip()
min_contour_len = int(min_contour_len_str) if min_contour_len_str else 50
# 【为什么要过滤短轮廓？】
# 聚类和清理后，可能还残留一些极小的碎片区域（几个像素大小的小点）。
# 这些碎片的轮廓周长很短（比如只有 10~20 像素），不是真正的地层边界。
# 设 min_contour_len=50 表示：周长 < 50 像素的轮廓直接丢弃。
#
# 周长和面积的关系（近似）：
# 一个正方形区域面积 = 100 像素 → 边长 = 10 像素 → 周长 = 40 像素
# 所以 min_contour_len=50 大约对应 150 像素左右面积的小碎片。

contour_thickness_str = input(
    "请输入验证图轮廓线粗细（推荐 1~3，默认 2）："
).strip()
contour_thickness = int(contour_thickness_str) if contour_thickness_str else 2
# 绘制在验证图上的轮廓线宽度（像素）。
# 1 = 细线，适合高分辨率图；2 = 中等；3 = 粗线，更醒目。

# 打印参数确认
print("\n===== 参数确认 =====")
print(f"  标签图路径:       {label_path}")
print(f"  原始图像路径:     {original_img_path}")
print(f"  输出目录:         {output_dir}")
print(f"  轮廓简化程度:     {epsilon_ratio}")
print(f"  最小轮廓周长:     {min_contour_len}")
print(f"  验证图线粗细:     {contour_thickness}")
print("\n开始处理...\n")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)


# ==================== 2. 加载数据 ====================

label_img = np.load(label_path)
# np.load() 读取 .npy 文件，返回保存时的 numpy 数组。
# 这里加载的是 blacklinemiss.py 中保存的标签图（中值滤波后的版本）。
# 形状：(高度, 宽度)，每个像素是 0 ~ n_clusters-1 的整数，表示它属于哪个地层。

h, w = label_img.shape
# h = height（图像高度），w = width（图像宽度）

n_clusters = label_img.max() + 1
# 标签值范围是 0 ~ max，所以地层总数 = max + 1
# 例如标签有 0,1,2,...,11 → max=11 → n_clusters=12

print(f"标签图尺寸: {w} × {h} 像素")
print(f"地层数量: {n_clusters}")

# 加载原始图像用于叠加验证
original_bgr = cv2.imread(original_img_path)
if original_bgr is None:
    print(f"⚠ 警告：找不到原始图像 {original_img_path}，将跳过验证图生成。")
    overlay = None
else:
    overlay = original_bgr.copy()
    # 复制一份原图，在副本上绘制轮廓，不修改原图


# ==================== 3. 为每个地层分配一种独特的轮廓颜色 ====================
# 验证图上要用不同颜色区分不同地层的轮廓线。
# 这里用 HSV 色环均匀取色，保证每种颜色差异明显。

colors = []
for i in range(n_clusters):
    # 【HSV 色环取色原理】
    # HSV 颜色空间：
    #   H = Hue（色相）：0~179（OpenCV 中），代表颜色在色环上的位置
    #     0=红，30=橙，60=黄，90=绿，120=蓝，150=紫
    #   S = Saturation（饱和度）：0~255，越大颜色越鲜艳
    #   V = Value（明度）：0~255，越大越亮
    #
    # 把 0~179 的色环均匀分成 n_clusters 份，每份取一个色相值
    # 这样不管地层数量多少，颜色都能均匀分布、互不混淆

    hue = int(179 * i / n_clusters)
    # 第 i 个地层的色相值

    color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    # 构造一个 1×1 的 HSV "图像"（其实就是一个颜色点）
    # 饱和度和明度都拉满（255），保证颜色鲜艳明亮

    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    # 转换成 BGR 颜色（OpenCV 用的颜色格式）
    # [0][0] 是因为 cvtColor 要求输入是"图像"格式，返回的也是图像格式
    # 取 [0][0] 就是取出那个 1×1 图像中唯一的像素值

    colors.append(tuple(int(c) for c in color_bgr))
    # 把 numpy 数组转成 Python 元组，如 (255, 0, 128)
    # cv2.drawContours() 需要 Python 元组格式的颜色

print(f"已生成 {n_clusters} 种轮廓颜色。\n")


# ==================== 4. 逐层提取轮廓 ====================

all_contours_data = []
# 用来收集所有轮廓的坐标数据，最终写入 CSV
# 每条记录格式：[cluster_id, contour_id, point_index, x, y]

total_contours = 0        # 所有地层的轮廓总数
total_points = 0           # 所有轮廓的点总数
total_points_original = 0  # 简化前的原始点总数（用于统计简化效果）

for cluster_id in range(n_clusters):

    # ---- 步骤 4.1：从标签图生成该地层的二值 mask ----
    mask = (label_img == cluster_id).astype(np.uint8) * 255
    # label_img == cluster_id → 布尔数组，属于该地层的像素=True
    # .astype(np.uint8)       → True→1, False→0
    # * 255                   → 1→255, 0→0
    # 最终得到一张黑白图：白色（255）= 属于该地层，黑色（0）= 不属于

    # ---- 步骤 4.2：用 OpenCV 提取轮廓 ----
    contours_raw, hierarchy = cv2.findContours(
        mask,                       # 输入：二值 mask（黑白图）
        cv2.RETR_EXTERNAL,          # 轮廓检索模式
        cv2.CHAIN_APPROX_SIMPLE     # 轮廓近似方法
    )
    # 【cv2.findContours() 详解】
    #
    # 这个函数扫描二值图像，找到所有白色区域的边界，返回边界上的点坐标。
    #
    # 参数解释：
    #
    #   cv2.RETR_EXTERNAL（检索模式）：
    #     RETR = Retrieve（检索）
    #     EXTERNAL = 外部
    #     含义：只提取最外层轮廓，忽略轮廓内部的孔洞。
    #     为什么？我们只关心地层区域的外边界，不关心内部可能的小孔洞。
    #     其他可选模式：
    #       RETR_LIST  → 提取所有轮廓（不管层级）
    #       RETR_TREE  → 提取所有轮廓并保留嵌套层级关系
    #
    #   cv2.CHAIN_APPROX_SIMPLE（近似方法）：
    #     CHAIN = 链式表示
    #     APPROX = Approximation（近似）
    #     SIMPLE = 简单近似
    #     含义：自动压缩水平、垂直和对角线方向上的连续点。
    #     例如一条直线边界从 (0,0) 到 (100,0)，原本有 101 个点，
    #     SIMPLE 模式只保留两个端点 (0,0) 和 (100,0)。
    #     这已经是一种初步简化，但对曲线边界效果有限。
    #     另一个选项：
    #       CHAIN_APPROX_NONE → 保留所有边界点，不做任何压缩
    #
    # 返回值：
    #   contours_raw：一个 Python 列表，每个元素是一个轮廓
    #     每个轮廓是一个 numpy 数组，形状 (点数, 1, 2)
    #     最内层的 2 个值是 [x, y] 坐标（注意：x=列号, y=行号）
    #     例如 contours_raw[0] 是第一个轮廓，contours_raw[0][0][0] 是它的第一个点 [x, y]
    #
    #   hierarchy：轮廓的层级关系信息
    #     因为我们用 RETR_EXTERNAL 只取外层轮廓，这里不需要用到

    # ---- 步骤 4.3：过滤 + 简化每个轮廓 ----
    cluster_contour_count = 0  # 这个地层有效轮廓数

    for contour in contours_raw:

        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, closed=True)
        # cv2.arcLength(轮廓, closed)
        # 计算轮廓的总长度（周长）。
        # closed=True 表示这是一个闭合轮廓（首尾相连）。
        # 返回值：周长（浮点数，单位是像素）

        # 过滤太短的轮廓
        if perimeter < min_contour_len:
            continue
            # continue = 跳过当前循环的剩余部分，直接进入下一个轮廓
            # 太短的轮廓通常是噪点碎片，不是真正的地层边界

        # Douglas-Peucker 轮廓简化
        epsilon = epsilon_ratio * perimeter
        # epsilon = 允许的最大偏差距离
        # 例如周长 1000 像素，epsilon_ratio=0.002 → epsilon=2 像素

        contour_simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
        # cv2.approxPolyDP(轮廓, epsilon, closed)
        # approx = Approximate（近似）
        # PolyDP = Polygon Douglas-Peucker（Douglas-Peucker 多边形近似）
        #
        # 【Douglas-Peucker 算法详解】
        # 这是一种经典的曲线简化算法，目标是用尽量少的点来近似一条曲线。
        #
        # 算法步骤（以一段开放曲线为例）：
        #   1. 取曲线的起点 A 和终点 B，连成一条直线 AB
        #   2. 在 AB 之间的所有点中，找到离直线 AB 最远的点 C
        #   3. 如果 C 到 AB 的距离 > epsilon：
        #        → C 是一个"重要拐点"，必须保留
        #        → 把曲线分成 A-C 和 C-B 两段，分别递归处理
        #   4. 如果 C 到 AB 的距离 ≤ epsilon：
        #        → A 到 B 之间的所有点都"不重要"
        #        → 直接用直线 AB 代替这段曲线
        #
        # 最终结果：只保留了那些"转弯幅度超过 epsilon"的拐点。
        # 直线段上的冗余点被删除，但关键的拐点都被保留。
        #
        # closed=True 表示把轮廓当作闭合多边形处理（首尾相连）
        #
        # 返回值：简化后的轮廓，格式和输入一样 (简化后点数, 1, 2)

        # 记录简化前后的点数（统计用）
        total_points_original += len(contour)
        total_points += len(contour_simplified)

        # ---- 步骤 4.4：收集轮廓坐标数据 ----
        for point_idx, point in enumerate(contour_simplified):
            # enumerate() 同时获取"序号"和"元素"
            # point_idx = 这是轮廓上的第几个点（从 0 开始）
            # point 的形状是 (1, 2)，即 [[x, y]]

            x = int(point[0][0])  # x 坐标（列号，从左往右）
            y = int(point[0][1])  # y 坐标（行号，从上往下）

            all_contours_data.append([cluster_id, cluster_contour_count, point_idx, x, y])
            # 把这个点的信息添加到总列表中
            # 格式：[地层编号, 该地层内的轮廓编号, 点序号, x, y]

        # ---- 步骤 4.5：在验证图上绘制轮廓 ----
        if overlay is not None:
            cv2.drawContours(
                overlay,                    # 在这张图上画
                [contour_simplified],       # 要画的轮廓列表（必须是列表套数组）
                -1,                         # -1 表示画列表中的所有轮廓
                colors[cluster_id],         # 线条颜色（BGR 元组）
                contour_thickness           # 线条粗细（像素）
            )
            # cv2.drawContours() 在图像上绘制轮廓线。
            # 第二个参数是一个列表，每个元素是一个轮廓数组。
            # 这里 [contour_simplified] 表示只画当前这一个轮廓。
            # 第三个参数 -1 表示"画列表中的所有轮廓"（这里列表里只有一个）。

        cluster_contour_count += 1

    total_contours += cluster_contour_count
    print(f"  cluster_{cluster_id}: {cluster_contour_count} 个轮廓")


# ==================== 5. 保存 CSV 文件 ====================

csv_path = os.path.join(output_dir, "all_contours.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    # open() 打开文件用于写入
    # "w" = write 模式（写入，如果文件存在会覆盖）
    # newline="" → CSV 标准要求：不要让 Python 自动添加额外的换行符
    # encoding="utf-8" → 使用 UTF-8 编码，支持中文

    writer = csv.writer(f)
    # csv.writer() 创建一个 CSV 写入器
    # 它会自动处理：逗号分隔、引号转义等 CSV 格式细节

    # 写入表头
    writer.writerow(["cluster_id", "contour_id", "point_index", "x", "y"])
    # writerow() 写入一行数据
    # 第一行通常是列名（表头）

    # 写入所有数据行
    writer.writerows(all_contours_data)
    # writerows() 一次写入多行数据
    # all_contours_data 是一个二维列表，每个子列表是一行

print(f"\n✓ CSV 已保存: {csv_path}")
print(f"  共 {len(all_contours_data):,} 行数据")


# ==================== 6. 保存验证图 ====================

if overlay is not None:
    overlay_path = os.path.join(output_dir, "contours_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"✓ 验证图已保存: {overlay_path}")
    print(f"  → 请查看验证图，确认轮廓线是否准确贴合地层边界。")


# ==================== 7. 输出统计信息 ====================

print(f"\n===== 提取统计 =====")
print(f"  地层数量:     {n_clusters}")
print(f"  有效轮廓总数: {total_contours}")
print(f"  原始点数:     {total_points_original:,}")
print(f"  简化后点数:   {total_points:,}")
if total_points_original > 0:
    reduction = (1 - total_points / total_points_original) * 100
    print(f"  简化率:       {reduction:.1f}% 的冗余点被移除")
    # 简化率 = 被删除的点占原始点数的百分比
    # 例如原始 10000 点，简化后 2000 点 → 简化率 = 80%
print(f"\n  CSV 文件:     {csv_path}")
if overlay is not None:
    print(f"  验证图:       {overlay_path}")
print(f"\n✓ 完成！")
