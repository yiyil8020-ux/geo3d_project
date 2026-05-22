# ============================================================
# 地质图颜色聚类（去除黑色线条版）- 交互式输入版
# 流程：降噪 → 提取黑线 → 排除黑线做聚类 → 填补黑线区域 → 清理碎片 → 输出 mask
# ============================================================


# ==================== 0. 导入第三方库 ====================

import os
# os = Operating System（操作系统）模块
# 提供与操作系统交互的功能，比如创建文件夹、拼接文件路径等
# 这里用到：
#   os.makedirs()   → 创建文件夹（如果不存在）
#   os.path.join()  → 把文件夹路径和文件名拼接成完整路径

import cv2
# cv2 = OpenCV（Open Source Computer Vision Library）的 Python 接口
# 这是最常用的计算机视觉库，可以读取/保存/处理图像
# 这里用到：
#   cv2.imread()          → 从文件读取图像，返回一个三维数组 (高度, 宽度, 3个颜色通道)
#   cv2.cvtColor()        → 转换颜色空间（比如 BGR→RGB、RGB→灰度、RGB→LAB）
#   cv2.GaussianBlur()    → 高斯模糊，用来降低图像噪声
#   cv2.dilate()          → 膨胀操作，让白色区域变"胖"
#   cv2.morphologyEx()    → 形态学运算（这里用闭运算 MORPH_CLOSE）
#   cv2.imwrite()         → 把数组保存为图像文件

import numpy as np
# numpy = Numerical Python，Python 中处理数值数组的核心库
# 图像在程序里本质就是一个多维数组（矩阵），numpy 让你可以高效地操作它
# 这里用到：
#   np.ones()             → 创建一个全是 1 的数组
#   np.random.choice()    → 从数组中随机抽取若干个元素
#   np.save()             → 把数组保存为 .npy 文件（numpy 专用格式，下次可以直接加载）
#   数组的布尔索引         → 比如 pixels[~mask] 表示"取出 mask 为 False 的那些像素"

from sklearn.cluster import KMeans
# sklearn = scikit-learn，Python 最常用的机器学习库
# KMeans = K 均值聚类算法
#
# 【什么是 KMeans？】
# 假设你有一堆彩色点，你想把它们自动分成 k 组，让同一组的颜色尽量接近。
# KMeans 的做法：
#   1. 随机选 k 个"中心点"
#   2. 把每个点分配给离它最近的中心
#   3. 重新计算每组的平均位置作为新中心
#   4. 重复 2-3 步，直到中心不再变化
# 最终每个点都有了一个类别标签（0, 1, 2, ..., k-1）
#
# 在本程序中：
#   - "彩色点"就是图像中每个像素的 LAB 颜色值
#   - "k 组"就是 k 种地层颜色
#   - 聚类结果：每个像素被分配到某一种地层

from skimage.morphology import remove_small_objects, remove_small_holes
# skimage = scikit-image，专注于图像处理的 Python 库
# remove_small_objects(布尔图, min_size=面积阈值)
#   → 在一张黑白图中，把面积小于 min_size 的白色碎片删掉（变成黑色）
#   → 用途：删掉聚类结果中的小噪点
# remove_small_holes(布尔图, area_threshold=面积阈值)
#   → 在一张黑白图中，把面积小于 area_threshold 的黑色小洞填上（变成白色）
#   → 用途：填补地层区域中间的小空洞

from scipy.ndimage import distance_transform_edt
# scipy = Scientific Python，Python 科学计算库
# ndimage = N-dimensional image，多维图像处理模块
# distance_transform_edt = 欧氏距离变换（Euclidean Distance Transform）
#
# 【什么是距离变换？】
# 给你一张黑白图：白色=有效区域，黑色=空洞
# 距离变换会计算每个黑色像素"离最近的白色像素有多远"
#
# 【这里为什么需要它？】
# 黑线区域的像素被标记为 -1（未知类别），我们需要把它们填上正确的地层标签。
# 最合理的做法：每个黑线像素应该被填上"离它最近的已知地层像素"的标签。
# distance_transform_edt 的 return_indices=True 参数会返回"最近有效像素的坐标"，
# 这样我们就能直接查到最近像素的标签，一步到位。
#
# 【旧代码为什么有 bug？】
# 旧代码用的是 cv2.distanceTransformWithLabels()，它返回的是"连通分量编号"，
# 不是像素在数组中的下标。旧代码把连通分量编号当下标用，导致取到的标签完全随机。

import matplotlib.pyplot as plt
# matplotlib = Python 最常用的绑图库
# plt = pyplot，matplotlib 的快捷绑图接口
# 这里用来：
#   plt.figure()   → 创建一张画布
#   plt.imshow()   → 在画布上显示图像
#   plt.savefig()  → 把画布保存为图片文件
#   plt.close()    → 关闭画布，释放内存


# ==================== 1. 用户输入参数 ====================
# input("提示文字") 会暂停程序，等待用户在终端输入文字，按回车后返回一个字符串。
# .strip() 去掉用户输入首尾的空格和换行符。
# 如果用户直接按回车（空字符串 ""），Python 中空字符串等价于 False，
# 所以 "" or "默认值" 的结果就是 "默认值"。

print("===== 地质图颜色聚类配置 =====")
print("（直接回车使用默认值）\n")

# --- 路径类参数 ---
input_path = input("请输入图像路径（默认 data/cropped_map.png）：").strip() or "data/cropped_map.png"
debug_dir  = input("请输入调试输出目录（默认 outputs/debug）：").strip() or "outputs/debug"
mask_dir   = input("请输入 mask 输出目录（默认 outputs/masks_clean）：").strip() or "outputs/masks_clean"

# --- 数值类参数 ---
# input() 返回的永远是字符串，需要用 int() 转成整数
# 示例：用户输入 "12" → int("12") → 整数 12

n_clusters_str = input("请输入聚类数量 即地层种类数（推荐 10~15，默认 12）：").strip()
n_clusters = int(n_clusters_str) if n_clusters_str else 12
# 【为什么默认 12？】
# 一张典型的彩色地质图通常包含 8~15 种不同的地层填色。
# 示例图中可以看到 Q、E₁、E₂、K₁、K₂、P₁、P₂、P₃、T₁、O₁、O₂、O₃、S₁ 共约 13 种。
# 设成 12 是一个合理的起始值。如果发现某些地层被合并，再增大这个数字。

line_threshold_str = input("请输入黑线灰度阈值 0~255（推荐 80~120，默认 90，越大排除越多）：").strip()
line_threshold = int(line_threshold_str) if line_threshold_str else 90
# 【什么是灰度阈值？】
# 彩色图像转成灰度后，每个像素的值在 0（纯黑）到 255（纯白）之间。
# 设阈值=90 意味着：灰度值 < 90 的像素被认为是"深色线条"（地质界线、等高线、文字等）。
# 阈值越大 → 更多像素被判为"黑线" → 排除越多 → 地层颜色越干净，但也可能误伤深色地层。
# 阈值越小 → 只排除最黑的像素 → 可能漏掉灰色的等高线。
# 建议先用默认值跑一次，看 outputs/debug/line_mask.png 来判断是否需要调整。

blur_size_str = input("请输入高斯模糊核大小（必须奇数，推荐 3 或 5，默认 3）：").strip()
blur_size = int(blur_size_str) if blur_size_str else 3
# 【什么是高斯模糊？】
# 对图像做一次"柔化"处理，让每个像素的颜色变成周围像素的加权平均值。
# 权重呈高斯（钟形曲线）分布：越近的像素权重越大，越远的越小。
# 核大小（kernel size）= 参与平均的邻居范围。3 表示 3×3 的邻居，5 表示 5×5。
# 【为什么要模糊？】
# 扫描版地质图会有像素级别的噪点（颗粒感），模糊可以消除这些噪点。
# 但模糊太多会让窄条状的地层边界变模糊，所以默认用较小的 3。

min_area_str = input("请输入最小区域面积/像素（小于此值的碎片删除，推荐 300~2000，默认 500）：").strip()
min_area = int(min_area_str) if min_area_str else 500
# 【为什么要设最小面积？】
# 聚类后，某些地层区域内部会出现一些"碎点"——
# 几十个像素被错误地标记成了另一种地层，看起来像是噪点。
# 设 min_area=500 表示：面积小于 500 像素的碎片会被删除。
# 注意：如果地图上有面积很小的地层（如 E₁、S₁），这个值设太大会误删它们。

sample_size_str = input("请输入 KMeans 采样像素数（推荐 100000~300000，默认 200000）：").strip()
sample_size_cfg = int(sample_size_str) if sample_size_str else 200000
# 【为什么要采样，而不是用全部像素？】
# 一张 2000×1500 的图有 300 万像素。如果让 KMeans 处理 300 万个数据点，
# 会非常慢（可能需要几分钟甚至更久）。
# 所以我们随机抽取一部分像素来"学习"颜色分组规律（fit），
# 然后再用学到的规律对所有像素进行分类（predict）。
# 采样太少（如 50000）→ 面积很小的地层可能完全没被采到，被忽略。
# 采样太多（如 500000）→ 运行变慢。200000 是一个折中的好选择。

morph_kernel_str = input("请输入闭运算核大小（必须奇数，推荐 3~7，默认 5）：").strip()
morph_kernel_size = int(morph_kernel_str) if morph_kernel_str else 5
# 【什么是闭运算（Morphological Close）？】
# 闭运算 = 先膨胀再腐蚀。
# 膨胀：白色区域向外扩张（"变胖"）→ 能填上小裂缝和小孔洞
# 腐蚀：白色区域向内收缩（"变瘦"）→ 恢复原来的大小
# 两步合在一起的效果：小孔洞被填上了，但整体形状基本不变。
# 核大小决定了"多大的孔洞能被填上"。5×5 是一个常用值。

dilate_iter_str = input("请输入黑线膨胀次数（推荐 1~3，默认 1，越大黑线 mask 越粗）：").strip()
dilate_iter = int(dilate_iter_str) if dilate_iter_str else 1
# 【什么是膨胀（Dilate）？】
# 在黑线 mask 中，白色=黑线，黑色=非黑线。
# 膨胀让白色区域向四周扩展一圈，相当于让黑线 mask "变胖"。
# 为什么要膨胀？因为黑线的边缘像素颜色不纯黑，灰度阈值可能漏掉边缘，
# 膨胀 1 次可以把这些边缘也纳入 mask。
# 膨胀太多会吃掉周围的地层像素。

print("\n===== 参数确认 =====")
print(f"  图像路径:       {input_path}")
print(f"  聚类数:         {n_clusters}")
print(f"  黑线阈值:       {line_threshold}")
print(f"  模糊核大小:     {blur_size}")
print(f"  最小区域面积:   {min_area}")
print(f"  采样像素数:     {sample_size_cfg}")
print(f"  闭运算核大小:   {morph_kernel_size}")
print(f"  黑线膨胀次数:   {dilate_iter}")
print(f"  调试输出目录:   {debug_dir}")
print(f"  mask 输出目录:  {mask_dir}")
print("\n开始处理...\n")

# 创建输出目录
# exist_ok=True 表示：如果目录已存在就不报错，直接跳过
os.makedirs(debug_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)


# ==================== 2. 读取图像 ====================
img_bgr = cv2.imread(input_path)
# cv2.imread() 读取图片文件，返回一个三维 numpy 数组：
#   形状 = (高度, 宽度, 3)
#   3 个通道的顺序是 BGR（蓝、绿、红）—— 注意不是常见的 RGB！
#   这是 OpenCV 的历史遗留设计。
# 如果文件不存在或格式不支持，返回 None。

if img_bgr is None:
    raise FileNotFoundError(f"找不到图像文件: {input_path}")
    # raise = 抛出异常，程序会立即停止并显示错误信息

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# cvtColor = Convert Color，转换颜色空间
# COLOR_BGR2RGB：把 BGR 顺序转成 RGB 顺序
# 后续的 matplotlib 绑图需要 RGB 格式，所以这里转换一下

print(f"图像尺寸: {img_rgb.shape[1]} × {img_rgb.shape[0]} 像素 (宽×高)")
print(f"总像素数: {img_rgb.shape[0] * img_rgb.shape[1]:,}")


# ==================== 3. 高斯模糊降噪 ====================
img_blur = cv2.GaussianBlur(img_rgb, (blur_size, blur_size), 0)
# 参数说明：
#   img_rgb              → 输入图像
#   (blur_size, blur_size) → 模糊核的大小，必须是奇数（如 3×3、5×5）
#   0                    → sigma（高斯分布的标准差），设 0 让 OpenCV 根据核大小自动计算

h, w, _ = img_blur.shape
# h = height（高度），w = width（宽度），_ = 通道数（3，我们不需要用到所以用 _ 忽略）


# ==================== 4. 提取深色线条 mask ====================
gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
# 把彩色图转成灰度图。灰度图只有一个通道，每个像素是 0~255 的单个数值。
# 转换公式大致是：灰度 = 0.299×R + 0.587×G + 0.114×B

line_mask = gray < line_threshold
# 这是一个布尔运算，返回一个和 gray 同样大小的布尔数组：
#   True  = 这个像素的灰度值 < 阈值 → 判定为深色线条
#   False = 这个像素的灰度值 ≥ 阈值 → 判定为地层填色

kernel = np.ones((3, 3), np.uint8)
# 创建一个 3×3 的全 1 矩阵，作为膨胀操作的"模板"（也叫结构元素）
# uint8 = unsigned int 8-bit，即 0~255 的无符号整数

line_mask = cv2.dilate(
    line_mask.astype(np.uint8),  # 布尔数组转成 0/1 的 uint8 数组（OpenCV 要求）
    kernel,                       # 膨胀的模板
    iterations=dilate_iter        # 膨胀次数：用户输入的值
).astype(bool)                    # 膨胀完再转回布尔数组

# 统计信息
line_pixel_count = np.sum(line_mask)
line_ratio = line_pixel_count / (h * w) * 100
print(f"黑线 mask 像素数: {line_pixel_count:,} ({line_ratio:.1f}%)")

# 保存 line_mask 可视化图像
cv2.imwrite(
    os.path.join(debug_dir, "line_mask.png"),
    line_mask.astype(np.uint8) * 255
    # 布尔 True=1 → 乘 255 → 白色（255）；False=0 → 黑色（0）
    # 所以保存出来的图：白色=被识别为黑线的区域，黑色=地层区域
)
print(f"已保存黑线 mask: {os.path.join(debug_dir, 'line_mask.png')}")
print("  → 请查看这张图，确认黑线、等高线、文字是否被标白。如果遗漏太多，增大 line_threshold。\n")


# ==================== 5. 转 LAB 颜色空间 + 准备聚类数据 ====================
img_lab = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)
# 【什么是 LAB 颜色空间？】
# 我们平时说的 RGB 是用红、绿、蓝三个通道描述颜色。
# LAB 是另一种描述颜色的方式，有三个通道：
#   L = Lightness（明度）：0=纯黑，255=纯白
#   A = 绿-红轴：值小偏绿，值大偏红
#   B = 蓝-黄轴：值小偏蓝，值大偏黄
#
# 【为什么用 LAB 而不是 RGB？】
# LAB 的设计目标是让"人眼感受到的颜色差异"和"数值差异"成正比。
# 在 RGB 空间中，两个颜色的数值差距大不一定意味着人眼看起来差很多。
# 用 LAB 做聚类，分出来的颜色组更符合人的视觉直觉。

pixels = img_lab.reshape(-1, 3)
# reshape(-1, 3) 把三维数组 (高, 宽, 3) 变成二维数组 (像素总数, 3)
# 每一行是一个像素的 [L, A, B] 值
# -1 表示"自动计算这个维度的大小"，等价于 h*w

flat_line_mask = line_mask.reshape(-1)
# 把二维的 line_mask (高, 宽) 也拉平成一维 (像素总数,)
# 这样 flat_line_mask[i] 就对应 pixels[i] 是否是黑线

valid_pixels = pixels[~flat_line_mask]
# ~ 是"取反"运算符：True→False, False→True
# ~flat_line_mask = "不是黑线的像素"
# pixels[~flat_line_mask] = 只保留非黑线像素的 LAB 值
# 这就实现了"排除黑线后再聚类"

print(f"有效像素数（排除黑线后）: {valid_pixels.shape[0]:,}")


# ==================== 6. KMeans 聚类（仅用非线条区域的采样） ====================
actual_sample_size = min(sample_size_cfg, valid_pixels.shape[0])
# 如果有效像素总数还不到 sample_size_cfg，就用全部像素
# min(a, b) 返回 a 和 b 中较小的那个

sample_idx = np.random.choice(valid_pixels.shape[0], actual_sample_size, replace=False)
# np.random.choice(总数, 抽取数, replace=False)
# 从 0 ~ 总数-1 中随机抽取 actual_sample_size 个不重复的整数
# replace=False 表示"不放回抽样"，即同一个像素不会被抽到两次

sample_pixels = valid_pixels[sample_idx]
# 用抽到的下标从 valid_pixels 中取出对应的像素颜色值

print(f"采样 {actual_sample_size:,} 个像素进行 KMeans 训练...")

kmeans = KMeans(
    n_clusters=n_clusters,  # 要分成几组（用户输入）
    random_state=42,        # 随机种子，设固定值让每次运行结果一致（可复现）
    n_init=10               # KMeans 会用 10 组不同的初始中心分别跑一遍，选最好的结果
                            # 因为 KMeans 对初始中心敏感，多跑几次能提高结果质量
)
kmeans.fit(sample_pixels)
# fit() = 训练/拟合
# 让 KMeans 在这些采样像素上学习出 k 个"聚类中心"（代表色）
# 训练完成后，kmeans.cluster_centers_ 里存着 k 个中心的 LAB 坐标

print(f"KMeans 训练完成。{n_clusters} 个聚类中心已确定。\n")


# ==================== 7. 对所有像素预测类别 ====================
labels_all = kmeans.predict(pixels)
# predict() = 预测
# 对图中的每一个像素（包括黑线区域的像素），根据它的 LAB 颜色值，
# 找到离它最近的那个聚类中心，把那个中心的编号作为它的类别标签。
# 返回值：一维数组，长度 = 像素总数，每个值是 0 ~ n_clusters-1 的整数

label_img = labels_all.reshape(h, w)
# 把一维标签数组重新变回 (高, 宽) 的二维图像形式

label_img = label_img.astype(np.int32)
# 转成 int32（32位有符号整数），因为下一步要把黑线位置设为 -1
# uint8 类型不能存负数

label_img[line_mask] = -1
# 把黑线位置的标签设为 -1，表示"这个像素的类别暂时未知，需要后续填补"
# 虽然上面 predict() 也给黑线像素分配了类别，但那个类别是不可靠的
# （因为训练时排除了黑线，但预测时用了黑线的颜色值）


# ==================== 8. 用距离变换填补黑线区域（已修复的版本） ====================
# 【目标】
# 黑线位置（label_img == -1）的像素需要被填上正确的地层标签。
# 策略：每个黑线像素应该被填上"离它最近的非黑线像素"的标签。
#
# 【方法：scipy.ndimage.distance_transform_edt】
# 这个函数接收一个布尔数组（True=需要填补的区域，False=已知区域）
# 返回两样东西：
#   distances      → 每个 True 像素离最近的 False 像素的欧氏距离
#   nearest_coords → 每个 True 像素对应的"最近 False 像素"的坐标
#
# nearest_coords 的形状是 (2, h, w)：
#   nearest_coords[0] = 最近有效像素的 行号（y 坐标）
#   nearest_coords[1] = 最近有效像素的 列号（x 坐标）
#
# 这样我们直接用这个坐标去 label_img 里取标签就行了。
#
# 【旧代码为什么有 bug？】
# 旧代码用 cv2.distanceTransformWithLabels()，它返回的 nearest_labels 是
# "最近有效像素所在的连通分量编号"，而不是像素的数组下标。
# 旧代码把它当数组下标用了，导致取到的标签完全是错的。

invalid_mask = (label_img == -1)
# True = 需要填补的黑线像素，False = 已经有正确标签的地层像素

print(f"需要填补的黑线像素数: {np.sum(invalid_mask):,}")

distances, nearest_coords = distance_transform_edt(
    invalid_mask,           # 输入：True 的位置需要计算距离
    return_distances=True,  # 返回距离值（虽然我们主要用坐标，但也保留距离信息）
    return_indices=True     # 返回最近有效像素的坐标（这是关键！）
)
# nearest_coords.shape = (2, h, w)
# 对于 invalid_mask 为 True 的像素 (y, x)：
#   nearest_coords[0, y, x] = 最近有效像素的 y 坐标
#   nearest_coords[1, y, x] = 最近有效像素的 x 坐标
# 对于 invalid_mask 为 False 的像素：坐标就是它自己

filled_label_img = label_img.copy()
# 复制一份，避免修改原始数据

filled_label_img[invalid_mask] = label_img[
    nearest_coords[0][invalid_mask],   # 最近有效像素的 y 坐标们
    nearest_coords[1][invalid_mask]    # 最近有效像素的 x 坐标们
]
# 这行代码做了什么：
# 1. invalid_mask 筛选出所有黑线像素
# 2. 对每个黑线像素，用 nearest_coords 找到它最近的有效像素的坐标
# 3. 从 label_img 中取出那个有效像素的标签
# 4. 把这个标签赋给黑线像素
# 注意：这是向量化操作（一次处理所有像素），不需要 for 循环，速度非常快

label_img = filled_label_img

# 验证：确保没有未填补的像素
unfilled_count = np.sum(label_img == -1)
if unfilled_count > 0:
    print(f"⚠ 警告：仍有 {unfilled_count} 个像素未被填补！")
else:
    print("✓ 所有黑线像素已成功填补。")


# ==================== 9. 生成预览图 ====================
centers_lab = kmeans.cluster_centers_.astype(np.uint8)
# kmeans.cluster_centers_ = 每个聚类中心的 LAB 坐标，形状 (n_clusters, 3)
# 转成 uint8 才能被 cv2 的颜色转换函数处理

centers_rgb = cv2.cvtColor(
    centers_lab.reshape(1, n_clusters, 3),  # 变成 (1, n_clusters, 3) 的"图像"格式
    cv2.COLOR_LAB2RGB                        # LAB → RGB
).reshape(n_clusters, 3)
# 现在 centers_rgb[i] 就是第 i 个聚类中心的 RGB 颜色

preview = centers_rgb[label_img]
# 神奇的 numpy 索引操作！
# label_img 的每个像素值是 0 ~ n_clusters-1 的整数
# centers_rgb[label_img] 把每个标签替换成对应的 RGB 颜色
# 结果：一张"只有 k 种颜色"的图像，相当于地层分区的彩色预览

plt.figure(figsize=(12, 8))          # 创建 12×8 英寸的画布
plt.imshow(preview)                   # 显示预览图
plt.axis("off")                       # 隐藏坐标轴
plt.title(f"KMeans clustering result (k={n_clusters}, line_threshold={line_threshold})")
plt.savefig(
    os.path.join(debug_dir, "cluster_no_lines_preview.png"),
    dpi=200,            # dpi = dots per inch（每英寸点数），越大图片越清晰
    bbox_inches="tight"  # 裁剪掉画布周围的空白
)
plt.close()  # 关闭画布，释放内存

print(f"\n已保存预览图: {os.path.join(debug_dir, 'cluster_no_lines_preview.png')}")
print("  → 请查看这张图，确认每种地层是否被独立识别出来。")
print("  → 如果某些地层被合并了，增大 n_clusters 再试。\n")


# ==================== 10. 生成并清理每个 mask ====================
print(f"正在生成 {n_clusters} 个地层 mask...")

for i in range(n_clusters):
    mask = (label_img == i)
    # 生成第 i 个聚类的二值 mask：属于这个类的像素=True，其他=False

    # ---- 清理步骤 1：删除小碎片 ----
    mask = remove_small_objects(mask, min_size=min_area)
    # 把面积小于 min_area 的 True 区域（小碎片）变成 False
    # 效果：消除散落在其他地层中的零星噪点

    # ---- 清理步骤 2：填充小孔洞 ----
    mask = remove_small_holes(mask, area_threshold=min_area)
    # 把面积小于 min_area 的 False 区域（小孔洞）变成 True
    # 效果：填补地层区域内部的小空洞

    # ---- 清理步骤 3：形态学闭运算 ----
    mask_uint8 = mask.astype(np.uint8) * 255
    # 转成 0/255 的 uint8 图像，OpenCV 形态学操作需要这种格式

    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    # morphologyEx = 形态学扩展运算
    # MORPH_CLOSE = 闭运算 = 先膨胀后腐蚀
    # 效果：填补 mask 中的小裂缝和窄缝隙，让地层区域更完整

    # ---- 保存 ----
    out_path = os.path.join(mask_dir, f"cluster_{i}_clean.png")
    cv2.imwrite(out_path, mask_uint8)

    # 统计这个 mask 覆盖的面积
    area = np.sum(mask_uint8 > 0)
    area_pct = area / (h * w) * 100
    print(f"  cluster_{i}: {area:>8,} 像素 ({area_pct:5.1f}%)")

# 保存完整的标签图（numpy 格式，后续步骤可以直接加载）
np.save(os.path.join(debug_dir, "label_img_no_lines.npy"), label_img)
# .npy 文件可以用 np.load("label_img_no_lines.npy") 加载回来
# 加载后得到的就是 (h, w) 的整数数组，每个值是地层标签 0 ~ n_clusters-1

print(f"\n✓ 完成！共生成 {n_clusters} 个 mask")
print(f"  mask 保存目录:  {mask_dir}")
print(f"  调试文件目录:   {debug_dir}")
print(f"  标签数组文件:   {os.path.join(debug_dir, 'label_img_no_lines.npy')}")
