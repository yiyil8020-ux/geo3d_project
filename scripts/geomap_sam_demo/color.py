# ============================================================
# 地质图颜色聚类脚本
# 流程：读图 → LAB 颜色空间 → KMeans 聚类 → 输出预览图和每类的 mask
# ============================================================

import os                         # 标准库：处理文件路径、创建文件夹
import cv2                        # OpenCV：图像读写与颜色空间转换
import numpy as np                # NumPy：用 N 维数组处理像素矩阵
import matplotlib.pyplot as plt   # Matplotlib：保存预览图
from sklearn.cluster import KMeans  # scikit-learn 提供的 KMeans 聚类算法


# ======================
# 1. 路径设置
# ======================
input_path = input("请输入图像路径: ").strip()  # 输入：要处理的地质图（相对于运行目录）
debug_dir = "outputs/debug"           # 输出：调试用的中间结果（预览图、label 数组）
mask_dir = "outputs/masks"            # 输出：每个颜色类别对应的 mask 图

# os.makedirs 递归创建文件夹；exist_ok=True 表示已存在不报错
os.makedirs(debug_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)


# ======================
# 2. 读取图像
# ======================
# cv2.imread：读图，返回值是一个三维 numpy 数组 (H, W, 3)，通道顺序为 BGR
img_bgr = cv2.imread(input_path)

# 如果文件路径错误或图像损坏，imread 不会抛错而是返回 None，所以必须手动判空
if img_bgr is None:
    raise FileNotFoundError(f"找不到图像文件: {input_path}")

# OpenCV 是 BGR、matplotlib 是 RGB，要先转 RGB 才能正确显示颜色
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# numpy 数组的 .shape 返回 (高, 宽, 通道数)
h, w, c = img_rgb.shape
print("图像大小:", w, "x", h)


# ======================
# 3. 转换到 LAB 颜色空间
# LAB 把"亮度"和"色彩"分离，对光照变化更鲁棒，聚类效果通常优于 RGB
# ======================
img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)


# ======================
# 4. 准备聚类数据
# KMeans 期望输入是二维数组：每行一个样本（这里就是一个像素的 LAB 值）
# ======================
# reshape(-1, 3)：把 (H, W, 3) 拉平成 (H*W, 3)。-1 表示这维度由其他维度自动推算
pixels = img_lab.reshape(-1, 3)

# 像素总数可能上百万，全用来训练太慢；随机采样最多 5 万个像素就够了
sample_size = min(50000, pixels.shape[0])

# np.random.choice：从 [0, pixels.shape[0]) 范围随机选 sample_size 个不重复的下标
sample_idx = np.random.choice(pixels.shape[0], sample_size, replace=False)

# 用花式索引（fancy indexing）取出这些像素，得到训练样本
sample_pixels = pixels[sample_idx]


# ======================
# 5. KMeans 颜色聚类
# 把像素按颜色相似度自动分成 n_clusters 组
# 经验值：地质图大约 5~6 种主色，就设 6
# ======================
n_clusters = int(input("请输入聚类数量: ").strip())  # 从用户输入获取聚类数量

# 创建 KMeans 模型对象（这里只是配置参数，还没开始算）
kmeans = KMeans(
    n_clusters=n_clusters,  # 想分成几类
    random_state=42,        # 随机种子，保证多次运行结果一致
    n_init=10               # 用不同初始化跑 10 次，取最好的那次（避免陷入局部最优）
)

# .fit：用采样像素训练模型，确定 n_clusters 个聚类中心
kmeans.fit(sample_pixels)

# .predict：对全部像素打标签，每个像素得到一个整数 0 ~ n_clusters-1
labels = kmeans.predict(pixels)

# 把一维标签数组还原成 (H, W) 的二维"标签图"，方便按位置查询
label_img = labels.reshape(h, w)


# ======================
# 6. 生成聚类预览图（把每个像素染成它所属类别的"代表色"）
# ======================
# .cluster_centers_：训练完成后得到的 n_clusters 个聚类中心（LAB 坐标，浮点数）
# 转成 uint8 才能再传给 cv2 做颜色空间转换
centers_lab = kmeans.cluster_centers_.astype(np.uint8)

# 把 (n_clusters, 3) 临时整形成一张 1×n_clusters 的图，从 LAB 转回 RGB，再变回 (n_clusters, 3)
centers_rgb = cv2.cvtColor(
    centers_lab.reshape(1, n_clusters, 3),
    cv2.COLOR_LAB2RGB
).reshape(n_clusters, 3)

# 神奇的索引：用标签图作为下标去查"代表色"表，得到一张和原图一样大的"色块图"
# 等价于：preview[i, j] = centers_rgb[label_img[i, j]]
preview = centers_rgb[label_img]

# 用 matplotlib 把预览图保存到磁盘
plt.figure(figsize=(10, 8))     # 创建画布，单位是英寸
plt.imshow(preview)             # 显示图像
plt.axis("off")                 # 关掉坐标轴
plt.title(f"KMeans color clusters: {n_clusters}")
plt.savefig(
    os.path.join(debug_dir, "cluster_preview.png"),  # 拼出输出路径
    dpi=200,                                         # 分辨率
    bbox_inches="tight"                              # 裁掉多余白边
)
plt.close()                     # 关闭画布释放内存

print("已保存聚类预览图: outputs/debug/cluster_preview.png")


# ======================
# 7. 为每个颜色类别生成单独的 mask
# mask 是黑白图，白色 = 属于这一类，黑色 = 不属于
# ======================
for i in range(n_clusters):
    # 创建一张全黑的图（高 h、宽 w、单通道、uint8）
    mask = np.zeros((h, w), dtype=np.uint8)

    # 布尔索引：把"标签等于 i"的位置赋值为 255（白）
    mask[label_img == i] = 255

    # 拼出输出路径，例如 outputs/masks/cluster_3.png
    out_path = os.path.join(mask_dir, f"cluster_{i}.png")

    # cv2.imwrite：把 numpy 数组保存为图像文件
    cv2.imwrite(out_path, mask)

print("已保存每个类别的 mask 到 outputs/masks/")


# ======================
# 8. 保存 label 数组（.npy 是 numpy 自带的二进制格式，读写都比图片快）
# 后续脚本可以用 np.load 直接读回来继续处理
# ======================
np.save(os.path.join(debug_dir, "label_img.npy"), label_img)

print("完成。")
