# ============================================================
# 基于 OpenCV 的 Canny 边缘检测交互式演示
# 功能：加载图像，通过滑动条实时调节 Canny 阈值，显示边缘检测结果
# ============================================================

from __future__ import print_function  # 兼容 Python 2 的 print 函数写法
import cv2 as cv                       # 导入 OpenCV 库，别名为 cv
import argparse                        # 导入命令行参数解析库
import numpy as np                     # 用于处理中文路径读图（imdecode）

# -------------------- 全局参数 --------------------
max_lowThreshold = 500    # 滑动条的最大值（Canny 低阈值的上限）
window_name = 'Edge Map'  # 显示窗口的名称
title_trackbar = 'Min Threshold:'  # 滑动条的标题
ratio = 3                 # Canny 高阈值 = 低阈值 * ratio（推荐 2~3）
kernel_size = 5        # Sobel 算子的核大小（Canny 内部使用）
last_edges = None         # 记录最后一次的边缘结果，用于保存

# -------------------- 回调函数 --------------------
def CannyThreshold(val):
    """滑动条回调函数：每次拖动滑动条时被调用，重新计算并显示边缘"""
    global last_edges
    low_threshold = val  # 获取当前滑动条的值作为低阈值
    img_blur = cv.blur(src_gray, (3, 3))  # 对灰度图进行 3x3 均值模糊，降噪
    # 执行 Canny 边缘检测：低阈值、高阈值（低阈值*ratio）、Sobel核大小
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    # 原始写法（彩色边缘显示），保留作对比：
    # mask = detected_edges != 0  # 生成布尔掩膜：边缘像素为 True，其余为 False
    # dst = src * (mask[:, :, None].astype(src.dtype))  # 用掩膜提取原图中的边缘区域
    # cv.imshow(window_name, dst)  # 在窗口中显示彩色边缘图
    last_edges = detected_edges
    cv.imshow(window_name, detected_edges)  # 显示黑白边缘图

# -------------------- 命令行参数解析 --------------------
# 原始方法：通过命令行参数传路径（已注释，改用手动输入）
# parser = argparse.ArgumentParser(description='Canny 边缘检测器教程的代码。')
# parser.add_argument('--input', help='Path to input image.', default='fruits.jpg')
# args = parser.parse_args()  # 解析命令行参数
# image_path = args.input

# 新方法：运行时手动输入图像路径
image_path = input("请输入图像路径: ").strip()
# 允许用户输入带引号的路径，例如 '.../a.jpg' 或 ".../a.jpg"
if (image_path.startswith("'") and image_path.endswith("'")) or (image_path.startswith('"') and image_path.endswith('"')):
    image_path = image_path[1:-1]

def read_image(path: str):
    """更健壮的读图：先用 imread，失败则用 imdecode 支持中文/特殊字符路径。"""
    img = cv.imread(path)
    if img is not None:
        return img
    try:
        data = np.fromfile(path, dtype=np.uint8)  # 直接按字节读文件
        if data.size == 0:
            return None
        return cv.imdecode(data, cv.IMREAD_COLOR)
    except Exception:
        return None

# -------------------- 图像加载 --------------------
# 原始写法（保留对比）：
# src = cv.imread(image_path)  # 读取输入图像（BGR 格式）

src = read_image(image_path)
if src is None:  # 如果图像加载失败
    print('Could not open or find the image: ', image_path)  # 打印错误信息
    print('提示：请确认路径存在；输入时不要多余引号；若文件名含中文，本版本已尝试兼容。')
    exit(0)  # 退出程序

# -------------------- 图像预处理 --------------------
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)  # 将 BGR 彩色图转换为灰度图

# -------------------- 创建窗口和滑动条 --------------------
cv.namedWindow(window_name)  # 创建一个名为 'Edge Map' 的显示窗口
# 在窗口中创建滑动条：标题、所属窗口、初始值0、最大值100、回调函数
cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)

# -------------------- 启动显示 --------------------
CannyThreshold(0)  # 以初始阈值 0 调用一次，显示初始结果
cv.waitKey()       # 等待用户按键（程序在此阻塞，直到用户关闭窗口或按任意键）
if last_edges is not None:
    cv.imwrite('edges.png', last_edges)  # 保存黑白边缘图
