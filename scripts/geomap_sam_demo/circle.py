import cv2 as cv
import numpy as np

# ==================== 原始代码（保留对比） ====================
# # 读取二值灰度图（单通道）
# inp = input("请输入图像路径: ").strip()
# img = cv.imread(inp, cv.IMREAD_GRAYSCALE)
# 
# # 查找轮廓（输入必须是单通道二值图）
# contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# 
# # 将灰度图转为 BGR，以便绘制彩色轮廓
# canvas = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# 
# # 在画布上绘制绿色轮廓
# re = cv.drawContours(canvas, contours, -1, (100,100,100), 1)
# 
# # 显示结果
# cv.imshow("Contours", re)
# cv.waitKey(0)
# cv.destroyAllWindows()

# ==================== 改进版：更稳的读图 + 防呆 ====================
inp = input("请输入图像路径: ").strip()

# 允许输入带引号的路径：'.../a.jpg' 或 ".../a.jpg"
if (inp.startswith("'") and inp.endswith("'")) or (inp.startswith('"') and inp.endswith('"')):
    inp = inp[1:-1]


def read_gray(path: str):
    """优先 imread；失败时用 imdecode 兼容中文/特殊字符路径。"""
    img0 = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img0 is not None:
        return img0
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv.imdecode(data, cv.IMREAD_GRAYSCALE)
    except Exception:
        return None


img = read_gray(inp)
if img is None:
    print("读图失败：", inp)
    print("提示：不要输入多余引号；确认路径存在；若含中文路径，本版本已尝试兼容。")
    raise SystemExit(1)

# findContours 更推荐输入“黑白二值图”：这里把灰度图变成黑白
_, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 灰度转彩色画布，才能画彩色轮廓
canvas = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)

re = cv.drawContours(canvas, contours, -1, (0, 255, 0), 2)

cv.imshow("Contours", re)
cv.imwrite("contours.png", re)
cv.waitKey(0)
cv.destroyAllWindows()
