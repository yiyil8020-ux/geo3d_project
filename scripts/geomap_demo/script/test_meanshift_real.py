import cv2
import numpy as np
from sklearn.cluster import MeanShift
import time

img = cv2.imread('ae89f190492c1a9125e9c3809896b209.png')
img_blur = cv2.GaussianBlur(img, (3, 3), 0)
img_lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
pixels = img_lab.reshape(-1, 3)

sample_idx = np.random.choice(pixels.shape[0], 10000, replace=False)
sample = pixels[sample_idx]

for bw in [8, 10, 12, 15]:
    t0 = time.time()
    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    ms.fit(sample)
    print(f"Bandwidth {bw}: found {len(ms.cluster_centers_)} clusters in {time.time()-t0:.2f} s")

