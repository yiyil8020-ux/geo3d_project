import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import time

# Create dummy image
img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
img[0:500, 0:500] = [255, 0, 0]
img[500:, 500:] = [0, 255, 0]
img[0:500, 500:] = [0, 0, 255]

img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
pixels = img_lab.reshape(-1, 3)

# Sample 10000 pixels
sample_idx = np.random.choice(pixels.shape[0], 10000, replace=False)
sample = pixels[sample_idx]

t0 = time.time()
bandwidth = estimate_bandwidth(sample, quantile=0.1, n_samples=3000)
print(f"Estimated bandwidth: {bandwidth}")
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(sample)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
t1 = time.time()

print(f"MeanShift found {len(cluster_centers)} clusters in {t1-t0:.2f} seconds.")
