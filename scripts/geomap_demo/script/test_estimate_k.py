import cv2
import numpy as np
from sklearn.cluster import KMeans
import time

def estimate_k(pixels, max_k=25, merge_threshold=12.0):
    kmeans = KMeans(n_clusters=max_k, random_state=42, n_init=3)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_
    
    merged_centers = []
    for c in centers:
        if not merged_centers:
            merged_centers.append(c)
        else:
            dists = np.linalg.norm(np.array(merged_centers) - c, axis=1)
            if np.min(dists) > merge_threshold:
                merged_centers.append(c)
    
    return len(merged_centers)

# Create a dummy image with exactly 7 distinct colors
np.random.seed(0)
img = np.zeros((100, 100, 3), dtype=np.uint8)
colors = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], 
    [255, 0, 255], [0, 255, 255], [128, 128, 128]
]
for i in range(100):
    for j in range(100):
        c = colors[np.random.randint(0, 7)]
        # add some noise
        c = np.clip(np.array(c) + np.random.randint(-20, 20, 3), 0, 255)
        img[i, j] = c

img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
pixels = img_lab.reshape(-1, 3)

t0 = time.time()
k = estimate_k(pixels)
t1 = time.time()
print(f"Estimated K: {k} (Expected: 7). Time: {t1-t0:.2f}s")
