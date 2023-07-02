import numpy as np
import cv2

def cross(gray, kernel):
    # Init res
    res = np.zeros_like(gray, dtype=np.float32)

    # Cross
    for row in range(1, res.shape[0] - 1):
        for col in range(1, res.shape[1] - 1):
            roi = gray[row-1:row+2, col-1:col+2]
            res[row, col] = (roi * kernel).sum()
    
    print(res)

kernel = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

img = np.array([
    [210, 210, 220, 210, 220],
    [220, 210, 180, 130, 120],
    [210, 220, 150, 160, 160],
    [70, 60, 60, 80, 90],
    [60, 50, 70, 80, 60]
], dtype=np.uint8)

cross(img, kernel)
