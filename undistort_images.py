import cv2
import numpy as np
import os
from glob import glob

# === Camera intrinsics from COLMAP (OPENCV model) ===
fx, fy = 1438.08, 1440.12
cx, cy = 958.239, 721.1
K = np.array([[fx,  0, cx],
              [0,  fy, cy],
              [0,   0,  1]], dtype=np.float32)

# Distortion coefficients: [k1, k2, p1, p2]
dist = np.array([0.0669857, -0.0778138, -0.00018165, 0.000210369], dtype=np.float32)

# === Input/output directories ===
input_dir = "train_data/images"          # folder with original images
output_dir = "train_data/undistorted"    # folder to save undistorted images
os.makedirs(output_dir, exist_ok=True)

# Process all JPG/PNG images
image_paths = glob(os.path.join(input_dir, "*.[jp][pn]g"))  # jpg, jpeg, png
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # Compute new optimal camera matrix (keeps FOV, crops black borders)
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    # Undistort
    undistorted = cv2.undistort(img, K, dist, None, newK)

    # Optionally crop the valid region of interest (ROI)
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]

    # Save
    filename = os.path.basename(path)
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, undistorted)
    print(f"Saved: {out_path}")
