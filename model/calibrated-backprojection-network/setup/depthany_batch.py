#!/usr/bin/env python3
# batch_run_depthanything.py

import os
import cv2
import torch
import numpy as np
from sklearn.linear_model import RANSACRegressor
import sys

# add DepthAnything wrapper to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
)
from depth_anything_wrapper import DepthAnythingWrapper

# add data_utils to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
)
import data_utils

WEIGHTS_PATH = (
    "../../Depth-Anything-V2-main/"
    "depth_anything_v2/depth_anything_v2_vitl.pth"
)
ENCODER = "vitl"

# List of (patient_id, rgb_path, sensor_depth_path)
CASES = [
    ("patient5",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/segmentation/subpixel-embedding-segmentation/evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient5.png",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/data/scans_testing/VS/VS3Cback/data/depth/0000000543.png"
    ),
    ("patient6",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/segmentation/subpixel-embedding-segmentation/evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient6.png",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/data/scans_testing/VS/VS3Rback/data/depth/0000000543.png"
    ),
    ("patient1",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/segmentation/subpixel-embedding-segmentation/evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient1.png",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/data/scans_testing/SB/SB1Cback/data/depth/"
     "0000003618-1681416747736.9832.png"
    ),
    ("patient2",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/segmentation/subpixel-embedding-segmentation/evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient2.png",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/data/scans_testing/SB/SB1Cback/data/depth/"
     "0000032026-1681420278112.6125.png"
    ),
    ("patient3",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/segmentation/subpixel-embedding-segmentation/evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient3.png",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/data/scans_testing/SB/SB1Rback/data/depth/"
     "0000001050-1681413477438.56.png"
    ),
    ("patient4",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/segmentation/subpixel-embedding-segmentation/evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient4.png",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/data/scans_testing/SB/SB1Rback/data/depth/"
     "0000007064-1681414211362.1973.png"
    ),
    ("patient14",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/segmentation/subpixel-embedding-segmentation/evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient14.png",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/data/scans_testing/ON/ON1Cback/data/depth/0000000100.png"
    ),
    ("patient15",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/segmentation/subpixel-embedding-segmentation/evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient15.png",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/data/scans_testing/ON/ON1Cback/data/depth/0000007100.png"
    ),
    ("patient16",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/segmentation/subpixel-embedding-segmentation/evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient16.png",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/data/scans_testing/ON/ON1Rback/data/depth/0000000100.png"
    ),
    ("patient17",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/segmentation/subpixel-embedding-segmentation/evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient17.png",
     "/Users/primary/Documents/GitHub/cs-thesis/3d-ergonomics/data/scans_testing/ON/ON1Rback/data/depth/0000007100.png"
    ),
]

METHOD = "ransac"
def compute_scale_and_bias(method, sensor_inv, dense_vals):
    if method == "ransac":
        r = RANSACRegressor(random_state=0)
        r.fit(dense_vals, sensor_inv.ravel())
        s = r.estimator_.coef_[0]
        b = r.estimator_.intercept_
    elif method == "percentile":
        p0d, p90d = np.percentile(dense_vals,  [0, 90])
        p0s, p90s = np.percentile(sensor_inv,  [0, 90])
        s  = (p90s - p0s) / (p90d - p0d)
        b  = p0s - s * p0d
    elif method == "linear":
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(dense_vals, sensor_inv.ravel())
        s, b = lr.coef_[0], lr.intercept_
    else:  # median
        s = np.median(sensor_inv) / np.median(dense_vals)
        b = 0.0
    return s, b

def main():
    # init wrapper
    wrapper = DepthAnythingWrapper(
        model_weights_path=WEIGHTS_PATH,
        encoder=ENCODER
    )

    for pid, rgb_path, depth_path in CASES:
        print(f"\n→ {pid}")
        # build output path by swapping "depth" → "dense_depth"
        if "/depth/" not in depth_path:
            raise ValueError(f"expected '/depth/' in path: {depth_path}")
        out_path = depth_path.replace("/depth/", "/dense_depth/")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            print(f"   skip (already exists): {out_path}")
            continue
        if not os.path.exists(depth_path):
            print(f"   skip (no sensor depth): {depth_path}")
            continue

        print("   running depthanything...")
        dense_pred = wrapper.infer(rgb_path)

        sensor = data_utils.load_depth(depth_path)  # H×W float
        if sensor.shape != dense_pred.shape:
            print(f"   resizing pred {dense_pred.shape} → {sensor.shape}")
            dense_pred = cv2.resize(
                dense_pred,
                (sensor.shape[1], sensor.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        mask = (sensor > 0)
        if mask.sum() < 10:
            print("   too few valid points, skipping")
            continue

        inv_sensor = (1.0 / sensor[mask]).reshape(-1, 1)
        dense_vals = dense_pred[mask].reshape(-1, 1)

        s, b = compute_scale_and_bias(METHOD, inv_sensor, dense_vals)
        fused = 1.0 / (dense_pred * s + b)

        data_utils.save_depth(fused, out_path)
        print(f"   saved → {out_path}")
        print(f"   scale={s:.4f},  bias={b:.4f}")

if __name__ == "__main__":
    main()
