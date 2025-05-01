#!/usr/bin/env python3
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2
import argparse
import sys
import torch
import torch.multiprocessing as mp
from sklearn.linear_model import RANSACRegressor, LinearRegression
import numpy as np

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


def compute_scale_and_bias(method, sensor_valid_inv, dense_valid, debug=True):
    if method == "ransac":
        ransac = RANSACRegressor(random_state=0)
        ransac.fit(dense_valid, sensor_valid_inv.ravel())
        scale = ransac.estimator_.coef_[0]
        bias  = ransac.estimator_.intercept_
        if debug:
            print(f"RANSAC fit for example -> scale: {scale:.3f}, bias: {bias:.3f}")
    elif method == "percentile":
        p0_d, p90_d = np.percentile(dense_valid,  [0, 90])
        p0_s, p90_s = np.percentile(sensor_valid_inv, [0, 90])
        
        print(f"p25_d: {p0_d:.3f}, p75_d: {p90_d:.3f}")
        print(f"p25_s: {p0_s:.3f}, p75_s: {p90_s:.3f}")
        
        # solve:  real ≈ scale·dense + bias
        scale = (p90_s - p0_s) / (p90_d - p0_d)
        bias  = p0_s - scale * p0_d

        if debug:
            print(f"Percentile fit → scale: {scale:.3f}, bias: {bias:.3f}")
    elif method == "median":
        scale = np.median(sensor_valid_inv) / np.median(dense_valid)
        bias  = 0.0
        
        if debug:
            print(f"Median fit for example -> scale: {scale:.3f}, bias: {bias:.3f}")
    elif method == "linear":
        linear = LinearRegression()
        linear.fit(dense_valid, sensor_valid_inv.ravel())
        scale = linear.coef_[0]
        bias  = linear.intercept_
        if debug:
            print(f"Linear fit for example -> scale: {scale:.3f}, bias: {bias:.3f}")
        
    return scale, bias

def process_shard(gpu_id: int, args):
    """
    Worker function for GPU `gpu_id`: scans all image files under args.data3d_root,
    and processes only those whose file_counter % args.num_gpus == gpu_id.
    """
    # Make only this GPU visible in this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Worker {gpu_id}] running on {device_name}")

    # initialize wrapper here so it picks the right device
    wrapper = DepthAnythingWrapper(
        model_weights_path=args.weights_path,
        encoder=args.encoder
    )

    file_counter = 0
    for root, dirs, _ in os.walk(args.data3d_root):
        if "data" not in dirs:
            continue

        data_dir  = os.path.join(root, "data")
        image_dir = os.path.join(data_dir, "image")
        depth_dir = os.path.join(data_dir, "depth")
        if not (os.path.isdir(image_dir) and os.path.isdir(depth_dir)):
            continue

        # ensure output folder exists
        dense_dir = os.path.join(data_dir, "dense_depth")
        os.makedirs(dense_dir, exist_ok=True)

        for fname in sorted(os.listdir(image_dir)):
            if not fname.lower().endswith(".png"):
                continue

            idx = file_counter
            file_counter += 1

            if idx % args.num_gpus != gpu_id:
                continue

            img_path   = os.path.join(image_dir, fname)
            depth_path = os.path.join(depth_dir, fname)
            out_path   = os.path.join(dense_dir, fname)

            if os.path.exists(out_path) and not args.override:
                print(f"[{gpu_id}] skip (exists): {out_path}", flush=True)
                continue
            if not os.path.exists(depth_path):
                print(f"[{gpu_id}] skip (no depth): {depth_path}", flush=True)
                continue

            print(f"[{gpu_id}] processing {img_path}")
            try:
                dense_pred = wrapper.infer(img_path)
            except Exception as e:
                print(f"[{gpu_id}] ERROR infer {img_path}: {e}", flush=True)
                continue

            try:
                sensor_depth = data_utils.load_depth(depth_path)
            except Exception as e:
                print(f"[{gpu_id}] ERROR load depth {depth_path}: {e}", flush=True)
                continue
            # Resize the dense prediction to match sensor depth shape if necessary
            if sensor_depth.shape != dense_pred.shape:
                print(f"Resizing dense prediction from {dense_pred.shape} to {sensor_depth.shape}")
                dense_pred = cv2.resize(dense_pred, (sensor_depth.shape[1], sensor_depth.shape[0]))

            valid_mask = sensor_depth > 0
            if valid_mask.sum() < 10:
                print(f"[{gpu_id}] too few valid points in {img_path}", flush=True)
                continue

            sensor_inv = 1.0 / sensor_depth[valid_mask]
            sensor_valid_inv = sensor_inv.reshape(-1, 1)
            dense_valid_mask = dense_pred[valid_mask]
            dense_valid       = dense_valid_mask.reshape(-1, 1)
            
            scale, bias = compute_scale_and_bias(
                method="ransac",
                sensor_valid_inv=sensor_valid_inv,
                dense_valid=dense_valid,
                debug=False
            )

            fused_dense = 1.0 / (dense_pred * scale + bias)

            data_utils.save_depth(fused_dense, out_path)
            print(f"[{gpu_id}] saved {out_path} with scale {scale:.4f} and bias {bias:.4f}", flush=True)

    print(f"[Worker {gpu_id}] done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed per-file depth fusion with DepthAnything"
    )
    parser.add_argument(
        "--encoder", type=str, default="vitb",
        help="which encoder to use for DepthAnything"
    )
    parser.add_argument(
        "--weights_path", type=str, required=True,
        help="path to your depth_anything .pth file"
    )
    parser.add_argument(
        "--data3d_root", type=str, required=True,
        help="root folder under which '*/data/image' & '*/data/depth' live"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=None,
        help="override torch.cuda.device_count()"
    )
    parser.add_argument(
        "--shard", type=int, default=-1,
        help="if >=0, run only that single shard (GPU id); else auto-spawn on all GPUs"
    )
    parser.add_argument(
        "--override", action="store_true",
        help="if set, override existing dense depth files"
    )

    args = parser.parse_args()

    # detect GPU count
    available = torch.cuda.device_count() or 0
    args.num_gpus = args.num_gpus or available or 1
    print(f"Detected {available} CUDA devices; using num_gpus={args.num_gpus}")

    # If no manual shard, and multiple GPUs, spawn one worker per GPU
    if args.shard < 0 and args.num_gpus > 1:
        mp.spawn(
            process_shard,
            args=(args,),
            nprocs=args.num_gpus,
            join=True
        )
    else:
        # single‐GPU or manual override
        gpu_id = 0 if args.shard < 0 else args.shard
        if gpu_id >= args.num_gpus:
            raise ValueError(f"shard {gpu_id} >= num_gpus {args.num_gpus}")
        process_shard(gpu_id, args)
