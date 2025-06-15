#!/usr/bin/env python3

import cv2
import numpy as np
import os
import argparse

def apply_yaw_projection(image, yaw_deg, fov_deg=90):
    h, w = image.shape[:2]
    f = w / (2 * np.tan(np.deg2rad(fov_deg / 2)))  # 焦点距離

    # 3D -> 2D 射影行列を作成する
    K = np.array([
        [f, 0, w / 2],
        [0, f, h / 2],
        [0, 0,     1]
    ])

    yaw = np.deg2rad(yaw_deg)
    R = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0,           1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # 3D -> 2D 射影変換 H = K * R * K^-1
    K_inv = np.linalg.inv(K)
    H = K @ R @ K_inv
    warped = cv2.warpPerspective(image, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--fov", type=float, default=90.0, help="Horizontal FOV in degrees (default: 90)")
    args = parser.parse_args()

    image_path = os.path.expanduser(args.image_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    for yaw in [-5, 0, 5]:
        out_img = apply_yaw_projection(img, yaw_deg=yaw, fov_deg=args.fov)
        out_path = f"yaw_{yaw:+d}deg.png"
        cv2.imwrite(out_path, out_img)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
