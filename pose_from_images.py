import argparse
import json
from pathlib import Path

import numpy as np

from pose_utils import (
    full_image_cloud,
    get_models,
    load_image,
    match_keypoints,
    paired_matches_to_3d,
    principal_point_from_image,
    save_alignment_viz,
    save_match_viz,
    save_points,
    similarity_transform_3d,
    depth_single_frame,
)


def compute_transform(img0_path, img1_path, out_dir, downsample=4, device_override=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roma_model, depth_model, depth_processor, device = get_models(device_override)

    img0 = load_image(img0_path)
    img1 = load_image(img1_path)
    w0, h0 = img0.size
    w1, h1 = img1.size

    pp0 = principal_point_from_image(img0)
    pp1 = principal_point_from_image(img1)

    depth0, f0 = depth_single_frame(img0, depth_model, depth_processor, device)
    depth1, f1 = depth_single_frame(img1, depth_model, depth_processor, device)

    kpts0, kpts1 = match_keypoints(
        img0_path, img1_path, roma_model, (w0, h0), (w1, h1), device
    )
    if len(kpts0) == 0:
        raise RuntimeError("No matches returned by RoMa.")

    pts0, pts1 = paired_matches_to_3d(
        kpts0, kpts1, depth0, depth1, pp0, pp1, f0, f1, (w0, h0), (w1, h1)
    )
    if len(pts0) < 3:
        raise RuntimeError("Not enough valid depth-backed matches for alignment.")

    T_BA = similarity_transform_3d(pts1, pts0)

    # Visuals
    save_match_viz(img0, img1, kpts0, kpts1, out_dir / "matches.png")

    pts0_full, cols0 = full_image_cloud(img0, depth0, f0, pp0, step=downsample)
    pts1_full, cols1 = full_image_cloud(img1, depth1, f1, pp1, step=downsample)
    pts1_aligned = (T_BA[:3, :3] @ pts1_full.T).T + T_BA[:3, 3]
    save_alignment_viz(
        pts0_full, pts1_full, pts1_aligned, out_dir / "alignment.png"
    )

    # Merge cloud for downstream inspection.
    merged_pts = np.vstack([pts0_full, pts1_aligned])
    merged_cols = np.vstack([cols0, cols1])
    save_points(merged_pts, merged_cols, out_dir / "cloud_merged.ply")

    # Save matrix to disk.
    np.save(out_dir / "transform.npy", T_BA)
    np.savetxt(out_dir / "transform.txt", T_BA, fmt="%.6f")
    with open(out_dir / "transform.json", "w") as f:
        json.dump({"transform_B_to_A": T_BA.tolist()}, f, indent=2)

    return T_BA


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate relative pose between two images using RoMa+DepthPro."
    )
    parser.add_argument("image0", type=Path, help="Path to first image")
    parser.add_argument("image1", type=Path, help="Path to second image")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Root output directory for results and visualizations",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Stride for cloud/plot generation (higher -> fewer points)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Force device; defaults to CUDA if available else CPU",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    img0 = args.image0
    img1 = args.image1
    if not img0.exists() or not img1.exists():
        raise FileNotFoundError("Input image path does not exist.")

    run_dir = args.out_dir / f"{img0.stem}__{img1.stem}"
    device_override = None
    if args.device is not None:
        device_override = (
            "cuda" if args.device == "cuda" else "cpu"
        )

    T = compute_transform(
        img0, img1, run_dir, downsample=args.downsample, device_override=device_override
    )
    np.set_printoptions(precision=4, suppress=True)
    print("Estimated T (B -> A):")
    print(T)


if __name__ == "__main__":
    main()
