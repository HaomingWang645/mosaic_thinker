import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from romatch import roma_indoor
from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast

# Simple cache so repeated calls do not reload large models.
_MODEL_CACHE = {
    "device": None,
    "roma": None,
    "depth_model": None,
    "depth_processor": None,
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_models(device=None):
    """
    Lazily load RoMa and DepthPro models once per process.
    Returns (roma_model, depth_model, depth_processor, device).
    """
    if device is not None and not isinstance(device, torch.device):
        device = torch.device(device)
    if device is None:
        device = get_device()
    cached = _MODEL_CACHE["roma"]
    if cached is not None and _MODEL_CACHE["device"] == device:
        return (
            _MODEL_CACHE["roma"],
            _MODEL_CACHE["depth_model"],
            _MODEL_CACHE["depth_processor"],
            device,
        )

    roma_model = roma_indoor(device="cuda" if device.type == "cuda" else "cpu")
    depth_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    depth_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(
        device
    )

    _MODEL_CACHE["device"] = device
    _MODEL_CACHE["roma"] = roma_model
    _MODEL_CACHE["depth_model"] = depth_model
    _MODEL_CACHE["depth_processor"] = depth_processor
    return roma_model, depth_model, depth_processor, device


def load_image(path):
    return Image.open(path).convert("RGB")


def principal_point_from_image(img: Image.Image):
    w, h = img.size
    return (np.array([w, h]) - 1) / 2.0


def depth_single_frame(image: Image.Image, depth_model, depth_processor, device):
    inputs = depth_processor(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depth_model(**inputs)
    post = depth_processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]
    depth = post["predicted_depth"].detach().cpu().numpy()
    focal_length = float(post["focal_length"])
    return depth, focal_length


def to_discrete_indices_np(kpts, w, h):
    xs = np.clip(np.rint(kpts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(kpts[:, 1]).astype(int), 0, h - 1)
    return xs, ys


def matched_points_to_3d(kpts, depth_map, pp, f, image_size):
    w, h = image_size
    xs, ys = to_discrete_indices_np(kpts, w, h)
    depth_vals = depth_map[ys, xs]
    valid = depth_vals > 0
    xs = xs[valid]
    ys = ys[valid]
    depth_vals = depth_vals[valid]
    if len(depth_vals) == 0:
        return np.empty((0, 3))
    x3d = (xs - pp[0]) * depth_vals / f
    y3d = (ys - pp[1]) * depth_vals / f
    return np.stack([x3d, y3d, depth_vals], axis=1)


def paired_matches_to_3d(
    kpts0, kpts1, depth0, depth1, pp0, pp1, f0, f1, size0, size1
):
    """
    Convert matched pixel coords to paired 3D points using a shared validity mask.
    Returns pts0, pts1 with the same length/order.
    """
    w0, h0 = size0
    w1, h1 = size1
    x0, y0 = to_discrete_indices_np(kpts0, w0, h0)
    x1, y1 = to_discrete_indices_np(kpts1, w1, h1)
    z0 = depth0[y0, x0]
    z1 = depth1[y1, x1]
    mask = (z0 > 0) & (z1 > 0)
    if not np.any(mask):
        return np.empty((0, 3)), np.empty((0, 3))
    x0 = x0[mask]
    y0 = y0[mask]
    x1 = x1[mask]
    y1 = y1[mask]
    z0 = z0[mask]
    z1 = z1[mask]
    x3d_0 = (x0 - pp0[0]) * z0 / f0
    y3d_0 = (y0 - pp0[1]) * z0 / f0
    x3d_1 = (x1 - pp1[0]) * z1 / f1
    y3d_1 = (y1 - pp1[1]) * z1 / f1
    pts0 = np.stack([x3d_0, y3d_0, z0], axis=1)
    pts1 = np.stack([x3d_1, y3d_1, z1], axis=1)
    return pts0, pts1


def similarity_transform_3d(src, dst):
    """
    Umeyama similarity (scale+rot+trans) aligning src->dst. Returns 4x4 matrix.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    assert src.shape == dst.shape and src.shape[1] == 3
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    S = X.T @ Y / src.shape[0]
    U, _, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    var_X = (X**2).sum() / src.shape[0]
    s = np.trace(R @ S) / var_X
    t = mu_d - s * (R @ mu_s)
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def match_keypoints(img_path0, img_path1, roma_model, img0_size, img1_size, device):
    device_str = "cuda" if device.type == "cuda" else "cpu"
    warp, certainty = roma_model.match(img_path0, img_path1, device=device_str)
    matches, certainty = roma_model.sample(warp, certainty)
    # Use the same ordering as the notebook example (width, height).
    w0, h0 = img0_size
    w1, h1 = img1_size
    kpts0, kpts1 = roma_model.to_pixel_coordinates(matches, w0, h0, w1, h1)
    kpts0 = kpts0.cpu().numpy()
    kpts1 = kpts1.cpu().numpy()
    return kpts0, kpts1


def ensure_rgb(img_np):
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    if img_np.shape[2] == 4:
        img_np = img_np[..., :3]
    return img_np.astype(np.uint8)


def full_image_cloud(image_pil, depth, f, pp, step=1):
    img_np = ensure_rgb(np.array(image_pil))
    depth_np = np.array(depth)
    H, W = depth_np.shape
    ys, xs = np.meshgrid(
        np.arange(0, H, step), np.arange(0, W, step), indexing="ij"
    )
    z = depth_np[ys, xs]
    valid = z > 0
    xs = xs[valid].ravel()
    ys = ys[valid].ravel()
    z = z[valid].ravel()
    x3d = (xs - pp[0]) * z / f
    y3d = (ys - pp[1]) * z / f
    pts = np.stack([x3d, y3d, z], axis=1)
    cols = img_np[ys, xs]
    return pts, cols


def save_points(points, colors, file_path):
    points = np.asarray(points)
    colors = np.asarray(colors)
    if colors.max() < 1.1:
        colors = (colors * 255).astype(np.uint8)
    ply_header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(file_path, "w") as f:
        f.write(ply_header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")


def save_match_viz(img0, img1, kpts0, kpts1, out_path, max_lines=500):
    imA = np.array(img0)
    imB = np.array(img1)
    hA, wA = imA.shape[:2]
    hB, wB = imB.shape[:2]
    max_h = max(hA, hB)
    canvas = np.zeros((max_h, wA + wB, 3), dtype=np.uint8)
    canvas[:hA, :wA] = imA
    canvas[:hB, wA : wA + wB] = imB

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(canvas)
    ax.axis("off")

    kpts0_np = np.asarray(kpts0)
    kpts1_np = np.asarray(kpts1)
    kpts1_shifted = kpts1_np.copy()
    kpts1_shifted[:, 0] += wA

    num = min(len(kpts0_np), max_lines)
    if num > 0:
        idx = np.random.choice(len(kpts0_np), num, replace=False)
        for i in idx:
            ax.plot(
                [kpts0_np[i, 0], kpts1_shifted[i, 0]],
                [kpts0_np[i, 1], kpts1_shifted[i, 1]],
                c="lime",
                linewidth=0.5,
                alpha=0.7,
            )
            ax.scatter(kpts0_np[i, 0], kpts0_np[i, 1], c="lime", s=6)
            ax.scatter(kpts1_shifted[i, 0], kpts1_shifted[i, 1], c="lime", s=6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_alignment_viz(pts0, pts1_raw, pts1_aligned, out_path, max_points=50000):
    # Downsample for plotting if clouds are too dense.
    def sample_pts(pts, n):
        if len(pts) <= n:
            return pts
        idx = np.random.choice(len(pts), n, replace=False)
        return pts[idx]

    pts0_s = sample_pts(pts0, max_points)
    pts1_r = sample_pts(pts1_raw, max_points)
    pts1_a = sample_pts(pts1_aligned, max_points)

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(pts0_s[:, 0], pts0_s[:, 2], pts0_s[:, 1], s=0.5, c="blue", alpha=0.6)
    ax1.scatter(pts1_r[:, 0], pts1_r[:, 2], pts1_r[:, 1], s=0.5, c="red", alpha=0.4)
    ax1.set_title("Before alignment")
    ax1.invert_zaxis()

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(
        pts0_s[:, 0], pts0_s[:, 2], pts0_s[:, 1], s=0.5, c="blue", alpha=0.6
    )
    ax2.scatter(
        pts1_a[:, 0], pts1_a[:, 2], pts1_a[:, 1], s=0.5, c="green", alpha=0.5
    )
    ax2.set_title("After alignment")
    ax2.invert_zaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
