#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import cv2
from PIL import Image

# Local imports
from vggt.dependency.vggsfm_utils import initialize_feature_extractors
import pycolmap
import trimesh


def _load_image_as_tensor_rgb01(image_path: str) -> torch.Tensor:
    """
    Load an image at its original resolution as a torch.FloatTensor in range [0, 1] with shape (3, H, W).
    """
    img = Image.open(image_path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def _extract_features(
    image_tensor: torch.Tensor,
    extractors: Dict[str, torch.nn.Module],
    device: torch.device,
) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Extract keypoints and descriptors using the provided extractors per modality.

    Returns a dict mapping extractor name -> (keypoints, descriptors)
        keypoints: (N, 2) float32 pixel coordinates (x, y)
        descriptors: (N, D) float32 descriptors or None if unavailable
    """
    image_tensor = image_tensor.to(device)
    feat_dict: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
    with torch.no_grad():
        for name, extractor in extractors.items():
            # LightGlue extractors typically provide .extract with keys: keypoints (1, N, 2), descriptors (1, N, C)
            out = extractor.extract(image_tensor, invalid_mask=None)
            kps = out.get("keypoints", None)
            desc = out.get("descriptors", None)
            if kps is None:
                continue
            # kps: (1, N, 2) or (N, 2)
            if kps.dim() == 3:
                kps_np = kps.squeeze(0).cpu().numpy()
            else:
                kps_np = kps.cpu().numpy()
            desc_np = None
            if desc is not None:
                # desc: (1, N, C) or (N, C) → stack along feature dim
                if desc.dim() == 3:
                    desc_np = desc.squeeze(0).cpu().numpy()
                else:
                    desc_np = desc.cpu().numpy()
                desc_np = desc_np.astype(np.float32)

            feat_dict[name] = (kps_np.astype(np.float32), desc_np)

    return feat_dict


def _mutual_nn_match(
    desc_q: np.ndarray,
    desc_d: np.ndarray,
    ratio: float = 0.8,
) -> np.ndarray:
    """
    Mutual nearest neighbor matching with Lowe's ratio test.

    Returns:
        matches: (M, 2) int array of index pairs (i_query, i_db)
    """
    if desc_q is None or desc_d is None or len(desc_q) == 0 or len(desc_d) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    # Compute distances (use dot product on normalized if desired). Here, use L2 distance.
    # Normalize descriptors to unit length for cosine-like distance stability
    eps = 1e-8
    q = desc_q / (np.linalg.norm(desc_q, axis=1, keepdims=True) + eps)
    d = desc_d / (np.linalg.norm(desc_d, axis=1, keepdims=True) + eps)

    # Compute pairwise distances via 1 - cosine similarity
    sim = q @ d.T  # (Nq, Nd)
    # For ratio test, we need ranking per query
    idx_sorted = np.argsort(-sim, axis=1)  # descending similarity
    best = idx_sorted[:, 0]
    if idx_sorted.shape[1] > 1:
        second = idx_sorted[:, 1]
        best_sim = sim[np.arange(sim.shape[0]), best]
        second_sim = sim[np.arange(sim.shape[0]), second]
        ratio_mask = best_sim >= (ratio * second_sim)
    else:
        ratio_mask = np.ones(sim.shape[0], dtype=bool)

    # Mutual check: ensure query->db and db->query are consistent
    # Compute db best back to query
    idx_sorted_d = np.argsort(-sim, axis=0)  # per db column
    best_back = idx_sorted_d[0, best]
    mutual = best_back == np.arange(sim.shape[0])

    final_mask = ratio_mask & mutual
    matches = np.stack([np.arange(sim.shape[0])[final_mask], best[final_mask]], axis=1).astype(np.int32)
    return matches


def _gather_2d3d_from_matches(
    matches: np.ndarray,
    kpts_q: np.ndarray,
    kpts_d: np.ndarray,
    recon_points2d_xy: np.ndarray,
    recon_points3d_xyz: np.ndarray,
    snap_px_thresh: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each matched db keypoint, snap to the nearest reconstruction 2D point of that db image
    within a pixel threshold, and collect its 3D point. Build 2D–3D correspondences for PnP.

    Inputs are all in the SAME image resolution as the reconstruction images.

    Returns:
        pts2d_query: (M, 2)
        pts3d_world: (M, 3)
    """
    if matches.size == 0 or kpts_d.size == 0 or recon_points2d_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    pts2d_query = []
    pts3d_world = []

    # Build a k-d tree on reconstruction 2D points for fast snapping
    # If scipy not available, do a simple brute-force for small sets
    # Here we do brute-force which is fine for typical sizes
    for qi, di in matches:
        db_xy = kpts_d[di]
        # Find nearest recon 2D
        diff = recon_points2d_xy - db_xy[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        j = np.argmin(dist2)
        dist = np.sqrt(dist2[j])
        if dist <= snap_px_thresh:
            pts2d_query.append(kpts_q[qi])
            pts3d_world.append(recon_points3d_xyz[j])

    if len(pts2d_query) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    return np.asarray(pts2d_query, dtype=np.float32), np.asarray(pts3d_world, dtype=np.float32)


def _median_intrinsics_from_recon(recon: pycolmap.Reconstruction) -> np.ndarray:
    """
    Compute a median intrinsics matrix K from all cameras in the reconstruction.
    Assumes PINHOLE-like models and same camera across frames.
    """
    fx_list = []
    fy_list = []
    cx_list = []
    cy_list = []
    for cam_id in recon.cameras:
        cam = recon.cameras[cam_id]
        K = cam.calibration_matrix()
        fx_list.append(K[0, 0])
        fy_list.append(K[1, 1])
        cx_list.append(K[0, 2])
        cy_list.append(K[1, 2])
    fx = float(np.median(fx_list))
    fy = float(np.median(fy_list))
    cx = float(np.median(cx_list))
    cy = float(np.median(cy_list))
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def _project_points3d(E_3x4: np.ndarray, K_3x3: np.ndarray, pts3d_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points into the image using extrinsics (cam_from_world [R|t]) and intrinsics K.

    Returns:
        uv: (M, 2) projected pixel coordinates
        mask: (M,) boolean mask of valid points (z>0)
    """
    # Convert to homogeneous world points
    P = pts3d_xyz.astype(np.float64)
    N = P.shape[0]
    Pw = np.hstack([P, np.ones((N, 1), dtype=np.float64)])  # (N,4)
    # Camera coordinates: Xc = E * Xw_h
    Xc = (E_3x4 @ Pw.T)  # (3,N)
    z = Xc[2, :]
    valid = z > 1e-6
    Xc_valid = Xc[:, valid]
    # Normalize
    Xn = Xc_valid / Xc_valid[2:3, :]
    # Pixels: u = K * Xn
    uv_h = (K_3x3 @ Xn)  # (3,M)
    uv = uv_h[:2, :].T.astype(np.float32)  # (M,2)
    return uv, valid


def localize_query_image(
    scene_dir: str,
    query_image_path: str,
    extractor_method: str = "aliked+sp",
    top_k_db: int = 20,
    min_pnp_inliers: int = 64,
    px_thresh_snap: float = 3.0,
    ransac_reproj_error: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Localize a query image against an existing COLMAP reconstruction using ALIKED+SuperPoint features.

    Returns:
        extrinsic_cam_from_world: (3,4)
        intrinsics_K: (3,3)
        debug: dict with diagnostic arrays
    """
    sparse_dir = os.path.join(scene_dir, "sparse")
    if not os.path.isdir(sparse_dir):
        raise FileNotFoundError(f"No sparse reconstruction found at {sparse_dir}")

    recon = pycolmap.Reconstruction(sparse_dir)

    # Build database list and 2D-3D mapping (registered 2D->3D)
    db_image_items = []  # (image_id, image_path, points2D_xy, points3D_xyz)
    images_dir = os.path.join(scene_dir, "images")

    for image_id in recon.images:
        img = recon.images[image_id]
        name = img.name
        image_path = os.path.join(images_dir, name)
        points2d_xy = []
        points3d_xyz = []
        for p2d in img.points2D:
            pid = p2d.point3D_id
            if pid == -1:
                continue
            xy = p2d.xy
            points2d_xy.append(np.array([xy[0], xy[1]], dtype=np.float32))
            p3d = recon.points3D[pid]
            points3d_xyz.append(p3d.xyz.astype(np.float32))
        if len(points2d_xy) > 0:
            db_image_items.append((image_id, image_path, np.stack(points2d_xy, axis=0), np.stack(points3d_xyz, axis=0)))

    # Load query image
    query_tensor = _load_image_as_tensor_rgb01(query_image_path)

    # Initialize extractors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractors = initialize_feature_extractors(4096, extractor_method=extractor_method, device=device)

    # Extract query features
    feats_q = _extract_features(query_tensor, extractors, device)

    # Choose top-K database images (fallback to all if K<=0)
    all_image_ids = list(recon.images.keys())
    ordered_ids = all_image_ids[: top_k_db if top_k_db > 0 else len(all_image_ids)]

    pts2d_all: List[np.ndarray] = []
    pts3d_all: List[np.ndarray] = []

    # If we have no registered 2D-3D, fall back to projection-based 2D-3D building
    use_projection_fallback = len(db_image_items) == 0

    # Precompute points3D for projection fallback
    proj_pts3d = None
    if use_projection_fallback:
        # Collect all points3D
        pts_list = []
        for pid in recon.points3D:
            pts_list.append(recon.points3D[pid].xyz.astype(np.float32))
        if not pts_list:
            # Fallback: try loading points from points.ply
            ply_path = os.path.join(sparse_dir, "points.ply")
            if not os.path.exists(ply_path):
                raise RuntimeError("Reconstruction has no 3D points; cannot localize.")
            cloud = trimesh.load(ply_path)
            if not hasattr(cloud, "vertices") or cloud.vertices is None or len(cloud.vertices) == 0:
                raise RuntimeError("points.ply has no vertices; cannot localize.")
            proj_pts3d = np.asarray(cloud.vertices, dtype=np.float32)
        else:
            proj_pts3d = np.stack(pts_list, axis=0)
        # Randomly subsample to keep memory reasonable
        if proj_pts3d.shape[0] > 200000:
            sel = np.random.choice(proj_pts3d.shape[0], 200000, replace=False)
            proj_pts3d = proj_pts3d[sel]

    # Iterate DB images
    for image_id in ordered_ids:
        img = recon.images[image_id]
        name = img.name
        image_path = os.path.join(images_dir, name)

        # Load DB image and extract features
        db_tensor = _load_image_as_tensor_rgb01(image_path)
        feats_d = _extract_features(db_tensor, extractors, device)

        # Intersect extractor keys and aggregate matches per extractor
        common_keys = set(feats_q.keys()).intersection(set(feats_d.keys()))
        if not common_keys:
            continue

        per_image_pts2d = []
        per_image_pts3d = []

        for key in common_keys:
            kpts_q_mod, desc_q_mod = feats_q[key]
            kpts_d_mod, desc_d_mod = feats_d[key]
            if kpts_d_mod.shape[0] == 0 or kpts_q_mod.shape[0] == 0:
                continue
            matches = _mutual_nn_match(desc_q_mod, desc_d_mod, ratio=0.8)
            if matches.shape[0] == 0:
                continue

            if not use_projection_fallback:
                # Use registered tracks
                item = next((it for it in db_image_items if it[0] == image_id), None)
                if item is None:
                    continue
                _, _, recon_xy, recon_xyz = item
                pts2d_q, pts3d = _gather_2d3d_from_matches(
                    matches, kpts_q_mod, kpts_d_mod, recon_xy, recon_xyz, snap_px_thresh=px_thresh_snap
                )
            else:
                # Projection fallback
                K_cam = recon.cameras[img.camera_id].calibration_matrix().astype(np.float32)
                E = img.cam_from_world.matrix()[:3, :].astype(np.float32)  # 3x4
                uv, valid_mask = _project_points3d(E, K_cam, proj_pts3d)
                width = recon.cameras[img.camera_id].width
                height = recon.cameras[img.camera_id].height
                in_bounds = (
                    (uv[:, 0] >= 0)
                    & (uv[:, 0] < width)
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < height)
                )
                uv = uv[in_bounds]
                pts3d_proj = proj_pts3d[valid_mask][in_bounds]
                if uv.shape[0] == 0:
                    continue
                pts2d_q, pts3d = _gather_2d3d_from_matches(
                    matches, kpts_q_mod, kpts_d_mod, uv, pts3d_proj, snap_px_thresh=px_thresh_snap
                )

            if pts2d_q.shape[0] > 0:
                per_image_pts2d.append(pts2d_q)
                per_image_pts3d.append(pts3d)

        if per_image_pts2d:
            pts2d_all.append(np.concatenate(per_image_pts2d, axis=0))
            pts3d_all.append(np.concatenate(per_image_pts3d, axis=0))

    if not pts2d_all:
        raise RuntimeError("No 2D–3D correspondences found. Try increasing top_k_db, using 'aliked+sp+sift', or relaxing px_thresh_snap.")

    pts2d = np.concatenate(pts2d_all, axis=0)
    pts3d = np.concatenate(pts3d_all, axis=0)

    # Intrinsics for query: assume same camera as reconstruction → median K
    K = _median_intrinsics_from_recon(recon)

    # Run PnP RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=pts3d.astype(np.float32),
        imagePoints=pts2d.astype(np.float32),
        cameraMatrix=K.astype(np.float32),
        distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=float(ransac_reproj_error),
        confidence=0.999,
        iterationsCount=10000,
    )

    if not success or inliers is None or inliers.shape[0] < min_pnp_inliers:
        raise RuntimeError(f"PnP failed or insufficient inliers: {0 if inliers is None else inliers.shape[0]} found.")

    R, _ = cv2.Rodrigues(rvec)
    extrinsic = np.hstack([R.astype(np.float32), tvec.reshape(3, 1).astype(np.float32)])

    debug = {
        "num_corr": np.array([pts2d.shape[0]], dtype=np.int32),
        "num_inliers": np.array([inliers.shape[0]], dtype=np.int32),
    }

    return extrinsic, K, debug


def main():
    parser = argparse.ArgumentParser(description="Localize a query image against a VGGT/pycolmap reconstruction.")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory with images/ and sparse/ from demo_colmap.py")
    parser.add_argument("--query_image", type=str, required=True, help="Path to query image to localize")
    parser.add_argument("--extractor", type=str, default="aliked+sp", help="Feature extractor combo, e.g., 'aliked+sp' or 'aliked+sp+sift'")
    parser.add_argument("--top_k_db", type=int, default=20, help="Number of database images to match")
    parser.add_argument("--px_thresh_snap", type=float, default=3.0, help="Max px distance to snap db keypoint to recon 2D point")
    parser.add_argument("--ransac_reproj_error", type=float, default=8.0, help="RANSAC reprojection error in pixels")
    parser.add_argument("--min_pnp_inliers", type=int, default=64, help="Minimum inliers to accept PnP solution")
    args = parser.parse_args()

    extrinsic, K, dbg = localize_query_image(
        scene_dir=args.scene_dir,
        query_image_path=args.query_image,
        extractor_method=args.extractor,
        top_k_db=args.top_k_db,
        px_thresh_snap=args.px_thresh_snap,
        ransac_reproj_error=args.ransac_reproj_error,
        min_pnp_inliers=args.min_pnp_inliers,
    )

    print("Localization successful.")
    print("Estimated extrinsic (cam_from_world [R|t]):")
    print(extrinsic)
    print("Estimated intrinsics K:")
    print(K)
    print(f"Num correspondences: {int(dbg['num_corr'][0])}, inliers: {int(dbg['num_inliers'][0])}")


if __name__ == "__main__":
    with torch.no_grad():
        main() 