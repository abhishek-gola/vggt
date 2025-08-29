#!/usr/bin/env python3
# Copyright (c) Meta Platforms...
# All rights reserved.

import argparse
import os
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import torch
import cv2
from PIL import Image

# Local imports
from vggt.dependency.vggsfm_utils import initialize_feature_extractors
import pycolmap
import trimesh
from vggt.utils.rotation import mat_to_quat


# ----------------------------- Utilities ---------------------------------- #

def _pil_open_rgb(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert("RGB")


def _tensor_from_pil_rgb01(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W), [0,1]


def _resize_long_side_keep_aspect(img: Image.Image, max_long_side: int) -> Tuple[Image.Image, float, float]:
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_long_side:
        return img, 1.0, 1.0
    scale = max_long_side / long_side
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    return img_resized, (w / new_w), (h / new_h)  # return scale-back factors


def _prepare_image_for_extractor(image_path: str, max_long_side: int) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[float, float]]:
    """
    Load PIL image, resize (keeping aspect) so that max(H,W) <= max_long_side,
    return tensor (3,H',W') in [0,1], original size (W,H), and scale-back factors (sx, sy)
    so that kpts_in_resized * [sx,sy] -> kpts_in_original.
    """
    img = _pil_open_rgb(image_path)
    orig_w, orig_h = img.size
    img_r, sx, sy = _resize_long_side_keep_aspect(img, max_long_side)
    tensor = _tensor_from_pil_rgb01(img_r)
    return tensor, (orig_w, orig_h), (sx, sy)


def _extract_features_with_scaling(
    image_tensor_resized: torch.Tensor,
    extractors: Dict[str, torch.nn.Module],
    device: torch.device,
    scale_back: Tuple[float, float],
) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Run extractor on the resized tensor (BCHW), then scale keypoints back to original pixel frame.
    Returns dict[name] = (kpts_xy_in_original_frame, desc or None)
    """
    sx, sy = scale_back
    image_tensor_resized = image_tensor_resized.to(device)
    if image_tensor_resized.dim() == 3:
        image_tensor_resized = image_tensor_resized.unsqueeze(0)  # (1,3,H,W)

    feat_dict: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
    with torch.no_grad():
        for name, extractor in extractors.items():
            out = extractor.extract(image_tensor_resized, invalid_mask=None)
            kps = out.get("keypoints", None)
            desc = out.get("descriptors", None)
            if kps is None:
                continue

            # Expect either (1,N,2) or (N,2)
            if kps.dim() == 3:
                k = kps.squeeze(0).detach().cpu().numpy()
            else:
                k = kps.detach().cpu().numpy()

            # Scale back to ORIGINAL image resolution
            if k.size > 0:
                k[:, 0] *= sx
                k[:, 1] *= sy
            k = k.astype(np.float32)

            d_np = None
            if desc is not None:
                if desc.dim() == 3:
                    d_np = desc.squeeze(0).detach().cpu().numpy().astype(np.float32)
                else:
                    d_np = desc.detach().cpu().numpy().astype(np.float32)

            feat_dict[name] = (k, d_np)

    return feat_dict


def _mutual_nn_match_cosine_ratio_on_distance(
    desc_q: np.ndarray,
    desc_d: np.ndarray,
    ratio: float = 0.8,
) -> np.ndarray:
    """
    Mutual NN using cosine similarity but applying Lowe ratio on the equivalent distances.
    Handles pathological negatives by operating in distance domain.
    Returns (M,2) int indices.
    """
    if desc_q is None or desc_d is None or len(desc_q) == 0 or len(desc_d) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    eps = 1e-8
    q = desc_q / (np.linalg.norm(desc_q, axis=1, keepdims=True) + eps)
    d = desc_d / (np.linalg.norm(desc_d, axis=1, keepdims=True) + eps)
    sim = q @ d.T  # (Nq,Nd)
    # Convert to distances where smaller is better: cosine distance = 1 - cos_sim in [0,2]
    dist = 1.0 - sim

    # For each query, get best and second-best distances
    idx_sorted = np.argsort(dist, axis=1)  # ascending distance
    best = idx_sorted[:, 0]
    has_second = idx_sorted.shape[1] > 1
    if has_second:
        second = idx_sorted[:, 1]
        best_d = dist[np.arange(dist.shape[0]), best]
        second_d = dist[np.arange(dist.shape[0]), second]
        # classic Lowe: best_d <= ratio * second_d
        ratio_mask = best_d <= (ratio * second_d)
    else:
        ratio_mask = np.ones(dist.shape[0], dtype=bool)

    # Mutual check in distance space (column-wise argmin)
    idx_sorted_d = np.argsort(dist, axis=0)  # for each db, queries ranked
    best_back = idx_sorted_d[0, best]
    mutual = best_back == np.arange(dist.shape[0])

    final_mask = ratio_mask & mutual
    if not np.any(final_mask):
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([np.arange(dist.shape[0])[final_mask], best[final_mask]], axis=1).astype(np.int32)


def _gather_2d3d_from_matches(
    matches: np.ndarray,
    kpts_q: np.ndarray,
    kpts_d: np.ndarray,
    recon_points2d_xy: np.ndarray,
    recon_points3d_xyz: np.ndarray,
    snap_px_thresh: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Snap DB keypoints to nearest registered 2D track and return query 2D ↔ world 3D correspondences.
    All coordinates MUST be in the ORIGINAL db image pixel frame (which we now ensure).
    """
    if matches.size == 0 or kpts_d.size == 0 or recon_points2d_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    pts2d_query = []
    pts3d_world = []

    # Brute-force nearest neighbor (OK at typical sizes)
    for qi, di in matches:
        db_xy = kpts_d[di]
        diff = recon_points2d_xy - db_xy[None, :]
        j = np.argmin((diff * diff).sum(axis=1))
        dist = np.sqrt(((recon_points2d_xy[j] - db_xy) ** 2).sum())
        if dist <= snap_px_thresh:
            pts2d_query.append(kpts_q[qi])
            pts3d_world.append(recon_points3d_xyz[j])

    if not pts2d_query:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    return np.asarray(pts2d_query, dtype=np.float32), np.asarray(pts3d_world, dtype=np.float32)


def _median_intrinsics_from_subset(recon: pycolmap.Reconstruction, image_ids: List[int]) -> np.ndarray:
    """
    Compute median K using only the provided subset of images (e.g., retrieved top-K),
    which is more robust than global-median across the whole scene.
    """
    fx_list = []
    fy_list = []
    cx_list = []
    cy_list = []
    for image_id in image_ids:
        cam = recon.cameras[recon.images[image_id].camera_id]
        K = cam.calibration_matrix()
        fx_list.append(K[0, 0])
        fy_list.append(K[1, 1])
        cx_list.append(K[0, 2])
        cy_list.append(K[1, 2])
    fx = float(np.median(fx_list))
    fy = float(np.median(fy_list))
    cx = float(np.median(cx_list))
    cy = float(np.median(cy_list))
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _project_points3d(E_3x4: np.ndarray, K_3x3: np.ndarray, pts3d_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    P = pts3d_xyz.astype(np.float64)
    N = P.shape[0]
    Pw = np.hstack([P, np.ones((N, 1), dtype=np.float64)])
    Xc = (E_3x4 @ Pw.T)
    z = Xc[2, :]
    valid = z > 1e-6
    Xn = Xc[:, valid] / Xc[2:3, valid]
    uv_h = (K_3x3 @ Xn)
    uv = uv_h[:2, :].T.astype(np.float32)
    return uv, valid


# ------------------- Lightweight image retrieval (pHash) ------------------- #

def _phash(image_path: str, hash_size: int = 16) -> np.ndarray:
    """
    Tiny perceptual hash (DCT) for retrieval. Returns an L2-normalized vector.
    """
    img = _pil_open_rgb(image_path)
    img = img.convert("L").resize((hash_size*4, hash_size*4), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    dct = cv2.dct(arr)
    dct_low = dct[:hash_size, :hash_size].flatten()
    v = dct_low - dct_low.mean()
    n = np.linalg.norm(v) + 1e-8
    return (v / n).astype(np.float32)


def _retrieve_top_k_by_phash(query_path: str, db_paths: List[Tuple[int, str]], k: int) -> List[int]:
    """
    db_paths: list of (image_id, path). Returns a list of selected image_ids.
    """
    qv = _phash(query_path)
    sims = []
    for iid, p in db_paths:
        dv = _phash(p)
        sims.append((iid, float(qv @ dv)))  # cosine since already L2-normalized
    sims.sort(key=lambda x: x[1], reverse=True)
    return [iid for iid, _ in sims[:k]] if k > 0 else [iid for iid, _ in sims]


# ------------------------------ Main logic --------------------------------- #

def localize_query_image(
    scene_dir: str,
    query_image_path: str,
    extractor_method: str = "aliked+sp",
    top_k_db: int = 20,
    min_pnp_inliers: int = 64,
    px_thresh_snap: float = 3.0,
    ransac_reproj_error: float = 8.0,
    extractor_max_long_side: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Localize a query image against an existing COLMAP reconstruction.

    Key fixes:
    - We control resizing and SCALE KEYPOINTS BACK to the ORIGINAL pixel frame.
    - Use perceptual-hash retrieval to actually pick top-K neighbors.
    - Ratio test applied in distance domain for cosine-based descriptors.
    - Intrinsics estimated from the subset of retrieved db images.
    """
    sparse_dir = os.path.join(scene_dir, "sparse")
    if not os.path.isdir(sparse_dir):
        raise FileNotFoundError(f"No sparse reconstruction found at {sparse_dir}")

    recon = pycolmap.Reconstruction(sparse_dir)

    images_dir = os.path.join(scene_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"No images/ folder found at {images_dir}")

    # Build fast map: image_id -> (name, path, registered 2D-3D)
    id_to_item = {}
    db_id_and_paths = []
    for image_id, img in recon.images.items():
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
            id_to_item[image_id] = (image_path, np.stack(points2d_xy, axis=0), np.stack(points3d_xyz, axis=0))
        db_id_and_paths.append((image_id, image_path))

    # Load query in ORIGINAL size and also prepare resized tensor for extractor
    q_tensor_resized, (qW, qH), (q_sx, q_sy) = _prepare_image_for_extractor(query_image_path, extractor_max_long_side)

    # Initialize extractors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractors = initialize_feature_extractors(extractor_max_long_side, extractor_method=extractor_method, device=device)

    # Extract query features (and scale them back to original pixel frame)
    feats_q = _extract_features_with_scaling(q_tensor_resized, extractors, device, (q_sx, q_sy))

    # Choose top-K database images using lightweight retrieval
    retrieved_ids = _retrieve_top_k_by_phash(query_image_path, db_id_and_paths, top_k_db)
    if top_k_db <= 0 or len(retrieved_ids) == 0:
        retrieved_ids = [iid for iid, _ in db_id_and_paths]

    pts2d_all: List[np.ndarray] = []
    pts3d_all: List[np.ndarray] = []

    # Check if we have registered 2D-3D tracks at all
    use_projection_fallback = (len(id_to_item) == 0)

    # Precompute 3D points for fallback projection
    proj_pts3d = None
    if use_projection_fallback:
        pts_list = [recon.points3D[pid].xyz.astype(np.float32) for pid in recon.points3D]
        if not pts_list:
            ply_path = os.path.join(sparse_dir, "points.ply")
            if not os.path.exists(ply_path):
                raise RuntimeError("Reconstruction has no 3D points; cannot localize.")
            cloud = trimesh.load(ply_path)
            if not hasattr(cloud, "vertices") or cloud.vertices is None or len(cloud.vertices) == 0:
                raise RuntimeError("points.ply has no vertices; cannot localize.")
            proj_pts3d = np.asarray(cloud.vertices, dtype=np.float32)
        else:
            proj_pts3d = np.stack(pts_list, axis=0)
        if proj_pts3d.shape[0] > 200000:
            sel = np.random.choice(proj_pts3d.shape[0], 200000, replace=False)
            proj_pts3d = proj_pts3d[sel]

    # Iterate DB images
    for image_id in retrieved_ids:
        img = recon.images[image_id]
        image_path = os.path.join(images_dir, img.name)

        # Prepare DB tensor (resized for extractor) and scale factors back to ORIGINAL DB pixel frame
        db_tensor_resized, (dbW, dbH), (db_sx, db_sy) = _prepare_image_for_extractor(image_path, extractor_max_long_side)
        feats_d = _extract_features_with_scaling(db_tensor_resized, extractors, device, (db_sx, db_sy))

        # Intersect modalities and match
        common_keys = set(feats_q.keys()).intersection(set(feats_d.keys()))
        if not common_keys:
            continue

        per_image_pts2d = []
        per_image_pts3d = []

        # Either use registered tracks or do projection fallback
        if not use_projection_fallback and (image_id in id_to_item):
            recon_xy, recon_xyz = id_to_item[image_id][1], id_to_item[image_id][2]
            have_tracks = True
        else:
            have_tracks = False

        for key in common_keys:
            kpts_q_mod, desc_q_mod = feats_q[key]
            kpts_d_mod, desc_d_mod = feats_d[key]
            if kpts_d_mod.shape[0] == 0 or kpts_q_mod.shape[0] == 0:
                continue

            matches = _mutual_nn_match_cosine_ratio_on_distance(desc_q_mod, desc_d_mod, ratio=0.8)
            if matches.shape[0] == 0:
                continue

            if have_tracks:
                pts2d_q, pts3d = _gather_2d3d_from_matches(
                    matches, kpts_q_mod, kpts_d_mod, recon_xy, recon_xyz, snap_px_thresh=px_thresh_snap
                )
            else:
                # Fallback: project global 3D into this DB view, then snap
                K_cam = recon.cameras[img.camera_id].calibration_matrix().astype(np.float32)
                E = img.cam_from_world.matrix()[:3, :].astype(np.float32)  # 3x4
                uv, valid_mask = _project_points3d(E, K_cam, proj_pts3d)
                width = recon.cameras[img.camera_id].width
                height = recon.cameras[img.camera_id].height
                in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
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
        raise RuntimeError(
            "No 2D–3D correspondences found. Try increasing --top_k_db, using 'aliked+sp+sift', "
            "or relaxing --px_thresh_snap. Also verify your scene has registered points."
        )

    pts2d = np.concatenate(pts2d_all, axis=0)
    pts3d = np.concatenate(pts3d_all, axis=0)

    # Intrinsics for query: prefer dominant camera among retrieved images, else median over retrieved
    retrieved_cam_ids = [recon.images[iid].camera_id for iid in retrieved_ids]
    cam_count = Counter(retrieved_cam_ids)
    dom_cam_id, _ = cam_count.most_common(1)[0]
    Ks = []
    for iid in retrieved_ids:
        if recon.images[iid].camera_id == dom_cam_id:
            Ks.append(recon.cameras[dom_cam_id].calibration_matrix())
    if Ks:
        K = np.median(np.stack(Ks, axis=0), axis=0).astype(np.float32)
    else:
        K = _median_intrinsics_from_subset(recon, retrieved_ids)

    # PnP RANSAC
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


# New helper to convert extrinsic [R|t] (cam_from_world) to (qw, qx, qy, qz, tx, ty, tz)
def _extrinsic_to_qwxyz_txyz(extrinsic: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    R = extrinsic[:, :3].astype(np.float32)
    t = extrinsic[:, 3].astype(np.float32)
    # mat_to_quat returns (x, y, z, w) with scalar last; convert to scalar-first (w, x, y, z)
    quat_xyzw = mat_to_quat(torch.from_numpy(R)[None, ...]).detach().cpu().numpy()[0]
    qwxyz = (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))
    return (*qwxyz, float(t[0]), float(t[1]), float(t[2]))


# ------------------------------ Main logic --------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Localize a query image against a VGGT/pycolmap reconstruction.")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory with images/ and sparse/ from demo_colmap.py")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query_image", type=str, help="Path to a single query image to localize")
    group.add_argument("--query_dir", type=str, help="Path to a directory of query images to localize")
    parser.add_argument("--output_txt", type=str, default="poses.txt", help="Output txt file to write poses as: name qw qx qy qz tx ty tz")
    parser.add_argument("--extractor", type=str, default="aliked+sp", help="Feature extractor combo, e.g., 'aliked+sp' or 'aliked+sp+sift'")
    parser.add_argument("--top_k_db", type=int, default=20, help="Number of database images to match (retrieved by pHash)")
    parser.add_argument("--px_thresh_snap", type=float, default=3.0, help="Max px distance to snap db keypoint to recon 2D point")
    parser.add_argument("--ransac_reproj_error", type=float, default=8.0, help="RANSAC reprojection error in pixels")
    parser.add_argument("--min_pnp_inliers", type=int, default=64, help="Minimum inliers to accept PnP solution")
    parser.add_argument("--extractor_max_long_side", type=int, default=2048, help="Resizes inputs for extractor; kpts are scaled back")
    args = parser.parse_args()

    if args.query_dir is None:
        # Single-image mode
        extrinsic, K, dbg = localize_query_image(
            scene_dir=args.scene_dir,
            query_image_path=args.query_image,
            extractor_method=args.extractor,
            top_k_db=args.top_k_db,
            px_thresh_snap=args.px_thresh_snap,
            ransac_reproj_error=args.ransac_reproj_error,
            min_pnp_inliers=args.min_pnp_inliers,
            extractor_max_long_side=args.extractor_max_long_side,
        )

        print("Localization successful.")
        print("Estimated extrinsic (cam_from_world [R|t]):")
        print(extrinsic)
        print("Estimated intrinsics K:")
        print(K)
        print(f"Num correspondences: {int(dbg['num_corr'][0])}, inliers: {int(dbg['num_inliers'][0])}")

        qwxyz_t = _extrinsic_to_qwxyz_txyz(extrinsic)
        with open(args.output_txt, "a") as f:
            f.write(f"{os.path.basename(args.query_image)} {qwxyz_t[0]:.8f} {qwxyz_t[1]:.8f} {qwxyz_t[2]:.8f} {qwxyz_t[3]:.8f} {qwxyz_t[4]:.6f} {qwxyz_t[5]:.6f} {qwxyz_t[6]:.6f}\n")
        print(f"Saved pose to {args.output_txt}")
        return

    # Directory mode
    if not os.path.isdir(args.query_dir):
        raise FileNotFoundError(f"Query directory not found: {args.query_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    entries = sorted([e for e in os.listdir(args.query_dir) if os.path.splitext(e)[1].lower() in exts])

    if len(entries) == 0:
        raise RuntimeError(f"No images found in directory: {args.query_dir}")

    failures = []
    count_success = 0
    with open(args.output_txt, "w") as f:
        for name in entries:
            qpath = os.path.join(args.query_dir, name)
            try:
                print(f"Processing {name} ...", flush=True)
                extrinsic, K, dbg = localize_query_image(
                    scene_dir=args.scene_dir,
                    query_image_path=qpath,
                    extractor_method=args.extractor,
                    top_k_db=args.top_k_db,
                    px_thresh_snap=args.px_thresh_snap,
                    ransac_reproj_error=args.ransac_reproj_error,
                    min_pnp_inliers=args.min_pnp_inliers,
                    extractor_max_long_side=args.extractor_max_long_side,
                )
                qwxyz_t = _extrinsic_to_qwxyz_txyz(extrinsic)
                # write one line per image immediately
                f.write(
                    f"{name} {qwxyz_t[0]:.8f} {qwxyz_t[1]:.8f} {qwxyz_t[2]:.8f} {qwxyz_t[3]:.8f} {qwxyz_t[4]:.6f} {qwxyz_t[5]:.6f} {qwxyz_t[6]:.6f}\n"
                )
                f.flush()
                num_corr = int(dbg.get("num_corr", np.array([0], dtype=np.int32))[0])
                num_inl = int(dbg.get("num_inliers", np.array([0], dtype=np.int32))[0])
                print(f"OK {name}: corr={num_corr}, inliers={num_inl}", flush=True)
                count_success += 1
            except Exception as e:
                failures.append((name, str(e)))
                print(f"FAIL {name}: {e}", flush=True)
                continue

    print(f"Processed {count_success} images, {len(failures)} failed. Saved poses to {args.output_txt}")
    if failures:
        print("Failures (image -> reason):")
        for name, reason in failures[:20]:
            print(f"  {name} -> {reason}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    with torch.no_grad():
        main()
