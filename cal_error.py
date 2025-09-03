#!/usr/bin/env python3
import math
from typing import Tuple, List, Dict, Optional
import argparse
import re


def parse_pose_from_line(line: str) -> Tuple[List[float], List[float]]:
    """Extract the first 7 floats as (qw,qx,qy,qz, tx,ty,tz) from a line."""
    nums = line.strip().split(' ')
    # print(nums)
    # for tok in line.strip().split():
    #     try:
    #         nums.append(float(tok))
    #     except ValueError:
    #         continue  # ignore non-numeric tokens (e.g., filenames)
    if len(nums) < 7:
        raise ValueError("Need at least 7 numeric values: qw qx qy qz tx ty tz")
    qw, qx, qy, qz, tx, ty, tz = nums[1:8]
    return [float(qw), float(qx), float(qy), float(qz)], [float(tx), float(ty), float(tz)]


def q_normalize(q):
    n = math.sqrt(sum(x*x for x in q))
    return [x / n for x in q]


def q_conj(q):
    w, x, y, z = q
    return [w, -x, -y, -z]


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]


def rotation_error_deg(q1, q2) -> float:
    """Shortest-angle rotation error between q1 and q2 in degrees."""
    q1 = q_normalize(q1)
    q2 = q_normalize(q2)
    # Ensure same hemisphere for shortest arc
    dot = sum(a*b for a, b in zip(q1, q2))
    if dot < 0:
        q2 = [-v for v in q2]
    # Relative rotation q_err = conj(q1) * q2
    qe = q_mul(q_conj(q1), q2)
    qe = q_normalize(qe)
    w = max(-1.0, min(1.0, abs(qe[0])))
    angle_rad = 2.0 * math.acos(w)
    return math.degrees(angle_rad)


def translation_error(t1, t2):
    """Return (L2 distance, per-axis delta vector t1 - t2)."""
    dx = t1[0] - t2[0]
    dy = t1[1] - t2[1]
    dz = t1[2] - t2[2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    return dist, (dx, dy, dz)


# === New helpers: camera center error (origin-shift robust) ===

def quat_wxyz_to_R(q):
    q = q_normalize(q)
    w, x, y, z = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return [
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)    ],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)    ],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
    ]


def camera_center_from_qt(q, t):
    # For world->cam: x_cam = R X_w + t, the camera center is C = -R^T t
    R = quat_wxyz_to_R(q)
    # R^T t
    r00, r01, r02 = R[0]
    r10, r11, r12 = R[1]
    r20, r21, r22 = R[2]
    tx, ty, tz = t
    Rt_t_x = r00*tx + r10*ty + r20*tz
    Rt_t_y = r01*tx + r11*ty + r21*tz
    Rt_t_z = r02*tx + r12*ty + r22*tz
    return [-Rt_t_x, -Rt_t_y, -Rt_t_z]


def compute_pair_error(line1: str, line2: str):
    q1, t1 = parse_pose_from_line(line1)
    q2, t2 = parse_pose_from_line(line2)
    rot_err_deg = rotation_error_deg(q1, q2)
    trans_dist, (dx, dy, dz) = translation_error(t1, t2)
    return rot_err_deg, trans_dist, (dx, dy, dz)


def compute_pair_error_centers(line1: str, line2: str):
    q1, t1 = parse_pose_from_line(line1)
    q2, t2 = parse_pose_from_line(line2)
    c1 = camera_center_from_qt(q1, t1)
    c2 = camera_center_from_qt(q2, t2)
    dist, (dx, dy, dz) = translation_error(c1, c2)
    rot_err_deg = rotation_error_deg(q1, q2)
    return rot_err_deg, dist, (dx, dy, dz)


# Keep old function name but parameterize the metric via a flag for reuse

def process_pairs(pairs: List[List[str]], use_centers: bool = False, align_translation: bool = False) -> None:
    if not pairs:
        print("No pairs provided.")
        return

    # Optional global translation alignment in camera-center space
    offset = (0.0, 0.0, 0.0)
    if align_translation:
        # Estimate mean offset s that minimizes ||(C1 + s) - C2|| over pairs
        dx_sum = dy_sum = dz_sum = 0.0
        count = 0
        for pair in pairs:
            if len(pair) != 2:
                continue
            q1, t1 = parse_pose_from_line(pair[0])
            q2, t2 = parse_pose_from_line(pair[1])
            c1 = camera_center_from_qt(q1, t1)
            c2 = camera_center_from_qt(q2, t2)
            dx_sum += (c2[0] - c1[0])
            dy_sum += (c2[1] - c1[1])
            dz_sum += (c2[2] - c1[2])
            count += 1
        if count > 0:
            offset = (dx_sum / count, dy_sum / count, dz_sum / count)
            print(f"Applying global center offset (meters): {offset}")

    rot_errors: List[float] = []
    trans_errors: List[float] = []
    dx_list: List[float] = []
    dy_list: List[float] = []
    dz_list: List[float] = []
    for idx, pair in enumerate(pairs, start=1):
        if len(pair) != 2:
            print(f"Skipping entry {idx}: expected 2 lines, got {len(pair)}")
            continue
        line1, line2 = pair

        if use_centers:
            rot_err_deg, trans_dist, (dx, dy, dz) = compute_pair_error_centers(line1, line2)
            # Apply global offset if requested
            if align_translation:
                # Apply estimated global offset s to predicted centers: (C1 + s) - C2
                dx += offset[0]
                dy += offset[1]
                dz += offset[2]
                trans_dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        else:
            rot_err_deg, trans_dist, (dx, dy, dz) = compute_pair_error(line1, line2)

        print(f"Pair {idx}: Rotational error: {rot_err_deg:.6f} deg")
        print(f"Pair {idx}: Translational error (L2): {trans_dist*100:.6f} cm")
        print(
            f"Pair {idx}: Per-axis translation delta (t1 - t2): "
            f"dx={dx*100:.6f} cm, dy={dy*100:.6f} cm, dz={dz*100:.6f} cm"
        )
        if(rot_err_deg > 10):
            print("Big error: ", rot_err_deg)
            continue
        rot_errors.append(rot_err_deg)
        trans_errors.append(trans_dist)
        dx_list.append(dx)
        dy_list.append(dy)
        dz_list.append(dz)
    if rot_errors:
        avg_rot = sum(rot_errors) / len(rot_errors)
        avg_trans = sum(trans_errors) / len(trans_errors)
        avg_dx = sum(dx_list) / len(dx_list)
        avg_dy = sum(dy_list) / len(dy_list)
        avg_dz = sum(dz_list) / len(dz_list)
        print(f"Average rotational error: {avg_rot:.6f} deg")
        print(f"Average translational error (L2): {avg_trans*100:.6f} cm")
        print(
            f"Average per-axis translation delta (t1 - t2): "
            f"dx={avg_dx*100:.6f} cm, dy={avg_dy*100:.6f} cm, dz={avg_dz*100:.6f} cm"
        )


# ===== New helpers for file parsing and name mapping =====

def parse_predictions_file(pred_path: str) -> Dict[str, str]:
    """Parse a predictions file with lines: NAME qw qx qy qz tx ty tz.

    Returns mapping NAME -> original line string.
    """
    name_to_line: Dict[str, str] = {}
    with open(pred_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # Expect at least: name + 7 numbers
            if len(parts) < 8:
                continue
            name = parts[0]
            name_to_line[name] = line
    return name_to_line


def parse_colmap_images_txt(gt_images_path: str) -> Dict[str, str]:
    """Parse COLMAP images.txt into mapping NAME -> original metadata line.

    The images.txt format is two lines per image; we only take the first line which is:
    IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    """
    name_to_line: Dict[str, str] = {}
    with open(gt_images_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # Heuristic: first token is IMAGE_ID (int), last is NAME, and there are >= 10 tokens
            try:
                _ = int(parts[0])
            except Exception:
                continue
            if len(parts) < 10:
                continue
            name = parts[-1]
            name_to_line[name] = line
    return name_to_line


def map_pred_name_to_gt_name(pred_name: str) -> Optional[str]:
    """Map e.g. frame_0021.jpg -> frame_000200.jpg using rule: gt_idx = (idx-1)*10.

    Returns None if the pattern cannot be parsed.
    """
    m = re.match(r"^(.*?)(\d+)(\.\w+)$", pred_name)
    if not m:
        return None
    prefix, digits, suffix = m.groups()
    try:
        idx = int(digits)
    except Exception:
        return None
    if idx <= 0:
        return None
    gt_idx = (idx - 1) * 10
    # GT images use 6-digit zero padding as in examples
    return f"{prefix}{gt_idx:06d}{suffix}"



def main():
    parser = argparse.ArgumentParser(description="Compute pose errors between predictions and COLMAP GT using name mapping.")
    parser.add_argument("--pred", required=True, help="Path to predictions file (absolute_poses.txt)")
    parser.add_argument("--gt_images", required=True, help="Path to COLMAP images.txt (ground truth)")
    parser.add_argument("--use_centers", action="store_true", help="Measure translation error on camera centers (origin-shift robust)")
    parser.add_argument("--align_translation", action="store_true", help="Estimate and remove a single global center offset before measuring errors")
    args = parser.parse_args()

    pred_map = parse_predictions_file(args.pred)
    gt_map = parse_colmap_images_txt(args.gt_images)

    if not pred_map:
        print("No predictions parsed.")
        return
    if not gt_map:
        print("No GT images parsed.")
        return

    pairs: List[List[str]] = []
    missing: List[str] = []

    cnt = 1
    for pred_name, pred_line in pred_map.items():
        gt_name = map_pred_name_to_gt_name(pred_name)
        print(f"pred_name: {cnt}", pred_name, "gt_name: ", gt_name)
        cnt += 1
        if gt_name is None:
            missing.append(pred_name)
            continue
        gt_line = gt_map.get(gt_name)
        if gt_line is None:
            missing.append(pred_name)
            continue
        pairs.append([pred_line, gt_line])

    if missing:
        print(f"Warning: {len(missing)} predictions had no matching GT and were skipped.")
        # Uncomment to list missing names
        # for n in missing:
        #     print(f"  missing GT for {n}")

    if not pairs:
        print("No valid prediction/GT pairs to evaluate.")
        return

    # Report and summarize
    process_pairs(pairs, use_centers=args.use_centers, align_translation=args.align_translation)


if __name__ == "__main__":
    main()
