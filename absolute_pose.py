# compose_relative_to_absolute.py
# Usage: just run it as-is to see the example. Replace the inputs under "USER INPUTS".

import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import os


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float).reshape(4)
    q = q / np.linalg.norm(q)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=float).reshape(4)
    q = q / np.linalg.norm(q)
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def compose_absolute_from_relative(R_rel: np.ndarray, t_rel: np.ndarray,
                                   q_init_wxyz: np.ndarray, t_init: np.ndarray):
    """
    Inputs:
      - R_rel (3x3), t_rel (3,): relative pose cam2_from_cam1
      - q_init_wxyz (4,), t_init (3,): absolute init pose cam1_from_world (world->cam)
    Returns:
      - R_abs (3x3), t_abs (3,), q_abs_wxyz (4,), Rt_abs (3x4)
    Composition (world->cam):
        cam2_from_world = (cam2_from_cam1) @ (cam1_from_world)
        R_abs = R_rel * R_init
        t_abs = R_rel @ t_init + t_rel
    """
    R_rel = np.asarray(R_rel, dtype=float).reshape(3,3)
    t_rel = np.asarray(t_rel, dtype=float).reshape(3)
    t_init = np.asarray(t_init, dtype=float).reshape(3)

    R_init = R.from_quat(wxyz_to_xyzw(q_init_wxyz))
    R_rel_obj = R.from_matrix(R_rel)

    R_abs_obj = R_rel_obj * R_init
    R_abs = R_abs_obj.as_matrix()
    t_abs = R_rel @ t_init + t_rel

    q_abs_wxyz = xyzw_to_wxyz(R_abs_obj.as_quat())
    Rt_abs = np.c_[R_abs, t_abs]
    return R_abs, t_abs, q_abs_wxyz, Rt_abs


def geodesic_angle_deg(q_a_wxyz: np.ndarray, q_b_wxyz: np.ndarray) -> float:
    qa = R.from_quat(wxyz_to_xyzw(q_a_wxyz))
    qb = R.from_quat(wxyz_to_xyzw(q_b_wxyz))
    dR = qa.inv() * qb
    return float(np.degrees(dR.magnitude()))


def translation_errors(t_est: np.ndarray, t_gt: np.ndarray):
    diff = np.asarray(t_est, float).reshape(3) - np.asarray(t_gt, float).reshape(3)
    return float(np.linalg.norm(diff)), diff


def _parse_pose_line(line: str):
    """Parses a single line with format:
    <frame_id> qw qx qy qz [tx ty tz]

    Returns (frame_id, q_wxyz (4,), t (3,) or None if absent)
    """
    raw = line.strip()
    print("raw: ", raw)
    if not raw or raw.startswith('#'):
        return None
    parts = raw.split()
    if len(parts) < 5:
        return None
    frame_id = parts[0]
    nums = [float(x) for x in parts[1:]]
    print("nums: ", nums)
    print("frame_id: ", frame_id)
    if len(nums) >= 7:
        q = np.array(nums[:4], dtype=float)
        t = np.array(nums[4:7], dtype=float)
    elif len(nums) == 4:
        q = np.array(nums[:4], dtype=float)
        t = np.zeros(3, dtype=float)
    else:
        # Not enough numbers
        return None
    return frame_id, q, t


def process_file_of_rel_quaternions(input_path: str, output_path: str,
                                    q_init: np.ndarray, t_init: np.ndarray) -> int:
    """Reads relative poses from input file and writes absolute poses to output.

    Input file lines: <frame> qw qx qy qz [tx ty tz]
    Output file lines: <frame> qw qx qy qz tx ty tz   (absolute world->cam)
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Current absolute pose accumulators
    q_init = np.array(q_init, dtype=float).reshape(4)
    t_init = np.array(t_init, dtype=float).reshape(3)
    q_curr = np.array(q_init, dtype=float).reshape(4)
    t_curr = np.array(t_init, dtype=float).reshape(3)

    out_lines = []
    for line in lines:
        parsed = _parse_pose_line(line)
        if parsed is None:
            continue
        frame_id, q_rel_wxyz, t_rel = parsed

        # Build R_rel from quaternion
        R_rel = R.from_quat(wxyz_to_xyzw(q_rel_wxyz)).as_matrix()

        # Compose with current absolute
        _, t_abs, q_abs_wxyz, _ = compose_absolute_from_relative(R_rel, t_rel, q_init, t_init)

        # Update accumulators
        q_curr = q_abs_wxyz
        t_curr = t_abs

        out_lines.append(
            f"{frame_id} "
            f"{q_curr[0]:.8f} {q_curr[1]:.8f} {q_curr[2]:.8f} {q_curr[3]:.8f} "
            f"{t_curr[0]:.8f} {t_curr[1]:.8f} {t_curr[2]:.8f}\n"
        )

    with open(output_path, 'a') as f:
        f.writelines(out_lines)

    return len(out_lines)


if __name__ == "__main__":
    # ======== USER INPUTS (replace with yours) ========
    # Fixed init pose (world->cam): (qw, qx, qy, qz, tx, ty, tz)
    q_init = np.array([0.534053, 0.447417, -0.473307, 0.53906], dtype=float)
    t_init = np.array([3.66724, 1.69314, -1.05368], dtype=float)
    # q_init = np.array([0.631464, 0.753969, 0.137941, -0.117284], dtype=float)
    # t_init = np.array([-2.4357, 1.73943, 0.164075], dtype=float)

    parser = argparse.ArgumentParser(description="Accumulate absolute poses from a file of relative quaternions (and optional translations)")
    parser.add_argument('-i', '--input', type=str, default='poses.txt', help="Input file with lines: <frame> qw qx qy qz [tx ty tz]")
    parser.add_argument('-o', '--output', type=str, default=None, help="Output file for absolute poses. Defaults to <dir>/absolute_<input_basename>")
    parser.add_argument('--single_example', action='store_true', help="Run the original single-example composition and print results")
    args = parser.parse_args()

    # If user requests single example or input does not exist, fall back to demo
    run_demo = args.single_example or (args.input is None) or (not os.path.exists(args.input))

    if run_demo:
        # Demo relative pose cam2_from_cam1: [R|t]
        R_rel = np.array([
            [ 0.69416946, -0.12765369,  0.70840190],
            [-0.10634849,  0.95516020,  0.27633128],
            [-0.71191204, -0.26715820,  0.64946730]
        ], dtype=float)
        t_rel = np.array([0.06464452, 0.00756567, 0.04645512], dtype=float)

        R_abs, t_abs, q_abs_wxyz, Rt_abs = compose_absolute_from_relative(R_rel, t_rel, q_init, t_init)

        np.set_printoptions(precision=8, suppress=True)
        print("Absolute pose (world->cam):")
        print("Quaternion (qw, qx, qy, qz):", q_abs_wxyz)
        print("Translation (tx, ty, tz):   ", t_abs)
        print("[R|t]:\n", Rt_abs)
    else:
        input_path = args.input
        if args.output is None:
            in_dir = os.path.dirname(os.path.abspath(input_path))
            in_base = os.path.basename(input_path)
            output_path = os.path.join(in_dir, f"absolute_{in_base}")
        else:
            output_path = args.output

        count = process_file_of_rel_quaternions(input_path, output_path, q_init, t_init)
        print(f"Wrote {count} absolute poses to: {output_path}")
