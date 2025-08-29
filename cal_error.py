#!/usr/bin/env python3
import math
from typing import Tuple, List


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


def compute_pair_error(line1: str, line2: str):
    q1, t1 = parse_pose_from_line(line1)
    q2, t2 = parse_pose_from_line(line2)
    rot_err_deg = rotation_error_deg(q1, q2)
    trans_dist, (dx, dy, dz) = translation_error(t1, t2)
    return rot_err_deg, trans_dist, (dx, dy, dz)


def process_pairs(pairs: List[List[str]]) -> None:
    if not pairs:
        print("No pairs provided.")
        return
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
        rot_err_deg, trans_dist, (dx, dy, dz) = compute_pair_error(line1, line2)
        print(f"Pair {idx}: Rotational error: {rot_err_deg:.6f} deg")
        print(f"Pair {idx}: Translational error (L2): {trans_dist:.6f}")
        print(
            f"Pair {idx}: Per-axis translation delta (t1 - t2): "
            f"dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}"
        )
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
        print(f"Average translational error (L2): {avg_trans:.6f}")
        print(
            f"Average per-axis translation delta (t1 - t2): "
            f"dx={avg_dx:.6f}, dy={avg_dy:.6f}, dz={avg_dz:.6f}"
        )


def main():
    # Define your list of [line1, line2] pairs here
    pairs: List[List[str]] = [
        [
            "frame_0283.jpg 0.61121935 0.61238429 0.35238566 -0.35668016 -2.95560394 1.92719296 0.14825538",
            "242 0.623393 0.623624 0.328777 -0.338201 -2.1353 1.89435 1.71956 1 frame_002820.jpg",
        ],
        [
            "frame_0816.jpg 0.58669496 0.63284002 0.35625867 -0.35830475 -2.99473443 1.93474735 0.40533786",
            "718 0.596963 0.643566 0.335981 -0.34143 -2.30137 1.70746 2.11709 1 frame_008150.jpg",
        ],
        [
            "frame_0719.jpg 0.34219882 0.34844806 0.62576049 -0.60820040 -2.11557172 1.76619976 2.66966293",
            "635 0.352252 0.361109 0.622525 -0.598315 -1.43216 1.63696 2.67944 1 frame_007180.jpg",
        ],
        [
            "frame_0659.jpg 0.08456995 -0.05157535 -0.89447958 0.43599791 1.96129040 -1.11738391 3.51460129",
            "596 -0.074797 0.0551423 0.894126 -0.438068 3.04187 -1.42598 3.01129 1 frame_006580.jpg",
        ],
    ]
    # pairs: List[List[str]] = [
    #     [
    #         "frame_0223.jpg 0.48060592 0.72622739 0.39776445 -0.28878223 -2.12631355 1.04781736 2.17917804",
    #         "196 0.494903 0.71723 0.39695 -0.288242 -2.47685 0.816623 2.43163 1 frame_002220.jpg",
    #     ],
    #     [
    #         "frame_0021.jpg 0.63301151 0.54355637 0.35904509 -0.41824577 -1.73195717 1.97337240 1.42231751",
    #         "14 0.625944 0.549281 0.3596 -0.420917 -1.66833 2.01945 1.5345 1 frame_000200.jpg",
    #     ],
    #     [
    #         "frame_0056.jpg 0.75458507 0.64872440 -0.07495499 0.06434109 -1.72828622 1.58801676 -1.68794386",
    #         "49 0.754136 0.650895 -0.070399 0.0515645 -1.45047 1.63617 -1.70768 1 frame_000550.jpg",
    #     ],
    #     [
    #         "frame_0212.jpg 0.77930586 0.61150198 0.08013668 -0.11102170 -2.81855864 1.68216254 -0.58736744",
    #         "185 0.797858 0.584817 0.0734038 -0.126584 -3.44741 1.53376 -0.471858 1 frame_002110.jpg",
    #     ],
    #     [
    #         "frame_0224.jpg 0.47743418 0.72375038 0.39925949 -0.29805007 -2.10969554 1.06616932 2.18763071",
    #         "197 0.489119 0.718755 0.396938 -0.294269 -2.49374 0.847405 2.40637 1 frame_002230.jpg",
    #     ],
    #     [
    #         "frame_0225.jpg 0.47384316 0.73370389 0.39411653 -0.28604795 -2.17658220 1.04197187 2.13061184",
    #         "198 0.483597 0.728151 0.39367 -0.284524 -2.60027 0.835147 2.31962 1 frame_002240.jpg",
    #     ],
    #     [
    #         "frame_0226.jpg 0.46162784 0.73723505 0.39727894 -0.29249558 -2.15412601 1.03316281 2.17398427",
    #         "199 0.47233 0.733856 0.394279 -0.287933 -2.61912 0.845583 2.31691 1 frame_002250.jpg",
    #     ],
    #     [
    #         "frame_0227.jpg 0.45494583 0.74424219 0.39408926 -0.28951944 -2.17326499 1.03091369 2.16926523",
    #         "200 0.465916 0.740351 0.391307 -0.285799 -2.66903 0.860439 2.27046 1 frame_002260.jpg",
    #     ],
    #     [
    #         "frame_0228.jpg 0.45115199 0.74967119 0.39083654 -0.28583524 -2.19339627 1.02920680 2.14546595",
    #         "201 0.46116 0.744657 0.389726 -0.284483 -2.7136 0.871448 2.22418 1 frame_002270.jpg",
    #     ],
    #     [
    #         "frame_0229.jpg 0.43401776 0.76137532 0.40300258 -0.26367618 -2.26757065 0.86576814 2.14794612",
    #         "202 0.445134 0.756729 0.401512 -0.260777 -2.81688 0.704531 2.15886 1 frame_002280.jpg",
    #     ],
    #     [
    #         "frame_0230.jpg 0.41376488 0.77235058 0.41768759 -0.24043771 -2.32945703 0.67715722 2.12636084",
    #         "203 0.423336 0.768596 0.416226 -0.238335 -2.89912 0.525012 2.10095 1 frame_002290.jpg",
    #     ],
    #     [
    #         "frame_0231.jpg 0.40616087 0.78393102 0.41624779 -0.21730917 -2.42506343 0.60108556 2.03554272",
    #         "204 0.415771 0.779794 0.415312 -0.215806 -3.02063 0.471294 1.94338 1 frame_002300.jpg",
    #     ],
    #     [
    #         "frame_0232.jpg 0.39124778 0.80049125 0.40584267 -0.20354524 -2.48226491 0.59416776 1.97768312",
    #         "205 0.403002 0.796446 0.405454 -0.197153 -3.11521 0.496107 1.79201 1 frame_002310.jpg",
    #     ],
    #     [
    #         "frame_0233.jpg 0.38897313 0.80281213 0.40659974 -0.19715281 -2.51469859 0.57814767 1.93747919",
    #         "206 0.398875 0.798946 0.407343 -0.191458 -3.17121 0.503732 1.72593 1 frame_002320.jpg",
    #     ],
    #     [
    #         "frame_0234.jpg 0.39072512 0.80119234 0.40826058 -0.19684514 -2.53589030 0.59643917 1.90886339",
    #         "207 0.403738 0.795006 0.410142 -0.191689 -3.22326 0.532162 1.65548 1 frame_002330.jpg",
    #     ],
    #     [
    #         "frame_0235.jpg 0.40594210 0.78996050 0.41420379 -0.19901920 -2.56971724 0.60801135 1.87359724",
    #         "208 0.412489 0.786424 0.415216 -0.197447 -3.28225 0.572281 1.60676 1 frame_002340.jpg",
    #     ],
    #     [
    #         "frame_0238.jpg 0.41073291 0.75624250 0.45073240 -0.23714143 -2.47575923 0.61323528 2.03896533",
    #         "211 0.415401 0.751265 0.454136 -0.238334 -3.32664 0.588975 1.7818 1 frame_002370.jpg",
    #     ],
    #     [
    #         "frame_0236.jpg 0.40968002 0.78104322 0.42587585 -0.20189982 -2.58104329 0.58586927 1.87398663",
    #         "209 0.415653 0.776142 0.428049 -0.203986 -3.33674 0.562238 1.6109 1 frame_002350.jpg",
    #     ],
    # ]
    process_pairs(pairs)


if __name__ == "__main__":
    main()
