#!/usr/bin/env python3
import argparse
import os
import struct
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ImagePose:
    image_id: int
    name: str
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int


def parse_absolute_poses(file_path: str, quat_order: str) -> List[Tuple[str, Tuple[float, float, float, float], Tuple[float, float, float]]]:
    poses = []
    with open(file_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                # Expect: name q1 q2 q3 q4 t1 t2 t3
                continue
            name = parts[0]
            q1, q2, q3, q4 = (float(x) for x in parts[1:5])
            t1, t2, t3 = (float(x) for x in parts[5:8])
            if quat_order.lower() == 'wxyz':
                qw, qx, qy, qz = q1, q2, q3, q4
            elif quat_order.lower() == 'xyzw':
                qx, qy, qz, qw = q1, q2, q3, q4
            else:
                raise ValueError("quat_order must be either 'wxyz' or 'xyzw'")
            poses.append((name, (qw, qx, qy, qz), (t1, t2, t3)))
    return poses


# Optional: try to obtain a camera_id from cameras.bin
# This parser only reads the first camera's id and ignores the rest.
# If parsing fails, return None and the caller should fallback to a provided --camera_id.
CAMERA_MODEL_NUM_PARAMS = {
    # Matches COLMAP's read_write_model.py model IDs
    0: 3,   # SIMPLE_PINHOLE
    1: 4,   # PINHOLE
    2: 4,   # SIMPLE_RADIAL
    3: 5,   # RADIAL
    4: 8,   # OPENCV
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12, # THIN_PRISM_FISHEYE
    11: 0,  # EQUIRECTANGULAR
}

def try_read_first_camera_id_from_bin(cameras_bin_path: str) -> Optional[int]:
    try:
        with open(cameras_bin_path, 'rb') as f:
            # uint64: num_cameras
            num_cams_bytes = f.read(8)
            if len(num_cams_bytes) != 8:
                return None
            (num_cameras,) = struct.unpack('<Q', num_cams_bytes)
            if num_cameras == 0:
                return None
            # int32: camera_id
            cam_id_bytes = f.read(4)
            if len(cam_id_bytes) != 4:
                return None
            (camera_id,) = struct.unpack('<i', cam_id_bytes)
            # int32: model_id
            model_id_bytes = f.read(4)
            if len(model_id_bytes) != 4:
                return None
            (model_id,) = struct.unpack('<i', model_id_bytes)
            # uint64: width, height
            wh_bytes = f.read(16)
            if len(wh_bytes) != 16:
                return None
            _width, _height = struct.unpack('<QQ', wh_bytes)
            # double[num_params]
            num_params = CAMERA_MODEL_NUM_PARAMS.get(model_id)
            if num_params is None:
                # Unknown model id; bail but still return id we already got
                return camera_id
            params_bytes = f.read(8 * num_params)
            if len(params_bytes) != 8 * num_params:
                return camera_id
            return camera_id
    except Exception:
        return None


def write_images_txt(out_path: str, image_poses: List[ImagePose]) -> None:
    with open(out_path, 'w') as f:
        f.write("# COLMAP images.txt\n"
                "# Image list with two lines of data per image:\n"
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for im in image_poses:
            f.write(f"{im.image_id} {im.qw:.12f} {im.qx:.12f} {im.qy:.12f} {im.qz:.12f} {im.tx:.12f} {im.ty:.12f} {im.tz:.12f} {im.camera_id} {im.name}\n")
            f.write('\n')  # Empty line for 2D points


def write_images_bin(out_path: str, image_poses: List[ImagePose]) -> None:
    with open(out_path, 'wb') as f:
        # uint64: number of registered images (we register all provided)
        f.write(struct.pack('<Q', len(image_poses)))
        for im in image_poses:
            # int32: IMAGE_ID
            f.write(struct.pack('<i', im.image_id))
            # double[4]: quaternion (qw, qx, qy, qz)
            f.write(struct.pack('<4d', im.qw, im.qx, im.qy, im.qz))
            # double[3]: translation (tx, ty, tz)
            f.write(struct.pack('<3d', im.tx, im.ty, im.tz))
            # int32: CAMERA_ID
            f.write(struct.pack('<i', im.camera_id))
            # char[]: NAME + null terminator
            f.write(im.name.encode('utf-8') + b'\x00')
            # uint64: num_points2D (we set 0 and skip list)
            f.write(struct.pack('<Q', 0))
            # No 2D points


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert absolute pose file to COLMAP images.txt and images.bin')
    parser.add_argument('--poses_path', required=True, help='Path to absolute poses txt (lines: name q1 q2 q3 q4 t1 t2 t3)')
    parser.add_argument('--output_dir', required=True, help='Directory to write images.txt and images.bin')
    parser.add_argument('--quat_order', default='wxyz', choices=['wxyz', 'xyzw'], help='Order of quaternion in input file')
    parser.add_argument('--camera_id', type=int, default=None, help='Camera ID to assign to all images (overrides detection)')
    parser.add_argument('--cameras_bin', type=str, default=None, help='Optional path to cameras.bin to auto-detect camera_id')
    parser.add_argument('--start_image_id', type=int, default=1, help='Starting IMAGE_ID')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    poses = parse_absolute_poses(args.poses_path, args.quat_order)
    if len(poses) == 0:
        raise ValueError('No valid poses parsed from poses file.')

    detected_cam_id: Optional[int] = None
    if args.camera_id is not None:
        camera_id = args.camera_id
    else:
        if args.cameras_bin and os.path.exists(args.cameras_bin):
            detected_cam_id = try_read_first_camera_id_from_bin(args.cameras_bin)
        camera_id = detected_cam_id if detected_cam_id is not None else 1

    image_poses: List[ImagePose] = []
    next_id = args.start_image_id
    for name, (qw, qx, qy, qz), (tx, ty, tz) in poses:
        image_poses.append(ImagePose(
            image_id=next_id,
            name=name,
            qw=qw, qx=qx, qy=qy, qz=qz,
            tx=tx, ty=ty, tz=tz,
            camera_id=camera_id,
        ))
        next_id += 1

    txt_path = os.path.join(args.output_dir, 'images.txt')
    bin_path = os.path.join(args.output_dir, 'images.bin')

    write_images_txt(txt_path, image_poses)
    write_images_bin(bin_path, image_poses)

    print(f'Wrote {len(image_poses)} images to:')
    print(f'  {txt_path}')
    print(f'  {bin_path}')
    if args.camera_id is None and detected_cam_id is None:
        print('Note: camera_id defaulted to 1. If your cameras.bin uses a different ID, pass --camera_id or --cameras_bin.')


if __name__ == '__main__':
    main() 