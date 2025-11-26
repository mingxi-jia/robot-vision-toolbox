#!/usr/bin/env python3
"""
Visualize a single point cloud frame with hand trajectory shown as coordinate frames.
"""

import os
import sys
import numpy as np
import open3d as o3d
from pathlib import Path
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.pcd_utils import np2o3d
from scipy.spatial.transform import Rotation as Rotation


def create_coordinate_frame(pose, size=0.05):
    """Create a coordinate frame from a 4x4 transformation matrix."""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size).transform(pose)


def load_pcd(path):
    """Load point cloud from .ply or .npy file."""
    path = Path(path)
    if path.suffix == '.ply':
        return o3d.io.read_point_cloud(str(path))
    elif path.suffix == '.npy':
        arr = np.load(path, allow_pickle=True)
        assert arr.shape[0] > 0, f"Empty point cloud in file: {path}"
        points, colors = arr[:, :3], arr[:, 3:6]
        return np2o3d(points, colors)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize point cloud with hand trajectory as coordinate frames"
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        help="Path to folder containing pcd_no_hand subfolder and hand_poses_wrt_world.npy"
    )
    parser.add_argument(
        "--frame-size",
        type=float,
        default=0.05,
        help="Size of coordinate frames (default: 0.05)"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Show every Nth frame in trajectory (default: 1, show all)"
    )
    parser.add_argument(
        "--pcd-frame",
        type=int,
        default=0,
        help="Which point cloud frame to visualize (default: 0, first frame)"
    )

    args = parser.parse_args()

    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path}")
        sys.exit(1)

    # Find pcd_no_hand folder
    pcd_folder = folder_path / "pcd_no_hand"
    if not pcd_folder.exists():
        print(f"❌ pcd_no_hand folder not found in {folder_path}")
        sys.exit(1)

    # Find hand poses file
    hand_poses_file = folder_path / "hand_poses_wrt_world.npy"
    if not hand_poses_file.exists():
        print(f"❌ hand_poses_wrt_world.npy not found in {folder_path}")
        sys.exit(1)

    # Load first point cloud
    pcd_files = sorted(pcd_folder.glob("*.npy"))
    if not pcd_files:
        pcd_files = sorted(pcd_folder.glob("*.ply"))

    if not pcd_files:
        print(f"❌ No point cloud files found in {pcd_folder}")
        sys.exit(1)

    if args.pcd_frame >= len(pcd_files):
        print(f"❌ Frame {args.pcd_frame} out of range (0-{len(pcd_files)-1})")
        sys.exit(1)

    print(f"📁 Loading point cloud frame {args.pcd_frame}: {pcd_files[args.pcd_frame].name}")
    pcd = load_pcd(pcd_files[args.pcd_frame])
    print(f"   Points: {len(pcd.points)}")

    # Load hand trajectory
    print(f"📊 Loading hand trajectory: {hand_poses_file.name}")
    hand_poses = np.load(hand_poses_file, allow_pickle=True)[()]
    items = sorted(hand_poses.items(), key=lambda kv: int(kv[0]))
    hand_poses = np.stack([np.asarray(v) for _, v in items], axis=0)

    print(f"   Shape: {hand_poses.shape}")
    print(f"   Frames: {len(hand_poses)}")

    

    # Create visualization
    print(f"\n🎨 Creating visualization...")
    print(f"   Coordinate frame size: {args.frame_size}")
    print(f"   Subsampling: every {args.subsample} frame(s)")

    geometries = [pcd]

    # Add coordinate frames for hand trajectory
    num_frames_shown = 0
    for i in range(0, len(hand_poses), args.subsample):
        pose = hand_poses[i]
        # pose is [tx, ty, tz, qx, qy, qz, qw] (xyz, xyzw)

        t = np.asarray(pose[:3], dtype=float)
        q = np.asarray(pose[3:7], dtype=float)  

        # scipy Rotation.from_quat expects (x, y, z, w)
        R_mat = Rotation.from_quat(q).as_matrix()

        # compose 4x4 transformation matrix and overwrite pose variable
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = R_mat
        pose[:3, 3] = t
        frame = create_coordinate_frame(pose, size=args.frame_size)
        geometries.append(frame)
        num_frames_shown += 1

    print(f"   Showing {num_frames_shown} coordinate frames")

    # Visualize
    print("\n🖼️  Launching visualizer...")
    print("\nControls:")
    print("  Mouse Left: Rotate")
    print("  Mouse Middle: Pan")
    print("  Mouse Scroll: Zoom")
    print("  Q/Esc: Quit")
    print()

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Point Cloud + Hand Trajectory (Frame {args.pcd_frame})",
        width=1280,
        height=720
    )


if __name__ == "__main__":
    main()
