#!/usr/bin/env python3
"""
Visualize HDF5 robomimic dataset: point cloud with action/state trajectory.
"""

import os
import sys
import numpy as np
import open3d as o3d
import h5py
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.pcd_utils import np2o3d


def create_coordinate_frame(pose, size=0.05):
    """Create a coordinate frame from a 4x4 transformation matrix."""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size).transform(pose)


def pose_7d_to_matrix(pose):
    """Convert 7D pose [x, y, z, qx, qy, qz, qw] to 4x4 transformation matrix.

    Args:
        pose: array of shape (7,) with [x, y, z, qx, qy, qz, qw]

    Returns:
        4x4 transformation matrix
    """
    t = pose[:3]
    q = pose[3:7]  # quaternion in (x, y, z, w) format

    # Convert quaternion to rotation matrix
    R_mat = R.from_quat(q).as_matrix()

    # Compose 4x4 transformation matrix
    T = np.eye(4, dtype=float)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


def reconstruct_poses_from_actions(actions, initial_state):
    """Reconstruct absolute poses from action deltas.

    Args:
        actions: (T, 7) array of [delta_x, delta_y, delta_z, delta_qx, delta_qy, delta_qz, gripper]
        initial_state: (7,) array of initial [x, y, z, qx, qy, qz, qw]

    Returns:
        (T, 7) array of absolute poses
    """
    poses = [initial_state]

    for i in range(len(actions)):
        prev_pose = poses[-1]
        action = actions[i]

        # Extract delta position and delta rotation
        delta_pos = action[:3]
        delta_rot = action[3:6]  # axis-angle representation

        # Current position
        new_pos = prev_pose[:3] + delta_pos

        # Current rotation: apply delta rotation to previous rotation
        prev_quat = prev_pose[3:7]
        prev_R = R.from_quat(prev_quat)
        delta_R = R.from_rotvec(delta_rot)
        new_R = prev_R * delta_R
        new_quat = new_R.as_quat()

        # Combine into new pose
        new_pose = np.concatenate([new_pos, new_quat])
        poses.append(new_pose)

    return np.array(poses[:-1])  # Return T poses (excluding the extra one)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize HDF5 robomimic dataset with point cloud and trajectory"
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="Path to HDF5 dataset file"
    )
    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        help="Demo name to visualize (e.g., 'demo_0'). Overrides --demo-index if provided."
    )
    parser.add_argument(
        "--demo-index",
        type=int,
        default=0,
        help="Demo index to visualize (default: 0, first demo). Ignored if --demo is provided."
    )
    parser.add_argument(
        "--pcd-frame",
        type=int,
        default=0,
        help="Which point cloud frame to visualize (default: 0, first frame)"
    )
    parser.add_argument(
        "--origin-frame-size",
        type=float,
        default=0.1,
        help="Size of world origin coordinate frame (default: 0.1)"
    )
    parser.add_argument(
        "--frame-size",
        type=float,
        default=0.03,
        help="Size of coordinate frames (default: 0.03)"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Show every Nth frame in trajectory (default: 1, show all)"
    )
    parser.add_argument(
        "--use-actions",
        action="store_true",
        help="Use actions instead of states for trajectory (will reconstruct poses from deltas)"
    )

    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_path)
    if not hdf5_path.exists():
        print(f"❌ HDF5 file not found: {hdf5_path}")
        sys.exit(1)

    # Load HDF5 dataset
    print(f"📂 Loading HDF5 dataset: {hdf5_path.name}")
    with h5py.File(hdf5_path, 'r') as f:
        # Determine demo name
        if args.demo is not None:
            demo_name = args.demo
        else:
            demo_name = f"demo_{args.demo_index}"

        # Check if demo exists
        if demo_name not in f['data']:
            available_demos = list(f['data'].keys())
            print(f"❌ Demo '{demo_name}' not found.")
            print(f"   Available demos: {', '.join(available_demos)}")
            sys.exit(1)

        print(f"📊 Loading demo: {demo_name}")
        demo = f['data'][demo_name]

        # Load point cloud
        if 'obs/pcd' not in demo:
            print(f"❌ No point cloud data found in obs/pcd")
            print(f"   Available obs keys: {list(demo['obs'].keys())}")
            sys.exit(1)

        pcd_data = demo['obs/pcd'][:]
        print(f"   Point cloud shape: {pcd_data.shape}")

        if args.pcd_frame >= len(pcd_data):
            print(f"❌ Frame {args.pcd_frame} out of range (0-{len(pcd_data)-1})")
            sys.exit(1)

        # Load trajectory (either from actions or states)
        if args.use_actions:
            print(f"📊 Using actions to reconstruct trajectory")
            actions = demo['actions'][:]
            states = demo['states'][:]
            initial_state = states[0]  # First state as initial pose

            # Reconstruct poses from actions
            poses = reconstruct_poses_from_actions(actions, initial_state)
            print(f"   Actions shape: {actions.shape}")
            print(f"   Reconstructed poses: {len(poses)}")
        else:
            print(f"📊 Using states for trajectory")
            states = demo['states'][:]
            poses = states  # States are absolute poses
            print(f"   States shape: {states.shape}")

        # Load grasp states
        grasp_states = None
        if 'obs/robot0_gripper_qpos' in demo:
            grasp_states = demo['obs/robot0_gripper_qpos'][:]
            print(f"   Grasp states shape: {grasp_states.shape}")
        else:
            print(f"   ⚠️  No grasp state data found in obs/robot0_gripper_qpos")

        # Get point cloud frame
        pcd_frame = pcd_data[args.pcd_frame]
        print(f"\n📍 Visualizing frame {args.pcd_frame}")
        print(f"   Point cloud points: {len(pcd_frame)}")

        # Display grasp state for this frame
        if grasp_states is not None and args.pcd_frame < len(grasp_states):
            grasp_value = grasp_states[args.pcd_frame]
            print(f"   Grasp state: {grasp_value} {'(Open)' if grasp_value < 0.5 else '(Closed)'}")

        # Convert to Open3D point cloud
        points = pcd_frame[:, :3]
        colors = pcd_frame[:, 3:6]
        pcd = np2o3d(points, colors)

    # Create visualization
    print(f"\n🎨 Creating visualization...")
    print(f"   World origin frame size: {args.origin_frame_size}")
    print(f"   Trajectory frame size: {args.frame_size}")
    print(f"   Subsampling: every {args.subsample} frame(s)")

    geometries = [pcd]

    # Add world origin coordinate frame
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.origin_frame_size)
    geometries.append(origin_frame)

    # Add coordinate frames for trajectory
    num_frames_shown = 0
    for i in range(0, len(poses), args.subsample):
        pose_7d = poses[i]
        pose_mat = pose_7d_to_matrix(pose_7d)

        frame = create_coordinate_frame(pose_mat, size=args.frame_size)
        geometries.append(frame)
        num_frames_shown += 1

    print(f"   Showing world origin + {num_frames_shown} trajectory frames")

    # Print trajectory info
    print(f"\n=� Trajectory statistics:")
    print(f"   Total frames: {len(poses)}")
    if len(poses) > 0:
        print(f"   Start position: [{poses[0][0]:.3f}, {poses[0][1]:.3f}, {poses[0][2]:.3f}]")
        print(f"   End position: [{poses[-1][0]:.3f}, {poses[-1][1]:.3f}, {poses[-1][2]:.3f}]")
        displacement = np.linalg.norm(poses[-1][:3] - poses[0][:3])
        print(f"   Total displacement: {displacement:.3f}m")

    # Visualize
    print("\n=�  Launching visualizer...")
    print("\nControls:")
    print("  Mouse Left: Rotate")
    print("  Mouse Middle: Pan")
    print("  Mouse Scroll: Zoom")
    print("  Q/Esc: Quit")
    print()

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{args.demo} - Frame {args.pcd_frame} + {'Actions' if args.use_actions else 'States'} Trajectory",
        width=1280,
        height=720
    )


if __name__ == "__main__":
    main()
