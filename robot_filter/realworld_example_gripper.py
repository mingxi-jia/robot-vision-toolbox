import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from robot_filter.arm_segmentor import RobotArmSegmentation
from utils.camera_utils import load_camera_info
from utils.transform_utils import create_transform_matrix
from utils.pcd_utils import filter_by_workspace, rgbd_to_o3d
from PIL import Image

def get_pcd(data_path):

    cam_views = [1, 2, 3]

    # 3. Load camera information and visualize poses
    cam_info = load_camera_info(os.path.join("configs", 'camera_info.yaml'))
    cam_info_dict = {f'cam{i}': cam_info[f'cam{i}'] for i in cam_views}

    merged_pcd = o3d.geometry.PointCloud()
    rgb_images = []
    for i in cam_views:
        cam_name = f'cam{i}'
        intrinsics = np.array(cam_info[cam_name]['k']).reshape(3, 3)
        extrinsics = create_transform_matrix(cam_info[cam_name]['t'], cam_info[cam_name]['q'])

        rgb_folder = os.path.join(data_path, cam_name, 'rgb')
        rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.png')])
        if not rgb_files:
            raise FileNotFoundError(f"No PNG files found in {rgb_folder}")
        first_frame = rgb_files[0].replace('.png', '')
        rgb_path = os.path.join(data_path, cam_name, 'rgb', f'{first_frame}.png')
        depth_path = os.path.join(data_path, cam_name, 'depth', f'{first_frame}.npy')
        rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
        depth = np.load(depth_path) / 1000.0  # Convert to meters

        rgb_images.append((cam_name, rgb))
        _, pcd_o3d = rgbd_to_o3d(rgb, depth, intrinsics, extrinsics)
        merged_pcd += pcd_o3d

    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.02)

    return merged_pcd, rgb_images

data_path = "example_data/multiview_rgbd"
pcd, rgb_images = get_pcd(data_path)

# Create robot segmentation instance
robot_seg = RobotArmSegmentation()
# Load the whole robot URDF for segmentation (if needed)
robot_seg.load_urdf("robot_filter/panda_description/urdf/panda_arm_hand_finray_wrist.urdf")
# Load the gripper-only URDF for gripper point cloud generation
robot_seg.load_gripper_urdf("robot_filter/panda_description/urdf/panda_gripper_wrist.urdf")

# ============================================================
# OPTION 1: If you have joint states, use FK (original method)
# ============================================================
USE_JOINT_STATES = False  # Set to True to use joint states, False to use direct EEF pose

if USE_JOINT_STATES:
    # Load joint states
    joint_states = np.load(os.path.join(data_path, "state", "joint_state.npy"))[0]
    joint_states = np.concatenate([joint_states, [0.07]])

    # Segment the robot from the scene
    filtered_pcd = robot_seg.segment(pcd, joint_states)

    # Extract finger width from joint states (last value)
    finger_width = joint_states[-1]
    print(f"Finger width: {finger_width}")

    # Get gripper point cloud in gripper frame
    gripper_pcd_local = robot_seg.get_gripper_pcd(finger_width=finger_width)
    print(f"Gripper point cloud has {len(gripper_pcd_local)} points")

    # Get end-effector transformation using link FK
    joint_names = [j.name for j in robot_seg.robot_urdf.actuated_joints]
    joint_angles = dict(zip(joint_names, joint_states))
    link_fk = robot_seg.robot_urdf.link_fk(cfg=joint_angles)

    # Get panda_hand transformation
    link_names = {link.name: link for link in link_fk.keys()}
    if 'panda_hand' in link_names:
        T_hand = link_fk[link_names['panda_hand']]
    elif 'panda_link8' in link_names:
        T_hand = link_fk[link_names['panda_link8']]
        T_hand_offset = np.eye(4)
        T_hand_offset[:3, :3] = R.from_euler('xyz', [0, 0, -0.785398163397]).as_matrix()
        T_hand = T_hand @ T_hand_offset
    else:
        raise ValueError(f"Could not find panda_hand or panda_link8 in link FK")

    # Transform gripper points to world frame
    T_world = robot_seg.T_world_urdf @ T_hand

else:
    # ============================================================
    # OPTION 2: Use direct 6D end-effector pose (NEW METHOD)
    # ============================================================
    # Define end-effector pose as 6D vector: [x, y, z, roll, pitch, yaw]
    # or as separate position and quaternion

    # Example 1: Using position + euler angles (more intuitive)
    eef_position = np.array([0.3, 0.0, 0.4])  # xyz in world frame (meters)
    eef_euler = np.array([0.0, 0.0, 0.0])   # roll, pitch, yaw (radians)

    # Example 2: Or load from your real-world data
    # eef_pose_data = np.load(os.path.join(data_path, "state", "eef_pose.npy"))[0]
    # eef_position = eef_pose_data[:3]
    # eef_euler = eef_pose_data[3:6]  # or eef_quat = eef_pose_data[3:7] if quaternion

    # Convert to transformation matrix
    T_hand = np.eye(4)
    T_hand[:3, :3] = R.from_euler('xyz', eef_euler).as_matrix()  # or R.from_quat(eef_quat).as_matrix()
    T_hand[:3, 3] = eef_position

    # Set finger width (gripper opening)
    finger_width = 0.08  # meters (0 = closed, 0.08 = fully open)
    print(f"Finger width: {finger_width}")

    # Get gripper point cloud in gripper/hand frame
    gripper_pcd_local = robot_seg.get_gripper_pcd(finger_width=finger_width)
    print(f"Gripper point cloud has {len(gripper_pcd_local)} points")

    # Transform to world frame
    # Note: T_world_urdf handles robot base transformation (usually identity for real robot)
    T_world = robot_seg.T_world_urdf @ T_hand

    # We skip segmentation since we don't have joint states for the full arm
    # Just use the original point cloud
    filtered_pcd = np.concatenate([np.asarray(pcd.points), np.asarray(pcd.colors)], axis=1)
    print("Skipping robot arm segmentation (no joint states available)")

# Transform gripper points to world frame
gripper_pcd_homogeneous = np.hstack([gripper_pcd_local, np.ones((len(gripper_pcd_local), 1))])
gripper_pcd_world = (T_world @ gripper_pcd_homogeneous.T).T[:, :3]

# Create Open3D point cloud for gripper
gripper_pcd = o3d.geometry.PointCloud()
gripper_pcd.points = o3d.utility.Vector3dVector(gripper_pcd_world)
gripper_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Grey color for gripper

# Create filtered scene point cloud
filtered_pcd_vis = o3d.geometry.PointCloud()
filtered_pcd_vis.points = o3d.utility.Vector3dVector(filtered_pcd[:, :3])
filtered_pcd_vis.colors = o3d.utility.Vector3dVector(filtered_pcd[:, 3:])

print("Visualizing: Filtered scene (colored) + Gripper only (grey)")
o3d.visualization.draw_geometries([filtered_pcd_vis, gripper_pcd])
