"""Utility functions for dataset conversion."""

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from vision_utils.pcd_utils import get_extrinsics_matrix


def load_camera_info_dict(info_path: str) -> dict:
    """Load camera information from YAML file.

    Args:
        info_path: Path to camera info YAML file

    Returns:
        Dictionary with camera info including intrinsics and extrinsics
    """
    assert info_path.endswith(".yaml"), "Info file should be a yaml file"

    with open(info_path, 'r') as f:
        info_dict = yaml.safe_load(f)

    for cam_name, cam_info in info_dict.items():
        info_dict[cam_name]['intrinsics'] = np.array(cam_info['k']).reshape(3, 3)
        info_dict[cam_name]['extrinsics'] = get_extrinsics_matrix(cam_info['t'], cam_info['q'])

    return info_dict


def convert_state_to_action(ee_pose: np.ndarray) -> np.ndarray:
    """Convert end-effector poses to actions.

    Args:
        ee_pose: Array of shape (N, 8) with xyz, quat, gripper

    Returns:
        Array of shape (N, 7) with xyz, axis-angle, gripper
    """
    action = np.concatenate((ee_pose[1:], ee_pose[-1:]), axis=0)
    xyz = action[:, :3]
    quat = action[:, 3:7]
    gripper = action[:, 7:]

    axis_angle = R.from_quat(quat).as_rotvec()
    actions = np.concatenate((xyz, axis_angle, gripper), axis=-1)

    return actions
