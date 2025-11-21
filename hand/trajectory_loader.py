"""Trajectory loading and processing."""

import os
from matplotlib import axis
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R
from hand.hand_utils import convert_state_to_action

from utils.pcd_utils import (depth2fgpcd, np2o3d, o3d2np, pcd_to_voxel, render_pcd_from_pose)
from configs.workspace import WORKSPACE, MAX_POINT_NUM_HDF5

def save_pcd(pcd):
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
    pcd_o3d.colors = o3d.utility.Vector3dVector(pcd[:, 3:])
    o3d.io.write_point_cloud("test.ply", pcd_o3d, write_ascii=True)

def convert_pose_from_hand_to_fingertip(ee_poses: dict) -> dict:
    offset = None
    corrected_hand_poss = dict()
    for frame_idx, hand_pos in ee_poses.items():
        # First frame: calculate corrective rotation
        if offset is None:
            default_pose = np.eye(4)
            default_pose[:3, :3] = R.from_euler('XYZ', [180, 0, 0], degrees=True).as_matrix()
            default_pose[:3, 3] = hand_pos[:3]

            init_hand_mat = np.eye(4)
            init_hand_mat[:3, :3] = R.from_quat(hand_pos[3:]).as_matrix()
            init_hand_mat[:3, 3] = hand_pos[:3]
            offset = np.linalg.inv(init_hand_mat) @ default_pose
            # Add translation along the hand's local coordinates by rotating the local vector
            local_trans = np.array([0.05, 0.01, 0.03])
            offset[:3, 3] = offset[:3, 3] + offset[:3, :3] @ local_trans


        # Apply corrective rotation (calculated from first frame)
        hand_mat = np.eye(4)
        hand_mat[:3, :3] = R.from_quat(hand_pos[3:]).as_matrix()
        hand_mat[:3, 3] = hand_pos[:3]

        corrected_hand_mat = hand_mat @ offset
        hand_pos[:3] = corrected_hand_mat[:3, 3]
        hand_pos[3:] = R.from_matrix(corrected_hand_mat[:3,:3]).as_quat()
        hand_pos[2] = np.clip(hand_pos[2], 0.0, None)  # prevent z from going below 0
        corrected_hand_poss[frame_idx] = hand_pos
    return corrected_hand_poss

class PointCloudProcessor:
    """Processes point clouds for dataset conversion."""

    def __init__(self, workspace: np.ndarray=WORKSPACE, fix_point_num: int=MAX_POINT_NUM_HDF5, robomimic_center: np.ndarray=None):
        """Initialize processor.

        Args:
            workspace: 3x2 array defining workspace boundaries
            fix_point_num: Target number of points after processing
            robomimic_center: Center point for robomimic coordinate system
        """
        self.workspace = workspace
        self.fix_point_num = fix_point_num
        self.robomimic_center = robomimic_center

    def process_raw_pcd(self, pcd: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, o3d.geometry.PointCloud]:
        """Process raw point cloud.

        Args:
            pcd: Raw point cloud array (N, 6) with xyz and rgb

        Returns:
            Tuple of (processed numpy array, processed o3d point cloud)
        """
        # Filter by workspace
        pcd_np = pcd[np.where(
            (pcd[:, 0] > self.workspace[0, 0]) & (pcd[:, 0] < self.workspace[0, 1]) &
            (pcd[:, 1] > self.workspace[1, 0]) & (pcd[:, 1] < self.workspace[1, 1]) &
            (pcd[:, 2] > self.workspace[2, 0]) & (pcd[:, 2] < self.workspace[2, 1])
        )]
        # filter z
        # pcd_np = pcd_np[pcd_np[:, 2] > 0.02]

        point_num = pcd_np.shape[0]
        assert point_num > 0, "Too few points in the point cloud after filtering."

        # render sphere
        pcd_np = self.get_render_pcd(pcd_np, pose)
        return pcd_np

    def downsample_pcd(self, pcd: np.ndarray) -> np.ndarray:
        point_num = pcd.shape[0]
        if point_num >= self.fix_point_num:
            # Farthest point down sample
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
            pcd_o3d.colors = o3d.utility.Vector3dVector(pcd[:, 3:])
            pcd_o3d = pcd_o3d.farthest_point_down_sample(self.fix_point_num)
            pcd = o3d2np(pcd_o3d)
        else:
            # Upsample by random selection
            extra_choice = np.random.choice(point_num, self.fix_point_num - point_num, replace=True)
            pcd = np.concatenate([pcd, pcd[extra_choice]], axis=0)
        return pcd

    def get_pcd_obs(self, pcd: np.ndarray, pose: np.ndarray, obs_type: str) -> tuple[np.ndarray, np.ndarray]:
        """Get point cloud observation.

        Args:
            pcd: Point cloud array (N, 6)
            pose: Pose array (7,)
            obs_type: 'pcd', 'pcd_t3', 'voxel', 'voxel_render'

        Returns:
            Tuple of (global pcd, local pcd)
        """
        # obs_dict = {}
        # if obs_type == 'voxel':
        #     global_obs = pcd_to_voxel(pcd[None, ...])[0]
        #     np_pcd_se3_rel = localize_pcd_batch(pcd[None,...], pose, local_type='se3')[0]
        #     local_obs = pcd_to_voxel(np_pcd_se3_rel[None,...], 'gripper')[0]
        # elif obs_type == 'pcd_se3':
        #     global_obs = crop_pcd(np_pcd_se3_rel, input_type='relative')
        #     np_pcd_se3_rel = localize_pcd_batch(pcd[None,...], pose, local_type='se3')[0]
        #     local_obs = crop_pcd(np_pcd_se3_rel, input_type='gripper')
        # elif obs_type == 'pcd_t3':
        #     np_pcd_se3_rel = localize_pcd_batch(pcd[None,...], pose, local_type='xyz')[0]
        #     global_obs = crop_pcd(np_pcd_se3_rel, input_type='relative')
        #     local_obs = crop_pcd(np_pcd_se3_rel, input_type='gripper')
        # elif obs_type == 'pcd':
        #     global_obs = crop_pcd(pcd, input_type='absolute')
        # else:
        #     raise NotImplementedError(f"Observation type {obs_type} not implemented.")
        
        global_obs = pcd
        return global_obs
            
    def get_render_pcd(self, pcd_no_robot: np.ndarray, ee_pose: np.ndarray) -> np.ndarray:
        """Get voxelized rendered point cloud with sphere.

        Args:
            pcd_no_robot: Point cloud without robot
            ee_pose: End-effector pose

        Returns:
            Voxelized point cloud
        """
        geco = render_pcd_from_pose(ee_pose, 1024, 'sphere')
        pcd_render = np.concatenate([pcd_no_robot, geco], axis=0)
        pcd_render = self.downsample_pcd(pcd_render)
        return pcd_render


class TrajectoryLoader:
    """Loads and processes trajectories from episodes."""

    def __init__(self, real_dataset_path: str, process_path: str, obs_type: str,
                 cam_list: list[str], main_cam: str, pcd_processor: PointCloudProcessor):
        """Initialize loader.

        Args:
            real_dataset_path: Path to real dataset
            process_path: Path to processed data
            cam_list: List of camera names
            main_cam: Main camera name
            pcd_processor: PointCloudProcessor instance
        """
        self.real_dataset_path = real_dataset_path
        self.process_path = process_path
        self.cam_list = cam_list
        self.main_cam = main_cam
        self.pcd_processor = pcd_processor
        self.obs_type = obs_type

    def get_traj_length(self, episode_name: str) -> int:
        """Get trajectory length for episode.

        Args:
            episode_name: Name of episode

        Returns:
            Number of frames in trajectory
        """
        episode_path = os.path.join(self.real_dataset_path, episode_name)
        traj_list = [
            f for f in os.listdir(os.path.join(episode_path, self.main_cam, "rgb"))
            if f.endswith(".png")
        ]
        traj_list.sort(key=lambda x: int(x.split(".")[0].split("_")[0]))
        return len(traj_list)

    def get_obs_from_episode(self, episode_path: str, cam: str,
                            frame_idx: str) -> tuple[np.ndarray, np.ndarray]:
        """Get RGB and depth observation.

        Args:
            episode_path: Path to episode
            cam: Camera name
            frame_idx: Frame index string

        Returns:
            Tuple of (rgb, depth)
        """
        rgb_path = os.path.join(episode_path, cam, "rgb", f"{frame_idx}.png")
        depth_path = os.path.join(episode_path, cam, "depth", f"{frame_idx}.npy")

        rgb = np.array(Image.open(rgb_path))
        depth = np.load(depth_path)

        return rgb, depth

    def get_pcd_from_episode(self, process_path: str,
                            frame_idx: str) -> tuple[np.ndarray, np.ndarray]:
        """Get point clouds from episode.

        Args:
            process_path: Path to processed data
            frame_idx: Frame index string

        Returns:
            Tuple of (pcd, pcd_no_robot)
        """
        pcd_path = os.path.join(process_path, "pcd", f"{frame_idx}.npy")
        pcd_no_robot_path = os.path.join(process_path, "pcd_no_hand", f"{frame_idx}.npy")

        pcd = np.load(pcd_path)
        pcd_no_robot = np.load(pcd_no_robot_path)

        
        return pcd, pcd_no_robot

    def load_trajectory(self, episode_name: str) -> dict:
        """Load complete trajectory.

        Args:
            episode_name: Name of episode

        Returns:
            Dictionary with obs, states, actions, rewards, dones
        """
        episode_path = os.path.join(self.real_dataset_path, episode_name)
        process_episode_path = os.path.join(self.process_path, episode_name)
        traj_length = self.get_traj_length(episode_name)

        ee_poss = np.load(
            os.path.join(process_episode_path, "hand_poses_wrt_world.npy"),
            allow_pickle=True
        )[()]
        ee_poss = convert_pose_from_hand_to_fingertip(ee_poss)
        grasps_state = np.load(os.path.join(process_episode_path, "grasp.npy"))[:,None]
        assert len(ee_poss) == len(grasps_state), "Mismatch in ee_poss and grasps signals."

        rgb_dict = {f'{cam}_image': [] for cam in self.cam_list}
        depth_dict = {f'{cam}_depth': [] for cam in self.cam_list}
        pcd_seq, local_pcd_seq, ee_pos_seq = [], [], []

        for frame_idx, pose in ee_poss.items():
            pcd, pcd_no_robot = self.get_pcd_from_episode(process_episode_path, frame_idx)

            for cam in self.cam_list:
                rgb, depth = self.get_obs_from_episode(episode_path, cam, frame_idx)
                rgb_dict[f'{cam}_image'].append(rgb)
                depth_dict[f'{cam}_depth'].append(depth)

            np_pcd = self.pcd_processor.process_raw_pcd(pcd, pose)
            np_pcd_no_robot = self.pcd_processor.process_raw_pcd(pcd_no_robot, pose)

            pcd_seq.append(np_pcd_no_robot)
            ee_pos_seq.append(pose)

        ee_pos_seq = np.stack(ee_pos_seq)
        # offset grasps by one timestep
        actions = convert_state_to_action(np.concatenate((ee_pos_seq, grasps_state), axis=-1))

        rewards = np.zeros((traj_length, 1), dtype=np.float32)
        rewards[-1] = 1.0
        dones = rewards.copy().astype(bool)

        state_dict = {
            'robot0_eef_pos': ee_pos_seq[:, :3].copy(),
            'robot0_eef_quat': ee_pos_seq[:, 3:7].copy(),
            'robot0_gripper_qpos': grasps_state.copy()
        }

        pcd_dict = {
           self.obs_type: np.stack(pcd_seq), 
        }

        obss = {**rgb_dict, **depth_dict, **state_dict, **pcd_dict, 'pcd': np.stack(pcd_seq)}

        return {
            'obs': obss,
            'states': ee_pos_seq,
            'actions': actions,
            'rewards': rewards,
            'dones': dones
        }
