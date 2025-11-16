"""Trajectory loading and processing."""

import os
import numpy as np
import open3d as o3d
from PIL import Image
from hand.hand_utils import convert_state_to_action

from robomimic.utils.obs_utils import (depth2fgpcd, np2o3d, o3d2np, pcd_to_voxel, localize_pcd_batch, 
                                       enlarge_mask, crop_pcd, get_clipspace, get_workspace, get_pcd_z_min)

class PointCloudProcessor:
    """Processes point clouds for dataset conversion."""

    def __init__(self, workspace: np.ndarray, fix_point_num: int, robomimic_center: np.ndarray):
        """Initialize processor.

        Args:
            workspace: 3x2 array defining workspace boundaries
            fix_point_num: Target number of points after processing
            robomimic_center: Center point for robomimic coordinate system
        """
        self.workspace = workspace
        self.fix_point_num = fix_point_num
        self.robomimic_center = robomimic_center

    def process_raw_pcd(self, pcd: np.ndarray) -> tuple[np.ndarray, o3d.geometry.PointCloud]:
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

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
        pcd_o3d.colors = o3d.utility.Vector3dVector(pcd_np[:, 3:])

        point_num = pcd_np.shape[0]
        assert point_num > 1024, "Too few points in the point cloud after filtering."

        if pcd_np.shape[0] >= self.fix_point_num:
            # Farthest point down sample
            pcd_o3d = pcd_o3d.farthest_point_down_sample(self.fix_point_num)
        else:
            # Upsample by random selection
            extra_choice = np.random.choice(point_num, self.fix_point_num - pcd.shape[0], replace=True)
            pcd = np.concatenate([pcd, pcd[extra_choice]], axis=0)

        pcd_np = o3d2np(pcd_o3d)
        return pcd_np, pcd_o3d

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
        
        global_obs = crop_pcd(pcd, input_type='absolute')
        return global_obs
            
    def get_render_pcd(self, pcd_no_robot: np.ndarray, ee_pos: np.ndarray) -> np.ndarray:
        """Get voxelized rendered point cloud with sphere.

        Args:
            pcd_no_robot: Point cloud without robot
            ee_pos: End-effector position

        Returns:
            Voxelized point cloud
        """
        geco = render_pcd_from_pose(ee_pos, 1024, 'sphere')
        pcd_render = np.concatenate([pcd_no_robot, geco], axis=0)
        np_voxels_render = pcd_to_voxel(pcd_render[None, ...])[0]
        return np_voxels_render


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
        pcd_no_robot = np.load(pcd_path)

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
        grasps = np.load(os.path.join(process_episode_path, "grasp.npy"))
        assert len(ee_poss) == len(grasps), "Mismatch in ee_poss and grasps signals."

        rgb_dict = {f'{cam}_image': [] for cam in self.cam_list}
        depth_dict = {f'{cam}_depth': [] for cam in self.cam_list}
        pcd_seq, local_pcd_seq, ee_pos_seq = [], [], []

        for frame_idx, pose in ee_poss.items():
            pcd, pcd_no_robot = self.get_pcd_from_episode(process_episode_path, frame_idx)

            for cam in self.cam_list:
                rgb, depth = self.get_obs_from_episode(episode_path, cam, frame_idx)
                rgb_dict[f'{cam}_image'].append(rgb)
                depth_dict[f'{cam}_depth'].append(depth)

            np_pcd, _ = self.pcd_processor.process_raw_pcd(pcd)
            np_pcd_no_robot, _ = self.pcd_processor.process_raw_pcd(pcd_no_robot)

            pcd_seq.append(np_pcd_no_robot)
            ee_pos_seq.append(pose)

        ee_pos_seq = np.stack(ee_pos_seq)
        actions = convert_state_to_action(np.concatenate((ee_pos_seq, grasps), axis=-1))

        rewards = np.zeros((traj_length, 1), dtype=np.float32)
        rewards[-1] = 1.0
        dones = rewards.copy().astype(bool)

        state_dict = {
            'robot0_eef_pos': ee_pos_seq[:, :3].copy(),
            'robot0_eef_quat': ee_pos_seq[:, 3:7].copy(),
            'robot0_gripper_qpos': gripper_state_seq.copy()
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
