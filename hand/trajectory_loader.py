"""Trajectory loading and processing."""

import os
import numpy as np
import open3d as o3d
from PIL import Image
from vision_utils.pcd_utils import o3d2np, pcd_to_voxel
from hand.utils import convert_state_to_action


class TrajectoryLoader:
    """Loads and processes trajectories from episodes."""

    def __init__(self, real_dataset_path: str, process_path: str,
                 cam_list: list[str], main_cam: str, pcd_processor):
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

        rgb_dict = {f'{cam}_image': [] for cam in self.cam_list}
        depth_dict = {f'{cam}_depth': [] for cam in self.cam_list}
        pcd_seq, voxel_seq, voxel_render_seq, ee_pos_seq = [], [], [], []

        for frame_idx, pose in ee_poss.items():
            pcd, pcd_no_robot = self.get_pcd_from_episode(process_episode_path, frame_idx)

            for cam in self.cam_list:
                rgb, depth = self.get_obs_from_episode(episode_path, cam, frame_idx)
                rgb_dict[f'{cam}_image'].append(rgb)
                depth_dict[f'{cam}_depth'].append(depth)

            np_pcd, _ = self.pcd_processor.process_raw_pcd(pcd)
            np_pcd_no_robot, _ = self.pcd_processor.process_raw_pcd(pcd_no_robot)
            np_voxels = pcd_to_voxel(np_pcd[None, ...])[0]
            np_voxels_render = self.pcd_processor.get_render_pcd(np_pcd_no_robot, pose)

            pcd_seq.append(np_pcd)
            voxel_seq.append(np_voxels)
            voxel_render_seq.append(np_voxels_render)
            ee_pos_seq.append(pose)

        ee_pos_seq = np.stack(ee_pos_seq)
        gripper_state_seq = np.zeros((len(ee_pos_seq), 1), dtype=np.float32)
        actions = convert_state_to_action(np.concatenate((ee_pos_seq, gripper_state_seq), axis=-1))

        rewards = np.zeros((traj_length, 1), dtype=np.float32)
        rewards[-1] = 1.0
        dones = rewards.copy().astype(bool)

        state_dict = {
            'robot0_eef_pos': ee_pos_seq[:, :3].copy(),
            'robot0_eef_quat': ee_pos_seq[:, 3:7].copy(),
            'robot0_gripper_qpos': gripper_state_seq.copy()
        }

        voxel_dict = {
            'voxel': np.stack(voxel_seq).astype(np.uint8),
            'voxel_render': np.stack(voxel_render_seq).astype(np.uint8)
        }

        obss = {**rgb_dict, **depth_dict, **state_dict, **voxel_dict, 'pcd': np.stack(pcd_seq)}

        return {
            'obs': obss,
            'states': ee_pos_seq,
            'actions': actions,
            'rewards': rewards,
            'dones': dones
        }
