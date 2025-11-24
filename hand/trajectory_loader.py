"""Trajectory loading and processing."""

import os
from matplotlib import axis
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R, Slerp
from hand.hand_utils import convert_state_to_action

from robot_filter.arm_segmentor import RobotArmSegmentation
from utils.pcd_utils import (depth2fgpcd, np2o3d, o3d2np, pcd_to_voxel, render_pcd_from_pose, convert_RGBD_fast)
from configs.workspace import WORKSPACE, MAX_POINT_NUM_HDF5
from tqdm import tqdm

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
            local_trans = np.array([0.06, 0.0, 0.03])
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

def convert_pose_from_robot_to_fingertip(ee_poses: dict) -> dict:
    corrected_hand_poss = []
    for frame_idx, hand_pos in enumerate(ee_poses):
        # Apply corrective rotation (calculated from first frame)
        hand_mat = np.eye(4)
        hand_mat[:3, :3] = R.from_quat(hand_pos[3:]).as_matrix()
        hand_mat[:3, 3] = hand_pos[:3]

        # Apply fixed translation offset from robot EE to hand fingertip
        eTf = np.eye(4)
        eTf[:3,3] = np.array([0.0, 0.0, 0.06])

        corrected_hand_mat = hand_mat @ eTf
        hand_pos[:3] = corrected_hand_mat[:3, 3]
        hand_pos[3:] = R.from_matrix(corrected_hand_mat[:3,:3]).as_quat()
        hand_pos[2] = np.clip(hand_pos[2], 0.0, None)  # prevent z from going below 0
        corrected_hand_poss.append(hand_pos)
         
    return np.array(corrected_hand_poss)

class ObservationProcessor:
    """Processes point clouds for dataset conversion."""

    def __init__(self, workspace: np.ndarray=WORKSPACE, fix_point_num: int=MAX_POINT_NUM_HDF5, data_type: str="robot"):
        """Initialize processor.

        Args:
            workspace: 3x2 array defining workspace boundaries
            fix_point_num: Target number of points after processing
        """
        self.workspace = workspace
        self.fix_point_num = fix_point_num
        if data_type == "robot":
            self.robot_filter = RobotArmSegmentation()

    def filter_pcd_by_workspace(self, pcd: np.ndarray) -> np.ndarray:
        """Filter point cloud by workspace boundaries.

        Args:
            pcd: Input point cloud array (N, 6) with xyz and rgb

        Returns:
            Filtered point cloud array
        """
        pcd_np = pcd[np.where(
            (pcd[:, 0] > self.workspace[0, 0]) & (pcd[:, 0] < self.workspace[0, 1]) &
            (pcd[:, 1] > self.workspace[1, 0]) & (pcd[:, 1] < self.workspace[1, 1]) &
            (pcd[:, 2] > self.workspace[2, 0]) & (pcd[:, 2] < self.workspace[2, 1])
        )]
        pcd_np = pcd_np[pcd_np[:, 2] > 0.02]
        return pcd_np
    
    def process_raw_pcd(self, pcd: np.ndarray, pose: np.ndarray=None, render: bool=False) -> tuple[np.ndarray, o3d.geometry.PointCloud]:
        """Process raw point cloud.

        Args:
            pcd: Raw point cloud array (N, 6) with xyz and rgb

        Returns:
            processed numpy array
        """
        # Filter by workspace
        pcd_np = self.filter_pcd_by_workspace(pcd)

        point_num = pcd_np.shape[0]
        assert point_num > 0, "Too few points in the point cloud after filtering."

        if render:
            assert pose is not None, "Pose must be provided for rendering."
            # render sphere
            pcd_np = self.get_render_pcd(pcd_np, pose)
        
        pcd_np = self.downsample_pcd(pcd_np)
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
    
    def resize_image(self, image: np.ndarray, target_size: tuple[int, int]=(84,84)) -> np.ndarray:
        h, w = image.shape[:2]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        cropped = image[top:top+min_dim, left:left+min_dim]
        if image.ndim == 2:  # depth image (single channel)
            resized = np.array(Image.fromarray(cropped).resize(target_size, Image.BILINEAR))
        else:  # RGB image (3 channels)
            resized = np.array(Image.fromarray(cropped).resize(target_size, Image.BILINEAR))
        return resized
    
    def localize_wrist_cam(self, rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        depth_threshold = 0.3
        mask = depth > depth_threshold
        if np.mean(mask) > 0.5:
            rgb = np.zeros_like(rgb)
            depth = np.ones_like(depth) * depth_threshold
        return rgb, depth
    
    def get_policy_images(self, rgb_dict: dict, depth_dict: dict) -> tuple[dict, dict]:
        for cam in rgb_dict.keys():
            rgb_resized = self.resize_image(rgb_dict[cam])
            depth_resized = self.resize_image(depth_dict[cam])

            if cam == 'cam4':
                rgb_resized, depth_resized = self.localize_wrist_cam(rgb_resized, depth_resized)

            rgb_dict[cam] = rgb_resized
            depth_dict[cam] = depth_resized

        return rgb_dict, depth_dict
            
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
        return pcd_render


    def get_policy_obs(self, pcd, pose, joint):
        np_pcd = self.process_raw_pcd(pcd, pose, render=False)

        np_pcd = self.filter_pcd_by_workspace(np_pcd)
        pcd_no_robot = self.robot_filter.segment(np_pcd, joint)
        render_pcd = self.process_raw_pcd(pcd_no_robot, pose, render=True)
        return np_pcd, render_pcd

class TrajectoryLoader:
    """Loads and processes trajectories from episodes."""

    def __init__(self, real_dataset_path: str, process_path: str, data_type: str, camera_info: dict,
                 cam_list: list[str], main_cam: str, obs_processor: ObservationProcessor):
        """Initialize loader.

        Args:
            real_dataset_path: Path to real dataset
            process_path: Path to processed data
            cam_list: List of camera names
            main_cam: Main camera name
            obs_processor: ObservationProcessor instance
        """

        self.real_dataset_path = real_dataset_path
        self.process_path = process_path
        self.cam_list = cam_list
        # exclude cam4 from pcd list
        self.pcd_cam_list = [cam for cam in cam_list if cam != 'cam4']
        print(f"Manually setting pcd cameras to: {self.pcd_cam_list} to exclude cam4 (wrist cam).")
        self.main_cam = main_cam
        self.obs_processor = obs_processor
        self.data_type = data_type
        self.camera_info = camera_info

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
            frame_idx: Frame index integer

        Returns:
            Tuple of (rgb, depth)
        """
        # camera frames are save as 'timestamp'.png so we need to find the correct file
        image_files = sorted([f for f in os.listdir(os.path.join(episode_path, cam, "rgb")) if f.endswith(".png")])
        depth_files = sorted([f for f in os.listdir(os.path.join(episode_path, cam, "depth")) if f.endswith(".npy")])
        rgb_path = os.path.join(episode_path, cam, "rgb", image_files[frame_idx])
        depth_path = os.path.join(episode_path, cam, "depth", depth_files[frame_idx])

        rgb = np.array(Image.open(rgb_path))
        depth = np.load(depth_path) / 1000.0  # convert mm to meters

        return rgb, depth

    def get_joint_state_from_episode(self, episode_path: str, frame_idx: str) -> np.ndarray:
        """Get joint state from episode.

        Args:
            episode_path: Path to episode
            frame_idx: Frame index string

        Returns:
            Joint state array
        """
        joint_state_path = os.path.join(episode_path, "joint_states.npy")
        return np.load(joint_state_path)[frame_idx]

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
        if self.data_type == "hand":
            return self.load_trajectory_hand(episode_name)
        else:
            return self.load_trajectory_robot(episode_name)

    def get_pcd_from_rgbd(self, episode_path: str, frame_idx: str) -> tuple[np.ndarray, np.ndarray]:
        """Get point clouds from rgbd"""
        pcds = []
        rgbs, depths = {}, {}
        for cam in self.cam_list:
            rgb, depth = self.get_obs_from_episode(episode_path, cam, frame_idx)
            rgbs[cam] = rgb
            depths[cam] = depth
            if cam in self.pcd_cam_list:
                intrinsics, extrinsics = self.camera_info[cam]['intrinsics'], self.camera_info[cam]['extrinsics']
                pcd_cam = convert_RGBD_fast(rgb, depth, intrinsics, extrinsics)
                pcds.append(pcd_cam)
        pcd = np.concatenate(pcds, axis=0)

        return pcd, rgbs, depths

    def check_integrity(self, episode_path: str) -> None:
        """Check integrity of episode data. It need to have cam1, cam2, cam3, grasp.npy, joint_states.npy, pose_wrt_world.npy

        Args:
            episode_path: Path to episode
        """
        required_files = [self.main_cam, "grasp.npy", "joint_states.npy", "pose_wrt_world.npy"]
        for f in required_files:
            if f.startswith("cam"):
                cam_path = os.path.join(episode_path, f)
                if not os.path.exists(cam_path):
                    raise FileNotFoundError(f"Camera folder {f} not found in episode {episode_path}")
            else:
                file_path = os.path.join(episode_path, "state", f)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required file {f} not found in episode {file_path}")
    
    
    def load_trajectory_robot(self, episode_name: str) -> dict:
        """Load robot trajectory.

        Args:
            episode_name: Name of episode

        Returns:
            Dictionary with obs, states, actions, rewards, dones
        """
        episode_path = os.path.join(self.real_dataset_path, episode_name)
        self.check_integrity(episode_path)
        traj_length = self.get_traj_length(episode_name)

        ee_poss = np.load(os.path.join(episode_path, "state", "pose_wrt_world.npy"))
        ee_poss = convert_pose_from_robot_to_fingertip(ee_poss)
        joint_states = np.load(os.path.join(episode_path, "state", "joint_states.npy"))
        grasps_state = np.load(os.path.join(episode_path, "state", "grasp.npy"))[:,None]
        assert len(ee_poss) == len(grasps_state), "Mismatch in ee_poss and grasps signals."

        rgb_dict = {f'{cam}_image': [] for cam in self.cam_list}
        depth_dict = {f'{cam}_depth': [] for cam in self.cam_list}
        pcd_seq, render_pcd_seq, ee_pos_seq = [], [], []

        for frame_idx, pose in tqdm(enumerate(ee_poss), total=traj_length, desc=f"Loading {episode_name}"):
            joint = joint_states[frame_idx][1:]
            pcd, rgbs, depths = self.get_pcd_from_rgbd(episode_path, frame_idx)
            
            rgbs, depths = self.obs_processor.get_policy_images(rgbs, depths)   
            np_pcd, np_pcd_no_robot = self.obs_processor.get_policy_obs(pcd, pose, joint)
            

            for cam in self.cam_list:
                rgb, depth = rgbs[cam], depths[cam]
                rgb_dict[f'{cam}_image'].append(rgb)
                depth_dict[f'{cam}_depth'].append(depth)
            pcd_seq.append(np_pcd)
            render_pcd_seq.append(np_pcd_no_robot)
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
            'robot0_gripper_qpos': grasps_state.copy().repeat(2, axis=1), # repeat to match mimicgen format
        }

        pcd_dict = {
            'pcd': np.stack(pcd_seq),
            'render_pcd': np.stack(render_pcd_seq), 
        }
        # rename cam4 to wrist_cam for clarity
        if 'cam4_image' in rgb_dict:
            rgb_dict['robot0_eye_in_hand_image'] = rgb_dict.pop('cam4_image')
        if 'cam4_depth' in depth_dict:
            depth_dict['robot0_eye_in_hand_depth'] = depth_dict.pop('cam4_depth')

        obss = {**rgb_dict, **depth_dict, **state_dict, **pcd_dict, 'pcd': np.stack(pcd_seq)}

        return {
            'obs': obss,
            'states': ee_pos_seq,
            'actions': actions,
            'rewards': rewards,
            'dones': dones
        }
    
    def load_trajectory_hand(self, episode_name: str) -> dict:
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
        pcd_seq, render_pcd_seq, ee_pos_seq = [], [], []
        
        for i, (frame_idx, pose) in enumerate(ee_poss.items()):
            pcd, pcd_no_robot = self.get_pcd_from_episode(process_episode_path, frame_idx)

            for cam in self.cam_list:
                rgb, depth = self.get_obs_from_episode(episode_path, cam, i)
                rgb_dict[f'{cam}_image'].append(rgb)
                depth_dict[f'{cam}_depth'].append(depth)

            np_pcd = self.obs_processor.process_raw_pcd(pcd, pose)
            np_pcd_no_robot = self.obs_processor.process_raw_pcd(pcd_no_robot, pose, render=True)
            
            pcd_seq.append(np_pcd)
            render_pcd_seq.append(np_pcd_no_robot)
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
            'robot0_gripper_qpos': grasps_state.copy().repeat(2, axis=1), # repeat to match mimicgen format
        }

        pcd_dict = {
            'pcd': np.stack(pcd_seq),
            'render_pcd': np.stack(render_pcd_seq), 
        }

        obss = {**rgb_dict, **depth_dict, **state_dict, **pcd_dict, 'pcd': np.stack(pcd_seq)}

        return {
            'obs': obss,
            'states': ee_pos_seq,
            'actions': actions,
            'rewards': rewards,
            'dones': dones
        }
