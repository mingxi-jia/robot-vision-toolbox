'''
This script converts the real dataset to the robomimic format.
The real dataset is in the format of:
dataset/
    episode_0/
        cam0/
            rgb/
            depth/
        cam1/
            rgb/
            depth/
        ...
    episode_1/
        cam0/
            rgb/
            depth/
        cam1/
            rgb/
            depth/
        ...
    ...
'''
import os
import h5py
import glob
import json
import time
from time import perf_counter
import torch
import copy
import yaml
import open3d as o3d
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append("./")
from vision_utils.pcd_utils import convert_RGBD_to_open3d, o3d2np
from dataset_utils.hand_preprocessor import HandPreprocessor as Hamer
from human_segmentor.sphere_pcd import generate_pcd_sequence

from vision_utils.pcd_utils import get_extrinsics_matrix, pcd_to_voxel, render_pcd_from_pose

from configs.workspace import WORKSPACE, MAX_POINT_NUM, VOXEL_SIZE


def load_camera_info_dict(info_path: str):
    assert info_path.endswith(".yaml"), "Info file should be a yaml file"
    with open(info_path, 'r') as f:
        info_dict = yaml.safe_load(f)
    for cam_name, cam_info in info_dict.items():
        info_dict[cam_name]['intrinsics'] = np.array(cam_info['k']).reshape(3, 3)
        info_dict[cam_name]['extrinsics'] = get_extrinsics_matrix(cam_info['t'], cam_info['q'])
    return info_dict

def convert_state_to_action(ee_pose: np.ndarray):
    # Instead of converting to lists and stacking, we can directly use NumPy slicing and concatenation.
    # This appends the last row to the sliced array from index 1 onward.
    action = np.concatenate((ee_pose[1:], ee_pose[-1:]), axis=0)
    xyz = action[:, :3]  # Extract xyz positions
    quat = action[:, 3:7]  # Extract quaternion orientations
    gripper = action[:, 7:]  # Extract gripper states (if any)

    axis_angle = R.from_quat(quat).as_rotvec()  # Convert quaternion to axis-angle representation
    actions = np.concatenate((xyz, axis_angle, gripper), axis=-1)
    return actions

class RealToRobomimicConverter:
    def __init__(self, real_dataset_path: str, output_robomimic_path: str):
        self.episode_list = [f for f in os.listdir(os.path.join(real_dataset_path)) if f.startswith("episode")]
        cam_list = [f for f in os.listdir(os.path.join(real_dataset_path, self.episode_list[0])) if f.startswith("cam")]
        num_cams = len(cam_list)
        num_episodes = len([f for f in os.listdir(real_dataset_path) if f.startswith("episode_")])

        self.info_dict = load_camera_info_dict(os.path.join('configs', "camera_info.yaml"))

        self.hamer = Hamer(real_dataset_path) #TODO(ivy)

        self.real_dataset_path = real_dataset_path
        self.process_path = os.path.join(real_dataset_path, "output")
        self.robomimic_dataset_path = output_robomimic_path
        self.robomimic_center = np.array([0, 0, 0.7])

        self.main_cam = 'cam3'
        self.num_cams = num_cams
        self.cam_list = cam_list
        self.low_dim_list = ['robot0_eef_pos', 'robot0_eef_quat']
        self.num_episodes = num_episodes
        self.fix_point_num = MAX_POINT_NUM
        self.workspace = WORKSPACE


        print(f"Extracting actions from real dataset using HAMER...")
        self.preprocess(self.hamer)

    def preprocess(self, hamer: Hamer):
        preprocess_start_time = perf_counter()
        for episode_name in tqdm(self.episode_list, desc="Preprocessing episodes"):
            # done_indicator = os.path.join(self.process_path, episode_name, "DONE")
            done_indicator = os.path.join(self.process_path, episode_name, "hand_poses_wrt_world.npy")
            if os.path.exists(done_indicator):
                print(f"Episode {episode_name} already processed. Skipping...")
                continue

            starting_time = perf_counter()
            episode_path = os.path.join(self.real_dataset_path, episode_name)
            for cam_id in [1, 2, 3]:
                print(f"\n========= Processing Camera {cam_id} =========")
                hamer.process(episode_path, cam_id)
                torch.cuda.empty_cache()
            pcd_gen_start = perf_counter()
            # generate pcd with rendered sphere
            poses_wrt_world = generate_pcd_sequence(episode_path, self.hamer.process_path, self.info_dict, sphere_cam=3, segment=False, visualize_coordinate_axis=True)
            elapsed_segment_false = perf_counter() - pcd_gen_start
            # generate pcd with human in it
            pcd_gen_start = perf_counter()
            poses_wrt_world = generate_pcd_sequence(episode_path, self.hamer.process_path, self.info_dict, sphere_cam=3, segment=True, visualize_coordinate_axis=True)
            elapsed_segment_true = perf_counter() - pcd_gen_start
            pcd_gen_start = perf_counter()
            torch.cuda.empty_cache()
            np.save(os.path.join(self.process_path, episode_name, "hand_poses_wrt_world.npy"), poses_wrt_world, allow_pickle=True)
            elapsed_save = perf_counter() - pcd_gen_start
            print(f"pcd generation (segment=False) takes {elapsed_segment_false:.2f} seconds.")
            print(f"pcd generation (segment=True) takes {elapsed_segment_true:.2f} seconds.")
            print(f"pcd saving takes {elapsed_save:.2f} seconds.")
            print(f"Episode {episode_name} processed in {perf_counter() - starting_time:.2f} seconds.")

            with open(os.path.join(self.process_path, episode_name, 'DONE'), "a") as f:
                f.write("This is a marker file to indicate that the preprocessing is done for this episode.\n")
        print(f"Preprocessing done in {perf_counter() - preprocess_start_time:.2f} seconds.")


    def process_raw_pcd(self, pcd: np.ndarray):
        pcd_np = pcd[np.where((pcd[:, 0] > self.workspace[0, 0]) & (pcd[:, 0] < self.workspace[0, 1]) &
                             (pcd[:, 1] > self.workspace[1, 0]) & (pcd[:, 1] < self.workspace[1, 1]) &
                             (pcd[:, 2] > self.workspace[2, 0]) & (pcd[:, 2] < self.workspace[2, 1]))]
        
        # pcd_np[:, 0] -= (self.workspace[0, 0] + self.workspace[0, 1]) / 2  # Center X coordinate
        # pcd_np[:, 1] -= (self.workspace[1, 0] + self.workspace[1, 1]) / 2  # Center X coordinate
        # pcd_np[:, 2] += self.robomimic_center[2]  # Adjust Z coordinate to match the table height in robomimic

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
        pcd_o3d.colors = o3d.utility.Vector3dVector(pcd_np[:, 3:])

        point_num = pcd_np.shape[0]
        assert point_num > 1024, "Too few points in the point cloud after filtering."
        if pcd_np.shape[0] >= self.fix_point_num:
            #farthest point down sample to self.fix_point_num
            pcd_o3d = pcd_o3d.farthest_point_down_sample(self.fix_point_num)
        else:
            extra_choice = np.random.choice(point_num, self.fix_point_num-pcd.shape[0], replace=True)
            pcd = np.concatenate([pcd, pcd[extra_choice]], axis=0)

        pcd_np = o3d2np(pcd_o3d)
        return pcd_np, pcd_o3d

    def get_traj_length(self, episode_name: str):
        episode_path = os.path.join(self.real_dataset_path, episode_name)
        traj_list = [f for f in os.listdir(os.path.join(episode_path, self.main_cam, "rgb")) if f.endswith(".png")]
        # sort in terms of time based on file name "sec_nanosec.png"
        traj_list.sort(key=lambda x: int(x.split(".")[0].split("_")[0]))
        return len(traj_list)

    def get_obs_from_episode(self, episode_path: int, cam: str, frame_idx: str):
        rgb_path = os.path.join(episode_path, cam, "rgb", f"{frame_idx}.png")
        depth_path = os.path.join(episode_path, cam, "depth", f"{frame_idx}.npy")

        rgb = np.array(Image.open(rgb_path))
        depth = np.load(depth_path)

        return rgb, depth
    
    def get_pcd_from_episode(self, process_path, frame_idx: str):
        pcd_path = os.path.join(process_path, "pcd", f"{frame_idx}.ply")
        pcd_no_robot_path = os.path.join(process_path, "pcd_no_hand", f"{frame_idx}.ply")

        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd_no_robot = o3d.io.read_point_cloud(pcd_no_robot_path)

        return o3d2np(pcd), o3d2np(pcd_no_robot)
    
    def get_render_pcd(self, pcd_no_robot: np.ndarray, ee_pos: np.ndarray):
        geco = render_pcd_from_pose(ee_pos, 1024, 'sphere')
        pcd_render = np.concatenate([pcd_no_robot, geco], axis=0)
        np_voxels_render = pcd_to_voxel(pcd_render[None,...])[0]
        return np_voxels_render
    
    def load_trajectory(self, episode_name: str):
        episode_path = os.path.join(self.real_dataset_path, episode_name)
        process_episode_path = os.path.join(self.process_path, episode_name)
        traj_length = self.get_traj_length(episode_name)
        ee_poss: dict = np.load(os.path.join(process_episode_path, "hand_poses_wrt_world.npy"), allow_pickle=True)[()]
        
        rgb_dict = {f'{cam}_image': [] for cam in self.cam_list}
        depth_dict = {f'{cam}_depth': [] for cam in self.cam_list}
        pcd_seq, voxel_seq, voxel_render_seq, ee_pos_seq = [], [], [], []

        for frame_idx, pose in ee_poss.items():
            pcd, pcd_no_robot = self.get_pcd_from_episode(process_episode_path, frame_idx)
            for cam in self.cam_list:
                rgb, depth = self.get_obs_from_episode(episode_path, cam, frame_idx)

                rgb_dict[f'{cam}_image'].append(rgb)
                depth_dict[f'{cam}_depth'].append(depth)

            np_pcd, _ = self.process_raw_pcd(pcd)
            np_pcd_no_robot, _ = self.process_raw_pcd(pcd_no_robot)
            np_voxels = pcd_to_voxel(np_pcd[None,...])[0]
            np_voxels_render = self.get_render_pcd(np_pcd_no_robot, pose)

            pcd_seq.append(np_pcd)
            voxel_seq.append(np_voxels)
            voxel_render_seq.append(np_voxels_render)
            ee_pos_seq.append(pose)

        ee_pos_seq = np.stack(ee_pos_seq)
        gripper_state_seq = np.zeros((len(ee_pos_seq), 1), dtype=np.float32)  #TODO(mingxi)
        actions = convert_state_to_action(np.concatenate((ee_pos_seq, gripper_state_seq), axis=-1))  # Convert ee_pos to actions
        
        rewards = np.zeros((traj_length, 1), dtype=np.float32)
        rewards[-1] = 1.0
        dones = rewards.copy().astype(bool)

        state_dict = dict()
        state_dict['robot0_eef_pos'] = ee_pos_seq[:, :3].copy()  # Assuming first 3 are position
        state_dict['robot0_eef_quat'] = ee_pos_seq[:, 3:7].copy()
        state_dict['robot0_gripper_qpos'] = gripper_state_seq.copy()  # Assuming gripper state is the last element

        voxel_dict = dict()
        voxel_dict['voxel'] = np.stack(voxel_seq).astype(np.uint8) 
        voxel_dict['voxel_render'] = np.stack(voxel_render_seq).astype(np.uint8)

        obss = {**rgb_dict, **depth_dict, **state_dict, **voxel_dict, 'pcd': np.stack(pcd_seq)}
        # merge obs and rgb_dict

        trajectory = {
            'obs': obss,
            'states': ee_pos_seq,
            'actions': actions,
            'rewards': rewards,
            'dones': dones
        }


        return trajectory
    
    def convert(self):
        '''
            Robomimic format (dataset.hdf5):
            data/
                demo_0/
                    actions: (N, 7)  # xyz, axis-angle, gripper
                    states: (N, 8)  # xyz, quat, gripper
                    rewards: (N, 1)  # scalar rewards
                    dones: (N, 1)  # boolean done flags
                    obs/
                        cam0_image: (N, H, W, 3)  # RGB images
                        cam0_depth: (N, H, W)  # Depth images
                        cam1_image: (N, H, W, 3)  # RGB images
                        cam1_depth: (N, H, W)  # Depth images
                        ...
                        pcd: (N, M, 6)  # Point cloud data (M points, 6 channels: x, y, z, r, g, b)
            where
            - N is the number of time steps in the episode
            - M is the number of points in the point cloud (fixed to self.fix_point_num)
            - H and W are the height and width of the RGB and depth images (e.g., 480, 640)
            - The actions and states are assumed to be in the format
              [x, y, z, axis-angle (qx, qy, qz, qw), gripper]
            - The obs dictionary contains the RGB and depth images for each camera,
              as well as the point cloud data.
            - The rewards and dones are scalar values for each time step.
            - The states are assumed to be in the format
        '''
        print(f"Converting data to robomimic format...")
        f_out = h5py.File(self.robomimic_dataset_path, "w")
        data_grp = f_out.create_group("data")

        bar = tqdm(total=len(self.episode_list), desc="Converting episodes")
        for episode_idx, episode_name in enumerate(self.episode_list):
            traj = self.load_trajectory(episode_name)

            ep = f"demo_{episode_idx}"

            ep_data_grp = data_grp.create_group(ep)
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            for k in traj["obs"].keys():
                data = np.array(traj["obs"][k])
                assert data.dtype != np.dtype('O'), "Data type should not be object, but got {}".format(data.dtype)
                ep_data_grp.create_dataset("obs/{}".format(k), data=data, compression="gzip")

            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
            tqdm.write("ep {}: wrote {} transitions to group {}".format(
            episode_idx, ep_data_grp.attrs["num_samples"], ep))
            bar.update(1)
        bar.close()

if __name__ == "__main__":
    # converter = RealToRobomimicConverter(real_dataset_path="/home/mingxi/data/realworld/test", output_robomimic_path="/home/mingxi/data/realworld/hdf5_hand_datasets/test_multiview_abs.hdf5")
    # converter = RealToRobomimicConverter(real_dataset_path="/home/mingxi/data/realworld/test", output_robomimic_path="/home/mingxi/data/realworld/test_multiview_abs.hdf5")
    converter = RealToRobomimicConverter(real_dataset_path="/media/mingxi/home2/data/real world/human_blockStacking_60", output_robomimic_path="/media/mingxi/home2/data/real world/test_multiview_abs.hdf5")
    converter.convert()
    
