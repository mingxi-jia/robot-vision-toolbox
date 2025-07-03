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
import copy
import yaml
import open3d as o3d

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from vision_utils.pcd_utils import convert_RGBD_to_open3d, o3d2np
from hand_preprocessor import HandPreprocessor as Hamer
from human_segmentor.sphere_pcd import generate_pcd_sequence

def load_info_dict(info_path: str):
    assert info_path.endswith(".yaml"), "Info file should be a yaml file"
    with open(info_path, 'r') as f:
        info_dict = yaml.safe_load(f)
    return info_dict

def convert_state_to_action(states: list):
    action = copy.deepcopy(states[1:])
    action.append(states[-1])
    action = np.stack(action)
    xyz = action[:, :3]  # Extract xyz positions
    quat = action[:, 3:7]  # Extract quaternion orientations
    gripper = action[:, 7:]  # Extract gripper states (if any)

    axis_angle = R.from_quat(quat).as_rotvec()  # Convert quaternion to axis-angle representation
    actions = np.concatenate((xyz, axis_angle, gripper), axis=-1)
    return actions

class RealToRobomimicConverter:
    def __init__(self, real_dataset_path: str, output_robomimic_path: str):
        cam_list = [f for f in os.listdir(os.path.join(real_dataset_path, "episode_0")) if f.startswith("cam")]
        num_cams = len(cam_list)
        num_episodes = len([f for f in os.listdir(real_dataset_path) if f.startswith("episode_")])

        self.info_dict = load_info_dict(os.path.join(real_dataset_path, "camera_info.yaml"))

        self.hamer = Hamer() #TODO(ivy)
        
        

        self.real_dataset_path = real_dataset_path
        self.robomimic_dataset_path = output_robomimic_path
        self.robomimic_center = np.array([0, 0, 0.7])

        self.main_cam = 'cam3'
        self.num_cams = num_cams
        self.cam_list = cam_list
        self.low_dim_list = ['robot0_eef_pos', 'robot0_eef_quat']
        self.num_episodes = num_episodes
        self.fix_point_num = 4412
        self.workspace = np.array([[0.2, 0.6], 
                                   [-0.2, 0.2], 
                                   [0.0, 0.4]])


        print(f"Extracting actions from real dataset using HAMER...")
        self.preprocess(self.hamer)

    def preprocess(self, hamer: Hamer):
        for episode_idx in tqdm(range(self.num_episodes), desc="Preprocessing episodes"):
            episode_path = os.path.join(self.real_dataset_path, f"episode_{episode_idx}")
            for cam_id in [1, 2, 3]:
                print(f"\n========= Processing Camera {cam_id} =========")
                hamer.process(episode_path, cam_id)
            generate_pcd_sequence(episode_path, start_frame=0, sphere_cam=3)
            # self.extract_actions(episode_idx)
            # self.extract_pcds(episode_idx)

    def extract_actions(self, episode_idx):
        episode_path = os.path.join(self.real_dataset_path, f"episode_{episode_idx}")
        traj_length = self.get_traj_length(episode_idx)

        states = []
        rewards = np.zeros(traj_length)
        for frame_idx in range(traj_length):
            rgb_path = os.path.join(episode_path, self.main_cam, "rgb", f"{frame_idx}.png")
            depth_path = os.path.join(episode_path, self.main_cam, "depth", f"{frame_idx}.npy")

            rgb = np.array(Image.open(rgb_path))
            depth = np.load(depth_path)
            
            hand_pose = self.hamer.get_hand_pose(rgb, depth) #TODO(ivy)
            assert hand_pose.shape == (7,), f"Hand pose shape is {hand_pose.shape} but should be (7,), aka [x, y, z, qx, qy, qz, qw]"

            states.append(hand_pose)

        # shift states by 1 time step
        actions = convert_state_to_action(states)
        states = np.stack(states)
        
        rewards[-1] = 1
        dones = rewards.copy().astype(bool)

        np.save(os.path.join(episode_path, "states.npy"), states)
        np.save(os.path.join(episode_path, "actions.npy"), actions)
        np.save(os.path.join(episode_path, "rewards.npy"), rewards)
        np.save(os.path.join(episode_path, "dones.npy"), dones)

    def process_raw_pcd(self, pcd: np.ndarray):
        pcd_np = pcd[np.where((pcd[:, 0] > self.workspace[0, 0]) & (pcd[:, 0] < self.workspace[0, 1]) &
                             (pcd[:, 1] > self.workspace[1, 0]) & (pcd[:, 1] < self.workspace[1, 1]) &
                             (pcd[:, 2] > self.workspace[2, 0]) & (pcd[:, 2] < self.workspace[2, 1]))]
        
        pcd_np[:, 0] -= (self.workspace[0, 0] + self.workspace[0, 1]) / 2  # Center X coordinate
        pcd_np[:, 1] -= (self.workspace[1, 0] + self.workspace[1, 1]) / 2  # Center X coordinate
        pcd_np[:, 2] += self.robomimic_center[2]  # Adjust Z coordinate to match the table height in robomimic

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
        pcd_o3d.colors = o3d.utility.Vector3dVector(pcd_np[:, 3:])

        #farthest point down sample to self.fix_point_num
        pcd_o3d = pcd_o3d.farthest_point_down_sample(self.fix_point_num)

        pcd_np = o3d2np(pcd_o3d)
        return pcd_np, pcd_o3d

    def extract_pcds(self, episode_idx):
        episode_path = os.path.join(self.real_dataset_path, f"episode_{episode_idx}")
        traj_length = self.get_traj_length(episode_idx)

        os.makedirs(os.path.join(episode_path, "pcds"), exist_ok=True)

        for frame_idx in range(traj_length):
            step_pcd = []
            for cam in self.cam_list:
                rgb_path = os.path.join(episode_path, cam, "rgb", f"{frame_idx}.png")
                depth_path = os.path.join(episode_path, cam, "depth", f"{frame_idx}.npy")

                rgb = np.array(Image.open(rgb_path))
                depth = np.load(depth_path)

                pcd_np, _ = convert_RGBD_to_open3d(rgb, depth, self.info_dict[cam]['intrinsics'], self.info_dict[cam]['extrinsics'])
                step_pcd.append(pcd_np)

            step_pcd = np.concatenate(step_pcd, axis=0)
            _, pcd_o3d = self.process_raw_pcd(step_pcd)
            o3d.io.write_point_cloud(os.path.join(episode_path, "pcds", f"{frame_idx}.ply"), pcd_o3d)
        

    def get_traj_length(self, episode_idx: int):
        episode_path = os.path.join(self.real_dataset_path, f"episode_{episode_idx}")
        traj_list = [f for f in os.listdir(os.path.join(episode_path, self.main_cam, "rgb")) if f.endswith(".png")]
        # sort in terms of time based on file name "sec_nanosec.png"
        traj_list.sort(key=lambda x: int(x.split(".")[0].split("_")[0]))
        return len(traj_list)

    def load_trajectory(self, episode_idx: int):
        episode_path = os.path.join(self.real_dataset_path, f"episode_{episode_idx}")
        traj_length = self.get_traj_length(episode_idx)
        
        rgb_dict = {cam: [] for cam in self.cam_list}
        depth_dict = {cam: [] for cam in self.cam_list}
        pcd_seq = []
        for frame_idx in range(traj_length):
            pcds = []
            for cam in self.cam_list:
                rgb_path = os.path.join(episode_path, cam, "rgb", f"{frame_idx}.png")
                depth_path = os.path.join(episode_path, cam, "depth", f"{frame_idx}.npy")
                pcd_path = os.path.join(episode_path, cam, "pcd", f"{frame_idx}.ply")

                rgb = np.array(Image.open(rgb_path))
                depth = np.load(depth_path)
                pcd = o3d.io.read_point_cloud(pcd_path)

                rgb_dict[cam].append(rgb)
                depth_dict[cam].append(depth)
                pcds.append(pcd)

            pcd_seq.append(np.concatenate(pcds, axis=0))


        actions = np.load(os.path.join(episode_path, "actions.npy"))
        states = np.load(os.path.join(episode_path, "states.npy"))
        rewards = np.load(os.path.join(episode_path, "rewards.npy"))
        dones = np.load(os.path.join(episode_path, "dones.npy"))

        obs = {**rgb_dict, **depth_dict, 'pcd': np.stack(pcd_seq)}

        trajectory = {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'dones': dones
        }

        return trajectory

    def load_trajectory_dummy(self, episode_idx: int):
        episode_path = os.path.join(self.real_dataset_path, f"episode_{episode_idx}")
        traj_length = self.get_traj_length(episode_idx)
        
        rgb_dict = {f'{cam}_image': [] for cam in self.cam_list}
        depth_dict = {f'{cam}_depth': [] for cam in self.cam_list}
        pcd_seq = []
        for frame_idx in range(traj_length):
            pcds = []
            for cam in self.cam_list:
                rgb = np.zeros((480, 640, 3), dtype=np.uint8) # Dummy RGB
                depth = np.zeros((480, 640), dtype=np.float32) # Dummy Depth
                pcd = np.zeros((self.fix_point_num, 6), dtype=np.float32) # Dummy PCD
                pcd[:, :3] = np.random.uniform(self.workspace[:, 0], self.workspace[:, 1], size=(self.fix_point_num, 3))  # Random points in workspace

                rgb_dict[f'{cam}_image'].append(rgb)
                depth_dict[f'{cam}_depth'].append(depth)
                pcds.append(pcd)

            p = np.concatenate(pcds, axis=0)
            pcd_seq.append(self.process_raw_pcd(p)[0])  # Processed PCD


        actions = np.zeros((traj_length, 7), dtype=np.float32) # xyz, axis-angle, gripper
        states = np.zeros((traj_length, 10), dtype=np.float32)
        rewards = np.zeros((traj_length, 1), dtype=np.float32)
        rewards[-1] = 1.0
        dones = rewards.copy().astype(bool)

        state_dict = dict()
        state_dict['robot0_eef_pos'] = np.zeros((traj_length, 3), dtype=np.float32)  # Assuming first 3 are position
        state_dict['robot0_eef_quat'] = np.zeros((traj_length, 4), dtype=np.float32)
        obs = {**rgb_dict, **depth_dict, **state_dict, 'pcd': np.stack(pcd_seq)}
        # merge obs and rgb_dict


        trajectory = {
            'obs': obs,
            'states': states,
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

        for episode_idx in tqdm(range(self.num_episodes), desc="Converting episodes"):
            traj = self.load_trajectory_dummy(episode_idx)

            ep = f"demo_{episode_idx}"

            ep_data_grp = data_grp.create_group(ep)
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            for k in traj["obs"].keys():
                data = np.array(traj["obs"][k])
                assert data.dtype != np.dtype('O'), "Data type should not be object, but got {}".format(data.dtype)
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")

            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # 
            print("ep {}: wrote {} transitions to group {}".format(episode_idx, ep_data_grp.attrs["num_samples"], ep))

if __name__ == "__main__":
    # converter = RealToRobomimicConverter(real_dataset_path="/home/mingxi/data/realworld/test", output_robomimic_path="/home/mingxi/data/realworld/hdf5_hand_datasets/test_multiview_abs.hdf5")
    converter = RealToRobomimicConverter(real_dataset_path="/home/xhe71/Desktop/robotool_data/06232025/", output_robomimic_path="/home/xhe71/Desktop/robotool_data/06232025/episode_0/test_multiview_abs.hdf5")
    converter.convert()
    
