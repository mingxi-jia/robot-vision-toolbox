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
import open3d as o3d

import numpy as np
from PIL import Image
from tqdm import tqdm

from vision_utils.ply_utils import convert_RGBD_to_open3d, o3d2np

def load_info_dict(info_path: str):
    with open(info_path, "r") as f:
        info_dict = json.load(f)
    return info_dict

class RealToRobomimicConverter:
    def __init__(self, real_dataset_path: str, output_robomimic_path: str):
        cam_list = [f for f in os.listdir(os.path.join(real_dataset_path, "episode_0")) if f.startswith("cam")]
        num_cams = len(cam_list)
        num_episodes = len([f for f in os.listdir(real_dataset_path) if f.startswith("episode_")])

        self.info_dict = load_info_dict(os.path.join(real_dataset_path, "info.json"))

        self.hamer = HAMER() #TODO(ivy)
        
        print(f"Extracting actions from real dataset using HAMER...")
        self.extract_actions()

        self.real_dataset_path = real_dataset_path
        self.robomimic_dataset_path = output_robomimic_path

        self.main_cam = 'cam3'
        self.num_cams = num_cams
        self.cam_list = cam_list
        self.num_episodes = num_episodes
        self.fix_point_num = 1024
        self.workspace = np.array([[-0.2, 0.2], 
                                   [-0.2, 0.2], 
                                   [0.0, 0.2]])

    def preprocess(self):
        for episode_idx in tqdm(range(self.num_episodes), desc="Preprocessing episodes"):
            self.extract_actions(episode_idx)
            self.extract_pcds(episode_idx)

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
            assert hand_pose.shape == (1, 10), f"Hand pose shape is {hand_pose.shape} but should be (1, 10), aka [x, y, z, qx, qy, qz, qw]"

            states.append(hand_pose)

        # shift states by 1 time step
        actions = copy.deepcopy(states[1:])
        actions.append(states[-1])
        actions = np.stack(actions)
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
        pcd_dict = {'pcd':[]}
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

            pcd_dict['pcd'].append(np.concatenate(pcds, axis=0))


        actions = np.load(os.path.join(episode_path, "actions.npy"))
        states = np.load(os.path.join(episode_path, "states.npy"))
        rewards = np.load(os.path.join(episode_path, "rewards.npy"))
        dones = np.load(os.path.join(episode_path, "dones.npy"))

        obs = {
            'rgb': rgb_dict,
            'depth': depth_dict,
            'pcd': pcd_dict,
            'states': states
        }

        trajectory = {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'dones': dones
        }

        return trajectory

    def convert(self):
        print(f"Converting data to robomimic format...")
        f_out = h5py.File(self.robomimic_dataset_path, "w")
        data_grp = f_out.create_group("data")

        for episode_idx in tqdm(range(self.num_episodes), desc="Converting episodes"):
            traj = self.load_trajectory(episode_idx)

            ep = f"demo_{episode_idx}"

            ep_data_grp = data_grp.create_group(ep)
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            for k in traj["obs"]:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")

            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # 
            print("ep {}: wrote {} transitions to group {}".format(episode_idx, ep_data_grp.attrs["num_samples"], ep))

if __name__ == "__main__":
    converter = RealToRobomimicConverter(real_dataset_path="dataset", output_robomimic_path="hand_dataset.hdf5")
    converter.convert()