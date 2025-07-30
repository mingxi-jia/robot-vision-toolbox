import os
import sys
import open3d as o3d
import numpy as np
import time
import json
import yaml
from scipy.spatial.transform import Rotation as R
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

sys.path.append('./')
from vision_utils.pcd_utils import *
from configs.workspace import WORKSPACE, MAX_POINT_NUM
from concurrent.futures import ThreadPoolExecutor


frame_cache = {}
preload_cache = {}

def uvz_to_world(u, v, z, intrinsics, extrinsics):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Step 1: UVZ to Camera Frame
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    P_cam = np.array([x, y, z, 1])  # homogeneous

    # Step 2: Camera Frame to World Frame
    P_world = extrinsics @ P_cam
    return P_world[:3]


def convert_hamer_pose_to_extrinsic(hand_data, extrinsics):
    global_pose = {}
    for frame_name, pose in hand_data.items():
        xyz_wrt_cam, quat_wrt_cam = pose[:3], pose[3:7]
        hand_pos_wrt_cam = np.eye(4)
        hand_pos_wrt_cam[:3, :3] = R.from_quat(quat_wrt_cam).as_matrix()
        hand_pos_wrt_cam[:3, 3] = xyz_wrt_cam
        hand_pos_wrt_world = extrinsics @ hand_pos_wrt_cam

        xyz_wrt_world, quat_wrt_world = hand_pos_wrt_world[:3, 3], R.from_matrix(hand_pos_wrt_world[:3, :3]).as_quat()
        global_pose[frame_name] = np.concatenate([xyz_wrt_world, quat_wrt_world])
    return global_pose
        


def apply_alignment_rotation(pose, yaml_path='./configs/hamer_alignment_matrix.yaml'):
    """
    Apply saved alignment rotation (as quaternion) from YAML to a given pose.
    Assumes pose is in [x, y, z, qx, qy, qz, qw] format.

    Returns:
        aligned_pose: Pose with rotated orientation, original translation.
    """
    # Load alignment config
    with open(yaml_path, 'r') as f:
        alignment_config = yaml.safe_load(f)
    
    if 'quat_transform' not in alignment_config:
        raise KeyError("quat_transform not found in the YAML config.")
    
    # Extract rotation matrix from quaternion
    align_quat = alignment_config['quat_transform']
    R_align = R.from_quat(align_quat).as_matrix()
    
    # Extract input pose rotation and translation
    t = np.array(pose[:3])
    R_pose = R.from_quat(pose[3:]).as_matrix()

    # Apply rotation to rotation
    R_aligned =  R_pose @ R_align

    # Reconstruct quaternion from aligned rotation
    quat_aligned = R.from_matrix(R_aligned).as_quat()
    return np.concatenate([t, quat_aligned])


def preload_all_frames(image_path, cam_views, frame_indices, rgb_dir, depth_dir):
    global preload_cache
    start_time = time.time()
    # print(f"Preloading all frames for cameras {cam_views} and frames {frame_indices}...")
    for cam_name in [f'cam{v}' for v in cam_views]:
        preload_cache[cam_name] = {}
        for frame_idx in frame_indices:
            rgb_path = os.path.join(image_path, cam_name, rgb_dir, f'{frame_idx}.png')
            depth_path = os.path.join(image_path, cam_name, depth_dir, f'{frame_idx}.npy')
            if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                raise FileNotFoundError(f"Missing rgb or depth file for {cam_name} frame {frame_idx}")
            rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
            depth = np.load(depth_path) / 1000.0
            preload_cache[cam_name][frame_idx] = (rgb, depth)
    end_time = time.time()
    # print(f"Preloading completed in {end_time - start_time:.2f} seconds.")


def load_single_camera_view(image_path, cam_name, frame_idx, rgb_dir, depth_dir, intrinsics, extrinsics):
    global preload_cache
    decode_start = time.time()
    if cam_name in preload_cache and frame_idx in preload_cache[cam_name]:
        rgb, depth = preload_cache[cam_name][frame_idx]
    else:
        rgb_path = os.path.join(image_path, cam_name, rgb_dir, f'{frame_idx}.png')
        depth_path = os.path.join(image_path, cam_name, depth_dir, f'{frame_idx}.npy')
        assert os.path.exists(rgb_path) and os.path.exists(depth_path)
        rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
        depth = np.load(depth_path) / 1000.0
    decode_end = time.time()

    convert_start = time.time()
    pcd_np = convert_RGBD_fast(rgb, depth, intrinsics, extrinsics)
    
    # _, pcd_o3d = convert_RGBD_to_open3d(rgb, depth, intrinsics, extrinsics)
    convert_end = time.time()
    points = pcd_np[:, :3]
    colors = pcd_np[:, 3:6]
    # points = np.asarray(pcd_o3d.points)
    # colors = np.asarray(pcd_o3d.colors)

    # print(f"Load single camera view {cam_name} frame {frame_idx} timing:")
    # print(f"  Image decode and depth load: {decode_end - decode_start:.3f}s")
    # print(f"  Point cloud conversion: {convert_end - convert_start:.3f}s")

    return points, colors


def load_camera_data(image_path, cam_name_list, frame_idx, rgb_dir, depth_dir, cam_info_dict):
    global frame_cache
    cache_key = (frame_idx, tuple(cam_name_list))
    if cache_key in frame_cache:
        return frame_cache[cache_key]

    points_list = []
    colors_list = []
    with ThreadPoolExecutor(max_workers=len(cam_name_list)) as executor:
        futures = []
        for cam_name in cam_name_list:
            intrinsics = cam_info_dict[cam_name]['intrinsics']
            extrinsics = cam_info_dict[cam_name]['extrinsics']
            futures.append(executor.submit(load_single_camera_view, image_path, cam_name, frame_idx, rgb_dir, depth_dir, intrinsics, extrinsics))
        for future in futures:
            points, colors = future.result()
            points_list.append(points)
            colors_list.append(colors)

    frame_cache[cache_key] = (points_list, colors_list)
    return points_list, colors_list


def assemble_combined_pcd(points_list, colors_list):
    combined_points = np.concatenate(points_list, axis=0)
    combined_colors = np.concatenate(colors_list, axis=0)
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    return combined_pcd


def add_sphere_to_pcd(combined_pcd, pose, visualize_axis=False):
    pose_aligned = apply_alignment_rotation(pose)
    sphere = render_pcd_from_pose(pose_aligned[None, ...], fix_point_num=1024, model_type='sphere')
    sphere_pcd = np2o3d(sphere[:, :3], sphere[:, 3:6])
    if visualize_axis:
        frame_pcd = add_coordinate_axes_from_pose(pose_aligned[:3], pose_aligned[3:])
        sphere_pcd += frame_pcd
    combined_pcd += sphere_pcd
    return combined_pcd


def save_pcd_async(path, pcd):
    o3d.io.write_point_cloud(path, pcd, write_ascii=False, compressed = False)


def save_ply_from_npy(npy_file, ply_file):
    """
    Load a saved .npy point cloud, convert to Open3D PointCloud, save as .ply, and delete the .npy.
    """
    arr = np.load(npy_file)
    points, colors = arr[:, :3], arr[:, 3:6]
    pcd_o3d = np2o3d(points, colors)
    o3d.io.write_point_cloud(ply_file, pcd_o3d, write_ascii=False, compressed=False)
    os.remove(npy_file) # Remove the .npy file after conversion


def generate_pcd_sequence(episode_path, output_path, cam_info_dict, sphere_cam=3, segment=True, visualize_coordinate_axis = False):
    """
    Generate a sequence of point clouds from RGB-D images and sphere poses.
    
    Parameters:
        episode_path (str): Path to the episode directory.
        output_path (str): Path where output point clouds and intermediate files are saved.
        cam_info_dict (dict): Dictionary containing camera intrinsics and extrinsics.
        start_frame (int, optional): Frame index to start from.
        sphere_cam (int, optional): Camera ID used for sphere pose.
    
    Returns:
        List[o3d.geometry.PointCloud]: List of point cloud objects for each frame.
    """
    _, episode_name = os.path.split(episode_path)
    if segment:
        image_path = os.path.join(output_path, episode_name)
        rgb_dir, depth_dir, pcd_dir = 'segmented_rgb', 'segmented_depth', 'pcd_no_hand'
    else:
        image_path = episode_path
        rgb_dir, depth_dir, pcd_dir = 'rgb', 'depth', 'pcd'

    main_cam_name = f'cam{sphere_cam}'
    cam_views = [1, 2, 3]

    cam_intrinsics = cam_info_dict[main_cam_name]['intrinsics']
    cam_extrinsics = cam_info_dict[main_cam_name]['extrinsics']

    pose_path = os.path.join(output_path, episode_name, f'hand_poses_wrt_cam{sphere_cam}.npy')
    hand_pose = np.load(pose_path, allow_pickle=True)[()]
    pose_wrt_world = convert_hamer_pose_to_extrinsic(hand_pose, cam_extrinsics)
    
    max_point_num = MAX_POINT_NUM

    save_dir = os.path.join(output_path, episode_name, pcd_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    frame_indices = sorted(pose_wrt_world.keys())
    preload_all_frames(image_path, cam_views, frame_indices, rgb_dir, depth_dir)

    def process_frame(frame_idx):
        start_total = time.time()
        # Load camera data
        start_load = time.time()
        all_points, all_colors = load_camera_data(image_path, [f'cam{v}' for v in cam_views], frame_idx, rgb_dir, depth_dir, cam_info_dict)
        end_load = time.time()

        # Assemble combined pcd
        start_assemble = time.time()
        combined_pcd = assemble_combined_pcd(all_points, all_colors)
        end_assemble = time.time()

        # Filter and downsample
        start_filter = time.time()
        combined_pcd = filter_point_cloud_by_workspace(
            combined_pcd,
            WORKSPACE
        )
        combined_pcd = combined_pcd.farthest_point_down_sample(num_samples=min(max_point_num, len(combined_pcd.points)))
        # Use random_downsample if too many points, else keep as is
        # if len(combined_pcd.points) > max_point_num:
        #     combined_pcd = random_downsample(combined_pcd, max_point_num)
        # end_filter = time.time()

        # Add sphere to pcd
        start_sphere = time.time()
        pose = pose_wrt_world[frame_idx]
        combined_pcd = add_sphere_to_pcd(combined_pcd, pose, visualize_axis=visualize_coordinate_axis)
        end_sphere = time.time()

        # Save npy file
        start_save = time.time()
        npy_file = os.path.join(save_dir, f"{frame_idx}.npy")
        points_np = np.asarray(combined_pcd.points)
        colors_np = np.asarray(combined_pcd.colors)
        np.save(npy_file, np.hstack([points_np, colors_np]))
        end_save = time.time()

        end_total = time.time()

        print(f"Frame {frame_idx} processing times:")
        print(f"  Loading data: {end_load - start_load:.2f}s")
        print(f"  Assembling PCD: {end_assemble - start_assemble:.2f}s")
        print(f"  Filtering/Downsampling: {end_filter - start_filter:.2f}s")
        print(f"  Adding sphere: {end_sphere - start_sphere:.2f}s")
        print(f"  Saving npy: {end_save - start_save:.2f}s")
        print(f"  Total: {end_total - start_total:.2f}s")

        return combined_pcd

    print(f"Generating point cloud sequence from episode: {episode_path} with {len(pose_wrt_world.keys())} frames")
    frame_sequence = []

    executor = ThreadPoolExecutor(max_workers=4) # Adjust as needed: multithreading
    frame_sequence = list(executor.map(process_frame, frame_indices))
    executor.shutdown(wait=True)

    # After all frames processed, batch convert npy to ply in parallel
    def convert_and_save(frame_idx):
        npy_file = os.path.join(save_dir, f"{frame_idx}.npy")
        ply_file = os.path.join(save_dir, f"{frame_idx}.ply")
        save_ply_from_npy(npy_file, ply_file)

    with ThreadPoolExecutor(max_workers=4) as executor2:
        executor2.map(convert_and_save, frame_indices)

    return pose_wrt_world


def convert_pose_to_world(episode_list, process_path, info_dict, main_cam=3):
    from tqdm import tqdm
    for episode_name in tqdm(episode_list, desc="Preprocessing episodes"):
            # done_indicator = os.path.join(self.process_path, episode_name, "DONE")
        done_indicator = os.path.join(process_path, episode_name, "hand_poses.npy")
        if os.path.exists(done_indicator):
            print(f"Episode {episode_name} already processed. Skipping...")
            hand_poses_path = os.path.join(process_path, episode_name, "hand_poses.npy")
            hand_pose = np.load(hand_poses_path, allow_pickle=True)[()]
            pose_wrt_world = convert_hamer_pose_to_extrinsic(hand_pose, info_dict[main_cam]['extrinsics'])
            np.save(os.path.join(process_path, episode_name, "hand_poses_wrt_world.npy"), pose_wrt_world, allow_pickle=True)
            print(f"saved to hand_poses_wrt_world")

if __name__ == "__main__":
    import yaml
    from PIL import Image
    import matplotlib.pyplot as plt
    import subprocess

    def load_camera_info_dict(info_path: str):
        assert info_path.endswith(".yaml"), "Info file should be a yaml file"
        with open(info_path, 'r') as f:
            info_dict = yaml.safe_load(f)
        for cam_name, cam_info in info_dict.items():
            info_dict[cam_name]['intrinsics'] = np.array(cam_info['k']).reshape(3, 3)
            info_dict[cam_name]['extrinsics'] = get_extrinsics_matrix(cam_info['t'], cam_info['q'])
        return info_dict
    # Example usage
    data_path = "/home/xhe71/Desktop/robotool_data//home/xhe71/Desktop/robotool_data/hand_rotation_test/output/episode_0/"
    out_path = "/home/xhe71/Desktop/robotool_data/hand_rotation_test/output/episode_0/"
    cam_info = load_camera_info_dict("configs/camera_info.yaml")
    frame_sequence = generate_pcd_sequence(data_path, out_path, cam_info,
                                           segment=True, visualize_coordinate_axis = True)
