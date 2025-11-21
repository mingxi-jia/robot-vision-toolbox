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
from utils.pcd_utils import *
from configs.workspace import WORKSPACE, MAX_POINT_NUM
from concurrent.futures import ThreadPoolExecutor


frame_cache = {}
preload_cache = {}

from utils.pcd_utils import get_extrinsics_matrix

def simple_downsample_for_fixed_scene(pcd: o3d.geometry.PointCloud, 
                                    target_points: int = 4412,
                                    voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """
    Simple two-stage downsampling: voxel downsample first, then precise control
    
    Args:
        pcd: Input point cloud
        target_points: Target number of points (default 4412)
        voxel_size: Voxel size in meters (default 0.01m = 1cm)
    
    Returns:
        Downsampled point cloud
    """
    if len(pcd.points) <= target_points:
        return pcd
    
    # Stage 1: Voxel downsampling for fast point reduction
    voxel_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Stage 2: Precise control to target number of points
    if len(voxel_pcd.points) > target_points:
        # If still too many points, random downsample to target
        sampling_ratio = target_points / len(voxel_pcd.points)
        final_pcd = voxel_pcd.random_down_sample(sampling_ratio=sampling_ratio)
    else:
        # If voxel downsampling already reduced below target, return as is
        final_pcd = voxel_pcd
    
    return final_pcd


def find_optimal_voxel_size(test_pcd: o3d.geometry.PointCloud, 
                          target_points: int = 4412) -> float:
    """
    Find optimal voxel size for your fixed scene
    Run once to find suitable voxel_size, then use it directly afterwards
    """
    print(f"Finding optimal voxel size for {len(test_pcd.points)} -> {target_points} points")
    print("-" * 60)
    
    # Test different voxel sizes
    voxel_sizes = [0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03]
    
    best_voxel_size = 0.01
    best_error = float('inf')
    
    for voxel_size in voxel_sizes:
        start_time = time.time()
        result = simple_downsample_for_fixed_scene(test_pcd, target_points, voxel_size)
        end_time = time.time()
        
        result_points = len(result.points)
        error = abs(result_points - target_points) / target_points
        
        print(f"Voxel size: {voxel_size:.3f}m -> {result_points:4d} points, "
              f"error: {error:.3f}, time: {end_time-start_time:.4f}s")
        
        if error < best_error:
            best_error = error
            best_voxel_size = voxel_size
    
    print(f"\nBest voxel size: {best_voxel_size:.3f}m (error: {best_error:.3f})")
    return best_voxel_size


def replace_fps_with_simple_voxel(combined_pcd: o3d.geometry.PointCloud, 
                                max_point_num: int) -> o3d.geometry.PointCloud:
    """
    Direct replacement for farthest_point_down_sample in your original code
    
    Usage:
    # Original code:
    # combined_pcd = combined_pcd.farthest_point_down_sample(num_samples=min(max_point_num, len(combined_pcd.points)))
    
    # Replace with:
    # combined_pcd = replace_fps_with_simple_voxel(combined_pcd, max_point_num)
    """
    
    # Recommended voxel size based on your scene
    # You can determine this value using find_optimal_voxel_size function
    RECOMMENDED_VOXEL_SIZE = 0.012  # 1.2cm, you can adjust this value
    
    return simple_downsample_for_fixed_scene(combined_pcd, max_point_num, RECOMMENDED_VOXEL_SIZE)


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


def transform_hamer_cam_to_robot(hand_data, extrinsics):
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
    Also applies Z-flip transformation to match Z-down coordinate convention.
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

    # Mirror transformation to flip Y-axis (inverts Y rotation direction)
    R_mirror_y = np.array([
        [ 1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1]
    ])

    # Extract input pose rotation and translation
    t = np.array(pose[:3])
    R_pose = R.from_quat(pose[3:]).as_matrix()

    # Apply Y-mirror to flip Y-axis rotation, then apply alignment
    R_aligned = R_mirror_y @ R_pose @ R_mirror_y @ R_align

    # Reconstruct quaternion from aligned rotation
    quat_aligned = R.from_matrix(R_aligned).as_quat()
    return np.concatenate([t, quat_aligned])


def preload_all_frames(image_path, cam_views, frame_indices, rgb_dir, depth_dir, variant_key='default'):
    """
    Preload frames into cache with variant-specific keys to avoid overwrites.

    Args:
        variant_key: String identifier for this variant (e.g., 'pcd', 'pcd_no_hand')
    """
    global preload_cache
    start_time = time.time()
    # print(f"Preloading all frames for cameras {cam_views} and frames {frame_indices}...")
    for cam_name in [f'cam{v}' for v in cam_views]:
        # Use variant-specific cache key to avoid overwriting
        cache_key = f"{cam_name}_{variant_key}"
        preload_cache[cache_key] = {}
        for frame_idx in frame_indices:
            rgb_path = os.path.join(image_path, cam_name, rgb_dir, f'{frame_idx}.png')
            depth_path = os.path.join(image_path, cam_name, depth_dir, f'{frame_idx}.npy')
            if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                raise FileNotFoundError(f"Missing rgb or depth file for {cam_name} frame {frame_idx}")
            rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
            depth = np.load(depth_path) / 1000.0
            preload_cache[cache_key][frame_idx] = (rgb, depth)
    end_time = time.time()
    # print(f"Preloading completed in {end_time - start_time:.2f} seconds.")


def load_single_camera_view(image_path, cam_name, frame_idx, rgb_dir, depth_dir, intrinsics, extrinsics, variant_key='default'):
    """
    Load a single camera view with variant-specific caching.

    Args:
        variant_key: String identifier matching the preload variant key
    """
    global preload_cache
    decode_start = time.time()

    # Use variant-specific cache key
    cache_key = f"{cam_name}_{variant_key}"

    if cache_key in preload_cache and frame_idx in preload_cache[cache_key]:
        rgb, depth = preload_cache[cache_key][frame_idx]
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


def load_camera_data(image_path, cam_name_list, frame_idx, rgb_dir, depth_dir, cam_info_dict, variant_key='default'):
    """
    Load camera data for all cameras with variant-specific caching.

    Args:
        variant_key: String identifier for the variant (e.g., 'pcd', 'pcd_no_hand')
    """
    global frame_cache
    # Include variant_key in cache key to separate variants
    cache_key = (frame_idx, tuple(cam_name_list), variant_key)
    if cache_key in frame_cache:
        return frame_cache[cache_key]

    points_list = []
    colors_list = []
    with ThreadPoolExecutor(max_workers=len(cam_name_list)) as executor:
        futures = []
        for cam_name in cam_name_list:
            intrinsics = cam_info_dict[cam_name]['intrinsics']
            extrinsics = cam_info_dict[cam_name]['extrinsics']
            futures.append(executor.submit(load_single_camera_view, image_path, cam_name, frame_idx, rgb_dir, depth_dir, intrinsics, extrinsics, variant_key))
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


def add_frame_to_pcd(combined_pcd, pose):
    # pose_aligned = apply_alignment_rotation(pose)
    pose_aligned = pose
    # sphere = render_pcd_from_pose(pose_aligned[None, ...], fix_point_num=1024, model_type='sphere')
    # sphere_pcd = np2o3d(sphere[:, :3], sphere[:, 3:6])
    frame_pcd = add_coordinate_axes_from_pose(pose_aligned[:3], pose_aligned[3:])
    combined_pcd += frame_pcd
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


def generate_pcd_sequence(episode_path, output_path, cam_info_dict, sphere_cam=3, segment=True, visualize_coordinate_axis = False, downsample_method = "simple_voxel"):
    """
    Generate a sequence of point clouds from RGB-D images and sphere poses.

    Parameters:
        episode_path (str): Path to the episode directory.
        output_path (str): Path where output point clouds and intermediate files are saved.
        cam_info_dict (dict): Dictionary containing camera intrinsics and extrinsics.
        sphere_cam (int): Camera ID used for sphere pose.
        segment (bool): Whether to use segmented images (only used if generate_both_variants=False).
        visualize_coordinate_axis (bool): Whether to add coordinate axis visualization.
        downsample_method (str): Method for downsampling ('simple_voxel', 'fps', 'random').
        generate_both_variants (bool): If True, generates both segmented and non-segmented PCDs in one pass.

    Returns:
        dict: Hand pose dictionary (pose_wrt_world).
    """
    # CRITICAL: Clear global caches at start of each episode to prevent cross-episode contamination
    global preload_cache, frame_cache
    preload_cache.clear()
    frame_cache.clear()

    _, episode_name = os.path.split(episode_path)

    
    # Single variant generation (backward compatibility)
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

    pose_path = os.path.join(output_path, episode_name, f'hand_poses_wrt_world.npy')
    pose_wrt_world = np.load(pose_path, allow_pickle=True)[()]
    assert len(pose_wrt_world) > 0, "No hand poses found!"

    max_point_num = MAX_POINT_NUM

    # Create save directories
    save_dir = os.path.join(output_path, episode_name, pcd_dir)
    os.makedirs(save_dir, exist_ok=True)

    frame_indices = sorted(pose_wrt_world.keys())

    # Preload frames for both variants if needed
    preload_all_frames(image_path, cam_views, frame_indices, rgb_dir, depth_dir, variant_key='default')

    def process_frame(frame_idx):
        # OPTIMIZATION: Process both variants in one pass
    
        # Original single-variant processing (backward compatibility)
        all_points, all_colors = load_camera_data(image_path, [f'cam{v}' for v in cam_views], frame_idx, rgb_dir, depth_dir, cam_info_dict, variant_key='default')

        combined_pcd = assemble_combined_pcd(all_points, all_colors)

        combined_pcd = filter_point_cloud_by_workspace(combined_pcd, WORKSPACE)

        if downsample_method == "simple_voxel":
            combined_pcd = replace_fps_with_simple_voxel(combined_pcd, max_point_num)
        elif downsample_method == "fps":
            combined_pcd = combined_pcd.farthest_point_down_sample(num_samples=min(max_point_num, len(combined_pcd.points)))
        elif downsample_method == "random":
            if len(combined_pcd.points) > max_point_num:
                combined_pcd = random_downsample(combined_pcd, max_point_num)

        if visualize_coordinate_axis:
            pose = pose_wrt_world[frame_idx]
            combined_pcd = add_frame_to_pcd(combined_pcd, pose)
        

        npy_file = os.path.join(save_dir, f"{frame_idx}.npy")
        points_np = np.asarray(combined_pcd.points)
        colors_np = np.asarray(combined_pcd.colors)
        np.save(npy_file, np.hstack([points_np, colors_np]))

        return combined_pcd

    print(f"Generating point cloud sequence from episode: {episode_path} with {len(pose_wrt_world.keys())} frames")
    frame_sequence = []

    executor = ThreadPoolExecutor(max_workers=4) # Adjust as needed: multithreading
    frame_sequence = list(executor.map(process_frame, frame_indices))
    executor.shutdown(wait=True)
    print(f"✅ Saved {len(frame_indices)} point clouds as .npy files (skipped .ply conversion for speed)")

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
            pose_wrt_world = transform_hamer_cam_to_robot(hand_pose, info_dict[main_cam]['extrinsics'])
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
