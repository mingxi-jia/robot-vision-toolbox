import os
import sys
import open3d as o3d
import numpy as np
import time
import json
from scipy.spatial.transform import Rotation as R
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

sys.path.append('./')
from example_data.pcd_utils import *
    


def visualize_pcd_loop(pcd_sequence):
    assert len(pcd_sequence) > 0, "No point clouds provided"
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D Loop Viewer", width=640, height=380)
    vis.get_render_option().point_size = 2.0

    current_idx = 0

    def update_view(vis):
        nonlocal current_idx
        vis.clear_geometries()
        vis.add_geometry(pcd_sequence[current_idx])
        current_idx = (current_idx + 1) % len(pcd_sequence)
        return False  # Continue animation

    vis.register_animation_callback(update_view)
    vis.add_geometry(pcd_sequence[0])
    vis.run()
    vis.destroy_window()

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


def convert_hamer_pose_to_extrinsic(hand_data, intrinsics, extrinsics):
    pose_dict = {}

    for frame_idx, key in enumerate(sorted(hand_data.keys())):
        sphere_center = np.array(hand_data[key]["uvz_handmask"])
        global_orient = np.array(hand_data[key]["global_orient_quat"])
        u,v,z = sphere_center
        frame_name = sorted(hand_data.keys())[frame_idx]
        trans_pose = uvz_to_world(u,v, z, intrinsics, extrinsics)
        quat = global_orient
        example_pose = np.hstack([trans_pose, quat])
        pose_dict[frame_name] = example_pose.tolist()
        # Removed cam_frame.transform(extrinsics)
    # Save to json
    # output_path = os.path.join(data_path, f'pose_dict.json')
    # with open(output_path, 'w') as f:
    #     json.dump(pose_dict, f, indent=4)
    # print(f"Saved pose dict to {output_path}")
    return pose_dict
        
        
def generate_pcd_sequence(episode_path, output_path, cam_info_dict, start_frame=0, sphere_cam = 3):
    main_cam_name = f'cam{sphere_cam}'
    data_path, episode_name = os.path.split(episode_path)

    
    cam_info = cam_info_dict

    cam_views = [1,2, 3]
    
            
    cam_intrinsics = cam_info[main_cam_name]['intrinsics']
    cam_extrinsics = cam_info[main_cam_name]['extrinsics']

    
    sphere_cam_name = f'cam{sphere_cam}'
    with open(os.path.join(output_path, episode_name, sphere_cam_name, 'sphere_pose.json'), 'r') as f:
        hand_pose = json.load(f)
    pose_dict = convert_hamer_pose_to_extrinsic(hand_pose, cam_intrinsics, cam_extrinsics)

    frame_sequence = []
    
    for i, frame_idx in enumerate(sorted(pose_dict.keys())):
        all_pcds = o3d.geometry.PointCloud()
        for i in cam_views:
            cam_name = f'cam{i}'

            # simulate different RGB and depth images for each view
            rgb_path = os.path.join(output_path, episode_name, cam_name, 'segment_out', 'segmented_rgb', f'{frame_idx}_segmented.png')
            depth_path = os.path.join(output_path, episode_name, cam_name, 'segment_out','segmented_depth', f'{frame_idx}_segmented.npy')
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                continue
            # Load RGB and depth images
            rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
            depth = np.load(depth_path) / 1000.
            # Convert RGB and depth to Open3D point cloud
            pcd_np, pcd_o3d = convert_RGBD_to_open3d(rgb, depth, cam_info[cam_name]['intrinsics'], cam_info[cam_name]['extrinsics'])
            all_pcds += pcd_o3d
        
        # Downsample and filter point cloud
        max_point_num = 4812
        x_min, y_min, z_min, ws_size = 0.2, -0.2, -0.05, 0.4
        all_pcds = filter_point_cloud_by_workspace(all_pcds, x_min, x_min+ws_size, y_min, y_min+ws_size, z_min, z_min+ws_size)
        all_pcds = all_pcds.farthest_point_down_sample(num_samples=max_point_num)

        # use the pose from the dictionary to render the sphere
        example_pose = np.array(pose_dict[frame_idx])
        sphere = render_pcd_from_pose(example_pose[None,...], fix_point_num=1024, model_type='sphere')[0]
        all_pcds += np2o3d(sphere[:, :3], sphere[:, 3:6])

       
        # Save the point cloud for this frame
        pcd_path = os.path.join(output_path, episode_name, 'pcd')
        if not os.path.exists(pcd_path):
            os.makedirs(pcd_path)
        o3d.io.write_point_cloud(os.path.join(pcd_path, f"{frame_idx}.ply"), all_pcds)
        print(f"Saved PCD to {pcd_path}")
        
        frame_sequence.append(all_pcds)

    return frame_sequence

if __name__ == "__main__":
    import yaml
    from PIL import Image
    import matplotlib.pyplot as plt
    import subprocess

    # Example usage
    data_path = "/home/xhe71/Desktop/robotool_data/06232025/episode_20250623_173802_495"
    frame_sequence = generate_pcd_sequence(data_path, start_frame=0, sphere_cam=3)


