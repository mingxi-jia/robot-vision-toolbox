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
    

# data_path = "/home/xhe71/Desktop/robotool_data/06232025/slow"
# output_folder = os.path.join(data_path, "pcd")
# os.makedirs(output_folder,  exist_ok=True)


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


def convert_hamer_pose_to_extrinsic(data_path, cam_name = 'cam3', intrinsics = None, extrinsics = None):
    pose_dict = {}
    with open(os.path.join('setup/camera_info.yaml'), 'r') as f:
        cam_info = yaml.safe_load(f)
        
    hand_path = os.path.join(data_path, cam_name, 'output', 'sphere_pose.json')
    if not os.path.exists(hand_path):
        return None
    with open(hand_path, 'r') as f:
        hand_data = json.load(f)

    if extrinsics is None:
        extrinsics = np.eye(4)
    for frame_idx, key in enumerate(sorted(hand_data.keys())):
        sphere_center = np.array(hand_data[key]["uvz_handmask"])
        global_orient = np.array(hand_data[key]["global_orient_quat"])
        u,v,z = sphere_center
        trans_pose = uvz_to_world(u,v, z, intrinsics, extrinsics)
        quat = global_orient
        example_pose = np.hstack([trans_pose, quat])
        pose_dict[frame_idx] = example_pose.tolist()
        # Removed cam_frame.transform(extrinsics)
    # Save to json
    output_path = os.path.join(data_path, f'pose_dict.json')
    with open(output_path, 'w') as f:
        json.dump(pose_dict, f, indent=4)
    print(f"Saved pose dict to {output_path}")
    return pose_dict
        
        
def generate_pcd_sequence(data_path, start_frame=0, sphere_cam = 3):
    # load camera_info.yaml
    with open(os.path.join('setup/camera_info.yaml'), 'r') as f:
        cam_info = yaml.safe_load(f)

    cam_views = [1,2, 3]
    
            
    frame_dict = dict()
    info_dict = dict()
    camera_frames = []
    num_frames = 0
    sphere_extrinsics = None
    sphere_intrinsics = None
    for i in cam_views:
        cam_name = f'cam{i}'
        intrinsics = np.array(cam_info[cam_name]['k']).reshape(3, 3)
        extrinsics = get_extrinsics_matrix(cam_info[cam_name]['t'], cam_info[cam_name]['q'])
        if i == sphere_cam:
            sphere_extrinsics = extrinsics
            sphere_intrinsics = intrinsics
        info_dict[cam_name] = {'intrinsics': intrinsics, 'extrinsics': extrinsics}

        frame_list = [f.split('.png')[0] for f in os.listdir(os.path.join(data_path, cam_name, 'rgb')) if f.endswith('.png')]
        frame_list.sort(key=lambda x: (int(x.split('_')[0])))
        if num_frames == 0:
            num_frames = len(frame_list)
        frame_dict[cam_name] = frame_list

        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        cam_frame.transform(extrinsics)
        camera_frames.append(cam_frame)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.3, origin=[0, 0, 0])
    camera_frames.append(coordinate_frame)

    # o3d.visualization.draw_geometries(camera_frames)

    
    sphere_cam_name = f'cam{sphere_cam}'
    pose_path = os.path.join(data_path, 'pose_dict.json')
    if not os.path.exists(pose_path):
        pose_dict = convert_hamer_pose_to_extrinsic(data_path, cam_name = sphere_cam_name, extrinsics = sphere_extrinsics)
    else:
        with open(pose_path, 'r') as f:
            pose_dict = json.load(f)
    # pose_dict = convert_hamer_pose_to_extrinsic(data_path, cam_name = sphere_cam_name, intrinsics = sphere_intrinsics, extrinsics = sphere_extrinsics)

    # multi-view reconstruction example

    all_pcds = o3d.geometry.PointCloud()
    frame_sequence = []
        
    for frame_i in range(start_frame, num_frames):
        for i in cam_views:
            cam_name = f'cam{i}'

            # simulate different RGB and depth images for each view
            frame_name = frame_dict[cam_name][frame_i]
            rgb_path = os.path.join(data_path, cam_name, 'rgb_final', 'sphere_rgb', f'{frame_name}_final.png')
            # if i == 3:
            #     depth_path = os.path.join(data_path, cam_name, 'rgb_final','sphere_depth', f'{frame_name}_sphere.npy')
            # else:
            #     depth_path = os.path.join(data_path, cam_name, 'rgb_segmented','segmented_depth', f'{frame_name}_segmented.npy')
            depth_path = os.path.join(data_path, cam_name, 'rgb_segmented','segmented_depth', f'{frame_name}_segmented.npy')
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                continue
            rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
            depth = np.load(depth_path) / 1000.

            vis = False
            if vis:
                # Plot side-by-side
                plt.figure(figsize=(12, 5))

                # Show color image
                plt.subplot(1, 2, 1)
                plt.title(f"{cam_name} Color Image")
                plt.imshow(rgb)
                plt.axis('off')

                # Show depth map
                plt.subplot(1, 2, 2)
                plt.title(f"{cam_name} Depth Map")
                plt.imshow(depth, cmap='plasma')  # or 'gray' or 'viridis'
                plt.colorbar(label='Depth')
                plt.axis('off')

                plt.tight_layout()
                # plt.show()

            pcd_np, pcd_o3d = convert_RGBD_to_open3d(rgb, depth, info_dict[cam_name]['intrinsics'], info_dict[cam_name]['extrinsics'])
            all_pcds += pcd_o3d
            

            # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([pcd_o3d, coordinate_frame])

        max_point_num = 4812
        x_min, y_min, z_min, ws_size = 0, -1, -0.05, 2
        all_pcds = filter_point_cloud_by_workspace(all_pcds, x_min, x_min+ws_size, y_min, y_min+ws_size, z_min, z_min+ws_size)
        all_pcds = all_pcds.farthest_point_down_sample(num_samples=max_point_num)


        # use the pose from the dictionary
        if pose_dict and frame_i in pose_dict:
            example_pose = np.array(pose_dict[frame_i])
            sphere = render_pcd_from_pose(example_pose[None,...], fix_point_num=1024, model_type='sphere')[0]
            all_pcds += np2o3d(sphere[:, :3], sphere[:, 3:6])

        # # Display the frame
        # o3d.visualization.draw_geometries([all_pcds, coordinate_frame],
        #                                   front=[1, 0.0, 0.8],
        #                                   lookat=[0.0, 0.0, 0.0],
        #                                   up=[-3.14, 0.98, 0.1],
        #                                   zoom=0.7)

        # # Wait briefly to hold the window before screenshot
        # time.sleep(0.2)

        # Save the point cloud for this frame
        pcd_save_path = os.path.join(output_folder, f"{frame_i:06d}.ply")
        o3d.io.write_point_cloud(pcd_save_path, all_pcds)
        print(f"Saved PCD to {pcd_save_path}")
        all_pcds = o3d.geometry.PointCloud()
        
        frame_sequence.append(all_pcds)
        
    return frame_sequence

# frame_sequence = generate_pcd_sequence(data_path, start_frame=0, sphere_cam=3)

