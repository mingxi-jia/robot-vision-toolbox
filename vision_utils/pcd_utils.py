import os
from pathlib import Path

import open3d as o3d
import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def np2o3d(pcd, color=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None and color.shape[0] > 0:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d

def populate_point_num(pcd, point_num):
    if pcd.shape[0] < point_num:
        extra_choice = np.random.choice(pcd.shape[0], point_num-pcd.shape[0], replace=True)
        pcd = np.concatenate([pcd, pcd[extra_choice]], axis=0)
    else:
        shuffle_idx = np.random.permutation(pcd.shape[0])[:point_num]
        pcd = pcd[shuffle_idx]
    return pcd

def o3d2np(pcd_o3d, num_samples=4412):
    # pcd: (n, 3)
    # color: (n, 3)
    xyz = np.asarray(pcd_o3d.points)
    rgb = np.asarray(pcd_o3d.colors)
    num_points = xyz.shape[0]

    if num_points > num_samples:
        pcd_o3d = pcd_o3d.farthest_point_down_sample(num_samples=num_samples)
        xyz = np.asarray(pcd_o3d.points)
        rgb = np.asarray(pcd_o3d.colors)
        pcd_np = np.concatenate([xyz, rgb], axis=1)
    else:
        pcd_np = np.concatenate([xyz, rgb], axis=1)
        pcd_np = populate_point_num(pcd_np, num_samples)

    return pcd_np


def depth2fgpcd(depth, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    mask = (depth > 0.0) & (depth < 2.0)  # filter out invalid depth values
    # mask = (depth <= 0.599/0.8)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd, mask

def convert_RGBD_to_open3d(rgb, depth, intrinsics, extrinsics):
    assert rgb.shape[0] == depth.shape[0] and rgb.shape[1] == depth.shape[1]
    assert intrinsics.shape == (3, 3)
    assert extrinsics.shape == (4, 4)
    assert rgb.dtype == np.uint8
    assert depth.dtype == np.float32 or depth.dtype == np.float64

    cam_param = [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]
    pcd, mask = depth2fgpcd(depth, cam_param)
    # pose = np.linalg.inv(ext_mat)
    pose = extrinsics
    # trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
    pcd_np = np.einsum('ij,jk->ik', pose, np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0))
    pcd_np = pcd_np[:3, :].T
    pcd_o3d = np2o3d(pcd_np, rgb[mask].reshape(-1, 3).astype(np.float64) / 255)
    return pcd_np, pcd_o3d

def get_extrinsics_matrix(t, q):
    """
    Convert translation and quaternion to extrinsics matrix.
    :param t: Translation vector (list or np.array of shape (3,))
    :param q: Quaternion (list or np.array of shape (4,))
    :return: Extrinsics matrix (np.array of shape (4, 4))
    """
    assert len(t) == 3 and len(q) == 4, "Translation must be of length 3 and quaternion of length 4."
    
    # Convert quaternion to rotation matrix
    x, y, z, w = q
    rot = R.from_quat([x, y, z, w]).as_matrix()
    
    # Create extrinsics matrix
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rot
    extrinsics[:3, 3] = t
    
    return extrinsics

if __name__ == "__main__":

    data_path = "example_data/multiview_rgbd"
    cam_views = [1, 2, 3]

    # load camera_info.yaml
    with open(os.path.join(data_path, 'camera_info.yaml'), 'r') as f:
        cam_info = yaml.safe_load(f)

    frame_dict = dict()
    info_dict = dict()
    camera_frames = []
    num_frames = 0
    for i in cam_views:
        cam_name = f'cam{i}'
        intrinsics = np.array(cam_info[cam_name]['k']).reshape(3, 3)
        extrinsics = get_extrinsics_matrix(cam_info[cam_name]['t'], cam_info[cam_name]['q'])
        info_dict[cam_name] = {'intrinsics': intrinsics, 'extrinsics': extrinsics}

        frame_list = [f.split('.png')[0] for f in os.listdir(os.path.join(data_path, cam_name, 'rgb')) if f.endswith('.png')]
        frame_list.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
        if num_frames == 0:
            num_frames = len(frame_list)
        else:
            assert num_frames == len(frame_list), f"Number of frames mismatch in {cam_name}: {num_frames} vs {len(frame_list)}"
        frame_dict[cam_name] = frame_list

        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        cam_frame.transform(extrinsics)
        camera_frames.append(cam_frame)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    camera_frames.append(coordinate_frame)
    o3d.visualization.draw_geometries(camera_frames)
    

    # multi-view reconstruction example
    
    all_pcds = []

    for frame_i in range(num_frames):
        for i in cam_views:
            cam_name = f'cam{i}'

            # simulate different RGB and depth images for each view
            frame_name = frame_dict[cam_name][frame_i]
            rgb = np.asarray(Image.open(os.path.join(data_path, cam_name, 'rgb', f'{frame_name}.png')).convert('RGB'))
            depth = np.load(os.path.join(data_path, cam_name, 'depth', f'{frame_name}.npy')) / 1000.


            pcd_np, pcd_o3d = convert_RGBD_to_open3d(rgb, depth, info_dict[cam_name]['intrinsics'], info_dict[cam_name]['extrinsics'])
            all_pcds.append(pcd_o3d)

            # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([pcd_o3d, coordinate_frame])

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8, origin=[0, 0, 0])
        all_pcds.append(coordinate_frame)
        o3d.visualization.draw_geometries(all_pcds)