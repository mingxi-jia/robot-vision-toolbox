import io
import os
os.environ["MUJOCO_GL"] = "osmesa"
import robosuite as suite
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
from robosuite.utils.camera_utils import get_real_depth_map
# from robosuite.controllers import load_controller_config
import numpy as np
import open3d as o3d
from urdfpy import URDF
import trimesh 
import matplotlib.pyplot as plt
import tempfile

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


def depth2fgpcd(depth, mask, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    mask = np.logical_and(mask, depth > 0)
    # mask = (depth <= 0.599/0.8)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd

def convert_RGBD_to_open3d(rgb, depth, intrinsics, extrinsics):
    assert rgb.shape[0] == depth.shape[0] and rgb.shape[1] == depth.shape[1]
    assert intrinsics.shape == (3, 3)
    assert extrinsics.shape == (4, 4)
    assert rgb.dtype == np.uint8
    assert depth.dtype == np.float32 or depth.dtype == np.float64

    cam_param = [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]
    pcd = depth2fgpcd(depth,  np.ones_like(depth, dtype=bool), cam_param)
    # pose = np.linalg.inv(ext_mat)
    pose = extrinsics
    # trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
    trans_pcd = np.einsum('ij,jk->ik', pose, np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0))
    trans_pcd = trans_pcd[:3, :].T
    pcd_o3d = np2o3d(trans_pcd, rgb.reshape(-1, 3).astype(np.float64) / 255)
    return pcd_o3d