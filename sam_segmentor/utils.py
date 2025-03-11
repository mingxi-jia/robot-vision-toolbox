import os
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

from sam_segmentor.configs import DEPTH_MINMAXS, INTRINSICS, EXTRINSICS

def project_rgb_and_mask_to_cloud(rgb, depth, mask, camera_name):
    intrinsic_matrix, extrinsic_matrix, depth_minmax = INTRINSICS[camera_name], EXTRINSICS[camera_name], DEPTH_MINMAXS[camera_name]

     # resize the segmentation mask
    if mask.shape != depth.shape[:2]:
        segmentation_mask_resized = cv2.resize(
            mask[0].astype(np.uint8),  # Convert bool to uint8
            (depth.shape[1], depth.shape[0])  # Resize to (width, height)
        ).astype(bool)  # Convert back to boolean if needed
    else:
        segmentation_mask_resized = mask[0]

    depth[segmentation_mask_resized == False] = 10.0

    # Compute point cloud
    cloud = get_cloud_from_depth(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsic_matrix,
        extrinsics=extrinsic_matrix
    )

    z_min, z_max = depth_minmax
    # filter ws z
    z_cond = (cloud[:, 2] < z_max) * (cloud[:, 2] > z_min)
    cloud = cloud[z_cond]
    return cloud

def get_cloud_from_depth(rgb, depth, intrinsics, extrinsics):
    height, width = depth.shape[:2]

    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth, rgb[..., 0], rgb[..., 1], rgb[..., 2]]).transpose(1, 2, 0)
    cloud_RT_camera = points.reshape(-1,6)
    xyz_RT_robot = transform(cloud_RT_camera[:,:3], extrinsics)
    cloud_RT_robot = np.concatenate([xyz_RT_robot, cloud_RT_camera[:,3:]], axis=-1)
    return cloud_RT_robot

def transform(cloud: np.array, world_T_cam: np.array, isPosition=True):
    ''' Apply the homogeneous transform T to the point cloud. Use isPosition=False if transforming unit vectors.'''
    n = cloud.shape[0]
    cloud = cloud.T
    augment = np.ones((1, n)) if isPosition else np.zeros((1, n))
    cloud = np.concatenate((cloud, augment), axis=0)
    cloud = np.dot(world_T_cam, cloud)
    cloud = cloud[0:3, :].T
    return cloud

def normalize_depth(depth: np.array, normalize_range: list):
    depth_min, depth_max = normalize_range
    return (depth - depth_min) / (depth_max - depth_min)

def visualize_prompt(first_frame, points, labels, first_mask=None):
    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.imshow(first_frame)
    show_points(points, labels, plt.gca())
    if first_mask is not None:
        show_mask(first_mask, plt.gca(), obj_id=1)
    plt.show(block=True)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))