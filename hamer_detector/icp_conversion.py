import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import open3d as o3d


def get_mask_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        return contours[0].reshape(-1, 2)  # Use the largest contour
    else:
        return np.zeros((0, 2), dtype=np.float32)

def compute_2d_alignment(src_pts, dst_pts):
    # src_pts: projected mesh (N,2), dst_pts: mask contour (M,2)
    # Simple translation matching using centroids for now
    src_centroid = src_pts.mean(axis=0)
    dst_centroid = dst_pts.mean(axis=0)
    translation = dst_centroid - src_centroid
    return translation

# === Project 3D points to 2D image coordinates ===
def project_points_to_image(points_3d, intrinsics):
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    u = (points_3d[:,0] * fx / points_3d[:,2]) + cx
    v = (points_3d[:,1] * fy / points_3d[:,2]) + cy
    return np.stack((u, v), axis=-1)

def pixel_to_camera(u, v, depth, camera_intrinsics):
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])

def extract_hand_point_cloud(mask, depth_img, camera_intrinsics):
    """
    Given a hand mask and aligned depth image, extract 3D points in camera frame.
    """
    points = []
    v_coords, u_coords = np.where(mask > 0)
    for u, v in zip(u_coords, v_coords):
        depth = depth_img[v, u]
        if depth > 0:
            point = pixel_to_camera(u, v, depth, camera_intrinsics)
            points.append(point)
    return np.array(points)



def compute_aligned_hamer_translation(hamer_vertices, hand_point_cloud, mask, camera_intrinsics):
    # Flip and scale
    hamer_vertices = hamer_vertices.copy()



    # Z alignment
    if hand_point_cloud.shape[0] > 0:
        # filter out the points with unreasonable depth < 0.2 m and > 2m
        valid_points = hand_point_cloud[(hand_point_cloud[:, 2] > 0.2) & (hand_point_cloud[:, 2] < 2)]
        hand_point_cloud = valid_points
        # Only keep points with reasonable Z values (in meters)
        valid_z = (hand_point_cloud[:, 2] > 0.2) & (hand_point_cloud[:, 2] < 1.5)
        num_valid_points = np.sum(valid_z)

        if num_valid_points < 50:
            print("⚠️ Not enough valid depth points. Skipping alignment.")
            return None  # or return unaligned vertices if preferred

        filtered_pc = hand_point_cloud[valid_z]
        pointcloud_z = np.percentile(filtered_pc[:, 2], 10)  # or use np.median(filtered_pc[:, 2])
        mesh_z = np.percentile(hamer_vertices[:, 2], 10)
        z_offset = pointcloud_z - mesh_z
        hamer_vertices[:, 2] += z_offset
    else:
        print("⚠️ Empty point cloud!")
        return None


    # 2D projection
    verts_2d = project_points_to_image(hamer_vertices, camera_intrinsics)
    mesh_contour = verts_2d
    mask_contour = get_mask_contour(mask)

    if mask_contour.shape[0] > 0:
        translation_2d = compute_2d_alignment(mesh_contour, mask_contour)
        # print("2D translation (u,v):", translation_2d)
        hamer_vertices[:, 0] += translation_2d[0] / camera_intrinsics['fx'] * hamer_vertices[:, 2]
        hamer_vertices[:, 1] += translation_2d[1] / camera_intrinsics['fy'] * hamer_vertices[:, 2]


    return hamer_vertices
