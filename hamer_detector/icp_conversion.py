import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import json
import trimesh
import os
import open3d as o3d
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord


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

def visualize_full_depth_image(depth_img):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.imshow(depth_img, cmap='gray')
    plt.colorbar(label="Depth Value")
    plt.title("Full Depth Image")
    plt.axis('off')
    plt.show()

def create_open3d_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def apply_icp(source_points, target_points, threshold=0.01):
    """
    Run ICP to align source_points (HaMeR mesh) to target_points (real point cloud).
    """
    source_pcd = create_open3d_point_cloud(source_points)
    target_pcd = create_open3d_point_cloud(target_points)

    # Initial alignment is identity
    trans_init = np.eye(4)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    return reg_p2p.transformation


def compute_aligned_hamer_translation(hamer_vertices, hand_point_cloud, mask, camera_intrinsics):
    # Flip and scale
    hamer_vertices = hamer_vertices.copy()


    # print(f"💡hamer: {len(hamer_vertices)} points", f"💡 Point cloud size: {hand_point_cloud.shape[0]} points")
    # Z alignment
    if hand_point_cloud.shape[0] > 0:
        # filter out the points with unreasonable depth < 0.2 m and > 2m
        valid_points = hand_point_cloud[
        (hand_point_cloud[:, 2] > 0.2) & (hand_point_cloud[:, 2] < 2)]
        hand_point_cloud = valid_points
        # Only keep points with reasonable Z values (in meters)
        valid_z = (hand_point_cloud[:, 2] > 0.2) & (hand_point_cloud[:, 2] < 1.5)
        num_valid_points = np.sum(valid_z)
        print(f"✅ Valid depth points: {num_valid_points}")

        if num_valid_points < 100:
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
    else:
        print("Warning: No valid mask contour, skipping 2D alignment.")

    # Debug visualization of ICP alignment (depth point cloud, original, aligned)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(hand_point_cloud[:, 0], hand_point_cloud[:, 1], hand_point_cloud[:, 2], c='b', label='Depth Point Cloud', alpha=0.3)
    ax.scatter(hamer_vertices[:, 0], hamer_vertices[:, 1], hamer_vertices[:, 2], c='r', label='Original HAMER Vertices', alpha=0.3)
    ax.scatter(hamer_vertices[:, 0], hamer_vertices[:, 1], hamer_vertices[:, 2], c='g', label='Aligned HAMER Vertices', alpha=0.6)

    ax.set_title('ICP Alignment Debug Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    # plt.show()

    return hamer_vertices


def visualize_projection_overlay(color_img, hamer_vertices, camera_intrinsics, mask=None):
    """
    Overlays the projected HaMeR mesh onto the color image and optionally the hand mask.
    """
    img_overlay = color_img.copy()
    verts_2d = project_points_to_image(hamer_vertices, camera_intrinsics)
    for (u, v) in verts_2d:
        u = int(round(u))
        v = int(round(v))
        if 0 <= u < img_overlay.shape[1] and 0 <= v < img_overlay.shape[0]:
            cv2.circle(img_overlay, (u, v), 1, (0, 0, 255), -1)  # Red dots for projection

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
    if mask is not None:
        plt.imshow(mask, alpha=0.3, cmap='jet')
    plt.title("HaMeR Projection Overlay")
    plt.axis('off')
    plt.show()


# === Projected 3D Sphere Overlay ===
def render_projected_sphere_overlay(color_img, translation, camera_intrinsics, radius=0.06):
    import trimesh
    sphere = trimesh.creation.uv_sphere(radius=radius)
    sphere.apply_translation(translation)
    verts_2d = project_points_to_image(np.array(sphere.vertices), camera_intrinsics)
    img_overlay = color_img.copy()
    for (u, v) in verts_2d:
        u = int(round(u))
        v = int(round(v))
        if 0 <= u < img_overlay.shape[1] and 0 <= v < img_overlay.shape[0]:
            cv2.circle(img_overlay, (u, v), 1, (0, 255, 0), -1)  # Green for sphere
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
    plt.title("Projected Sphere Overlay (Green)")
    plt.axis('off')
    plt.show()
def render_textured_sphere_projection(color_img, translation, camera_intrinsics, radius=0.06):
    import pyrender
    import trimesh
    from PIL import Image

    # Create icosphere and apply translation
    sphere = trimesh.creation.icosphere(radius=radius)
    sphere.apply_translation(translation)

    print("Sphere translation (Z):", translation[2])
    print("Sphere bounds:", sphere.bounds)

    # DEBUG: Use solid color instead of texture to test visibility
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.0, 1.0, 0.0, 1.0],  # Bright green
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True  # Fix
    )
    mesh = pyrender.Mesh.from_trimesh(sphere, material=material, smooth=True)
    
    # Setup scene with ambient light and background color
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0.0, 0.0, 0.0])

    # Add camera node and light
    fx, fy, cx, cy = camera_intrinsics['fx'], camera_intrinsics['fy'], camera_intrinsics['cx'], camera_intrinsics['cy']
    height, width = color_img.shape[:2]
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.01, zfar=3.0)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.array([
        [1,  0,  0],
        [ 0,  -1,  0],
        [ 0,  0, -1]
    ])  # 180-degree rotation around Y axis

    scene.add(camera, pose=camera_pose)

    # Print camera field of view and pose
    print("Camera FOV:")
    print(f"  fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
    print("Camera pose matrix:\n", camera_pose)

    light = pyrender.SpotLight(color=np.ones(3), intensity=5.0,
                                innerConeAngle=np.pi/16.0,
                                outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)

    scene.add(mesh)
    scene.ambient_light = np.array([0.5, 0.5, 0.5, 1.0])

    # Render
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color, depth = renderer.render(scene)
    # Debug: Save intermediate outputs
    cv2.imwrite("debug_color_rendered.png", color)
    cv2.imwrite("debug_depth_rendered.png", (depth * 255).astype(np.uint8))
    print("Rendered depth min/max:", depth.min(), depth.max())
    # Visualize
    alpha = (depth > 0).astype(np.float32)[..., None]
    blended = color_img.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.title("Textured Sphere Projection")
    plt.axis('off')
    plt.show()


def visualize_3d_scene_with_sphere_and_camera(translation, hand_point_cloud, depth_image = None, camera_intrinsics = None):
    
    def create_point_cloud_from_depth(depth_img, camera_intrinsics):
        height, width = depth_img.shape
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
        u_coords = u_coords.flatten()
        v_coords = v_coords.flatten()
        depth = depth_img.flatten().astype(np.float32)
        valid = depth > 0
        u_coords = u_coords[valid]
        v_coords = v_coords[valid]
        depth = depth[valid]
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        X = (u_coords - cx) * depth / fx
        Y = (v_coords - cy) * depth / fy
        Z = depth
        return np.stack((X, Y, Z), axis=-1)

    geometries = []

    # Camera origin as coordinate frame
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(camera_frame)

    # Sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.06)
    sphere.paint_uniform_color([0.2, 0.8, 0.2])  # green
    sphere.translate(translation)
    geometries.append(sphere)

    # Hand point cloud
    if hand_point_cloud is not None and hand_point_cloud.shape[0] > 0:
        hand_pcd = o3d.geometry.PointCloud()
        hand_pcd.points = o3d.utility.Vector3dVector(hand_point_cloud)
        hand_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red
        geometries.append(hand_pcd)

    # Full depth point cloud
    if depth_image is not None and camera_intrinsics is not None:
        full_points = create_point_cloud_from_depth(depth_image, camera_intrinsics)
        full_points = full_points[(full_points[:,2] > 0) & (full_points[:,2] < 3)]
        if full_points.shape[0] > 0:
            full_pcd = o3d.geometry.PointCloud()
            full_pcd.points = o3d.utility.Vector3dVector(full_points)
            full_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # gray
            geometries.append(full_pcd)

    o3d.visualization.draw_geometries(geometries)
def render_scene_to_rgbd(translation, hand_point_cloud, depth_image, camera_intrinsics, image_size=(640, 360)):
    

    def create_point_cloud_from_depth_with_color(depth_img, color_img, camera_intrinsics):
        height, width = depth_img.shape
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
        u_coords = u_coords.flatten()
        v_coords = v_coords.flatten()
        depth = depth_img.flatten().astype(np.float32)
        valid = depth > 0
        u_coords = u_coords[valid]
        v_coords = v_coords[valid]
        depth = depth[valid]
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        X = (u_coords - cx) * depth / fx
        Y = (v_coords - cy) * depth / fy
        Z = depth
        points = np.stack((X, Y, Z), axis=-1)
        colors = color_img[v_coords, u_coords][:, ::-1] / 255.0  # BGR → RGB
        return points, colors

    # Create geometry list
    geometries = []
    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.06)
    # sphere.paint_uniform_color([0.2, 0.8, 0.2])
    # sphere.translate(translation)
    # geometries.append(("sphere", sphere))

    # if hand_point_cloud is not None and hand_point_cloud.shape[0] > 0:
    #     hand_pcd = o3d.geometry.PointCloud()
    #     hand_pcd.points = o3d.utility.Vector3dVector(hand_point_cloud)
    #     hand_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    #     geometries.append(("hand", hand_pcd))
    if depth_image is not None and color_img is not None:
        full_points, full_colors = create_point_cloud_from_depth_with_color(depth_image, color_img, camera_intrinsics)
        valid = (full_points[:, 2] > 0) & (full_points[:, 2] < 3)
        full_points = full_points[valid]
        full_colors = full_colors[valid]

        # Remove points near hand_point_cloud
        # Remove points within the bounding box of the hand point cloud
        if hand_point_cloud is not None and hand_point_cloud.shape[0] > 0:
            min_corner = hand_point_cloud.min(axis=0)
            max_corner = hand_point_cloud.max(axis=0)
            in_box = np.all((full_points >= min_corner) & (full_points <= max_corner), axis=1)
            full_points = full_points[~in_box]
            full_colors = full_colors[~in_box]

        if full_points.shape[0] > 0:
            full_pcd = o3d.geometry.PointCloud()
            full_pcd.points = o3d.utility.Vector3dVector(full_points)
            full_pcd.colors = o3d.utility.Vector3dVector(full_colors)
            geometries.append(("scene", full_pcd))
    
    w, h = image_size
    renderer = OffscreenRenderer(w, h)
    renderer.scene.set_background([0, 0, 0, 1])

    # Setup camera
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(w, h, camera_intrinsics['fx'], camera_intrinsics['fy'],
                             camera_intrinsics['cx'], camera_intrinsics['cy'])
    extrinsic = np.eye(4)  # camera at origin, looking along +Z
    extrinsic[:3, :3] = np.array([
        [1,  0,  0],
        [0, 1,  0],
        [0,  0, 1]
    ])  # Flip Y and Z to match input convention
    renderer.setup_camera(intrinsic, extrinsic)

    # Add geometry
    for name, geom in geometries:
        material = MaterialRecord()
        material.shader = "defaultUnlit"
        renderer.scene.add_geometry(name, geom, material)

    # Render and return
    color = renderer.render_to_image()
    depth = renderer.render_to_depth_image(z_in_view_space=False)
    color_np = np.asarray(color)
    depth_np = np.asarray(depth)
    return color_np, depth_np
