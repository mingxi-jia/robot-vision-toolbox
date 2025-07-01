import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import trimesh
import numpy as np
import pyrender
from PIL import Image
import cv2
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from hamer.utils.renderer import create_raymond_lights
import sys
sys.path.append('./')
from human_segmentor.util import convert_images_to_video, get_first_frame
from util import convert_images_to_video
os.environ["PYOPENGL_PLATFORM"] = "egl"  # or "osmesa" if EGL fails
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
scene_bg_color = (0, 0, 0)

def add_sphere(r = 0.06):
    # 1. Create a UV sphere
    sphere = trimesh.creation.uv_sphere(radius=r)

    # 2. Load texture image
    texture_img =Image.open("hamer_detector/sphere_textures/bigger_distinguishable_checkerboard_texture.png").convert("RGB")
    texture_np = np.array(texture_img)

    # 3. Generate UV coordinates from normals (for wrapping)
    normals = sphere.vertices / np.linalg.norm(sphere.vertices, axis=1, keepdims=True)
    u = 0.5 + np.arctan2(normals[:, 2], normals[:, 0]) / (2 * np.pi)
    v = 0.5 - np.arcsin(normals[:, 1]) / np.pi
    uv = np.stack([u, v], axis=1)

    # 4. Assign UVs and texture
    sphere.visual = trimesh.visual.TextureVisuals(uv=uv)
    sphere.visual.material.image = texture_np
    sphere.metadata["type"] = "sphere"
    return sphere

def transform_mesh(mesh, translation, rotation_matrix=None):
    # Removed the extra rotation block for sphere

    if rotation_matrix is not None and len(rotation_matrix) > 0:
        if mesh.metadata.get("type") == "sphere":
            # Flip rotation by applying 180 deg around Z axis before rotation_matrix
            # flip_rot = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
            rot4x4 = np.eye(4)
            rot4x4[:3, :3] = rotation_matrix
            total_rot = rot4x4
            mesh.apply_transform(total_rot)
        else:
            rot4x4 = np.eye(4)
            rot4x4[:3, :3] = rotation_matrix
            mesh.apply_transform(rot4x4)

    # Apply translation
    trans_matrix = trimesh.transformations.translation_matrix(translation)
    mesh.apply_transform(trans_matrix)

    return mesh


def render_rgba_multiple(
        mesh,
        cam_t,
        camera_intrinsics, 
        rot_matrix = None,
        rot_axis=[1,0,0],
        rot_angle=0,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0,0,0),
        render_res=[256, 256],
        focal_length=None,
        is_right=False,
        renderer=None,
        debug=True,
    ):

    # renderer is now passed in from outside and reused

    transformed_mesh = transform_mesh(mesh, cam_t, rot_matrix)

    scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                            ambient_light=(0.4, 0.3, 0.3))
    pyrender_mesh = pyrender.Mesh.from_trimesh(transformed_mesh, smooth=True)
    scene.add(pyrender_mesh)

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.array([
        [1,  0,  0],
        [ 0,  -1,  0],
        [ 0,  0, -1]
    ])  # 180-degree rotation around Y axis
    
    # camera_center = [render_res[0] / 2., render_res[1] / 2.]
    
    # if isinstance(focal_length, dict):
    #     fx = focal_length.get("fx", 1.0)
    #     fy = focal_length.get("fy", 1.0)
    #     cx = focal_length.get("cx", camera_center[0])
    #     cy = focal_length.get("cy", camera_center[1])
    # else:
    #     fx = fy = focal_length
    #     cx = camera_center[0]
    #     cy = camera_center[1]
    
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy,
                                        cx=cx, cy=cy, zfar=1e12)

    # Create camera node and add it to pyRender scene
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(camera_node)
    # add light
    
    # Add directional light at camera pose
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
    light_node = pyrender.Node(light=light, matrix=camera_pose)
    scene.add_node(light_node)
    
    # Add orientation arrows for Z (red) and X (green) axes in local frame
    from trimesh.creation import cylinder, cone

    def make_axis_arrow(color_rgb, direction='z'):
        arrow_length = 0.15
        arrow_radius = 0.01
        base = cylinder(radius=arrow_radius, height=arrow_length, sections=12)
        tip = cone(radius=arrow_radius * 2, height=arrow_length * 0.3, sections=12)
        tip.apply_translation([0, 0, arrow_length])
        arrow = base + tip
        arrow.visual.vertex_colors = np.tile(np.array(color_rgb) * 255, (arrow.vertices.shape[0], 1))
        if direction == 'x':
            rot = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
            arrow.apply_transform(rot)
        elif direction == 'y':
            rot = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
            arrow.apply_transform(rot)
        return arrow

    if debug:
        z_arrow = make_axis_arrow((1, 0, 0), 'z')    # Red Z
        x_arrow = make_axis_arrow((0, 1, 0), 'x')    # Green X
        y_arrow = make_axis_arrow((0, 0.6, 1), 'y')  # Blue Y
        for arr in [z_arrow, x_arrow, y_arrow]:
            transformed_arrow = transform_mesh(arr.copy(), cam_t, rot_matrix)
            arrow_mesh = pyrender.Mesh.from_trimesh(transformed_arrow, smooth=False)
            scene.add(arrow_mesh)
    
    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    scene.clear()

    return color


def replace_sphere(mesh_folder, image_folder, depth_folder, output_folder, intrinsics_path, ori_depth_folder, debug=False):
    import json
    from multiprocessing import Pool
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith("_segmented.png")])
    first_image_path = os.path.join(image_folder, image_files[0])
    print(first_image_path)
    image = cv2.imread(first_image_path)
    height, width, _ = image.shape
    # # Load hand mesh info
    # with open(os.path.join(mesh_folder, "hand_pose_camera_info.json"), "r") as f:
    #     hand_data = json.load(f)

    with open(os.path.join(mesh_folder, "hand_pose_camera_info_smoothed.json"), "r") as f:
        hand_data = json.load(f)

    # Load camera intrinsics
    if os.path.exists(intrinsics_path):
        with open(intrinsics_path, "r") as f:
            camera_intrinsics = json.load(f)
    else:
        camera_intrinsics = None

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
    tasks = [(fname, mesh_folder, image_folder, depth_folder, output_folder, camera_intrinsics, hand_data, renderer, ori_depth_folder, debug) for fname in image_files]

    for args in tasks:
        process_frame(*args)
    renderer.delete()


def process_frame(fname, mesh_folder, image_folder, depth_folder, output_folder, camera_intrinsics, hand_data, renderer, ori_depth_folder, debug):
    frame_id = os.path.splitext(fname)[0].split('_')[0]
    img_path = os.path.join(image_folder, fname)
    img = np.asarray(Image.open(img_path).convert("RGB"))
    height, width, _ = img.shape
    mesh = add_sphere()

    color_output_dir = os.path.join(output_folder, 'sphere_rgb')
    os.makedirs(color_output_dir, exist_ok=True)
    depth_output_dir = os.path.join(output_folder, 'sphere_depth')
    os.makedirs(depth_output_dir, exist_ok=True)
    
    matched_key = next((key for key in hand_data if frame_id in key), None)
    if matched_key is None:
        # print(f"Skipping frame {frame_id} hand data not found)")
        return

    cam_t = np.array(hand_data[matched_key]["pred_cam_t"])
    rot_mat = np.array(hand_data[matched_key]['global_orient'][0])

    # Apply flip rotation to rot_mat
    flip_rot = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    rot4x4 = np.eye(4)
    rot4x4[:3, :3] = rot_mat
    flipped_rot_mat = (flip_rot @ rot4x4)[:3, :3]


    # Save sphere pose (cam_t and quaternion from flipped_rot_mat)
    from scipy.spatial.transform import Rotation as R
    sphere_pose_dict = {}
    pose_output_path = os.path.join(output_folder, 'sphere_pose.json')
    if os.path.exists(pose_output_path):
        with open(pose_output_path, 'r') as f:
            sphere_pose_dict = json.load(f)
    
    quat = R.from_matrix(flipped_rot_mat).as_quat()
    # Camera intrinsics
    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]

    # Project cam_t to (u, v, z)
    x, y, z = cam_t
    u = int(round((x * fx) / z + cx))
    v = int(round((y * fy) / z + cy))

    sphere_pose_dict[frame_id] = {
        "pred_cam_t": cam_t.tolist(),
        "global_orient_quat": quat.tolist(),
        "uvz_center": [u, v, z*1000]
    }

    # Additional code to compute handmask center and average depth
    handmask_path = os.path.join(mesh_folder, f"{frame_id}_handmask.png")
    if os.path.exists(handmask_path):
        hand_mask = cv2.imread(handmask_path, cv2.IMREAD_GRAYSCALE)
        ys, xs = np.where(hand_mask > 127)
        if len(xs) > 0:
            center_u = int(np.mean(xs))
            center_v = int(np.mean(ys))
            depth_path = os.path.join(ori_depth_folder, f"{int(frame_id):06d}.npy")
            if os.path.exists(depth_path):
                depth_img = np.load(depth_path)
                hand_depth_vals = depth_img[ys, xs]
                valid_depths = hand_depth_vals[hand_depth_vals > 0]
                if valid_depths.size > 0:
                    avg_depth = float(np.mean(valid_depths))/1000
                    if 0.1 <= avg_depth <= 3.0:
                        if frame_id not in sphere_pose_dict:
                            sphere_pose_dict[frame_id] = {}
                        sphere_pose_dict[frame_id]['uvz_handmask'] = [center_u, center_v, avg_depth]

    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
    )

    rendered_img = render_rgba_multiple(mesh, cam_t, camera_intrinsics, rot_matrix=flipped_rot_mat, render_res=[width, height], focal_length=camera_intrinsics, **misc_args, renderer=renderer, debug=debug)
    input_img = img.astype(np.float32)[:, :, ::-1] / 255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
    input_img_overlay = input_img[:, :, :3] * (1 - rendered_img[:, :, 3:]) + rendered_img[:, :, :3] * rendered_img[:, :, 3:]
    rgb_img = 255 * input_img_overlay[:, :, ::-1]
    cv2.imwrite(os.path.join(color_output_dir, f'{frame_id}_final.png'), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

    # Add the full sphere geometry to update the depth map
    if camera_intrinsics is not None and depth_folder is not None:
        depth_path = os.path.join(depth_folder, f"{int(frame_id):06d}_segmented.npy")
        if os.path.exists(depth_path):
            depth = np.load(depth_path)
            H, W = depth.shape

            # Transform sphere mesh
            sphere_mesh = transform_mesh(add_sphere(), cam_t, flipped_rot_mat)
            vertices = sphere_mesh.vertices
            sphere_center = sphere_mesh.bounding_box.centroid
            center_x, center_y, center_z = sphere_center

            # Project into image coordinates
            center_u = int(round((center_x * fx) / center_z + cx))
            center_v = int(round((center_y * fy) / center_z + cy))
            sphere_pose_dict[frame_id]['uvz_center'] = [center_u, center_v, center_z]
            z_vals = vertices[:, 2]
            fx, fy = camera_intrinsics["fx"], camera_intrinsics["fy"]
            cx, cy = camera_intrinsics["cx"], camera_intrinsics["cy"]
            x_proj = (vertices[:, 0] * fx) / z_vals + cx
            y_proj = (vertices[:, 1] * fy) / z_vals + cy
            u_proj = np.round(x_proj).astype(int)
            v_proj = np.round(y_proj).astype(int)

            updated_depth = depth.copy()
            for i in range(len(vertices)):
                u, v = u_proj[i], v_proj[i]
                z = z_vals[i]
                if z < 10:
                    z *= 1000  # convert to mm if needed
                if 0 <= v < H and 0 <= u < W:
                    if updated_depth[v, u] == 0 or z < updated_depth[v, u]:
                        updated_depth[v, u] = z

            np.save(os.path.join(depth_output_dir, f'{frame_id}_sphere.npy'), updated_depth)

            # Save visualization
            import matplotlib.pyplot as plt
            vis_path = os.path.join(depth_output_dir, f'{frame_id}_sphere_debug.png')
            plt.imsave(vis_path, updated_depth, cmap='plasma')
        
    with open(pose_output_path, 'w') as f:
        json.dump(sphere_pose_dict, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Replace hand with sphere in segmented images")
    parser.add_argument("--mesh_folder", type=str, required=True, help="Folder containing hand mesh data")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing segmented images")
    parser.add_argument("--depth_folder", type=str, required=True, help="Folder containing depth images")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save output images and depth maps")
    parser.add_argument("--intrinsics_path", type=str, required=True, help="Path to camera intrinsics JSON file")
    parser.add_argument("--ori_depth_folder", type=str, required=True, help="Original depth folder for reference depth images")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for additional visualizations") 
    replace_sphere(
        mesh_folder=parser.parse_args().mesh_folder,
        image_folder=parser.parse_args().image_folder,
        depth_folder=parser.parse_args().depth_folder,
        output_folder=parser.parse_args().output_folder,
        intrinsics_path=parser.parse_args().intrinsics_path,
        ori_depth_folder=parser.parse_args().ori_depth_folder,
        debug=parser.parse_args().debug
    )
    convert_images_to_video(parser.parse_args().output_folder, framerate=10)