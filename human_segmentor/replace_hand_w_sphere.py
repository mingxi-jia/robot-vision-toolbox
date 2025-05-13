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


def transform_mesh(mesh, translation, rotation_matrix = None, rot_axis=[1, 0, 0], rot_angle=0):
    if rotation_matrix is not None:
        rot4x4 = np.eye(4)
        if len(rotation_matrix) > 0:
            rot4x4[:3, :3] = rotation_matrix

        # If mesh is a sphere, rotate it to match HaMeR hand orientation
        if mesh.metadata.get("type") == "sphere":
            align = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
            mesh.apply_transform(align)

        mesh.apply_transform(rot4x4)

    # # Optional additional rotation
    # rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
    # mesh.apply_transform(rot)

    # Flip upside down (180 degrees around X-axis)
    # flip_matrix = trimesh.transformations.rotation_matrix(
    #     np.radians(180), [1, 0, 0]
    # )
    # mesh.apply_transform(flip_matrix)

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
    ):

    renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                            viewport_height=render_res[1],
                                            point_size=1.0)

    # if is_right is not None and not is_right:
    #     # flip rotation for left hand
    #     rot_matrix[:, 0] *= -1
    
    transformed_mesh = transform_mesh(mesh, cam_t, rot_matrix, rot_axis=[1, 0, 0], rot_angle=0)

    scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                            ambient_light=(0.4, 0.3, 0.3))
    pyrender_mesh = pyrender.Mesh.from_trimesh(transformed_mesh, smooth=False)
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

    z_arrow = make_axis_arrow((1, 0, 0), 'z')    # Red Z
    x_arrow = make_axis_arrow((0, 1, 0), 'x')    # Green X
    y_arrow = make_axis_arrow((0, 0.6, 1), 'y')  # Blue Y
    for arr in [z_arrow, x_arrow, y_arrow]:
        arr = transform_mesh(arr, cam_t, rot_matrix, rot_axis=[1, 0, 0], rot_angle=0)
        arrow_mesh = pyrender.Mesh.from_trimesh(arr, smooth=False)
        scene.add(arrow_mesh)
    
    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    renderer.delete()

    return color


def replace_sphere(mesh_folder, image_folder, output_folder, intrinsics_path):
    import yaml
    
    # Load hand mesh info
    # json_path = os.path.join(mesh_folder, "hand_pose_camera_info.json")
    json_path = os.path.join(mesh_folder, "hand_pose_camera_info_smoothed.json")
    with open(json_path, "r") as f:
        hand_data = json.load(f)
    
    # Load camera centroids
    # centroids_path = os.path.join(mesh_folder, "centroids.yml")
    # centroids_path = os.path.join(mesh_folder, "centroids_smoothed.yml")
    # with open(centroids_path, "r") as f:
    #     centroids = yaml.safe_load(f)
    
    # intrinsics_path = args.intrinsics_path if hasattr(args, "intrinsics_path") else None
    if os.path.exists(intrinsics_path):
        with open(intrinsics_path, "r") as f:
            camera_intrinsics = json.load(f)
    else:
        camera_intrinsics = None

    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith("_final.png")])

    for fname in image_files:
        frame_id = os.path.splitext(fname)[0].split('_')[1]  # assumes filename like '000123_final.png'
        print(f'processing frame-> {frame_id}')
        img_path = os.path.join(image_folder, fname)
        img = np.asarray(Image.open(img_path).convert("RGB"))  # float32 in [0,1]
        height, width, _ = img.shape
        mesh = add_sphere()

        # Find matching hand_data key
        matched_key = None
        for key in hand_data:
            if frame_id in key:
                matched_key = key
                break

        if matched_key is None:
            print(f"Skipping frame {frame_id} hand data not found)")
            continue
        
        # cam_t = np.array(centroids[frame_id])  # translation vector
        cam_t = np.array(hand_data[matched_key]["pred_cam_t"])


        misc_args = dict(
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )

        # Apply reorientation to global_orient
        rot_mat = np.array(hand_data[matched_key]['global_orient'][0])

        # rot_mat_converted = rot_mat
        focal_param = camera_intrinsics 
        rendered_img = render_rgba_multiple(mesh, cam_t, focal_param, rot_matrix = rot_mat, render_res = [width, height], focal_length=focal_param, **misc_args)

        input_img = img.astype(np.float32)[:,:,::-1]/255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
        input_img_overlay = input_img[:,:,:3] * (1-rendered_img[:,:,3:]) + rendered_img[:,:,:3] * rendered_img[:,:,3:]
        rgb_img = 255*input_img_overlay[:, :, ::-1]
        cv2.imwrite(os.path.join(output_folder, f'{frame_id}_final.jpg'), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        
if __name__ == "__main__":

    mesh_folder = "/home/xhe71/Desktop/robotool_data/Color_hamer"
    image_folder = "/home/xhe71/Desktop/robotool_data/Color_segmented"
    output_folder = "/home/xhe71/Desktop/robotool_data/Color_final"
    intrinsic_path = "/home/xhe71/Desktop/robotool_data/camera_intrinsics.json"
    replace_sphere(mesh_folder, image_folder, output_folder, intrinsic_path)