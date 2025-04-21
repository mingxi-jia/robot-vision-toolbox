# cam_intrinsic = np.array([
#     [387.12786865234375, 0.0,               321.97259521484375], 
#     [0.0,                386.7669677734375, 243.21249389648438], 
#     [0.0,                0.0,               1.0]
# ])
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
    return sphere


def transform_mesh(mesh, translation, rotation_matrix = None, rot_axis=[1, 0, 0], rot_angle=0):
    if rotation_matrix is not None:
        rot4x4 = np.eye(4)
        rot4x4[:3, :3] = rotation_matrix

        mesh.apply_transform(rot4x4)

    # Optional additional rotation
    rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
    mesh.apply_transform(rot)

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

    if is_right is not None and not is_right:
        # flip rotation for left hand
        rot_matrix[:, 0] *= -1
    
    transformed_mesh = transform_mesh(mesh, cam_t, rot_matrix, rot_axis=[1, 0, 0], rot_angle=0)

    scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                            ambient_light=(0.3, 0.3, 0.3))
    pyrender_mesh = pyrender.Mesh.from_trimesh(transformed_mesh, smooth=False)
    scene.add(pyrender_mesh)

    camera_pose = np.eye(4)
    
    camera_center = [render_res[0] / 2., render_res[1] / 2.]
    camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                        cx=camera_center[0], cy=camera_center[1], zfar=1e12)

    # Create camera node and add it to pyRender scene
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(camera_node)
    # add light
    
    light_nodes = create_raymond_lights()
    for node in light_nodes:
        node.light.intensity = 3.0
        scene.add_node(node)
    
    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    renderer.delete()

    return color


def replace_sphere(mesh_folder, image_folder, output_folder):
    import yaml
    
    # Load hand mesh info
    json_path = os.path.join(mesh_folder, "hand_pose_camera_info_smoothed.json")
    with open(json_path, "r") as f:
        hand_data = json.load(f)
    
    # Load camera centroids
    # centroids_path = os.path.join(mesh_folder, "centroids.yml")
    centroids_path = os.path.join(mesh_folder, "centroids_smoothed.yml")
    with open(centroids_path, "r") as f:
        centroids = yaml.safe_load(f)
    
    cam_intrinsic = np.array([
        [387.12786865234375, 0.0,               321.97259521484375], 
        [0.0,                386.7669677734375, 243.21249389648438], 
        [0.0,                0.0,               1.0]
    ])
    
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith("_final.png")])

    for fname in image_files:
        frame_id = os.path.splitext(fname)[0].split('_')[1]  # assumes filename like '000123_final.png'
        print(f'processing frame-> {frame_id}')
        img_path = os.path.join(image_folder, fname)
        img = np.asarray(Image.open(img_path).convert("RGB"))  # float32 in [0,1]
        frame_id = 'frame_'+frame_id+'_left'
        mesh = add_sphere(r = 0.06)
        frame_id_right = 'frame_'+frame_id+'_right'
        if frame_id not in hand_data or frame_id not in centroids:
            if frame_id_right in hand_data:
                frame_id =frame_id_right
            else:
                print(f"Skipping {frame_id} (missing data)")
                continue
        
        cam_t = np.array(centroids[frame_id])  # translation vector
        scaled_focal_length = hand_data[frame_id]['scaled_focal_length']

        misc_args = dict(
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )

        # Apply reorientation to global_orient
        rot_mat = np.array(hand_data[frame_id]['global_orient'][0])
        # rot_mat_converted = rot_mat
        rendered_img = render_rgba_multiple(mesh, cam_t, rot_matrix = rot_mat, render_res = [640, 480], is_right=False, focal_length=scaled_focal_length, **misc_args)

        input_img = img.astype(np.float32)[:,:,::-1]/255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
        input_img_overlay = input_img[:,:,:3] * (1-rendered_img[:,:,3:]) + rendered_img[:,:,:3] * rendered_img[:,:,3:]
        rgb_img = 255*input_img_overlay[:, :, ::-1]
        cv2.imwrite(os.path.join(output_folder, f'{frame_id}_final.jpg'), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        
        
    convert_images_to_video(output_folder)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay wrist-centered spheres onto images using .obj hand meshes.")
    parser.add_argument("--mesh_folder", type=str, default = "hamer_detector/hamer_output", help="Path to folder containing .obj files")
    parser.add_argument("--image_folder", type=str, default="hamer_detector/segmentation_output", help="Path to folder containing *_final.png images")
    parser.add_argument("--output_folder", type=str, default="hamer_detector/sphere_overlay", help="Folder to save blended results")
    args = parser.parse_args()
    replace_sphere(args.mesh_folder, args.image_folder, args.output_folder)


