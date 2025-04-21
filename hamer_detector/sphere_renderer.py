import copy
import numpy as np
import pyrender
import os
import trimesh
from PIL import Image
from pyrender.constants import RenderFlags
from typing import List
import yaml
# from submodules.hamer.hamer.utils.renderer import Renderer, create_raymond_lights



class HandMarker:
    """
    JointMarker class to draw actions on images
    """

    def __init__(
        self,
        image_width,
        image_height,
        camera_scales,
        sphere_radius=0.008,
        znear=0.00001,
        zfar=3.0,
    ):
        """
        Initializes an instance of the JointMarker class

        Parameters:
        - image_width (int): image width
        - image_height (int): image height
        - camera_scales (List[float]): scaling of spheres per camera
        - sphere_radius (float): sphere radius
        - znear (float): distance to near clipping plane for intrinsic camera
        - zfar (float): distance to the far clipping plane for intrinsic camera

        Returns:
        - None
        """
        self._image_width = image_width
        self._image_height = image_height
        self._sphere_radius = sphere_radius
        self._znear = znear
        self._zfar = zfar

        # Initialize offscreen renderer
        self._offscreen_renderer = pyrender.OffscreenRenderer(
            self._image_width, self._image_height
        )

        # Cache meshes and materials
        self.mesh_cache = {}  # camera_scale: (sphere_mesh, sphere_material)
        for camera_scale in camera_scales:
            per_cam_cache_dict = {}
            sphere_mesh = trimesh.creation.uv_sphere(
                radius=self._sphere_radius * camera_scale
            )
            sphere_material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0)
            per_cam_cache_dict["sphere"] = (sphere_mesh, sphere_material)
            self.mesh_cache[camera_scale] = per_cam_cache_dict

    def render_action(
        self,
        cam_intrinsic,
        cam_extrinsic,
        joint_matrices,
        joint_opens,
        camera_scale=1.0,
        sphere_colors=List[None],
    ):
        """
        Renders an action in a pyrender scene

        Parameters:
        - cam_intrinsic (np.array): camera intrinsics
        - cam_extrinsic (np.array): camera extrinsics
        - joint_matrices (List[np.array]): list of joint pose matrices
        - joint_opens (List[int]): list of joint open/close states \
            (only relevant for gripper joints)
        - camera_scale (int): camera scaling for sphere
        - sphere_colors (List[string]): colors for spheres

        Returns:
        - rendered_img: (h, w, 3) uint8 array
        """

        # Create a PyRender scene
        scene = pyrender.Scene()

        # Camera extrinsics
        cam_extrinsic = np.array(cam_extrinsic)

        # Create an intrinsic camera
        camera = pyrender.IntrinsicsCamera(
            fx=cam_intrinsic[0, 0],
            fy=cam_intrinsic[1, 1],
            cx=cam_intrinsic[0, 2],
            cy=cam_intrinsic[1, 2],
            znear=self._znear,
            zfar=self._zfar,
        )
        print(cam_extrinsic)
        print(cam_intrinsic)
        rotation_matrix = cam_extrinsic[:3, :3]

        # Define the rotation angle in radians (180 degrees)
        angle_degrees = -180
        angle_radians = np.radians(angle_degrees)
        rotation_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_radians), -np.sin(angle_radians)],
                [0, np.sin(angle_radians), np.cos(angle_radians)],
            ]
        )

        # Perform the rotation by multiplying the matrices
        rotated_rotation_matrix = np.dot(rotation_matrix, rotation_x)

        # Update the original 4x4 matrix with the rotated 3x3 rotation matrix
        cam_extrinsic[:3, :3] = rotated_rotation_matrix

        # Add the camera to the scene
        _ = scene.add(camera, pose=cam_extrinsic)

        # Draw spheres
        for i, (gripper_matrix, gripper_open) in enumerate(
            zip(joint_matrices, joint_opens)
        ):
            # Draw sphere
            if sphere_colors[i] is None:
                sphere_texture_name = (
                    "sphere_yellow_stripe_texture.png"
                    if gripper_open <= 0.1
                    else "sphere_cyan_stripe_texture.png"
                )
            else:
                color_mapping = {
                    "green": "bigger_distinguishable_checkerboard_texture.png",
                    "red": "sphere_red_stripe_texture.png",
                    "purple": "sphere_purple_stripe_texture.png",
                }
                sphere_texture_name = color_mapping[sphere_colors[i]]

            current_path = os.path.dirname(os.path.realpath(__file__))
            sphere_texture = pyrender.Texture(
                source=Image.open(
                    os.path.join(current_path, "sphere_textures", sphere_texture_name)
                ).convert("RGBA"),
                source_channels="RGBA",
            )

            sphere_mesh, sphere_material = self.mesh_cache[camera_scale]["sphere"]
            sphere_mesh, sphere_material = copy.deepcopy(sphere_mesh), copy.deepcopy(
                sphere_material
            )
            texture_image = np.ones((1, 1, 3), dtype=np.uint8) * 255
            texture = trimesh.visual.texture.TextureVisuals(
                uv=(
                    sphere_mesh.vertices[:, :2]
                    - np.min(sphere_mesh.vertices[:, :2], axis=0)
                )
                / np.ptp(sphere_mesh.vertices[:, :2], axis=0),
                image=texture_image,
            )
            sphere_mesh._visual = texture
            sphere_material.baseColorTexture = sphere_texture

            # Add base color factor
            # NOTE: this is not necessary, but kept for
            # reproducing original sphere colors from the paper
            base_color = (
                tuple([0.60392156862, 0.86274509803, 1.0, 1.0])
                if gripper_open > 0.1
                else tuple([1.0, 1.0, 0.0, 1.0])
            )
            sphere_material.baseColorFactor = base_color

            sphere = pyrender.Mesh.from_trimesh(sphere_mesh, material=sphere_material)
            _ = scene.add(sphere, pose=gripper_matrix)

        # Render the scene in offscreen mode
        flags = RenderFlags.FLAT
        rendered_img, _ = self._offscreen_renderer.render(scene, flags)
        return rendered_img


    def create_raymond_lights() -> List[pyrender.Node]:
        """
        Return raymond light nodes for the scene.
        """
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3,:3] = np.c_[x,y,z]
            nodes.append(pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix
            ))

        return nodes

    def render_rgba_multiple(
            self,
            vertices: List[np.array],
            cam_t: List[np.array],
            rot_axis=[1,0,0],
            rot_angle=0,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
        ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]

        # Load your texture image (must be square or UV-mapped properly)
        texture_img = Image.open("hamer_detector/sphere_textures/bigger_distinguishable_checkerboard_texture.png")  # Replace with your texture path
        texture_img = np.array(texture_img)

        mesh_list = []
        for t in cam_t:
            # Create a UV-mapped sphere
            sphere = trimesh.creation.uv_sphere(radius=0.05)
            sphere.visual = trimesh.visual.TextureVisuals(image=texture_img)

            # Position the sphere
            sphere.apply_translation(t)

            # Convert to pyrender mesh with texture
            mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
            mesh_list.append(mesh)
        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        for i,mesh in enumerate(mesh_list):
            scene.add(mesh, f'mesh_{i}')

        camera_pose = np.eye(4)
        # camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        focal_length = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        Renderer.add_point_lighting(scene, camera_node)
        Renderer.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color


def main():
    # Example setup:
    # Folder containing 20 images: "images/"
    # A file containing 20 (x,y,z) positions: "positions.npy" (shape: (20,3))
    # Modify these paths as needed.
    image_folder = "/home/xhe71/Desktop/demo_data/0417_test"
    position_file = "/home/xhe71/Desktop/demo_data/0417_test_hamer/centroids.yml"
    out_folder = "/home/xhe71/Desktop/demo_data/0417_test_out"

    os.makedirs(out_folder, exist_ok=True)

    # Load your 20 positions (shape: (20,3))
    positions_xyz = yaml.load(open(position_file, "r"), Loader=yaml.FullLoader)

    # Example: image size (adjust for your images)
    width, height = 640, 480

    # Example camera intrinsics for (640 x 480). Adjust to your real camera:
    # fx, fy ~ focal lengths; cx, cy ~ principal point
    cam_intrinsic = np.array([[389.0,   0.0, 323.8],
                              [  0.0, 389.0, 237.1],
                              [  0.0,   0.0,   1.0]], dtype=np.float32)

    # Example camera extrinsic as identity (if your images are already in correct orientation).
    # If you have a real extrinsic matrix, load/use that.
    cam_extrinsic = np.eye(4, dtype=np.float32)

    # Initialize the JointMarker
    # Suppose we only want one scale factor = 1.0
    marker = HandMarker(
        image_width=width,
        image_height=height,
        camera_scales=[1.0],
        sphere_radius=0.05,  # pick a radius you prefer
        znear=0.4,
        zfar=20.0,
    )

    img_path_list = os.listdir(image_folder)
    img_path_list = [i for i in img_path_list if i.endswith(".png")]
    img_path_list.sort()
    # Loop over all 20 images
    for i in img_path_list:
        # Load image (assuming they are named img_00.jpg, img_01.jpg, etc.)
        img_path = os.path.join(image_folder, i)
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found.")
            continue
        pil_img = Image.open(img_path)
        pil_img = pil_img.resize((width, height))  # ensure it matches the renderer size if needed

        # Build a 4x4 transform for the sphere at positions_xyz[i]
        image_fn = i.split(".")[0] + '_left'
        x, y, z = positions_xyz[image_fn]
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = x
        T[1, 3] = -y
        T[2, 3] = -z

        # Render the sphere
        # We pass a list of transforms, in this case just one sphere
        # 'joint_opens' can be arbitrary, e.g., [1] means "open" => sphere_cyan_stripe_texture
        # sphere_colors can be, e.g., ["green"] or [None]

        # Convert rendered to a PIL Image
        rendered_pil = Image.fromarray(rendered)
        # print()
        # Optional Step: Overlay or alpha-blend rendered sphere with the real image.
        # In simple form, you can just take the rendered sphere's color wherever it's non-black.
        # A more advanced approach would involve alpha channels, segmentation, etc.
        # Here is a basic blend approach:
        background = pil_img.convert("RGBA")
        foreground = rendered_pil.convert("RGBA")
        # Use alpha=0.7, for example
        blended = Image.blend(background, foreground, alpha=0.7)

        # Save the result
        out_path = os.path.join(out_folder, i)
        blended.save(out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()