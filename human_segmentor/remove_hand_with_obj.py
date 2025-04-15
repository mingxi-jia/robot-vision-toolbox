import os
import trimesh
import numpy as np
import pyrender
from PIL import Image
import cv2
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_and_save_orientation(mesh_path, overlay_image, output_image_path):
    if not os.path.exists(mesh_path):
        print(f"❌ File does not exist: {mesh_path}")
        return

    mesh = trimesh.load(mesh_path, process=False)
    verts = np.array(mesh.vertices)
    if verts.shape[0] == 0:
        print(f"❌ Empty mesh: {mesh_path}")
        return

    centered = verts - verts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered)
    orientation = Vt[2]
    if orientation[1] < 0:
        orientation = -orientation
    center = verts.mean(axis=0)

    # xlim = [-1, 1]
    # ylim = [-1, 1]
    # zlim = [0, -10]

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=5, alpha=0.2, color='gray')
    ax1.scatter(center[0], center[1], center[2], color='green', s=40, label='Center')
    ax1.quiver(center[0], center[1], center[2], *orientation, color='red', linewidth=1, length=0.1, normalize=True)
    ax1.set_title("3D Mesh + Orientation")
    # ax1.set_xlim(*xlim)
    # ax1.set_ylim(*ylim)
    # ax1.set_zlim(*zlim)
    ax1.view_init(elev=-30, azim=45)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(overlay_image)
    ax2.set_title("2D Projection Overlay")
    ax2.axis("off")
    # plt.show()
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=100)
    plt.close()


def add_sphere(mesh_folder, image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    texture_image = Image.open("hamer_detector/sphere_textures/bigger_distinguishable_checkerboard_texture.png").convert("RGBA")

    for fname in sorted(os.listdir(mesh_folder)):
        if not fname.endswith(".obj"):
            continue

        mesh_path = os.path.join(mesh_folder, fname)
        mesh = trimesh.load(mesh_path, process=False)
        verts = np.array(mesh.vertices)
        if verts.shape[0] == 0:
            print(f"❌ Skipping empty mesh: {fname}")
            continue

        # center = verts.mean(axis=0)
        # center = verts[0]
        # centered = verts - center
        # _, _, Vt = np.linalg.svd(centered)
        # orientation = Vt[2]
        # if orientation[1] < 0:
        #     orientation = -orientation
        # Use consistent origin and reference direction
        # wrist = verts[0]  # or a stable wrist index
        # middle = verts[744]  # or average over middle MCP region
        # ref_dir = middle - wrist
        # ref_dir /= np.linalg.norm(ref_dir)

        # # PCA orientation
        # centered = verts - wrist
        # _, _, Vt = np.linalg.svd(centered)
        # orientation = Vt[2]

        # # Flip PCA orientation if it opposes the reference
        # if np.dot(orientation, ref_dir) < 0:
        #     orientation = -orientation
        # center = wrist

        center = mesh.center_mass
        orientation = mesh.principal_inertia_vectors[2] 

        sphere = trimesh.creation.uv_sphere(radius=0.1)
        normals = sphere.vertices / np.linalg.norm(sphere.vertices, axis=1, keepdims=True)
        u = 0.5 + np.arctan2(normals[:, 2], normals[:, 0]) / (2 * np.pi)
        v = 0.5 - np.arcsin(normals[:, 1]) / np.pi
        uv = np.stack([u, v], axis=1)
        sphere.visual = trimesh.visual.texture.TextureVisuals(uv=uv)
        sphere.visual.material.image = np.array(texture_image)

        T = np.eye(4)
        T[:3, 3] = center

        z_axis = np.array([0, 0, 1])
        target = orientation / np.linalg.norm(orientation)
        v = np.cross(z_axis, target)
        c = np.dot(z_axis, target)
        if np.linalg.norm(v) > 1e-6:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))
            T[:3, :3] = R

        scene = pyrender.Scene()
        material = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=pyrender.Texture(source=texture_image, source_channels="RGBA"),
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
        sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=material, smooth=False)
        scene.add(sphere_mesh, pose=T)
        cam = pyrender.IntrinsicsCamera(fx=387.12, fy=386.77, cx=321.97, cy=243.21)
        # cam = pyrender.IntrinsicsCamera(fx=615.35, fy=615.40, cx=313.39, cy=251.59)
        scene.add(cam, pose=np.eye(4))
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(light, pose=np.eye(4))

        r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
        color, _ = r.render(scene)
        rendered = color.astype(np.float32) / 255.0

        frame_id = fname.replace(".obj", "").split("_")[1]
        img_name = f"frame_{frame_id}_final.png"
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_name}")
            continue

        original_img = cv2.imread(img_path)
        original_img = cv2.resize(original_img, (640, 480))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mask = np.any(rendered < 0.99, axis=-1).astype(np.float32)[..., None]
        blended = original_img * (1 - mask) + rendered * mask

        out_path = os.path.join(output_folder, img_name)
        blended_save = (blended * 255).astype(np.uint8)[..., ::-1]
        cv2.imwrite(out_path, blended_save)
        print(f"✅ Saved: {out_path}")

        debug_image_name = f"frame_{frame_id}_debug.png"
        debug_out_path = os.path.join(output_folder, debug_image_name)
        visualize_and_save_orientation(mesh_path=mesh_path, overlay_image=blended, output_image_path=debug_out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay wrist-centered spheres onto images using .obj hand meshes.")
    parser.add_argument("--mesh_folder", type=str, default = "hamer_detector/hamer_output", help="Path to folder containing .obj files")
    parser.add_argument("--image_folder", type=str, default="hamer_detector/segmentation_output", help="Path to folder containing *_final.png images")
    parser.add_argument("--output_folder", type=str, default="hamer_detector/sphere_overlay", help="Folder to save blended results")
    args = parser.parse_args()
    add_sphere(args.mesh_folder, args.image_folder, args.output_folder)


