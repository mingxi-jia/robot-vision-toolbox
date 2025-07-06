# now we start coding the function
import io
import time
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
import imageio.v2 as imageio

# === STEP 1: Initialize robosuite environment ===
H, W = 128, 128

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["frontview"],
    camera_heights=128,
    camera_widths=128,
    render_camera="frontview",
)

intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=W, height=H, fx=154.50966799187808, fy=154.50966799187808, cx=W/2, cy=H/2
)

# grab the extrinsic of the camera so we can align the coordinate
extrinsic = np.linalg.inv(get_camera_extrinsic_matrix(env.sim, "frontview")) # world to pixel space

obs = env.reset()


NUM_STEPS = 10
for step in range(NUM_STEPS):

    action = np.random.uniform(-1, 1, size=env.robots[0].action_dim) * 2
    obs,reward, done,_ = env.step(action)

    rgb_image, depth_image = env.sim.render(
        camera_name="frontview",
        width=W,
        height=H,
        depth=True
    )
    rgb_image, depth_image = rgb_image[::-1], depth_image[::-1]
    # store the rgb_image locally
    rgb_image = np.array(rgb_image)
    imageio.imwrite(f"rgb_image_{step}.png", rgb_image)

    depth_image = get_real_depth_map(env.sim, depth_image)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.geometry.Image(rgb_image.astype(np.uint8)),
        depth=o3d.geometry.Image((depth_image * 1000).astype(np.uint16)),  # mm for Open3D
        depth_scale=1000.0,
        convert_rgb_to_intensity=False
    )
    # Integrate the RGBD image into the TSDF volume
    # === Initialize TSDF volume ===
    # Updated constructor - we now use the o3d.pipelines.integration.ScalableTSDFVolume
    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,  # 1 cm voxel size
        sdf_trunc=0.04,     # 4 cm truncation distance
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    tsdf_volume.integrate(
        rgbd,
        intrinsic=intrinsics,
        # Assuming the camera is at the origin for this example
        extrinsic=extrinsic  # camera at originf
    )

    #visualize this rgbd
    rgbd_vis = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.geometry.Image(rgb_image.astype(np.uint8)),
        depth=o3d.geometry.Image((depth_image * 1000).astype(np.uint16)),  # mm for Open3D
        depth_scale=1000.0,
        convert_rgb_to_intensity=False
    )
    # o3d.visualization.draw_geometries([rgbd_vis.color, rgbd_vis.depth])


    # # === STEP 4: Create point cloud for visualization/debugging ===
    pcd = tsdf_volume.extract_point_cloud()
    # o3d.visualization.draw_geometries([pcd])

    # print(f"Scene point cloud (fused) has {len(pcd.points):,} points")

    # === STEP 5: Load URDF and create a mesh for the robot arm ===
    urdf_path = "/home/mingxi/Desktop/rosella_work_folder/panda_description/urdf/panda_arm_hand.urdf"

    with open(urdf_path, 'r') as f:
        urdf_content = f.read()

    absolute_base_path = "/home/mingxi/Desktop/rosella_work_folder/panda_description/"
    # urdf_str = urdf_content.replace("package://panda_description/", absolute_base_path)
    urdf_str = (
        urdf_content
            .replace("package://panda_description", absolute_base_path)
            .replace("panda_description/meshes", f"{absolute_base_path}/meshes")
    )


    with tempfile.NamedTemporaryFile(mode="w+", suffix=".urdf", delete=False) as f:
        f.write(urdf_str)
        f.flush()
        robot_urdf = URDF.load(f.name)


    # -------- Get world frame pose of the robot state ---------
    # base_body = "robot0_link0" # hmmmm
    base_body = "robot0_base" # hmmmm
    base_pose = env.sim.data.get_body_xpos(base_body)  # (x, y, z, qx, qy, qz, qw)
    print("robot0_base position =", base_pose)
    print("robot0_base height z =", base_pose[2])
    base_quat = env.sim.data.get_body_xquat(base_body)  # (qx, qy, qz, qw)
    print("robot0_base quaternion =", base_quat)

    # build a 4*4 homogeneous transformation matrix
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial import cKDTree
    T_world_urdf = np.eye(4)
    T_world_urdf[:3, :3] = R.from_quat(base_quat, scalar_first=True).as_matrix()  # rotation part
    T_world_urdf[:3, 3] = base_pose[:3]  # translation part
    # ------------------------------------------------------------------

    # Get joint angles from simulation
    joint_names = [j.name for j in robot_urdf.actuated_joints]
    joint_positions = obs['robot0_joint_pos']  # robosuite joint angles

    # Map names to values
    joint_angles = dict(zip(joint_names, joint_positions))

    # Compute transformed robot meshes
    robot_mesh_dict = robot_urdf.visual_trimesh_fk(cfg=joint_angles) # returns mesh, transformation matrix relative to base


    # Sample points from each mesh surface
    sampled_points = []
    for mesh, pose in robot_mesh_dict.items():
        transformed = mesh.copy()
        transformed.apply_transform(pose)
        transformed.apply_transform(T_world_urdf)  # transform to world frame
        sampled_points.append(transformed.sample(5000))

    robot_points = np.vstack(sampled_points)
    print("PASS CHECK FOR MESHES!")
    # Create Open3D point cloud for KDTree
    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.points = o3d.utility.Vector3dVector(robot_points)
    robot_kdtree = o3d.geometry.KDTreeFlann(robot_pcd)



    # ------------------------------------------------------------------
    #  Visual check: original scene (green) + robot samples (red)
    # ------------------------------------------------------------------
    scene_vis  = o3d.geometry.PointCloud(pcd)
    scene_vis.paint_uniform_color([0, 1, 0])      # green
    robot_vis  = o3d.geometry.PointCloud(robot_pcd)
    robot_vis.paint_uniform_color([1, 0, 0])      # red

    print("Previewing alignment (green = scene, red = robot mesh)…")
    # o3d.visualization.draw_geometries([scene_vis, robot_vis])

    # ------------------------------------------------------------------
    #  Filter scene points close to robot mesh
    # ------------------------------------------------------------------
    robot_pcd = robot_pcd.voxel_down_sample(voxel_size=0.01)  # downsample for efficiency
    pcd = pcd.voxel_down_sample(voxel_size=0.01)  # downsample for efficiency

    FILTER_THRESH = 0.03
    # Convert robot points to numpy array for fast distance computation
    robot_points_np = np.asarray(robot_pcd.points)
    scene_points_np = np.asarray(pcd.points)
    print(scene_points_np.shape, robot_points_np.shape)

    

    starting_time = time.time()

    # Use scipy.spatial.cKDTree for efficient vectorized distance queries
    robot_tree = cKDTree(robot_points_np)
    # Query for the minimum distance from each scene point to the robot mesh points
    dists, _ = robot_tree.query(scene_points_np)
    kept_mask = dists > FILTER_THRESH
    kept = scene_points_np[kept_mask]

    print(f"Filtering took {time.time() - starting_time:.2f} seconds")

    filtered_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray(kept))
    )
    filtered_pcd.paint_uniform_color([0, 0, 1])   # blue

    print(f"Filtered point cloud has {len(filtered_pcd.points):,} points")

    # ------------------------------------------------------------------
    #  Optional: alpha-shape surface for debug
    # ------------------------------------------------------------------
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        filtered_pcd, alpha=0.02
    )
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("filtered_mesh.ply", mesh)
    o3d.io.write_point_cloud("filtered_point_cloud.ply", filtered_pcd)

    print(f"Filtered mesh  : {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris")

    # ------------------------------------------------------------------
    #  Visualise BEFORE vs AFTER
    # ------------------------------------------------------------------
    print("⬛  Showing green (orig) + blue (kept) overlay …")
    o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([pcd, filtered_pcd])
