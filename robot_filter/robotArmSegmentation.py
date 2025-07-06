# now we start coding the function

import time
import os
os.environ["MUJOCO_GL"] = "osmesa"
# from robosuite.controllers import load_controller_config
import numpy as np
import open3d as o3d
from urdfpy import URDF
import matplotlib.pyplot as plt
import tempfile
import imageio.v2 as imageio
import json
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

# take the camera data out of the function
# make this into a class that can be reused, so the camera data is stored in the class
# and the function can be called with different rgb, depth, and joints
# class RobotArmSegmentation:
class RobotArmSegmentation:
    def __init__(self, camera_json_path):
        with open(camera_json_path, "r") as f:
            self.camera_json = json.load(f)
        self.robot_urdf = None
        self.robot_urdf = None
        self.T_world_urdf = None  # Will be set later
        self.camera_name = None  # Will be set later
            
    def set_camera(self, camera_name):  
        if camera_name not in self.camera_json:
            raise ValueError(f"Camera '{camera_name}' not found in camera metadata.")
        self.camera_name = camera_name
        
    def get_intrinsics_extrinsics(self):
        if not hasattr(self, 'camera_name'):
            raise ValueError("Camera name not set. Use set_camera() to set it.")
        intrinsics = np.array(self.camera_json[self.camera_name]["intrinsic"])
        extrinsics = np.array(self.camera_json[self.camera_name]["extrinsic"])
        extrinsics = np.linalg.inv(extrinsics)

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        width = 128
        height = 128

        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(width, height, fx, fy, cx, cy)
        return intrinsics, extrinsics
    
    #load urdf as part of the class as well
    def load_urdf(self, urdf_path):
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
            self.robot_urdf = URDF.load(f.name)

        # Assuming the robot is at the origin, no transformation needed
        # If you have a specific pose, set T_world_urdf accordingly

        base_pose = np.array([-0.56, 0., 0.912])
        base_quat = np.array([1., 0., 0., 0.])  # Assuming no rotation, identity quaternion

        self.T_world_urdf = np.eye(4)  # Identity matrix for no transformation
        self.T_world_urdf[:3, :3] = R.from_quat(base_quat, scalar_first=True).as_matrix()  # rotation part
        self.T_world_urdf[:3, 3] = base_pose[:3]  # translation part

        return self.robot_urdf

    # Robot arm segmentation function
    def segment(self, rgb, depth, joints):
        intrinsics, extrinsics = self.get_intrinsics_extrinsics()

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(rgb.astype(np.uint8)),
            depth=o3d.geometry.Image((depth * 1000).astype(np.uint16)),  # mm for Open3D
            depth_scale=1000.0,
            convert_rgb_to_intensity=False
        )

        tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01,  # 1 cm voxel size
            sdf_trunc=0.04,     # 4 cm truncation distance
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        tsdf_volume.integrate(
            rgbd,
            intrinsic=intrinsics,
            extrinsic=extrinsics  # camera at origin
        )

        pcd = tsdf_volume.extract_point_cloud()

        # Load URDF and create a mesh for the robot arm
        if not hasattr(self, 'robot_urdf'):
            raise ValueError("URDF not loaded. Use load_urdf() to load it.")

        # Get joint angles from simulation
        joint_names = [j.name for j in self.robot_urdf.actuated_joints]
        joint_positions = joints  # robosuite joint angles

        # Map names to values
        joint_angles = dict(zip(joint_names, joint_positions))

        # Compute transformed robot meshes
        robot_mesh_dict = self.robot_urdf.visual_trimesh_fk(cfg=joint_angles)  # returns mesh, transformation matrix relative to base

        # Sample points from each mesh surface
        sampled_points = []
        for mesh, pose in robot_mesh_dict.items():
            transformed = mesh.copy()
            transformed.apply_transform(pose)
            transformed.apply_transform(self.T_world_urdf)  # transform to world frame
            sampled_points.append(transformed.sample(5000))

        robot_points = np.vstack(sampled_points)

        # Create Open3D point cloud for KDTree
        robot_pcd = o3d.geometry.PointCloud()
        robot_pcd.points = o3d.utility.Vector3dVector(robot_points)

        # Visual check: original scene (green) + robot samples (red)
        scene_vis = o3d.geometry.PointCloud(pcd)
        scene_vis.paint_uniform_color([0, 1, 0])  # green
        robot_vis = o3d.geometry.PointCloud(robot_pcd)
        robot_vis.paint_uniform_color([1, 0, 0])  # red

        print("Previewing alignment (green = scene, red = robot mesh)…")
        o3d.visualization.draw_geometries([scene_vis, robot_vis])

        # Filter scene points close to robot mesh
        robot_pcd = robot_pcd.voxel_down_sample(voxel_size=0.01)  # downsample for efficiency
        pcd = pcd.voxel_down_sample(voxel_size=0.01)  # downsample for efficiency

        FILTER_THRESH = 0.03
        robot_points_np = np.asarray(robot_pcd.points)  # Convert to numpy array for fast distance computation          
        scene_points_np = np.asarray(pcd.points)
        
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
        filtered_pcd.paint_uniform_color([0, 0, 1])
        print(f"Filtered point cloud has {len(filtered_pcd.points):,} points")
        
        # Optional: alpha-shape surface for debug
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            filtered_pcd, alpha=0.02
        )
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh("filtered_mesh.ply", mesh)
        o3d.io.write_point_cloud("filtered_point_cloud.ply", filtered_pcd)
        
        print(f"Filtered mesh  : {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris")

        # Visualise BEFORE vs AFTER
        print("⬛  Showing green (orig) + blue (kept) overlay …")
        o3d.visualization.draw_geometries([filtered_pcd])

        return filtered_pcd, pcd  # Return the filtered point cloud and original scene point cloud




# def robot_arm_segmentation(rgb, depth, joints):

#     with open("example_data/robomimic/camera_meta.json", "r") as f:
#         camera_json = json.load(f)

#     camera_name = "frontview"

#     #get the intrinsics and extrinsics from the camera
#     # Get intrinsic and extrinsic
#     intrinsics = np.array(camera_json[camera_name]["intrinsic"])
#     extrinsics = np.array(camera_json[camera_name]["extrinsic"])


#     # grab the extrinsic of the camera so we can align the coordinate
#     extrinsics = np.linalg.inv(extrinsics) # world to pixel space

#     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#         color=o3d.geometry.Image(rgb.astype(np.uint8)),
#         depth=o3d.geometry.Image((depth * 1000).astype(np.uint16)),  # mm for Open3D
#         depth_scale=1000.0,
#         convert_rgb_to_intensity=False
#     )
#     # Integrate the RGBD image into the TSDF volume
#     # === Initialize TSDF volume ===
#     tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
#         voxel_length=0.01,  # 1 cm voxel size
#         sdf_trunc=0.04,     # 4 cm truncation distance
#         color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
#     )

#     tsdf_volume.integrate(
#         rgbd,
#         intrinsic=intrinsics,
#         # Assuming the camera is at the origin for this example
#         extrinsic=extrinsics  # camera at originf
#     )

#     #visualize this rgbd
#     # rgbd_vis = o3d.geometry.RGBDImage.create_from_color_and_depth(
#     #     color=o3d.geometry.Image(rgb.astype(np.uint8)),
#     #     depth=o3d.geometry.Image((depth * 1000).astype(np.uint16)),  # mm for Open3D
#     #     depth_scale=1000.0,
#     #     convert_rgb_to_intensity=False
#     # )
#     # o3d.visualization.draw_geometries([rgbd_vis.color, rgbd_vis.depth])


#     # # === STEP 4: Create point cloud for visualization/debugging ===
#     pcd = tsdf_volume.extract_point_cloud()
#     # o3d.visualization.draw_geometries([pcd])

#     # === STEP 5: Load URDF and create a mesh for the robot arm ===
#     urdf_path = "/home/mingxi/Desktop/rosella_work_folder/panda_description/urdf/panda_arm_hand.urdf"

#     with open(urdf_path, 'r') as f:
#         urdf_content = f.read()

#     absolute_base_path = "/home/mingxi/Desktop/rosella_work_folder/panda_description/"
#     # urdf_str = urdf_content.replace("package://panda_description/", absolute_base_path)
#     urdf_str = (
#         urdf_content
#             .replace("package://panda_description", absolute_base_path)
#             .replace("panda_description/meshes", f"{absolute_base_path}/meshes")
#     )


#     with tempfile.NamedTemporaryFile(mode="w+", suffix=".urdf", delete=False) as f:
#         f.write(urdf_str)
#         f.flush()
#         robot_urdf = URDF.load(f.name)


#     # -------- Get world frame pose of the robot state ---------
#     # base_body = "robot0_link0" # hmmmm
#     # base_body = "robot0_base" # hmmmm
#     # base_pose = env.sim.data.get_body_xpos(base_body)  # (x, y, z, qx, qy, qz, qw)
#     # print("robot0_base height z =", base_pose[2])
#     # base_quat = env.sim.data.get_body_xquat(base_body)  # (qx, qy, qz, qw)

#     # build a 4*4 homogeneous transformation matrix

#     T_world_urdf = np.eye(4)
#     # T_world_urdf[:3, :3] = R.from_quat(base_quat, scalar_first=True).as_matrix()  # rotation part
#     # T_world_urdf[:3, 3] = base_pose[:3]  # translation part
#     # ------------------------------------------------------------------

#     # Get joint angles from simulation
#     joint_names = [j.name for j in robot_urdf.actuated_joints]
#     joint_positions = joints  # robosuite joint angles

#     # Map names to values
#     joint_angles = dict(zip(joint_names, joint_positions))

#     # Compute transformed robot meshes
#     robot_mesh_dict = robot_urdf.visual_trimesh_fk(cfg=joint_angles) # returns mesh, transformation matrix relative to base


#     # Sample points from each mesh surface
#     sampled_points = []
#     for mesh, pose in robot_mesh_dict.items():
#         transformed = mesh.copy()
#         transformed.apply_transform(pose)
#         transformed.apply_transform(T_world_urdf)  # transform to world frame
#         sampled_points.append(transformed.sample(5000))

#     robot_points = np.vstack(sampled_points)
#     print("PASS CHECK FOR MESHES!")

#     # Create Open3D point cloud for KDTree
#     robot_pcd = o3d.geometry.PointCloud()
#     robot_pcd.points = o3d.utility.Vector3dVector(robot_points)

#     # ------------------------------------------------------------------
#     #  Visual check: original scene (green) + robot samples (red)
#     # ------------------------------------------------------------------
#     scene_vis  = o3d.geometry.PointCloud(pcd)
#     scene_vis.paint_uniform_color([0, 1, 0])      # green
#     robot_vis  = o3d.geometry.PointCloud(robot_pcd)
#     robot_vis.paint_uniform_color([1, 0, 0])      # red

#     print("Previewing alignment (green = scene, red = robot mesh)…")
#     # o3d.visualization.draw_geometries([scene_vis, robot_vis])

#     # ------------------------------------------------------------------
#     #  Filter scene points close to robot mesh
#     # ------------------------------------------------------------------
#     robot_pcd = robot_pcd.voxel_down_sample(voxel_size=0.01)  # downsample for efficiency
#     pcd = pcd.voxel_down_sample(voxel_size=0.01)  # downsample for efficiency

#     FILTER_THRESH = 0.03
#     # Convert robot points to numpy array for fast distance computation
#     robot_points_np = np.asarray(robot_pcd.points)
#     scene_points_np = np.asarray(pcd.points)
#     print(scene_points_np.shape, robot_points_np.shape)

#     starting_time = time.time()

#     # Use scipy.spatial.cKDTree for efficient vectorized distance queries
#     robot_tree = cKDTree(robot_points_np)
#     # Query for the minimum distance from each scene point to the robot mesh points
#     dists, _ = robot_tree.query(scene_points_np)
#     kept_mask = dists > FILTER_THRESH
#     kept = scene_points_np[kept_mask]

#     print(f"Filtering took {time.time() - starting_time:.2f} seconds")

#     filtered_pcd = o3d.geometry.PointCloud(
#         o3d.utility.Vector3dVector(np.asarray(kept))
#     )
#     filtered_pcd.paint_uniform_color([0, 0, 1])   # blue

#     print(f"Filtered point cloud has {len(filtered_pcd.points):,} points")

#     # ------------------------------------------------------------------
#     #  Optional: alpha-shape surface for debug
#     # ------------------------------------------------------------------
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
#         filtered_pcd, alpha=0.02
#     )
#     mesh.compute_vertex_normals()
#     o3d.io.write_triangle_mesh("filtered_mesh.ply", mesh)
#     o3d.io.write_point_cloud("filtered_point_cloud.ply", filtered_pcd)

#     print(f"Filtered mesh  : {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris")

#     # ------------------------------------------------------------------
#     #  Visualise BEFORE vs AFTER
#     # ------------------------------------------------------------------
#     print("⬛  Showing green (orig) + blue (kept) overlay …")
#     o3d.visualization.draw_geometries([filtered_pcd])
