import os
import copy
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import open3d as o3d
import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

class PointCloudUtils:
    """
    A comprehensive utility class for point cloud manipulation in robotics research.

    This class provides a set of tools for creating, converting, transforming,
    and processing point clouds. It is designed to be easy to use and versatile
    for various robotics applications, such as perception, planning, and simulation.

    The class can pre-load and generate primitive point cloud models (e.g., a gripper, a sphere)
    for efficient rendering and manipulation.

    Example:
        >>> pcd_utils = PointCloudUtils()
        >>> # Load an RGB-D image and convert it to a point cloud
        >>> rgb_image = np.asarray(Image.open('rgb.png'))
        >>> depth_image = np.load('depth.npy')
        >>> intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        >>> extrinsics = np.eye(4)
        >>> pcd, _ = pcd_utils.rgbd_to_o3d(rgb_image, depth_image, intrinsics, extrinsics)
        >>> # Visualize the point cloud
        >>> pcd_utils.visualize([pcd])
    """

    def __init__(self, gripper_asset_path: Optional[str] = None):
        """
        Initializes the PointCloudUtils class.

        Args:
            gripper_asset_path (Optional[str]): Path to the gripper's .ply file.
                If None, it defaults to './gripper.ply' relative to this script's location.
        """
        self.GRIPPER = self._load_gripper_model(gripper_asset_path)
        self.SPHERE = self._create_colored_sphere()

    def _load_gripper_model(self, gripper_asset_path: Optional[str]) -> np.ndarray:
        """
        Loads the gripper point cloud model.

        Args:
            gripper_asset_path (Optional[str]): The path to the gripper .ply file.

        Returns:
            np.ndarray: The gripper point cloud as an (N, 6) numpy array (xyzrgb).
        """
        if gripper_asset_path is None:
            # Using `pathlib` for robust path handling
            current_file_path = Path(__file__).parent.absolute()
            gripper_asset_path = current_file_path / "gripper.ply"

        try:
            gripper_asset = o3d.io.read_point_cloud(str(gripper_asset_path))
            xyz_numpy = np.asarray(gripper_asset.points, dtype=np.float32)
            color_numpy = np.asarray(gripper_asset.colors, dtype=np.float32)
            return np.concatenate([xyz_numpy, color_numpy], axis=1)
        except Exception as e:
            print(f"Warning: Could not load gripper model from {gripper_asset_path}. Error: {e}")
            return np.empty((0, 6), dtype=np.float32)

    def _create_colored_sphere(self, num_points: int = 1024, radius: float = 0.05) -> np.ndarray:
        """
        Creates a sphere point cloud with colors indicating the octant.

        This uses the Fibonacci lattice method for evenly distributing points on a sphere.

        Args:
            num_points (int): The number of points in the sphere.
            radius (float): The radius of the sphere.

        Returns:
            np.ndarray: The sphere point cloud as an (N, 6) numpy array (xyzrgb).
        """
        indices = np.arange(num_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / num_points)
        theta = np.pi * (1 + 5**0.5) * indices
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        colors = np.zeros((num_points, 3), dtype=np.float32)
        masks = {
            'red': (x >= 0) & (y >= 0) & (z >= 0),
            'green': (x < 0) & (y >= 0) & (z >= 0),
            'blue': (x < 0) & (y < 0) & (z >= 0),
            'yellow': (x >= 0) & (y < 0) & (z >= 0),
            'magenta': (x >= 0) & (y >= 0) & (z < 0),
            'cyan': (x < 0) & (y >= 0) & (z < 0),
            'grey': (x < 0) & (y < 0) & (z < 0),
            'orange': (x >= 0) & (y < 0) & (z < 0),
        }
        color_map = {
            'red': [1.0, 0.0, 0.0],
            'green': [0.0, 1.0, 0.0],
            'blue': [0.0, 0.0, 1.0],
            'yellow': [1.0, 1.0, 0.0],
            'magenta': [1.0, 0.0, 1.0],
            'cyan': [0.0, 1.0, 1.0],
            'grey': [0.5, 0.5, 0.5],
            'orange': [1.0, 0.5, 0.0],
        }

        for color_name, mask in masks.items():
            colors[mask] = color_map[color_name]

        return np.concatenate([np.stack([x, y, z], axis=1), colors], axis=1)

    def numpy_to_open3d(self, pcd_np: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Converts a numpy array to an Open3D PointCloud object.

        Args:
            pcd_np (np.ndarray): The input numpy array. Can be (N, 3) for xyz
                                 or (N, 6) for xyzrgb.

        Returns:
            o3d.geometry.PointCloud: The converted Open3D point cloud.
        """
        if pcd_np.shape[1] not in [3, 6]:
            raise ValueError("Input numpy array must have 3 or 6 columns (xyz or xyzrgb).")

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np[:, :3])

        if pcd_np.shape[1] == 6:
            colors = pcd_np[:, 3:6]
            if np.max(colors) > 1.0 or np.min(colors) < 0.0:
                # Clamping color values to be safe
                colors = np.clip(colors, 0, 1)
            pcd_o3d.colors = o3d.utility.Vector3dVector(colors)

        return pcd_o3d

    def open3d_to_numpy(self, pcd_o3d: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Converts an Open3D PointCloud object to a numpy array.

        Args:
            pcd_o3d (o3d.geometry.PointCloud): The input Open3D point cloud.

        Returns:
            np.ndarray: The converted numpy array (N, 6) for xyzrgb if colors are present,
                        otherwise (N, 3) for xyz.
        """
        xyz = np.asarray(pcd_o3d.points)
        if pcd_o3d.has_colors():
            rgb = np.asarray(pcd_o3d.colors)
            return np.concatenate([xyz, rgb], axis=1)
        return xyz

    def resample_point_cloud(self, pcd_np: np.ndarray, num_points: int) -> np.ndarray:
        """
        Resamples a point cloud to a specific number of points.

        If the point cloud has fewer points, it upsamples by duplicating random points.
        If it has more, it downsamples by random permutation.

        Args:
            pcd_np (np.ndarray): The input point cloud as a numpy array.
            num_points (int): The desired number of points.

        Returns:
            np.ndarray: The resampled point cloud.
        """
        current_num_points = pcd_np.shape[0]
        if current_num_points == 0:
            return np.zeros((num_points, pcd_np.shape[1]), dtype=pcd_np.dtype)
            
        if current_num_points < num_points:
            extra_indices = np.random.choice(current_num_points, num_points - current_num_points, replace=True)
            return np.concatenate([pcd_np, pcd_np[extra_indices]], axis=0)
        
        shuffled_indices = np.random.permutation(current_num_points)[:num_points]
        return pcd_np[shuffled_indices]

    def farthest_point_downsample(self, pcd_o3d: o3d.geometry.PointCloud, num_samples: int) -> o3d.geometry.PointCloud:
        """
        Downsamples a point cloud using the farthest point sampling algorithm.

        Args:
            pcd_o3d (o3d.geometry.PointCloud): The input Open3D point cloud.
            num_samples (int): The number of points to sample.

        Returns:
            o3d.geometry.PointCloud: The downsampled point cloud.
        """
        if len(pcd_o3d.points) <= num_samples:
            return pcd_o3d
        return pcd_o3d.farthest_point_down_sample(num_samples=num_samples)

    def depth_to_point_cloud(self, depth_image: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a depth image to a point cloud in the camera frame.

        Args:
            depth_image (np.ndarray): The (H, W) depth image.
            intrinsics (np.ndarray): The (3, 3) camera intrinsics matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - np.ndarray: The foreground point cloud (N, 3).
                - np.ndarray: The (H, W) boolean mask of the foreground points.
        """
        h, w = depth_image.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        mask = (depth_image > 0.0) & (depth_image < 2.0)  # Filter invalid depth
        
        y_map, x_map = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        x_map = x_map[mask]
        y_map = y_map[mask]

        points = np.zeros((mask.sum(), 3), dtype=np.float32)
        points[:, 0] = (x_map - cx) * depth_image[mask] / fx
        points[:, 1] = (y_map - cy) * depth_image[mask] / fy
        points[:, 2] = depth_image[mask]

        return points, mask

    def rgbd_to_o3d(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
        """
        Converts RGB and depth images to a transformed Open3D point cloud in the world frame.

        Args:
            rgb_image (np.ndarray): The (H, W, 3) RGB image (uint8).
            depth_image (np.ndarray): The (H, W) depth image (float).
            intrinsics (np.ndarray): The (3, 3) camera intrinsics matrix.
            extrinsics (np.ndarray): The (4, 4) camera extrinsics matrix (camera to world).

        Returns:
            Tuple[np.ndarray, o3d.geometry.PointCloud]: A tuple containing:
                - np.ndarray: The transformed point cloud as a numpy array (N, 3).
                - o3d.geometry.PointCloud: The transformed Open3D point cloud.
        """
        if not (rgb_image.shape[:2] == depth_image.shape and rgb_image.ndim == 3 and depth_image.ndim == 2):
            raise ValueError("RGB and Depth image dimensions do not match.")
        if not (intrinsics.shape == (3, 3) and extrinsics.shape == (4, 4)):
            raise ValueError("Invalid intrinsics or extrinsics matrix shape.")

        pcd_local, mask = self.depth_to_point_cloud(depth_image, intrinsics)
        
        # Apply the extrinsics to transform points from camera to world frame
        pcd_world_np = self.apply_transform(pcd_local, extrinsics)

        colors = rgb_image[mask].astype(np.float64) / 255.0
        pcd_world_o3d = self.numpy_to_open3d(np.hstack([pcd_world_np, colors]))

        return pcd_world_np, pcd_world_o3d

    def create_transform_matrix(self, translation: Union[List[float], np.ndarray], quaternion: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Creates a 4x4 SE(3) transformation matrix from translation and quaternion.

        Args:
            translation (Union[List[float], np.ndarray]): A (3,) translation vector [x, y, z].
            quaternion (Union[List[float], np.ndarray]): A (4,) quaternion [qx, qy, qz, qw].

        Returns:
            np.ndarray: The (4, 4) transformation matrix.
        """
        if len(translation) != 3 or len(quaternion) != 4:
            raise ValueError("Translation must be of length 3 and quaternion of length 4.")
        
        transform = np.eye(4)
        transform[:3, :3] = R.from_quat(quaternion).as_matrix()
        transform[:3, 3] = translation
        
        return transform

    def apply_transform(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Applies an SE(3) transformation to a set of points.

        Args:
            points (np.ndarray): An (N, 3) or (N, 6) array of points.
            transform (np.ndarray): A (4, 4) transformation matrix.

        Returns:
            np.ndarray: The transformed points, maintaining original shape.
        """
        if points.shape[0] == 0:
            return points
            
        points_copy = points.copy()
        xyz = points_copy[:, :3]
        
        points_homogeneous = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        transformed_points = (transform @ points_homogeneous.T).T
        
        points_copy[:, :3] = transformed_points[:, :3]
        return points_copy

    def render_model_at_pose(self, pose: np.ndarray, model_type: str = 'gripper', num_points: int = 1024) -> np.ndarray:
        """
        Renders a pre-defined model (e.g., gripper, sphere) at given pose(s).

        Args:
            pose (np.ndarray): A (B, 7) array of poses, where B is the batch size and
                               7 corresponds to (x, y, z, qx, qy, qz, qw).
            model_type (str): The type of model to render ('gripper' or 'sphere').
            num_points (int): The number of points for the rendered model.

        Returns:
            np.ndarray: The rendered point cloud(s) with shape (B, num_points, 6).
        """
        if model_type == 'gripper':
            model_pcd = self.GRIPPER
        elif model_type == 'sphere':
            model_pcd = self.SPHERE
        else:
            raise NotImplementedError(f"Model type '{model_type}' not implemented.")

        original_shape = list(pose.shape[:-1])
        pose = pose.reshape(-1, 7)
        batch_size = pose.shape[0]
        
        pcds = []
        for i in range(batch_size):
            transform_matrix = self.create_transform_matrix(pose[i, :3], pose[i, 3:7])
            transformed_pcd = self.apply_transform(model_pcd, transform_matrix)
            resampled_pcd = self.resample_point_cloud(transformed_pcd, num_points)
            pcds.append(resampled_pcd)

        final_shape = [*original_shape, num_points, model_pcd.shape[1]]
        return np.stack(pcds).reshape(final_shape)

    def filter_by_workspace(self, pcd: o3d.geometry.PointCloud, workspace: List[float]) -> o3d.geometry.PointCloud:
        """
        Filters an Open3D point cloud based on workspace limits (a bounding box).

        Args:
            pcd (o3d.geometry.PointCloud): The input point cloud.
            workspace (List[float]): A list of 6 floats [x_min, x_max, y_min, y_max, z_min, z_max].

        Returns:
            o3d.geometry.PointCloud: The filtered point cloud.
        """
        if len(workspace) != 6:
            raise ValueError("Workspace must be a list of 6 floats: [x_min, x_max, y_min, y_max, z_min, z_max].")
        
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(workspace[0], workspace[2], workspace[4]),
            max_bound=(workspace[1], workspace[3], workspace[5])
        )
        return pcd.crop(bbox)

    def visualize(self, geometries: List, window_name: str = "Open3D Visualization"):
        """
        Visualizes a list of Open3D geometries.

        Args:
            geometries (List): A list of Open3D geometry objects to visualize.
            window_name (str): The title of the visualization window.
        """
        o3d.visualization.draw_geometries(geometries, window_name=window_name)


if __name__ == "__main__":
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Example Usage of the PointCloudUtils Class
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    # 1. Initialize the utility class
    # This will pre-load the gripper model and generate the sphere model.
    # To run this example, you need the 'example_data' directory and a 'gripper.ply' file.
    
    # Create a dummy gripper.ply if it doesn't exist for demonstration
    if not os.path.exists("gripper.ply"):
        print("Creating a dummy 'gripper.ply' for demonstration purposes.")
        dummy_pcd = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.02, depth=0.1)
        dummy_pcd.compute_vertex_normals()
        dummy_pcd = dummy_pcd.sample_points_uniformly(number_of_points=1000)
        dummy_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.io.write_point_cloud("gripper.ply", dummy_pcd)

    pcd_utils = PointCloudUtils()

    # 2. Define data paths and check for existence
    data_path = "example_data/multiview_rgbd"
    if not os.path.exists(data_path):
        print(f"Error: The example data path '{data_path}' does not exist.")
        print("Please ensure the example data is available to run the full demonstration.")
    else:
        cam_views = [1, 2, 3]

        # 3. Load camera information and visualize poses
        with open(os.path.join(data_path, 'camera_info.yaml'), 'r') as f:
            cam_info = yaml.safe_load(f)

        camera_frames = []
        info_dict = {}
        frame_dict = {}

        for i in cam_views:
            cam_name = f'cam{i}'
            intrinsics = np.array(cam_info[cam_name]['k']).reshape(3, 3)
            extrinsics = pcd_utils.create_transform_matrix(cam_info[cam_name]['t'], cam_info[cam_name]['q'])
            info_dict[cam_name] = {'intrinsics': intrinsics, 'extrinsics': extrinsics}

            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            cam_frame.transform(extrinsics)
            camera_frames.append(cam_frame)
            
            frame_list = sorted(
                [f.split('.png')[0] for f in os.listdir(os.path.join(data_path, cam_name, 'rgb')) if f.endswith('.png')],
                key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1]))
            )
            frame_dict[cam_name] = frame_list
        
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        camera_frames.append(world_frame)

        print("Visualizing camera poses and the world frame...")
        pcd_utils.visualize(camera_frames, window_name="Camera Poses")

        # 4. Multi-view reconstruction for the first frame
        print("\nStarting multi-view reconstruction for the first frame...")
        
        merged_pcd = o3d.geometry.PointCloud()

        for i in cam_views:
            cam_name = f'cam{i}'
            frame_name = frame_dict[cam_name][0] # Process only the first frame

            rgb_path = os.path.join(data_path, cam_name, 'rgb', f'{frame_name}.png')
            depth_path = os.path.join(data_path, cam_name, 'depth', f'{frame_name}.npy')

            rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
            depth = np.load(depth_path) / 1000.0  # Convert to meters

            _, pcd_o3d = pcd_utils.rgbd_to_o3d(
                rgb, depth, info_dict[cam_name]['intrinsics'], info_dict[cam_name]['extrinsics']
            )
            merged_pcd += pcd_o3d
        
        print(f"Total points from merging all views: {len(merged_pcd.points)}")

        # 5. Process the merged point cloud
        workspace_limits = [0.2, 0.7, -0.2, 0.3, -0.04, 0.5]
        filtered_pcd = pcd_utils.filter_by_workspace(merged_pcd, workspace_limits)
        print(f"Points after workspace filtering: {len(filtered_pcd.points)}")

        downsampled_pcd = pcd_utils.farthest_point_downsample(filtered_pcd, num_samples=4096)
        print(f"Points after farthest point downsampling: {len(downsampled_pcd.points)}")
        
        # 6. Render a model at a specific pose
        # Note the batch dimension for the pose array: (1, 7)
        example_pose = np.array([[0.4, 0.0, 0.2, 0., 0., 0., 1.]])
        rendered_sphere_np = pcd_utils.render_model_at_pose(example_pose, model_type='sphere', num_points=1024)[0]
        rendered_sphere_o3d = pcd_utils.numpy_to_open3d(rendered_sphere_np)

        # 7. Final Visualization
        print("\nVisualizing final scene: reconstructed point cloud, a rendered sphere, and world frame.")
        pcd_utils.visualize([downsampled_pcd, rendered_sphere_o3d, world_frame], window_name="Final Reconstruction")