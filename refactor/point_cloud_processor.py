"""Point cloud processing utilities."""

import numpy as np
import open3d as o3d
from vision_utils.pcd_utils import o3d2np, pcd_to_voxel, render_pcd_from_pose


class PointCloudProcessor:
    """Processes point clouds for dataset conversion."""

    def __init__(self, workspace: np.ndarray, fix_point_num: int, robomimic_center: np.ndarray):
        """Initialize processor.

        Args:
            workspace: 3x2 array defining workspace boundaries
            fix_point_num: Target number of points after processing
            robomimic_center: Center point for robomimic coordinate system
        """
        self.workspace = workspace
        self.fix_point_num = fix_point_num
        self.robomimic_center = robomimic_center

    def process_raw_pcd(self, pcd: np.ndarray) -> tuple[np.ndarray, o3d.geometry.PointCloud]:
        """Process raw point cloud.

        Args:
            pcd: Raw point cloud array (N, 6) with xyz and rgb

        Returns:
            Tuple of (processed numpy array, processed o3d point cloud)
        """
        # Filter by workspace
        pcd_np = pcd[np.where(
            (pcd[:, 0] > self.workspace[0, 0]) & (pcd[:, 0] < self.workspace[0, 1]) &
            (pcd[:, 1] > self.workspace[1, 0]) & (pcd[:, 1] < self.workspace[1, 1]) &
            (pcd[:, 2] > self.workspace[2, 0]) & (pcd[:, 2] < self.workspace[2, 1])
        )]

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
        pcd_o3d.colors = o3d.utility.Vector3dVector(pcd_np[:, 3:])

        point_num = pcd_np.shape[0]
        assert point_num > 1024, "Too few points in the point cloud after filtering."

        if pcd_np.shape[0] >= self.fix_point_num:
            # Farthest point down sample
            pcd_o3d = pcd_o3d.farthest_point_down_sample(self.fix_point_num)
        else:
            # Upsample by random selection
            extra_choice = np.random.choice(point_num, self.fix_point_num - pcd.shape[0], replace=True)
            pcd = np.concatenate([pcd, pcd[extra_choice]], axis=0)

        pcd_np = o3d2np(pcd_o3d)
        return pcd_np, pcd_o3d

    def get_render_pcd(self, pcd_no_robot: np.ndarray, ee_pos: np.ndarray) -> np.ndarray:
        """Get voxelized rendered point cloud with sphere.

        Args:
            pcd_no_robot: Point cloud without robot
            ee_pos: End-effector position

        Returns:
            Voxelized point cloud
        """
        geco = render_pcd_from_pose(ee_pos, 1024, 'sphere')
        pcd_render = np.concatenate([pcd_no_robot, geco], axis=0)
        np_voxels_render = pcd_to_voxel(pcd_render[None, ...])[0]
        return np_voxels_render
