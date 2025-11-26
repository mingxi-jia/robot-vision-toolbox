import torch
import numpy as np
import open3d as o3d
from pytorch3d.transforms import rotation_6d_to_matrix, quaternion_to_matrix

def visualize_pcd(points: np.array, mode='color'):
    assert mode in ['color', 'xyz'], "Mode must be 'color' or 'xyz'"
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    
    if mode == 'color':
        assert points.shape[1] >= 6
        pcd.colors = o3d.utility.Vector3dVector(points[:,3:6])
    
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, origin])

def visualize_pcds(points: list, mode='color'):
    assert mode in ['color', 'xyz'], "Mode must be 'color' or 'xyz'"
    
    pcds = []
    for p in points:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p[:,:3])
    
        if mode == 'color':
            assert p.shape[1] >= 6
            pcd.colors = o3d.utility.Vector3dVector(p[:,3:6])
        
        pcds.append(pcd)
    
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([*pcds, origin])
