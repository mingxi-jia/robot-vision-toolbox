import open3d as o3d
import numpy as np
import time

def simple_downsample_for_fixed_scene(pcd: o3d.geometry.PointCloud, 
                                    target_points: int = 4412,
                                    voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """
    Simple two-stage downsampling: voxel downsample first, then precise control
    
    Args:
        pcd: Input point cloud
        target_points: Target number of points (default 4412)
        voxel_size: Voxel size in meters (default 0.01m = 1cm)
    
    Returns:
        Downsampled point cloud
    """
    if len(pcd.points) <= target_points:
        return pcd
    
    # Stage 1: Voxel downsampling for fast point reduction
    voxel_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Stage 2: Precise control to target number of points
    if len(voxel_pcd.points) > target_points:
        # If still too many points, random downsample to target
        sampling_ratio = target_points / len(voxel_pcd.points)
        final_pcd = voxel_pcd.random_down_sample(sampling_ratio=sampling_ratio)
    else:
        # If voxel downsampling already reduced below target, return as is
        final_pcd = voxel_pcd
    
    return final_pcd


def find_optimal_voxel_size(test_pcd: o3d.geometry.PointCloud, 
                          target_points: int = 4412) -> float:
    """
    Find optimal voxel size for your fixed scene
    Run once to find suitable voxel_size, then use it directly afterwards
    """
    print(f"Finding optimal voxel size for {len(test_pcd.points)} -> {target_points} points")
    print("-" * 60)
    
    # Test different voxel sizes
    voxel_sizes = [0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03]
    
    best_voxel_size = 0.01
    best_error = float('inf')
    
    for voxel_size in voxel_sizes:
        start_time = time.time()
        result = simple_downsample_for_fixed_scene(test_pcd, target_points, voxel_size)
        end_time = time.time()
        
        result_points = len(result.points)
        error = abs(result_points - target_points) / target_points
        
        print(f"Voxel size: {voxel_size:.3f}m -> {result_points:4d} points, "
              f"error: {error:.3f}, time: {end_time-start_time:.4f}s")
        
        if error < best_error:
            best_error = error
            best_voxel_size = voxel_size
    
    print(f"\nBest voxel size: {best_voxel_size:.3f}m (error: {best_error:.3f})")
    return best_voxel_size


def replace_fps_with_simple_voxel(combined_pcd: o3d.geometry.PointCloud, 
                                max_point_num: int) -> o3d.geometry.PointCloud:
    """
    Direct replacement for farthest_point_down_sample in your original code
    
    Usage:
    # Original code:
    # combined_pcd = combined_pcd.farthest_point_down_sample(num_samples=min(max_point_num, len(combined_pcd.points)))
    
    # Replace with:
    # combined_pcd = replace_fps_with_simple_voxel(combined_pcd, max_point_num)
    """
    
    # Recommended voxel size based on your scene
    # You can determine this value using find_optimal_voxel_size function
    RECOMMENDED_VOXEL_SIZE = 0.012  # 1.2cm, you can adjust this value
    
    return simple_downsample_for_fixed_scene(combined_pcd, max_point_num, RECOMMENDED_VOXEL_SIZE)


def quick_performance_test():
    """Quick performance test"""
    # Create test point cloud simulating your scene
    def create_test_pcd(n_points=25000):  # Simulate point count after multi-camera fusion
        pcd = o3d.geometry.PointCloud()
        
        # Simulate desktop scene
        points = []
        
        # Table surface
        table_points = np.random.uniform(-0.5, 0.5, (n_points//2, 2))
        table_z = np.full((n_points//2, 1), 0.8)
        table_points = np.hstack([table_points, table_z])
        points.append(table_points)
        
        # Objects on table
        object_points = np.random.uniform(-0.3, 0.3, (n_points//2, 3))
        object_points[:, 2] += 0.6
        points.append(object_points)
        
        all_points = np.vstack(points)
        colors = np.random.rand(len(all_points), 3)
        
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    test_pcd = create_test_pcd()
    target_points = 4412
    
    print("Performance Comparison:")
    print("=" * 50)
    
    # Test original method
    start_time = time.time()
    fps_result = test_pcd.farthest_point_down_sample(num_samples=target_points)
    fps_time = time.time() - start_time
    
    print(f"Original FPS:    {len(fps_result.points):4d} points, {fps_time:.4f}s")
    
    # Test simple voxel method
    start_time = time.time()
    voxel_result = replace_fps_with_simple_voxel(test_pcd, target_points)
    voxel_time = time.time() - start_time
    
    error = abs(len(voxel_result.points) - target_points) / target_points
    speedup = fps_time / voxel_time if voxel_time > 0 else float('inf')
    
    print(f"Simple Voxel:    {len(voxel_result.points):4d} points, {voxel_time:.4f}s, "
          f"error: {error:.3f}, speedup: {speedup:.1f}x")


if __name__ == "__main__":
    print("Simple Voxel Downsampling for Fixed Scene")
    print("=" * 50)
    
    # Run performance test
    quick_performance_test()
    
    print("\n" + "=" * 50)
    print("HOW TO USE:")
    print("=" * 50)
    
    usage_example = '''
# In your original code, find this line:
# combined_pcd = combined_pcd.farthest_point_down_sample(num_samples=min(max_point_num, len(combined_pcd.points)))

# Replace with:
combined_pcd = replace_fps_with_simple_voxel(combined_pcd, max_point_num)

# If you want to adjust voxel size, use directly:
combined_pcd = simple_downsample_for_fixed_scene(combined_pcd, max_point_num, voxel_size=0.015)

# Recommended voxel size settings:
# - 0.008-0.010m (8-10mm): Fine scenes, need to preserve more details
# - 0.012-0.015m (12-15mm): Balanced choice, suitable for most indoor robot scenes  
# - 0.018-0.025m (18-25mm): Coarse downsampling, pursue fastest speed
    '''
    
    print(usage_example)
    
    print("\n" + "=" * 50)
    print("QUICK SETUP GUIDE:")
    print("=" * 50)
    print("1. Use recommended setting (voxel_size=0.012) directly, should work well")
    print("2. If too many points, increase voxel_size to 0.015 or 0.018")  
    print("3. If too few points, decrease voxel_size to 0.008 or 0.010")
    print("4. Once you find the right value, use it fixed without recalculation")