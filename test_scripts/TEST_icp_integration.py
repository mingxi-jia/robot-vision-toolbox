"""
Test script for ICP optimization integration

This script tests the optimized ICP implementation against the original
to ensure correctness and measure performance improvement.

Test data location: ../../../Downloads/raw

Usage:
    python test_scripts/TEST_icp_integration.py --data_path <path_to_test_data>
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import time
import argparse
from pathlib import Path

# Import both implementations
from hamer_detector.icp_conversion import (
    extract_hand_point_cloud as extract_hand_point_cloud_original,
    compute_aligned_hamer_translation as compute_aligned_hamer_translation_original
)

from hamer_detector.icp_conversion_optimized import (
    extract_hand_point_cloud_vectorized as extract_hand_point_cloud_optimized,
    compute_aligned_hamer_translation_optimized as compute_aligned_hamer_translation_optimized
)


def load_test_data(data_path):
    """Load test data from the specified path"""
    data_path = Path(data_path)

    # Find mask, depth, and check for hamer vertices
    mask_files = list(data_path.glob("*handmask.png"))
    depth_files = list(data_path.glob("*.npy"))

    if not mask_files or not depth_files:
        print(f"❌ No test data found in {data_path}")
        print(f"   Mask files: {len(mask_files)}")
        print(f"   Depth files: {len(depth_files)}")
        return None

    # Use first available files
    mask_path = mask_files[0]
    depth_path = depth_files[0]

    print(f"📂 Loading test data:")
    print(f"   Mask: {mask_path.name}")
    print(f"   Depth: {depth_path.name}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    depth = np.load(str(depth_path))

    # Convert depth to meters if needed
    if depth.max() > 10:  # Likely in mm
        depth = depth / 1000.0

    return mask, depth


def test_extract_hand_point_cloud(mask, depth, intrinsics, num_trials=10):
    """Test point cloud extraction performance"""

    print("\n" + "="*70)
    print("TEST 1: extract_hand_point_cloud")
    print("="*70)

    # Warm up
    _ = extract_hand_point_cloud_original(mask, depth, intrinsics)
    _ = extract_hand_point_cloud_optimized(mask, depth, intrinsics)

    # Benchmark original
    times_original = []
    for _ in range(num_trials):
        start = time.perf_counter()
        points_original = extract_hand_point_cloud_original(mask, depth, intrinsics)
        elapsed = time.perf_counter() - start
        times_original.append(elapsed)

    # Benchmark optimized
    times_optimized = []
    for _ in range(num_trials):
        start = time.perf_counter()
        points_optimized = extract_hand_point_cloud_optimized(mask, depth, intrinsics)
        elapsed = time.perf_counter() - start
        times_optimized.append(elapsed)

    # Calculate statistics
    avg_original = np.mean(times_original)
    avg_optimized = np.mean(times_optimized)
    speedup = avg_original / avg_optimized

    # Check numerical accuracy
    if points_original.shape == points_optimized.shape:
        # Sort points for comparison (order may differ)
        points_original_sorted = np.sort(points_original.flatten())
        points_optimized_sorted = np.sort(points_optimized.flatten())

        max_diff = np.max(np.abs(points_original_sorted - points_optimized_sorted))
        is_accurate = np.allclose(points_original_sorted, points_optimized_sorted, rtol=1e-5)
    else:
        is_accurate = False
        max_diff = float('inf')

    # Print results
    print(f"\n📊 Performance Results ({num_trials} trials):")
    print(f"   Original implementation:  {avg_original*1000:.3f}ms ± {np.std(times_original)*1000:.3f}ms")
    print(f"   Optimized implementation: {avg_optimized*1000:.3f}ms ± {np.std(times_optimized)*1000:.3f}ms")
    print(f"   Speedup: {speedup:.1f}x faster")

    print(f"\n🔍 Accuracy Check:")
    print(f"   Point cloud shape: {points_original.shape} vs {points_optimized.shape}")
    print(f"   Max difference: {max_diff:.2e}")
    print(f"   Numerically accurate: {'✅ PASS' if is_accurate else '❌ FAIL'}")

    return is_accurate and speedup > 10, speedup


def test_compute_aligned_translation(mask, depth, intrinsics, num_trials=5):
    """Test full ICP alignment performance"""

    print("\n" + "="*70)
    print("TEST 2: compute_aligned_hamer_translation")
    print("="*70)

    # Create synthetic HaMeR vertices for testing
    num_verts = 778
    hamer_vertices = np.random.randn(num_verts, 3) * 0.05
    hamer_vertices[:, 2] += 0.5  # Set to ~0.5m depth

    # Extract point cloud for ICP
    hand_pcd = extract_hand_point_cloud_optimized(mask, depth, intrinsics)

    if hand_pcd.shape[0] < 50:
        print("⚠️  Warning: Very few points in hand point cloud, test may not be representative")
        return False, 1.0

    print(f"\n📊 Test setup:")
    print(f"   HaMeR vertices: {hamer_vertices.shape[0]}")
    print(f"   Hand point cloud: {hand_pcd.shape[0]} points")

    # Warm up
    _ = compute_aligned_hamer_translation_original(hamer_vertices.copy(), hand_pcd, mask, intrinsics)
    _ = compute_aligned_hamer_translation_optimized(hamer_vertices.copy(), hand_pcd, mask, intrinsics)

    # Benchmark original
    times_original = []
    for _ in range(num_trials):
        start = time.perf_counter()
        aligned_original = compute_aligned_hamer_translation_original(
            hamer_vertices.copy(), hand_pcd, mask, intrinsics
        )
        elapsed = time.perf_counter() - start
        times_original.append(elapsed)

    # Benchmark optimized
    times_optimized = []
    for _ in range(num_trials):
        start = time.perf_counter()
        aligned_optimized = compute_aligned_hamer_translation_optimized(
            hamer_vertices.copy(), hand_pcd, mask, intrinsics
        )
        elapsed = time.perf_counter() - start
        times_optimized.append(elapsed)

    # Calculate statistics
    avg_original = np.mean(times_original)
    avg_optimized = np.mean(times_optimized)
    speedup = avg_original / avg_optimized

    # Check numerical accuracy
    if aligned_original is not None and aligned_optimized is not None:
        max_diff = np.max(np.abs(aligned_original - aligned_optimized))
        is_accurate = np.allclose(aligned_original, aligned_optimized, rtol=1e-4, atol=1e-6)
    else:
        is_accurate = (aligned_original is None) == (aligned_optimized is None)
        max_diff = 0.0

    # Print results
    print(f"\n📊 Performance Results ({num_trials} trials):")
    print(f"   Original implementation:  {avg_original*1000:.3f}ms ± {np.std(times_original)*1000:.3f}ms")
    print(f"   Optimized implementation: {avg_optimized*1000:.3f}ms ± {np.std(times_optimized)*1000:.3f}ms")
    print(f"   Speedup: {speedup:.1f}x faster")

    print(f"\n🔍 Accuracy Check:")
    if aligned_original is not None:
        print(f"   Alignment successful: ✅")
        print(f"   Max difference: {max_diff:.2e}")
        print(f"   Numerically accurate: {'✅ PASS' if is_accurate else '❌ FAIL'}")
    else:
        print(f"   Alignment failed (both implementations): {'✅ PASS' if is_accurate else '❌ FAIL'}")

    return is_accurate and speedup > 5, speedup


def main():
    parser = argparse.ArgumentParser(description="Test ICP optimization integration")
    parser.add_argument('--data_path', type=str, default='../../../Downloads/raw',
                       help='Path to test data directory')
    parser.add_argument('--num_trials', type=int, default=10,
                       help='Number of trials for benchmarking')

    args = parser.parse_args()

    print("="*70)
    print("  ICP OPTIMIZATION INTEGRATION TEST")
    print("="*70)

    # Load test data
    test_data = load_test_data(args.data_path)
    if test_data is None:
        print("\n❌ TEST FAILED: Could not load test data")
        print(f"   Please ensure test data exists at: {args.data_path}")
        print(f"   Required files: *handmask.png, *.npy (depth)")
        return 1

    mask, depth = test_data

    # Define camera intrinsics (example values, adjust as needed)
    intrinsics = {
        'fx': 389.0,
        'fy': 389.0,
        'cx': 320.0,
        'cy': 180.0
    }

    print(f"\n📐 Camera intrinsics: fx={intrinsics['fx']}, fy={intrinsics['fy']}")
    print(f"   Image size: {mask.shape}")
    print(f"   Depth range: [{depth[depth>0].min():.3f}, {depth.max():.3f}] meters")

    # Run tests
    test1_pass, speedup1 = test_extract_hand_point_cloud(mask, depth, intrinsics, args.num_trials)
    test2_pass, speedup2 = test_compute_aligned_translation(mask, depth, intrinsics, max(5, args.num_trials // 2))

    # Final summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print(f"\n✅ Test 1 (Point Cloud Extraction): {'PASS' if test1_pass else 'FAIL'} ({speedup1:.1f}x speedup)")
    print(f"✅ Test 2 (Full ICP Alignment):     {'PASS' if test2_pass else 'FAIL'} ({speedup2:.1f}x speedup)")

    all_pass = test1_pass and test2_pass

    print(f"\n{'='*70}")
    if all_pass:
        print("🎉 ALL TESTS PASSED - Safe to integrate into main pipeline")
        print("="*70)
        print("\nNext steps:")
        print("  1. Review docs/03_INTEGRATION_GUIDE.md")
        print("  2. Integrate into real_to_robomimic_converter.py")
        print("  3. Comment out old code (don't delete)")
        print("  4. Delete this test script after integration")
        return 0
    else:
        print("❌ SOME TESTS FAILED - DO NOT integrate yet")
        print("="*70)
        print("\nTroubleshooting:")
        print("  - Check test data quality")
        print("  - Verify camera intrinsics are correct")
        print("  - Review error messages above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
