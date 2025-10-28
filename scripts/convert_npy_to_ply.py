#!/usr/bin/env python3
"""
Convert .npy point cloud files to .ply format for visualization.
This script is needed because the optimization skipped .ply conversion for speed.
"""

import os
import sys
import numpy as np
import open3d as o3d
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vision_utils.pcd_utils import np2o3d


def convert_npy_to_ply(npy_path, ply_path):
    """Convert a single .npy file to .ply format."""
    arr = np.load(npy_path)
    points, colors = arr[:, :3], arr[:, 3:6]
    pcd_o3d = np2o3d(points, colors)
    o3d.io.write_point_cloud(str(ply_path), pcd_o3d, write_ascii=False, compressed=False)


def convert_episode(pcd_dir):
    """Convert all .npy files in a directory to .ply."""
    npy_files = sorted(Path(pcd_dir).glob("*.npy"))

    if not npy_files:
        print(f"⚠️  No .npy files found in {pcd_dir}")
        return 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for npy_path in npy_files:
            ply_path = npy_path.with_suffix('.ply')
            futures.append(executor.submit(convert_npy_to_ply, npy_path, ply_path))

        # Wait for all conversions
        for future in tqdm(futures, desc=f"Converting {pcd_dir.name}", leave=False):
            future.result()

    return len(npy_files)


def main():
    import argparse
    import subprocess
    parser = argparse.ArgumentParser(description="Convert .npy point clouds to .ply format")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to output directory (e.g., /path/to/output)")
    parser.add_argument("--episode", type=str, default=None,
                       help="Specific episode name to convert (default: all episodes)")
    parser.add_argument("--variant", type=str, choices=['pcd', 'pcd_no_hand', 'both'],
                       default='both', help="Which variant to convert")
    parser.add_argument("--render_video", action='store_true',
                       help="Render videos using mesh-sequence-player after conversion")

    args = parser.parse_args()

    output_path = Path(args.output_path)
    if not output_path.exists():
        print(f"❌ Output path does not exist: {output_path}")
        sys.exit(1)

    # Get list of episodes to process
    if args.episode:
        episodes = [output_path / args.episode]
    else:
        episodes = sorted([d for d in output_path.iterdir()
                          if d.is_dir() and d.name.startswith('episode_')])

    if not episodes:
        print(f"❌ No episodes found in {output_path}")
        sys.exit(1)

    print(f"Found {len(episodes)} episode(s) to process")

    total_converted = 0
    videos_to_render = []  # Track directories for video rendering

    for episode_dir in episodes:
        variants_to_process = []

        if args.variant in ['pcd', 'both']:
            pcd_dir = episode_dir / 'pcd'
            if pcd_dir.exists():
                variants_to_process.append(pcd_dir)

        if args.variant in ['pcd_no_hand', 'both']:
            pcd_no_hand_dir = episode_dir / 'pcd_no_hand'
            if pcd_no_hand_dir.exists():
                variants_to_process.append(pcd_no_hand_dir)

        if not variants_to_process:
            print(f"⚠️  Skipping {episode_dir.name} - no pcd directories found")
            continue

        print(f"\n📁 Processing {episode_dir.name}")
        for pcd_dir in variants_to_process:
            count = convert_episode(pcd_dir)
            total_converted += count
            print(f"   ✅ Converted {count} files in {pcd_dir.name}/")

            # Track for video rendering
            if args.render_video:
                videos_to_render.append(pcd_dir)

    print(f"\n✅ Total: Converted {total_converted} .npy files to .ply")

    # Render videos if requested
    if args.render_video and videos_to_render:
        print(f"\n🎬 Rendering videos with mesh-sequence-player...")
        for pcd_dir in videos_to_render:
            variant_name = pcd_dir.name  # 'pcd' or 'pcd_no_hand'
            episode_name = pcd_dir.parent.name
            output_video = pcd_dir.parent / f"{episode_name}_{variant_name}.mp4"

            print(f"\n   📹 Rendering {episode_name}/{variant_name}...")
            cmd = ['mesh-sequence-player', str(pcd_dir), '-p', '--output', str(output_video)]

            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"   ✅ Video saved: {output_video}")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to render video: {e}")
                if e.stdout:
                    print(f"      stdout: {e.stdout}")
                if e.stderr:
                    print(f"      stderr: {e.stderr}")
            except FileNotFoundError:
                print(f"   ❌ mesh-sequence-player not found. Please install it first.")
                break

        print(f"\n✅ Video rendering complete")


if __name__ == "__main__":
    main()
