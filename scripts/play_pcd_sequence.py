#!/usr/bin/env python3
"""
Play point cloud sequence with Open3D visualization.
Supports both .ply and .npy formats.
"""

import os
import sys
import numpy as np
import open3d as o3d
from pathlib import Path
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vision_utils.pcd_utils import np2o3d


def load_pcd(path):
    """Load point cloud from .ply or .npy file."""
    if path.suffix == '.ply':
        return o3d.io.read_point_cloud(str(path))
    elif path.suffix == '.npy':
        arr = np.load(path, allow_pickle=True)
        assert arr.shape[0] > 0, f"Empty point cloud in file: {path}"
        points, colors = arr[:, :3], arr[:, 3:6]
        return np2o3d(points, colors)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Play point cloud sequence")
    parser.add_argument("pcd_dir", type=str, help="Directory containing .ply or .npy files")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second (default: 10)")
    parser.add_argument("--loop", action="store_true", help="Loop the sequence")
    parser.add_argument("--format", type=str, choices=['ply', 'npy', 'auto'],
                       default='auto', help="File format (default: auto)")

    args = parser.parse_args()

    pcd_dir = Path(args.pcd_dir)
    if not pcd_dir.exists():
        print(f"❌ Directory not found: {pcd_dir}")
        sys.exit(1)

    # Find files
    if args.format == 'auto':
        ply_files = sorted(pcd_dir.glob("*.ply"))
        npy_files = sorted(pcd_dir.glob("*.npy"))
        if ply_files:
            files = ply_files
        elif npy_files:
            files = npy_files
        else:
            print(f"❌ No .ply or .npy files found in {pcd_dir}")
            sys.exit(1)
    else:
        files = sorted(pcd_dir.glob(f"*.{args.format}"))
        if not files:
            print(f"❌ No .{args.format} files found in {pcd_dir}")
            sys.exit(1)

    print(f"📁 Found {len(files)} files in {pcd_dir}")
    print(f"🎬 Playing at {args.fps} FPS")
    print(f"{'🔁 Looping enabled' if args.loop else '▶️  Playing once'}")
    print("\nControls:")
    print("  Space: Pause/Resume")
    print("  Q/Esc: Quit")
    print("  Left/Right Arrow: Previous/Next frame (when paused)")
    print("  Mouse: Rotate (left), Pan (middle), Zoom (scroll)")
    print()

    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Sequence Player", width=1280, height=720)

    # Load first frame
    current_idx = 0
    pcd = load_pcd(files[current_idx])
    vis.add_geometry(pcd)

    # Set initial view
    view_ctrl = vis.get_view_control()
    view_ctrl.set_zoom(0.5)

    # Playback state
    paused = False
    last_frame_time = time.time()
    frame_duration = 1.0 / args.fps

    def update_frame(idx):
        nonlocal pcd
        new_pcd = load_pcd(files[idx])
        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        print(f"\rFrame: {idx+1}/{len(files)} ({files[idx].name})", end='', flush=True)

    print(f"Frame: {current_idx+1}/{len(files)} ({files[current_idx].name})", end='', flush=True)

    # Main loop
    try:
        while True:
            if not vis.poll_events():
                break

            current_time = time.time()

            # Auto-advance if not paused and enough time has elapsed
            if not paused and (current_time - last_frame_time >= frame_duration):
                current_idx += 1
                if current_idx >= len(files):
                    if args.loop:
                        current_idx = 0
                    else:
                        print("\n✅ Sequence complete")
                        break

                update_frame(current_idx)
                last_frame_time = current_time

            vis.update_renderer()

            # Small sleep to prevent CPU spinning
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    main()
