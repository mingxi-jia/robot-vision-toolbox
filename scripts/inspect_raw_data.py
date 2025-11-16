#!/usr/bin/env python3
"""
Interactive viewer for raw dataset RGB images from multiple cameras and episodes.
Navigate through episodes and frames using arrow keys.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def load_all_episodes_images(episodes_path, num_cams=3):
    """
    Load all RGB images from all episodes.

    Returns:
        episodes_data: list of dicts with {
            'name': episode name,
            'frames': {frame_idx: {cam_idx: image_path}},
            'grasp_actions': numpy array of gripper values
        }
    """
    episodes_path = Path(episodes_path)

    # Find all episode directories
    episodes = sorted([d for d in episodes_path.iterdir()
                      if d.is_dir() and d.name.startswith("episode_")])

    if not episodes:
        print(f"No episodes found in {episodes_path}")
        return []

    print(f"Found {len(episodes)} episodes")

    episodes_data = []

    for episode_dir in episodes:
        print(f"Loading {episode_dir.name}...", end=" ")

        # Collect all images from all cameras
        cam_images = {}

        for cam_idx in range(1, num_cams + 1):
            rgb_dir = episode_dir / f"cam{cam_idx}" / "rgb"
            if not rgb_dir.exists():
                print(f"\n  Warning: {rgb_dir} does not exist")
                continue

            images = sorted(rgb_dir.glob("*.png"))
            cam_images[cam_idx] = images

        # Verify all cameras have the same number of images
        num_frames_list = [len(imgs) for imgs in cam_images.values()]
        if not num_frames_list:
            print("No images found")
            continue

        if len(set(num_frames_list)) > 1:
            print(f"\n  Warning: Cameras have different number of frames: {num_frames_list}")
            min_frames = min(num_frames_list)
            print(f"  Using minimum: {min_frames} frames")
        else:
            min_frames = num_frames_list[0]

        # Organize by frame index
        frame_data = {}
        for frame_idx in range(min_frames):
            frame_data[frame_idx] = {
                cam_idx: cam_images[cam_idx][frame_idx]
                for cam_idx in cam_images.keys()
            }

        # Load gripper actions
        grasp_file = episode_dir / "state" / "grasp.npy"
        grasp_actions = None
        if grasp_file.exists():
            try:
                grasp_actions = np.load(grasp_file)
                if len(grasp_actions) != min_frames:
                    print(f"\n  Warning: Grasp actions length ({len(grasp_actions)}) != frames ({min_frames})")
                    # Pad or trim to match frames
                    if len(grasp_actions) < min_frames:
                        grasp_actions = np.pad(grasp_actions, (0, min_frames - len(grasp_actions)), mode='edge')
                    else:
                        grasp_actions = grasp_actions[:min_frames]
            except Exception as e:
                print(f"\n  Warning: Could not load grasp actions: {e}")
        else:
            print(f"\n  Warning: {grasp_file} does not exist")

        episodes_data.append({
            'name': episode_dir.name,
            'path': episode_dir,
            'frames': frame_data,
            'num_frames': min_frames,
            'grasp_actions': grasp_actions
        })

        print(f"{min_frames} frames")

    return episodes_data


def create_gripper_bar(width, gripper_value, bar_height=40):
    """
    Create a gripper action visualization bar.

    Args:
        width: width of the bar
        gripper_value: value between 0 (open) and 1 (closed)
        bar_height: height of the bar

    Returns:
        numpy array: bar visualization
    """
    bar = np.zeros((bar_height, width, 3), dtype=np.uint8)

    # Background (dark gray)
    bar[:, :] = (40, 40, 40)

    # Border
    cv2.rectangle(bar, (0, 0), (width - 1, bar_height - 1), (100, 100, 100), 2)

    # Filled portion (green for closed gripper)
    if gripper_value > 0:
        fill_width = int(width * gripper_value)
        bar[:, :fill_width] = (0, int(255 * gripper_value), 0)

    # Add text label
    label = f"Gripper: {gripper_value:.2f}"
    if gripper_value >= 0.5:
        label += " (CLOSED)"
        text_color = (255, 255, 255)
    else:
        label += " (OPEN)"
        text_color = (200, 200, 200)

    cv2.putText(bar, label, (10, bar_height - 12),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    return bar


def create_episode_frame_visualization(episode_data, frame_idx, max_height=600):
    """
    Create visualization for a single frame from one episode.

    Args:
        episode_data: episode data dict
        frame_idx: which frame to visualize
        max_height: maximum height for each camera image

    Returns:
        numpy array: concatenated image with all cameras and gripper bar
    """
    if frame_idx >= episode_data['num_frames']:
        return None

    frame_data = episode_data['frames'][frame_idx]
    images = []

    for cam_idx in sorted(frame_data.keys()):
        img = cv2.imread(str(frame_data[cam_idx]))

        if img is None:
            print(f"Warning: Could not load {frame_data[cam_idx]}")
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, f"Cam {cam_idx} - Not Found", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Resize to max_height while maintaining aspect ratio
        if img.shape[0] > max_height:
            scale = max_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            new_height = max_height
            img = cv2.resize(img, (new_width, new_height))

        # Add camera label
        label = f"Cam {cam_idx}"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)

        images.append(img)

    if not images:
        return None

    # Concatenate cameras horizontally
    concat_img = np.concatenate(images, axis=1)

    # Add header with episode name
    header_height = 80
    header = np.zeros((header_height, concat_img.shape[1], 3), dtype=np.uint8)

    # Episode name
    episode_name = episode_data['name']
    cv2.putText(header, episode_name, (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    # Add gripper action bar below images
    gripper_value = 0.0
    if episode_data['grasp_actions'] is not None and frame_idx < len(episode_data['grasp_actions']):
        gripper_value = float(episode_data['grasp_actions'][frame_idx])

    gripper_bar = create_gripper_bar(concat_img.shape[1], gripper_value, bar_height=50)

    # Combine header, image, and gripper bar
    final_img = np.vstack([header, concat_img, gripper_bar])

    return final_img


def visualize_all_episodes(episodes_path, num_cams=3):
    """
    Interactive visualization of episodes' RGB images.

    Controls:
        - Up/Down arrow: Navigate between episodes
        - Left/Right arrow: Navigate frames within episode
        - 'q' or ESC: Quit
        - Space: Auto-play frames
        - 'r': Reset to first frame
        - 'n': Next episode
        - 'p': Previous episode
    """
    episodes_data = load_all_episodes_images(episodes_path, num_cams)

    if not episodes_data:
        print("No episodes found or loaded!")
        return

    print(f"\nTotal episodes: {len(episodes_data)}")

    window_name = "Episode Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    current_episode_idx = 0
    current_frame = 0
    auto_play = False

    while True:
        episode = episodes_data[current_episode_idx]
        num_frames = episode['num_frames']

        # Ensure frame is within bounds
        current_frame = max(0, min(current_frame, num_frames - 1))

        # Create visualization
        concat_img = create_episode_frame_visualization(episode, current_frame)

        if concat_img is None:
            print("Failed to create visualization")
            break

        # Add info overlay at the bottom
        info_height = 60
        info_bar = np.zeros((info_height, concat_img.shape[1], 3), dtype=np.uint8)

        # Episode info
        episode_info = f"Episode: {current_episode_idx + 1}/{len(episodes_data)}"
        cv2.putText(info_bar, episode_info, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Frame info
        frame_info = f"Frame: {current_frame + 1}/{num_frames}"
        cv2.putText(info_bar, frame_info, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Controls
        controls_text = "Up/Down: Episodes | Left/Right: Frames | Space: Auto-play | Q: Quit"
        text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(info_bar, controls_text,
                   (concat_img.shape[1] - text_size[0] - 10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        final_img = np.vstack([concat_img, info_bar])

        cv2.imshow(window_name, final_img)

        # Handle keyboard input
        wait_time = 30 if auto_play else 0
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == 81 or key == 2:  # Left arrow
            current_frame = max(0, current_frame - 1)
            auto_play = False
        elif key == 83 or key == 3:  # Right arrow
            current_frame = min(num_frames - 1, current_frame + 1)
            auto_play = False
        elif key == 82 or key == 0:  # Up arrow
            # Previous episode
            current_episode_idx = max(0, current_episode_idx - 1)
            current_frame = 0  # Reset to first frame of new episode
            auto_play = False
            print(f"Switched to episode {current_episode_idx + 1}: {episodes_data[current_episode_idx]['name']}")
        elif key == 84 or key == 1:  # Down arrow
            # Next episode
            current_episode_idx = min(len(episodes_data) - 1, current_episode_idx + 1)
            current_frame = 0  # Reset to first frame of new episode
            auto_play = False
            print(f"Switched to episode {current_episode_idx + 1}: {episodes_data[current_episode_idx]['name']}")
        elif key == ord(' '):  # Space
            auto_play = not auto_play
            print(f"Auto-play: {'ON' if auto_play else 'OFF'}")
        elif key == ord('r'):  # Reset
            current_frame = 0
            auto_play = False
            print("Reset to first frame")
        elif key == ord('n'):  # Next episode
            current_episode_idx = min(len(episodes_data) - 1, current_episode_idx + 1)
            current_frame = 0
            auto_play = False
            print(f"Switched to episode {current_episode_idx + 1}: {episodes_data[current_episode_idx]['name']}")
        elif key == ord('p'):  # Previous episode
            current_episode_idx = max(0, current_episode_idx - 1)
            current_frame = 0
            auto_play = False
            print(f"Switched to episode {current_episode_idx + 1}: {episodes_data[current_episode_idx]['name']}")

        # Auto-play: advance frame
        if auto_play:
            current_frame += 1
            if current_frame >= num_frames:
                # Move to next episode when frames end
                current_episode_idx = min(len(episodes_data) - 1, current_episode_idx + 1)
                current_frame = 0
                if current_episode_idx == len(episodes_data) - 1:
                    # Loop back to first episode
                    current_episode_idx = 0
                print(f"Auto-play: Switched to episode {current_episode_idx + 1}")

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize episodes' RGB images from multiple cameras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python inspect_raw_data.py /home/mingxi/code/h2r_franka_ROS2/raw_datasets/episodes

Controls:
  Up/Down arrows: Navigate between episodes
  Left/Right arrows: Navigate frames within current episode
  Space: Toggle auto-play (plays frames and moves to next episode)
  N/P: Next/Previous episode
  R: Reset to first frame
  Q/ESC: Quit
        """
    )

    parser.add_argument(
        "path",
        type=str,
        help="Path to episodes root directory"
    )

    parser.add_argument(
        "--num-cams",
        type=int,
        default=3,
        help="Number of cameras (default: 3)"
    )

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return

    visualize_all_episodes(path, args.num_cams)


if __name__ == "__main__":
    main()
