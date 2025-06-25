import os
import re
import subprocess
import cv2

def extract_frame_index(filename):
    """Extracts frame index from a filename like frame_00023_final.jpg"""
    match = re.search(r'frame_(\d+)', filename)
    return int(match.group(1)) if match else None

def convert_images_to_video(image_folder: str, framerate: int = 24):
    """
    Converts a folder of images into a video using ffmpeg, adjusting duration by frame number difference.
    
    Args:
        image_folder (str): Path to the folder containing images.
        base_framerate (int): Original FPS to infer real time between frames.
    """
    base_framerate = framerate
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.endswith('_final.jpg') or f.endswith('_all.jpg')
    ])

    if not image_files:
        print("❌ No images found.")
        return

    list_path = os.path.join(image_folder, "images.txt")
    durations = []

    # Get frame indices, filter out files with None indices
    indexed_images = [(f, extract_frame_index(f)) for f in image_files]
    indexed_images = [(f, idx) for f, idx in indexed_images if idx is not None]
    if not indexed_images:
        print("❌ No images with valid frame indices found.")
        return
    image_files, frame_indices = zip(*indexed_images)
    image_files = list(image_files)
    frame_indices = list(frame_indices)

    with open(list_path, "w") as f:
        for i in range(len(image_files)):
            f.write(f"file '{image_files[i]}'\n")
            if i < len(image_files) - 1:
                frame_gap = frame_indices[i+1] - frame_indices[i]
            else:
                frame_gap = 1  # assume 1 for the last frame
            duration = frame_gap / base_framerate
            durations.append(duration)
            f.write(f"duration {duration:.4f}\n")
        f.write(f"file '{image_files[-1]}'\n")  # repeat last frame

    output_path = os.path.join(image_folder, "output.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "images.txt",
        "-pix_fmt", "yuv420p",
        "output.mp4"
    ]

    cwd = os.getcwd()
    os.chdir(image_folder)
    subprocess.run(cmd)
    os.chdir(cwd)

    print(f"✅ Saved video to {output_path}")

def get_first_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"✅ First frame saved to {output_path}")
    else:
        print("❌ Failed to read first frame.")
    cap.release()
