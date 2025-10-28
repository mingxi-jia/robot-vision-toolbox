import os
import re
import subprocess
import cv2

# Global verbosity flag - set to False to suppress non-essential prints
VERBOSE = False

def convert_images_to_video(image_folder: str, framerate: int = 24):
    """
    Converts a folder of images into a video using OpenCV's VideoWriter.
    
    Args:
        image_folder (str): Path to the folder containing images.
        framerate (int): Frame rate of the output video.
    """
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.endswith('_final.png') or f.endswith('.png') or f.endswith('_all.png')
    ])

    if not image_files:
        print("❌ No images found.")
        return

    # Read the first image to get the frame size
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"❌ Failed to read the first image: {first_image_path}")
        return
    height, width, channels = first_image.shape

    output_path = os.path.join(image_folder, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, framerate, (width, height))

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Skipping unreadable image: {image_path}")
            continue
        if (image.shape[1], image.shape[0]) != (width, height):
            print(f"⚠️ Skipping image with mismatched size: {image_path}")
            continue
        video_writer.write(image)

    video_writer.release()
    if VERBOSE:
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


def rename_images_sequentially(folder_path: str, ext: str = ".png"):
    """
    Renames all image files in a folder to sequential format like 000001.png.
    
    Args:
        folder_path (str): The directory containing image files.
        ext (str): Target file extension for renaming (e.g., '.png', '.jpg', '.npy').
    """
    valid_exts = (".png", ".jpg", ".jpeg", ".npy")
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
    for idx, filename in enumerate(files, start=0):
        new_name = f"{idx:06d}{ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
    if VERBOSE:
        print(f"✅ Renamed {len(files)} files to sequential names with extension {ext}")


def convert_image_format(folder_path: str, target_ext: str = ".jpg"):
    """
    Converts all images in a folder from PNG to JPG or vice versa.
    
    Args:
        folder_path (str): The directory containing image files.
        target_ext (str): Target file extension (e.g., '.jpg' or '.png').
    """
    assert target_ext in (".jpg", ".png"), "Only .jpg and .png are supported."
    source_ext = ".png" if target_ext == ".jpg" else ".jpg"
    
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(source_ext)])
    for filename in files:
        src_path = os.path.join(folder_path, filename)
        image = cv2.imread(src_path)
        base = os.path.splitext(filename)[0]
        dst_path = os.path.join(folder_path, f"{base}{target_ext}")
        cv2.imwrite(dst_path, image)
    if VERBOSE:
        print(f"✅ Converted {len(files)} files from {source_ext} to {target_ext}")
