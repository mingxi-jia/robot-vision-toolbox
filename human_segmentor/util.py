import subprocess
import cv2
def convert_images_to_video(image_folder: str, framerate: int = 24):
    """
    Converts a folder of images into a video using ffmpeg.
    
    Args:
        image_folder (str): Path to the folder containing image frames (e.g., *_final.jpg).
        framerate (int, optional): Frames per second for the output video. Defaults to 24.
    
    Output:
        Saves a file named 'output.mp4' in the same directory.
    """
    import os

    # Get list of images sorted
    image_files = sorted([f for f in os.listdir(image_folder) if (f.endswith('_final.jpg') or f.endswith('_all.jpg'))])
    if not image_files:
        print("❌ No images found to convert.")
        return

    list_path = os.path.join(image_folder, "images.txt")
    with open(list_path, "w") as f:
        for img in image_files:
            f.write(f"file '{img}'\n")

    output_path = os.path.join(image_folder, "output.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "images.txt",
        "-vf", f"fps={framerate}",
        "-pix_fmt", "yuv420p",
        "output.mp4"
    ]

    cwd = os.getcwd()
    os.chdir(image_folder)
    print(f"curr_Dir: {os.getcwd()}")
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
        print("❌ Failed to read video or grab first frame.")

    cap.release()
