import cv2
import os

def subsample_video(video_path, skip_every_frame=10, output_dir="./data", new_size=None):
    """
    Subsamples frames from a video and optionally reshapes (resizes) them.

    :param video_path: Path to the video file (e.g. 'test.MOV').
    :param skip_every_frame: Capture a frame every 'skip_every_frame' frames.
    :param output_dir: Directory where the frames will be saved.
    :param new_size: Tuple (width, height) for resizing frames. 
                     e.g. (640, 360). If None, keeps the original size.
    """

    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_index = 0  # Keeps track of how many frames we've processed
    saved_count = 0  # Counts how many frames we've actually saved

    while True:
        ret, frame = cap.read()
        if not ret:
            # No more frames; we've reached the end of the video
            break

        # Save the frame if it matches the skipping condition
        if frame_index % skip_every_frame == 0:
            # Resize the frame if a new size is specified
            if new_size is not None:
                frame = cv2.resize(frame, new_size)

            frame_filename = os.path.join(output_dir, f"frame_{frame_index:06d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_index += 1

    # Make sure to release the capture object
    cap.release()
    print(f"Extraction complete. Saved {saved_count} frames to {output_dir}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Subsample frames from a video")
    parser.add_argument("--video_path", type=str, default="/home/xhe71/Downloads/human (1).mp4", help="Path to input video file")
    parser.add_argument("--skip_every_frame", type=int, default=1, help="Number of frames to skip between each capture")
    parser.add_argument("--output_dir", type=str, default="./hamer_detector/example_data/test-env", help="Directory to save extracted frames")
    parser.add_argument("--resize_width", type=int, default=640, help="Width to resize frames to")
    parser.add_argument("--resize_height", type=int, default=480, help="Height to resize frames to")
    args = parser.parse_args()

    subsample_video(
        args.video_path,
        skip_every_frame=args.skip_every_frame,
        output_dir=args.output_dir,
        new_size=(args.resize_width, args.resize_height)
    )
