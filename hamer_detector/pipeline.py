import os
import argparse

from video_preprocessor import subsample_video
from demo import main as hamer_main
from sphere_renderer import main as sphere_render_main
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
import sys
sys.path.append('./')
from human_segmentor.human_pose_segmentor_mp_sam import process_video


def main(video_path, tmp_img_dir, segmentation_out_dir, hamer_out_dir, sphere_out_dir, background_img, handedness = 'right'):
    # Make all paths absolute
    video_path = os.path.abspath(video_path)
    tmp_img_dir = os.path.abspath(tmp_img_dir)
    segmentation_out_dir = os.path.abspath(segmentation_out_dir)
    hamer_out_dir = os.path.abspath(hamer_out_dir)
    sphere_out_dir = os.path.abspath(sphere_out_dir)
    background_img = os.path.abspath(background_img)

    print("🔹 Step 1: Extracting video frames...")
    subsample_video(video_path, skip_every_frame=1, output_dir=tmp_img_dir, new_size=(640, 480))

    print("🔹 Step 2: Running HaMeR hand detection and 3D reconstruction...")
    hamer_args = argparse.Namespace(
        checkpoint=DEFAULT_CHECKPOINT,
        img_folder=tmp_img_dir,
        out_folder=hamer_out_dir,
        side_view=False,
        full_frame=True,
        save_mesh=True,
        batch_size=48,
        rescale_factor=2.0,
        body_detector="vitdet",
        file_type=["*.jpg", "*.png"]
    )
    hamer_main(hamer_args)

    print("🔹 Step 3: Segmenting and removing human from video...")
    process_video(video_path=video_path, output_folder=segmentation_out_dir, background_path=background_img)

    print("🔹 Step 4: Rendering spheres and blending with background...")
    sphere_render_main(segmentation_out_dir, os.path.join(hamer_out_dir, "centroids.yml"), sphere_out_dir, handedness)

    print("🔹 Step 5: Compiling final blended video...")
    os.system(f"ffmpeg -framerate 30 -i {sphere_out_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p {sphere_out_dir}/output_final_video.mp4")
    print(f"🎬 Final video saved to: {sphere_out_dir}/output_final_video.mp4")

    print("✅ All steps complete. Final images saved to:", sphere_out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline: extract video ➜ segment ➜ reconstruct ➜ render spheres")
    parser.add_argument("--video_path", type=str, default = "/home/xhe71/Downloads/human (1).mp4", help="Path to the input video file")
    parser.add_argument("--tmp_img_dir", type=str, default="hamer_detector/tmp_imgs", help="Temp folder to store extracted frames")
    parser.add_argument("--segmentation_out_dir", type=str, default="hamer_detector/segmentation_output", help="Folder to store segmentation results")
    parser.add_argument("--hamer_out_dir", type=str, default="hamer_detector/hamer_output", help="Folder to store HaMeR outputs")
    parser.add_argument("--sphere_out_dir", type=str, default="hamer_detector/final_output", help="Folder to store blended sphere output")
    parser.add_argument("--background_img", type=str, default = "hamer_detector/example_data/frame_000340_final.png", help="Path to background image to use for replacement")
    parser.add_argument("--handedness", type=str, default="right",
                    help="Select sphere overlay on designated hand: left or right(defualt)")
    args = parser.parse_args()

    main(args.video_path, args.tmp_img_dir, args.segmentation_out_dir, args.hamer_out_dir, args.sphere_out_dir, args.background_img, args.handedness)