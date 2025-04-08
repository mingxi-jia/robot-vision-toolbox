import os
import argparse

from hamer_detector.video_preprocessor import subsample_video
from hamer_detector.demo import main as hamer_main
from human_segmentor.human_pose_segmentor_mp_sam import process_video
from hamer_detector.sphere_renderer import main as sphere_render_main

def main(video_path, tmp_img_dir, segmentation_out_dir, hamer_out_dir, sphere_out_dir, background_img):
    print("🔹 Step 1: Extracting video frames...")
    subsample_video(video_path, skip_every_frame=1, output_dir=tmp_img_dir, new_size=(640, 480))

    print("🔹 Step 2: Running HaMeR hand detection and 3D reconstruction...")
    hamer_args = argparse.Namespace(
        checkpoint=None,
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
    sphere_render_main(segmentation_out_dir, os.path.join(hamer_out_dir, "centroids.yml"), sphere_out_dir)

    print("✅ All steps complete. Final images saved to:", sphere_out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline: extract video ➜ segment ➜ reconstruct ➜ render spheres")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--tmp_img_dir", type=str, default="hamer_detector/tmp_imgs", help="Temp folder to store extracted frames")
    parser.add_argument("--segmentation_out_dir", type=str, default="hamer_detector/segmentation_output", help="Folder to store segmentation results")
    parser.add_argument("--hamer_out_dir", type=str, default="hamer_detector/hamer_output", help="Folder to store HaMeR outputs")
    parser.add_argument("--sphere_out_dir", type=str, default="hamer_detector/final_output", help="Folder to store blended sphere output")
    parser.add_argument("--background_img", type=str, required=True, help="Path to background image to use for replacement")
    args = parser.parse_args()

    main(args.video_path, args.tmp_img_dir, args.segmentation_out_dir, args.hamer_out_dir, args.sphere_out_dir, args.background_img)