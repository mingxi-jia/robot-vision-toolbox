import os
import argparse

from hamer_detector.video_preprocessor import subsample_video
from hamer_detector.demo import detect_hand 
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
import sys
sys.path.append('./')
from human_segmentor.human_pose_segmentor_mp_sam import process_image_folder
from human_segmentor.replace_hand_w_sphere import replace_sphere
from human_segmentor.util import convert_images_to_video, get_first_frame
# import os
# os.environ["PYOPENGL_PLATFORM"] = "glfw"  # force use of native OpenGL
import time
import matplotlib.pyplot as plt
from hamer_detector.KF_smoothing import smooth_KF_yml, smooth_hand_pose_json_KF


def main(video_path, tmp_img_dir, segmentation_out_dir, hamer_out_dir, sphere_out_dir, background_img, handedness = 'right'):
    # Make all paths absolute
    video_path = os.path.abspath(video_path)
    tmp_img_dir = os.path.abspath(tmp_img_dir)
    segmentation_out_dir = os.path.abspath(segmentation_out_dir)
    hamer_out_dir = os.path.abspath(hamer_out_dir)
    sphere_out_dir = os.path.abspath(sphere_out_dir)
    if background_img is None:
        background_img = os.path.join(tmp_img_dir + 'background_img.png')
        get_first_frame(video_path, background_img)
    else:
        background_img = os.path.abspath(background_img)

    # print("🔹 Step 1: Extracting video frames...")
    # subsample_video(video_path, skip_every_frame=1, output_dir=tmp_img_dir, new_size=(640, 480))

    # print("🔹 Step 2: Running HaMeR hand detection and 3D reconstruction...")
    # hamer_args = argparse.Namespace(
    #     checkpoint=DEFAULT_CHECKPOINT,
    #     img_folder=tmp_img_dir,
    #     out_folder=hamer_out_dir,
    #     side_view=False,
    #     full_frame=True,
    #     save_mesh=True,
    #     batch_size=48,
    #     rescale_factor=1.0,
    #     body_detector="vitdet",
    #     file_type=["*.jpg", "*.png"]
    # )
    # detect_hand(hamer_args)
    
    # convert_images_to_video(hamer_out_dir)
    # step 2.5 Smooth centroid data
    smooth_KF_yml(os.path.join(hamer_out_dir, 'centroids.yml'))

    smooth_KF_yml(os.path.join(hamer_out_dir, 'wrists.yml'))
    smooth_hand_pose_json_KF(os.path.join(hamer_out_dir, 'hand_pose_camera_info.json'))
    print("saved smoothed KF")
    
    # print("🔹 Step 3: Segmenting and removing human from video...")
    # process_image_folder(image_folder=tmp_img_dir, output_folder=segmentation_out_dir, background_path=background_img, hand_model_path = hamer_out_dir)
    # convert_images_to_video(segmentation_out_dir)
    
    print("🔹 Step 4: Rendering spheres and blending with background...")
    replace_sphere(hamer_out_dir, segmentation_out_dir, sphere_out_dir)
    print("✅ All steps complete. Final images saved to:", sphere_out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline: extract video ➜ segment ➜ reconstruct ➜ render spheres")
    parser.add_argument("--video_path", type=str, default = "/home/xhe71/Downloads/test2_2.mp4", help="Path to the input video file")
    parser.add_argument("--background_img", type=str, default = None, help="Path to background image to use for replacement")
    parser.add_argument("--handedness", type=str, default="left",
                    help="Select sphere overlay on designated hand: left or right(defualt)")
    args = parser.parse_args()
        # Derive output folders from video_path
    base_dir = os.path.splitext(os.path.abspath(args.video_path))[0]
    tmp_img_dir = base_dir + "_frames"
    segmentation_out_dir = base_dir + "_segmented"
    hamer_out_dir = base_dir + "_hamer"
    sphere_out_dir = base_dir + "_final"
    main(args.video_path, tmp_img_dir, segmentation_out_dir, hamer_out_dir, sphere_out_dir, args.background_img, args.handedness)
