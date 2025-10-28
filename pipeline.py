import os
import argparse
import time
from hamer_detector.video_preprocessor import subsample_video
from hamer_detector.detector import detect_hand 
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
import sys
sys.path.append('./')
# from human_segmentor.human_pose_segmentor_mp_sam import process_image_folder
from human_segmentor.human_pose_sam2_video import run_sam2_segmentation
from human_segmentor.replace_hand_w_sphere import replace_sphere
from human_segmentor.util import convert_images_to_video, get_first_frame
# import os
# os.environ["PYOPENGL_PLATFORM"] = "glfw"  # force use of native OpenGL
import time
import matplotlib.pyplot as plt
from hamer_detector.hamer_smoothing import smooth_hand_pose_json_KF
from human_segmentor.util import rename_images_sequentially
SAMPLE_RATE = 1
start_time = time.time()
def main(video_path, tmp_img_dir, segmentation_out_dir, hamer_out_dir, sphere_out_dir, background_img, depth_img_dir, intrinsics_path, cam_num, debug):
    # Make all paths absolute
    video_path = os.path.abspath(video_path)
    rename_images_sequentially(video_path, '.png')
    rename_images_sequentially(depth_img_dir, '.npy')
    tmp_img_dir = os.path.abspath(tmp_img_dir)
    segmentation_out_dir = os.path.abspath(segmentation_out_dir)
    hamer_out_dir = os.path.abspath(hamer_out_dir)
    sphere_out_dir = os.path.abspath(sphere_out_dir)
    img_count = len([f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]) // SAMPLE_RATE
    print(f"Number of files: {img_count}")
    if background_img is None:
        if os.path.isdir(video_path):
            # Use the first image file in the folder
            image_fnames = sorted([
                f for f in os.listdir(video_path)
                if f.lower().endswith(('.jpg', '.png'))
            ])
            if image_fnames:
                background_img = os.path.join(video_path, image_fnames[0])
            else:
                raise FileNotFoundError("No image files found in the provided image folder.")
        else:
            background_img = os.path.join(tmp_img_dir, 'background_img.png')
            get_first_frame(video_path, background_img)
    else:
        background_img = os.path.abspath(background_img)

    if not os.path.exists(tmp_img_dir) or len(os.listdir(tmp_img_dir)) == 0:
        if os.path.isdir(video_path):
            print("🔹 Step 1: Using provided image folder with frame skipping...")
            os.makedirs(tmp_img_dir, exist_ok=True)
            image_fnames = sorted([
                f for f in os.listdir(video_path)
                if f.lower().endswith(('.jpg', '.png'))
            ])
            img_count = len(image_fnames)
            skip_rate = SAMPLE_RATE
            for i, fname in enumerate(image_fnames):
                if i % skip_rate == 0:
                    os.symlink(os.path.abspath(os.path.join(video_path, fname)),
                               os.path.join(tmp_img_dir, fname))
                
        else:
            print("🔹 Step 1: Extracting video frames...")
            subsample_video(video_path, skip_every_frame=SAMPLE_RATE, output_dir=tmp_img_dir, new_size=(640, 360))
    else:
        print("🔹 Step 1: Skipping frame extraction, images already exist.")

    print("🔹 Step 2: Running HaMeR hand detection and 3D reconstruction...")
    hamer_start_time = time.time()
    hamer_args = argparse.Namespace(
        checkpoint=DEFAULT_CHECKPOINT,
        img_folder=tmp_img_dir,
        out_folder=hamer_out_dir,
        depth_folder = depth_img_dir,
        intrinsics_path = intrinsics_path,
        side_view=False,
        full_frame=True,
        save_mesh=debug,
        batch_size=48,
        rescale_factor=1.0,
        body_detector="regnety",
        file_type=["*.jpg", "*.png"],
        debug=debug
    )
    # Add depth and intrinsics to args if available
    if hasattr(hamer_args, "depth_folder") and hamer_args.depth_folder is not None:
        hamer_args.depth_folder = args.depth_folder
    if hasattr(hamer_args, "intrinsics_path") and hamer_args.intrinsics_path is not None:
        hamer_args.intrinsics_path = args.intrinsics_path
    
    if cam_num == 1:
        background_img = "configs/cam1_background.png"
        depth_background_img = "configs/cam1_background.npy"
        hamer_args.intrinsics_path = "configs/intrinsics_cam1.json"
    elif cam_num == 2:
        background_img = "configs/cam2_background.png"
        depth_background_img = "configs/cam2_background.npy"
        hamer_args.intrinsics_path = "configs/intrinsics_cam2.json"
    elif cam_num == 3:
        background_img = "configs/cam3_background.png"
        depth_background_img = "configs/cam3_background.npy"
        hamer_args.intrinsics_path = "configs/intrinsics_cam3.json"
    intrinsics_path = hamer_args.intrinsics_path

    detect_hand(hamer_args)
    hamer_end_time = time.time()
    convert_images_to_video(hamer_out_dir, framerate=30//SAMPLE_RATE)
    # step 2.5 Smooth centroid data
    # smooth_hand_pose_json_KF(os.path.join(hamer_out_dir, 'hand_pose_camera_info.json'), skip_rate = SAMPLE_RATE)
    # print("saved smoothed KF")
    
    print("🔹 Step 3: Segmenting and removing human from video...")
    seg_start_time = time.time()
    run_sam2_segmentation(tmp_img_dir, hamer_out_dir, depth_img_dir, intrinsics_path, segmentation_out_dir, background_img, debug, ref_cam = cam_num)

    segmented_rgb_dir = os.path.join(segmentation_out_dir, "segmented_rgb")
    segmented_depth_dir = os.path.join(segmentation_out_dir, "segmented_depth")
    seg_end_time = time.time()
    convert_images_to_video(segmented_rgb_dir, framerate=30//SAMPLE_RATE)
    
    print("🔹 Step 4: Rendering spheres and blending with background...")
    sphere_start_time = time.time()
    replace_sphere(hamer_out_dir, segmented_rgb_dir, segmented_depth_dir, sphere_out_dir, intrinsics_path, ori_depth_folder=depth_img_dir, debug=debug)
    sphere_end_time = time.time()
    convert_images_to_video(sphere_out_dir, framerate=30//SAMPLE_RATE)
    print("✅ All steps complete. Final images saved to:", sphere_out_dir)

    end_time = time.time()
    print("----------------------------------")
    processing_frames = img_count//SAMPLE_RATE
    total_time =  end_time - start_time
    avg_time = round(total_time / processing_frames, 4)
    hamer_avg_time = round((hamer_end_time-hamer_start_time) / processing_frames, 4)
    seg_avg_time = round((seg_end_time-seg_start_time) / processing_frames, 4)
    sphere_avg_time = round((sphere_end_time-sphere_start_time) / processing_frames, 4)
    print("total processing frames: ", processing_frames, "frames")
    print("total processing time: ", total_time, "seconds")
    print("Average: ", avg_time, "seconds/frame")
    print("Average Hamer:             |", hamer_avg_time, "seconds/frame |", round((hamer_avg_time/avg_time)*100, 2), '\%')
    print("Average Segmentation:      |", seg_avg_time, "seconds/frame | ", round((seg_avg_time/avg_time)*100, 2), '\%')
    print("Average Sphere Replacement:|", sphere_avg_time, "seconds/frame |", round((sphere_avg_time/avg_time)*100, 2), '\%')
    print("----------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline: extract video ➜ segment ➜ reconstruct ➜ render spheres")


    parser.add_argument("--video_path", type=str, help="Path to the input video file or image folder")
    parser.add_argument("--cam_num", type=int, default = 1, help="Camera id(int): 1-(right side), 2-(left side), or 3-(front) ")
    parser.add_argument("--background_img", type=str, default = None, help="Path to background image to use for replacement")
    parser.add_argument("--depth_folder", type=str,  help="Folder with depth images matching image frames")
    parser.add_argument("--intrinsics_path", type=str, default=None, help="Path to camera intrinsics .json file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with full rendering and mesh saving")
    args = parser.parse_args()
    # Derive output folders from video_path    
    base_dir = os.path.splitext(os.path.abspath(args.video_path))[0]
    tmp_img_dir = base_dir + "_frames"
    segmentation_out_dir = base_dir + "_segmented"
    hamer_out_dir = base_dir + "_hamer"
    sphere_out_dir = base_dir + "_final"

    main(args.video_path, tmp_img_dir, segmentation_out_dir, hamer_out_dir, sphere_out_dir, args.background_img, args.depth_folder, args.intrinsics_path, args.cam_num, args.debug)
