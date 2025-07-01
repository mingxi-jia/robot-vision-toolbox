import os
import argparse
import time

import torch
from hamer.configs import CACHE_DIR_HAMER
from hamer_detector.video_preprocessor import subsample_video
from hamer_detector.demo import detect_hand 
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
import sys
sys.path.append('./')
from human_segmentor.human_pose_sam2_video import run_sam2_segmentation
from human_segmentor.replace_hand_w_sphere import replace_sphere
from human_segmentor.util import convert_images_to_video, get_first_frame
# import os
# os.environ["PYOPENGL_PLATFORM"] = "glfw"  # force use of native OpenGL
import time
import matplotlib.pyplot as plt
from hamer_detector.KF_smoothing import smooth_hand_pose_json_KF
from human_segmentor.util import rename_images_sequentially
from human_segmentor.sphere_pcd import generate_pcd_sequence
from vitpose_model import ViTPoseModel
from hamer.utils.renderer import Renderer, cam_crop_to_full

from sam2.build_sam import build_sam2_video_predictor

class HandPreprocessor:
    SAMPLE_RATE = 2

    def __init__(self):
        # Download and load checkpoints
        download_models(CACHE_DIR_HAMER)
        model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)

        # Setup HaMeR model
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        torch.backends.cudnn.benchmark = True

        self.model_cfg = model_cfg
        self.model = model.to(device)
        self.model.eval()

        # keypoint detector
        self.cpm = ViTPoseModel(device)

        # Setup the renderer
        self.renderer = Renderer(model_cfg, faces=model.mano.faces)

        # init sam
        # Load SAM2 Model
        CHECKPOINT = "submodules/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
        CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT)

        self.debug = False  # Set to True to save mesh and visualize

    def hamer_mask(self):
        """Run the HaMeR hand detection model."""
        hamer_args = argparse.Namespace(
            checkpoint=DEFAULT_CHECKPOINT,
            img_folder=self.tmp_img_dir,
            out_folder=self.hamer_out_dir,
            depth_folder=self.depth_img_dir,
            intrinsics_path=self.intrinsics_path,
            side_view=False,
            full_frame=True,
            save_mesh=False,
            batch_size=48,
            rescale_factor=1.0,
            body_detector="regnety",
            file_type=["*.jpg", "*.png"],
        )
        detect_hand(hamer_args, self.model, self.model_cfg, self.cpm, self.renderer)
        smooth_hand_pose_json_KF(os.path.join(self.hamer_out_dir, 'hand_pose_camera_info.json'),
                                 skip_rate=self.SAMPLE_RATE)
        convert_images_to_video(self.hamer_out_dir, framerate=30 // self.SAMPLE_RATE)
        print("Step 2: HaMeR hand detection completed.")
        
    def segment_human(self, cam_num):
        """Segment the human from the background using SAM."""
        run_sam2_segmentation(self.predictor, self.tmp_img_dir, self.hamer_out_dir, self.depth_img_dir,
                              self.intrinsics_path, self.segmentation_out_dir,
                              self.background_img, self.debug, ref_cam=cam_num)
        segmented_rgb_dir = os.path.join(self.segmentation_out_dir, "segmented_rgb")
        segmented_depth_dir = os.path.join(self.segmentation_out_dir, "segmented_depth")
        convert_images_to_video(segmented_rgb_dir, framerate=30 // self.SAMPLE_RATE)
        print("Step 3: Human segmentation completed.")  
    
    def render_spheres(self):
        """Render spheres in place of hands."""
        segmented_rgb_dir = os.path.join(self.segmentation_out_dir, "segmented_rgb")
        segmented_depth_dir = os.path.join(self.segmentation_out_dir, "segmented_depth")
        replace_sphere(self.hamer_out_dir, segmented_rgb_dir, segmented_depth_dir,
                       self.sphere_out_dir, self.intrinsics_path,
                       ori_depth_folder=self.depth_img_dir, debug=self.debug)
        convert_images_to_video(self.sphere_out_dir, framerate=30 // self.SAMPLE_RATE)
        print("Step 4: Sphere rendering completed.")    
    
    
    def process(self, episode_path, cam_num):
        """Run the entire preprocessing pipeline."""
        self.cam_dir = os.path.join(os.path.abspath(episode_path), f'cam{cam_num}')
        self.video_path = os.path.join(self.cam_dir, 'rgb')
        self.depth_img_dir =  os.path.join(self.cam_dir, 'depth')
        self.tmp_img_dir = os.path.join(self.cam_dir, 'tmp_images')
        rename_images_sequentially(self.video_path, '.png')
        rename_images_sequentially(self.depth_img_dir, '.npy')
        self.segmentation_out_dir =os.path.join(self.cam_dir, 'segment_out')
        self.hamer_out_dir = os.path.join(self.cam_dir, 'hamer_out')
        self.sphere_out_dir = os.path.join(self.cam_dir, 'output')
        self.background_img = f"setup/cam{cam_num}_background.png"
        self.intrinsics_path = f"setup/intrinsics_cam{cam_num}.json"

        start_time = time.time()
        print("🔹 Step 0: Preparing image frames and background...")

        img_count = len([f for f in os.listdir(self.video_path)
                         if f.lower().endswith(('.jpg', '.png'))]) // self.SAMPLE_RATE

        if not os.path.exists(self.tmp_img_dir) or len(os.listdir(self.tmp_img_dir)) == 0:
            print("🔹 Step 1: Copying frames...")
            os.makedirs(self.tmp_img_dir, exist_ok=True)
            image_fnames = sorted([
                f for f in os.listdir(self.video_path)
                if f.lower().endswith(('.jpg', '.png'))
            ])
            for i, fname in enumerate(image_fnames):
                if i % self.SAMPLE_RATE == 0:
                    src = os.path.abspath(os.path.join(self.video_path, fname))
                    dst = os.path.join(self.tmp_img_dir, fname)
                    if not os.path.exists(dst):
                        os.symlink(src, dst)
        else:
            print("🔹 Step 1: Skipping frame copy, tmp images already exist.")

        self.hamer_mask()
        self.segment_human(cam_num)
        self.render_spheres()

        end_time = time.time()
        print(f"✅ Pipeline completed. Processed {img_count} frames in {round(end_time - start_time, 2)} seconds.")
        # delete the tmp images
        if os.path.exists(self.tmp_img_dir):
            for f in os.listdir(self.tmp_img_dir):
                os.remove(os.path.join(self.tmp_img_dir, f))
            os.rmdir(self.tmp_img_dir)
            print(f"🔹 Temporary images directory {self.tmp_img_dir} cleaned up.")
        

# =================== Run all 3 camera views ===================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PreprocessSinglePipeline on all three camera views.")
    parser.add_argument("--episode_path", type=str, required=True, help="Path to episode folder containing cam1, cam2, cam3")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for mesh saving and visualization")

    args = parser.parse_args()

    for cam_id in [1, 2, 3]:
        print(f"\n========= Processing Camera {cam_id} =========")
        pipeline = HandPreprocessor(args.episode_path, cam_num=cam_id, debug=args.debug)
        pipeline.process()
    generate_pcd_sequence(args.episode_path, start_frame=0, sphere_cam=3)