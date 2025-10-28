import os
import argparse
import time
import yaml
from pathlib import Path
import numpy as np

import torch
from hamer.configs import CACHE_DIR_HAMER
from hamer_detector.video_preprocessor import subsample_video
from hamer_detector.detector import detect_hand_pipeline_batch, detect_hand_pipeline 
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
from hamer_detector.hamer_smoothing import smooth_hand_pose_json
from human_segmentor.util import rename_images_sequentially
from human_segmentor.sphere_pcd import generate_pcd_sequence

sys.path.append("submodules/hamer")
from vitpose_model import ViTPoseModel
from hamer.utils.renderer import Renderer, cam_crop_to_full

from sam2.build_sam import build_sam2_video_predictor

def read_camera_info(camera_info_path: str) -> dict:
    # process raw camera info to hamer-compatible format
    with open(camera_info_path, 'r') as f:
        camera_info = yaml.safe_load(f)
    cam_list = camera_info.keys()
    for cam in cam_list:
        intrinsics = np.array(camera_info[cam]['k']).reshape(3, 3)
        camera_info[cam]['fx'] = intrinsics[0, 0]
        camera_info[cam]['fy'] = intrinsics[1, 1]
        camera_info[cam]['cx'] = intrinsics[0, 2]
        camera_info[cam]['cy'] = intrinsics[1, 2]
    return camera_info

class HandPreprocessor:
    SAMPLE_RATE = 1

    def __init__(self, dataset_path: str, main_cam_idx: int = 3):

        # Download and load checkpoints
        # download_models(CACHE_DIR_HAMER)
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

        # Load detector
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        body_detector = "regnety"
        if body_detector == 'vitdet':
            from detectron2.config import LazyConfig
            import hamer
            cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5
            detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif body_detector == 'regnety':
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
            detector       = DefaultPredictor_Lazy(detectron2_cfg)
        self.detectron = detector

        # keypoint detector
        self.cpm = ViTPoseModel(device)

        # Setup the renderer
        self.renderer = Renderer(model_cfg, faces=model.mano.faces)

        # Load SAM2 Model
        CHECKPOINT = "submodules/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
        CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT)
 
        self.process_path = os.path.join(dataset_path, "output")
        os.makedirs(self.process_path, exist_ok=True)

        self.main_cam_idx = main_cam_idx
        self.camera_info = read_camera_info('configs/camera_info.yaml')
        self.debug = False 

    def detect_hand(self, args, cam_num, batch_mode=False):
        camera_intrinsics = self.camera_info[f'cam{cam_num}']
        shortened = False if cam_num == 3 else True
        if batch_mode:
            hand_poss = detect_hand_pipeline_batch(args, 
                                                   self.model, 
                                                   self.model_cfg, 
                                                   self.cpm, 
                                                   self.detectron, 
                                                   self.renderer, 
                                                   camera_intrinsics, 
                                                   shortened)
        else:
            hand_poss = detect_hand_pipeline(args, 
                                             self.model, 
                                             self.model_cfg, 
                                             self.cpm, 
                                             self.detectron, 
                                             self.renderer, 
                                             camera_intrinsics, 
                                             shortened)
        return hand_poss

    def get_hamer_poses(self, hamer_args, cam_num, shortened=False):
        """Run the HaMeR hand detection model."""
        
        camera_intrinsics = self.camera_info[f'cam{cam_num}']
        hand_poss = self.detect_hand(hamer_args, cam_num, batch_mode=True)
        if cam_num == 3:
            start_tmp= time.time()
            
            hand_poss = smooth_hand_pose_json(os.path.join(self.hamer_out_dir, 'hand_pose_camera_info.json'), skip_rate=self.SAMPLE_RATE)
            print(f"smoothed_hand_pose_json takes {time.time() - start_tmp:.2f} seconds")
            convert_images_to_video(self.hamer_out_dir, framerate=30 // self.SAMPLE_RATE)
        return hand_poss
        
    def segment_human(self, cam_num):
        """Segment the human from the background using SAM."""
        run_sam2_segmentation(self.predictor, 
                              self.tmp_img_dir, 
                              self.hamer_out_dir, 
                              self.depth_img_dir,
                              self.camera_info[f'cam{cam_num}'], 
                              self.segmentation_out_dir,
                              self.background_img, 
                              self.debug, 
                              ref_cam=cam_num)
        segmented_rgb_dir = os.path.join(self.segmentation_out_dir, "segmented_rgb")
        convert_images_to_video(segmented_rgb_dir, framerate=30 // self.SAMPLE_RATE)
    
    def render_spheres(self, hand_poss, cam_num):
        """Render spheres in place of hands."""
        segmented_rgb_dir = os.path.join(self.segmentation_out_dir, "segmented_rgb")
        segmented_depth_dir = os.path.join(self.segmentation_out_dir, "segmented_depth")
        sphere_poses = replace_sphere(hand_poss, self.hamer_out_dir, segmented_rgb_dir, segmented_depth_dir,
                       self.sphere_out_dir, self.camera_info[f'cam{cam_num}'],
                       ori_depth_folder=self.depth_img_dir, debug=self.debug)
        convert_images_to_video(self.sphere_out_dir, framerate=30 // self.SAMPLE_RATE)
        return sphere_poses
    
    def prepare(self):
        print("🔹 Step 0: Preparing image frames and background...")

        img_count = len([f for f in os.listdir(self.video_path)
                         if f.lower().endswith(('.jpg', '.png'))]) // self.SAMPLE_RATE

        # create tmp data for processing
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
        return img_count
    
    def get_hamer_args(self):
        return argparse.Namespace(
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

    def clean_tmp_images(self):
        if os.path.exists(self.tmp_img_dir):
            for f in os.listdir(self.tmp_img_dir):
                os.remove(os.path.join(self.tmp_img_dir, f))
            os.rmdir(self.tmp_img_dir)
            print(f"🔹 Temporary images directory {self.tmp_img_dir} cleaned up.")

    def process(self, episode_path, cam_num):
        """
        Run the entire preprocessing pipeline.
        """
        total_start = time.time()
        # ----------Prepare paths and directories----------------
        data_path, episode_name = os.path.split(episode_path)
        # Set up directories and file paths for this camera
        self.cam_dir = os.path.join(os.path.abspath(episode_path), f'cam{cam_num}')
        self.video_path = os.path.join(self.cam_dir, 'rgb')
        self.depth_img_dir = os.path.join(self.cam_dir, 'depth')
        self.tmp_img_dir = os.path.join(self.cam_dir, 'tmp_images')
        self.background_img = f"configs/cam{cam_num}_background.png"
        self.intrinsics_path = f"configs/intrinsics_cam{cam_num}.json"

        # Rename images to a sequential order
        rename_images_sequentially(self.video_path, '.png')
        rename_images_sequentially(self.depth_img_dir, '.npy')

        # Define processing output directories
        process_data_path = os.path.join(self.process_path, episode_name, f'cam{cam_num}')
        self.segmentation_out_dir = process_data_path
        self.hamer_out_dir = os.path.join(process_data_path, 'hamer_out')
        self.sphere_out_dir = process_data_path

        hamer_args = self.get_hamer_args()

        prep_start = time.time()
        # Prepare temporary images for processing
        img_count = self.prepare()
        # print(f"Preparation took {time.time() - prep_start:.2f} seconds.")

        hamer_start = time.time()
        # ----------Run HaMeR hand detection----------------
        shortened = False if cam_num == 3 else True
        print("hammer processing")
        hand_poss = self.get_hamer_poses(hamer_args, cam_num, shortened=shortened)
        print(f"HaMeR hand detection took {time.time() - hamer_start:.2f} seconds.")

        seg_start = time.time()
        # ----------Run human segmentation using SAM2----------
        self.segment_human(cam_num)
        print(f"Human segmentation took {time.time() - seg_start:.2f} seconds.")

        # Optional: Render spheres in place of hands (currently commented out)
        # if cam_num == 3:
        #     hand_poss = self.render_spheres(hand_poss, cam_num=cam_num)

        clean_start = time.time()
        # Clean up temporary images directory
        self.clean_tmp_images()
        print(f"Cleanup took {time.time() - clean_start:.2f} seconds.")

        # Save hand poses to file 
        if cam_num == self.main_cam_idx:
            np.save(os.path.join(self.process_path, episode_name, f'hand_poses_wrt_cam{self.main_cam_idx}.npy'), hand_poss)

        print(f"Total processing time: {time.time() - total_start:.2f} seconds.")

        

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